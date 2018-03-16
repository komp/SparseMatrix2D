
/*
 *  Based on code found at the Nax.io website:
 *  https://nax.io/2017/05/21/implementing-a-thread-pool-in-c++
 *  It also includes excellent descriptive text about the implementation.
 */
#include "decoder_pool.h"

DecoderPool::DecoderPool (H_matrix *hmat, unsigned int maxIterations, size_t decoder_count)
  : P_running(true)
  , P_job_size(0u)
  , P_hmat(hmat)
  , P_maxIterations(maxIterations)
{
  P_threads.reserve(decoder_count);
  for (size_t i = 0; i < decoder_count; i++) {
    P_threads.emplace_back(&DecoderPool::worker_main, this);
  }
}

DecoderPool::~DecoderPool() {
  std::unique_lock<std::mutex> lock(P_mutex);
  P_running = false;
  P_cv_worker.notify_all();
  lock.unlock();
  for (auto& t : P_threads) t.join();
}

void DecoderPool::schedule_job(bundleElt *packet_address, bundleElt *decode_address) {
  std::lock_guard<std::mutex> lock(P_mutex);

  P_job_packet_address.push_back(packet_address);
  P_job_decode_address.push_back(decode_address);
  P_job_size++;
  P_cv_worker.notify_one();
}

void DecoderPool::worker_main() {

  std::unique_lock<std::mutex> lock(P_mutex);

  unsigned int  returnVal;

  bundleElt *dev_rSig;
  bundleElt *dev_eta;
  bundleElt *dev_estimate;
  bundleElt *dev_etaByBitIndex;
  bundleElt *dev_lambdaByCheckIndex;
  bundleElt *dev_cHat;
  bundleElt *dev_parityBits;

  unsigned int *dev_mapRC;
  unsigned int *dev_mapCR;

  unsigned int *mapRows2Cols = P_hmat->mapRows2Cols;
  unsigned int *mapCols2Rows = P_hmat->mapCols2Rows;
  unsigned int numBits = P_hmat->numBits;
  unsigned int numChecks = P_hmat->numChecks;
  unsigned int maxBitsPerCheck = P_hmat->maxBitsPerCheck;
  unsigned int maxChecksPerBit = P_hmat->maxChecksPerBit;

  unsigned int nChecksByBits = numChecks*(maxBitsPerCheck+1);
  unsigned int nBitsByChecks = numBits*(maxChecksPerBit+1);
  HANDLE_ERROR( cudaMalloc( (void**)&dev_rSig, numBits * sizeof(bundleElt)));
  HANDLE_ERROR( cudaMalloc( (void**)&dev_eta, nChecksByBits * sizeof(bundleElt)));
  HANDLE_ERROR( cudaMalloc( (void**)&dev_estimate, numBits * sizeof(bundleElt)));
  HANDLE_ERROR( cudaMalloc( (void**)&dev_etaByBitIndex,  nBitsByChecks * sizeof(bundleElt)));
  HANDLE_ERROR( cudaMalloc( (void**)&dev_lambdaByCheckIndex, nChecksByBits * sizeof(bundleElt)));
  HANDLE_ERROR( cudaMalloc( (void**)&dev_mapRC, nChecksByBits * sizeof(unsigned int)));
  HANDLE_ERROR( cudaMalloc( (void**)&dev_mapCR, nBitsByChecks * sizeof(unsigned int)));
  HANDLE_ERROR( cudaMalloc( (void**)&dev_cHat, nChecksByBits * sizeof(bundleElt)));
  HANDLE_ERROR( cudaMalloc( (void**)&dev_parityBits, numChecks * sizeof(bundleElt)));

  HANDLE_ERROR(cudaMemcpy(dev_mapRC, mapRows2Cols, nChecksByBits * sizeof(unsigned int), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_mapCR, mapCols2Rows, nBitsByChecks * sizeof(unsigned int), cudaMemcpyHostToDevice));

  // All matrices are stored in column order.  For Check arrays
  // There are numCheck columns, and maxBitsPerCheck+1 rows.
  // row 0 always contains the number of contributors for this check node.

  bundleElt *checksWeight = (bundleElt *) malloc (numChecks * sizeof(bundleElt));
  bundleElt *bitsWeight = (bundleElt *) malloc (numBits * sizeof(bundleElt));

  for (unsigned int check=0; check<numChecks; check++) checksWeight[check] = make_bundleElt((float)mapRows2Cols[check]);

  for (unsigned int bit=0; bit<numBits; bit++) bitsWeight[bit] = make_bundleElt((float)mapCols2Rows[bit]);

  HANDLE_ERROR(cudaMemcpy(dev_etaByBitIndex, bitsWeight, numBits * sizeof(bundleElt), cudaMemcpyHostToDevice));

  HANDLE_ERROR(cudaMemcpy(dev_eta, checksWeight, numChecks * sizeof(bundleElt), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_lambdaByCheckIndex, checksWeight, numChecks * sizeof(bundleElt), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_cHat, checksWeight, numChecks * sizeof(bundleElt), cudaMemcpyHostToDevice));

  while (P_running) {
    P_cv_worker.wait(lock);
    for (;;) {
      if (P_job_size == 0) break;
      bundleElt *packet_address = P_job_packet_address.back();
      bundleElt *decode_address = P_job_decode_address.back();
      P_job_packet_address.pop_back();
      P_job_decode_address.pop_back();
      P_job_size--;

      lock.unlock();

      returnVal =  ldpcDecoder (P_hmat, P_maxIterations, (packet_address+1), (decode_address +1),
                                dev_rSig, dev_estimate, dev_eta, dev_etaByBitIndex, dev_lambdaByCheckIndex,
                                dev_parityBits, dev_mapRC, dev_mapCR);
      decode_address[0] = make_bundleElt(float(returnVal));
      lock.lock();
    }
  }
}
