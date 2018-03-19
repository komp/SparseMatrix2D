#include "loader_pool.h"

#include <random>
#include <thread>
#include <vector>
#include <functional>
#include <condition_variable>
#include <mutex>

#include "LDPC.h"

LoaderPool::LoaderPool (unsigned int infoLength, unsigned int numBits, unsigned int numParityBits,
                            unsigned int* W_row_rom, unsigned int numRowsW, unsigned int numColsW, unsigned int shiftRegLength,
                            float sigma2, float lc)
: P_running(true)
    , P_job_size(0u)
    , P_infoLeng(infoLength)
    , P_numBits(numBits)
    , P_numParityBits(numParityBits)
    , P_W_row_rom(W_row_rom)
    , P_numRowsW(numRowsW)
    , P_numColsW(numColsW)
, P_shiftRegLength(shiftRegLength)
,  P_sigma2 (sigma2)
, P_lc (lc)

{
  P_threads.reserve(4);
  for (size_t i = 0; i < 4; i++) {
    P_threads.emplace_back(&LoaderPool::worker_main, this);
  }
}

LoaderPool::~LoaderPool() {
  std::unique_lock<std::mutex> lock(P_mutex);
  P_running = false;
  P_cv_worker.notify_all();
  lock.unlock();
  for (auto& t : P_threads) t.join();
}

void LoaderPool::schedule_job(Tpkt *packet) {
  std::lock_guard<std::mutex> lock(P_mutex);

  P_job_packet.push_back(packet);
  P_job_size++;
  P_cv_worker.notify_one();
}

void LoaderPool::worker_main() {

  std::unique_lock<std::mutex> lock(P_mutex);

  unsigned int  seed = 163331;
  /*  or use this to get a fresh sequence each time the program is run.
  std::random_device  rd;  //Will be used to obtain a seed for the random number engine
  std::mt19937 generator(rd()); //Standard mersenne_twister_engine seeded with rd()
  */
  std::mt19937 generator(seed); //Standard mersenne_twister_engine
  std::uniform_real_distribution<> rDist(0, 1);

  // normal distribution for noise.
  // NOTE,  we are using the same random number generator ("generator") here.
  std::normal_distribution<float> normDist(0.0, 1.0);
  unsigned int* infoWord;
  unsigned int* codeWord;
  float s, noise;
  float* receivedSig;

  infoWord = (unsigned int *)malloc(P_infoLeng * sizeof(unsigned int));
  codeWord = (unsigned int *)malloc((P_infoLeng + P_numParityBits) * sizeof(unsigned int));
  receivedSig = (float *)malloc(P_numBits * sizeof(float));

  while (P_running) {
    P_cv_worker.wait(lock);
    for (;;) {
      if (P_job_size == 0) break;

      Tpkt *packet = P_job_packet.back();
      P_job_packet.pop_back();
      P_job_size--;

      lock.unlock();

      for (unsigned int slot=0; slot < SLOTS_PER_ELT; slot++) {
        for (unsigned int j=0; j < P_infoLeng; j++) {
          infoWord[j] = (0.5 >= rDist(generator))? 1:0;
        }
        ldpcEncoder(infoWord, P_W_row_rom, P_infoLeng, P_numRowsW, P_numColsW, P_shiftRegLength, codeWord);

        for (unsigned int j=0; j < (P_infoLeng+P_numParityBits) ; j++) {
          s     = 2*float(codeWord[j]) - 1;
          // AWGN channel
          noise = sqrt(P_sigma2) * normDist(generator);
          // When r is scaled by Lc it results in precisely scaled LLRs
          receivedSig[j]  = P_lc*(s + noise);
        }
        // The LDPC codes are punctured, so the r we feed to the decoder is
        // longer than the r we got from the channel. The punctured positions are filled in as zeros
        for (unsigned int j=(P_infoLeng+P_numParityBits); j<P_numBits; j++) receivedSig[j] = 0.0;
        for (unsigned int j=0; j < P_numBits; j++ ) packet->receivedSigs[j].s[slot] = receivedSig[j];
      }
      lock.lock();

      packet->loadStamp = 1.0;
    }
  }
}
