// Based on the Eric's C-code implementation of ldpc decode.
//
#include <cstdio>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

#include "GPUincludes.h"
#include "LDPC.h"

#define NTHREADS   64
#define CNP_THREADS   20  // checkNodeProcessing threads

unsigned int nChecksByBits;
unsigned int nBitsByChecks;

bundleElt *eta;
bundleElt *etaByBitIndex;
bundleElt *lambdaByCheckIndex;
bundleElt *cHat;
bundleElt *parityBits;
bundleElt paritySum;

bundleElt *dev_rSig;
bundleElt *dev_eta;
bundleElt *dev_lambda;
bundleElt *dev_etaByBitIndex;
bundleElt *dev_lambdaByCheckIndex;
bundleElt *dev_cHat;
bundleElt *dev_parityBits;

unsigned int *dev_mapRC;
unsigned int *dev_mapCR;

size_t temp_storage_bytes;
int* temp_storage=NULL;

void initLdpcDecoder  (H_matrix *hmat) {

  unsigned int *mapRows2Cols = hmat->mapRows2Cols;
  unsigned int *mapCols2Rows = hmat->mapCols2Rows;
  unsigned int numBits = hmat->numBits;
  unsigned int numChecks = hmat->numChecks;
  unsigned int maxBitsPerCheck = hmat->maxBitsPerCheck;
  unsigned int maxChecksPerBit = hmat->maxChecksPerBit;

  unsigned int numContributors;
  unsigned int nChecksByBits = numChecks*(maxBitsPerCheck+1);
  unsigned int nBitsByChecks = numBits*(maxChecksPerBit+1);

  eta = (bundleElt *)malloc(nChecksByBits* sizeof(bundleElt));
  etaByBitIndex= (bundleElt *)malloc(nBitsByChecks* sizeof(bundleElt));
  lambdaByCheckIndex = (bundleElt *)malloc(nChecksByBits* sizeof(bundleElt));
  cHat = (bundleElt *)malloc(nChecksByBits* sizeof(bundleElt));
  parityBits = (bundleElt *)malloc(numChecks * sizeof(bundleElt));

  HANDLE_ERROR( cudaMalloc( (void**)&dev_rSig, numBits * sizeof(bundleElt) ));
  HANDLE_ERROR( cudaMalloc( (void**)&dev_eta, nChecksByBits * sizeof(bundleElt)));
  HANDLE_ERROR( cudaMalloc( (void**)&dev_lambda, numBits * sizeof(bundleElt)));
  HANDLE_ERROR( cudaMalloc( (void**)&dev_etaByBitIndex,  nBitsByChecks * sizeof(bundleElt)));
  HANDLE_ERROR( cudaMalloc( (void**)&dev_lambdaByCheckIndex, nChecksByBits * sizeof(bundleElt)));
  HANDLE_ERROR( cudaMalloc( (void**)&dev_mapRC, nChecksByBits * sizeof(unsigned int)));
  HANDLE_ERROR( cudaMalloc( (void**)&dev_mapCR, nBitsByChecks * sizeof(unsigned int)));
  HANDLE_ERROR( cudaMalloc( (void**)&dev_cHat, nChecksByBits * sizeof(bundleElt)));
  HANDLE_ERROR( cudaMalloc( (void**)&dev_parityBits, numChecks * sizeof(bundleElt)));

  HANDLE_ERROR(cudaMemcpy(dev_mapRC, mapRows2Cols, nChecksByBits * sizeof(unsigned int), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_mapCR, mapCols2Rows, nBitsByChecks * sizeof(unsigned int), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_cHat, cHat, nChecksByBits * sizeof(bundleElt), cudaMemcpyHostToDevice));

  memset(eta, 0, nChecksByBits*sizeof(eta[0]));
  memset(etaByBitIndex, 0, nBitsByChecks*sizeof(etaByBitIndex[0]));
  memset(lambdaByCheckIndex, 0, nChecksByBits*sizeof(lambdaByCheckIndex[0]));

  // All matrices are stored in column order.
  // For eta and lambdaByCheckIndex, each column represents a Check node.
  // There are maxBitsPerCheck+1 rows.
  // row 0 always contains the number of contributors for this check node.
  for (unsigned int check=0; check<numChecks; check++) {
    numContributors = mapRows2Cols[check];
    eta[check] = make_bundleElt((float)numContributors);
    lambdaByCheckIndex[check] = make_bundleElt((float)numContributors);
    cHat[check] = make_bundleElt((float)numContributors);
  }
  // Need to have row 0 (see preceding code segment) in lambdaByCheckIndex and cHat into device memory, now.
  // For each new record, these device memory matrices are updated with a kernel
  HANDLE_ERROR(cudaMemcpy(dev_lambdaByCheckIndex, lambdaByCheckIndex, nChecksByBits * sizeof(bundleElt), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_cHat, cHat, nChecksByBits * sizeof(bundleElt), cudaMemcpyHostToDevice));

  // For etaByBitIndex, each column corresponds to a bit node.
  // row 0, contains the number of contributors for this bit.
  for (unsigned int bit=0; bit<numBits; bit++) {
    etaByBitIndex[bit] = make_bundleElt((float)mapCols2Rows[bit]);
  }
}

int ldpcDecoderWithInit (H_matrix *hmat, bundleElt *rSig, unsigned int  maxIterations, unsigned int *decision, bundleElt *estimates) {

  unsigned int numBits = hmat->numBits;
  unsigned int numChecks = hmat->numChecks;
  unsigned int maxBitsPerCheck = hmat->maxBitsPerCheck;
  unsigned int maxChecksPerBit = hmat->maxChecksPerBit;
  unsigned int nChecksByBits = numChecks*(maxBitsPerCheck+1);
  unsigned int nBitsByChecks = numBits*(maxChecksPerBit+1);

  unsigned int iterCounter;
  bool allChecksPassed = false;
  unsigned int successCount;
  unsigned int returnVal;

  HANDLE_ERROR(cudaMemcpy(dev_rSig, rSig, numBits * sizeof(bundleElt), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_etaByBitIndex, etaByBitIndex, nBitsByChecks * sizeof(bundleElt), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_eta, eta, nChecksByBits * sizeof(bundleElt), cudaMemcpyHostToDevice));
  copyBitsToCheckmatrix<<<numBits, NTHREADS>>>(dev_mapCR, dev_rSig, dev_lambdaByCheckIndex, numBits, maxChecksPerBit);

  ////////////////////////////////////////////////////////////////////////////
  // Main iteration loop
  ////////////////////////////////////////////////////////////////////////////

  for(iterCounter=1;iterCounter<=maxIterations;iterCounter++) {
    checkNodeProcessingOptimalBlock <<<numChecks, CNP_THREADS>>>(numChecks, maxBitsPerCheck,
                                                              dev_lambdaByCheckIndex, dev_eta,
                                                              dev_mapRC, dev_etaByBitIndex);

    bitEstimates<<<(numBits)/NTHREADS+1,NTHREADS>>>(dev_rSig, dev_etaByBitIndex, dev_lambdaByCheckIndex, dev_cHat,
                                                    dev_mapCR, numBits,maxChecksPerBit);

    calcParityBits <<<numChecks/ NTHREADS+1 , NTHREADS>>>(dev_cHat, dev_parityBits, numChecks, maxBitsPerCheck);
    allChecksPassed = true;

    //  The cpu is slightly faster than GPU DeviceReduce  to determine if any paritycheck is non-zero.
    HANDLE_ERROR(cudaMemcpy(parityBits, dev_parityBits, numChecks*sizeof(bundleElt),cudaMemcpyDeviceToHost));
    paritySum = make_bundleElt(0.0);
    for (unsigned int check=0; check < numChecks; check++) {
      for (unsigned int slot=0; slot< SLOTS_PER_ELT; slot++) if ((int)parityBits[check].s[slot] != 0) allChecksPassed = false;
      if (! allChecksPassed) break;
    }
    if (allChecksPassed) break;
  }
  // Return our best guess.
  // if iterCounter < maxIterations, then successful.
  paritySum = make_bundleElt(0.0);
  for (unsigned int check=0; check < numChecks; check++) paritySum += parityBits[check];
  successCount = 0;
  for (unsigned int slot=0; slot< SLOTS_PER_ELT; slot++) if ((int)paritySum.s[slot] == 0) successCount++;

  returnVal = (iterCounter << 4) + successCount;
  return (returnVal);
}
