// #define INTERNAL_TIMINGS_4_DECODER ON

// Based on the Eric's C-code implementation of ldpc decode.
//
#include <cstdio>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

#include "GPUincludes.h"
#include "LDPC.h"
// For the CUB library
#include "cub.cuh"

#define NTHREADS   16
#define CNP_THREADS   20  // checkNodeProcessing threads

int ldpcDecoder (float *rSig, unsigned int numChecks, unsigned int numBits,
                 unsigned int maxBitsForCheck, unsigned int maxChecksForBit,
                 unsigned int *mapRows2Cols,
                 unsigned int *mapCols2Rows,
                 unsigned int maxIterations,
                 unsigned int *decision,
                 float *estimates) {

  // Number of elements in a checkRows matrix (matrix with rowNum = CheckIndex)
  unsigned int nChecksByBits = numChecks*(maxBitsForCheck+1);
  // Number of elements in a bitRows matrix (matrix with rowNum = BitIndex)
  unsigned int nBitsByChecks = numBits*(maxChecksForBit+1);

  float eta[nChecksByBits];
  float lambda[numBits];
  float etaByBitIndex[nBitsByChecks];
  float lambdaByCheckIndex[nChecksByBits];
  unsigned int cHat [nChecksByBits];
  unsigned int paritySum;
  // unsigned int parityBits[numChecks];

  unsigned int iterCounter;
  bool allChecksPassed = false;

  unsigned int oneDindex;
  unsigned int rowStart;
  unsigned int rowLength;

  float *dev_rSig;
  float *dev_eta;
  float *dev_lambda;
  float *dev_etaByBitIndex;
  float *dev_lambdaByCheckIndex;
  unsigned int *dev_cHat;
  unsigned int *dev_parityBits;
  unsigned int *dev_paritySum;

  unsigned int *dev_mapRC;
  unsigned int *dev_mapCR;

  HANDLE_ERROR( cudaMalloc( (void**)&dev_rSig, numBits * sizeof(float) ));
  HANDLE_ERROR( cudaMalloc( (void**)&dev_eta, nChecksByBits * sizeof(float)));
  HANDLE_ERROR( cudaMalloc( (void**)&dev_lambda, numBits * sizeof(float)));
  HANDLE_ERROR( cudaMalloc( (void**)&dev_etaByBitIndex,  nBitsByChecks * sizeof(float)));
  HANDLE_ERROR( cudaMalloc( (void**)&dev_lambdaByCheckIndex, nChecksByBits * sizeof(float)));
  HANDLE_ERROR( cudaMalloc( (void**)&dev_mapRC, nChecksByBits * sizeof(unsigned int)));
  HANDLE_ERROR( cudaMalloc( (void**)&dev_mapCR, nBitsByChecks * sizeof(unsigned int)));
  HANDLE_ERROR( cudaMalloc( (void**)&dev_cHat, nChecksByBits * sizeof(unsigned int)));
  HANDLE_ERROR( cudaMalloc( (void**)&dev_parityBits, numChecks * sizeof(unsigned int)));
  HANDLE_ERROR( cudaMalloc( (void**)&dev_paritySum, 1 * sizeof(unsigned int)));

  memcpy(lambda, rSig, numBits*sizeof(lambda[0]));
  memset(eta, 0, nChecksByBits*sizeof(eta[0]));
  memset(etaByBitIndex, 0, nBitsByChecks*sizeof(etaByBitIndex[0]));
  memset(lambdaByCheckIndex, 0, nChecksByBits*sizeof(lambdaByCheckIndex[0]));

  // Need to insert rowLengths into eta (and lambdaByCheckIndex)
  // with rows corresponding to parity checks.
  rowStart = 0;
  for (unsigned int check=0; check<numChecks; check++) {
    rowLength = mapRows2Cols[rowStart];
    eta[rowStart] = (float)rowLength;
    lambdaByCheckIndex[rowStart] = (float)rowLength;
    cHat[rowStart] = rowLength;
    rowStart = rowStart + (maxBitsForCheck+1);
  }

  // Need to insert rowLengths into etaByBitIndex
  rowStart = 0;
  for (unsigned int bit=0; bit<numBits; bit++) {
    etaByBitIndex[rowStart] = (float)mapCols2Rows[rowStart];
    rowStart = rowStart + (maxChecksForBit+1);
  }

  // initialization
  // Build a matrix in which every row represents a check
  // and the elements, are the estimates for the bits contributing to this check.

  rowStart = 0;
  for (unsigned int bit=0; bit<numBits; bit++) {
    for (unsigned int index=1; index<=mapCols2Rows[rowStart]; index++) {
      oneDindex  = mapCols2Rows[rowStart +index];
      lambdaByCheckIndex[oneDindex] = lambda[bit];
    }
    rowStart = rowStart + (maxChecksForBit+1);
  }

  // Determine temporary device storage requirements for CUB reduce; and then allocate the space.
  size_t temp_storage_bytes;
  int* temp_storage=NULL;
  cub::DeviceReduce::Sum(temp_storage, temp_storage_bytes, dev_parityBits, dev_paritySum, numChecks);
  cudaMalloc(&temp_storage,temp_storage_bytes);

  HANDLE_ERROR(cudaMemcpy(dev_rSig, rSig, numBits * sizeof(float), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_mapRC, mapRows2Cols, nChecksByBits * sizeof(unsigned int), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_mapCR, mapCols2Rows, nBitsByChecks * sizeof(unsigned int), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_etaByBitIndex, etaByBitIndex, nBitsByChecks * sizeof(float), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_cHat, cHat, nChecksByBits * sizeof(unsigned int), cudaMemcpyHostToDevice));

  ////////////////////////////////////////////////////////////////////////////
  // Main iteration loop
  ////////////////////////////////////////////////////////////////////////////

  HANDLE_ERROR(cudaMemcpy(dev_eta, eta, nChecksByBits * sizeof(float), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_lambdaByCheckIndex, lambdaByCheckIndex, nChecksByBits * sizeof(float), cudaMemcpyHostToDevice));

  for(iterCounter=1;iterCounter<=maxIterations;iterCounter++) {

    // checkNodeProcessingMinSum <<< (numChecks)/NTHREADS+1,NTHREADS>>>(numChecks, maxBitsForCheck, dev_lambdaByCheckIndex, dev_eta);
    // checkNodeProcessingMinSumBlock <<< numChecks,32>>>(numChecks, maxBitsForCheck, dev_lambdaByCheckIndex, dev_eta);
    //  1 thread per block is significantly slower than 2.  (23300 :: 26000 msec) though I think I'm using just 1.
    //  >2 does not help much more.
    // checkNodeProcessingOptimal <<<numChecks/2,2>>>(numChecks, maxBitsForCheck, dev_lambdaByCheckIndex, dev_eta);

    checkNodeProcessingOptimalBlock <<<numChecks, 32>>>(numChecks, maxBitsForCheck, dev_lambdaByCheckIndex, dev_eta);

    transposeRC<<<(numChecks),32>>>(dev_mapRC, dev_eta, dev_etaByBitIndex, numChecks, maxBitsForCheck);

    // bit estimates update
    bitEstimates<<<(numBits)/NTHREADS+1,NTHREADS>>>(dev_rSig, dev_etaByBitIndex, dev_lambda, numBits,maxChecksForBit);

    // This resembles the earlier transpose operation, and so
    // this time is accumulated with it.

    // copy lambda (current bit estimates) into
    // a checkMatrix (where each row represents a check
    copyBitsToCheckmatrix<<<numBits,16>>>(dev_mapCR, dev_lambda, dev_lambdaByCheckIndex,
                                            dev_cHat, numBits, maxChecksForBit);
    calcParityBits <<<numChecks, 2>>>(dev_cHat, dev_parityBits, numChecks, maxBitsForCheck);

    // HANDLE_ERROR(cudaMemcpy(& parityBits,dev_parityBits, numChecks* sizeof(int),cudaMemcpyDeviceToHost));
    // allChecksPassed = true;
    // for (int j=0; j< numChecks; j++) {
    //   if (parityBits[j] != 0) {
    //     allChecksPassed = false;
    //     break;
    //   }
    // }

    cub::DeviceReduce::Sum(temp_storage, temp_storage_bytes, dev_parityBits, dev_paritySum, numChecks);
    HANDLE_ERROR(cudaMemcpy(& paritySum,dev_paritySum,sizeof(int),cudaMemcpyDeviceToHost));
    allChecksPassed =  (paritySum == 0)? true : false;

    if (allChecksPassed) {break;}
  }

  if(allChecksPassed == false) {
    // printf("Decoding failure after %d iterations\n", iterCounter);
  } else {
    HANDLE_ERROR(cudaMemcpy(cHat, dev_cHat, nChecksByBits * sizeof(unsigned int),cudaMemcpyDeviceToHost));
    // Print a status message on the iteration loop
    // printf("Success at %i iterations\n",iterCounter);
  }
  HANDLE_ERROR( cudaFree( dev_rSig));
  HANDLE_ERROR( cudaFree( dev_eta));
  HANDLE_ERROR( cudaFree( dev_lambda));
  HANDLE_ERROR( cudaFree( dev_etaByBitIndex));
  HANDLE_ERROR( cudaFree( dev_lambdaByCheckIndex));
  HANDLE_ERROR( cudaFree( dev_mapRC));
  HANDLE_ERROR( cudaFree( dev_mapCR));
  HANDLE_ERROR( cudaFree( dev_cHat));
  HANDLE_ERROR( cudaFree( dev_parityBits));
  HANDLE_ERROR( cudaFree( dev_paritySum));
  HANDLE_ERROR( cudaFree( temp_storage));

  return (iterCounter);
}
