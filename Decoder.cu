// Based on the Eric's C-code implementation of ldpc decode.
//
#include <cstdio>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

#include "GPUincludes.h"
#include "LDPC.h"

#include "cub.cuh"

#define NTHREADS   32
#define CNP_THREADS   20  // checkNodeProcessing threads

unsigned int numBits, numChecks;
unsigned int maxChecksForBit, maxBitsForCheck;

  unsigned int nChecksByBits;
unsigned int nBitsByChecks;

float *eta;
float *etaByBitIndex;
unsigned int *paritySum;

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

size_t temp_storage_bytes;
int* temp_storage=NULL;

void initLdpcDecoder  (unsigned int numChecksI, unsigned int numBitsI,
                      unsigned int maxBitsForCheckI, unsigned int maxChecksForBitI,
                      unsigned int *mapRows2Cols, unsigned int *mapCols2Rows) {

  unsigned int rowStart, rowLength;
  // NOTE:  these two matrices are local, since the contents are written to device memory
  // and never used again on the host side.
  float *lambdaByCheckIndex;
  unsigned int *cHat;


  numBits = numBitsI;
  numChecks = numChecksI;
  maxBitsForCheck = maxBitsForCheckI;
  maxChecksForBit = maxChecksForBitI;

  nChecksByBits = numChecks*(maxBitsForCheck+1);
  nBitsByChecks = numBits*(maxChecksForBit+1);

  eta = (float *)malloc(nChecksByBits* sizeof(float));
  etaByBitIndex= (float *)malloc(nBitsByChecks* sizeof(float));
  lambdaByCheckIndex = (float *)malloc(nChecksByBits* sizeof(float));
  cHat = (unsigned int *)malloc(nChecksByBits* sizeof(unsigned int));

  // cudaMallocHost for paritySum ensures the value is in "pinned memory",
  // so the DeviceToHost transfer should be faster.
  // Unfortunately, early tests show no improvement
  HANDLE_ERROR( cudaMallocHost((void**)&paritySum, sizeof(unsigned int)));

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

  // Determine temporary device storage requirements for CUB reduce; and then allocate the space.
  cub::DeviceReduce::Sum(temp_storage, temp_storage_bytes, dev_parityBits, dev_paritySum, numChecks);
  HANDLE_ERROR(cudaMalloc(&temp_storage,temp_storage_bytes));

  HANDLE_ERROR(cudaMemcpy(dev_mapRC, mapRows2Cols, nChecksByBits * sizeof(unsigned int), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_mapCR, mapCols2Rows, nBitsByChecks * sizeof(unsigned int), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_cHat, cHat, nChecksByBits * sizeof(unsigned int), cudaMemcpyHostToDevice));

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
  // Need to the rowLengths in lambdaByCheckIndex and cHat into device memory, now.
  // For each new record, these device memory matrices are updated with a kernel
  // (that expects these matrices to contain the rowLengths as the first element of each row.
  HANDLE_ERROR(cudaMemcpy(dev_lambdaByCheckIndex, lambdaByCheckIndex, nChecksByBits * sizeof(float), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_cHat, cHat, nChecksByBits * sizeof(unsigned int), cudaMemcpyHostToDevice));

  // Need to insert rowLengths into etaByBitIndex
  rowStart = 0;
  for (unsigned int bit=0; bit<numBits; bit++) {
    etaByBitIndex[rowStart] = (float)mapCols2Rows[rowStart];
    rowStart = rowStart + (maxChecksForBit+1);
  }
}

int ldpcDecoderWithInit (float *rSig, unsigned int  maxIterations, unsigned int *decision, float *estimates) {

  unsigned int iterCounter;
  bool allChecksPassed = false;

  HANDLE_ERROR(cudaMemcpy(dev_rSig, rSig, numBits * sizeof(float), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_etaByBitIndex, etaByBitIndex, nBitsByChecks * sizeof(float), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_eta, eta, nChecksByBits * sizeof(float), cudaMemcpyHostToDevice));
  copyBitsToCheckmatrix<<<numBits,16>>>(dev_mapCR, dev_rSig, dev_lambdaByCheckIndex, numBits, maxChecksForBit);

  ////////////////////////////////////////////////////////////////////////////
  // Main iteration loop
  ////////////////////////////////////////////////////////////////////////////

  for(iterCounter=1;iterCounter<=maxIterations;iterCounter++) {
    checkNodeProcessingOptimalBlock <<<numChecks, 32>>>(numChecks, maxBitsForCheck, dev_lambdaByCheckIndex, dev_eta,
                                                        dev_mapRC, dev_etaByBitIndex);

      bitEstimates<<<(numBits)/NTHREADS+1,NTHREADS>>>(dev_rSig, dev_etaByBitIndex, dev_lambdaByCheckIndex, dev_cHat,
                                                     dev_mapCR, numBits,maxChecksForBit);

    calcParityBits <<<numChecks, 2>>>(dev_cHat, dev_parityBits, numChecks, maxBitsForCheck);
    cub::DeviceReduce::Sum(temp_storage, temp_storage_bytes, dev_parityBits, dev_paritySum, numChecks);
    HANDLE_ERROR(cudaMemcpy(paritySum,dev_paritySum,sizeof(int),cudaMemcpyDeviceToHost));
    allChecksPassed =  (*paritySum == 0)? true : false;

    if (allChecksPassed) {break;}
  }
  // Return our best guess.
  // if iterCounter < maxIterations, then successful.
  HANDLE_ERROR(cudaMemcpy(estimates, dev_lambda, numBits * sizeof(unsigned int),cudaMemcpyDeviceToHost));
  for (unsigned int i=0; i<numBits; i++) decision[i] = estimates[i] > 0;
  return (iterCounter);
}
