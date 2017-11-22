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

unsigned int nChecksByBits;
unsigned int nBitsByChecks;

float *eta;
float *etaByBitIndex;
float *lambdaByCheckIndex;
unsigned int *cHat;
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

void initLdpcDecoder  (H_matrix *hmat) {

  unsigned int *mapRows2Cols;
  unsigned int *mapCols2Rows;

  unsigned int numContributors;

  unsigned int numBits = hmat->numBits;
  unsigned int numChecks = hmat->numChecks;
  unsigned int maxBitsPerCheck = hmat->maxBitsPerCheck;
  unsigned int maxChecksPerBit = hmat->maxChecksPerBit;
  unsigned int nChecksByBits = numChecks*(maxBitsPerCheck+1);
  unsigned int nBitsByChecks = numBits*(maxChecksPerBit+1);

  eta = (float *)malloc(nChecksByBits* sizeof(float));
  etaByBitIndex= (float *)malloc(nBitsByChecks* sizeof(float));
  lambdaByCheckIndex = (float *)malloc(nChecksByBits* sizeof(float));
  cHat = (unsigned int *)malloc(nChecksByBits* sizeof(unsigned int));


  // Currently, mapRows2Cols and mapCols2Rows are provided in a row-based order.
  // Each row of a matrix represents a check (or bit) node.
  // In the "new" order, more appropriate for GPU striding, we use a column-based order.
  // Ech column of a matrix represetns a check (or bit) node.
  // So, here, I "transpose" these matrices.
  mapRows2Cols = (unsigned int *) malloc(nChecksByBits * sizeof(unsigned int));
  mapCols2Rows = (unsigned int *) malloc(nBitsByChecks * sizeof(unsigned int));

  remapRows2Cols(numChecks, numBits, maxBitsPerCheck, maxChecksPerBit, hmat->mapRows2Cols, mapRows2Cols);
  remapCols2Rows(numChecks, numBits, maxBitsPerCheck, maxChecksPerBit, hmat->mapCols2Rows, mapCols2Rows);
  hmat->mapRows2Cols = mapRows2Cols;
  hmat->mapCols2Rows = mapCols2Rows;

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

  // All matrices are stored in column order.
  // For eta and lambdaByCheckIndex, each column represents a Check node.
  // There are maxBitsPerCheck+1 rows.
  // row 0 always contains the number of contributors for this check node.
  for (unsigned int check=0; check<numChecks; check++) {
    numContributors = mapRows2Cols[check];
    eta[check] = (float)numContributors;
    lambdaByCheckIndex[check] = (float)numContributors;
    cHat[check] = numContributors;
  }
  // Need to have row 0 (see preceding code segment) in lambdaByCheckIndex and cHat into device memory, now.
  // For each new record, these device memory matrices are updated with a kernel
  HANDLE_ERROR(cudaMemcpy(dev_lambdaByCheckIndex, lambdaByCheckIndex, nChecksByBits * sizeof(float), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_cHat, cHat, nChecksByBits * sizeof(unsigned int), cudaMemcpyHostToDevice));

  // For etaByBitIndex, each column corresponds to a bit node.
  // row 0, contains the number of contributors for this bit.
  for (unsigned int bit=0; bit<numBits; bit++) {
    etaByBitIndex[bit] = (float)mapCols2Rows[bit];
  }
}

int ldpcDecoderWithInit (H_matrix *hmat, float *rSig, unsigned int  maxIterations, unsigned int *decision, float *estimates) {

  unsigned int numBits = hmat->numBits;
  unsigned int numChecks = hmat->numChecks;
  unsigned int maxBitsPerCheck = hmat->maxBitsPerCheck;
  unsigned int maxChecksPerBit = hmat->maxChecksPerBit;
  unsigned int nChecksByBits = numChecks*(maxBitsPerCheck+1);
  unsigned int nBitsByChecks = numBits*(maxChecksPerBit+1);

  unsigned int iterCounter;
  bool allChecksPassed = false;

  HANDLE_ERROR(cudaMemcpy(dev_rSig, rSig, numBits * sizeof(float), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_etaByBitIndex, etaByBitIndex, nBitsByChecks * sizeof(float), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_eta, eta, nChecksByBits * sizeof(float), cudaMemcpyHostToDevice));
  copyBitsToCheckmatrix<<<numBits, NTHREADS>>>(dev_mapCR, dev_rSig, dev_lambdaByCheckIndex, numBits, maxChecksPerBit);

  ////////////////////////////////////////////////////////////////////////////
  // Main iteration loop
  ////////////////////////////////////////////////////////////////////////////

  for(iterCounter=1;iterCounter<=maxIterations;iterCounter++) {
    checkNodeProcessingOptimalBlock <<<numChecks, NTHREADS>>>(numChecks, maxBitsPerCheck,
                                                              dev_lambdaByCheckIndex, dev_eta,
                                                              dev_mapRC, dev_etaByBitIndex);

    bitEstimates<<<(numBits)/NTHREADS+1,NTHREADS>>>(dev_rSig, dev_etaByBitIndex, dev_lambdaByCheckIndex, dev_cHat,
                                                    dev_mapCR, numBits,maxChecksPerBit);

    calcParityBits <<<numChecks/ NTHREADS+1 , NTHREADS>>>(dev_cHat, dev_parityBits, numChecks, maxBitsPerCheck);
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
