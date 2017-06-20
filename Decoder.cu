// #define INTERNAL_TIMINGS_4_DECODER ON

// Based on the Eric's C-code implementation of ldpc decode.
//
#include <math.h>
#include <string.h>
#include <time.h>

#include "GPUincludes.h"

// #define MIN(a,b)  (((a) < (b)) ? (a) : (b))
#define ABS(a)  (((a) < (0)) ? (-(a)) : (a))
#define MAX_ETA                1e6
#define SCALE_FACTOR           0.75

#define NTHREADS    128

__global__ void
checkNodeProcessing (unsigned int numChecks, unsigned int maxBitsForCheck,
                      // eta is IN and OUT
                      float *lambdaByCheckIndex, float *eta) {
  // edk  HACK !!!
  // This was signs[maxBitsForCheck], which generates the error:
  // error: constant value is not known.
  // Since we are in a kernel function, we probably need a compile-time constant.
  // 128 should be much larger than maxBitsForCheck for any reasonable LDPC encoding.
  unsigned int signs[128];
  unsigned int signProduct;
  float value, min1, min2;
  unsigned int minIndex;

  // index
  unsigned int m;
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int thisRowLength, thisRowStart, currentIndex;

  if (tid < numChecks) {
    m = tid;
    thisRowStart = m * (maxBitsForCheck+1);
    // signs[n]  == 0  ==>  positive; 1  ==>  negative
    memset(signs, 0, (maxBitsForCheck+1)*sizeof(signs[0]));
    signProduct = 0;
    min1 = MAX_ETA;
    min2 =  MAX_ETA;
    minIndex = 1;
    thisRowLength = eta[thisRowStart];
    for (unsigned int n=1; n<= thisRowLength ; n++) {
      currentIndex = thisRowStart+n;
      value = eta[currentIndex] - lambdaByCheckIndex[currentIndex];
      signs[n] = (value < 0)? 1 : 0;
      signProduct = (signProduct != signs[n])? 1 : 0;
      value = ABS(value);
      if (value < min1) {
        min2 = min1;
        min1 = value;
        minIndex = n;
      } else if ( value < min2) {
        min2 = value;
      }
    }
    min1 = min1 * SCALE_FACTOR * (-1);
    min2 = min2 * SCALE_FACTOR * (-1);
    for (unsigned int n=1; n<= thisRowLength; n++) {
      currentIndex = thisRowStart+n;
      eta[currentIndex] =  (n == minIndex) ? min2 : min1;
      if (signs[n] != signProduct) {eta[currentIndex] = -eta[currentIndex];}
    }
  }
}

__global__ void
bitEstimates(float *rSig, float *etaByBitIndex, float *lambda,
             unsigned int numBits, unsigned int maxChecksForBit) {

  unsigned int n;
  unsigned int thisRowLength, thisRowStart;
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid < numBits) {
    n = tid;
    float sum = rSig[n];
    thisRowStart = n*(maxChecksForBit+1);
    thisRowLength = etaByBitIndex[thisRowStart];
    for (unsigned int m=1; m<=thisRowLength; m++) {
      sum = sum + etaByBitIndex[thisRowStart +m];
    }
    lambda[n] = sum;
  }
}

int ldpcDecoder (float *rSig, unsigned int numChecks, unsigned int numBits,
                 unsigned int maxBitsForCheck, unsigned int maxChecksForBit,
                 unsigned int *mapRows2Cols,
                 unsigned int *mapCols2Rows,
                 unsigned int maxIterations,
                 unsigned int *decision,
                 float *estimates) {

  unsigned int nChecksByBits = numChecks*(maxBitsForCheck+1);
  unsigned int nBitsByChecks = numBits*(maxChecksForBit+1);

  float eta[nChecksByBits];
  float lambda[numBits];
  float etaByBitIndex[nBitsByChecks];
  float lambdaByCheckIndex[nChecksByBits];
  unsigned int cHat [nChecksByBits];

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

  HANDLE_ERROR( cudaMalloc( (void**)&dev_rSig, numBits * sizeof(float) ) );
  HANDLE_ERROR( cudaMalloc( (void**)&dev_eta, nChecksByBits * sizeof(float) ) );
  HANDLE_ERROR( cudaMalloc( (void**)&dev_lambda, numBits * sizeof(float) ) );
  HANDLE_ERROR( cudaMalloc( (void**)&dev_etaByBitIndex,  nBitsByChecks * sizeof(float) ) );
  HANDLE_ERROR( cudaMalloc( (void**)&dev_lambdaByCheckIndex, nChecksByBits * sizeof(float) ) );
  //  HANDLE_ERROR( cudaMalloc( (void**)&dev_cHat, nChecksByBits * sizeof(unsigned int) ) );

  memcpy(lambda, rSig, numBits*sizeof(lambda[0]));
  memset(eta, 0, nChecksByBits*sizeof(eta[0]));
  memset(lambdaByCheckIndex, 0, nChecksByBits*sizeof(eta[0]));

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

  HANDLE_ERROR(cudaMemcpy(dev_rSig, rSig, numBits * sizeof(float), cudaMemcpyHostToDevice));

#ifdef INTERNAL_TIMINGS_4_DECODER
  float elapsedTime, partTimes;
  float  allTime = 0.0, nodeProcessingTime = 0.0, bitEstimateTime = 0.0, transposeTime = 0.0;
  cudaEvent_t globalStart;
  HANDLE_ERROR(cudaEventCreate(&globalStart));
  cudaEvent_t startAt;
  HANDLE_ERROR(cudaEventCreate(&startAt));
  cudaEvent_t stopAt;
  HANDLE_ERROR(cudaEventCreate(&stopAt));
  HANDLE_ERROR(cudaEventRecord(globalStart, NULL));
#endif

  ////////////////////////////////////////////////////////////////////////////
  // Main iteration loop
  ////////////////////////////////////////////////////////////////////////////

  for(iterCounter=1;iterCounter<=maxIterations;iterCounter++) {

    HANDLE_ERROR(cudaMemcpy(dev_eta, eta, nChecksByBits * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_lambdaByCheckIndex, lambdaByCheckIndex, nChecksByBits * sizeof(float), cudaMemcpyHostToDevice));

#ifdef INTERNAL_TIMINGS_4_DECODER
    HANDLE_ERROR(cudaEventRecord(startAt, NULL));
#endif
    // checkNode Processing  (1536)
    checkNodeProcessing<<< (1535+NTHREADS)/NTHREADS,NTHREADS>>>(numChecks, maxBitsForCheck, dev_lambdaByCheckIndex, dev_eta);
    cudaDeviceSynchronize();

#ifdef INTERNAL_TIMINGS_4_DECODER
    HANDLE_ERROR( cudaEventRecord(stopAt, NULL));
    HANDLE_ERROR( cudaEventSynchronize(stopAt));
    HANDLE_ERROR( cudaEventElapsedTime(&elapsedTime, startAt, stopAt));
    nodeProcessingTime = nodeProcessingTime + elapsedTime;
#endif
    HANDLE_ERROR(cudaMemcpy(eta, dev_eta, nChecksByBits * sizeof(float), cudaMemcpyDeviceToHost));

#ifdef INTERNAL_TIMINGS_4_DECODER
    HANDLE_ERROR(cudaEventRecord(startAt, NULL));
#endif
    // Transpose  eta with rows == parity checks, to rows == bits
    rowStart = 0;
    for (unsigned int check=0; check<numChecks; check++) {
      for (unsigned int index=1; index<= mapRows2Cols[rowStart]; index++) {
        oneDindex = mapRows2Cols[rowStart + index];
        etaByBitIndex[oneDindex] = eta[rowStart + index];
      }
      rowStart = rowStart + (maxBitsForCheck+1);
    }

#ifdef INTERNAL_TIMINGS_4_DECODER
    HANDLE_ERROR( cudaEventRecord(stopAt, NULL));
    HANDLE_ERROR( cudaEventSynchronize(stopAt));
    HANDLE_ERROR( cudaEventElapsedTime(&elapsedTime, startAt, stopAt));
    transposeTime = transposeTime + elapsedTime;
#endif

    // bit estimates update
    HANDLE_ERROR(cudaMemcpy(dev_etaByBitIndex, etaByBitIndex, nBitsByChecks * sizeof(float), cudaMemcpyHostToDevice));
#ifdef INTERNAL_TIMINGS_4_DECODER
    HANDLE_ERROR(cudaEventRecord(startAt, NULL));
#endif
    bitEstimates<<<(2560+NTHREADS)/NTHREADS,NTHREADS>>>(dev_rSig, dev_etaByBitIndex, dev_lambda, numBits,maxChecksForBit);
    cudaDeviceSynchronize();
#ifdef INTERNAL_TIMINGS_4_DECODER
    HANDLE_ERROR( cudaEventRecord(stopAt, NULL));
    HANDLE_ERROR( cudaEventSynchronize(stopAt));
    HANDLE_ERROR( cudaEventElapsedTime(&elapsedTime, startAt, stopAt));
    bitEstimateTime = bitEstimateTime + elapsedTime;
#endif
    HANDLE_ERROR(cudaMemcpy(lambda, dev_lambda, numBits * sizeof(float), cudaMemcpyDeviceToHost));

#ifdef INTERNAL_TIMINGS_4_DECODER
    HANDLE_ERROR(cudaEventRecord(startAt, NULL));
#endif
    // Transpose  lambda with rows == bits, to rows == parity checks
    rowStart = 0;
    for (unsigned int n=0; n<numBits; n++) {
      decision[n] = (lambda[n] >= 0) ? 1 : 0;
      estimates[n] = lambda[n];
      for (unsigned int index=1; index<=mapCols2Rows[rowStart]; index++) {
        oneDindex  = mapCols2Rows[rowStart + index];
        lambdaByCheckIndex[oneDindex] = lambda[n];
        cHat[oneDindex] = decision[n];
      }
      rowStart = rowStart + (maxChecksForBit+1);
    }
#ifdef INTERNAL_TIMINGS_4_DECODER
    HANDLE_ERROR( cudaEventRecord(stopAt, NULL));
    HANDLE_ERROR( cudaEventSynchronize(stopAt));
    HANDLE_ERROR( cudaEventElapsedTime(&elapsedTime, startAt, stopAt));
    transposeTime = transposeTime + elapsedTime;
#endif

    // Check for correct decoding.
    rowStart = 0;
    allChecksPassed = true;
    for (unsigned int check=0; check<numChecks; check++) {
      unsigned int sum = 0;
      for (unsigned int index=1; index<= cHat[rowStart]; index++) {sum = sum + cHat[rowStart + index];}
      if ((sum % 2) != 0 ) {
        allChecksPassed = false;
        break;
      }
      rowStart = rowStart + maxBitsForCheck+1;
    }
    if (allChecksPassed) {
      break;}
  }

  if(allChecksPassed == false) {
    // printf("Decoding failure after %d iterations\n", iterCounter);
  } else {
    // Print a status message on the iteration loop
    // printf("Success at %i iterations\n",iterCounter);
  }

#ifdef INTERNAL_TIMINGS_4_DECODER
  HANDLE_ERROR( cudaEventRecord(stopAt, NULL));
  HANDLE_ERROR( cudaEventSynchronize(stopAt));
  HANDLE_ERROR( cudaEventElapsedTime(&elapsedTime, globalStart, stopAt));
  allTime = elapsedTime;
  partTimes = nodeProcessingTime + bitEstimateTime + transposeTime;

  printf("\n");
  printf ("Total Time      : %.1f microsec\n", 1000*allTime);
  printf ("node processing : %.1f microsec (%.2f%)\n", 1000*nodeProcessingTime, 100 *nodeProcessingTime/allTime);
  printf ("bit estimates   : %.1f microsec (%.2f%)\n", 1000*bitEstimateTime, 100 * bitEstimateTime/allTime);
  printf ("transpose       : %.1f microsec (%.2f%)\n", 1000*transposeTime, 100 * transposeTime/allTime);
  printf ("Other???        : %.1f microsec (%.2f%)\n", 1000*(allTime - partTimes) , 100 * (allTime - partTimes)/allTime);
  printf("\n");
#endif

  return (iterCounter);
}
