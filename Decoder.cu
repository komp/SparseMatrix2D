#define INTERNAL_TIMINGS_4_DECODER ON
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
checkNodeProcessingC (
                      // numChecks and bitsForCheck are constant for program lifecycle.
                      unsigned int numChecks, unsigned int *bitsForCheck,
                      unsigned int maxBitsForCheck,
                      // lambdaByCheckIndex is IN only (but changes on each invocation).
                      float *lambdaByCheckIndex,
                      // eta is IN and OUT
                      float *eta) {
  unsigned checkIndex;
  // edk  HACK !!!
  // This was signs[maxBitsForCheck], which generates the error:
  // error: constant value is not known.
  // Since we are in a kernel function, we probably need a compile-time constant.
  // 128 should be much larger than maxBitsForCheck for any reasonable LDPC encoding.
  int signs[128];
  unsigned int signProduct;
  float value, min1, min2;
  unsigned int minIndex;

  // index
  int m;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid < numChecks) {
    m = tid;
    checkIndex = tid * maxBitsForCheck;
    // signs[n]  == 0  ==>  positive
    //           == 1  ==>  negative
    memset(signs, 0, maxBitsForCheck*sizeof(signs[0]));
    signProduct = 0;
    min1 = MAX_ETA;
    min2 =  MAX_ETA;
    minIndex = 0;
    for (unsigned int n=0; n<bitsForCheck[m]; n++) {
      value = eta[checkIndex+n] - lambdaByCheckIndex[checkIndex+n];
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
    for (unsigned int n=0; n<bitsForCheck[m]; n++) {
      eta[checkIndex+n] =  (n == minIndex) ? min2 : min1;
      if (signs[n] != signProduct) {eta[checkIndex+n] = -eta[checkIndex+n];}
    }
  }
}

__global__ void
bitEstimatesC(float *rSig, unsigned int *checksForBit,
              // etaByBitIndex is IN only (but changes on each invocation).
              float *etaByBitIndex,
              // lambda and decision are OUT only
              float *lambda, int *decision,
              unsigned int numBits, unsigned int maxChecksForBit) {
  // index
  int n;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid < numBits) {
    n = tid;
    float sum = rSig[n];
    for (unsigned int m=0; m<checksForBit[n]; m++) {
      sum = sum + etaByBitIndex[tid*maxChecksForBit +m];
    }
    lambda[n] = sum;
    decision[n] = (sum >= 0) ? 1 : 0;
  }
}

int ldpcDecoder (float *rSig, unsigned int numChecks, unsigned int numBits,
                 unsigned int *bitsForCheck, unsigned int *checksForBit,
                 int maxBitsForCheck, int maxChecksForBit,
                 unsigned int *mapRows2Cols,
                 unsigned int *mapCols2Rows,
                 unsigned int maxIterations,
                 int *decision) {

  unsigned int nChecksByBits = numChecks*maxBitsForCheck;
  unsigned int nBitsByChecks = numBits*maxChecksForBit;

  float eta[nChecksByBits];
  float lambda[numBits];
  float etaByBitIndex[nBitsByChecks];
  float lambdaByCheckIndex[nChecksByBits];
  int cHat [nChecksByBits];

  unsigned int iterCounter;
  bool allChecksPassed = false;

  unsigned int oneDindex;
  unsigned int bitIndex = 0;
  unsigned int checkIndex = 0;

  float *dev_rSig;
  float *dev_eta;
  float *dev_lambda;
  float *dev_etaByBitIndex;
  float *dev_lambdaByCheckIndex;
  unsigned int *dev_bitsForCheck;
  unsigned int *dev_checksForBit;
  int *dev_decision;

  HANDLE_ERROR( cudaMalloc( (void**)&dev_rSig, numBits * sizeof(float) ) );
  HANDLE_ERROR( cudaMalloc( (void**)&dev_bitsForCheck, numChecks * sizeof(unsigned int) ) );
  HANDLE_ERROR( cudaMalloc( (void**)&dev_checksForBit, numBits * sizeof(unsigned int) ) );

  HANDLE_ERROR( cudaMalloc( (void**)&dev_eta, nChecksByBits * sizeof(float) ) );
  HANDLE_ERROR( cudaMalloc( (void**)&dev_lambda, numBits * sizeof(float) ) );
  HANDLE_ERROR( cudaMalloc( (void**)&dev_etaByBitIndex,  nBitsByChecks * sizeof(float) ) );
  HANDLE_ERROR( cudaMalloc( (void**)&dev_lambdaByCheckIndex, nChecksByBits * sizeof(float) ) );
  HANDLE_ERROR( cudaMalloc( (void**)&dev_decision, numBits * sizeof(unsigned int) ) );
  //  HANDLE_ERROR( cudaMalloc( (void**)&dev_cHat, nChecksByBits * sizeof(unsigned int) ) );

  memcpy(lambda, rSig, numBits*sizeof(lambda[0]));
  memset(eta, 0, numChecks*maxBitsForCheck*sizeof(eta[0]));

  // initialization
  // Build a matrix in which every row represents a check
  // and the elements, are the estimates for the bits contributing to this check.

  for (unsigned int bit=0; bit<numBits; bit++) {
    for (unsigned int index=0; index<checksForBit[bit]; index++) {
      oneDindex  = mapCols2Rows[bitIndex +index];
      lambdaByCheckIndex[oneDindex] = lambda[bit];
    }
    bitIndex = bitIndex + maxChecksForBit;
  }

  HANDLE_ERROR(cudaMemcpy(dev_rSig, rSig, numBits * sizeof(float), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_bitsForCheck, bitsForCheck, numChecks * sizeof(unsigned int), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_checksForBit, checksForBit, numBits  * sizeof(unsigned int), cudaMemcpyHostToDevice));

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
    checkNodeProcessingC<<< (1535+NTHREADS)/NTHREADS,NTHREADS>>>(numChecks, dev_bitsForCheck, maxBitsForCheck, dev_lambdaByCheckIndex, dev_eta);
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
    // Transpose  eta with rows == checkIndex, to rows == bitIndex
    checkIndex = 0;
    for (unsigned int check=0; check<numChecks; check++) {
      for (unsigned int index=0; index<bitsForCheck[check]; index++) {
        oneDindex = mapRows2Cols[checkIndex + index];
        etaByBitIndex[oneDindex] = eta[checkIndex + index];
      }
      checkIndex = checkIndex + maxBitsForCheck;
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
    bitEstimatesC<<<(2560+NTHREADS)/NTHREADS,NTHREADS>>>(dev_rSig, dev_checksForBit, dev_etaByBitIndex, dev_lambda, dev_decision, numBits,maxChecksForBit);
    cudaDeviceSynchronize();
#ifdef INTERNAL_TIMINGS_4_DECODER
    HANDLE_ERROR( cudaEventRecord(stopAt, NULL));
    HANDLE_ERROR( cudaEventSynchronize(stopAt));
    HANDLE_ERROR( cudaEventElapsedTime(&elapsedTime, startAt, stopAt));
    bitEstimateTime = bitEstimateTime + elapsedTime;
#endif
    HANDLE_ERROR(cudaMemcpy(lambda, dev_lambda, numBits * sizeof(float), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(decision, dev_decision, numBits * sizeof(unsigned int), cudaMemcpyDeviceToHost));

#ifdef INTERNAL_TIMINGS_4_DECODER
    HANDLE_ERROR(cudaEventRecord(startAt, NULL));
#endif
    // Transpose  lambda with rows == bitIndex, to rows == checkIndex
    bitIndex = 0;
    for (unsigned int n=0; n<numBits; n++) {
      for (unsigned int index=0; index<checksForBit[n]; index++) {
        oneDindex  = mapCols2Rows[bitIndex + index];
        lambdaByCheckIndex[oneDindex] = lambda[n];
        cHat[oneDindex] = decision[n];
      }
      bitIndex = bitIndex + maxChecksForBit;
    }
#ifdef INTERNAL_TIMINGS_4_DECODER
    HANDLE_ERROR( cudaEventRecord(stopAt, NULL));
    HANDLE_ERROR( cudaEventSynchronize(stopAt));
    HANDLE_ERROR( cudaEventElapsedTime(&elapsedTime, startAt, stopAt));
    transposeTime = transposeTime + elapsedTime;
#endif

    // Check for correct decoding.
    checkIndex = 0;
    allChecksPassed = true;
    for (unsigned int check=0; check<numChecks; check++) {
      int sum = 0;
      for (unsigned int index=0; index<bitsForCheck[check]; index++) {sum = sum + cHat[checkIndex + index];}
      if ((sum % 2) != 0 ) {
        allChecksPassed = false;
        break;
      }
      checkIndex = checkIndex + maxBitsForCheck;
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
