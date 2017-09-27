// #define INTERNAL_TIMINGS_4_DECODER ON

// Based on the Eric's C-code implementation of ldpc decode.
//
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

#include "GPUincludes.h"

// #define MIN(a,b)  (((a) < (b)) ? (a) : (b))
#define ABS(a)  (((a) < (0)) ? (-(a)) : (a))
#define MAX_ETA                1e6
#define MIN_TANH_MAGNITUDE     1e-10
#define SCALE_FACTOR           0.75

#define NTHREADS   16
#define CNP_THREADS   20  // checkNodeProcessing threads

__global__ void
checkNodeProcessingOptimal (unsigned int numChecks, unsigned int maxBitsForCheck,
                            float *lambdaByCheckIndex, float *eta);
__global__ void
checkNodeProcessingOptimalBlock (unsigned int numChecks, unsigned int maxBitsForCheck,
                                 float *lambdaByCheckIndex, float *eta);

__global__ void
checkNodeProcessingMinSum (unsigned int numChecks, unsigned int maxBitsForCheck,
                           float *lambdaByCheckIndex, float *eta);

__global__ void
checkNodeProcessingMinSumBlock (unsigned int numChecks, unsigned int maxBitsForCheck,
                                float *lambdaByCheckIndex, float *eta);


__global__ void
checkNodeProcessingOptimalNaive (unsigned int numChecks, unsigned int maxBitsForCheck,
                            float *lambdaByCheckIndex, float *eta) {
  unsigned int m;
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int thisRowLength, thisRowStart, currentIndex;
  float value, product;
  float rowVals[128];

  if (tid < numChecks) {
    m = tid;
    thisRowStart = m * (maxBitsForCheck+1);
    thisRowLength = eta[thisRowStart];

    // Optimal solution using tanh
    // Each thread processes an entire row.

    // compute the tanh values, and temporarily store back into eta
    for (unsigned int n=1; n<= thisRowLength ; n++) {
      currentIndex = thisRowStart+n;
      eta[currentIndex] = tanhf ((eta[currentIndex] - lambdaByCheckIndex[currentIndex]) / 2.0);
    }

    // Compute the product of the other tanh terms for each non-zero elements.
    for (unsigned int n=1; n<= thisRowLength ; n++) {
      product = 1.0;
      for (unsigned int newvar=1; newvar<= thisRowLength; newvar++) {
        if (newvar != n) product=product* eta[thisRowStart+newvar];
      }
      value = -2 *atanhf(product);
      value = (value > MAX_ETA)? MAX_ETA : value;
      value = (value < -MAX_ETA)? -MAX_ETA : value;
      rowVals[n] =  value;
    }
    for (unsigned int n=1; n<= thisRowLength ; n++) { eta[thisRowStart+n] = rowVals[n];}
  }
}

__global__ void
checkNodeProcessingOptimalCheckZero (unsigned int numChecks, unsigned int maxBitsForCheck,
                            float *lambdaByCheckIndex, float *eta) {
  unsigned int m, indexOfZero;
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int thisRowLength, thisRowStart, currentIndex;
  float value, product, arg;
  float rowVals[128];
  int nzeros;

  if (tid < numChecks) {
    nzeros=0;
    indexOfZero = 0;
    m = tid;
    thisRowStart = m * (maxBitsForCheck+1);
    thisRowLength = eta[thisRowStart];
    product = 1.0;

    // Optimal solution using tanh
    // Each thread processes an entire row.
    // Compute the product of the tanh terms for all non-zero elements.
    for (unsigned int n=1; n<= thisRowLength ; n++) {
      currentIndex = thisRowStart+n;
      value =  tanhf ((eta[currentIndex] - lambdaByCheckIndex[currentIndex]) / 2.0);
      rowVals[n] = value;
      if (value == 0.0) {
        nzeros++;
        indexOfZero = n;
      } else {
        product =  product * value;
      }
    }

    // Now, set the value for each element in this row.
    // If there are 2 or more zero's, then the product without
    // any single value will still be zero.
    if (nzeros > 1) {
      for (unsigned int n=1; n<= thisRowLength; n++) {
        currentIndex = thisRowStart+n;
        eta[currentIndex] =  0.0;
      }
    } else if (nzeros == 1) {
      for (unsigned int n=1; n<= thisRowLength; n++) {
        currentIndex = thisRowStart+n;
        eta[currentIndex] =  (n == indexOfZero)?product : 0.0;
      }
    } else {
      for (unsigned int n=1; n<= thisRowLength; n++) {
        currentIndex = thisRowStart+n;
        arg = product/rowVals[n];
        value = -2 *atanhf(arg);
        value = (value > MAX_ETA)? MAX_ETA : value;
        value = (value < -MAX_ETA)? -MAX_ETA : value;
        eta[currentIndex] =  value;
      }
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

// Transpose  checkRows matrix with rows == parity checks, to
//            bitRows matrix  with rows == bits
__global__ void
transposeRC (unsigned int* map, float *checkRows, float *bitRows,
             unsigned int numChecks, unsigned int maxBitsForCheck) {
  // index
  unsigned int check = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int rowStart;
  unsigned int cellIndex, oneDindex;

  if (check < numChecks) {
    rowStart = check * (maxBitsForCheck+1);
    for (unsigned int index=1; index<= map[rowStart]; index++) {
      cellIndex = rowStart + index;
      oneDindex = map[cellIndex];
      bitRows[oneDindex] = checkRows[cellIndex];
    }
  }
}

// copyBitsToCheckmatrix accepts a vector of the current bitEstimates
// and copies them into a checkRow matrix, where each row represents a check.
// It also generates a HardDecision copy of that output matrix checkRows.
__global__ void
copyBitsToCheckmatrix (unsigned int* map, float *bitEstimates, float *checkRows,
                       unsigned int *hd,
                       unsigned int numBits, unsigned int maxChecksForBit) {
  // index
  unsigned int bitIndex = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int rowStart = 0;
  unsigned int cellIndex, oneDindex;
  float thisBitEstimate;

  if (bitIndex < numBits) {
    rowStart = bitIndex * (maxChecksForBit+1);
    thisBitEstimate = bitEstimates[bitIndex];
    for (unsigned int index=1; index<= map[rowStart]; index++) {
      cellIndex = rowStart + index;
      oneDindex = map[cellIndex];
      checkRows[oneDindex] = thisBitEstimate;
      hd[oneDindex] = (thisBitEstimate >= 0) ? 1 : 0;
    }
  }
}

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

  HANDLE_ERROR(cudaMemcpy(dev_rSig, rSig, numBits * sizeof(float), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_mapRC, mapRows2Cols, nChecksByBits * sizeof(unsigned int), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_mapCR, mapCols2Rows, nBitsByChecks * sizeof(unsigned int), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_etaByBitIndex, etaByBitIndex, nBitsByChecks * sizeof(float), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_cHat, cHat, nChecksByBits * sizeof(unsigned int), cudaMemcpyHostToDevice));

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

  HANDLE_ERROR(cudaMemcpy(dev_eta, eta, nChecksByBits * sizeof(float), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_lambdaByCheckIndex, lambdaByCheckIndex, nChecksByBits * sizeof(float), cudaMemcpyHostToDevice));

  for(iterCounter=1;iterCounter<=maxIterations;iterCounter++) {

#ifdef INTERNAL_TIMINGS_4_DECODER
    HANDLE_ERROR(cudaEventRecord(startAt, NULL));
#endif
    // checkNodeProcessingMinSum <<< (numChecks)/NTHREADS+1,NTHREADS>>>(numChecks, maxBitsForCheck, dev_lambdaByCheckIndex, dev_eta);
    // checkNodeProcessingMinSumBlock <<< numChecks,32>>>(numChecks, maxBitsForCheck, dev_lambdaByCheckIndex, dev_eta);
    //  1 thread per block is significantly slower than 2.  (23300 :: 26000 msec) though I think I'm using just 1.
    //  >2 does not help much more.
    // checkNodeProcessingOptimal <<<numChecks/2,2>>>(numChecks, maxBitsForCheck, dev_lambdaByCheckIndex, dev_eta);

    checkNodeProcessingOptimalBlock <<<numChecks, 32>>>(numChecks, maxBitsForCheck, dev_lambdaByCheckIndex, dev_eta);

#ifdef INTERNAL_TIMINGS_4_DECODER
    HANDLE_ERROR( cudaEventRecord(stopAt, NULL));
    HANDLE_ERROR( cudaEventSynchronize(stopAt));
    HANDLE_ERROR( cudaEventElapsedTime(&elapsedTime, startAt, stopAt));
    nodeProcessingTime = nodeProcessingTime + elapsedTime;
#endif

#ifdef INTERNAL_TIMINGS_4_DECODER
    HANDLE_ERROR(cudaEventRecord(startAt, NULL));
#endif
    transposeRC<<<(numChecks)/NTHREADS+1,NTHREADS>>>(dev_mapRC, dev_eta, dev_etaByBitIndex, numChecks, maxBitsForCheck);

#ifdef INTERNAL_TIMINGS_4_DECODER
    HANDLE_ERROR( cudaEventRecord(stopAt, NULL));
    HANDLE_ERROR( cudaEventSynchronize(stopAt));
    HANDLE_ERROR( cudaEventElapsedTime(&elapsedTime, startAt, stopAt));
    transposeTime = transposeTime + elapsedTime;
#endif

    // bit estimates update
#ifdef INTERNAL_TIMINGS_4_DECODER
    HANDLE_ERROR(cudaEventRecord(startAt, NULL));
#endif
    bitEstimates<<<(numBits)/NTHREADS+1,NTHREADS>>>(dev_rSig, dev_etaByBitIndex, dev_lambda, numBits,maxChecksForBit);

#ifdef INTERNAL_TIMINGS_4_DECODER
    HANDLE_ERROR( cudaEventRecord(stopAt, NULL));
    HANDLE_ERROR( cudaEventSynchronize(stopAt));
    HANDLE_ERROR( cudaEventElapsedTime(&elapsedTime, startAt, stopAt));
    bitEstimateTime = bitEstimateTime + elapsedTime;
#endif

#ifdef INTERNAL_TIMINGS_4_DECODER
    HANDLE_ERROR(cudaEventRecord(startAt, NULL));
#endif
    // This resembles the earlier transpose operation, and so
    // this time is accumulated with it.

    // copy lambda (current bit estimates) into
    // a checkMatrix (where each row represents a check
    copyBitsToCheckmatrix<<<(numBits)/NTHREADS+1,NTHREADS>>>(dev_mapCR, dev_lambda, dev_lambdaByCheckIndex,
                                                                 dev_cHat, numBits, maxChecksForBit);

    HANDLE_ERROR(cudaMemcpy(cHat, dev_cHat, nChecksByBits * sizeof(unsigned int),cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(cHat, dev_cHat, nChecksByBits * sizeof(unsigned int),cudaMemcpyDeviceToHost));

#ifdef INTERNAL_TIMINGS_4_DECODER
    HANDLE_ERROR( cudaEventRecord(stopAt, NULL));
    HANDLE_ERROR( cudaEventSynchronize(stopAt));
    HANDLE_ERROR( cudaEventElapsedTime(&elapsedTime, startAt, stopAt));
    transposeTime = transposeTime + elapsedTime;
#endif

    //DEBUG
    // if( iterCounter == 1) {
    //   printf("rSig: %.2f,  %.2f,  %.2f,  %.2f,  %.2f,  %.2f\n",
    //          rSig[0],rSig[1],rSig[2],rSig[3],rSig[4],rSig[5]);
    // }

    // if( iterCounter < 10) {
    //   HANDLE_ERROR(cudaMemcpy(lambda, dev_lambda, numBits * sizeof(float), cudaMemcpyDeviceToHost));
    //   printf("Lambda  Iter: %i: %.2f,  %.2f,  %.2f,  %.2f,  %.2f,  %.2f\n",
    //           iterCounter, lambda[0],lambda[1],lambda[2],lambda[3],lambda[4],lambda[5]);
    // }

    // Check for correct decoding.
    rowStart = 0;
    allChecksPassed = true;
    for (unsigned int check=0; check<numChecks; check++) {
      unsigned int sum = 0;
      for (unsigned int index=1; index<= cHat[rowStart]; index++) {sum = sum + cHat[rowStart + index];}
      if ((sum % 2) != 0) {
        allChecksPassed = false;
        break;
      }
      rowStart = rowStart + maxBitsForCheck+1;
    }
    if (allChecksPassed) {break;}
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
