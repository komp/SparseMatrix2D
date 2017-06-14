// Based on the Eric's C-code implementation of ldpc decode.
//
#include <math.h>
#include <string.h>

#include <iostream>
#include <chrono>
#include <ctime>

#include "mat.h"
#include "matrix.h"

#define MIN(a,b)  (((a) < (b)) ? (a) : (b))
#define ABS(a)  (((a) < (0)) ? (-(a)) : (a))
#define MAX_ETA                1e6
#define SCALE_FACTOR           0.75

void checkNodeProcessing (
                          // numChecks and bitsForCheck are constant for program lifecycle.
                          const unsigned int numChecks, const unsigned int *bitsForCheck,
                          const unsigned int maxBitsForCheck,
                          // lambdaByCheckIndex is IN only (but changes on each invocation).
                          const double *lambdaByCheckIndex,
                          // eta is IN and OUT
                          double *eta) {
  int signs[maxBitsForCheck];
  int signProduct;
  double value, min1, min2;
  unsigned int minIndex;
  unsigned int checkIndex=0;

  for (unsigned int m =0; m<numChecks; m++) {
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
    checkIndex = checkIndex + maxBitsForCheck;
  }
}

void bitEstimates (
                   // rSig, numBits and checksForBit are constant for program lifecycle.
                   const double *rSig, const unsigned int numBits, const unsigned int *checksForBit,
                   const unsigned int maxChecksForBit,
                   // etaByBitIndex is IN only (but changes on each invocation).
                   const double *etaByBitIndex,
                   // lambda and decision are OUT only
                   double *lambda, unsigned char *decision) {

  unsigned int bitIndex = 0;
  for (unsigned int n=0; n<numBits; n++) {
    double sum = rSig[n];
    for (unsigned int m=0; m<checksForBit[n]; m++) {sum = sum + etaByBitIndex[bitIndex+m];}
    lambda[n] = sum;
    decision[n] = (sum >= 0) ? 1 : 0;
    bitIndex = bitIndex + maxChecksForBit;
  }
}

int ldpcDecoder (const double *rSig, const unsigned int numChecks, const unsigned int numBits,
                  const unsigned int *bitsForCheck, const unsigned int *checksForBit,
                  const int maxBitsForCheck, const int maxChecksForBit,
                  const unsigned int *mapRows2Cols,
                  const unsigned int *mapCols2Rows,
                  const unsigned int maxIterations,
                  unsigned char *decision) {

  double eta[numChecks*maxBitsForCheck];
  double lambda[numBits];

  double etaByBitIndex[numBits*maxChecksForBit];
  double lambdaByCheckIndex[numChecks*maxBitsForCheck];
  double cHat [numChecks*maxBitsForCheck];

  unsigned int iterCounter;
  bool allChecksPassed = false;

  unsigned int oneDindex;
  unsigned int ptrIndex;
  unsigned int bitIndex = 0;
  unsigned int checkIndex = 0;

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

  ////////////////////////////////////////////////////////////////////////////
  // Main iteration loop
  ////////////////////////////////////////////////////////////////////////////

#ifdef INTERNAL_TIMINGS_4_DECODER
  auto startTime = std::chrono::high_resolution_clock::now();
  auto endTime = std::chrono::high_resolution_clock::now();
  long int delta = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
  long int npTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
  long int beTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
  long int transposeTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
  beTime = 0;
  npTime = 0;
  transposeTime = 0;
#endif
  auto globalStart = std::chrono::high_resolution_clock::now();

  for(iterCounter=1;iterCounter<=maxIterations;iterCounter++) {

    // checkNode Processing
#ifdef INTERNAL_TIMINGS_4_DECODER
    startTime = std::chrono::high_resolution_clock::now();
#endif
    checkNodeProcessing(numChecks, bitsForCheck, maxBitsForCheck, lambdaByCheckIndex, eta);
#ifdef INTERNAL_TIMINGS_4_DECODER
    endTime = std::chrono::high_resolution_clock::now();
    delta = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
    npTime = npTime + delta;
#endif

    // Transpose  eta with rows == checkIndex, to rows == bitIndex
#ifdef INTERNAL_TIMINGS_4_DECODER
    startTime = std::chrono::high_resolution_clock::now();
#endif
    checkIndex = 0;
    for (unsigned int check=0; check<numChecks; check++) {
      for (unsigned int index=0; index<bitsForCheck[check]; index++) {
        oneDindex = mapRows2Cols[checkIndex + index];
        etaByBitIndex[oneDindex] = eta[checkIndex + index];
      }
      checkIndex = checkIndex + maxBitsForCheck;
    }
#ifdef INTERNAL_TIMINGS_4_DECODER
    endTime = std::chrono::high_resolution_clock::now();
    delta = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
    transposeTime = transposeTime + delta;
#endif

    // bit estimates update
#ifdef INTERNAL_TIMINGS_4_DECODER
    startTime = std::chrono::high_resolution_clock::now();
#endif
    bitEstimates(rSig, numBits, checksForBit, maxChecksForBit, etaByBitIndex, lambda, decision);
#ifdef INTERNAL_TIMINGS_4_DECODER
    endTime = std::chrono::high_resolution_clock::now();
    delta = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
    beTime = beTime + delta;
#endif
    // Transpose  lambda with rows == bitIndex, to rows == checkIndex
#ifdef INTERNAL_TIMINGS_4_DECODER
    startTime = std::chrono::high_resolution_clock::now();
#endif
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
    endTime = std::chrono::high_resolution_clock::now();
    delta = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
    transposeTime = transposeTime + delta;
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

#ifdef INTERNAL_TIMINGS_4_DECODER
  endTime = std::chrono::high_resolution_clock::now();
  long int allTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - globalStart).count();
#endif

  if(allChecksPassed == false) {
    // printf("Decoding failure after %d iterations\n", iterCounter);
  } else {
    // Print a status message on the iteration loop
    // printf("Success at %i iterations\n",iterCounter);
  }

#ifdef INTERNAL_TIMINGS_4_DECODER
  printf("Total Time     : %i microsec\n",  allTime);
  printf("node processing: %i microsec (%i%)\n",  npTime, 100* npTime / allTime);
  printf("bit esimates   : %i microsec (%i%)\n",  beTime, 100* beTime / allTime);
  printf("transpose      : %i microsec (%i%)\n",  transposeTime, 100 * transposeTime /allTime);
#endif
  return (iterCounter);
}
