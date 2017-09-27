#include <math.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

#include "GPUincludes.h"

#define ABS(a)  (((a) < (0)) ? (-(a)) : (a))
#define MAX_ETA                1e6
#define SCALE_FACTOR           0.75

__global__ void
checkNodeProcessingMinSumBlock (unsigned int numChecks, unsigned int maxBitsForCheck,
                           float *lambdaByCheckIndex, float *eta) {
  float value;
  unsigned int m, n;
  unsigned int thisRowLength, thisRowStart, currentIndex;

  m = blockIdx.x;
  n = threadIdx.x + 1;
  if (m < numChecks) {
    // signs[n]  == 0  ==>  positive; 1  ==>  negative
    __shared__ unsigned int signs[128];
    __shared__ float rowVals[128];
    __shared__ float min1, min2;
    __shared__ unsigned int minIndex, signProduct;

    thisRowStart = m * (maxBitsForCheck+1);
    thisRowLength = eta[thisRowStart];

    if ( n <= thisRowLength) {
        currentIndex = thisRowStart+n;
        value = eta[currentIndex] - lambdaByCheckIndex[currentIndex];
        signs[n] = (value < 0)? 1 : 0;
        rowVals[n] = ABS(value);

        __syncthreads();

      // Using JUST thread 0 to find min1, min2 and signProduct
      // Storing it in the shared location  rowVals[0]
      if (threadIdx.x == 0) {
        min1 = MAX_ETA;
        min2 =  MAX_ETA;
        minIndex = 1;
        signProduct = 0;
        for (unsigned int j=1; j<= thisRowLength; j++) {
          signProduct = (signProduct != signs[j])? 1 : 0;
          if (rowVals[j] < min1) {
            min2 = min1;
            min1 = rowVals[j];
            minIndex = j;
          } else if ( rowVals[j] < min2) {
            min2 = rowVals[j];
          }
        }
        min1 = min1 * SCALE_FACTOR * (-1);
        min2 = min2 * SCALE_FACTOR * (-1);
      }
      __syncthreads();

      rowVals[n] =  (n == minIndex) ? min2 : min1;
      if (signs[n] != signProduct) {rowVals[n] = -rowVals[n];}
      eta[currentIndex] = rowVals[n];
    }
  }
}
