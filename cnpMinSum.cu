#include <math.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

#include "GPUincludes.h"

#define ABS(a)  (((a) < (0)) ? (-(a)) : (a))
#define MAX_ETA                1e6
#define SCALE_FACTOR           0.75

__global__ void
checkNodeProcessingMinSum (unsigned int numChecks, unsigned int maxBitsForCheck,
                           float *lambdaByCheckIndex, float *eta) {
  // edk  HACK
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