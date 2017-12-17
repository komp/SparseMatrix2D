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
                           float *lambdaByCheckIndex, float *eta, unsigned int* mapRows2Cols,
                           float *etaByBitIndex) {
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
  unsigned int thisRowLength, currentIndex;
  float values[7];

  if (tid < numChecks) {
    m = tid;
    // signs[n]  == 0  ==>  positive; 1  ==>  negative
    memset(signs, 0, (maxBitsForCheck+1)*sizeof(signs[0]));
    signProduct = 0;
    min1 = MAX_ETA;
    min2 =  MAX_ETA;
    minIndex = 1;
    thisRowLength = (unsigned int) eta[m];
    for (unsigned int n=1; n<= thisRowLength ; n++) {
      currentIndex = (n * numChecks) + m;
      value = lambdaByCheckIndex[currentIndex];
      values[n] = value;
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
    if (m < 1) printf ("Check %i: %.1f %.1f %.1f %.1f %.1f %.1f\n",m, values[1],values[2], values[3], values[4], values[5], values[6]);
    //    min1 = min1 * SCALE_FACTOR;
    //    min2 = min2 * SCALE_FACTOR;
    for (unsigned int n=1; n<= thisRowLength; n++) {
      currentIndex = (n * numChecks) + m;
      value =  (n == minIndex) ? min2 : min1;
      if (signs[n] != signProduct) {value = -value;}
      values[n] = value;
      eta[currentIndex] =  value;
      etaByBitIndex[ mapRows2Cols[currentIndex] ] = value;
    }
    if (m < 1) printf ("PostCheck %i: %.1f %.1f %.1f %.1f %.1f %.1f\n",m, values[1],values[2], values[3], values[4], values[5], values[6]);

  }
}
