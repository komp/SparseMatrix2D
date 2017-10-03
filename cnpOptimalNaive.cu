#include <math.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

#include "GPUincludes.h"

#define MAX_ETA                1e6
#define MIN_TANH_MAGNITUDE     1e-10

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
