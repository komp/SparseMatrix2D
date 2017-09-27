#include <math.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

#include "GPUincludes.h"

#define MAX_ETA                1e6
#define MIN_TANH_MAGNITUDE     1e-10

// NOTE:  pragma unroll before the loops did not improve performance
// For implementation,
// I had to make the loop termination index constant; and wrap the
//  the loop body in a if (n < thisRowLength) ...
__global__ void
checkNodeProcessingOptimal (unsigned int numChecks, unsigned int maxBitsForCheck,
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
    product = 1.0;

    for (unsigned int n=1; n<= thisRowLength ; n++) {
      currentIndex = thisRowStart+n;
      value =  tanhf ((eta[currentIndex] - lambdaByCheckIndex[currentIndex]) / 2.0);
      if (value == 0.0) {
        value = MIN_TANH_MAGNITUDE;
      }
      rowVals[n] = value;
      product =  product * value;
    }

    for (unsigned int n=1; n<= thisRowLength; n++) {
      currentIndex = thisRowStart+n;
      value = -2 *atanhf(product/rowVals[n]);
      value = (value > MAX_ETA)? MAX_ETA : value;
      value = (value < -MAX_ETA)? -MAX_ETA : value;
      eta[currentIndex] =  value;
    }
  }
}
