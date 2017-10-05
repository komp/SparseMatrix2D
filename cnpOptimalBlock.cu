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
checkNodeProcessingOptimalBlock (unsigned int numChecks, unsigned int maxBitsForCheck,
                                 float *lambdaByCheckIndex, float *eta, unsigned int* mapRows2Cols,
                                 float *etaByBitIndex) {

  unsigned int m, n;
  unsigned int thisRowLength, thisRowStart, currentIndex;
  float value;

  m = blockIdx.x;
  n = threadIdx.x + 1;
  if (m < numChecks) {
    __shared__ float rowVals[128];

    thisRowStart = m * (maxBitsForCheck+1);
    thisRowLength = eta[thisRowStart];

    if (n <= thisRowLength) {
      currentIndex = thisRowStart+n;
      value =  tanhf ((eta[currentIndex] - lambdaByCheckIndex[currentIndex]) / 2.0);
      if (value == 0.0) {value = MIN_TANH_MAGNITUDE;}
      rowVals[n] = value;
      __syncthreads();

      // Using JUST thread 0 to compute the product of all terms.
      // Storing it in the shared location  rowVals[0]
      if (threadIdx.x == 0) {
        rowVals[0] = 1.0;
        for (unsigned int j=1; j<= thisRowLength; j++) rowVals[0] =  rowVals[0] * rowVals[j];
      }
      __syncthreads();

      currentIndex = thisRowStart+n;
      value = -2 *atanhf(rowVals[0]/rowVals[n]);
      value = (value > MAX_ETA)? MAX_ETA : value;
      value = (value < -MAX_ETA)? -MAX_ETA : value;
      eta[currentIndex] =  value;
      etaByBitIndex[ mapRows2Cols[currentIndex] ] = value;
    }
  }
}
