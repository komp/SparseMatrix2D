#include <math.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

#include "bundleElt.h"

#define MAX_ETA                1e6
#define MIN_TANH_MAGNITUDE     1e-10

// NOTE:  pragma unroll before the loops did not improve performance
// For implementation,
// I had to make the loop termination index constant; and wrap the
//  the loop body in a if (n < thisRowLength) ...
__global__ void
checkNodeProcessingOptimalBlock (unsigned int numChecks, unsigned int maxBitsForCheck,
                                 bundleElt *lambdaByCheckIndex, bundleElt *eta, unsigned int* mapRows2Cols,
                                 bundleElt *etaByBitIndex) {

  unsigned int m, n;
  unsigned int thisRowLength, currentIndex;
  bundleElt arg, value;

  m = blockIdx.x;
  n = threadIdx.x + 1;
  if (m < numChecks) {
    __shared__ bundleElt rowVals[128];

    thisRowLength = (int) ONEVAL(eta[m]);
    if (n <= thisRowLength) {
      currentIndex = m + (n* numChecks);
      arg =  (eta[currentIndex] - lambdaByCheckIndex[currentIndex]) / 2.0;
      value.x = tanhf(arg.x);
      value.y = tanhf(arg.y);
      value.z = tanhf(arg.z);
      value.w = tanhf(arg.w);
      if (value.x == 0.0) {value.x = MIN_TANH_MAGNITUDE;}
      if (value.y == 0.0) {value.y = MIN_TANH_MAGNITUDE;}
      if (value.z == 0.0) {value.z = MIN_TANH_MAGNITUDE;}
      if (value.w == 0.0) {value.w = MIN_TANH_MAGNITUDE;}
      rowVals[n] = value;
      __syncthreads();

      // Using JUST thread 0 to compute the product of all terms.
      // Storing it in the shared location  rowVals[0]
      if (threadIdx.x == 0) {
        rowVals[0] = makeBundleElt(1.0);
        for (unsigned int j=1; j<= thisRowLength; j++) rowVals[0] *= rowVals[j];
      }
      __syncthreads();

      // value = -2 *atanhf(rowVals[0]/rowVals[n]);
      arg = rowVals[0]/rowVals[n];
      value.x = -2 * atanhf(arg.x);
      value.y = -2 * atanhf(arg.y);
      value.z = -2 * atanhf(arg.z);
      value.w = -2 * atanhf(arg.w);

      value = clamp(value, -MAX_ETA, MAX_ETA);
      eta[currentIndex] =  value;
      etaByBitIndex[ mapRows2Cols[currentIndex] ] = value;
    }
  }
}
