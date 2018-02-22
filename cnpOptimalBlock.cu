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
  bundleElt myprod;

  m = blockIdx.x;
  n = threadIdx.x + 1;
  if (m < numChecks) {
    __shared__ bundleElt rowVals[128];

    thisRowLength = (int) ONEVAL(eta[m]);
    if (n <= thisRowLength) {
      currentIndex = m + (n* numChecks);
      arg =  (eta[currentIndex] - lambdaByCheckIndex[currentIndex]) / 2.0;
      for (unsigned int slot=0; slot< SLOTS_PER_ELT; slot++) {
        value.s[slot] = tanhf(arg.s[slot]);
        if (value.s[slot] == 0.0) {value.s[slot] = MIN_TANH_MAGNITUDE;}
      }
      rowVals[n] = value;
      __syncthreads();

      myprod = make_bundleElt(1.0);
      for (unsigned int j=1; j<= thisRowLength; j++) if (j != n) myprod *= rowVals[j];

      for (unsigned int slot=0; slot< SLOTS_PER_ELT; slot++) value.s[slot] = -2 * atanhf(myprod.s[slot]);
      value = clamp(value, -MAX_ETA, MAX_ETA);
      eta[currentIndex] =  value;
      etaByBitIndex[ mapRows2Cols[currentIndex] ] = value;
    }
  }
}
