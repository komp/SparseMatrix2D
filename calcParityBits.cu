#include "bundleElt.h"

__global__ void
calcParityBits (bundleElt *cHat, bundleElt *parityBits, unsigned int numChecks, unsigned int maxBitsForCheck) {

  unsigned int m = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int thisRowLength;
  bundleElt sum = make_bundleElt(0.0);

  if (m < numChecks) {
    thisRowLength = (int)ONEVAL(cHat[m]);
    for (unsigned int n=1; n<= thisRowLength ; n++) {
      sum.s[0] += (cHat[n * numChecks + m].s[0] >= 0) ? 1 : 0;
      sum.s[1] += (cHat[n * numChecks + m].s[1] >= 0) ? 1 : 0;
      sum.s[2] += (cHat[n * numChecks + m].s[2] >= 0) ? 1 : 0;
      sum.s[3] += (cHat[n * numChecks + m].s[3] >= 0) ? 1 : 0;
    }

    parityBits[m].s[0] = ((int)sum.s[0]) % 2;
    parityBits[m].s[1] = ((int)sum.s[1]) % 2;
    parityBits[m].s[2] = ((int)sum.s[2]) % 2;
    parityBits[m].s[3] = ((int)sum.s[3]) % 2;
  }
}
