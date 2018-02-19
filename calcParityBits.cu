#include "bundleElt.h"

__global__ void
calcParityBits (bundleElt *cHat, bundleElt *parityBits, unsigned int numChecks, unsigned int maxBitsForCheck) {

  unsigned int m = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int thisRowLength;
  bundleElt sum = makeBundleElt(0.0);

  if (m < numChecks) {
    thisRowLength = (int)ONEVAL(cHat[m]);
    for (unsigned int n=1; n<= thisRowLength ; n++) {
      sum.x += (cHat[n * numChecks + m].x >= 0) ? 1 : 0;
      sum.y += (cHat[n * numChecks + m].y >= 0) ? 1 : 0;
      sum.z += (cHat[n * numChecks + m].z >= 0) ? 1 : 0;
      sum.w += (cHat[n * numChecks + m].w >= 0) ? 1 : 0;
    }
    parityBits[m].x = ((int)sum.x) % 2;
    parityBits[m].y = ((int)sum.y) % 2;
    parityBits[m].z = ((int)sum.z) % 2;
    parityBits[m].w = ((int)sum.w) % 2;
  }
}
