#include "bundleElt.h"

__global__ void
calcParityBits (bundleElt *cHat, bundleElt *parityBits, unsigned int numChecks, unsigned int maxBitsForCheck,
                unsigned int nChecksByBits, unsigned int nBundles) {

  unsigned int bundleIndex, bundleBase, etaIndex;
  unsigned int tid, m;
  unsigned int thisRowLength;

  bundleIndex = blockIdx.x / numChecks;
  tid = threadIdx.x + blockIdx.x * blockDim.x;
  m = tid % numChecks;
  if (bundleIndex < nBundles) {
    bundleBase = bundleIndex* nChecksByBits;
    thisRowLength = (int)ONEVAL(cHat[m]);
    bundleElt sum = make_bundleElt(0.0);
    for (unsigned int n=1; n<= thisRowLength ; n++) {
      for (unsigned int slot=0; slot< SLOTS_PER_ELT; slot++)
        if (cHat[bundleBase + n * numChecks + m].s[slot] >= 0) sum.s[slot]++;
    }
    for (unsigned int slot=0; slot< SLOTS_PER_ELT; slot++)
      parityBits[(bundleIndex*numChecks) + m].s[slot] = ((int)sum.s[slot]) % 2;
  }
}
