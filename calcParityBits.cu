#include "bundleElt.h"


// Inlining the SLOT loops made no difference.

__global__ void
calcParityBits (bundleElt *cHat, bundleElt *parityBits, unsigned int numChecks, unsigned int maxBitsForCheck) {

  unsigned int tid, m;
  unsigned int thisRowLength;

  tid = threadIdx.x + blockIdx.x * blockDim.x;
  m = tid;
  if (m < numChecks) {
    thisRowLength = (int)ONEVAL(cHat[m]);
    bundleElt sum = make_bundleElt(0.0);
    for (unsigned int n=1; n<= thisRowLength ; n++) {
      for (unsigned int slot=0; slot< SLOTS_PER_ELT; slot++)
        if (cHat[n * numChecks + m].s[slot] >= 0) sum.s[slot]++;
    }
    for (unsigned int slot=0; slot< SLOTS_PER_ELT; slot++)
      parityBits[m].s[slot] = ((int)sum.s[slot]) % 2;
  }
}
