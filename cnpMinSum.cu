#include <math.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

#include "bundleElt.h"

#define ABS(a)  (((a) < (0)) ? (-(a)) : (a))
#define MAX_ETA                1e6
#define SCALE_FACTOR           0.75

__global__ void
checkNodeProcessingMinSum (unsigned int numChecks, unsigned int maxBitsForCheck,
                           bundleElt *lambdaByCheckIndex, bundleElt *eta, unsigned int* mapRows2Cols,
                           bundleElt *etaByBitIndex, unsigned int nChecksByBits, unsigned int nBitsByChecks, unsigned int nBundles) {
  // edk  HACK
  // This was signs[maxBitsForCheck], which generates the error:
  // error: constant value is not known.
  // Since we are in a kernel function, we probably need a compile-time constant.
  // 128 should be much larger than maxBitsForCheck for any reasonable LDPC encoding.
  bundleEltI signs[128];
  bundleEltI signProduct;
  bundleElt min1, min2;
  bundleEltI minIndex;
  float value;

  unsigned int bundleIndex, bundleBase, etaIndex;
  unsigned int m;
  unsigned int thisRowLength, currentIndex;

  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  bundleIndex = tid / numChecks;
  m = tid % numChecks;

  if (bundleIndex < nBundles && m < numChecks) {
    bundleBase = bundleIndex* nChecksByBits;
    // signs[n]  == 0  ==>  positive; 1  ==>  negative
    memset(signs, 0, (maxBitsForCheck+1)*sizeof(signs[0]));
    signProduct = make_bundleEltI(0);
    min1 = make_bundleElt(MAX_ETA);
    min2 = make_bundleElt(MAX_ETA);
    minIndex = make_bundleEltI(1);

    thisRowLength = (unsigned int) ONEVAL(eta[m]);

    for (unsigned int n=1; n<= thisRowLength ; n++) {
      currentIndex = (n * numChecks) + m;
      etaIndex = bundleBase + currentIndex;

      for (unsigned int slot=0; slot< SLOTS_PER_ELT; slot++) {
        value = eta[etaIndex].s[slot] - lambdaByCheckIndex[etaIndex].s[slot];
        signs[n].s[slot] = (value < 0)? 1 : 0;
        signProduct.s[slot] = (signProduct.s[slot] != signs[n].s[slot])? 1 : 0;
        value = ABS(value);
        if (value < min1.s[slot]) {
          min2.s[slot] = min1.s[slot];
          min1.s[slot] = value;
          minIndex.s[slot] = n;
        } else if ( value < min2.s[slot]) {
          min2.s[slot] = value;
        }
      }
    }

    for (unsigned int slot=0; slot< SLOTS_PER_ELT; slot++) {
      min1.s[slot] = min1.s[slot] * SCALE_FACTOR * (-1);
      min2.s[slot] = min2.s[slot] * SCALE_FACTOR * (-1);
    }

    for (unsigned int n=1; n<= thisRowLength; n++) {
      currentIndex = (n * numChecks) + m;
      etaIndex = bundleBase + currentIndex;
      for (unsigned int slot=0; slot< SLOTS_PER_ELT; slot++) {
        value =  (n == minIndex.s[slot]) ? min2.s[slot] : min1.s[slot];
        if (signs[n].s[slot] != signProduct.s[slot]) {value = -value;}
        eta[etaIndex].s[slot] =  value;
        etaByBitIndex[(bundleIndex * nBitsByChecks) + mapRows2Cols[currentIndex] ].s[slot] = value;
      }
    }
  }
}
