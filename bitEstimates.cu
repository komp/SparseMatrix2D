#include "bundleElt.h"

__global__ void
bitEstimates(bundleElt *rSig, bundleElt *etaByBitIndex, bundleElt *lambdaByCheckIndex, bundleElt *hd,
             unsigned int *mapCols2Rows, unsigned int numBits, unsigned int maxChecksForBit,
             unsigned int nChecksByBits, unsigned int nBitsByChecks, unsigned int nBundles) {

  unsigned int bundleIndex, bundleBase;
  unsigned int n;
  unsigned int thisRowLength, cellIndex;
  unsigned oneDindex;
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  bundleIndex = tid / numBits;
  n = tid % numBits;

  if (bundleIndex < nBundles) {
    bundleBase = bundleIndex* nBitsByChecks;
    bundleElt sum = rSig[n];
    thisRowLength = (int)ONEVAL(etaByBitIndex[n]);

    for (unsigned int m=1; m<=thisRowLength; m++) sum += etaByBitIndex[m * numBits + n + bundleBase];

    for (unsigned int m=1; m<=thisRowLength; m++) {
      cellIndex = m * numBits + n;
      oneDindex = mapCols2Rows [cellIndex] + (bundleIndex * nChecksByBits);
      lambdaByCheckIndex [ oneDindex] = sum;
      //      hd[oneDindex] = (sum >= 0) ? 1 : 0;
      hd[oneDindex] = sum;
    }
  }
}
