#include "bundleElt.h"

__global__ void
bitEstimates(bundleElt *rSig, bundleElt *esitmate,  bundleElt *etaByBitIndex, bundleElt *lambdaByCheckIndex,
             unsigned int *mapCols2Rows, unsigned int numBits, unsigned int maxChecksForBit) {

  unsigned int thisRowLength, cellIndex;
  unsigned oneDindex;
  unsigned int n = threadIdx.x + blockIdx.x * blockDim.x;

  if (n < numBits) {
    bundleElt sum = rSig[n];
    thisRowLength = (int)ONEVAL(etaByBitIndex[n]);

    for (unsigned int m=1; m<=thisRowLength; m++) sum += etaByBitIndex[m * numBits + n];

    for (unsigned int m=1; m<=thisRowLength; m++) {
      cellIndex = m * numBits + n;
      oneDindex = mapCols2Rows [cellIndex];
      lambdaByCheckIndex [ oneDindex] = sum;
      //      hd[oneDindex] = (sum >= 0) ? 1 : 0;
    }
  }
}
