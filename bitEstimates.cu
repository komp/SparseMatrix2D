#include "bundleElt.h"

__global__ void
bitEstimates(bundleElt *rSig, bundleElt *etaByBitIndex, bundleElt *lambdaByCheckIndex, bundleElt *hd,
             unsigned int *mapCols2Rows, unsigned int numBits, unsigned int maxChecksForBit) {

  unsigned int n;
  unsigned int thisRowLength, cellIndex;
  unsigned oneDindex;
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid < numBits) {
    n = tid;
    bundleElt sum = rSig[n];
    thisRowLength = (int)ONEVAL(etaByBitIndex[n]);

    for (unsigned int m=1; m<=thisRowLength; m++) sum += etaByBitIndex[m * numBits + n];

    for (unsigned int m=1; m<=thisRowLength; m++) {
      cellIndex = m * numBits + n;
      oneDindex = mapCols2Rows [cellIndex];
      lambdaByCheckIndex [oneDindex] = sum;
      //      hd[oneDindex] = (sum >= 0) ? 1 : 0;
      hd[oneDindex] = sum;
    }
  }
}
