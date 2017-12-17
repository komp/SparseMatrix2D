#include <string.h>
#include <stdio.h>

__global__ void
bitEstimates(float *rSig, float *etaByBitIndex, float *lambdaByCheckIndex, unsigned int *hd,
             unsigned int *mapCols2Rows, unsigned int numBits, unsigned int maxChecksForBit) {

  unsigned int n;
  unsigned int thisRowLength, cellIndex;
  unsigned oneDindex;
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid < numBits) {
    n = tid;
    float sum = rSig[n];
    thisRowLength = etaByBitIndex[n];
    for (unsigned int m=1; m<=thisRowLength; m++) {
      sum = sum + etaByBitIndex[m * numBits + n];
    }
    if ( tid < 1) {
      printf("Bit %i  sum = %.1f (%.1f,%.1f, %.1f, %.1f)\n",
             n, sum, rSig[n], etaByBitIndex[numBits + n], etaByBitIndex[2*numBits + n], etaByBitIndex[3*numBits + n]);
    }
    for (unsigned int m=1; m<=thisRowLength; m++) {
      cellIndex = m * numBits + n;
      oneDindex = mapCols2Rows [cellIndex];
      // lambdaByCheckIndex [oneDindex] = sum;
      lambdaByCheckIndex [oneDindex] = sum - etaByBitIndex[m * numBits + n];
      hd[oneDindex] = (sum >= 0) ? 1 : 0;
    }
  }
}
