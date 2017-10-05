__global__ void
bitEstimates(float *rSig, float *etaByBitIndex, float *lambdaByCheckIndex, unsigned int *hd,
             unsigned int *mapCols2Rows, unsigned int numBits, unsigned int maxChecksForBit) {

  unsigned int n;
  unsigned int thisRowLength, thisRowStart;
  unsigned cellIndex, oneDindex;
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid < numBits) {
    n = tid;
    float sum = rSig[n];
    thisRowStart = n*(maxChecksForBit+1);
    thisRowLength = etaByBitIndex[thisRowStart];
    for (unsigned int m=1; m<=thisRowLength; m++) {
      sum = sum + etaByBitIndex[thisRowStart +m];
    }
    for (unsigned int m=1; m<=thisRowLength; m++) {
      cellIndex = thisRowStart + m;
      oneDindex = mapCols2Rows [cellIndex];
      lambdaByCheckIndex [oneDindex] = sum;
      hd[oneDindex] = (sum >= 0) ? 1 : 0;
    }
  }
}
