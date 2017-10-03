__global__ void
bitEstimates(float *rSig, float *etaByBitIndex, float *lambda,
             unsigned int numBits, unsigned int maxChecksForBit) {

  unsigned int n;
  unsigned int thisRowLength, thisRowStart;
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid < numBits) {
    n = tid;
    float sum = rSig[n];
    thisRowStart = n*(maxChecksForBit+1);
    thisRowLength = etaByBitIndex[thisRowStart];
    for (unsigned int m=1; m<=thisRowLength; m++) {
      sum = sum + etaByBitIndex[thisRowStart +m];
    }
    lambda[n] = sum;
  }
}
