__global__ void
calcParityBits (unsigned int* cHat, unsigned int *parityBits, unsigned int numChecks, unsigned int maxBitsForCheck) {

  unsigned int m = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int thisRowLength, thisRowStart;
  unsigned int sum;

  if (m < numChecks) {
    thisRowStart = m * (maxBitsForCheck+1);
    thisRowLength = cHat[thisRowStart];
    sum = 0;
    for (unsigned int n=1; n<= thisRowLength ; n++) {sum += cHat[thisRowStart+n];}
    parityBits[m] = sum % 2;
  }
}
