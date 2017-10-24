__global__ void
calcParityBits (unsigned int* cHat, unsigned int *parityBits, unsigned int numChecks, unsigned int maxBitsForCheck) {

  unsigned int m = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int thisRowLength;
  unsigned int sum;

  if (m < numChecks) {
    thisRowLength = cHat[m];
    sum = 0;
    for (unsigned int n=1; n<= thisRowLength ; n++) {sum += cHat[n * numChecks + m];}
    parityBits[m] = sum % 2;
  }
}
