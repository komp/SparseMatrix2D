// Transpose  checkRows matrix with rows == parity checks, to
//            bitRows matrix  with rows == bits
__global__ void
transposeRC (unsigned int* map, float *checkRows, float *bitRows,
             unsigned int numChecks, unsigned int maxBitsForCheck) {
  // index
  unsigned int m,n;
  unsigned int thisRowStart, thisRowLength;
  unsigned int cellIndex, oneDindex;

  m = blockIdx.x;
  n = threadIdx.x + 1;
  if (m < numChecks) {
    thisRowStart = m * (maxBitsForCheck+1);
    thisRowLength = map[thisRowStart];
    if (n <= thisRowLength) {
      cellIndex = thisRowStart + n;
      oneDindex = map[cellIndex];
      bitRows[oneDindex] = checkRows[cellIndex];
    }
  }
}
