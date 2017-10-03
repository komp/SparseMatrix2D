// copyBitsToCheckmatrix accepts a vector of the current bitEstimates
// and copies them into a checkRow matrix, where each row represents a check.
// It also generates a HardDecision copy of that output matrix checkRows.
__global__ void
copyBitsToCheckmatrix (unsigned int* map, float *bitEstimates, float *checkRows,
                       unsigned int *hd,
                       unsigned int numBits, unsigned int maxChecksForBit) {
  // index
  unsigned int m, n;
  unsigned int thisRowStart, thisRowLength;
  unsigned int cellIndex, oneDindex;
  float thisBitEstimate;

  n = blockIdx.x;
  m = threadIdx.x + 1;
  if (n < numBits) {
    thisRowStart = n * (maxChecksForBit+1);
    thisRowLength = map[thisRowStart];
    thisBitEstimate = bitEstimates[n];
    if (m <= thisRowLength) {
      cellIndex = thisRowStart + m;
      oneDindex = map[cellIndex];
      checkRows[oneDindex] = thisBitEstimate;
      hd[oneDindex] = (thisBitEstimate >= 0) ? 1 : 0;
    }
  }
}
