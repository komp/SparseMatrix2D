// copyBitsToCheckmatrix accepts a vector of the current bitEstimates
// and copies them into a checkRow matrix, where each row represents a check.
// It also generates a HardDecision copy of that output matrix checkRows.
__global__ void
copyBitsToCheckmatrix (unsigned int* map, float *bitEstimates, float *checkRows,
                       // argument no longer used.
                       //                       unsigned int *hd,
                       unsigned int numBits, unsigned int maxChecksForBit) {
  // index
  unsigned int m, n;
  unsigned int thisRowLength;
  unsigned int cellIndex, oneDindex;
  float thisBitEstimate;

  n = blockIdx.x;
  m = threadIdx.x + 1;
  if (n < numBits) {
    thisRowLength = map[n];
    thisBitEstimate = bitEstimates[n];
    if (m <= thisRowLength) {
      cellIndex = m * numBits + n;
      oneDindex = map[cellIndex];
      checkRows[oneDindex] = thisBitEstimate;
      // no longer used.
      //      hd[oneDindex] = (thisBitEstimate >= 0) ? 1 : 0;
    }
  }
}
