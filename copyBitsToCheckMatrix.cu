#include "bundleElt.h"

// copyBitsToCheckmatrix accepts a vector of the current bitEstimates
// and copies them into a checkRow matrix, where each row represents a check.
__global__ void
copyBitsToCheckmatrix (unsigned int* map, bundleElt *bitEstimates, bundleElt *checkRows,
                       unsigned int numBits, unsigned int maxChecksForBit,
                       unsigned int nChecksByBits, unsigned int nBitsByChecks, unsigned int nBundles) {
  // index
  unsigned int bundleIndex;
  unsigned int m, n;
  unsigned int thisRowLength;
  unsigned int cellIndex, oneDindex;
  bundleElt thisBitEstimate;

  bundleIndex = blockIdx.x / numBits;
  n = blockIdx.x % numBits;
  m = threadIdx.x + 1;
  if (bundleIndex < nBundles) {
    thisRowLength = map[n];
    thisBitEstimate = bitEstimates[(bundleIndex * nBitsByChecks) + n];
    if (m <= thisRowLength) {
      cellIndex =  m * numBits + n;
      oneDindex = map[cellIndex] + (bundleIndex * nChecksByBits);
      checkRows[oneDindex] = thisBitEstimate;
    }
  }
}
