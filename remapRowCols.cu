#include <stdlib.h>
#include <string.h>
#include <stdio.h>

void remapRows2Cols (unsigned int numChecks, unsigned int numBits,
                     unsigned int maxBitsPerCheck, unsigned int maxChecksPerBit,
                     unsigned int *r2c, unsigned int *newR2C) {

  unsigned int oldDest, oldDestBit, oldDestOffset, newDest;
  // First copy column 0 (actual number of entries for each row) to row 0
  for (unsigned int check = 0; check < numChecks; check++) newR2C[check] = r2c[check*(maxBitsPerCheck+1)];
  //  Now, the cell values have to be modified as we swap rows and cols.
  for (unsigned int rowIndex = 1; rowIndex<=maxBitsPerCheck; rowIndex++) {
    for (unsigned int check = 0; check < numChecks; check++) {
      oldDest = r2c[check*(maxBitsPerCheck+1) + rowIndex];
      if (oldDest != 0) {
        oldDestBit = oldDest / (maxChecksPerBit+1);
        oldDestOffset = oldDest - (oldDestBit * (maxChecksPerBit+1));
        newDest = oldDestOffset * numBits + oldDestBit;
        newR2C[ rowIndex * numChecks + check] = newDest;
      }
    }
  }
}

void remapCols2Rows (unsigned int numChecks, unsigned int numBits,
                     unsigned int maxBitsPerCheck, unsigned int maxChecksPerBit,
                     unsigned int *c2r, unsigned int *newC2R) {

  unsigned int oldDest, oldDestCheck, oldDestOffset, newDest;
  // First copy column 0 (actual number of entries for each row) to row 0
  for (unsigned int bit = 0; bit < numBits; bit++) newC2R[bit] = c2r[bit*(maxChecksPerBit +1)];
  //  Now, the cell values have to be modified as we swap rows and cols.
  for (unsigned int rowIndex = 1; rowIndex<=maxChecksPerBit; rowIndex++) {
    for (unsigned int bit = 0; bit < numBits; bit++) {
      oldDest = c2r[bit*(maxChecksPerBit+1) + rowIndex];
      if (oldDest != 0) {
        oldDestCheck = oldDest / (maxBitsPerCheck+1);
        oldDestOffset = oldDest - (oldDestCheck * (maxBitsPerCheck+1));
        newDest = oldDestOffset * numChecks + oldDestCheck;
        newC2R[ rowIndex * numBits + bit] = newDest;
      }
    }
  }
}
