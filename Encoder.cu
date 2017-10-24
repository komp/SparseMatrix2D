// Based on the Eric's Matlab implementation of ldpcEncoder1.
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

void ldpcEncoder (unsigned int *messageBits, unsigned int* W_ROW_ROM,
                  unsigned int numMsgBits, unsigned int numRowsInRom,
                  unsigned int numParBits, unsigned int shiftRegLength,
                  unsigned int *codeWord)  {
  unsigned int parityBits[numParBits];
  unsigned int numShiftRegs = numParBits/shiftRegLength;
  unsigned int cSR [numParBits];
  unsigned int msgPtr = 0;
  unsigned int wIndex;
  unsigned int wordSize = sizeof(unsigned int);
  unsigned int temp;

  memset(parityBits, 0, numParBits *sizeof(parityBits[0]));

// Loop on the number of rows in the ROM.
  for (int romRowIndex = 0; romRowIndex< numRowsInRom; romRowIndex++) {
    wIndex = romRowIndex * numParBits;
    memmove(cSR, &(W_ROW_ROM[wIndex]), numParBits * wordSize);

    // Loop through shiftRegLength cyclic (barrel) shifts of the registers
    for (int dummyIndex = 0; dummyIndex < shiftRegLength; dummyIndex++) {
      // Multiply (AND) the concatenated contents of shift registers with
      // the incoming message bit and add (XOR) them with the accumulated
      // parity bits, then store the result in the parity bit registers.
      if (messageBits[msgPtr] == 1) {
        for (unsigned int j=0; j< numParBits; j++) {
          parityBits[j] = (parityBits[j] + cSR[j]) % 2;
        }
      }
      // Clock the cyclic (barrel) shift registers. The values at the
      // BOTTOM of the register matrix need to be placed at the
      // TOP of the register matrix.
      // It is a bit tricky, since the linear elements of cSR,
      // contatin the shift register matrix in column order.
      for (unsigned int j = 0; j < numParBits; j += shiftRegLength) {
        temp = cSR[j - 1 + shiftRegLength];
        memmove(&(cSR[j+1]), &(cSR[j]),(shiftRegLength -1)*wordSize);
        cSR[j] = temp;
      }
      msgPtr++;
    }
  }
// Because this is a systematic code, we form the codeword by
// concatenating the parity bits at the end of the message bits.
  memmove(codeWord, messageBits, numMsgBits*wordSize);
  memmove(&(codeWord[numMsgBits]), parityBits, numParBits*wordSize);
}
