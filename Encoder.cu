// Based on the Eric's Matlab implementation of ldpcEncoder1.
//
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <time.h>


void ldpcEncoder (unsigned int *messageBits, unsigned int* W_ROW_ROM,
                  unsigned int numMsgBits, unsigned int numRowsInRom, unsigned int numParBits,
                  unsigned int shiftRegLength,
                 unsigned int *codeWord)  {

  unsigned int parityBits[numParBits];
  unsigned int numShiftRegs = numParBits/shiftRegLength;
  //  The row/col order seems reversed here, but I chose this order
  //  so that the barrel shift can be done with memcopy
  unsigned int allShiftRegs[shiftRegLength][numShiftRegs];
  unsigned int concatShiftRegs [numShiftRegs* shiftRegLength];
  unsigned int shiftBuffer[numShiftRegs];
  unsigned int msgPtr = 0;
  unsigned int wIndex;
  unsigned int wordSize = sizeof(unsigned int);
  unsigned int ctr = 0;

  memset(parityBits, 0, numParBits *sizeof(parityBits[0]));

// Loop on the number of rows in the ROM.
  for (unsigned int romRowIndex = 0; romRowIndex< numRowsInRom; romRowIndex++) {
    // Grab the row of the ROM indexed by romRowIndex and load it into the shift registers.
    // memmove(allShiftRegs,&(W_ROW_ROM[romRowIndex * numParBits]), numParBits*wordSize);
    wIndex = romRowIndex * numParBits;
    for (unsigned int col = 0; col < numShiftRegs; col++) {
      for (unsigned int row = 0; row < shiftRegLength; row++) {
        allShiftRegs[row][col] = W_ROW_ROM[wIndex++];}
    }

    // Loop through shiftRegLength cyclic (barrel) shifts of the registers
    for (unsigned int ll = 0; ll < shiftRegLength; ll++) {
      // Multiply (AND) the concatenated contents of the shift registers with
      // the incoming message bit and add (XOR) them with the accumulated
      // parity bits, then store the result in the parity bit registers.
      wIndex = 0;
      for (unsigned int col = 0; col < numShiftRegs; col++) {
        for (unsigned int row = 0; row < shiftRegLength; row++) {
          concatShiftRegs[wIndex++] = allShiftRegs[row][col];}
      }
      //  ParityBits = mod(ParityBits + messageBits(MessagePointer)*ConcatenatedContentsOfRegisters, 2);
      if (messageBits[msgPtr] == 1) {
        for (unsigned int j=0; j< numParBits; j++) {
          parityBits[j] = (parityBits[j] + concatShiftRegs[j]) % 2;
          if (parityBits[j] == 1) ctr++;
        }
      }

      // Clock the cyclic (barrel) shift registers. The values at the BOTTOM
      // of the register matrix need to be placed at the TOP of the register matrix.
      // ShiftRegisters = [ShiftRegisters(end,:); ShiftRegisters(1:end-1,:)];
      // unsigned int allShiftRegs[shiftRegLength, numShiftRegs];
      memmove( shiftBuffer, & (allShiftRegs[shiftRegLength-1][0]),  numShiftRegs*wordSize);
      memmove(&(allShiftRegs[1][0]) , allShiftRegs, (numParBits- numShiftRegs)*wordSize);
      memmove( allShiftRegs, shiftBuffer,  numShiftRegs*wordSize);

      msgPtr++;
    }
  }
// Because this is a systematic code, we form the codeword by concatenating the
// parity bits at the end of the message bits.
  memmove(codeWord, messageBits, numMsgBits*wordSize);
  memmove(&(codeWord[numMsgBits]), parityBits, numParBits*wordSize);

}
