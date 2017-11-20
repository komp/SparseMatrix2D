#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>

/*  Note: we accept irregular codes, those codes for which
    the number of checks per bit, and the number of bits per check
    are not constant.
    But, for irregular codes, the checksForBit and bitsForCheck matrices
    that are read from the file, must be regular.
    checksForBit will be numBits X maxWeightForBit in size.
    The unused elements must contain a zero.
*/

int ReadAlistFile(const char *AlistFile, unsigned int *mrc, unsigned int *mcr){

  FILE *fd;
  unsigned int k;
  bool test;

  unsigned int numBits, numChecks, maxWeightForBit, maxWeightForCheck;
  unsigned int *weightForBit; // H_bnw;
  unsigned int *weightForCheck; // H_cnw;
  unsigned int *bitsForCheck;  // H_bn;
  unsigned int *checksForBit; //  H_cn;
  unsigned int *mapRows2Cols;
  unsigned int *mapCols2Rows;

  unsigned int bit, check, rowIndex;
  int indexForBit, indexForCheck;

  fd = fopen(AlistFile,"r");
  if(fd == NULL){ fprintf(stdout,"Error while opening alist file\n"); return -1; }

  fscanf(fd,"%d", &numBits);
  fscanf(fd,"%d", &numChecks);
  fscanf(fd,"%d", &maxWeightForBit);
  fscanf(fd,"%d", &maxWeightForCheck);

  weightForBit = (unsigned int *) malloc(sizeof(unsigned int)* numBits);
  weightForCheck = (unsigned int *) malloc(sizeof(unsigned int)* numChecks);
  checksForBit = (unsigned int *) malloc(sizeof(unsigned int)* numBits * maxWeightForBit);
  bitsForCheck = (unsigned int *) malloc(sizeof(unsigned int)* numChecks * maxWeightForCheck);
  // The first element of each row will contain the actual weight for this row,
  // so the size must be modified appropriately.
  mapRows2Cols = (unsigned int *) malloc(sizeof(unsigned int)* numChecks * (maxWeightForCheck +1));
  mapCols2Rows = (unsigned int *) malloc(sizeof(unsigned int)* numBits * (maxWeightForBit +1));

  //Read weightForBit's
  for (unsigned int i = 0; i < numBits; i++)fscanf(fd,"%d",&weightForBit[i]);
  //and Read weightForCheck's
  for (unsigned int i = 0; i < numChecks; i++)fscanf(fd,"%d",&weightForCheck[i]);

  //Read checksForBit's
  for (unsigned int i = 0; i < (numBits * maxWeightForBit); i++) fscanf(fd,"%d",&checksForBit[i]);
  //Read bitsForCheck's
  for (unsigned int i = 0; i < (numChecks * maxWeightForCheck); i++) fscanf(fd,"%d",&bitsForCheck[i]);

  for (unsigned int check = 0; check < numChecks; check++) {
    rowIndex = check * maxWeightForCheck;
    mapRows2Cols[check * (maxWeightForCheck +1)] = weightForCheck[check];
    for(unsigned int j = 0; j < weightForCheck[check]; j++) {
      // WARNING:  we use 0-based indexing,
      // but the alist format uses 1-based indexes for bits and checks.
      // so we have to carefully go back and forth
      bit = bitsForCheck[rowIndex+j] -1;
      //  Need to find the position of "check" in the list checksForBit[bit, :]
      indexForCheck = -1;
      for (unsigned int k=0; k < weightForBit[bit]; k++) {
        if ( (check +1) == checksForBit[bit* maxWeightForBit + k] ) indexForCheck = k;
      }
      if (indexForCheck < 0) {
        printf ("ERROR:  Failure to create mapRows2Cols.\n");
        printf ("Cannot find check = %d, in checksForBit[%d,:]\n", check, bit);
        exit (-1);
      }
      // '+1' is required, since the first element of each row in mapRows2Cols (mapCols2Rows)
      // contains the actual row length.
      mapRows2Cols[check*(maxWeightForCheck +1)+1 + j] = bit*(maxWeightForBit+1)+1 + indexForCheck;
    }
  }
  for (unsigned int i = 0; i < (numChecks * (maxWeightForCheck +1)); i++) {
    if (mapRows2Cols[i] != mrc[i]) {
      printf("ERROR at index %i, mapRows2Cols(%i) != mrc(%i)\n", i, mapRows2Cols[i], mrc[i]);
      return (-1);
    }
  }

  for (unsigned int bit = 0; bit < numBits; bit++) {
    rowIndex = bit * maxWeightForBit;
    mapCols2Rows[bit * (maxWeightForBit +1)] = weightForBit[bit];
    for(unsigned int j = 0; j < weightForBit[bit]; j++) {
      // WARNING:  we use 0-based indexing,
      // but the alist format uses 1-based indexes for bits and checks.
      // so we have to carefully go back and forth
      check = checksForBit[rowIndex+j] -1;
      //  Need to find the position of "bit" in the list bitsForCheck[check, :]
      indexForBit = -1;
      for (unsigned int k=0; k < weightForCheck[check]; k++) {
        if ((bit +1) == bitsForCheck[check* maxWeightForCheck + k] ) indexForBit = k;
      }
      if (indexForBit < 0) {
        printf ("ERROR:  Failure to create mapCols2Rows.\n");
        printf ("Cannot find bit = %d, in bitsForCheck[%d,:]\n", bit, check);
        exit (-1);
      }
      // '+1' is required, since the first element of each row in mapRows2Cols (mapCols2Rows)
      // contains the actual row length.
      mapCols2Rows[bit*(maxWeightForBit+1)+1 + j] = check*(maxWeightForCheck +1)+1 + indexForBit;
    }
  }
  for (unsigned int i = 0; i < (numBits * (maxWeightForBit +1)); i++) {
    if (mapCols2Rows[i] != mcr[i]) {
      printf("ERROR at index %i, mapCols2Rows(%i) != mcr(%i)\n", i, mapCols2Rows[i], mcr[i]);
      return (-1);
    }
  }
  fclose(fd);

  return 0;
}
