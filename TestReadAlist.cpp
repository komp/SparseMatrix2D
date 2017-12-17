#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>

int ReadAlistFile(const char *AlistFile, unsigned int *mrc, unsigned int *mcr);

int main (int argc, char **argv) {

  unsigned int numChecks, numBits, maxBitsForCheck, maxChecksForBit;
  unsigned int  *mapRows2Cols, *mapCols2Rows;
  unsigned int numRowsW, numColsW, numParityBits, shiftRegLength;
  unsigned int *W_ROW_ROM;

  char mapFile[256];
  char wROM_File[256];
  FILE *src;
  int errnum;
  unsigned int infoLeng, rnum, rdenom;
  int status;

  infoLeng = atoi(argv[1]);
  rnum = atoi(argv[2]);
  rdenom = atoi(argv[3]);
  sprintf(mapFile, "./G_and_H_Matrices/Maps_%d%d_%d.bin", rnum, rdenom, infoLeng);

  src = fopen(mapFile, "r");
  if (src == NULL) {
    errnum = errno;
    printf("Value of errno: %d\n", errnum);
    perror("Error printed by perror");
    printf("Error opening file %s\n", mapFile);
    return(EXIT_FAILURE);
  }

  fread(& numBits, sizeof(unsigned int), 1, src);
  fread(& numChecks, sizeof(unsigned int), 1, src);
  fread(& maxBitsForCheck, sizeof(unsigned int), 1, src);
  fread(& maxChecksForBit, sizeof(unsigned int), 1, src);

  // These maps have an extra column (+1),
  // since each row begins with the actual length for the row.
  mapCols2Rows = (unsigned int*) malloc(numBits * (maxChecksForBit +1) * sizeof( unsigned int));
  mapRows2Cols = (unsigned int*) malloc(numChecks * (maxBitsForCheck +1) * sizeof( unsigned int));

  fread(mapCols2Rows, sizeof(unsigned int), numBits* (maxChecksForBit+1), src);
  fread(mapRows2Cols, sizeof(unsigned int), numChecks* (maxBitsForCheck+1), src);
  fclose(src);

  const char *alistFile = "./G_and_H_Matrices/H_45_1024.alist";
  status =  ReadAlistFile(alistFile, mapRows2Cols, mapCols2Rows);
  if (status == 0) {
    printf ("ReadAlistFile worked as expected\n");
  } else {
    printf ("FAILURE/\n");
  }
}
