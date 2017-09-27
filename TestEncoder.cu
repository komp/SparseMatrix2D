#include <random>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>

void ldpcEncoder (unsigned int *infoWord, unsigned int* W_ROW_ROM,
                  unsigned int numMsgBits, unsigned int numRowsinRom, unsigned int numParBits,
                  unsigned int shiftRegLength,
                  unsigned int *codeWord);

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

  unsigned int  seed = 163331;
  /*  or use this to get a fresh sequence each time the program is run.
  std::random_device  rd;  //Will be used to obtain a seed for the random number engine
  std::mt19937 generator(rd()); //Standard mersenne_twister_engine seeded with rd()
  */
  std::mt19937 generator(seed); //Standard mersenne_twister_engine
  std::uniform_real_distribution<> rDist(0, 1);

  if (argc < 4) {
    printf("usage:  TestEncoder <infoLength> <r-numerator> <r-denominator>\n" );
    exit(-1);
  }
  infoLeng = atoi(argv[1]);
  rnum = atoi(argv[2]);
  rdenom = atoi(argv[3]);
  sprintf(mapFile, "./G_and_H_Matrices/Maps_%d%d_%d.bin", rnum, rdenom, infoLeng);
  sprintf(wROM_File, "./G_and_H_Matrices/W_ROW_ROM_%d%d_%d.bin", rnum, rdenom, infoLeng);

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

  src = fopen(wROM_File, "r");
  if (src == NULL) {
    errnum = errno;
    printf("Value of errno: %d\n", errnum);
    perror("Error printed by perror");
    printf("Error opening file %s\n", mapFile);
    return(EXIT_FAILURE);
  }

  fread(& numRowsW, sizeof(unsigned int), 1, src);
  fread(& numColsW, sizeof(unsigned int), 1, src);
  fread(& shiftRegLength, sizeof(unsigned int), 1, src);
  W_ROW_ROM = (unsigned int*) malloc(numRowsW * numColsW * sizeof( unsigned int));
  fread(W_ROW_ROM, sizeof(unsigned int), numRowsW * numColsW, src);
  numParityBits = numColsW;
  fclose(src);

  printf("parameters have been read.\n");
  printf("numBits = %i, numChecks = %i\n", numBits, numChecks);
  printf("Max checks for bit: %i  Max bits for check %i\n", maxChecksForBit, maxBitsForCheck);
  // ///////////////////////////////////////////

  unsigned int* infoWord;
  unsigned int* codeWord;

  infoWord = (unsigned int *)malloc(infoLeng * sizeof(unsigned int));
  codeWord = (unsigned int *)malloc(numBits * sizeof(unsigned int));

    for (unsigned int j=0; j < infoLeng; j++) {
      infoWord[j] = (0.5 >= rDist(generator))? 1:0;
    }
    ldpcEncoder(infoWord, W_ROW_ROM, infoLeng, numRowsW, numColsW, shiftRegLength, codeWord);

    for (unsigned int j=0; j< numParityBits; j++) {
      printf(" %i", codeWord[infoLeng+j]);
      if ( (j % 40) == 39)  { printf("\n"); }
    }
    printf("\n");
}
