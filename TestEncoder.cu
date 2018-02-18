#include <random>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>

#include <chrono>

void ldpcEncoder (unsigned int *infoWord, unsigned int* W_ROW_ROM,
                  unsigned int numMsgBits, unsigned int numRowsinRom, unsigned int numParBits,
                  unsigned int shiftRegLength,
                  unsigned int *codeWord);

int main (int argc, char **argv) {

  unsigned int numChecks, numBits, maxBitsForCheck, maxChecksForBit;
  unsigned int numRowsW, numColsW, shiftRegLength;
  unsigned int *W_ROW_ROM;

  char alistFile[256];
  char wROM_File[256];
  FILE *src;
  int errnum;
  unsigned int infoLeng, rnum, rdenom;

  using clock = std::chrono::steady_clock;

  clock::time_point startTime;
  clock::time_point endTime;
  clock::duration encoderTime;

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
  sprintf(alistFile, "./G_and_H_Matrices/H_%d%d_%d.alist", rnum, rdenom, infoLeng);
  sprintf(wROM_File, "./G_and_H_Matrices/W_ROW_ROM_%d%d_%d.binary", rnum, rdenom, infoLeng);

  src = fopen(alistFile, "r");
  if (src == NULL) {
    errnum = errno;
    printf("Value of errno: %d\n", errnum);
    perror("Error printed by perror");
    printf("Error opening file %s\n", alistFile);
    return(EXIT_FAILURE);
  }

  fscanf(src,"%d", &numBits);
  fscanf(src ,"%d", &numChecks);
  fscanf(src,"%d", &maxChecksForBit);
  fscanf(src,"%d", &maxBitsForCheck);
  fclose(src);

  src = fopen(wROM_File, "r");
  if (src == NULL) {
    errnum = errno;
    printf("Value of errno: %d\n", errnum);
    perror("Error printed by perror");
    printf("Error opening file %s\n", wROM_File);
    return(EXIT_FAILURE);
  }

  fread(& numRowsW, sizeof(unsigned int), 1, src);
  fread(& numColsW, sizeof(unsigned int), 1, src);
  fread(& shiftRegLength, sizeof(unsigned int), 1, src);
  W_ROW_ROM = (unsigned int*) malloc(numRowsW * numColsW * sizeof( unsigned int));
  fread(W_ROW_ROM, sizeof(unsigned int), numRowsW * numColsW, src);
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
      //      infoWord[j] = (0.5 >= rDist(generator))? 1:0;
      infoWord[j] = j % 2;
    }
    startTime = clock::now();
    ldpcEncoder(infoWord, W_ROW_ROM, infoLeng, numRowsW, numColsW, shiftRegLength, codeWord);
    endTime = clock::now();
    encoderTime = endTime - startTime;

    printf("Time for encoder: %i microsec\n",
           std::chrono::duration_cast<std::chrono::microseconds>(encoderTime).count());


  char encodedFile[256];
  sprintf(encodedFile, "./evenodd%d.encoded", numBits);
  src = fopen(encodedFile, "w");
  if (src == NULL) {
    errnum = errno;
    printf("Value of errno: %d\n", errnum);
    perror("Error printed by perror");
    printf("Error opening file %s\n", encodedFile);
    return(EXIT_FAILURE);
  }
  for(unsigned int j=0; j<numBits; j++) fprintf(src,"%d\n", codeWord[j]);
  fclose(src);


    // Debug.
    // unsigned int numParityBits = numColsW;
    // for (unsigned int j=0; j< numParityBits; j++) {
    //   printf(" %i", codeWord[infoLeng+j]);
    //   if ( (j % 40) == 39)  { printf("\n"); }
    // }
    // printf("\n");
}
