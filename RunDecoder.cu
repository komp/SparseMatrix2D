#include <random>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>

#define ABS(a)  (((a) < (0)) ? (-(a)) : (a))

#define HISTORY_LENGTH  20

#include "GPUincludes.h"

#define MAXITERATIONS  60

void ldpcEncoder (unsigned int *infoWord, unsigned int* W_ROW_ROM,
                  unsigned int numMsgBits, unsigned int numRowsinRom, unsigned int numParBits,
                   unsigned int shiftRegLength,
                  unsigned int *codeWord);

int ldpcDecoder (float *rSig, unsigned int numChecks, unsigned int numBits,
                 unsigned int maxBitsForCheck, unsigned int maxChecksForBit,
                 unsigned int *mapRows2Cols, unsigned int *mapCols2Rows,
                 unsigned int maxIterations,
                 unsigned int *decision,
                 float *estimates);

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
  float ebno;
  unsigned int how_many;
  float rNominal;
  float No, sigma2, lc;

  unsigned int  seed = 163331;
  /*  or use this to get a fresh sequence each time the program is run.
  std::random_device  rd;  //Will be used to obtain a seed for the random number engine
  std::mt19937 generator(rd()); //Standard mersenne_twister_engine seeded with rd()
  */
  std::mt19937 generator(seed); //Standard mersenne_twister_engine
  std::uniform_real_distribution<> rDist(0, 1);

  // normal distribution for noise.
  // NOTE,  we are using the same random number generator ("generator") here.
  std::normal_distribution<float> normDist(0.0, 1.0);


  if (argc < 6) {
    printf("usage:  RunDecoder <infoLength> <r-numerator> <r-denominator> <ebno> <numpackets>\n" );
    exit(-1);
  }
  infoLeng = atoi(argv[1]);
  rnum = atoi(argv[2]);
  rdenom = atoi(argv[3]);
  rNominal = float(rnum)/float(rdenom);
  ebno = atof(argv[4]);
  how_many = atoi(argv[5]);
  sprintf(mapFile, "./G_and_H_Matrices/Maps_%d%d_%d.bin", rnum, rdenom, infoLeng);
  sprintf(wROM_File, "./G_and_H_Matrices/W_ROW_ROM_%d%d_%d.bin", rnum, rdenom, infoLeng);


  // Noise variance and log-likelihood ratio (LLR) scale factor. Because
  // Ec = R*Eb is unity (i.e. we transmit +1's and -1's), then No = 1/(R*EbNo).
  No = 1/(rNominal * pow(10,(ebno/10)));
  sigma2 = No/2;
  // When r is scaled by Lc it results in precisely scaled LLRs
  lc = 4/No;


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
  printf("infoLeng = %i, numParityBits = %i (%i), numBits = %i\n",
         infoLeng, numParityBits, infoLeng + numParityBits, numBits);
  printf("%i %i\n", maxChecksForBit, maxBitsForCheck);
  printf("ebn0 = %f, sigma = %f\n", ebno, sigma2);

  // ///////////////////////////////////////////

  unsigned int decision[numBits];
  float estimates[numBits];

  unsigned int successes = 0;
  unsigned int iterationSum = 0;

  // Allocate CUDA events that we'll use for timing
  cudaEvent_t start;
  HANDLE_ERROR(cudaEventCreate(&start));
  cudaEvent_t stop;
  HANDLE_ERROR(cudaEventCreate(&stop));
  float msecRecord = 0.0f;
  float msecTotal = 0.0f;

  int itersHistory[HISTORY_LENGTH+1];
  int iters;

  unsigned int* infoWord;
  unsigned int* codeWord;
  float s, noise;
  float* receivedSig;

  infoWord = (unsigned int *)malloc(infoLeng * sizeof(unsigned int));
  codeWord = (unsigned int *)malloc((infoLeng+numParityBits) * sizeof(unsigned int));
  receivedSig = (float *)malloc(numBits * sizeof(float));

  for (unsigned int i=1; i<= how_many; i++) {

    for (unsigned int j=0; j < infoLeng; j++) {
      infoWord[j] = (0.5 >= rDist(generator))? 1:0;
    }
    ldpcEncoder(infoWord, W_ROW_ROM, infoLeng, numRowsW, numColsW, shiftRegLength, codeWord);

    // Modulate the codeWord, and add noise
    for (unsigned int j=0; j < (infoLeng+numParityBits) ; j++) {
      s     = 2*float(codeWord[j]) - 1;
      // AWGN channel
      noise = sqrt(sigma2) * normDist(generator);
      // When r is scaled by Lc it results in precisely scaled LLRs
      receivedSig[j]  = lc*(s + noise);
    }

    // The LDPC codes are punctured, so the r we feed to the decoder is
    // longer than the r we got from the channel. The punctured positions are filled in as zeros
    for (unsigned int j=(infoLeng+numParityBits); j<numBits; j++) receivedSig[j] = 0.0;

    // Finally, ready to decode signal

    HANDLE_ERROR(cudaEventRecord(start, NULL));
    iters = ldpcDecoder(receivedSig, numChecks, numBits,
                        maxBitsForCheck, maxChecksForBit, mapRows2Cols, mapCols2Rows, MAXITERATIONS,
                        decision, estimates);
    // Record the stop event
    HANDLE_ERROR( cudaEventRecord(stop, NULL));
    HANDLE_ERROR( cudaEventSynchronize(stop));
    HANDLE_ERROR( cudaEventElapsedTime(&msecRecord, start, stop));
    msecTotal += msecRecord;

    if (iters < MAXITERATIONS) {successes++;}
    iterationSum = iterationSum + iters;
    if ( i <= HISTORY_LENGTH) { itersHistory[i] = iters;}
    if (i % 1000 == 0) printf(" %i Successes out of %i inputs (%.0f msec).\n", successes, i, msecTotal);
  }

  printf("%.0f msec to decode %i packets.\n", msecTotal, how_many);

  printf(" %i Successes out of %i inputs.\n", successes, how_many);
  printf(" %i cumulative iterations, or about %.1f per packet.\n", iterationSum, iterationSum/(float)how_many);
  printf("Number of iterations for the first few packets:  ");
  for (unsigned int i=1; i<= MIN(how_many, HISTORY_LENGTH); i++) {printf(" %i", itersHistory[i]);}
  printf ("\n");

  cudaDeviceReset();
}
