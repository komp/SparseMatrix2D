#include <math.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>

#define ABS(a)  (((a) < (0)) ? (-(a)) : (a))

#include "GPUincludes.h"

#define MAXITERATIONS  200

int ldpcDecoder (float *rSig, unsigned int numChecks, unsigned int numBits,
                 unsigned int maxBitsForCheck, unsigned int maxChecksForBit,
                 unsigned int *mapRows2Cols, unsigned int *mapCols2Rows,
                 unsigned int maxIterations,
                 unsigned int *decision,
                 float *estimates);

int main () {
  unsigned int numChecks, numBits, maxBitsForCheck, maxChecksForBit;
  unsigned int  *mapRows2Cols, *mapCols2Rows;
  float *receivedSigs;
  unsigned int sigLength, numSigs;
  char mapFile[] = "/users/komp/APSK/GPUtests/SparseMatrix2D/SampleData/Maps1024.bin";
  char sigFile[] = "/users/komp/APSK/GPUtests/SparseMatrix2D/SampleData/sig_2.5.bin";
  FILE *src;
  int errnum;

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

  src = fopen(sigFile, "r");
  if (src == NULL) {
    errnum = errno;
    printf("Value of errno: %d\n", errnum);
    perror("Error printed by perror");
    printf("Error opening file %s\n", sigFile);
    return(EXIT_FAILURE);
  }

  fread(& numSigs, sizeof(unsigned int), 1, src);
  fread(& sigLength, sizeof(unsigned int), 1, src);
  receivedSigs = (float *) malloc (sizeof(float)*numSigs*sigLength);
  fread (receivedSigs, sizeof(float), numSigs*sigLength, src);
  fclose(src);

  printf("parameters have been read.\n");
  printf("numBits = %i, numChecks = %i\n", numBits, numChecks);
  printf("%i %i %i %i\n", maxChecksForBit, maxBitsForCheck, sigLength, numSigs);
  // ///////////////////////////////////////////

  unsigned int decision[sigLength];
  unsigned int niters[sigLength];
  float estimates[sigLength];
  unsigned int sigStartIndex;

  unsigned int successes = 0;
  unsigned int iterationSum = 0;
  unsigned int numreps = 1;

  // Allocate CUDA events that we'll use for timing
  cudaEvent_t start;
  HANDLE_ERROR(cudaEventCreate(&start));
  cudaEvent_t stop;
  HANDLE_ERROR(cudaEventCreate(&stop));
  HANDLE_ERROR(cudaEventRecord(start, NULL));

  for (unsigned int reps = 0; reps < numreps; reps++) {
    for (unsigned int i=0; i<numSigs; i++) {
      sigStartIndex = i * sigLength;
      niters[i] = ldpcDecoder(& receivedSigs[sigStartIndex], numChecks, numBits,
                              maxBitsForCheck, maxChecksForBit, mapRows2Cols, mapCols2Rows, MAXITERATIONS,
                              decision, estimates);
      if (niters[i] < MAXITERATIONS) {successes++;}
      iterationSum = iterationSum + niters[i];  }
  }
  // Record the stop event
  HANDLE_ERROR( cudaEventRecord(stop, NULL));
  // Wait for the stop event to complete
  HANDLE_ERROR( cudaEventSynchronize(stop));
  float msecTotal = 0.0f;
  HANDLE_ERROR( cudaEventElapsedTime(&msecTotal, start, stop));
  printf("%f msec to decode %i packets.\n", msecTotal, numSigs* numreps);

  printf(" %i Successes out of %i inputs.\n", successes, numSigs);
  printf(" %i cumulative iterations, or about %.1f per packet.\n", iterationSum, iterationSum/(float)numSigs);
  printf("Number of iterations for the first few packets:  ");
  for (unsigned int i=0; i<10; i++) {printf(" %i", niters[i]);}
  printf ("\n");
}
