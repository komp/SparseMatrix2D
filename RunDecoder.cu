#include <math.h>
#include <string.h>

#include "mat.h"
#include "matrix.h"

#include "GPUincludes.h"

#define MAXITERATIONS  200

int ldpcDecoder (float *rSig, unsigned int numChecks, unsigned int numBits,
                 unsigned int *bitsForCheck, unsigned int *checksForBit,
                 int maxBitsForCheck, int maxChecksForBit,
                 unsigned int *mapRows2Cols, unsigned int *mapCols2Rows,
                 unsigned int maxIterations,
                 int *decision);

int main () {
  MATFile *pmat;
  mxArray *p1, *p2, *p3, *p4, *p5, *p6, *p7, *p8;
  unsigned int numChecks, numBits, maxBitsForCheck, maxChecksForBit, infoLength;
  unsigned int *bitsForCheck, *checksForBit;
  unsigned int  *mapRows2Cols, *mapCols2Rows;
  double *receivedSigs;
  unsigned int sigLength, numSigs;
  //  char filename[] ="~/Projects/CodedAPSK/matlabDecoder/Data/Signals/sig_2.5.mat";
  char filename[] ="~/APSK/Data/Signals/sig_2.5_short.mat";

  pmat = matOpen(filename, "r");
  if (pmat == NULL) {
    printf("Error opening file %s\n", filename);
    return(EXIT_FAILURE);
  }

  p1 = matGetVariable(pmat, "bitsForCheck");
  bitsForCheck = (unsigned int *)mxGetData(p1);
  numChecks = (unsigned int)mxGetM(p1);

  p2 = matGetVariable(pmat, "checksForBit");
  checksForBit = (unsigned int *)mxGetData(p2);
  numBits = (unsigned int)mxGetM(p2);

  p3 = matGetVariable(pmat, "maxBitsForCheck");
  maxBitsForCheck = (unsigned int)mxGetScalar(p3);

  p4 = matGetVariable(pmat, "maxChecksForBit");
  maxChecksForBit = (unsigned int)mxGetScalar(p4);

  p5 = matGetVariable(pmat, "infoLength");
  infoLength = (unsigned int)mxGetScalar(p5);

  p6 = matGetVariable(pmat, "mapRows2Cols");
  mapRows2Cols  = (unsigned int *)mxGetData(p6);

  p7 = matGetVariable(pmat, "mapCols2Rows");
  mapCols2Rows  = (unsigned int *)mxGetData(p7);

  p8 = matGetVariable(pmat, "receivedSigs");
  receivedSigs = mxGetPr(p8);
  sigLength = mxGetM(p8);
  numSigs = mxGetN(p8);

  matClose(pmat);
  // ///////////////////////////////////////////

  int decision[sigLength];
  int niters[sigLength];
  float rSig[sigLength];
  unsigned int sigStartIndex;

  int successes = 0;
  int iterationSum = 0;
  int numreps = 1;

  // Allocate CUDA events that we'll use for timing
  cudaEvent_t start;
  HANDLE_ERROR(cudaEventCreate(&start));
  cudaEvent_t stop;
  HANDLE_ERROR(cudaEventCreate(&stop));
  HANDLE_ERROR(cudaEventRecord(start, NULL));

  for (int reps = 0; reps < numreps; reps++) {
    for (unsigned int i=0; i<numSigs; i++) {
      sigStartIndex = i * sigLength;
      for (unsigned int j=0; j<sigLength; j++) {rSig[j] =  (float)receivedSigs[sigStartIndex+j];   }
      niters[i] = ldpcDecoder(rSig, numChecks, numBits, bitsForCheck, checksForBit,
                              maxBitsForCheck, maxChecksForBit, mapRows2Cols, mapCols2Rows, MAXITERATIONS, decision);
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
  for (int i=0; i<10; i++) {printf(" %i", niters[i]);}
  printf ("\n");
}
