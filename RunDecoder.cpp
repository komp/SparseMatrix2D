#include <math.h>
#include <string.h>

#include <iostream>
#include <chrono>
#include <ctime>

#include "mat.h"
#include "matrix.h"

#define MAXITERATIONS  200

int ldpcDecoder (const double *rSig, const unsigned int numChecks, const unsigned int numBits,
                  const unsigned int *bitsForCheck, const unsigned int *checksForBit,
                  const int maxBitsForCheck, const int maxChecksForBit,
                  const unsigned int *mapRows2Cols, const unsigned int *mapCols2Rows,
                  const unsigned int maxIterations,
                  unsigned char *decision);


int main () {
  double *pR = NULL;
  MATFile *pmat;
  mxArray *p1, *p2, *p3, *p4, *p5, *p6, *p7, *p8;
  unsigned int numChecks, numBits, maxBitsForCheck, maxChecksForBit, infoLength;
  unsigned int *bitsForCheck = NULL, *checksForBit = NULL;
  unsigned int  *mapRows2Cols = NULL, *mapCols2Rows = NULL;
  double *receivedSigs = NULL;
  double *sigPtr;
  unsigned int sigLength, numSigs;
  unsigned int ptrIndex = 0;
  unsigned int *xx_M = NULL;
  //  char filename[] ="~/Projects/CodedAPSK/matlabDecoder/Data/Signals/sig_2.5.mat";
  char filename[] ="~/APSK/Data/Signals/sig_2.5.mat";

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

  printf("(%i, %i), (%i,%i), %i, %i\n",
         numChecks, numBits, maxBitsForCheck, maxChecksForBit, infoLength, numSigs);

  unsigned char decision[sigLength];
  int niters[sigLength];
  int successes = 0;
  int iterationSum = 0;

  auto started = std::chrono::high_resolution_clock::now();

  sigPtr = receivedSigs;
  for (int i=0; i<numSigs; i++) {
    niters[i] = ldpcDecoder(sigPtr, numChecks, numBits, bitsForCheck, checksForBit,
                            maxBitsForCheck, maxChecksForBit, mapRows2Cols, mapCols2Rows, MAXITERATIONS,
                            decision);
    if (niters[i] < MAXITERATIONS) {successes++;}
    iterationSum = iterationSum + niters[i];
    sigPtr = sigPtr + sigLength;
  }
  auto done = std::chrono::high_resolution_clock::now();
  printf(" %i msec for %i packets.\n",
         std::chrono::duration_cast<std::chrono::milliseconds>(done-started).count(),
         numSigs);

  printf(" %i Successes out of %i packets.\n", successes, numSigs);
  printf(" %i cumulative iterations, or about %.1f per packet.\n", iterationSum, iterationSum/(float)numSigs);
  printf("Number of iterations for the first few packets:  ");
  for (int i=0; i<10; i++) {printf(" %i", niters[i]);}
  printf ("\n");
}
