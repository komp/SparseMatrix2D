// Based on the Eric's C-code implementation of ldpc decode.
//
#include <cstdio>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

#include "GPUincludes.h"
#include "LDPC.h"

#define NTHREADS   64
#define CNP_THREADS   20  // checkNodeProcessing threads

int ldpcDecoder (H_matrix *hmat, unsigned int  maxIterations, bundleElt *rSig, bundleElt *decodedPkt,
                 bundleElt *dev_rSig, bundleElt *dev_estimate, bundleElt *dev_eta, bundleElt *dev_etaByBitIndex,
                 bundleElt *dev_lambdaByCheckIndex, bundleElt *dev_parityBits,
                 unsigned int *dev_mapRC, unsigned int *dev_mapCR) {

  unsigned int numBits = hmat->numBits;
  unsigned int numChecks = hmat->numChecks;
  unsigned int maxBitsPerCheck = hmat->maxBitsPerCheck;
  unsigned int maxChecksPerBit = hmat->maxChecksPerBit;
  unsigned int nChecksByBits = numChecks*(maxBitsPerCheck+1);

  unsigned int iterCounter;
  bool allChecksPassed = false;
  unsigned int bitWeight;
  unsigned int successCount;
  unsigned int returnVal;
  unsigned int cellIndex, oneDindex ;

  unsigned int *mapRC = hmat->mapRows2Cols;
  unsigned int *mapCR = hmat->mapCols2Rows;
  bundleElt * checksArray = (bundleElt *)malloc(nChecksByBits * sizeof(bundleElt));
  bundleElt zeroBE = make_bundleElt(0.0);
  bundleElt* parityBits;
  bundleElt paritySum;

  parityBits = (bundleElt*) malloc(numChecks* sizeof(bundleElt));

  for (unsigned int check=0; check<numChecks; check++) checksArray[check] = make_bundleElt ((float)(mapRC[check]));
  for (unsigned int check=numChecks; check< nChecksByBits; check++) checksArray[check] = zeroBE;

  HANDLE_ERROR(cudaMemcpy(dev_eta, checksArray, nChecksByBits * sizeof(bundleElt), cudaMemcpyHostToDevice));

  // Re-use the checksArray.
  for (unsigned int bit = 0; bit < numBits; bit++) {
    bitWeight = mapCR[bit];
    for (unsigned int m=1; m<= bitWeight; m++) {
      cellIndex =  m * numBits + bit;
      oneDindex = mapCR[cellIndex];
      checksArray[oneDindex] = rSig[bit];
    }
  }

    HANDLE_ERROR(cudaMemcpy(dev_rSig, rSig, numBits * sizeof(bundleElt), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_lambdaByCheckIndex, checksArray, nChecksByBits * sizeof(bundleElt), cudaMemcpyHostToDevice));

    ////////////////////////////////////////////////////////////////////////////
    // Main iteration loop
    ////////////////////////////////////////////////////////////////////////////

    for(iterCounter=1;iterCounter<=maxIterations;iterCounter++) {
      checkNodeProcessingOptimalBlock <<<numChecks, CNP_THREADS>>>
        (numChecks, maxBitsPerCheck, dev_lambdaByCheckIndex, dev_eta, dev_mapRC, dev_etaByBitIndex);

      bitEstimates<<<numBits/NTHREADS+1,NTHREADS>>>
        (dev_rSig, dev_estimate, dev_etaByBitIndex, dev_lambdaByCheckIndex, dev_mapCR, numBits,maxChecksPerBit);

      calcParityBits <<<numChecks/NTHREADS+1, NTHREADS>>>
        (dev_lambdaByCheckIndex, dev_parityBits, numChecks, maxBitsPerCheck);

      allChecksPassed = true;

      //  The cpu is slightly faster than GPU DeviceReduce  to determine if any paritycheck is non-zero.
      HANDLE_ERROR(cudaMemcpy(parityBits, dev_parityBits, numChecks*sizeof(bundleElt),cudaMemcpyDeviceToHost));
      paritySum = zeroBE;
      for (unsigned int check=0; check < numChecks; check++) {
        for (unsigned int slot=0; slot< SLOTS_PER_ELT; slot++) if ((int)parityBits[check].s[slot] != 0) allChecksPassed = false;
        if (! allChecksPassed) break;
      }
      if (allChecksPassed) break;
    }
    if (iterCounter < maxIterations) {
      successCount = SLOTS_PER_ELT;
    } else {
      successCount = 0;
      paritySum = make_bundleElt(0.0);
      for (unsigned int check=0; check < numChecks; check++) paritySum += parityBits[check];
      for (unsigned int slot=0; slot< SLOTS_PER_ELT; slot++) if ((int)paritySum.s[slot] == 0) successCount++;
    }
    HANDLE_ERROR(cudaMemcpy(decodedPkt, dev_estimate, numBits*sizeof(bundleElt),cudaMemcpyDeviceToHost));

  returnVal = (successCount << 8) + iterCounter;
  return (returnVal);
}
