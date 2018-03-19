#include <random>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>

#include <chrono>

#include "GPUincludes.h"
#include "LDPC.h"

#include "loader_pool.h"
#include "decoder_pool.h"

#define HISTORY_LENGTH  20

int main (int argc, char **argv) {

  using clock = std::chrono::steady_clock;

  clock::time_point startTime;
  clock::time_point endTime;
  clock::duration allTime;

  int status;
  unsigned int numChecks, numBits;
  unsigned int numRowsW, numColsW, numParityBits, shiftRegLength;
  unsigned int numThreads, sigBufferLength;
  unsigned int *W_ROW_ROM;

  H_matrix *hmat = (H_matrix*) malloc(sizeof(H_matrix));

  int maxIterations;
  char H_Alist_File[256];
  char wROM_File[256];
  FILE *src;
  int errnum;
  unsigned int infoLeng, rnum, rdenom;
  float ebno;
  unsigned int how_many;
  float rNominal;
  float No, sigma2, lc;

  if (argc < 8) {
    printf("usage:  RunDecoder <infoLength> <r-numerator> <r-denominator> <ebno> <numpackets> <maxIterations> <# Threads>\n" );
    exit(-1);
  }
  infoLeng = atoi(argv[1]);
  rnum = atoi(argv[2]);
  rdenom = atoi(argv[3]);
  rNominal = float(rnum)/float(rdenom);
  ebno = atof(argv[4]);
  how_many = atoi(argv[5]);
  maxIterations = atoi(argv[6]);
  numThreads = atoi(argv[7]);

  sprintf(H_Alist_File, "./G_and_H_Matrices/H_%d%d_%d.alist", rnum, rdenom, infoLeng);
  sprintf(wROM_File, "./G_and_H_Matrices/W_ROW_ROM_%d%d_%d.binary", rnum, rdenom, infoLeng);


  // Noise variance and log-likelihood ratio (LLR) scale factor. Because
  // Ec = R*Eb is unity (i.e. we transmit +1's and -1's), then No = 1/(R*EbNo).
  No = 1/(rNominal * pow(10,(ebno/10)));
  sigma2 = No/2;
  // When r is scaled by Lc it results in precisely scaled LLRs
  lc = 4/No;

  status = ReadAlistFile(hmat, H_Alist_File);
  if ( status != 0) {
    printf ("Unable to read alist file: %s\n", H_Alist_File);
    exit(-1);
  }
  numBits = hmat->numBits;
  numChecks = hmat->numChecks;

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
  numParityBits = numColsW;
  fclose(src);


  printf("parameters have been read.\n");
  printf("SLOTS_PER_ELT = %d\n", SLOTS_PER_ELT);
  printf("numBits = %i, numChecks = %i\n", numBits, numChecks);
  printf("infoLeng = %i, numParityBits = %i (%i), numBits = %i\n",
         infoLeng, numParityBits, infoLeng + numParityBits, numBits);
  printf("maxChecksPerBit = %i maxBitsPerCheck = %i\n", hmat->maxChecksPerBit, hmat->maxBitsPerCheck);
  printf("ebn0 = %f, sigma = %f\n", ebno, sigma2);

  // ///////////////////////////////////////////

  bundleElt *receivedSigs;
  bundleElt *decodedSigs;
  bundleElt zeroBE = make_bundleElt(0.0);
  unsigned int sigIndex;

  unsigned int successes = 0;
  unsigned int iterationSum = 0;
  int iters;

  // An ugly way to intialize variable allTime (accumulated interesting time) to zero.
  startTime = clock::now();
  allTime = startTime - startTime;

  sigBufferLength = 2 * numThreads;
  receivedSigs = (bundleElt*) malloc(sigBufferLength * numBits * sizeof(bundleElt));
  decodedSigs  = (bundleElt*) malloc(sigBufferLength * numBits * sizeof(bundleElt));

  std::vector<Tpkt> buffer;
  buffer.reserve(sigBufferLength);

  DecoderPool* decoders = new DecoderPool(hmat, maxIterations, numThreads);
  LoaderPool* pktLoader = new LoaderPool(infoLeng, numBits, numParityBits, W_ROW_ROM, numRowsW, numColsW, shiftRegLength, sigma2, lc);

  for (unsigned int i=0; i< sigBufferLength; i++) {
    sigIndex = i* numBits;
    buffer.emplace_back(&receivedSigs[sigIndex],  & decodedSigs[sigIndex]);
    buffer[i].state = LOADING;
    pktLoader->schedule_job(&buffer[i]);
  }

  unsigned int pktsDecoded = 0;
  startTime = clock::now();

  while (pktsDecoded < how_many) {
    for (unsigned int i=0; i< sigBufferLength; i++) {
      sigIndex = i * (numBits+1);
      switch(buffer[i].state) {
      case LOADING :
        if (buffer[i].loadStamp != 0 ) {
          buffer[i].loadStamp = 0;
          buffer[i].state = DECODING;
          decoders->schedule_job(&buffer[i]);
        }
        break;
      case DECODING :
        if (buffer[i].decodeStamp != 0 ) {
          iters = buffer[i].decodeStamp;
          successes += iters >> 8;
          iters = iters & 0xff ;
          iterationSum = iterationSum + iters;
          pktsDecoded += SLOTS_PER_ELT;
          if ((pktsDecoded % 10000) == 0) {
            printf (" .");
            fflush(stdout);
          }
          buffer[i].decodeStamp = 0;
          buffer[i].state = LOADING;
          pktLoader->schedule_job(&buffer[i]);
        }
        break;
      }
    }
  }
  endTime = clock::now();
  allTime = endTime - startTime;

  printf("\n");
  delete pktLoader;
  delete decoders;

  printf("%i msec to decode %i packets.\n",std::chrono::duration_cast<std::chrono::milliseconds>(allTime).count(),pktsDecoded);
  printf(" %i Successes out of %i packets. (%.2f%%)\n", successes, pktsDecoded, 100.0 * successes/ pktsDecoded);
  printf("Information rate: %.2f Mbps\n", successes * infoLeng / (1000.0 * std::chrono::duration_cast<std::chrono::milliseconds>(allTime).count()));
  // SLOTS_PER_ELT packets are handled in each iteration, so...
  iterationSum = iterationSum * SLOTS_PER_ELT;
  printf(" %i cumulative iterations, or about %.1f per packet.\n", iterationSum, iterationSum/(float)pktsDecoded);
  //  printf("Number of iterations for the first few packets:  ");
  //  for (unsigned int i=1; i<= MIN(pktsDecoded, HISTORY_LENGTH); i++) {printf(" %i", itersHistory[i]);}
  //  printf ("\n");

  cudaDeviceReset();
}
