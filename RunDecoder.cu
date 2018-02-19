#include <random>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>

#include <chrono>

#include "GPUincludes.h"
#include "LDPC.h"

#define HISTORY_LENGTH  20

int main (int argc, char **argv) {

  using clock = std::chrono::steady_clock;

  clock::time_point startTime;
  clock::time_point endTime;
  clock::duration oneTime;
  clock::duration allTime;

  int status;
  unsigned int numChecks, numBits;
  unsigned int numRowsW, numColsW, numParityBits, shiftRegLength;
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


  if (argc < 7) {
    printf("usage:  RunDecoder <infoLength> <r-numerator> <r-denominator> <ebno> <numpackets> <maxIterations>\n" );
    exit(-1);
  }
  infoLeng = atoi(argv[1]);
  rnum = atoi(argv[2]);
  rdenom = atoi(argv[3]);
  rNominal = float(rnum)/float(rdenom);
  ebno = atof(argv[4]);
  how_many = atoi(argv[5]);
  maxIterations = atoi(argv[6]);
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
  printf("numBits = %i, numChecks = %i\n", numBits, numChecks);
  printf("infoLeng = %i, numParityBits = %i (%i), numBits = %i\n",
         infoLeng, numParityBits, infoLeng + numParityBits, numBits);
  printf("maxChecksPerBit = %i maxBitsPerCheck = %i\n", hmat->maxChecksPerBit, hmat->maxBitsPerCheck);
  printf("ebn0 = %f, sigma = %f\n", ebno, sigma2);

  // ///////////////////////////////////////////

  unsigned int decision[numBits];
  bundleElt estimates[numBits];

  unsigned int successes = 0;
  unsigned int iterationSum = 0;
  int totalPackets;

  int itersHistory[HISTORY_LENGTH+1];
  int iters;

  unsigned int* infoWord;
  unsigned int* codeWord;
  float s, noise;
  float* receivedSig;
  bundleElt* receivedBundle;

  infoWord = (unsigned int *)malloc(infoLeng * sizeof(unsigned int));
  codeWord = (unsigned int *)malloc((infoLeng+numParityBits) * sizeof(unsigned int));
  receivedSig = (float *)malloc(numBits * sizeof(float));
  receivedBundle = (bundleElt *)malloc(numBits * sizeof(bundleElt));

  // An ugly way to intialize variable allTime (accumulated interesting time) to zero.
  startTime = clock::now();
  allTime = startTime - startTime;

  initLdpcDecoder (hmat);

  FILE *fd;
  fd = fopen("./evenodd1408.encoded" , "r");
  if (fd == NULL) {
    printf("Error opening file %s\n", "./evenodd1408.encoded");
    return(-1);
  }
  for (int i=0; i< numBits; i++) fscanf(fd, "%d", &codeWord[i]);
  fclose(fd);

  for (unsigned int i=1; i<= how_many; i++) {
    for (unsigned int slot=0; slot < SLOTS_PER_ELT; slot++) {
      for (unsigned int j=0; j < infoLeng; j++) {
        infoWord[j] = (0.5 >= rDist(generator))? 1:0;
      }
      //edk  when commented, this implies we are using a "standard" infoWord
      //edk  of alternating 0/1.
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

      for (unsigned int j=0; j < numBits; j++ ) receivedBundle[j].s[slot] = receivedSig[j]; break;
    }

    // Finally, ready to decode signal

    startTime = clock::now();
    iters = ldpcDecoderWithInit (hmat, receivedBundle, maxIterations, decision, estimates);
    endTime = clock::now();
    oneTime = endTime - startTime;
    allTime = allTime + oneTime;

    successes += (iters & 7);
    iters = iters >> 3;
    iterationSum = iterationSum + iters;
    if ( i <= HISTORY_LENGTH) { itersHistory[i] = iters;}
    if (i % 1000 == 0) printf(" %i Successes out of %i inputs (%i msec).\n",
                              successes, i, std::chrono::duration_cast<std::chrono::milliseconds>(allTime).count());
  }

  totalPackets = SLOTS_PER_ELT * how_many;
  printf("%i msec to decode %i packets.\n", std::chrono::duration_cast<std::chrono::milliseconds>(allTime).count(), totalPackets);

  printf(" %i Successes out of %i packets. (%.1f%%)\n", successes, totalPackets, 100.0 * successes/ totalPackets);
  printf(" %i cumulative iterations, or about %.1f per packet.\n", iterationSum, iterationSum/(float)how_many);
  printf("Number of iterations for the first few packets:  ");
  for (unsigned int i=1; i<= MIN(how_many, HISTORY_LENGTH); i++) {printf(" %i", itersHistory[i]);}
  printf ("\n");

  cudaDeviceReset();
}
