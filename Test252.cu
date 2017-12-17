#include <random>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>

#include <chrono>

#include "GPUincludes.h"
#include "LDPC.h"

#define HISTORY_LENGTH  20
#define MAXITERATIONS  60

int main (int argc, char **argv) {

  using clock = std::chrono::steady_clock;

  int status;
  unsigned int numChecks, numBits;
  unsigned int numParityBits;

  H_matrix *hmat = (H_matrix*) malloc(sizeof(H_matrix));

  char H_Alist_File[256];
  FILE *src;
  //edk int errnum;
  unsigned int infoLeng, rnum, rdenom;

  if (argc < 4) {
    printf("usage:  Test252 <infoLength> <r-numerator> <r-denominator>\n" );
    exit(-1);
  }
  infoLeng = atoi(argv[1]);
  rnum = atoi(argv[2]);
  rdenom = atoi(argv[3]);

  sprintf(H_Alist_File, "./G_and_H_Matrices/A_252.alist");
  status = ReadAlistFile(hmat, H_Alist_File);
  if ( status != 0) {
    printf ("Unable to read alist file: %s\n", H_Alist_File);
    exit(-1);
  }
  numBits = hmat->numBits;
  numChecks = hmat->numChecks;

  numParityBits = 252;
  infoLeng = 252;
  unsigned int* codeWord;
  unsigned int* bits4check = hmat->bitsForCheck;

  codeWord = (unsigned int *)malloc((infoLeng+numParityBits) * sizeof(unsigned int));

  const char *encodedFile = "./evenodd.encoded";
  src = fopen(encodedFile, "r");
  if (src == NULL) {
    printf ("Unable to open encoded signal file: %s\n", encodedFile);
    exit(-1);
  }
  for (unsigned int locali=0; locali<infoLeng+numParityBits; locali++) {
    fscanf(src,"%d", &(codeWord[locali]));
  }
  for (unsigned int locali=0; locali<infoLeng+numParityBits; locali++) {
    printf("%1d", codeWord[locali]);
    if ((locali+1) % 100 == 0) printf("\n");
  }
  fclose(src);

  unsigned int sum;
  unsigned int bitValue, bit;
  for (unsigned int check = 0; check < numChecks; check++) {
    sum = 0;
    for (unsigned int index = 0; index < hmat->maxBitsPerCheck; index++) {
      bit = bits4check[check * hmat->maxBitsPerCheck + index] -1;
      bitValue = codeWord[bit];
      sum += bitValue;
      printf ("%4i", bit);
    }
    printf (":  %1i  %i  %s\n", sum %2, sum, (sum%2 != 0)? "ERROR" : "");
  }
}
