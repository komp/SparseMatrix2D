
__global__ void
checkNodeProcessingOptimal (unsigned int numChecks, unsigned int maxBitsForCheck,
                            float *lambdaByCheckIndex, float *eta);
__global__ void
checkNodeProcessingOptimalBlock (unsigned int numChecks, unsigned int maxBitsForCheck,
                                 float *lambdaByCheckIndex, float *eta,
                                 unsigned int* mapRows2Cols, float *etaByBitIndex);

__global__ void
checkNodeProcessingMinSum (unsigned int numChecks, unsigned int maxBitsForCheck,
                           float *lambdaByCheckIndex, float *eta);

__global__ void
checkNodeProcessingMinSumBlock (unsigned int numChecks, unsigned int maxBitsForCheck,
                                float *lambdaByCheckIndex, float *eta);
__global__ void
checkNodeProcessingOptimalNaive (unsigned int numChecks, unsigned int maxBitsForCheck,
                                 float *lambdaByCheckIndex, float *eta);
__global__ void
bitEstimates(float *rSig, float *etaByBitIndex, float *lambdaByCheckIndex, unsigned int *hd,
             unsigned int *mapCols2Rows, unsigned int numBits, unsigned int maxChecksForBit);

__global__ void
transposeRC (unsigned int* map, float *checkRows, float *bitRows,
             unsigned int numChecks, unsigned int maxBitsForCheck);

__global__ void
copyBitsToCheckmatrix (unsigned int* map, float *bitEstimates, float *checkRows,
                       unsigned int numBits, unsigned int maxChecksForBit);

__global__ void
calcParityBits (unsigned int* cHat, unsigned int *parityBits, unsigned int numChecks, unsigned int maxBitsForCheck);


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

void initLdpcDecoder  (unsigned int numChecksI, unsigned int numBitsI,
                       unsigned int maxBitsForCheckI, unsigned int maxChecksForBitI,
                       unsigned int *mapRows2Cols, unsigned int *mapCols2Rows);

int ldpcDecoderWithInit (float *rSig, unsigned int  maxIterations, unsigned int *decision, float *estimates);

void remapRows2Cols (unsigned int numChecks, unsigned int numBits,
                     unsigned int maxBitsPerCheck, unsigned int maxChecksPerBit,
                     unsigned int *r2c, unsigned int *newR2C);

void remapCols2Rows (unsigned int numChecks, unsigned int numBits,
                     unsigned int maxBitsPerCheck, unsigned int maxChecksPerBit,
                     unsigned int *c2r, unsigned int *newC2R);
