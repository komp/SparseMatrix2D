#include  "bundleElt.h"

typedef struct{
	unsigned int numBits;	     //!< Code length
	unsigned int numChecks;       //!< Number of Parity-Check Bits (equivalently equations)
	unsigned int maxChecksPerBit; //!< Maximum Bit Node Weight
	unsigned int maxBitsPerCheck; //!< Maximum Check Node Weight
	unsigned int *weightForBit;	 //!< Weight per Bit Node (redundant for regular LDPC codes)
	unsigned int *weightForCheck; //!< Weight per Check Node (redundant for regular LDPC codes)
	unsigned int *bitsForCheck;   //!< Indexes of Bit Nodes the Check Nodes are connected to
	unsigned int *checksForBit;   //!< Indexes of Check Nodes the Bit Nodes are connected to
	unsigned int *mapRows2Cols;   //!< Memory offsets LUT when Bit Nodes write to Check Nodes
	unsigned int *mapCols2Rows;   //!< Memory offsets LUT when Check Nodes write to Bit Nodes
} H_matrix;


__global__ void
checkNodProcessingOptimal (unsigned int numChecks, unsigned int maxBitsForCheck,
                           bundleElt *lambdaByCheckIndex, bundleElt *eta);

__global__ void
checkNodeProcessingOptimalBlock (unsigned int numChecks, unsigned int maxBitsForCheck,
                                 bundleElt *lambdaByCheckIndex, bundleElt *eta,
                                 unsigned int* mapRows2Cols, bundleElt *etaByBitIndex,
                                 unsigned int nChecksByBits, unsigned int nBitsByChecks, unsigned int nBundles);

__global__ void
checkNodeProcessingMinSum (unsigned int numChecks, unsigned int maxBitsForCheck,
                           bundleElt *lambdaByCheckIndex, bundleElt *eta,
                           unsigned int* mapRows2Cols, bundleElt *etaByBitIndex,
                           unsigned int nChecksByBits, unsigned int nBitsByChecks, unsigned int nBundles);

__global__ void
checkNodeProcessingMinSumBlock (unsigned int numChecks, unsigned int maxBitsForCheck,
                                bundleElt *lambdaByCheckIndex, bundleElt *eta);
__global__ void
checkNodeProcessingOptimalNaive (unsigned int numChecks, unsigned int maxBitsForCheck,
                                 bundleElt *lambdaByCheckIndex, bundleElt *eta);
__global__ void
bitEstimates(bundleElt *rSig, bundleElt *etaByBitIndex, bundleElt *lambdaByCheckIndex, bundleElt *hd,
             unsigned int *mapCols2Rows, unsigned int numBits, unsigned int maxChecksForBit,
             unsigned int nChecksByBits, unsigned int nBitsByChecks, unsigned int nBundles);


__global__ void
transposeRC (unsigned int* map, bundleElt *checkRows, bundleElt *bitRows,
             unsigned int numChecks, unsigned int maxBitsForCheck);

__global__ void
copyBitsToCheckmatrix (unsigned int* map, bundleElt *bitEstimates, bundleElt *checkRows,
                       unsigned int numBits, unsigned int maxChecksForBit,
                       unsigned int nChecksByBits, unsigned int nBitsByChecks, unsigned int nBundles);
__global__ void
calcParityBits (bundleElt *cHat, bundleElt *parityBits, unsigned int numChecks, unsigned int maxBitsForCheck,
                unsigned int nChecksByBits, unsigned int nBundles);



void ldpcEncoder (unsigned int *infoWord, unsigned int* W_ROW_ROM,
                  unsigned int numMsgBits, unsigned int numRowsinRom, unsigned int numParBits,
                   unsigned int shiftRegLength,
                  unsigned int *codeWord);

int ldpcDecoder (bundleElt *rSig, unsigned int numChecks, unsigned int numBits,
                 unsigned int maxBitsForCheck, unsigned int maxChecksForBit,
                 unsigned int *mapRows2Cols, unsigned int *mapCols2Rows,
                 unsigned int maxIterations,
                 unsigned int *decision,
                 bundleElt *estimates);

int ReadAlistFile(H_matrix *hmat, const char *AlistFile);

void initLdpcDecoder  (H_matrix *hmat, unsigned int nBundles);

int ldpcDecoderWithInit (H_matrix *hmat, bundleElt *rSig, unsigned int  maxIterations, unsigned int *decision, bundleElt *estimates, unsigned int nBundles);

void remapRows2Cols (unsigned int numChecks, unsigned int numBits,
                     unsigned int maxBitsPerCheck, unsigned int maxChecksPerBit,
                     unsigned int *r2c, unsigned int *newR2C);

void remapCols2Rows (unsigned int numChecks, unsigned int numBits,
                     unsigned int maxBitsPerCheck, unsigned int maxChecksPerBit,
                     unsigned int *c2r, unsigned int *newC2R);
