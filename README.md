# SparseMatrix2D
This repository is a study.
It will operate explicitly on the 1024 rate 1/2 LDPC (using min-sum).
It does not intend to do long SNR studies, etc.
I have verified that the original implementation
matches Erik's matlab implementation exactly to the level
of floating point accuracy for bit estimates at each iteration.
So, the work here is to find how quickly we can perform the
decode with the data structure(s) described below.

In the early state, at least,
this codes does NOT rely on the actual H matrix.
It reads a number of parameters and matrices that contain
the information needed for this algorithm
from a Matlab generated input file.
(These never change for the given encoding, so it did
not seem necessary to recode existing Matlab work into C.)

ALSO, this does not include an encoder section.
Read the noisy received signal, also generated with Matlab.

If the study shows promise, we start over, generalize,
and do complete testing.


===================  Key Idea =============================
the H Matrix for LDPC is very sparse.
For the case:  InfoLength = 1024, Rate 1/2
H is 1536 rows by 2560 columns
No row has more than 6 non-zero values; and
no col has more than 6 non-zero values.

We iteratively perform operations on non-zero values
first along the rows (node processing);
then  along the cols (bit esitamation).
The indexes themselves are not relevant.

the row operations are independent; and the
the col operations are independent.
This allows for very simple parallelization.
BUT,  for the row ops we want the Sparse Matrix in row-order;
while for the col ops we wnat the Sparse Matrix in col-order.

So, wth a row-order copy of H,
we perform the node processing in parallel.

Then we encounter the ONE problem, how to efficiently
transform this H matrix in row-order to one
in col-order, so that we can parallelize the bit estimates step.
(and then, transform this output back to row-order for the next iteration).
