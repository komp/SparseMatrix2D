#ifndef _BUNDLE_ELT_H_
#define _BUNDLE_ELT_H_

#include "GPUincludes.h"

/*! \file bundleElt.h
*    \brief Header file for data type to hold one element of a bundle.
*
*  A bundle is the group of packets that are decoded at the same time.
*  Each invocation of BN_processing or CN_processing is applied to
*  every packet in the bundle.
*
*  The packets are collected into bundleElements (bundleElt).
*  bundleElt[n]  contains the n'th value for each packet in the bundle
*
*  bundleElt is typically implemented as a vector of a basic type
*  (unsigned int, is an example).
*  One or more samples may be stored in each element of this vector
*  via packing.  (The original implementation, stored 4 samples
*  (8 bits) in each unsigned int.
*
* It is relatively simple to alter the number of packets placed in a bundle.
*
*  The number of packets in a bundle is controlled by two values:
*  SLOTS_PER_ELT   -- length of the vector
*  SAMPLES_PER_SLOT -- samples stored in each vector element.
*  and related:
*  SAMPLE_WIDTH    -- number of bits required for a sample.
*
*  You must provide an implementation of the bundleElt structure
*  that can satisfy the values you choose for these #define's.
*  For example, if you set  SLOTS_PER_ELT to 8
*  bundleElt must contain at least 8 addressable fields.
*  Likewise, (SAMPLE_WIDTH * SAMPLES_PER_SLOT) <= number of bits in a SLOT.
*
*  The rest of the code in this package functions without alteration
*  with any valid combination of #define's;
*  with the exception of:
*  slotAtIndex.cu  and
*  macros.h
*
*  slotAtIndex.cu
*  ===============
*  If you change this file, you will likely need to modify the functions:
*
*  int slotAtIndex (bundleElt *element, unsigned int index);
*  void saveAtIndex (bundleElt* element, unsigned int index, int value);
*
*  which provide the interface to acess/modify each slot of a bundleElt.
*
*  SoftDecode.cu
*  ==============
*  This file contains macro expansions for
*  DOITALL_CN
*  and
*  DOITALL_BN
*
*  for each slot in a bundleElt.
*
*  You must add/remove occurrences of these macros to match
*  SLOTS_PER_ELT
*  and, you may have to modify the first argument to these macros: "word"
*  this is the string (not quoted) necessary to address the slot.
*  Some examples:
*  If  bundleElt is  unsigned int slot[8],  then
*  slot[0], slot[1] ...
*  if  bundleElt is  uint4
*  (a specific cuda typedef that includes 4 unsigned ints,
*  that includes (multiple) specific addressing alternatives
*  via  union), then
*  x, y, z, w  could be used  or
*  s0, s1, s2, s3
*
*/

#define SLOTS_PER_ELT 4
#define SAMPLES_PER_SLOT 1
#define SAMPLE_WIDTH 32
// #define SAMPLE_MASK  ((1 << SAMPLE_WIDTH) -1)
/*  Just for SAMPLE_WIDTH == 32 */
#define SAMPLE_MASK (~0)

#define USED_BITS (SAMPLES_PER_SLOT * SAMPLE_WIDTH)

//  PKTS_PER_BUNDLE  - the number of packets handled collectively
//  in the basic data structure used here, bundleElt,
#define PKTS_PER_BUNDLE (SLOTS_PER_ELT * SAMPLES_PER_SLOT)

struct __builtin_align__(16) localBE
{
    float s[SLOTS_PER_ELT];
};
typedef localBE bundleElt;

// typedef float4 bundleElt;

#define makeBundleElt(x) ({x,x,x,x})

static __inline__ __host__ __device__ bundleElt  make_bundleElt(float x, float y, float z, float w)
{
    bundleElt t; t.s[0] = x; t.s[1] = y; t.s[2] = z; t.s[3] = w; return t;
}
static __inline__ __host__ __device__ bundleElt  make_bundleElt(float x)
{
    bundleElt t; t.s[0] = x; t.s[1] = x; t.s[2] = x; t.s[3] = x; return t;
}

inline __host__ __device__ void operator+=(bundleElt &a, bundleElt b)
{
    a.s[0] += b.s[0];
    a.s[1] += b.s[1];
    a.s[2] += b.s[2];
    a.s[3] += b.s[3];
}
inline __host__ __device__ void operator*=(bundleElt &a, bundleElt b)
{
    a.s[0] *= b.s[0];
    a.s[1] *= b.s[1];
    a.s[2] *= b.s[2];
    a.s[3] *= b.s[3];
}

inline __host__ __device__ bundleElt operator+(bundleElt a, bundleElt b)
{
    bundleElt be;
    be.s[0] = a.s[0] + b.s[0];
    be.s[1] = a.s[1] + b.s[1];
    be.s[2] = a.s[2] + b.s[2];
    be.s[3] = a.s[3] + b.s[3];
    return be;
}
inline __host__ __device__ bundleElt operator-(bundleElt a, bundleElt b)
{
    bundleElt be;
    be.s[0] = a.s[0] - b.s[0];
    be.s[1] = a.s[1] - b.s[1];
    be.s[2] = a.s[2] - b.s[2];
    be.s[3] = a.s[3] - b.s[3];
    return be;
}
inline __host__ __device__ bundleElt operator*(bundleElt a, bundleElt b)
{
    bundleElt be;
    be.s[0] = a.s[0] * b.s[0];
    be.s[1] = a.s[1] * b.s[1];
    be.s[2] = a.s[2] * b.s[2];
    be.s[3] = a.s[3] * b.s[3];
    return be;
}
inline __host__ __device__ bundleElt operator/(bundleElt a, bundleElt b)
{
    bundleElt be;
    be.s[0] = a.s[0] / b.s[0];
    be.s[1] = a.s[1] / b.s[1];
    be.s[2] = a.s[2] / b.s[2];
    be.s[3] = a.s[3] / b.s[3];
    return be;
}
inline __host__ __device__ bundleElt operator/(bundleElt a, float b)
{
    bundleElt be;
    be.s[0] = a.s[0] / b;
    be.s[1] = a.s[1] / b;
    be.s[2] = a.s[2] / b;
    be.s[3] = a.s[3] / b;
    return be;
}

inline __device__ __host__ bundleElt clamp(bundleElt v, float a, float b)
{
    return make_bundleElt(clamp(v.s[0], a, b), clamp(v.s[1], a, b), clamp(v.s[3], a, b), clamp(v.s[3], a, b));
}

#define ONEVAL(be)  (be).s[0]
#endif
