#ifndef _BUNDLE_ELT_H_
#define _BUNDLE_ELT_H_

#include "GPUincludes.h"

/*! \file bundleElt.h
*    \brief Header file for data type to hold one element of a bundle.
*
*  A bundle is the group of packets that are decoded at the same time.
*  Each kernel must handle every packet in the bundle.
*
*  bundleElt is typically implemented as a vector of a basic type
*  (float for example).
*  One or more samples may be stored in each element of this vector
*  via packing.  This capability would only be useful if the basic type
*  is (unsigned) int; and < 32 bits are needed for the range of values.
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
*  Several common operators are overloaded for bundleElt in this include file.
*  If you modiy bundleElt, you will need to update each of these definitions
*  at the same time.
*/

#define SLOTS_PER_ELT 8
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


static __inline__ __host__ __device__ bundleElt  make_bundleElt(float x0, float x1, float x2, float x3, float x4, float x5, float x6, float x7) {
    bundleElt be;
    be.s[0] = x0; be.s[1] = x1; be.s[2] = x2; be.s[3] = x3;
    be.s[4] = x4; be.s[5] = x5; be.s[6] = x6; be.s[7] = x7;
    return be;}

static __inline__ __host__ __device__ bundleElt  make_bundleElt(float x) {
    return make_bundleElt(x,x,x,x,x,x,x,x);}

inline __host__ __device__ void operator+=(bundleElt &a, bundleElt b) {
    a.s[0] += b.s[0];
    a.s[1] += b.s[1];
    a.s[2] += b.s[2];
    a.s[3] += b.s[3];
    a.s[4] += b.s[4];
    a.s[5] += b.s[5];
    a.s[6] += b.s[6];
    a.s[7] += b.s[7];}

inline __host__ __device__ void operator*=(bundleElt &a, bundleElt b) {
    a.s[0] *= b.s[0];
    a.s[1] *= b.s[1];
    a.s[2] *= b.s[2];
    a.s[3] *= b.s[3];
    a.s[4] *= b.s[4];
    a.s[5] *= b.s[5];
    a.s[6] *= b.s[6];
    a.s[7] *= b.s[7];}

inline __host__ __device__ bundleElt operator+(bundleElt a, bundleElt b) {
    bundleElt be;
    be.s[0] = a.s[0] + b.s[0];
    be.s[1] = a.s[1] + b.s[1];
    be.s[2] = a.s[2] + b.s[2];
    be.s[3] = a.s[3] + b.s[3];
    be.s[4] = a.s[4] + b.s[4];
    be.s[5] = a.s[5] + b.s[5];
    be.s[6] = a.s[6] + b.s[6];
    be.s[7] = a.s[7] + b.s[7];
    return be;}

inline __host__ __device__ bundleElt operator-(bundleElt a, bundleElt b) {
    bundleElt be;
    be.s[0] = a.s[0] - b.s[0];
    be.s[1] = a.s[1] - b.s[1];
    be.s[2] = a.s[2] - b.s[2];
    be.s[3] = a.s[3] - b.s[3];
    be.s[4] = a.s[4] - b.s[4];
    be.s[5] = a.s[5] - b.s[5];
    be.s[6] = a.s[6] - b.s[6];
    be.s[7] = a.s[7] - b.s[7];
    return be;}

inline __host__ __device__ bundleElt operator*(bundleElt a, bundleElt b) {
    bundleElt be;
    be.s[0] = a.s[0] * b.s[0];
    be.s[1] = a.s[1] * b.s[1];
    be.s[2] = a.s[2] * b.s[2];
    be.s[3] = a.s[3] * b.s[3];
    be.s[4] = a.s[4] * b.s[4];
    be.s[5] = a.s[5] * b.s[5];
    be.s[6] = a.s[6] * b.s[6];
    be.s[7] = a.s[7] * b.s[7];
    return be;}

inline __host__ __device__ bundleElt operator/(bundleElt a, bundleElt b) {
    bundleElt be;
    be.s[0] = a.s[0] / b.s[0];
    be.s[1] = a.s[1] / b.s[1];
    be.s[2] = a.s[2] / b.s[2];
    be.s[3] = a.s[3] / b.s[3];
    be.s[4] = a.s[4] / b.s[4];
    be.s[5] = a.s[5] / b.s[5];
    be.s[6] = a.s[6] / b.s[6];
    be.s[7] = a.s[7] / b.s[7];
    return be;}

inline __host__ __device__ bundleElt operator/(bundleElt a, float b) {
    bundleElt be;
    be.s[0] = a.s[0] / b;
    be.s[1] = a.s[1] / b;
    be.s[2] = a.s[2] / b;
    be.s[3] = a.s[3] / b;
    be.s[4] = a.s[4] / b;
    be.s[5] = a.s[5] / b;
    be.s[6] = a.s[6] / b;
    be.s[7] = a.s[7] / b;
    return be;}

inline __device__ __host__ bundleElt clamp(bundleElt v, float a, float b) {
    return make_bundleElt(clamp(v.s[0], a, b), clamp(v.s[1], a, b), clamp(v.s[2], a, b), clamp(v.s[3], a, b),
                          clamp(v.s[4], a, b), clamp(v.s[5], a, b), clamp(v.s[6], a, b), clamp(v.s[7], a, b));}

inline __host__ __device__ void fprintBE(FILE *fd, bundleElt a) {
    fprintf(fd, "[%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f]",
            a.s[0],a.s[1],a.s[2],a.s[3],a.s[4],a.s[5],a.s[6],a.s[7]);
}


#define ONEVAL(be)  (be).s[0]
#endif
