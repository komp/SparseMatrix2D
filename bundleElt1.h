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

#define SLOTS_PER_ELT 1
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


static __inline__ __host__ __device__ bundleElt  make_bundleElt(float x0) {
    bundleElt be;
    be.s[0] = x0;
    return be;}

inline __host__ __device__ void operator+=(bundleElt &a, bundleElt b) {
    a.s[0] += b.s[0];}

inline __host__ __device__ void operator*=(bundleElt &a, bundleElt b) {
    a.s[0] *= b.s[0];}

inline __host__ __device__ bundleElt operator+(bundleElt a, bundleElt b) {
    bundleElt be;
    be.s[0] = a.s[0] + b.s[0];
    return be;}

inline __host__ __device__ bundleElt operator-(bundleElt a, bundleElt b) {
    bundleElt be;
    be.s[0] = a.s[0] - b.s[0];
    return be;}

inline __host__ __device__ bundleElt operator*(bundleElt a, bundleElt b) {
    bundleElt be;
    be.s[0] = a.s[0] * b.s[0];
    return be;}

inline __host__ __device__ bundleElt operator/(bundleElt a, bundleElt b) {
    bundleElt be;
    be.s[0] = a.s[0] / b.s[0];
    return be;}

inline __host__ __device__ bundleElt operator/(bundleElt a, float b) {
    bundleElt be;
    be.s[0] = a.s[0] / b;
    return be;}

inline __device__ __host__ bundleElt clamp(bundleElt v, float a, float b) {
    return make_bundleElt(clamp(v.s[0], a, b)); }

inline __host__ __device__ void fprintBE(FILE *fd, bundleElt a) {
    fprintf(fd, "[%.2f] ", a.s[0]);
}

#define ONEVAL(be)  (be).s[0]
#endif
