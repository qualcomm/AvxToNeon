/*
 * Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

 * http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.

 */

#include <stdio.h>
#ifndef AVX2NEON_H
#error Never use <avxintrin512.h> directly; include " avx2neon.h" instead.
#endif


#include <arm_neon.h>

#include <math.h>
#include <stdlib.h>
#ifdef __cplusplus
using namespace std;
#endif

#include "typedefs.h"

typedef union {
    int8x16_t vect_s8;
    int16x8_t vect_s16;
    int32x4_t vect_s32;
    int64x2_t vect_s64;
    uint8x16_t vect_u8;
    uint16x8_t vect_u16;
    uint32x4_t vect_u32;
    uint64x2_t vect_u64;
} __m128i;

typedef float32x4_t __m128;

typedef float64x2_t __m128d;

typedef enum {
    _MM_CMPINT_EQ = 0,    /* Equal */
    _MM_CMPINT_LT = 1,    /* Less than */
    _MM_CMPINT_LE = 2,    /* Less than or Equal */
    _MM_CMPINT_FALSE = 3, /* Always False */
    _MM_CMPINT_NE = 4,    /* Not Equal */
    _MM_CMPINT_NLT = 5,   /* Not Less than */
    _MM_CMPINT_NLE = 6,   /* Not Less than or Equal */
    _MM_CMPINT_TRUE = 7   /* Always True */
} _MM_CMPINT_ENUM;

ALIGN_STRUCT(16) static uint64_t g_mask_epi64[2] = {0x01, 0x02};
ALIGN_STRUCT(16) static uint32_t g_mask_epi32[4] = {0x01, 0x02, 0x04, 0x08};
ALIGN_STRUCT(16) static uint16_t g_mask_epi16[8] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80};
ALIGN_STRUCT(16) static uint8_t g_mask_epi8[16] = {
    0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80};

#define _SIDD_UBYTE_OPS 0x00  // unsigned 8-bit characters
#define _SIDD_UWORD_OPS 0x01  // unsigned 16-bit characters
#define _SIDD_SBYTE_OPS 0x02  // signed 8-bit characters
#define _SIDD_SWORD_OPS 0x03  // signed 16-bit characters

#define _SIDD_CMP_EQUAL_ANY 0x00      // compare equal any
#define _SIDD_CMP_RANGES 0x04         // compare ranges
#define _SIDD_CMP_EQUAL_EACH 0x08     // compare equal each
#define _SIDD_CMP_EQUAL_ORDERED 0x0C  // compare equal ordered

#define _SIDD_POSITIVE_POLARITY 0x00
#define _SIDD_MASKED_POSITIVE_POLARITY 0x20
#define _SIDD_NEGATIVE_POLARITY 0x10         // negate results
#define _SIDD_MASKED_NEGATIVE_POLARITY 0x30  // negate results only before end of string

#define _SIDD_LEAST_SIGNIFICANT 0x00  // index only: return last significant bit
#define _SIDD_MOST_SIGNIFICANT 0x40   // index only: return most significant bit

#define _SIDD_BIT_MASK 0x00   // mask only: return bit mask
#define _SIDD_UNIT_MASK 0x40  // mask only: return byte/word mask

#define PCMPSTR_EQ_16x8(a, b, mtx)                                                                          \
    {                                                                                                       \
        mtx[0].vect_u16 = vceqq_u16(vdupq_n_u16(vgetq_lane_u16(b.vect_u16, 0)), a.vect_u16);                \
        mtx[1].vect_u16 = vceqq_u16(vdupq_n_u16(vgetq_lane_u16(b.vect_u16, 1)), a.vect_u16);                \
        mtx[2].vect_u16 = vceqq_u16(vdupq_n_u16(vgetq_lane_u16(b.vect_u16, 2)), a.vect_u16);                \
        mtx[3].vect_u16 = vceqq_u16(vdupq_n_u16(vgetq_lane_u16(b.vect_u16, 3)), a.vect_u16);                \
        mtx[4].vect_u16 = vceqq_u16(vdupq_n_u16(vgetq_lane_u16(b.vect_u16, 4)), a.vect_u16);                \
        mtx[5].vect_u16 = vceqq_u16(vdupq_n_u16(vgetq_lane_u16(b.vect_u16, 5)), a.vect_u16);                \
        mtx[6].vect_u16 = vceqq_u16(vdupq_n_u16(vgetq_lane_u16(b.vect_u16, 6)), a.vect_u16);                \
        mtx[7].vect_u16 = vceqq_u16(vdupq_n_u16(vgetq_lane_u16(b.vect_u16, 7)), a.vect_u16);                \
    }

#define PCMPSTR_EQ_8x16(a, b, mtx)                                                                          \
    {                                                                                                       \
        mtx[0].vect_u8 = vceqq_u8(vdupq_n_u8(vgetq_lane_u8(b.vect_u8, 0)), a.vect_u8);                      \
        mtx[1].vect_u8 = vceqq_u8(vdupq_n_u8(vgetq_lane_u8(b.vect_u8, 1)), a.vect_u8);                      \
        mtx[2].vect_u8 = vceqq_u8(vdupq_n_u8(vgetq_lane_u8(b.vect_u8, 2)), a.vect_u8);                      \
        mtx[3].vect_u8 = vceqq_u8(vdupq_n_u8(vgetq_lane_u8(b.vect_u8, 3)), a.vect_u8);                      \
        mtx[4].vect_u8 = vceqq_u8(vdupq_n_u8(vgetq_lane_u8(b.vect_u8, 4)), a.vect_u8);                      \
        mtx[5].vect_u8 = vceqq_u8(vdupq_n_u8(vgetq_lane_u8(b.vect_u8, 5)), a.vect_u8);                      \
        mtx[6].vect_u8 = vceqq_u8(vdupq_n_u8(vgetq_lane_u8(b.vect_u8, 6)), a.vect_u8);                      \
        mtx[7].vect_u8 = vceqq_u8(vdupq_n_u8(vgetq_lane_u8(b.vect_u8, 7)), a.vect_u8);                      \
        mtx[8].vect_u8 = vceqq_u8(vdupq_n_u8(vgetq_lane_u8(b.vect_u8, 8)), a.vect_u8);                      \
        mtx[9].vect_u8 = vceqq_u8(vdupq_n_u8(vgetq_lane_u8(b.vect_u8, 9)), a.vect_u8);                      \
        mtx[10].vect_u8 = vceqq_u8(vdupq_n_u8(vgetq_lane_u8(b.vect_u8, 10)), a.vect_u8);                    \
        mtx[11].vect_u8 = vceqq_u8(vdupq_n_u8(vgetq_lane_u8(b.vect_u8, 11)), a.vect_u8);                    \
        mtx[12].vect_u8 = vceqq_u8(vdupq_n_u8(vgetq_lane_u8(b.vect_u8, 12)), a.vect_u8);                    \
        mtx[13].vect_u8 = vceqq_u8(vdupq_n_u8(vgetq_lane_u8(b.vect_u8, 13)), a.vect_u8);                    \
        mtx[14].vect_u8 = vceqq_u8(vdupq_n_u8(vgetq_lane_u8(b.vect_u8, 14)), a.vect_u8);                    \
        mtx[15].vect_u8 = vceqq_u8(vdupq_n_u8(vgetq_lane_u8(b.vect_u8, 15)), a.vect_u8);                    \
    }

#define PCMPSTR_RNG_U16x8(a, b, mtx)                                                                                \
    {                                                                                                               \
        uint16x8_t vect_b[8];                                                                                       \
        __m128i mask;                                                                                               \
        mask.vect_u32 = vdupq_n_u32(0xffff);                                                                        \
        vect_b[0] = vdupq_n_u16(vgetq_lane_u16(b.vect_u16, 0));                                                     \
        vect_b[1] = vdupq_n_u16(vgetq_lane_u16(b.vect_u16, 1));                                                     \
        vect_b[2] = vdupq_n_u16(vgetq_lane_u16(b.vect_u16, 2));                                                     \
        vect_b[3] = vdupq_n_u16(vgetq_lane_u16(b.vect_u16, 3));                                                     \
        vect_b[4] = vdupq_n_u16(vgetq_lane_u16(b.vect_u16, 4));                                                     \
        vect_b[5] = vdupq_n_u16(vgetq_lane_u16(b.vect_u16, 5));                                                     \
        vect_b[6] = vdupq_n_u16(vgetq_lane_u16(b.vect_u16, 6));                                                     \
        vect_b[7] = vdupq_n_u16(vgetq_lane_u16(b.vect_u16, 7));                                                     \
        int i;                                                                                                      \
        for (i = 0; i < 8; i++) {                                                                                   \
            mtx[i].vect_u16 = vbslq_u16(mask.vect_u16, vcgeq_u16(vect_b[i], a.vect_u16),                            \
            vcleq_u16(vect_b[i], a.vect_u16));                                                                      \
        }                                                                                                           \
    }
#define PCMPSTR_RNG_S16x8(a, b, mtx)                                                                                \
    {                                                                                                               \
        int16x8_t vect_b[8];                                                                                        \
        __m128i mask;                                                                                               \
        mask.vect_u32 = vdupq_n_u32(0xffff);                                                                        \
        vect_b[0] = vdupq_n_s16(vgetq_lane_s16(b.vect_s16, 0));                                                     \
        vect_b[1] = vdupq_n_s16(vgetq_lane_s16(b.vect_s16, 1));                                                     \
        vect_b[2] = vdupq_n_s16(vgetq_lane_s16(b.vect_s16, 2));                                                     \
        vect_b[3] = vdupq_n_s16(vgetq_lane_s16(b.vect_s16, 3));                                                     \
        vect_b[4] = vdupq_n_s16(vgetq_lane_s16(b.vect_s16, 4));                                                     \
        vect_b[5] = vdupq_n_s16(vgetq_lane_s16(b.vect_s16, 5));                                                     \
        vect_b[6] = vdupq_n_s16(vgetq_lane_s16(b.vect_s16, 6));                                                     \
        vect_b[7] = vdupq_n_s16(vgetq_lane_s16(b.vect_s16, 7));                                                     \
        int i;                                                                                                      \
        for (i = 0; i < 8; i++) {                                                                                   \
            mtx[i].vect_u16 = vbslq_u16(mask.vect_u16, vcgeq_s16(vect_b[i], a.vect_s16),                            \
            vcleq_s16(vect_b[i], a.vect_s16));                                                                      \
        }                                                                                                           \
    }

#define PCMPSTR_RNG_U8x16(a, b, mtx)                                                                                \
    {                                                                                                               \
        uint8x16_t vect_b[16];                                                                                      \
        __m128i mask;                                                                                               \
        mask.vect_u16 = vdupq_n_u16(0xff);                                                                          \
        vect_b[0] = vdupq_n_u8(vgetq_lane_u8(b.vect_u8, 0));                                                        \
        vect_b[1] = vdupq_n_u8(vgetq_lane_u8(b.vect_u8, 1));                                                        \
        vect_b[2] = vdupq_n_u8(vgetq_lane_u8(b.vect_u8, 2));                                                        \
        vect_b[3] = vdupq_n_u8(vgetq_lane_u8(b.vect_u8, 3));                                                        \
        vect_b[4] = vdupq_n_u8(vgetq_lane_u8(b.vect_u8, 4));                                                        \
        vect_b[5] = vdupq_n_u8(vgetq_lane_u8(b.vect_u8, 5));                                                        \
        vect_b[6] = vdupq_n_u8(vgetq_lane_u8(b.vect_u8, 6));                                                        \
        vect_b[7] = vdupq_n_u8(vgetq_lane_u8(b.vect_u8, 7));                                                        \
        vect_b[8] = vdupq_n_u8(vgetq_lane_u8(b.vect_u8, 8));                                                        \
        vect_b[9] = vdupq_n_u8(vgetq_lane_u8(b.vect_u8, 9));                                                        \
        vect_b[10] = vdupq_n_u8(vgetq_lane_u8(b.vect_u8, 10));                                                      \
        vect_b[11] = vdupq_n_u8(vgetq_lane_u8(b.vect_u8, 11));                                                      \
        vect_b[12] = vdupq_n_u8(vgetq_lane_u8(b.vect_u8, 12));                                                      \
        vect_b[13] = vdupq_n_u8(vgetq_lane_u8(b.vect_u8, 13));                                                      \
        vect_b[14] = vdupq_n_u8(vgetq_lane_u8(b.vect_u8, 14));                                                      \
        vect_b[15] = vdupq_n_u8(vgetq_lane_u8(b.vect_u8, 15));                                                      \
        int i;                                                                                                      \
        for (i = 0; i < 16; i++) {                                                                                  \
            mtx[i].vect_u8 = vbslq_u8(mask.vect_u8, vcgeq_u8(vect_b[i], a.vect_u8), vcleq_u8(vect_b[i], a.vect_u8));\
        }                                                                                                           \
    }

#define PCMPSTR_RNG_S8x16(a, b, mtx)                                                                                \
    {                                                                                                               \
        int8x16_t vect_b[16];                                                                                       \
        __m128i mask;                                                                                               \
        mask.vect_u16 = vdupq_n_u16(0xff);                                                                          \
        vect_b[0] = vdupq_n_s8(vgetq_lane_s8(b.vect_s8, 0));                                                        \
        vect_b[1] = vdupq_n_s8(vgetq_lane_s8(b.vect_s8, 1));                                                        \
        vect_b[2] = vdupq_n_s8(vgetq_lane_s8(b.vect_s8, 2));                                                        \
        vect_b[3] = vdupq_n_s8(vgetq_lane_s8(b.vect_s8, 3));                                                        \
        vect_b[4] = vdupq_n_s8(vgetq_lane_s8(b.vect_s8, 4));                                                        \
        vect_b[5] = vdupq_n_s8(vgetq_lane_s8(b.vect_s8, 5));                                                        \
        vect_b[6] = vdupq_n_s8(vgetq_lane_s8(b.vect_s8, 6));                                                        \
        vect_b[7] = vdupq_n_s8(vgetq_lane_s8(b.vect_s8, 7));                                                        \
        vect_b[8] = vdupq_n_s8(vgetq_lane_s8(b.vect_s8, 8));                                                        \
        vect_b[9] = vdupq_n_s8(vgetq_lane_s8(b.vect_s8, 9));                                                        \
        vect_b[10] = vdupq_n_s8(vgetq_lane_s8(b.vect_s8, 10));                                                      \
        vect_b[11] = vdupq_n_s8(vgetq_lane_s8(b.vect_s8, 11));                                                      \
        vect_b[12] = vdupq_n_s8(vgetq_lane_s8(b.vect_s8, 12));                                                      \
        vect_b[13] = vdupq_n_s8(vgetq_lane_s8(b.vect_s8, 13));                                                      \
        vect_b[14] = vdupq_n_s8(vgetq_lane_s8(b.vect_s8, 14));                                                      \
        vect_b[15] = vdupq_n_s8(vgetq_lane_s8(b.vect_s8, 15));                                                      \
        int i;                                                                                                      \
        for (i = 0; i < 16; i++) {                                                                                  \
            mtx[i].vect_u8 = vbslq_u8(mask.vect_u8, vcgeq_s8(vect_b[i], a.vect_s8), vcleq_s8(vect_b[i], a.vect_s8));\
        }                                                                                                           \
    }

#if defined(_MSC_VER) && !defined(__clang__)
#define SET64x2(res, e0, e1)            \
    res = vsetq_lane_s64((e0), res, 0); \
    res = vsetq_lane_s64((e0), res, 1);
#define SET32x4(res, e0, e1, e2, e3) \
    res = vsetq_lane_s32((e0), res, 0); \
    res = vsetq_lane_s32((e1), res, 1); \
    res = vsetq_lane_s32((e2), res, 2); \
    res = vsetq_lane_s32((e3), res, 3);


#endif

#if defined(__GNUC__) || defined(__clang__)
#define SET64x2(res, e0, e1)               \
    __asm__ __volatile__(                  \
        "mov %[r].d[0], %[x]         \n\t" \
        "mov %[r].d[1], %[y]         \n\t" \
        : [r] "=w"(res)                    \
        : [x] "r"(e0), [y] "r"(e1));
#define SET32x4(res, e0, e1, e2, e3)       \
    __asm__ __volatile__(                  \
        "mov %[r].s[0], %w[x]        \n\t" \
        "mov %[r].s[1], %w[y]        \n\t" \
        "mov %[r].s[2], %w[z]        \n\t" \
        "mov %[r].s[3], %w[k]        \n\t" \
        : [r] "=w"(res)                    \
        : [x] "r"(e0), [y] "r"(e1), [z] "r"(e2), [k] "r"(e3));

#endif

/* extract highest bit from every 32bit */ 
#define PICK_HB_32x16(res, sign)                                                                             \
    {                                                                                                        \
        res.vect_u32[0] = vshrq_n_u32(res.vect_u32[0], 31);                                                  \
        res.vect_u32[1] = vshrq_n_u32(res.vect_u32[1], 31);                                                  \
        res.vect_u32[2] = vshrq_n_u32(res.vect_u32[2], 31);                                                  \
        res.vect_u32[3] = vshrq_n_u32(res.vect_u32[3], 31);                                                  \
        res.vect_u64[0] = vsraq_n_u64(res.vect_u64[0], res.vect_u64[0], 31);                                 \
        res.vect_u64[1] = vsraq_n_u64(res.vect_u64[1], res.vect_u64[1], 31);                                 \
        res.vect_u64[2] = vsraq_n_u64(res.vect_u64[2], res.vect_u64[2], 31);                                 \
        res.vect_u64[3] = vsraq_n_u64(res.vect_u64[3], res.vect_u64[3], 31);                                 \
        *sign = (vgetq_lane_u8(res.vect_u8[0], 0) | (vgetq_lane_u8(res.vect_u8[0], 8) << 2) |                \
                (vgetq_lane_u8(res.vect_u8[1], 0) << 4) | (vgetq_lane_u8(res.vect_u8[1], 8) << 6) |          \
                (vgetq_lane_u8(res.vect_u8[2], 0) << 8) | (vgetq_lane_u8(res.vect_u8[2], 8) << 10) |         \
                (vgetq_lane_u8(res.vect_u8[3], 0) << 12) | (vgetq_lane_u8(res.vect_u8[3], 8) << 14));        \
    };

FORCE_INLINE int _mm_popcnt_u32(unsigned int a)
{
    return (int)vaddlv_u8(vcnt_u8(vcreate_u8((unsigned __int64)a)));
}

FORCE_INLINE __int64 _mm_popcnt_u64(unsigned __int64 a)
{
    return (__int64)vaddlv_u8(vcnt_u8(vcreate_u8(a)));
}

FORCE_INLINE __m128i _mm_div_epi8(__m128i a, __m128i b)
{
    __m128i res_m128i;
    int16x8_t ta[2], tb[2];
    int16x8_t tmp_lo, tmp_hi;
    int32x4_t la[4], lb[4];
    float32x4_t res[4];

    ta[0] = vmovl_s8(vget_low_s8(a.vect_s8));
    tb[0] = vmovl_s8(vget_low_s8(b.vect_s8));
    ta[1] = vmovl_s8(vget_high_s8(a.vect_s8));
    tb[1] = vmovl_s8(vget_high_s8(b.vect_s8));

    la[0] = vmovl_s16(vget_low_s16(ta[0]));
    lb[0] = vmovl_s16(vget_low_s16(tb[0]));
    la[1] = vmovl_s16(vget_high_s16(ta[0]));
    lb[1] = vmovl_s16(vget_high_s16(tb[0]));
    la[2] = vmovl_s16(vget_low_s16(ta[1]));
    lb[2] = vmovl_s16(vget_low_s16(tb[1]));
    la[3] = vmovl_s16(vget_high_s16(ta[1]));
    lb[3] = vmovl_s16(vget_high_s16(tb[1]));

    res[1] = vdivq_f32(vcvtq_f32_s32(la[1]), vcvtq_f32_s32(lb[1]));
    res[2] = vdivq_f32(vcvtq_f32_s32(la[2]), vcvtq_f32_s32(lb[2]));
    res[0] = vdivq_f32(vcvtq_f32_s32(la[0]), vcvtq_f32_s32(lb[0]));
    res[3] = vdivq_f32(vcvtq_f32_s32(la[3]), vcvtq_f32_s32(lb[3]));
    tmp_lo = vcombine_s16(vmovn_s32(vcvtq_s32_f32(res[0])), vmovn_s32(vcvtq_s32_f32(res[1])));
    tmp_hi = vcombine_s16(vmovn_s32(vcvtq_s32_f32(res[2])), vmovn_s32(vcvtq_s32_f32(res[3])));
    res_m128i.vect_s8 = vcombine_s8(vmovn_s16(tmp_lo), vmovn_s16(tmp_hi));

    return res_m128i;
}

FORCE_INLINE __m128i _mm_div_epi16(__m128i a, __m128i b)
{
    __m128i res_m128i;
    float32x4_t fa[2], fb[2];
    float32x4_t res_lo, res_hi;
    fa[0] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(a.vect_s16)));
    fb[0] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(b.vect_s16)));    
    fa[1] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(a.vect_s16)));
    fb[1] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(b.vect_s16)));
    res_lo = vdivq_f32(fa[0], fb[0]);
    res_hi = vdivq_f32(fa[1], fb[1]);
    res_m128i.vect_s16 = vcombine_s16(vmovn_s32(vcvtq_s32_f32(res_lo)), vmovn_s32(vcvtq_s32_f32(res_hi)));
    return res_m128i;
}

FORCE_INLINE __m128i _mm_div_epu8(__m128i a, __m128i b)
{
    __m128i res_m128i;
    uint16x8_t ta[2], tb[2];
    uint16x8_t tmp_lo, tmp_hi;
    uint32x4_t la[4], lb[4];
    float32x4_t res[4];

    ta[0] = vmovl_u8(vget_low_u8(a.vect_u8));
    tb[0] = vmovl_u8(vget_low_u8(b.vect_u8));
    ta[1] = vmovl_u8(vget_high_u8(a.vect_u8));
    tb[1] = vmovl_u8(vget_high_u8(b.vect_u8));

    la[0] = vmovl_u16(vget_low_u16(ta[0]));
    lb[0] = vmovl_u16(vget_low_u16(tb[0]));
    la[1] = vmovl_u16(vget_high_u16(ta[0]));
    lb[1] = vmovl_u16(vget_high_u16(tb[0]));
    la[2] = vmovl_u16(vget_low_u16(ta[1]));
    lb[2] = vmovl_u16(vget_low_u16(tb[1]));
    la[3] = vmovl_u16(vget_high_u16(ta[1]));
    lb[3] = vmovl_u16(vget_high_u16(tb[1]));

    res[1] = vdivq_f32(vcvtq_f32_u32(la[1]), vcvtq_f32_u32(lb[1]));
    res[2] = vdivq_f32(vcvtq_f32_u32(la[2]), vcvtq_f32_u32(lb[2]));
    res[0] = vdivq_f32(vcvtq_f32_u32(la[0]), vcvtq_f32_u32(lb[0]));
    res[3] = vdivq_f32(vcvtq_f32_u32(la[3]), vcvtq_f32_u32(lb[3]));
    tmp_lo = vcombine_u16(vmovn_u32(vcvtq_u32_f32(res[0])), vmovn_u32(vcvtq_u32_f32(res[1])));
    tmp_hi = vcombine_u16(vmovn_u32(vcvtq_u32_f32(res[2])), vmovn_u32(vcvtq_u32_f32(res[3])));
    res_m128i.vect_u8 = vcombine_u8(vmovn_u16(tmp_lo), vmovn_u16(tmp_hi));

    return res_m128i;
}

FORCE_INLINE __m128i _mm_div_epu16(__m128i a, __m128i b)
{
    __m128i res_m128i;
    float32x4_t fa[2], fb[2];
    float32x4_t res_lo, res_hi;
    fa[0] = vcvtq_f32_u32(vmovl_u16(vget_low_u16(a.vect_u16)));
    fb[0] = vcvtq_f32_u32(vmovl_u16(vget_low_u16(b.vect_u16)));    
    fa[1] = vcvtq_f32_u32(vmovl_u16(vget_high_u16(a.vect_u16)));
    fb[1] = vcvtq_f32_u32(vmovl_u16(vget_high_u16(b.vect_u16)));
    res_lo = vdivq_f32(fa[0], fb[0]);
    res_hi = vdivq_f32(fa[1], fb[1]);
    res_m128i.vect_u16 = vcombine_u16(vmovn_u32(vcvtq_u32_f32(res_lo)), vmovn_u32(vcvtq_u32_f32(res_hi)));
    return res_m128i;
}

#define MM_SLL_VECT_S64_SET(res, a, mc) \
    res.vect_s64 = vshlq_n_s64(a.vect_s64, mc);
FORCE_INLINE __m128i _mm_sll_epi64(__m128i a, __m128i count)
{
    long long c = GET_LANE_S64_FROM128(count, 0);
    __m128i result_m128i;
    if (likely(c >= 0 && c < 64)) {
        switch (c)
        {
        case 0:
            MM_SLL_VECT_S64_SET(result_m128i, a, 0);
            break;
        case 1:
            MM_SLL_VECT_S64_SET(result_m128i, a, 1);
            break;
        case 2:
            MM_SLL_VECT_S64_SET(result_m128i, a, 2);
            break;
        case 3:
            MM_SLL_VECT_S64_SET(result_m128i, a, 3);
            break;
        case 4:
            MM_SLL_VECT_S64_SET(result_m128i, a, 4);
            break;
        case 5:
            MM_SLL_VECT_S64_SET(result_m128i, a, 5);
            break;
        case 6:
            MM_SLL_VECT_S64_SET(result_m128i, a, 6);
            break;
        case 7:
            MM_SLL_VECT_S64_SET(result_m128i, a, 7);
            break;
        case 8:
            MM_SLL_VECT_S64_SET(result_m128i, a, 8);
            break;
        case 9:
            MM_SLL_VECT_S64_SET(result_m128i, a, 9);
            break;
        case 10:
            MM_SLL_VECT_S64_SET(result_m128i, a, 10);
            break;
        case 11:
            MM_SLL_VECT_S64_SET(result_m128i, a, 11);
            break;
        case 12:
            MM_SLL_VECT_S64_SET(result_m128i, a, 12);
            break;
        case 13:
            MM_SLL_VECT_S64_SET(result_m128i, a, 13);
            break;
        case 14:
            MM_SLL_VECT_S64_SET(result_m128i, a, 14);
            break;
        case 15:
            MM_SLL_VECT_S64_SET(result_m128i, a, 15);
            break;
        case 16:
            MM_SLL_VECT_S64_SET(result_m128i, a, 16);
            break;
        case 17:
            MM_SLL_VECT_S64_SET(result_m128i, a, 17);
            break;
        case 18:
            MM_SLL_VECT_S64_SET(result_m128i, a, 18);
            break;
        case 19:
            MM_SLL_VECT_S64_SET(result_m128i, a, 19);
            break;
        case 20:
            MM_SLL_VECT_S64_SET(result_m128i, a, 20);
            break;
        case 21:
            MM_SLL_VECT_S64_SET(result_m128i, a, 21);
            break;
        case 22:
            MM_SLL_VECT_S64_SET(result_m128i, a, 22);
            break;
        case 23:
            MM_SLL_VECT_S64_SET(result_m128i, a, 23);
            break;
        case 24:
            MM_SLL_VECT_S64_SET(result_m128i, a, 24);
            break;
        case 25:
            MM_SLL_VECT_S64_SET(result_m128i, a, 25);
            break;
        case 26:
            MM_SLL_VECT_S64_SET(result_m128i, a, 26);
            break;
        case 27:
            MM_SLL_VECT_S64_SET(result_m128i, a, 27);
            break;
        case 28:
            MM_SLL_VECT_S64_SET(result_m128i, a, 28);
            break;
        case 29:
            MM_SLL_VECT_S64_SET(result_m128i, a, 29);
            break;
        case 30:
            MM_SLL_VECT_S64_SET(result_m128i, a, 30);
            break;
        case 31:
            MM_SLL_VECT_S64_SET(result_m128i, a, 31);
            break;
        case 32:
            MM_SLL_VECT_S64_SET(result_m128i, a, 32);
            break;
        case 33:
            MM_SLL_VECT_S64_SET(result_m128i, a, 33);
            break;
        case 34:
            MM_SLL_VECT_S64_SET(result_m128i, a, 34);
            break;
        case 35:
            MM_SLL_VECT_S64_SET(result_m128i, a, 35);
            break;
        case 36:
            MM_SLL_VECT_S64_SET(result_m128i, a, 36);
            break;
        case 37:
            MM_SLL_VECT_S64_SET(result_m128i, a, 37);
            break;
        case 38:
            MM_SLL_VECT_S64_SET(result_m128i, a, 38);
            break;
        case 39:
            MM_SLL_VECT_S64_SET(result_m128i, a, 39);
            break;
        case 40:
            MM_SLL_VECT_S64_SET(result_m128i, a, 40);
            break;
        case 41:
            MM_SLL_VECT_S64_SET(result_m128i, a, 41);
            break;
        case 42:
            MM_SLL_VECT_S64_SET(result_m128i, a, 42);
            break;
        case 43:
            MM_SLL_VECT_S64_SET(result_m128i, a, 43);
            break;
        case 44:
            MM_SLL_VECT_S64_SET(result_m128i, a, 44);
            break;
        case 45:
            MM_SLL_VECT_S64_SET(result_m128i, a, 45);
            break;
        case 46:
            MM_SLL_VECT_S64_SET(result_m128i, a, 46);
            break;
        case 47:
            MM_SLL_VECT_S64_SET(result_m128i, a, 47);
            break;
        case 48:
            MM_SLL_VECT_S64_SET(result_m128i, a, 48);
            break;
        case 49:
            MM_SLL_VECT_S64_SET(result_m128i, a, 49);
            break;
        case 50:
            MM_SLL_VECT_S64_SET(result_m128i, a, 50);
            break;
        case 51:
            MM_SLL_VECT_S64_SET(result_m128i, a, 51);
            break;
        case 52:
            MM_SLL_VECT_S64_SET(result_m128i, a, 52);
            break;
        case 53:
            MM_SLL_VECT_S64_SET(result_m128i, a, 53);
            break;
        case 54:
            MM_SLL_VECT_S64_SET(result_m128i, a, 54);
            break;
        case 55:
            MM_SLL_VECT_S64_SET(result_m128i, a, 55);
            break;
        case 56:
            MM_SLL_VECT_S64_SET(result_m128i, a, 56);
            break;
        case 57:
            MM_SLL_VECT_S64_SET(result_m128i, a, 57);
            break;
        case 58:
            MM_SLL_VECT_S64_SET(result_m128i, a, 58);
            break;
        case 59:
            MM_SLL_VECT_S64_SET(result_m128i, a, 59);
            break;
        case 60:
            MM_SLL_VECT_S64_SET(result_m128i, a, 60);
            break;
        case 61:
            MM_SLL_VECT_S64_SET(result_m128i, a, 61);
            break;
        case 62:
            MM_SLL_VECT_S64_SET(result_m128i, a, 62);
            break;
        case 63:
            MM_SLL_VECT_S64_SET(result_m128i, a, 63);
            break;
        }
    }
    else {
        result_m128i.vect_s64 = vdupq_n_s64(0);
    }
    return result_m128i;
}

FORCE_INLINE __m128i _mm_cmpeq_epi64(__m128i a, __m128i b)
{
    __m128i result_m128i;
    result_m128i.vect_u64 = vceqq_s64(a.vect_s64, b.vect_s64);
    return result_m128i;
}

// set b[0] to a[0]
FORCE_INLINE __m128 _mm_move_ss (__m128 a, __m128 b)
{
#if defined(_MSC_VER) && !defined(__clang__)
    return vsetq_lane_f32(vgetq_lane_f32(b, 0), a, 0);
#endif
#if defined(__GNUC__) || defined(__clang__)
    __asm__ __volatile__("mov %0.s[0], %1.s[0]        \n\t" : "+w"(a) : "w"(b));
    return a;
#endif
}

FORCE_INLINE __m128d _mm_move_sd(__m128d a, __m128d b)
{
#if defined(_MSC_VER) && !defined(__clang__)
    return vsetq_lane_f64(vgetq_lane_f64(b, 0), a, 0);
#endif
#if defined(__GNUC__) || defined(__clang__)
    __asm__ __volatile__("mov %0.d[0], %1.d[0]        \n\t" : "+w"(a) : "w"(b));
#endif
    return a;
}

FORCE_INLINE int _mm_testz_si128(__m128i a, __m128i b)
{
    __m128i res_m128i;
    res_m128i.vect_s64 = vandq_s64(a.vect_s64, b.vect_s64);
    return !(vgetq_lane_s64(res_m128i.vect_s64, 0) | vgetq_lane_s64(res_m128i.vect_s64, 1));
}

FORCE_INLINE int _mm_extract_epi32(__m128i a, const int imm8)
{
    assert(imm8 >= 0 && imm8 <= 3);
    switch (imm8) {
        case 0:
            return GET_LANE_S32_FROM128(a, 0);
        case 1:
            return GET_LANE_S32_FROM128(a, 1);
        case 2:
            return GET_LANE_S32_FROM128(a, 2);
        case 3:
            return GET_LANE_S32_FROM128(a, 3);
        default:
            // this will never access
            return GET_LANE_S32_FROM128(a, 0);
    }
}

FORCE_INLINE int _mm_extract_ps (__m128 a, const int imm8)
{
    assert(imm8 >= 0 && imm8 <= 3);
#if defined(_MSC_VER) && !defined(__clang__)
    int32x4_t val = vreinterpretq_s32_f32(a);
    switch (imm8) {
        case 0:
            return vgetq_lane_s32(val, 0);
        case 1:
            return vgetq_lane_s32(val, 1);
        case 2:
            return vgetq_lane_s32(val, 2);
        case 3:
            return vgetq_lane_s32(val, 3);
        default:
            // this will never access
            return vgetq_lane_s32(val, 0);
    }
#endif
#if defined(__GNUC__) || defined(__clang__)
    return vreinterpretq_s32_f32(a)[imm8];
#endif
}

FORCE_INLINE __int64 _mm_extract_epi64(__m128i a, const int imm8)
{
    assert(imm8 >= 0 && imm8 <= 1);
#if defined(_MSC_VER) && !defined(__clang__)
    switch (imm8) {
        case 0:
            return GET_LANE_S64_FROM128(a, 0);
        case 1:
            return GET_LANE_S64_FROM128(a, 1);
        default:
            // this will never access
            return GET_LANE_S64_FROM128(a, 0);
    }
#endif
#if defined(__GNUC__) || defined(__clang__)
    return a.vect_s64[imm8];
#endif
}

FORCE_INLINE unsigned int _mm_crc32_u8(unsigned int crc, unsigned char v)
{
#if defined(_MSC_VER) && !defined(__clang__)
    return __crc32cb(crc, v);
#endif
#if defined(__GNUC__) || defined(__clang__)
    __asm__ __volatile__("crc32cb %w[c], %w[c], %w[v]\n\t" : [c] "+r"(crc) : [v] "r"(v));
    return crc;
#endif
}

FORCE_INLINE unsigned int _mm_crc32_u16(unsigned int crc, unsigned short v)
{
#if defined(_MSC_VER) && !defined(__clang__)
    return __crc32ch(crc, v);
#endif
#if defined(__GNUC__) || defined(__clang__)
    __asm__ __volatile__("crc32ch %w[c], %w[c], %w[v]\n\t" : [c] "+r"(crc) : [v] "r"(v));
    return crc;
#endif
}

FORCE_INLINE unsigned int _mm_crc32_u32(unsigned int crc, unsigned int v)
{
#if defined(_MSC_VER) && !defined(__clang__)
    return __crc32cw(crc, v);
#endif
#if defined(__GNUC__) || defined(__clang__)
    __asm__ __volatile__("crc32cw %w[c], %w[c], %w[v]\n\t" : [c] "+r"(crc) : [v] "r"(v));
    return crc;
#endif
}

FORCE_INLINE unsigned __int64 _mm_crc32_u64(unsigned __int64 crc, unsigned __int64 v)
{
#if defined(_MSC_VER) && !defined(__clang__)
    return __crc32cd(crc, v);
#endif
#if defined(__GNUC__) || defined(__clang__)
    __asm__ __volatile__("crc32cx %w[c], %w[c], %x[v]\n\t" : [c] "+r"(crc) : [v] "r"(v));
    return crc;
#endif
}

FORCE_INLINE __m128d _mm_set_pd(double e1, double e0)
{
    __m128d res_m128d;
    res_m128d = vdupq_n_f64(0);
    //SET64x2(res_m128d, e0, e1);
    res_m128d = vsetq_lane_f64(e0, res_m128d, 0);
    res_m128d = vsetq_lane_f64(e1, res_m128d, 1);
    return res_m128d;
}

FORCE_INLINE __m128i _mm_set1_epi64x(int64_t a)
{
    __m128i ret;
    ret.vect_s64 = vdupq_n_s64(a);
    return ret;
}

FORCE_INLINE __m128d _mm_set1_pd(double a)
{
    return vdupq_n_f64(a);
}

FORCE_INLINE __m128i _mm_set_epi32(int e3, int e2, int e1, int e0)
{
    __m128i res_m128i;
    res_m128i.vect_s32 = vdupq_n_s32(0);
    SET32x4(res_m128i.vect_s32, e0, e1, e2, e3);
    return res_m128i;
}

FORCE_INLINE __m128i _mm_set_epi64x(int64_t e1, int64_t e0)
{
    __m128i res_m128i;
    res_m128i.vect_s64 = vdupq_n_s64(0);
    //SET64x2(res_m128i.vect_s64, e0, e1);
    res_m128i.vect_s64 = vsetq_lane_s64(e0, res_m128i.vect_s64, 0);
    res_m128i.vect_s64 = vsetq_lane_s64(e1, res_m128i.vect_s64, 1);
    return res_m128i;
}

FORCE_INLINE __m128 _mm_set_ps(float e3, float e2, float e1, float e0)
{
    __m128 res_m128;
    SET32x4(res_m128, e0, e1, e2, e3);
    return res_m128;
}

static int aggregate_equal_any_8x16(int la, int lb, __m128i mtx[16])
{
    int res = 0;
    int j;
    int m = (1 << la) - 1;
    uint8x8_t vect_mask = vld1_u8(g_mask_epi8);
    uint8x8_t t_lo = vtst_u8(vdup_n_u8(m & 0xff), vect_mask);
    uint8x8_t t_hi = vtst_u8(vdup_n_u8(m >> 8), vect_mask);
    uint8x16_t vect = vcombine_u8(t_lo, t_hi);
    for (j = 0; j < lb; j++) {
        mtx[j].vect_u8 = vandq_u8(vect, mtx[j].vect_u8);
        mtx[j].vect_u8 = vshrq_n_u8(mtx[j].vect_u8, 7);
        int tmp = vaddvq_u8(mtx[j].vect_u8) ? 1 : 0;
        res |= ( tmp << j);
    }
    return res;
}

static int aggregate_equal_any_16x8(int la, int lb, __m128i mtx[16])
{
    int res = 0;
    int j;
    int m = (1 << la) - 1;
    uint16x8_t vect = vtstq_u16(vdupq_n_u16(m), vld1q_u16(g_mask_epi16));
    for (j = 0; j < lb; j++) {
        mtx[j].vect_u16 = vandq_u16(vect, mtx[j].vect_u16);
        mtx[j].vect_u16 = vshrq_n_u16(mtx[j].vect_u16, 15);
        int tmp = vaddvq_u16(mtx[j].vect_u16) ? 1 : 0;
        res |= (tmp << j);
    }
    return res;
}

static int cal_res_byte_equal_any(__m128i a, int la, __m128i b, int lb)
{
    __m128i mtx[16];
    PCMPSTR_EQ_8x16(a, b, mtx);
    return aggregate_equal_any_8x16(la, lb, mtx);
}

static int cal_res_word_equal_any(__m128i a, int la, __m128i b, int lb)
{
    __m128i mtx[16];
    PCMPSTR_EQ_16x8(a, b, mtx);
    return aggregate_equal_any_16x8(la, lb, mtx);
}

static int aggregate_ranges_16x8(int la, int lb, __m128i mtx[16])
{
    int res = 0;
    int j;
    int m = (1 << la) - 1;
    uint16x8_t vect = vtstq_u16(vdupq_n_u16(m), vld1q_u16(g_mask_epi16));
    for (j = 0; j < lb; j++) {
        mtx[j].vect_u16 = vandq_u16(vect, mtx[j].vect_u16);
        mtx[j].vect_u16 = vshrq_n_u16(mtx[j].vect_u16, 15);
        __m128i tmp;
        tmp.vect_u32 = vshrq_n_u32(mtx[j].vect_u32, 16);
        uint32x4_t vect_res = vandq_u32(mtx[j].vect_u32, tmp.vect_u32);
        int t = vaddvq_u32(vect_res) ? 1 : 0;
        res |= (t << j);
    }
    return res;
}

static int aggregate_ranges_8x16(int la, int lb, __m128i mtx[16])
{
    int res = 0;
    int j;
    int m = (1 << la) - 1;
    uint8x8_t vect_mask = vld1_u8(g_mask_epi8);
    uint8x8_t t_lo = vtst_u8(vdup_n_u8(m & 0xff), vect_mask);
    uint8x8_t t_hi = vtst_u8(vdup_n_u8(m >> 8), vect_mask);
    uint8x16_t vect = vcombine_u8(t_lo, t_hi);
    for (j = 0; j < lb; j++) {
        mtx[j].vect_u8 = vandq_u8(vect, mtx[j].vect_u8);
        mtx[j].vect_u8 = vshrq_n_u8(mtx[j].vect_u8, 7);
        __m128i tmp;
        tmp.vect_u16 = vshrq_n_u16(mtx[j].vect_u16, 8);
        uint16x8_t vect_res = vandq_u16(mtx[j].vect_u16, tmp.vect_u16);
        int t = vaddvq_u16(vect_res) ? 1 : 0;
        res |= (t << j);
    }
    return res;
}

static int cal_res_ubyte_ranges(__m128i a, int la, __m128i b, int lb)
{
    __m128i mtx[16];
    PCMPSTR_RNG_U8x16(a, b, mtx);
    return aggregate_ranges_8x16(la, lb, mtx);
}

static int cal_res_sbyte_ranges(__m128i a, int la, __m128i b, int lb)
{
    __m128i mtx[16];
    PCMPSTR_RNG_S8x16(a, b, mtx);
    return aggregate_ranges_8x16(la, lb, mtx);
}

static int cal_res_uword_ranges(__m128i a, int la, __m128i b, int lb)
{
    __m128i mtx[16];
    PCMPSTR_RNG_U16x8(a, b, mtx);
    return aggregate_ranges_16x8(la, lb, mtx);
}

static int cal_res_sword_ranges(__m128i a, int la, __m128i b, int lb)
{
    __m128i mtx[16];
    PCMPSTR_RNG_S16x8(a, b, mtx);
    return aggregate_ranges_16x8(la, lb, mtx);
}

static int cal_res_byte_equal_each(__m128i a, int la, __m128i b, int lb)
{
    uint8x16_t mtx = vceqq_u8(a.vect_u8, b.vect_u8);
    int m0 = (la < lb) ? 0 : ((1 << la) - (1 << lb));
    int m1 = 0x10000 - (1 << la);
    int tb = 0x10000 - (1 << lb);
    uint8x8_t vect_mask, vect0_lo, vect0_hi, vect1_lo, vect1_hi;
    uint8x8_t tmp_lo, tmp_hi, res_lo, res_hi;
    vect_mask = vld1_u8(g_mask_epi8);
    vect0_lo = vtst_u8(vdup_n_u8(m0), vect_mask);
    vect0_hi = vtst_u8(vdup_n_u8(m0 >> 8), vect_mask);
    vect1_lo = vtst_u8(vdup_n_u8(m1), vect_mask);
    vect1_hi = vtst_u8(vdup_n_u8(m1 >> 8), vect_mask);
    tmp_lo = vtst_u8(vdup_n_u8(tb), vect_mask);
    tmp_hi = vtst_u8(vdup_n_u8(tb >> 8), vect_mask);

    res_lo = vbsl_u8(vect0_lo, vdup_n_u8(0), vget_low_u8(mtx));
    res_hi = vbsl_u8(vect0_hi, vdup_n_u8(0), vget_high_u8(mtx));
    res_lo = vbsl_u8(vect1_lo, tmp_lo, res_lo);
    res_hi = vbsl_u8(vect1_hi, tmp_hi, res_hi);
    res_lo = vand_u8(res_lo, vect_mask);
    res_hi = vand_u8(res_hi, vect_mask);

    int res = vaddv_u8(res_lo) + (vaddv_u8(res_hi) << 8);
    return res;
}

static int cal_res_word_equal_each(__m128i a, int la, __m128i b, int lb)
{
    uint16x8_t mtx = vceqq_u16(a.vect_u16, b.vect_u16);
    int m0 = (la < lb) ? 0 : ((1 << la) - (1 << lb));
    int m1 = 0x100 - (1 << la);
    int tb = 0x100 - (1 << lb);
    uint16x8_t vect_mask = vld1q_u16(g_mask_epi16);
    uint16x8_t vect0 = vtstq_u16(vdupq_n_u16(m0), vect_mask);
    uint16x8_t vect1 = vtstq_u16(vdupq_n_u16(m1), vect_mask);
    uint16x8_t tmp = vtstq_u16(vdupq_n_u16(tb), vect_mask);
    mtx = vbslq_u16(vect0, vdupq_n_u16(0), mtx);
    mtx = vbslq_u16(vect1, tmp, mtx);
    mtx = vandq_u16(mtx, vect_mask);
    return vaddvq_u16(mtx);
}

static int aggregate_equal_ordered_8x16(int bound, int la, int lb, __m128i mtx[16])
{
    int res = 0;
    int i, j, k;
    int m1 = 0x10000 - (1 << la);
    uint8x8_t vect_mask = vld1_u8(g_mask_epi8);
    uint8x16_t vect1 = vcombine_u8(vtst_u8(vdup_n_u8(m1), vect_mask), vtst_u8(vdup_n_u8(m1 >> 8), vect_mask));
    uint8x16_t vect_minusone = vdupq_n_u8(-1);
    uint8x16_t vect_zero = vdupq_n_u8(0);
    for (j = 0; j < lb; j++) {
        mtx[j].vect_u8 = vbslq_u8(vect1, vect_minusone, mtx[j].vect_u8);
    }
    for (j = lb; j < bound; j++) {
        mtx[j].vect_u8 = vbslq_u8(vect1, vect_minusone, vect_zero);
    }
    unsigned char *ptr = (unsigned char*)mtx;
    for (i = 0; i < bound; i++) {
        int val = 1;
        for (j = 0, k = i; j < bound - i && k < bound; j++, k++) {
            val &= ptr[k * bound + j];
        }
        res = (val << i) + res;
    }
    return res;
}

static int aggregate_equal_ordered_16x8(int bound, int la, int lb, __m128i mtx[16])
{
    int res = 0;
    int i, j, k;
    int m1 = 0x100 - (1 << la);
    uint16x8_t vect_mask = vld1q_u16(g_mask_epi16);
    uint16x8_t vect1 = vtstq_u16(vdupq_n_u16(m1), vect_mask);
    uint16x8_t vect_minusone = vdupq_n_u16(-1);
    uint16x8_t vect_zero = vdupq_n_u16(0);
    for (j = 0; j < lb; j++) {
        mtx[j].vect_u16 = vbslq_u16(vect1, vect_minusone, mtx[j].vect_u16);
    }
    for (j = lb; j < bound; j++) {
        mtx[j].vect_u16 = vbslq_u16(vect1, vect_minusone, vect_zero);
    }
    unsigned short *ptr = (unsigned short*)mtx;
    for (i = 0; i < bound; i++) {
        int val = 1;
        for (j = 0, k = i; j < bound - i && k < bound; j++, k++) {
            val &= ptr[k * bound + j];
        }
        res = (val << i) + res;
    }
    return res;
}

static int cal_res_byte_equal_ordered(__m128i a, int la, __m128i b, int lb)
{
    __m128i mtx[16];
    PCMPSTR_EQ_8x16(a, b, mtx);
    return aggregate_equal_ordered_8x16(16, la, lb, mtx);
}

static int cal_res_word_equal_ordered(__m128i a, int la, __m128i b, int lb)
{
    __m128i mtx[16];
    PCMPSTR_EQ_16x8(a, b, mtx);
    return aggregate_equal_ordered_16x8(8, la, lb, mtx);
}

typedef enum {
    CMP_UBYTE_EQUAL_ANY,
    CMP_UWORD_EQUAL_ANY,
    CMP_SBYTE_EQUAL_ANY,
    CMP_SWORD_EQUAL_ANY,
    CMP_UBYTE_RANGES,
    CMP_UWORD_RANGES,
    CMP_SBYTE_RANGES,
    CMP_SWORD_RANGES,
    CMP_UBYTE_EQUAL_EACH,
    CMP_UWORD_EQUAL_EACH,
    CMP_SBYTE_EQUAL_EACH,
    CMP_SWORD_EQUAL_EACH,
    CMP_UBYTE_EQUAL_ORDERED,
    CMP_UWORD_EQUAL_ORDERED,
    CMP_SBYTE_EQUAL_ORDERED,
    CMP_SWORD_EQUAL_ORDERED
} _MM_CMPESTR_ENUM;
typedef int (*CMPESTR)(__m128i a, int la, __m128i b, int lb);
typedef struct {
    _MM_CMPESTR_ENUM cmpintEnum;
    CMPESTR cmpFun;
} CmpestrFuncList;
static CmpestrFuncList g_CmpestrFuncList[] = {{CMP_UBYTE_EQUAL_ANY, cal_res_byte_equal_any},
    {CMP_UWORD_EQUAL_ANY, cal_res_word_equal_any},
    {CMP_SBYTE_EQUAL_ANY, cal_res_byte_equal_any},
    {CMP_SWORD_EQUAL_ANY, cal_res_word_equal_any},
    {CMP_UBYTE_RANGES, cal_res_ubyte_ranges},
    {CMP_UWORD_RANGES, cal_res_uword_ranges},
    {CMP_SBYTE_RANGES, cal_res_sbyte_ranges},
    {CMP_SWORD_RANGES, cal_res_sword_ranges},
    {CMP_UBYTE_EQUAL_EACH, cal_res_byte_equal_each},
    {CMP_UWORD_EQUAL_EACH, cal_res_word_equal_each},
    {CMP_SBYTE_EQUAL_EACH, cal_res_byte_equal_each},
    {CMP_SWORD_EQUAL_EACH, cal_res_word_equal_each},
    {CMP_UBYTE_EQUAL_ORDERED, cal_res_byte_equal_ordered},
    {CMP_UWORD_EQUAL_ORDERED, cal_res_word_equal_ordered},
    {CMP_SBYTE_EQUAL_ORDERED, cal_res_byte_equal_ordered},
    {CMP_SWORD_EQUAL_ORDERED, cal_res_word_equal_ordered}};

FORCE_INLINE int neg_fun(int res, int lb, int imm8, int bound)
{
    int m;
    switch (imm8 & 0x30) {
        case _SIDD_NEGATIVE_POLARITY:
            res ^= 0xffffffff;
            break;
        case _SIDD_MASKED_NEGATIVE_POLARITY:
            m = (1 << lb) - 1;
            res ^= m;
            break;
        default:
            break;
    }

    return res & ((bound == 8) ? 0xFF : 0xFFFF);
}

FORCE_INLINE int _mm_cmpestri(__m128i a, int la, __m128i b, int lb, const int imm8)
{
    int bound = (imm8 & 0x01) ? 8 : 16;
#if defined(_MSC_VER)
    // msvc and msvc-like compiler does not support 'asr' in inline asm for ARM, so use C code
    if (la < 0) la = 0;
    if (lb < 0) lb = 0;
    if (la > bound) la = bound;
    if (lb > bound) lb = bound;
#else
    __asm__ __volatile__ (
        "eor w0, %w[a], %w[a], asr31          \n\t"
        "sub %w[a], w0, %w[a], asr31          \n\t"
        "eor w1, %w[b], %w[b], asr31          \n\t"
        "sub %w[b], w1, %w[b], asr31          \n\t"
        "cmp %w[a], %w[bd]                  \n\t"
        "csel %w[a], %w[bd], %w[a], gt      \n\t"
        "cmp %w[b], %w[bd]                  \n\t"
        "csel %w[b], %w[bd], %w[b], gt      \n\t"
        :[a]"+r"(la), [b]"+r"(lb)
        :[bd]"r"(bound)
        :"w0", "w1"
    );
#endif

    int r2 = g_CmpestrFuncList[imm8 & 0x0f].cmpFun(a, la, b, lb);
    r2 = neg_fun(r2, lb, imm8, bound);
    return (r2 == 0) ? bound : ((imm8 & 0x40) ? (31 - __builtin_clz(r2)) : __builtin_ctz(r2));
}

FORCE_INLINE __m128i _mm_cmpestrm(__m128i a, int la, __m128i b, int lb, const int imm8)
{
    __m128i dst;
    int bound = (imm8 & 0x01) ? 8 : 16;
#if defined(_MSC_VER)
    // msvc and msvc-like compiler does not support 'asr' in inline asm for ARM, so use C code
    if (la < 0) la = 0;
    if (lb < 0) lb = 0;
    if (la > bound) la = bound;
    if (lb > bound) lb = bound;
#else
    __asm__ __volatile__ (
        "eor w0, %w[a], %w[a], asr31          \n\t"
        "sub %w[a], w0, %w[a], asr31          \n\t"
        "eor w1, %w[b], %w[b], asr31          \n\t"
        "sub %w[b], w1, %w[b], asr31          \n\t"
        "cmp %w[a], %w[bd]                  \n\t"
        "csel %w[a], %w[bd], %w[a], gt      \n\t"
        "cmp %w[b], %w[bd]                  \n\t"
        "csel %w[b], %w[bd], %w[b], gt      \n\t"
        :[a]"+r"(la), [b]"+r"(lb)
        :[bd]"r"(bound)
        :"w0", "w1"
    );
#endif

    int r2 = g_CmpestrFuncList[imm8 & 0x0f].cmpFun(a, la, b, lb);
    r2 = neg_fun(r2, lb, imm8, bound);

    dst.vect_u8 = vdupq_n_u8(0);
    if (imm8 & 0x40) {
        if (bound == 8) {
            uint16x8_t tmp = vtstq_u16(vdupq_n_u16(r2), vld1q_u16(g_mask_epi16));
            dst.vect_u16 = vbslq_u16(tmp, vdupq_n_u16(-1), dst.vect_u16);
        } else {
            uint8x16_t vect_r2 = vcombine_u8(vdup_n_u8(r2), vdup_n_u8(r2 >> 8));
            uint8x16_t tmp = vtstq_u8(vect_r2, vld1q_u8(g_mask_epi8));
            dst.vect_u8 = vbslq_u8(tmp, vdupq_n_u8(-1), dst.vect_u8);
        }
    } else {
        if (bound == 16) {
            dst.vect_u16 = vsetq_lane_u16(r2 & 0xffff, dst.vect_u16, 0);
        } else {
            dst.vect_u8 = vsetq_lane_u8(r2 & 0xff, dst.vect_u8, 0);
        }
    }

    return dst;
}

FORCE_INLINE __m128i _mm_insert_epi32(__m128i a, int i, const int imm8)
{
    assert(imm8 >= 0 && imm8 <= 3);
    switch (imm8)
    {
    case 0:
        a.vect_s32 = vsetq_lane_s32(i, a.vect_s32, 0);
        break;
    case 1:
        a.vect_s32 = vsetq_lane_s32(i, a.vect_s32, 1);
        break;
    case 2:
        a.vect_s32 = vsetq_lane_s32(i, a.vect_s32, 2);
        break;
    case 3:
        a.vect_s32 = vsetq_lane_s32(i, a.vect_s32, 3);
        break;
    default:
        // never do this
        a.vect_s32 = vsetq_lane_s32(i, a.vect_s32, 0);
        break;
    }
    return a;
}

FORCE_INLINE __m128i _mm_load_epi32 (void const* mem_addr)
{
    __m128i res;
    res.vect_s32 = vld1q_s32((const int32_t *)mem_addr);
    return res;
}

FORCE_INLINE __m128i _mm_load_epi64 (void const* mem_addr)
{
    __m128i res;
    res.vect_s64 = vld1q_s64((const int64_t *)mem_addr);
    return res;
}

FORCE_INLINE __m128i _mm_load_si128 (__m128i const* mem_addr)
{
    __m128i res;
    res.vect_s32 = vld1q_s32((const int32_t *)mem_addr);
    return res;
}

FORCE_INLINE __m128d _mm_load_pd (double const* mem_addr)
{
    __m128d res;
    res = vld1q_f64((const double *)mem_addr);
    return res;
}

FORCE_INLINE __m128 _mm_load_ps (float const* mem_addr)
{
    __m128 res;
    res = vld1q_f32((const float *)mem_addr);
    return res;
}

FORCE_INLINE void _mm_store_epi32 (void* mem_addr, __m128i a)
{
    vst1q_s32((int32_t *)mem_addr, a.vect_s32);
}

FORCE_INLINE void _mm_store_epi64 (void* mem_addr, __m128i a)
{
    vst1q_s64((int64_t *)mem_addr, a.vect_s64);
}

FORCE_INLINE void _mm_store_si128 (__m128i* mem_addr, __m128i a)
{
    vst1q_s32((int32_t *)mem_addr, a.vect_s32);
}

FORCE_INLINE void _mm_store_pd (double* mem_addr, __m128d a)
{
    vst1q_f64(mem_addr, a);
}

FORCE_INLINE void _mm_store_ps (float* mem_addr, __m128 a)
{
    vst1q_f32(mem_addr, a);
}

FORCE_INLINE __m128i _mm_set1_epi8(char w)
{
    __m128i res;
    res.vect_s8 = vdupq_n_s8(w);
    return res;
}

FORCE_INLINE __m128i _mm_set1_epi16(short a)
{
    __m128i res;
    res.vect_s16 = vdupq_n_s16(a);
    return res;
}

FORCE_INLINE __m128i _mm_set1_epi32(int _i)
{
    __m128i res;
    res.vect_s32 = vdupq_n_s32(_i);
    return res;
}

FORCE_INLINE __m128 _mm_set1_ps (float a)
{
    __m128 res;
    res = vdupq_n_f32(a);
    return res;
}

FORCE_INLINE __m128i _mm_cmpeq_epi8 (__m128i a, __m128i b)
{
    __m128i res;
    res.vect_u8 = vceqq_s8(a.vect_s8, b.vect_s8);
    return res;
}

FORCE_INLINE __m128i _mm_cmpeq_epi32(__m128i a, __m128i b)
{
    __m128i res;
    res.vect_u32 = vceqq_s32(a.vect_s32, b.vect_s32);
    return res;
}

FORCE_INLINE void _mm_storeu_si128(__m128i *p, __m128i a)
{
    vst1q_s32((int32_t*) p, a.vect_s32);
}

FORCE_INLINE __m128i _mm_loadu_si128(const __m128i *p)
{
    __m128i res;
    res.vect_s32 = vld1q_s32((const int32_t *)p);
    return res;
}

FORCE_INLINE __m128i _mm_and_si128(__m128i a, __m128i b)
{
    __m128i res;
    res.vect_s32 = vandq_s32(a.vect_s32, b.vect_s32);
    return res;
}

FORCE_INLINE __m128i _mm_or_si128(__m128i a, __m128i b)
{
    __m128i res;
    res.vect_s32 = vorrq_s32(a.vect_s32, b.vect_s32);
    return res;
}

FORCE_INLINE __m128i _mm_xor_si128(__m128i a, __m128i b)
{
    __m128i res;
    res.vect_s32 = veorq_s32(a.vect_s32, b.vect_s32);
    return res;
}

FORCE_INLINE __m128i _mm_andnot_si128(__m128i a, __m128i b)
{
    __m128i res;
    res.vect_s32 = vbicq_s32(b.vect_s32, a.vect_s32);
    return res;
}

FORCE_INLINE __m128 _mm_castsi128_ps(__m128i a)
{
    __m128 res;
    res = vreinterpretq_f32_s32(a.vect_s32);
    return res;
}

FORCE_INLINE __m128i _mm_max_epu8(__m128i a, __m128i b)
{
    __m128i res;
    res.vect_u8 = vmaxq_u8(a.vect_u8, b.vect_u8);
    return res;
}

FORCE_INLINE __m128i _mm_min_epu8(__m128i a, __m128i b)
{
    __m128i res;
    res.vect_u8 = vminq_u8(a.vect_u8, b.vect_u8);
    return res;
}

FORCE_INLINE __m128i _mm_sub_epi8(__m128i a, __m128i b)
{
    __m128i res;
    res.vect_s8 = vsubq_s8(a.vect_s8, b.vect_s8);
    return res;
}

FORCE_INLINE __m128i _mm_adds_epu8(__m128i a, __m128i b)
{
    __m128i res;
    res.vect_u8 = vqaddq_u8(a.vect_u8, b.vect_u8);
    return res;
}

FORCE_INLINE __m128i _mm_adds_epi16(__m128i a, __m128i b)
{
    __m128i res;
    res.vect_s16 = vqaddq_s16(a.vect_s16, b.vect_s16);
    return res;
}

FORCE_INLINE __m128i _mm_setzero_si128()
{
    __m128i res;
    res.vect_s32 = vdupq_n_s32(0);
    return res;
}
#define MM_SLLI_SI128_CASE_STAT(a, res, imm8)\
    res.vect_s8 = vextq_s8(zero, a.vect_s8, 16 - imm8);
FORCE_INLINE __m128i _mm_slli_si128(__m128i a, const int imm8)
{
    __m128i res;
    if (imm8 > 0 && imm8 <= 15) {
        int8x16_t zero = vdupq_n_s8(0);
#if defined(_MSC_VER)
        // msvc and msvc-like compiler does not support 'asr' in inline asm for ARM, so use C code
        switch (imm8)
        {
        case 1:
            MM_SLLI_SI128_CASE_STAT(a, res, 1);
            break;
        case 2:
            MM_SLLI_SI128_CASE_STAT(a, res, 2);
            break;
        case 3:
            MM_SLLI_SI128_CASE_STAT(a, res, 3);
            break;
        case 4:
            MM_SLLI_SI128_CASE_STAT(a, res, 4);
            break;
        case 5:
            MM_SLLI_SI128_CASE_STAT(a, res, 5);
            break;
        case 6:
            MM_SLLI_SI128_CASE_STAT(a, res, 6);
            break;
        case 7:
            MM_SLLI_SI128_CASE_STAT(a, res, 7);
            break;
        case 8:
            MM_SLLI_SI128_CASE_STAT(a, res, 8);
            break;
        case 9:
            MM_SLLI_SI128_CASE_STAT(a, res, 9);
            break;
        case 10:
            MM_SLLI_SI128_CASE_STAT(a, res, 10);
            break;
        case 11:
            MM_SLLI_SI128_CASE_STAT(a, res, 11);
            break;
        case 12:
            MM_SLLI_SI128_CASE_STAT(a, res, 12);
            break;
        case 13:
            MM_SLLI_SI128_CASE_STAT(a, res, 13);
            break;
        case 14:
            MM_SLLI_SI128_CASE_STAT(a, res, 14);
            break;
        case 15:
            MM_SLLI_SI128_CASE_STAT(a, res, 15);
            break;
        default:
            // this will never access
            MM_SLLI_SI128_CASE_STAT(a, res, 1);
            break;
        }
#else
        __asm__ __volatile__ (
            "ext %0.16b, %1.16b, %2.16b, #%3"
            : "=w"(res.vect_s8)
            : "w"(zero), "w"(a.vect_s8), "i"(16 - imm8)
            : /*No clobbers */);
#endif
    } else if (imm8 == 0) {
        res = a;
    } else {
        res.vect_s8 = vdupq_n_s8(0);
    }
    return res;
}

#define MM_SRLI_SI128_SET_CASE_STAT(a, res, imm8) \
    res.vect_s8 = vextq_s8(a.vect_s8, vdupq_n_s8(0), (imm8));
FORCE_INLINE __m128i _mm_srli_si128 (__m128i a, int imm8)
{
    assert(imm8 >=0 && imm8 < 256);
    __m128i res;
    if (likely(imm8 > 0 && imm8 <= 15)) {
        switch (imm8)
        {
        case 1:
            MM_SRLI_SI128_SET_CASE_STAT(a, res, 1);
            break;
        case 2:
            MM_SRLI_SI128_SET_CASE_STAT(a, res, 2);
            break;
        case 3:
            MM_SRLI_SI128_SET_CASE_STAT(a, res, 3);
            break;
        case 4:
            MM_SRLI_SI128_SET_CASE_STAT(a, res, 4);
            break;
        case 5:
            MM_SRLI_SI128_SET_CASE_STAT(a, res, 5);
            break;
        case 6:
            MM_SRLI_SI128_SET_CASE_STAT(a, res, 6);
            break;
        case 7:
            MM_SRLI_SI128_SET_CASE_STAT(a, res, 7);
            break;
        case 8:
            MM_SRLI_SI128_SET_CASE_STAT(a, res, 8);
            break;
        case 9:
            MM_SRLI_SI128_SET_CASE_STAT(a, res, 9);
            break;
        case 10:
            MM_SRLI_SI128_SET_CASE_STAT(a, res, 10);
            break;
        case 11:
            MM_SRLI_SI128_SET_CASE_STAT(a, res, 11);
            break;
        case 12:
            MM_SRLI_SI128_SET_CASE_STAT(a, res, 12);
            break;
        case 13:
            MM_SRLI_SI128_SET_CASE_STAT(a, res, 13);
            break;
        case 14:
            MM_SRLI_SI128_SET_CASE_STAT(a, res, 14);
            break;
        case 15:
            MM_SRLI_SI128_SET_CASE_STAT(a, res, 15);
            break;
        default:
            // this will never access
            MM_SRLI_SI128_SET_CASE_STAT(a, res, 1);
            break;
        }
    } else if (imm8 == 0) {
        res = a;
    } else {
        res.vect_s8 = vdupq_n_s8(0);
    }
    return res;
}

#define MM_SLLI_SEPI32_CASE_STAT(a, res, imm8) \
    res.vect_s32 = vshlq_n_s32(a.vect_s32, imm8);
FORCE_INLINE __m128i _mm_slli_epi32 (__m128i a, int imm8)
{
    __m128i res;
    if (likely(imm8 >= 0 && imm8 < 32)) {
        switch (imm8)
        {
        case 1:
            MM_SLLI_SEPI32_CASE_STAT(a, res, 1);
            break;
        case 2:
            MM_SLLI_SEPI32_CASE_STAT(a, res, 2);
            break;
        case 3:
            MM_SLLI_SEPI32_CASE_STAT(a, res, 3);
            break;
        case 4:
            MM_SLLI_SEPI32_CASE_STAT(a, res, 4);
            break;
        case 5:
            MM_SLLI_SEPI32_CASE_STAT(a, res, 5);
            break;
        case 6:
            MM_SLLI_SEPI32_CASE_STAT(a, res, 6);
            break;
        case 7:
            MM_SLLI_SEPI32_CASE_STAT(a, res, 7);
            break;
        case 8:
            MM_SLLI_SEPI32_CASE_STAT(a, res, 8);
            break;
        case 9:
            MM_SLLI_SEPI32_CASE_STAT(a, res, 9);
            break;
        case 10:
            MM_SLLI_SEPI32_CASE_STAT(a, res, 10);
            break;
        case 11:
            MM_SLLI_SEPI32_CASE_STAT(a, res, 11);
            break;
        case 12:
            MM_SLLI_SEPI32_CASE_STAT(a, res, 12);
            break;
        case 13:
            MM_SLLI_SEPI32_CASE_STAT(a, res, 13);
            break;
        case 14:
            MM_SLLI_SEPI32_CASE_STAT(a, res, 14);
            break;
        case 15:
            MM_SLLI_SEPI32_CASE_STAT(a, res, 15);
            break;
        case 16:
            MM_SLLI_SEPI32_CASE_STAT(a, res, 16);
            break;
        case 17:
            MM_SLLI_SEPI32_CASE_STAT(a, res, 17);
            break;
        case 18:
            MM_SLLI_SEPI32_CASE_STAT(a, res, 18);
            break;
        case 19:
            MM_SLLI_SEPI32_CASE_STAT(a, res, 19);
            break;
        case 20:
            MM_SLLI_SEPI32_CASE_STAT(a, res, 20);
            break;
        case 21:
            MM_SLLI_SEPI32_CASE_STAT(a, res, 21);
            break;
        case 22:
            MM_SLLI_SEPI32_CASE_STAT(a, res, 22);
            break;
        case 23:
            MM_SLLI_SEPI32_CASE_STAT(a, res, 23);
            break;
        case 24:
            MM_SLLI_SEPI32_CASE_STAT(a, res, 24);
            break;
        case 25:
            MM_SLLI_SEPI32_CASE_STAT(a, res, 25);
            break;
        case 26:
            MM_SLLI_SEPI32_CASE_STAT(a, res, 26);
            break;
        case 27:
            MM_SLLI_SEPI32_CASE_STAT(a, res, 27);
            break;
        case 28:
            MM_SLLI_SEPI32_CASE_STAT(a, res, 28);
            break;
        case 29:
            MM_SLLI_SEPI32_CASE_STAT(a, res, 29);
            break;
        case 30:
            MM_SLLI_SEPI32_CASE_STAT(a, res, 30);
            break;
        case 31:
            MM_SLLI_SEPI32_CASE_STAT(a, res, 31);
            break;
        default:
            // this will never access
            MM_SLLI_SEPI32_CASE_STAT(a, res, 1);
            break;
        }
    } else {
        res.vect_s32 = vdupq_n_s32(0);
    } 
    return res;
}

#define MM_SLLI_EPI64_CASE_STST(a, res, imm8) \
    res.vect_s64 = vshlq_n_s64(a.vect_s64, imm8);   
FORCE_INLINE __m128i _mm_slli_epi64 (__m128i a, int imm8)
{
    __m128i res;
    if (likely(imm8 >= 0 && imm8 < 64)) {
        switch (imm8)
        {
        case 0:
            res.vect_s64 = vshlq_n_s64(a.vect_s64, 0);
            break;
        case 1:
            res.vect_s64 = vshlq_n_s64(a.vect_s64, 1);
            break;
        case 2:
            res.vect_s64 = vshlq_n_s64(a.vect_s64, 2);
            break;
        case 3:
            res.vect_s64 = vshlq_n_s64(a.vect_s64, 3);
            break;
        case 4:
            res.vect_s64 = vshlq_n_s64(a.vect_s64, 4);
            break;
        case 5:
            res.vect_s64 = vshlq_n_s64(a.vect_s64, 5);
            break;
        case 6:
            res.vect_s64 = vshlq_n_s64(a.vect_s64, 6);
            break;
        case 7:
            res.vect_s64 = vshlq_n_s64(a.vect_s64, 7);
            break;
        case 8:
            res.vect_s64 = vshlq_n_s64(a.vect_s64, 8);
            break;
        case 9:
            res.vect_s64 = vshlq_n_s64(a.vect_s64, 9);
            break;
        case 10:
            res.vect_s64 = vshlq_n_s64(a.vect_s64, 10);
            break;
        case 11:
            res.vect_s64 = vshlq_n_s64(a.vect_s64, 11);
            break;
        case 12:
            res.vect_s64 = vshlq_n_s64(a.vect_s64, 12);
            break;
        case 13:
            res.vect_s64 = vshlq_n_s64(a.vect_s64, 13);
            break;
        case 14:
            res.vect_s64 = vshlq_n_s64(a.vect_s64, 14);
            break;
        case 15:
            res.vect_s64 = vshlq_n_s64(a.vect_s64, 15);
            break;
        case 16:
            res.vect_s64 = vshlq_n_s64(a.vect_s64, 16);
            break;
        case 17:
            res.vect_s64 = vshlq_n_s64(a.vect_s64, 17);
            break;
        case 18:
            res.vect_s64 = vshlq_n_s64(a.vect_s64, 18);
            break;
        case 19:
            res.vect_s64 = vshlq_n_s64(a.vect_s64, 19);
            break;
        case 20:
            res.vect_s64 = vshlq_n_s64(a.vect_s64, 20);
            break;
        case 21:
            res.vect_s64 = vshlq_n_s64(a.vect_s64, 21);
            break;
        case 22:
            res.vect_s64 = vshlq_n_s64(a.vect_s64, 22);
            break;
        case 23:
            res.vect_s64 = vshlq_n_s64(a.vect_s64, 23);
            break;
        case 24:
            res.vect_s64 = vshlq_n_s64(a.vect_s64, 24);
            break;
        case 25:
            res.vect_s64 = vshlq_n_s64(a.vect_s64, 25);
            break;
        case 26:
            res.vect_s64 = vshlq_n_s64(a.vect_s64, 26);
            break;
        case 27:
            res.vect_s64 = vshlq_n_s64(a.vect_s64, 27);
            break;
        case 28:
            res.vect_s64 = vshlq_n_s64(a.vect_s64, 28);
            break;
        case 29:
            res.vect_s64 = vshlq_n_s64(a.vect_s64, 29);
            break;
        case 30:
            res.vect_s64 = vshlq_n_s64(a.vect_s64, 30);
            break;
        case 31:
            res.vect_s64 = vshlq_n_s64(a.vect_s64, 31);
            break;
        case 32:
            res.vect_s64 = vshlq_n_s64(a.vect_s64, 32);
            break;
        case 33:
            res.vect_s64 = vshlq_n_s64(a.vect_s64, 33);
            break;
        case 34:
            res.vect_s64 = vshlq_n_s64(a.vect_s64, 34);
            break;
        case 35:
            res.vect_s64 = vshlq_n_s64(a.vect_s64, 35);
            break;
        case 36:
            res.vect_s64 = vshlq_n_s64(a.vect_s64, 36);
            break;
        case 37:
            res.vect_s64 = vshlq_n_s64(a.vect_s64, 37);
            break;
        case 38:
            res.vect_s64 = vshlq_n_s64(a.vect_s64, 38);
            break;
        case 39:
            res.vect_s64 = vshlq_n_s64(a.vect_s64, 39);
            break;
        case 40:
            res.vect_s64 = vshlq_n_s64(a.vect_s64, 40);
            break;
        case 41:
            res.vect_s64 = vshlq_n_s64(a.vect_s64, 41);
            break;
        case 42:
            res.vect_s64 = vshlq_n_s64(a.vect_s64, 42);
            break;
        case 43:
            res.vect_s64 = vshlq_n_s64(a.vect_s64, 43);
            break;
        case 44:
            res.vect_s64 = vshlq_n_s64(a.vect_s64, 44);
            break;
        case 45:
            res.vect_s64 = vshlq_n_s64(a.vect_s64, 45);
            break;
        case 46:
            res.vect_s64 = vshlq_n_s64(a.vect_s64, 46);
            break;
        case 47:
            res.vect_s64 = vshlq_n_s64(a.vect_s64, 47);
            break;
        case 48:
            res.vect_s64 = vshlq_n_s64(a.vect_s64, 48);
            break;
        case 49:
            res.vect_s64 = vshlq_n_s64(a.vect_s64, 49);
            break;
        case 50:
            res.vect_s64 = vshlq_n_s64(a.vect_s64, 50);
            break;
        case 51:
            res.vect_s64 = vshlq_n_s64(a.vect_s64, 51);
            break;
        case 52:
            res.vect_s64 = vshlq_n_s64(a.vect_s64, 52);
            break;
        case 53:
            res.vect_s64 = vshlq_n_s64(a.vect_s64, 53);
            break;
        case 54:
            res.vect_s64 = vshlq_n_s64(a.vect_s64, 54);
            break;
        case 55:
            res.vect_s64 = vshlq_n_s64(a.vect_s64, 55);
            break;
        case 56:
            res.vect_s64 = vshlq_n_s64(a.vect_s64, 56);
            break;
        case 57:
            res.vect_s64 = vshlq_n_s64(a.vect_s64, 57);
            break;
        case 58:
            res.vect_s64 = vshlq_n_s64(a.vect_s64, 58);
            break;
        case 59:
            res.vect_s64 = vshlq_n_s64(a.vect_s64, 59);
            break;
        case 60:
            res.vect_s64 = vshlq_n_s64(a.vect_s64, 60);
            break;
        case 61:
            res.vect_s64 = vshlq_n_s64(a.vect_s64, 61);
            break;
        case 62:
            res.vect_s64 = vshlq_n_s64(a.vect_s64, 62);
            break;
        case 63:
            res.vect_s64 = vshlq_n_s64(a.vect_s64, 63);
            break;
        default:
            // this will never access
            res.vect_s64 = vshlq_n_s64(a.vect_s64, 0);
            break;
        }
    } else {
        res.vect_s64 = vdupq_n_s64(0);
    } 
    return res;
}

FORCE_INLINE __m128i _mm_srli_epi64 (__m128i a, int imm8)
{
    __m128i res;
    if (likely(imm8 >= 0 && imm8 < 64)) {
        int64x2_t vect_imm = vdupq_n_s64(-imm8);
        res.vect_u64 = vshlq_u64(a.vect_u64, vect_imm);
    } else {
        res.vect_u64 = vdupq_n_u64(0);
    } 
    return res;
}

FORCE_INLINE __m128i _mm_cvtsi32_si128(int a)
{
    __m128i res;
    res.vect_s32 = vsetq_lane_s32(a, vdupq_n_s32(0), 0);
    return res;
}

FORCE_INLINE int _mm_cvtsi128_si32(__m128i a)
{
    return vgetq_lane_s32(a.vect_s32, 0);
}

FORCE_INLINE __m128i _mm_packs_epi16(__m128i a, __m128i b)
{
    __m128i res;
    res.vect_s8 = vcombine_s8(vqmovn_s16(a.vect_s16), vqmovn_s16(b.vect_s16));
    return res;
}

FORCE_INLINE __m128i _mm_packs_epi32(__m128i a, __m128i b)
{
    __m128i res;
    res.vect_s16 = vcombine_s16(vqmovn_s32(a.vect_s32), vqmovn_s32(b.vect_s32));
    return res;
}

FORCE_INLINE int _mm_movemask_ps(__m128 a)
{
    __m128i res_m128i;
    res_m128i.vect_u32 = vshrq_n_u32(vreinterpretq_u32_f32(a), 31);
    res_m128i.vect_u64 = vsraq_n_u64(res_m128i.vect_u64, res_m128i.vect_u64, 31);
    return (int)(vgetq_lane_u8(res_m128i.vect_u8, 0) | (vgetq_lane_u8(res_m128i.vect_u8, 8) << 2));
}

FORCE_INLINE int _mm_movemask_epi8(__m128i a)
{
#if defined(_MSC_VER) && !defined(__clang__)

	// 第一步：右移每个字节 7 位
	uint8x16_t v_u8 = vshrq_n_u8(a.vect_u8, 7);

	// 第二步：转换为 uint16x8_t 并 usra（右移 7 位后加回）
	uint16x8_t v_u16 = vreinterpretq_u16_u8(v_u8);
	v_u16 = vshrq_n_u16(v_u16, 7);
	v_u16 = vaddq_u16(v_u16, vreinterpretq_u16_u8(v_u8));

	// 第三步：转换为 uint32x4_t 并 usra（右移 14 位后加回）
	uint32x4_t v_u32 = vreinterpretq_u32_u16(v_u16);
	v_u32 = vaddq_u32(v_u32, vshrq_n_u32(v_u32, 14));

	// 第四步：转换为 uint64x2_t 并 usra（右移 28 位后加回）
	uint64x2_t v_u64 = vreinterpretq_u64_u32(v_u32);
	v_u64 = vaddq_u64(v_u64, vshrq_n_u64(v_u64, 28));

	// 第五步：将第 8 个字节插入到第 1 个字节
	uint8x16_t v_final = vreinterpretq_u8_u64(v_u64);
	v_final = vsetq_lane_u8(vgetq_lane_u8(v_final, 8), v_final, 1);

	// 第六步：提取第一个 halfword（uint16_t）
	uint16x8_t v_final_u16 = vreinterpretq_u16_u8(v_final);
	return vgetq_lane_u16(v_final_u16, 0);

#endif
#if defined(__GNUC__) || defined(__clang__)
    int res;
    __asm__ __volatile__(
        "ushr %[a0].16b, %[a0].16b, #7          \n\t"
        "usra %[a0].8h, %[a0].8h, #7            \n\t"
        "usra %[a0].4s, %[a0].4s, #14           \n\t"
        "usra %[a0].2d, %[a0].2d, #28           \n\t"
        "ins %[a0].b[1], %[a0].b[8]             \n\t"
        "umov %w[r], %[a0].h[0]"
        : [r] "=r"(res), [a0] "+w"(a.vect_u8)
        :
        :);
    return res;
#endif
}

FORCE_INLINE __m128i _mm_shuffle_epi8(__m128i a,__m128i b)
{
    __m128i res;
    uint8x16_t mask_and = vdupq_n_u8(0x8f);
    res.vect_u8 = vqtbl1q_u8(a.vect_u8, vandq_u8(b.vect_u8, mask_and));
    return res;
}

#ifdef _WIN32
#include <malloc.h>
#endif
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winfinite-recursion"
#endif
FORCE_INLINE void* _mm_malloc(size_t size, size_t align)
{
    void* ptr = NULL;
#ifdef _WIN32
    ptr = _aligned_malloc(size, align);
    printf("111111111111111111\n");
#else
    if (align == 1)
        return malloc(size);
    if (align == 2 || (sizeof(void*) == 8 && align == 4))
        align = sizeof(void*);
    if (posix_memalign(&ptr, align, size) != 0)
        ptr = NULL;
#endif
    printf("2222222222222222222\n");
    return ptr;
}

FORCE_INLINE void _mm_free(void* mem_addr)
{
#ifdef _WIN32
    _aligned_free(mem_addr);
#else
    free(mem_addr);
#endif
}
#ifdef __clang__
#pragma clang diagnostic pop
#endif

FORCE_INLINE __m128i _mm_subs_epu8(__m128i a, __m128i b)
{
	__m128i res;
	res.vect_u8 = vqsubq_u8(a.vect_u8, b.vect_u8);
	return res;
}

FORCE_INLINE __m128i _mm_subs_epu16(__m128i a, __m128i b)
{
	__m128i res;
	res.vect_u16 = vqsubq_u16(a.vect_u16, b.vect_u16);
	return res;
}

FORCE_INLINE int _mm_extract_epi16(__m128i a, const int imm8)
{
#if defined(_MSC_VER) && !defined(__clang__)
    const unsigned char index = imm8 & 0x7;
    switch (index) {
        case 0:
            return vgetq_lane_s16(a.vect_s16, 0);
        case 1:
            return vgetq_lane_s16(a.vect_s16, 1);
        case 2:
            return vgetq_lane_s16(a.vect_s16, 2);
        case 3:
            return vgetq_lane_s16(a.vect_s16, 3);
        case 4:
            return vgetq_lane_s16(a.vect_s16, 4);
        case 5:
            return vgetq_lane_s16(a.vect_s16, 5);
        case 6:
            return vgetq_lane_s16(a.vect_s16, 6);
        case 7:
            return vgetq_lane_s16(a.vect_s16, 7);
        default:
            // unreachable
            assert(0 && "Unexpected value imm8");
            return vgetq_lane_s16(a.vect_s16, 0);
    }
#endif
#if defined(__GNUC__) || defined(__clang__)
    return a.vect_s16[imm8 & 0x7] & 0xffff;
#endif
}

FORCE_INLINE __m128i _mm_max_epi16(__m128i a, __m128i b)
{
	__m128i dst;
	dst.vect_s16 = vmaxq_s16(a.vect_s16, b.vect_s16);
	return dst;
}

FORCE_INLINE __m128i _mm_cmpgt_epi16(__m128i a, __m128i b)
{
	__m128i dst;
	dst.vect_u16 = vcgtq_s16(a.vect_s16, b.vect_s16);
	return dst;
}

// for AVX FMA Intrinsic 3'rd param c is dest c = (a * b)
// for ARM NEON Intrinsic 1'st param is dest
FORCE_INLINE __m128 _mm_fmadd_ps(__m128 a, __m128 b, __m128 c)
{
    return vfmaq_f32(c, a, b);
}

// k is mask which decide whether to do FMA or not
// This is an extension of _mm_fmadd_ps
FORCE_INLINE __m128 _mm_mask_fmadd_ps(__m128 a, __mmask8 k, __m128 b, __m128 c)
{
    assert(k <= 0xF);
    uint32_t m0 = (k & 0x1) ? 0xFFFFFFFF : 0;
    uint32_t m1 = (k & 0x2) ? 0xFFFFFFFF : 0;
    uint32_t m2 = (k & 0x4) ? 0xFFFFFFFF : 0;
    uint32_t m3 = (k & 0x8) ? 0xFFFFFFFF : 0;
#if defined(_MSC_VER) && !defined(__clang__)
    uint32x4_t mask = vdupq_n_u32(0);
    mask = vsetq_lane_u32(m0, mask, 0);
    mask = vsetq_lane_u32(m1, mask, 1);
    mask = vsetq_lane_u32(m2, mask, 2);
    mask = vsetq_lane_u32(m3, mask, 3);
#endif
#if defined(__GNUC__) || defined(__clang__)
    uint32x4_t mask = (uint32x4_t){m0, m1, m2, m3};
#endif
    float32x4_t muladd = vfmaq_f32(c, a, b);
    float32x4_t result = vbslq_f32(mask, muladd, a);
    return result;
}

FORCE_INLINE __m128d _mm_fmadd_pd(__m128d a, __m128d b, __m128d c)
{
    return vfmaq_f64(c, a, b);
}

FORCE_INLINE __m128d _mm_mask_fmadd_pd(__m128d a, __mmask8 k, __m128d b, __m128d c)
{
    assert(k <= 0x3);
    uint64_t m0 = (k & 0x1) ? 0xFFFFFFFFFFFFFFFF : 0;
    uint64_t m1 = (k & 0x2) ? 0xFFFFFFFFFFFFFFFF : 0;
    uint64x2_t mask = (uint64x2_t){ m0, m1 };
    float64x2_t muladd = vfmaq_f64(c, a, b);
    float64x2_t result = vbslq_f64(mask, muladd, a);
    return result;
}