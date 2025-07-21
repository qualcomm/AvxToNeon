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

#ifndef AVX2NEON_H
#error " Never use <typedefs.h> directly; include " avx2neon.h" instead."
#endif

#ifndef TYPEDEFS_H
#define TYPEDEFS_H

#if defined(__GNUC__) || defined(__clang__)
#pragma push_macro("FORCE_INLINE")
#pragma push_macro("ALIGN_STRUCT")
#define FORCE_INLINE static inline __attribute__((always_inline))
#define ALIGN_STRUCT(x) __attribute__((aligned(x)))
#define GET_LANE_S64_FROM128(vec, index) (vec.vect_s64[(index)])
#define GET_LANE_S32_FROM128(vec, index) (vec.vect_s32[(index)])

#elif defined(_MSC_VER) && !defined(__clang__)
#pragma push_macro("FORCE_INLINE")
#pragma push_macro("ALIGN_STRUCT")
#define FORCE_INLINE __forceinline
#define ALIGN_STRUCT(x) __declspec(align(x))
#define GET_LANE_S64_FROM128(vec, index) vgetq_lane_s64((vec.vect_s64), (index))
#define GET_LANE_S32_FROM128(vec, index) vgetq_lane_s32((vec.vect_s32), (index))

#else
#error "Macro name collisions may happens with unknown compiler"
#ifdef FORCE_INLINE
#undef FORCE_INLINE
#endif

#define FORCE_INLINE static inline
#ifndef ALIGN_STRUCT
#define ALIGN_STRUCT(x) __declspec(align(x))
#endif

#endif

#define likely(x)     __builtin_expect(!!(x), 1)
#define unlikely(x)   __builtin_expect(!!(x), 0)

#ifndef __int32
#define __int32 int
#endif

#ifndef __int64
#define __int64 long long
#endif

#ifndef __mmask64
#define __mmask64 unsigned long long
#endif

#ifndef __mmask16
#define __mmask16 unsigned short
#endif

#ifndef __mmask8
#define __mmask8 unsigned char
#endif

#ifndef __builtin_expect
#define __builtin_expect(expr, val) (expr)
#endif

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#pragma intrinsic(_BitScanForward)
static FORCE_INLINE int __builtin_ctz(unsigned int x) {
    unsigned long index;
    if (x == 0) return 32;
    _BitScanForward(&index, x);
    return (int)index;
}
static FORCE_INLINE int __builtin_clz(unsigned int x) {
    if (x == 0) return 32;
    int n = 0;
    for (unsigned int mask = 0x80000000; (x & mask) == 0; mask >>= 1) {
        n++;
    }
    return n;
}
#endif

#endif //TYPEDEFS_H