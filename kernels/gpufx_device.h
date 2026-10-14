/*
    SPDX-FileCopyrightText: 2026 Kdenlive contributors
    SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

/*
 * Helpers shared by the CUDA kernels in gpufx_kernels.cu.
 *
 * The kernels are compiled to PTX with nvcc (see gpufx/CMakeLists.txt). For the
 * unit tests the very same source is also compiled as plain C++ with
 * GPUFX_EMULATION defined, and executed on the CPU by the mock CUDA driver.
 * Everything used by the kernels therefore has to be available in both worlds:
 * no CUDA headers, no libdevice math (pow, exp, ...), no shared memory or
 * __syncthreads(). Anything transcendental is precomputed on the host.
 */

#pragma once

#if defined(__CUDACC__)

typedef unsigned char gpufx_u8;
typedef unsigned short gpufx_u16;
typedef unsigned int gpufx_u32;
typedef unsigned long long gpufx_u64;
typedef long long gpufx_i64;

#define GPUFX_KERNEL extern "C" __global__
#define GPUFX_DEV static __device__ __forceinline__
#define GPUFX_UNROLL _Pragma("unroll")

GPUFX_DEV int gpufx_f2i_rn(float x)
{
    return __float2int_rn(x);
}
GPUFX_DEV unsigned gpufx_f2u_rz(float x)
{
    return __float2uint_rz(x);
}
GPUFX_DEV float gpufx_i2f_rn(int x)
{
    return __int2float_rn(x);
}
GPUFX_DEV float gpufx_fmul_rn(float a, float b)
{
    return __fmul_rn(a, b);
}
GPUFX_DEV float gpufx_fdiv_rn(float a, float b)
{
    return __fdiv_rn(a, b);
}

#else // CPU emulation used by the tests

#include <cmath>
#include <cstdint>

typedef uint8_t gpufx_u8;
typedef uint16_t gpufx_u16;
typedef uint32_t gpufx_u32;
typedef uint64_t gpufx_u64;
typedef int64_t gpufx_i64;

struct GpufxDim3
{
    unsigned x, y, z;
};
extern thread_local GpufxDim3 threadIdx;
extern thread_local GpufxDim3 blockIdx;
extern thread_local GpufxDim3 blockDim;
extern thread_local GpufxDim3 gridDim;

#ifndef __restrict__
#define __restrict__ __restrict
#endif

#define GPUFX_KERNEL extern "C"
#define GPUFX_DEV static inline
#define GPUFX_UNROLL

GPUFX_DEV int gpufx_f2i_rn(float x)
{
    return int(std::nearbyint(x));
}
GPUFX_DEV unsigned gpufx_f2u_rz(float x)
{
    return unsigned(x);
}
GPUFX_DEV float gpufx_i2f_rn(int x)
{
    return float(x);
}
GPUFX_DEV float gpufx_fmul_rn(float a, float b)
{
    volatile float r = a * b;
    return r;
}
GPUFX_DEV float gpufx_fdiv_rn(float a, float b)
{
    volatile float r = a / b;
    return r;
}

#endif

GPUFX_DEV int gpufx_min(int a, int b)
{
    return a < b ? a : b;
}
GPUFX_DEV int gpufx_max(int a, int b)
{
    return a > b ? a : b;
}
GPUFX_DEV int gpufx_clamp(int v, int lo, int hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

/* ------------------------------------------------------------------------
 * 8-bit and 16-bit premultiplied arithmetic, matching Qt's raster engine.
 * Pixels are RGBA8888 in memory, i.e. r | g << 8 | b << 16 | a << 24 when
 * loaded as a little-endian 32-bit word.
 * ------------------------------------------------------------------------ */

// round(t / 255) the way Qt's BYTE_MUL()/qPremultiply() do it, for t = c * a
GPUFX_DEV gpufx_u32 gpufx_div255(gpufx_u32 t)
{
    return (t + (t >> 8) + 0x80u) >> 8;
}

// qt_div_65535(): round(x / 65535) for x = c * a with 16-bit c and a
GPUFX_DEV gpufx_u32 gpufx_div65535(gpufx_u32 x)
{
    return (x + (x >> 16) + 0x8000u) >> 16;
}

// qt_div_257(): 16-bit to 8-bit channel conversion with rounding
GPUFX_DEV gpufx_u32 gpufx_div257(gpufx_u32 x)
{
    x += 128u;
    return (x - (x >> 8)) >> 8;
}

GPUFX_DEV gpufx_u32 gpufx_pack(gpufx_u32 r, gpufx_u32 g, gpufx_u32 b, gpufx_u32 a)
{
    return r | (g << 8) | (b << 16) | (a << 24);
}

// RGBA8888 (straight alpha) to 8-bit premultiplied, as qPremultiply().
GPUFX_DEV gpufx_u32 gpufx_premultiply8(gpufx_u32 p)
{
    gpufx_u32 a = p >> 24;
    return gpufx_pack(gpufx_div255((p & 0xffu) * a), gpufx_div255(((p >> 8) & 0xffu) * a), gpufx_div255(((p >> 16) & 0xffu) * a), a);
}

// BYTE_MUL(): multiply every channel of a premultiplied pixel by a / 255
GPUFX_DEV gpufx_u32 gpufx_byte_mul(gpufx_u32 p, gpufx_u32 a)
{
    return gpufx_pack(gpufx_div255((p & 0xffu) * a), gpufx_div255(((p >> 8) & 0xffu) * a), gpufx_div255(((p >> 16) & 0xffu) * a), gpufx_div255((p >> 24) * a));
}

struct GpufxRgba64
{
    gpufx_u32 r, g, b, a; // 16-bit premultiplied channels
};

/* Fetch of a straight-alpha RGBA8888 pixel into Qt's 16-bit premultiplied
 * pipeline (convertARGBToRGBA64PM, AVX2 variant: (c * a) >> 16 corrected by
 * the top bit). */
GPUFX_DEV GpufxRgba64 gpufx_fetch64(gpufx_u32 p)
{
    gpufx_u32 a8 = p >> 24;
    GpufxRgba64 v;
    if (a8 == 0) {
        v.r = v.g = v.b = v.a = 0;
        return v;
    }
    gpufx_u32 r16 = (p & 0xffu) * 257u;
    gpufx_u32 g16 = ((p >> 8) & 0xffu) * 257u;
    gpufx_u32 b16 = ((p >> 16) & 0xffu) * 257u;
    if (a8 == 255) {
        v.r = r16;
        v.g = g16;
        v.b = b16;
        v.a = 65535u;
        return v;
    }
    gpufx_u32 a16 = a8 * 257u;
    gpufx_u32 x = (r16 * a16) >> 16;
    v.r = x + (x >> 15);
    x = (g16 * a16) >> 16;
    v.g = x + (x >> 15);
    x = (b16 * a16) >> 16;
    v.b = x + (x >> 15);
    v.a = a16;
    return v;
}

// An 8-bit premultiplied pixel expanded to 16 bits (fetchARGB32PMToRGBA64PM)
GPUFX_DEV GpufxRgba64 gpufx_expand_pm8(gpufx_u32 pm)
{
    GpufxRgba64 v;
    v.r = (pm & 0xffu) * 257u;
    v.g = ((pm >> 8) & 0xffu) * 257u;
    v.b = ((pm >> 16) & 0xffu) * 257u;
    v.a = (pm >> 24) * 257u;
    return v;
}

// comp_func_SourceOver_rgb64()
GPUFX_DEV GpufxRgba64 gpufx_source_over64(GpufxRgba64 s, GpufxRgba64 d)
{
    if (s.a == 65535u) return s;
    if (s.a == 0) return d;
    gpufx_u32 ia = 65535u - s.a;
    GpufxRgba64 r;
    r.r = s.r + gpufx_div65535(d.r * ia);
    r.g = s.g + gpufx_div65535(d.g * ia);
    r.b = s.b + gpufx_div65535(d.b * ia);
    r.a = s.a + gpufx_div65535(d.a * ia);
    return r;
}

/* Store of a 16-bit premultiplied pixel to straight-alpha RGBA8888
 * (convertARGBFromRGBA64PM_sse4: un-premultiply with a float reciprocal). */
GPUFX_DEV gpufx_u32 gpufx_store64(GpufxRgba64 v)
{
    if (v.a == 0) return 0;
    if (v.a == 65535u) return gpufx_pack(gpufx_div257(v.r), gpufx_div257(v.g), gpufx_div257(v.b), 255u);
    float ia = gpufx_fdiv_rn(255.0f, gpufx_i2f_rn(int(v.a)));
    int r = gpufx_f2i_rn(gpufx_fmul_rn(gpufx_i2f_rn(int(v.r)), ia));
    int g = gpufx_f2i_rn(gpufx_fmul_rn(gpufx_i2f_rn(int(v.g)), ia));
    int b = gpufx_f2i_rn(gpufx_fmul_rn(gpufx_i2f_rn(int(v.b)), ia));
    return gpufx_pack(gpufx_clamp(r, 0, 255), gpufx_clamp(g, 0, 255), gpufx_clamp(b, 0, 255), gpufx_div257(v.a));
}

// Integer IIR step of Qt's expblur<12, 10, true> (qt_blurinner_alphaOnly)
GPUFX_DEV gpufx_u8 gpufx_expblur_step(int &z, int value, int alpha)
{
    z += alpha * ((value << 10) - (z >> 12));
    return gpufx_u8(z >> 22);
}
