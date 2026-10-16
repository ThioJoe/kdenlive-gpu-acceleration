/*
    SPDX-FileCopyrightText: 2026 ThioJoe
    SPDX-License-Identifier: GPL-3.0-only
*/

#pragma once

#include <cmath>
#include <cstdint>

// Same type as in kernels/gpufx_device.h: kernel arguments must match exactly
typedef uint32_t gpufx_u32;

namespace gpufx {

/** qRound() as implemented by Qt 6 on x86-64 */
inline int qtRound(double d)
{
    return int(d + std::copysign(0.5, d));
}

/** QColor(r, g, b, a) premultiplied the way QPainter does for a solid fill:
 *  premultiplied in 16 bits (QRgba64::premultiplied()) then converted back to
 *  8 bits (QColor::rgba()). Packed as RGBA8888. */
inline gpufx_u32 qtPremultipliedColor(uint8_t r, uint8_t g, uint8_t b, uint8_t a)
{
    const uint32_t a16 = a * 257u;
    auto div257 = [](uint32_t x) {
        x += 128u;
        return (x - (x >> 8)) >> 8;
    };
    auto channel = [&](uint8_t c) {
        const uint32_t c16 = c * 257u;
        uint32_t v;
        if (a16 == 65535u) {
            v = c16;
        } else if (a16 == 0) {
            v = 0;
        } else {
            const uint32_t t = c16 * a16;
            v = (t + (t >> 16) + 0x8000u) >> 16;
        }
        return div257(v);
    };
    return channel(r) | (channel(g) << 8) | (channel(b) << 16) | (div257(a16) << 24);
}

} // namespace gpufx
