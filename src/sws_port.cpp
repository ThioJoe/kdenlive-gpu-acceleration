/*
    SPDX-FileCopyrightText: FFmpeg developers
    SPDX-FileCopyrightText: 2026 Kdenlive contributors
    SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "sws_port.h"

#include <cstdint>
#include <cstdlib>

namespace gpufx {
namespace sws {

namespace {

// ff_yuv2rgb_coeffs (libswscale/yuv2rgb.c)
const int32_t kCoefficients[11][4] = {
    {117489, 138438, 13975, 34925}, // no sequence_display_extension
    {117489, 138438, 13975, 34925}, // ITU-R Rec. 709 (1990)
    {104597, 132201, 25675, 53279}, // unspecified
    {104597, 132201, 25675, 53279}, // reserved
    {104448, 132798, 24759, 53109}, // FCC
    {104597, 132201, 25675, 53279}, // ITU-R Rec. 624-4 System B, G
    {104597, 132201, 25675, 53279}, // SMPTE 170M
    {117579, 136230, 16907, 35559}, // SMPTE 240M (1987)
    {0},                            // YCgCo
    {110013, 140363, 12277, 42626}, // Bt-2020-NCL
    {110013, 140363, 12277, 42626}, // Bt-2020-CL
};

// sws_getCoefficients()
const int32_t *coefficients(int colorspace)
{
    if (colorspace > 10 || colorspace < 0 || colorspace == 8) {
        colorspace = CsDefault;
    }
    return kCoefficients[colorspace];
}

// ROUNDED_DIV() of libavutil
int64_t roundedDiv(int64_t a, int64_t b)
{
    return (a >= 0 ? a + (b >> 1) : a - (b >> 1)) / b;
}

// roundToInt16() of yuv2rgb.c
uint16_t roundToInt16(int64_t f)
{
    const int r = int((f + (1 << 15)) >> 16);
    if (r < -0x7FFF) {
        return 0x8000;
    } else if (r > 0x7FFF) {
        return 0x7FFF;
    }
    return uint16_t(r);
}

constexpr int kRgb2YuvShift = 15;

} // namespace

int colorspaceTable(int mltColorspace)
{
    switch (mltColorspace) {
    case 170: // mlt_colorspace_smpte170m
    case 470: // mlt_colorspace_bt470bg
    case 601: // mlt_colorspace_bt601
        return CsItu601;
    case 240: // mlt_colorspace_smpte240m
        return CsSmpte240m;
    case 709: // mlt_colorspace_bt709
        return CsItu709;
    case 2020: // mlt_colorspace_bt2020_ncl
    case 2021: // mlt_colorspace_bt2020_cl
        return CsBt2020;
    default:
        return CsDefault;
    }
}

YuvToRgb yuvToRgb(int table, bool fullRange)
{
    const int32_t *inv = coefficients(table);
    const int64_t contrast = 1 << 16;
    const int64_t saturation = 1 << 16;
    const int64_t brightness = 0;
    int64_t crv = inv[0];
    int64_t cbu = inv[1];
    int64_t cgu = -inv[2];
    int64_t cgv = -inv[3];
    int64_t cy = 1 << 16;
    int64_t oy = 0;
    if (!fullRange) {
        cy = (cy * 255) / 219;
        oy = 16 << 16;
    } else {
        crv = (crv * 224) / 255;
        cbu = (cbu * 224) / 255;
        cgu = (cgu * 224) / 255;
        cgv = (cgv * 224) / 255;
    }
    cy = (cy * contrast) >> 16;
    crv = (crv * contrast * saturation) >> 32;
    cbu = (cbu * contrast * saturation) >> 32;
    cgu = (cgu * contrast * saturation) >> 32;
    cgv = (cgv * contrast * saturation) >> 32;
    oy -= 256 * brightness;

    YuvToRgb result;
    result.yCoeff = int16_t(roundToInt16(cy * (1 << 13)));
    result.yOffset = int16_t(roundToInt16(oy * (1 << 9)));
    result.v2r = int16_t(roundToInt16(crv * (1 << 13)));
    result.v2g = int16_t(roundToInt16(cgv * (1 << 13)));
    result.u2g = int16_t(roundToInt16(cgu * (1 << 13)));
    result.u2b = int16_t(roundToInt16(cbu * (1 << 13)));
    return result;
}

RgbToYuv rgbToYuv(int table)
{
    // fill_rgb2yuv_table(), which always computes the limited range table
    const int32_t *t = coefficients(table);
    int64_t vr = t[0];
    int64_t ub = t[1];
    int64_t ug = -t[2];
    int64_t vg = -t[3];
    const int64_t one = 65536;
    const int64_t cy = one * 255 / 219;
    const int64_t w = roundedDiv(one * one * ug, ub);
    const int64_t v = roundedDiv(one * one * vg, vr);
    const int64_t z = one * one - w - v;
    const int64_t cY = roundedDiv(cy * z, one);
    const int64_t cU = roundedDiv(ub * z, one);
    const int64_t cV = roundedDiv(vr * z, one);
    const int64_t scale = int64_t(1) << kRgb2YuvShift;

    RgbToYuv result;
    result.ry = int(-roundedDiv(scale * v, cY));
    result.gy = int(roundedDiv(scale * one * one, cY));
    result.by = int(-roundedDiv(scale * w, cY));
    result.ru = int(roundedDiv(scale * v, cU));
    result.gu = int(-roundedDiv(scale * one * one, cU));
    result.bu = int(roundedDiv(scale * (z + w), cU));
    result.rv = int(roundedDiv(scale * (v + z), cV));
    result.gv = int(-roundedDiv(scale * one * one, cV));
    result.bv = int(roundedDiv(scale * w, cV));

    const int32_t *defaults = coefficients(CsDefault);
    if (t[0] == defaults[0] && t[1] == defaults[1] && t[2] == defaults[2] && t[3] == defaults[3]) {
        result.by = int(0.114 * 219 / 255 * (1 << kRgb2YuvShift) + 0.5);
        result.bv = -int(0.081 * 224 / 255 * (1 << kRgb2YuvShift) + 0.5);
        result.bu = int(0.500 * 224 / 255 * (1 << kRgb2YuvShift) + 0.5);
        result.gy = int(0.587 * 219 / 255 * (1 << kRgb2YuvShift) + 0.5);
        result.gv = -int(0.419 * 224 / 255 * (1 << kRgb2YuvShift) + 0.5);
        result.gu = -int(0.331 * 224 / 255 * (1 << kRgb2YuvShift) + 0.5);
        result.ry = int(0.299 * 219 / 255 * (1 << kRgb2YuvShift) + 0.5);
        result.rv = int(0.500 * 224 / 255 * (1 << kRgb2YuvShift) + 0.5);
        result.ru = -int(0.169 * 224 / 255 * (1 << kRgb2YuvShift) + 0.5);
    }
    return result;
}

int localPosition(int chromaSubsample, int position)
{
    if (position == -1 || position <= -513) {
        position = (128 << chromaSubsample) - 128;
    }
    position += 128; // relative to ideal left edge
    return position >> chromaSubsample;
}

void pointSamples(int srcSize, int dstSize, int srcPos, int dstPos, std::vector<int> &samples)
{
    samples.resize(size_t(dstSize));
    // lumXInc / chrXInc of sws_init_context()
    const int xInc = int(((int64_t(srcSize) << 16) + (dstSize >> 1)) / dstSize);
    if (std::abs(xInc - 0x10000) < 10 && srcPos == dstPos) {
        // unscaled
        for (int i = 0; i < dstSize; ++i) {
            samples[size_t(i)] = i < srcSize ? i : srcSize - 1;
        }
        return;
    }
    int64_t xDstInSrc = ((dstPos * int64_t(xInc)) >> 8) - ((srcPos * 0x8000LL) >> 7);
    for (int i = 0; i < dstSize; ++i) {
        int xx = int((xDstInSrc + (1 << 15)) >> 16);
        // The border handling moves the single tap inside the source
        if (xx < 0) {
            xx = 0;
        } else if (xx > srcSize - 1) {
            xx = srcSize - 1;
        }
        samples[size_t(i)] = xx;
        xDstInSrc += xInc;
    }
}

void chroma420Samples(int width, int height, bool fields, std::vector<int> &columns, std::vector<int> &rows)
{
    // The chroma of a 4:2:0 source has half the size, rounded up. RGB output
    // with SWS_FULL_CHR_H_INT has no chroma subsampling.
    pointSamples((width + 1) >> 1, width, localPosition(1, -513), localPosition(0, -513), columns);
    if (!fields) {
        pointSamples((height + 1) >> 1, height, localPosition(1, -513), localPosition(0, -513), rows);
        return;
    }
    // Each field is converted as an image of half the height, whose chroma
    // lines are every other line of the chroma plane
    const int fieldHeight = height / 2;
    std::vector<int> fieldRows;
    pointSamples((fieldHeight + 1) >> 1, fieldHeight, localPosition(1, -513), localPosition(0, -513), fieldRows);
    rows.resize(size_t(height));
    for (int field = 0; field < 2; ++field) {
        for (int y = 0; y < fieldHeight; ++y) {
            rows[size_t(2 * y + field)] = 2 * fieldRows[size_t(y)] + field;
        }
    }
}

} // namespace sws
} // namespace gpufx
