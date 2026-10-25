/*
    SPDX-FileCopyrightText: 2026 ThioJoe
    SPDX-License-Identifier: GPL-3.0-only
*/

/*
 * CUDA kernels for the gpufx MLT module.
 *
 * Every kernel reproduces the arithmetic of the CPU filter it replaces, so that
 * switching between CPU and GPU processing does not change the rendered result.
 * Images are 8-bit RGBA (mlt_image_rgba), tightly packed, one 32-bit word per
 * pixel.
 *
 * Keep the kernel list in gpufx_kernel_list.h in sync with this file.
 */

#include "gpufx_device.h"

#define GPUFX_GLOBAL_X (int(blockIdx.x * blockDim.x + threadIdx.x))
#define GPUFX_GLOBAL_Y (int(blockIdx.y * blockDim.y + threadIdx.y))

/* ==========================================================================
 * Drop shadow (MLT "dropshadow" filter, which renders with
 * QGraphicsDropShadowEffect). See gpufx/src/effect_dropshadow.cpp for the
 * geometry, computed on the host.
 * ========================================================================== */

GPUFX_DEV gpufx_u32 gpufx_alpha_at(const gpufx_u32 *__restrict__ src, int w, int h, int x, int y)
{
    if (x < 0 || y < 0 || x >= w || y >= h) {
        return 0;
    }
    return src[size_t(y) * size_t(w) + size_t(x)] >> 24;
}

/* Builds the input of the blur: the alpha channel of the source placed in the
 * padded effect rectangle and shifted by the rounded shadow offset, so that
 * padded (x, y) shows the image at (x + ox, y + oy). When half is set the
 * result is downscaled by two like qt_halfScaled() does (average of averages,
 * truncating). */
GPUFX_KERNEL void gpufx_dropshadow_alpha(const gpufx_u32 *__restrict__ src, int w, int h, gpufx_u8 *__restrict__ dst, int dw, int dh, int ox, int oy, int half)
{
    const int x = GPUFX_GLOBAL_X;
    const int y = GPUFX_GLOBAL_Y;
    if (x >= dw || y >= dh) {
        return;
    }
    gpufx_u32 v;
    if (half) {
        const int sx = 2 * x + ox;
        const int sy = 2 * y + oy;
        const gpufx_u32 t = (gpufx_alpha_at(src, w, h, sx, sy) + gpufx_alpha_at(src, w, h, sx + 1, sy)) >> 1;
        const gpufx_u32 b = (gpufx_alpha_at(src, w, h, sx, sy + 1) + gpufx_alpha_at(src, w, h, sx + 1, sy + 1)) >> 1;
        v = (t + b) >> 1;
    } else {
        v = gpufx_alpha_at(src, w, h, x + ox, y + oy);
    }
    dst[size_t(y) * size_t(dw) + size_t(x)] = gpufx_u8(v);
}

/* Horizontal pass of Qt's expblur(): an exponential IIR filter run forward and
 * then backward over each row, with the filter state carried over. One thread
 * per row. */
GPUFX_KERNEL void gpufx_expblur_rows(gpufx_u8 *__restrict__ buf, int bw, int bh, int alpha)
{
    const int y = GPUFX_GLOBAL_X;
    if (y >= bh) {
        return;
    }
    gpufx_u8 *row = buf + size_t(y) * size_t(bw);
    int z = 0;
    for (int i = 0; i < bw; ++i) {
        row[i] = gpufx_expblur_step(z, row[i], alpha);
    }
    for (int i = bw - 2; i >= 0; --i) {
        row[i] = gpufx_expblur_step(z, row[i], alpha);
    }
}

/* Vertical pass of Qt's expblur(). Qt rotates the image by 270 degrees and
 * blurs the rows of the rotated copy, which makes the forward pass run from
 * the bottom of each column to the top. One thread per column. */
GPUFX_KERNEL void gpufx_expblur_cols(gpufx_u8 *__restrict__ buf, int bw, int bh, int alpha)
{
    const int x = GPUFX_GLOBAL_X;
    if (x >= bw) {
        return;
    }
    gpufx_u8 *col = buf + x;
    const size_t stride = size_t(bw);
    int z = 0;
    for (int i = bh - 1; i >= 0; --i) {
        col[size_t(i) * stride] = gpufx_expblur_step(z, col[size_t(i) * stride], alpha);
    }
    for (int i = 1; i < bh; ++i) {
        col[size_t(i) * stride] = gpufx_expblur_step(z, col[size_t(i) * stride], alpha);
    }
}

/* Smooth 2x upscale of the half resolution blur, as done by QPainter with
 * SmoothPixmapTransform (fetchTransformedBilinearARGB32PM_simple_scale_helper):
 * 16.16 fixed point sampling with the 1/65536 offset of QSpanData::setupMatrix,
 * 8-bit weights, vertical interpolation first, clamped edges. */
GPUFX_DEV gpufx_u32 gpufx_qt_upscale2(const gpufx_u8 *__restrict__ img, int iw, int ih, int x, int y)
{
    const int fy = 32768 * y - 16385;
    int y1 = fy >> 16;
    int y2;
    const gpufx_u32 disty = gpufx_u32(fy & 0xffff) >> 8;
    if (y1 < 0) {
        y1 = y2 = 0;
    } else if (y1 >= ih - 1) {
        y1 = y2 = ih - 1;
    } else {
        y2 = y1 + 1;
    }
    const int fx = 32768 * x - 16385;
    int x1 = fx >> 16;
    int x2;
    const gpufx_u32 distx = gpufx_u32(fx & 0xffff) >> 8;
    if (x1 < 0) {
        x1 = x2 = 0;
    } else if (x1 >= iw - 1) {
        x1 = x2 = iw - 1;
    } else {
        x2 = x1 + 1;
    }
    const gpufx_u8 *r1 = img + size_t(y1) * size_t(iw);
    const gpufx_u8 *r2 = img + size_t(y2) * size_t(iw);
    const gpufx_u32 left = (gpufx_u32(r1[x1]) * (256u - disty) + gpufx_u32(r2[x1]) * disty) >> 8;
    const gpufx_u32 right = (gpufx_u32(r1[x2]) * (256u - disty) + gpufx_u32(r2[x2]) * disty) >> 8;
    return (left * (256u - distx) + right * distx) >> 8;
}

/* Final composition of the drop shadow onto the image: the colored shadow
 * (SourceIn fill of the blurred alpha) is drawn over the image, then the
 * original image is drawn over the result. Both draws go through Qt's 16-bit
 * pipeline because the target QImage is RGBA8888 (straight alpha).
 * (pl, pt) is the top left corner of the padded effect rectangle, which is
 * where the blur buffer starts in image coordinates. */
GPUFX_KERNEL void gpufx_dropshadow_composite(const gpufx_u32 *__restrict__ src, gpufx_u32 *__restrict__ dst, int w, int h, const gpufx_u8 *__restrict__ blur,
                                             int bw, int bh, int pl, int pt, int half, gpufx_u32 color_pm)
{
    const int x = GPUFX_GLOBAL_X;
    const int y = GPUFX_GLOBAL_Y;
    if (x >= w || y >= h) {
        return;
    }
    const int px = x - pl;
    const int py = y - pt;
    gpufx_u32 shadow_alpha;
    if (half) {
        shadow_alpha = (px < 2 * bw && py < 2 * bh) ? gpufx_qt_upscale2(blur, bw, bh, px, py) : 0u;
    } else {
        shadow_alpha = blur[size_t(py) * size_t(bw) + size_t(px)];
    }
    const gpufx_u32 shadow = gpufx_byte_mul(color_pm, shadow_alpha);
    const size_t i = size_t(y) * size_t(w) + size_t(x);
    const gpufx_u32 p = src[i];
    const gpufx_u32 with_shadow = gpufx_store64(gpufx_source_over64(gpufx_expand_pm8(shadow), gpufx_fetch64(p)));
    dst[i] = gpufx_store64(gpufx_source_over64(gpufx_expand_pm8(gpufx_premultiply8(p)), gpufx_fetch64(with_shadow)));
}

/* Drop shadow with no blur and no offset: Qt only draws the source item over
 * the image (QGraphicsEffect::drawSource()). */
GPUFX_KERNEL void gpufx_over_self(const gpufx_u32 *__restrict__ src, gpufx_u32 *__restrict__ dst, int n)
{
    const int i = GPUFX_GLOBAL_X;
    if (i >= n) {
        return;
    }
    const gpufx_u32 p = src[i];
    dst[i] = gpufx_store64(gpufx_source_over64(gpufx_expand_pm8(gpufx_premultiply8(p)), gpufx_fetch64(p)));
}

/* ==========================================================================
 * Box blur (MLT core "box_blur" filter, mlt_image_box_blur()): separable
 * moving average over 2 * radius + 1 pixels with edge pixels repeated,
 * rounded to nearest. Each thread handles a run of GPUFX_BOX_RUN pixels.
 * ========================================================================== */

#define GPUFX_BOX_RUN 32

GPUFX_DEV gpufx_u32 gpufx_box_average(int sum, int diameter)
{
    // lrint(sum / diameter): the diameter is odd so there are no ties
    return gpufx_u32((2 * sum + diameter) / (2 * diameter));
}

GPUFX_DEV void gpufx_add_px(int *acc, gpufx_u32 p, int sign)
{
    acc[0] += sign * int(p & 0xffu);
    acc[1] += sign * int((p >> 8) & 0xffu);
    acc[2] += sign * int((p >> 16) & 0xffu);
    acc[3] += sign * int(p >> 24);
}

GPUFX_DEV gpufx_u32 gpufx_box_output(const int *acc, int diameter, gpufx_u32 original, int preserve_alpha)
{
    const gpufx_u32 a = preserve_alpha ? (original >> 24) : gpufx_box_average(acc[3], diameter);
    return gpufx_pack(gpufx_box_average(acc[0], diameter), gpufx_box_average(acc[1], diameter), gpufx_box_average(acc[2], diameter), a);
}

GPUFX_KERNEL void gpufx_box_blur_h(const gpufx_u32 *__restrict__ src, gpufx_u32 *__restrict__ dst, int w, int h, int radius, int preserve_alpha)
{
    const int x0 = GPUFX_GLOBAL_X * GPUFX_BOX_RUN;
    const int y = GPUFX_GLOBAL_Y;
    if (x0 >= w || y >= h) {
        return;
    }
    const gpufx_u32 *row = src + size_t(y) * size_t(w);
    gpufx_u32 *out = dst + size_t(y) * size_t(w);
    const int last = w - 1;
    const int r = gpufx_min(radius, w / 2);
    const int diameter = 2 * r + 1;
    int acc[4] = {0, 0, 0, 0};
    for (int k = -r; k <= r; ++k) {
        gpufx_add_px(acc, row[gpufx_clamp(x0 + k, 0, last)], 1);
    }
    const int x1 = gpufx_min(x0 + GPUFX_BOX_RUN, w);
    for (int x = x0; x < x1; ++x) {
        out[x] = gpufx_box_output(acc, diameter, row[x], preserve_alpha);
        gpufx_add_px(acc, row[gpufx_clamp(x + r + 1, 0, last)], 1);
        gpufx_add_px(acc, row[gpufx_clamp(x - r, 0, last)], -1);
    }
}

GPUFX_KERNEL void gpufx_box_blur_v(const gpufx_u32 *__restrict__ src, gpufx_u32 *__restrict__ dst, int w, int h, int radius, int preserve_alpha)
{
    const int x = GPUFX_GLOBAL_X;
    const int y0 = GPUFX_GLOBAL_Y * GPUFX_BOX_RUN;
    if (x >= w || y0 >= h) {
        return;
    }
    const size_t stride = size_t(w);
    const gpufx_u32 *col = src + x;
    gpufx_u32 *out = dst + x;
    const int last = h - 1;
    const int r = gpufx_min(radius, h / 2);
    const int diameter = 2 * r + 1;
    int acc[4] = {0, 0, 0, 0};
    for (int k = -r; k <= r; ++k) {
        gpufx_add_px(acc, col[size_t(gpufx_clamp(y0 + k, 0, last)) * stride], 1);
    }
    const int y1 = gpufx_min(y0 + GPUFX_BOX_RUN, h);
    for (int y = y0; y < y1; ++y) {
        out[size_t(y) * stride] = gpufx_box_output(acc, diameter, col[size_t(y) * stride], preserve_alpha);
        gpufx_add_px(acc, col[size_t(gpufx_clamp(y + r + 1, 0, last)) * stride], 1);
        gpufx_add_px(acc, col[size_t(gpufx_clamp(y - r, 0, last)) * stride], -1);
    }
}

/* ==========================================================================
 * "Box Blur" of the MLT kdenlive module ("boxblur" filter). The CPU version
 * reads a summed area table at clamped corners, which amounts to summing the
 * pixels in (clamp(x - bw), clamp(x + bw)] x (clamp(y - bh), clamp(y + bh)]
 * and scaling by 1 / (2bw * 2bh) in float, truncated. Here the same sums are
 * computed in two exact integer passes.
 * ========================================================================== */

/* Horizontal sums, one int per channel (sums is 4 ints per pixel). */
GPUFX_KERNEL void gpufx_rect_blur_h(const gpufx_u32 *__restrict__ src, int *__restrict__ sums, int w, int h, int bw)
{
    const int x0 = GPUFX_GLOBAL_X * GPUFX_BOX_RUN;
    const int y = GPUFX_GLOBAL_Y;
    if (x0 >= w || y >= h) {
        return;
    }
    const gpufx_u32 *row = src + size_t(y) * size_t(w);
    int *out = sums + 4 * size_t(y) * size_t(w);
    const int last = w - 1;
    int lo = gpufx_clamp(x0 - bw, 0, last);
    int hi = gpufx_clamp(x0 + bw, 0, last);
    int acc[4] = {0, 0, 0, 0};
    for (int i = lo + 1; i <= hi; ++i) {
        gpufx_add_px(acc, row[i], 1);
    }
    const int x1 = gpufx_min(x0 + GPUFX_BOX_RUN, w);
    for (int x = x0; x < x1; ++x) {
        int *o = out + 4 * size_t(x);
        o[0] = acc[0];
        o[1] = acc[1];
        o[2] = acc[2];
        o[3] = acc[3];
        const int nlo = gpufx_clamp(x + 1 - bw, 0, last);
        const int nhi = gpufx_clamp(x + 1 + bw, 0, last);
        if (nhi > hi) {
            gpufx_add_px(acc, row[nhi], 1);
        }
        if (nlo > lo) {
            gpufx_add_px(acc, row[nlo], -1);
        }
        lo = nlo;
        hi = nhi;
    }
}

/* Vertical sums of the horizontal sums, then the CPU filter's float scaling. */
GPUFX_KERNEL void gpufx_rect_blur_v(const int *__restrict__ sums, gpufx_u32 *__restrict__ dst, int w, int h, int bh, float mul)
{
    const int x = GPUFX_GLOBAL_X;
    const int y0 = GPUFX_GLOBAL_Y * GPUFX_BOX_RUN;
    if (x >= w || y0 >= h) {
        return;
    }
    const size_t stride = 4 * size_t(w);
    const int *col = sums + 4 * size_t(x);
    const int last = h - 1;
    int lo = gpufx_clamp(y0 - bh, 0, last);
    int hi = gpufx_clamp(y0 + bh, 0, last);
    int acc[4] = {0, 0, 0, 0};
    for (int j = lo + 1; j <= hi; ++j) {
        const int *s = col + size_t(j) * stride;
        acc[0] += s[0];
        acc[1] += s[1];
        acc[2] += s[2];
        acc[3] += s[3];
    }
    const int y1 = gpufx_min(y0 + GPUFX_BOX_RUN, h);
    for (int y = y0; y < y1; ++y) {
        gpufx_u32 c[4];
        for (int k = 0; k < 4; ++k) {
            c[k] = gpufx_f2u_rz(gpufx_fmul_rn(gpufx_i2f_rn(acc[k]), mul)) & 0xffu;
        }
        dst[size_t(y) * size_t(w) + size_t(x)] = gpufx_pack(c[0], c[1], c[2], c[3]);
        const int nlo = gpufx_clamp(y + 1 - bh, 0, last);
        const int nhi = gpufx_clamp(y + 1 + bh, 0, last);
        if (nhi > hi) {
            const int *s = col + size_t(nhi) * stride;
            acc[0] += s[0];
            acc[1] += s[1];
            acc[2] += s[2];
            acc[3] += s[3];
        }
        if (nlo > lo) {
            const int *s = col + size_t(nlo) * stride;
            acc[0] -= s[0];
            acc[1] -= s[1];
            acc[2] -= s[2];
            acc[3] -= s[3];
        }
        lo = nlo;
        hi = nhi;
    }
}

/* ==========================================================================
 * Per channel lookup tables (brightness/fades, lift/gamma/gain). The tables
 * are computed on the host with the CPU filter's own formulas, which keeps
 * the results identical. lut holds 4 x 256 entries: R, G, B, A.
 * ========================================================================== */

GPUFX_KERNEL void gpufx_lut_rgba(const gpufx_u32 *__restrict__ src, gpufx_u32 *__restrict__ dst, int n, const gpufx_u8 *__restrict__ lut)
{
    const int i = GPUFX_GLOBAL_X;
    if (i >= n) {
        return;
    }
    const gpufx_u32 p = src[i];
    dst[i] = gpufx_pack(lut[p & 0xffu], lut[256 + ((p >> 8) & 0xffu)], lut[512 + ((p >> 16) & 0xffu)], lut[768 + (p >> 24)]);
}

/* ==========================================================================
 * frei0r square blur (include/frei0r/blur.h), used by the "Square Blur",
 * "Glow" and "Soft Glow" effects. The CPU version reads a summed area table:
 * every channel, alpha included, is the sum of the pixels in
 * [x - k, x + k] x [y - k, y + k] clipped to the image, divided by the number
 * of pixels in the clipped rectangle (truncated). The sums are unsigned 32-bit
 * like the table cells, so they wrap around the same way on huge images.
 * Each thread handles a run of pixels, longer runs for larger kernels.
 * ========================================================================== */

GPUFX_DEV void gpufx_add_px_u(gpufx_u32 *acc, gpufx_u32 p)
{
    acc[0] += p & 0xffu;
    acc[1] += (p >> 8) & 0xffu;
    acc[2] += (p >> 16) & 0xffu;
    acc[3] += p >> 24;
}

GPUFX_DEV void gpufx_sub_px_u(gpufx_u32 *acc, gpufx_u32 p)
{
    acc[0] -= p & 0xffu;
    acc[1] -= (p >> 8) & 0xffu;
    acc[2] -= (p >> 16) & 0xffu;
    acc[3] -= p >> 24;
}

/* Horizontal sums, 4 x u32 per pixel. */
GPUFX_KERNEL void gpufx_square_blur_h(const gpufx_u32 *__restrict__ src, gpufx_u32 *__restrict__ sums, int w, int h, int k, int run)
{
    const int x0 = GPUFX_GLOBAL_X * run;
    const int y = GPUFX_GLOBAL_Y;
    if (x0 >= w || y >= h) {
        return;
    }
    const gpufx_u32 *row = src + size_t(y) * size_t(w);
    gpufx_u32 *out = sums + 4 * size_t(y) * size_t(w);
    const int last = w - 1;
    gpufx_u32 acc[4] = {0, 0, 0, 0};
    const int hi = gpufx_min(x0 + k, last);
    for (int i = gpufx_max(x0 - k, 0); i <= hi; ++i) {
        gpufx_add_px_u(acc, row[i]);
    }
    const int x1 = gpufx_min(x0 + run, w);
    for (int x = x0; x < x1; ++x) {
        gpufx_u32 *o = out + 4 * size_t(x);
        o[0] = acc[0];
        o[1] = acc[1];
        o[2] = acc[2];
        o[3] = acc[3];
        if (x + 1 + k <= last) {
            gpufx_add_px_u(acc, row[x + 1 + k]);
        }
        if (x - k >= 0) {
            gpufx_sub_px_u(acc, row[x - k]);
        }
    }
}

// frei0r's INT_MULT(): a * b / 255 rounded
GPUFX_DEV gpufx_u32 gpufx_int_mult(gpufx_u32 a, gpufx_u32 b)
{
    const gpufx_u32 t = a * b + 0x80u;
    return ((t >> 8) + t) >> 8;
}

#define GPUFX_SQUARE_BLUR_PLAIN 0
#define GPUFX_SQUARE_BLUR_GLOW 1
#define GPUFX_SQUARE_BLUR_SOFTGLOW_SCREEN 2
#define GPUFX_SQUARE_BLUR_SOFTGLOW_OVERLAY 3
#define GPUFX_SQUARE_BLUR_SOFTGLOW_ADD 4

/* What the effects do with the blurred pixel b and the input pixel s. */
GPUFX_DEV gpufx_u32 gpufx_square_blur_output(const gpufx_u32 *b, gpufx_u32 s, int mode)
{
    gpufx_u32 c[4];
    for (int i = 0; i < 4; ++i) {
        const gpufx_u32 sc = (s >> (8 * i)) & 0xffu;
        switch (mode) {
        case GPUFX_SQUARE_BLUR_GLOW: // glow.c, alpha included
            c[i] = 255u - ((255u - sc) * (255u - b[i])) / 255u;
            break;
        case GPUFX_SQUARE_BLUR_SOFTGLOW_SCREEN: // softglow.c screen()
            c[i] = 255u - gpufx_int_mult(255u - b[i], 255u - sc);
            break;
        case GPUFX_SQUARE_BLUR_SOFTGLOW_OVERLAY: // softglow.c overlay()
            c[i] = gpufx_int_mult(b[i], b[i] + gpufx_int_mult(2u * sc, 255u - b[i]));
            break;
        case GPUFX_SQUARE_BLUR_SOFTGLOW_ADD: // softglow.c add()
            c[i] = b[i] + sc > 255u ? 255u : b[i] + sc;
            break;
        default:
            c[i] = b[i];
            break;
        }
        c[i] &= 0xffu;
    }
    if (mode >= GPUFX_SQUARE_BLUR_SOFTGLOW_SCREEN) {
        // The soft glow blends keep the smaller alpha
        const gpufx_u32 sa = s >> 24;
        c[3] = b[3] < sa ? b[3] : sa;
    }
    return gpufx_pack(c[0], c[1], c[2], c[3]);
}

/* Vertical sums of the horizontal sums, average, then the effect's blend with
 * the input image src (see gpufx_square_blur_output). */
GPUFX_KERNEL void gpufx_square_blur_v(const gpufx_u32 *__restrict__ sums, const gpufx_u32 *__restrict__ src, gpufx_u32 *__restrict__ dst, int w, int h, int k,
                                      int run, int mode)
{
    const int x = GPUFX_GLOBAL_X;
    const int y0 = GPUFX_GLOBAL_Y * run;
    if (x >= w || y0 >= h) {
        return;
    }
    const size_t stride = 4 * size_t(w);
    const gpufx_u32 *col = sums + 4 * size_t(x);
    const int last = h - 1;
    gpufx_u32 acc[4] = {0, 0, 0, 0};
    const int hi = gpufx_min(y0 + k, last);
    for (int j = gpufx_max(y0 - k, 0); j <= hi; ++j) {
        const gpufx_u32 *s = col + size_t(j) * stride;
        acc[0] += s[0];
        acc[1] += s[1];
        acc[2] += s[2];
        acc[3] += s[3];
    }
    const gpufx_u32 columns = gpufx_u32(gpufx_min(x + k, w - 1) - gpufx_max(x - k, 0) + 1);
    const int y1 = gpufx_min(y0 + run, h);
    for (int y = y0; y < y1; ++y) {
        const gpufx_u32 area = columns * gpufx_u32(gpufx_min(y + k, last) - gpufx_max(y - k, 0) + 1);
        gpufx_u32 b[4];
        for (int i = 0; i < 4; ++i) {
            b[i] = (acc[i] / area) & 0xffu;
        }
        const size_t index = size_t(y) * size_t(w) + size_t(x);
        dst[index] = gpufx_square_blur_output(b, src[index], mode);
        if (y + 1 + k <= last) {
            const gpufx_u32 *s = col + size_t(y + 1 + k) * stride;
            acc[0] += s[0];
            acc[1] += s[1];
            acc[2] += s[2];
            acc[3] += s[3];
        }
        if (y - k >= 0) {
            const gpufx_u32 *s = col + size_t(y - k) * stride;
            acc[0] -= s[0];
            acc[1] -= s[1];
            acc[2] -= s[2];
            acc[3] -= s[3];
        }
    }
}

/* First step of frei0r's "Soft Glow": the lightness of every pixel (GIMP's
 * (max + min) / 2, rounded) through the sigmoidal transfer, which is
 * precomputed on the host in curve. Alpha is kept. */
GPUFX_KERNEL void gpufx_softglow_highlights(const gpufx_u32 *__restrict__ src, gpufx_u32 *__restrict__ dst, int n, const gpufx_u8 *__restrict__ curve)
{
    const int i = GPUFX_GLOBAL_X;
    if (i >= n) {
        return;
    }
    const gpufx_u32 p = src[i];
    const int r = int(p & 0xffu);
    const int g = int((p >> 8) & 0xffu);
    const int b = int((p >> 16) & 0xffu);
    const int high = gpufx_max(r, gpufx_max(g, b));
    const int low = gpufx_min(r, gpufx_min(g, b));
    const gpufx_u32 v = curve[(high + low + 1) >> 1];
    dst[i] = v | (v << 8) | (v << 16) | (p & 0xff000000u);
}

/* ==========================================================================
 * frei0r "Sharpness" (Kdenlive's "Sharpen"), MPlayer's unsharp mask. The
 * blur is a cascade of [1 1] filters, which amounts to binomial weights over
 * 2 * steps + 1 pixels with edge pixels repeated, here in two separable
 * passes with the same exact integer sums. Red, green and blue are sharpened,
 * alpha is kept.
 * ========================================================================== */

#define GPUFX_UNSHARP_MAX_STEPS 6

GPUFX_DEV void gpufx_binomial(int *weights, int n)
{
    weights[0] = 1;
    for (int i = 0; i < n; ++i) {
        weights[i + 1] = weights[i] * (n - i) / (i + 1);
    }
}

/* Horizontal weighted sums of red, green and blue, 4 x u32 per pixel. */
GPUFX_KERNEL void gpufx_unsharp_h(const gpufx_u32 *__restrict__ src, gpufx_u32 *__restrict__ sums, int w, int h, int steps)
{
    const int x = GPUFX_GLOBAL_X;
    const int y = GPUFX_GLOBAL_Y;
    if (x >= w || y >= h) {
        return;
    }
    int weights[2 * GPUFX_UNSHARP_MAX_STEPS + 1];
    gpufx_binomial(weights, 2 * steps);
    const gpufx_u32 *row = src + size_t(y) * size_t(w);
    gpufx_u32 acc[3] = {0, 0, 0};
    for (int d = -steps; d <= steps; ++d) {
        const gpufx_u32 p = row[gpufx_clamp(x + d, 0, w - 1)];
        const gpufx_u32 weight = gpufx_u32(weights[d + steps]);
        acc[0] += weight * (p & 0xffu);
        acc[1] += weight * ((p >> 8) & 0xffu);
        acc[2] += weight * ((p >> 16) & 0xffu);
    }
    gpufx_u32 *o = sums + 4 * (size_t(y) * size_t(w) + size_t(x));
    o[0] = acc[0];
    o[1] = acc[1];
    o[2] = acc[2];
    o[3] = 0;
}

/* Vertical weighted sums, then unsharp(): the scaled difference between the
 * pixel and its blurred value is added to the pixel (amount in 16.16). */
GPUFX_KERNEL void gpufx_unsharp_v(const gpufx_u32 *__restrict__ sums, const gpufx_u32 *__restrict__ src, gpufx_u32 *__restrict__ dst, int w, int h, int steps,
                                  int amount)
{
    const int x = GPUFX_GLOBAL_X;
    const int y = GPUFX_GLOBAL_Y;
    if (x >= w || y >= h) {
        return;
    }
    int weights[2 * GPUFX_UNSHARP_MAX_STEPS + 1];
    gpufx_binomial(weights, 2 * steps);
    const gpufx_u32 *col = sums + 4 * size_t(x);
    gpufx_u32 acc[3] = {0, 0, 0};
    for (int d = -steps; d <= steps; ++d) {
        const gpufx_u32 *s = col + 4 * size_t(gpufx_clamp(y + d, 0, h - 1)) * size_t(w);
        const gpufx_u32 weight = gpufx_u32(weights[d + steps]);
        acc[0] += weight * s[0];
        acc[1] += weight * s[1];
        acc[2] += weight * s[2];
    }
    const int scalebits = 4 * steps;
    const gpufx_u32 halfscale = 1u << (scalebits - 1);
    const size_t index = size_t(y) * size_t(w) + size_t(x);
    const gpufx_u32 p = src[index];
    gpufx_u32 c[3];
    for (int i = 0; i < 3; ++i) {
        const int value = int((p >> (8 * i)) & 0xffu);
        const int blurred = int((acc[i] + halfscale) >> scalebits);
        const int res = value + (((value - blurred) * amount) >> 16);
        c[i] = gpufx_u32(gpufx_clamp(res, 0, 255));
    }
    dst[index] = gpufx_pack(c[0], c[1], c[2], p >> 24);
}

/* ==========================================================================
 * QPainter::drawImage() with a transform onto an RGBA8888 image, as used by
 * MLT's "qtblend" filter (Kdenlive's Transform). The host ports Qt's
 * rasterizer and passes, for every destination row, the spans with their
 * coverage and the runs that Qt fetches, blends and stores in one call (see
 * gpufx/src/qt_raster.cpp). Each thread then redoes for its pixel what Qt's
 * 16-bit pipeline does, including the groups of 4 pixels its SSE4 format
 * conversions work on: every thread computes the whole group of its pixel.
 *
 * under: the destination content before drawing, or null for a destination
 *        filled with background
 * rows: per destination row, first span, span count, first run, run count
 * spans: x, length, coverage
 * runs: x, length, then the source position of the first pixel: 16.16 fixed
 *       point for the transformed modes, integer pixel for untransformed
 * ========================================================================== */

#define GPUFX_QT_UNTRANSFORMED 0
#define GPUFX_QT_NEAREST 1
#define GPUFX_QT_BILINEAR 2

GPUFX_DEV GpufxRgba64 gpufx_qt_expand(gpufx_u32 p)
{
    GpufxRgba64 v;
    v.r = (p & 0xffu) * 257u;
    v.g = ((p >> 8) & 0xffu) * 257u;
    v.b = ((p >> 16) & 0xffu) * 257u;
    v.a = (p >> 24) * 257u;
    return v;
}

// QRgba64::fromArgb32().premultiplied(): the scalar tail of the conversions
GPUFX_DEV GpufxRgba64 gpufx_qt_premultiply_exact(gpufx_u32 p)
{
    GpufxRgba64 v = gpufx_qt_expand(p);
    if (v.a == 65535u) {
        return v;
    }
    if (v.a == 0) {
        v.r = v.g = v.b = 0;
        return v;
    }
    v.r = gpufx_div65535(v.r * v.a);
    v.g = gpufx_div65535(v.g * v.a);
    v.b = gpufx_div65535(v.b * v.a);
    return v;
}

// convertARGBToRGBA64PM_sse4 on a group that is neither all transparent nor all opaque
GPUFX_DEV GpufxRgba64 gpufx_qt_premultiply_simd(gpufx_u32 p)
{
    GpufxRgba64 v = gpufx_qt_expand(p);
    gpufx_u32 x = (v.r * v.a) >> 16;
    v.r = x + (x >> 15);
    x = (v.g * v.a) >> 16;
    v.g = x + (x >> 15);
    x = (v.b * v.a) >> 16;
    v.b = x + (x >> 15);
    return v;
}

/* Converts 4 RGBA8888 pixels to 16-bit premultiplied like
 * convertARGBToRGBA64PM_sse4: a full group of 4 is converted at once, the
 * pixels of a shorter group are the scalar tail of the call (entries past the
 * tail are converted too, they are not used). */
GPUFX_DEV void gpufx_qt_convert(const gpufx_u32 *p, GpufxRgba64 *out, bool full)
{
    const gpufx_u32 alphas = (p[0] >> 24) | (p[1] >> 24) | (p[2] >> 24) | (p[3] >> 24);
    const gpufx_u32 opaque = (p[0] >> 24) & (p[1] >> 24) & (p[2] >> 24) & (p[3] >> 24);
    GPUFX_UNROLL
    for (int k = 0; k < 4; ++k) {
        if (!full) {
            out[k] = gpufx_qt_premultiply_exact(p[k]);
        } else if (alphas == 0) {
            out[k].r = out[k].g = out[k].b = out[k].a = 0;
        } else if (opaque == 255u) {
            out[k] = gpufx_qt_expand(p[k]);
        } else {
            out[k] = gpufx_qt_premultiply_simd(p[k]);
        }
    }
}

GPUFX_DEV gpufx_u32 gpufx_mulhi16(gpufx_u32 a, gpufx_u32 b)
{
    return (a * b) >> 16;
}

/* interpolate_4_pixels_rgb64() (SSE2), 16-bit weights */
GPUFX_DEV GpufxRgba64 gpufx_qt_interpolate(GpufxRgba64 tl, GpufxRgba64 tr, GpufxRgba64 bl, GpufxRgba64 br, gpufx_u32 distx, gpufx_u32 disty)
{
    if (disty) {
        const gpufx_u32 idy = 0x10000u - disty;
        tl.r = (gpufx_mulhi16(tl.r, idy) + gpufx_mulhi16(bl.r, disty)) & 0xffffu;
        tl.g = (gpufx_mulhi16(tl.g, idy) + gpufx_mulhi16(bl.g, disty)) & 0xffffu;
        tl.b = (gpufx_mulhi16(tl.b, idy) + gpufx_mulhi16(bl.b, disty)) & 0xffffu;
        tl.a = (gpufx_mulhi16(tl.a, idy) + gpufx_mulhi16(bl.a, disty)) & 0xffffu;
        tr.r = (gpufx_mulhi16(tr.r, idy) + gpufx_mulhi16(br.r, disty)) & 0xffffu;
        tr.g = (gpufx_mulhi16(tr.g, idy) + gpufx_mulhi16(br.g, disty)) & 0xffffu;
        tr.b = (gpufx_mulhi16(tr.b, idy) + gpufx_mulhi16(br.b, disty)) & 0xffffu;
        tr.a = (gpufx_mulhi16(tr.a, idy) + gpufx_mulhi16(br.a, disty)) & 0xffffu;
    }
    if (distx) {
        const gpufx_u32 idx = 0x10000u - distx;
        tl.r = (gpufx_mulhi16(tl.r, idx) + gpufx_mulhi16(tr.r, distx)) & 0xffffu;
        tl.g = (gpufx_mulhi16(tl.g, idx) + gpufx_mulhi16(tr.g, distx)) & 0xffffu;
        tl.b = (gpufx_mulhi16(tl.b, idx) + gpufx_mulhi16(tr.b, distx)) & 0xffffu;
        tl.a = (gpufx_mulhi16(tl.a, idx) + gpufx_mulhi16(tr.a, distx)) & 0xffffu;
    }
    return tl;
}

GPUFX_DEV GpufxRgba64 gpufx_qt_multiply(GpufxRgba64 v, gpufx_u32 a)
{
    v.r = gpufx_div65535(v.r * a);
    v.g = gpufx_div65535(v.g * a);
    v.b = gpufx_div65535(v.b * a);
    v.a = gpufx_div65535(v.a * a);
    return v;
}

/* comp_func_SourceOver_rgb64() with const_alpha = coverage * opacity */
GPUFX_DEV GpufxRgba64 gpufx_qt_source_over(GpufxRgba64 s, GpufxRgba64 d, gpufx_u32 constAlpha)
{
    if (constAlpha == 255u) {
        if (s.a == 65535u) {
            return s;
        }
        if (s.a == 0) {
            return d;
        }
    } else {
        s = gpufx_qt_multiply(s, constAlpha * 257u);
    }
    const GpufxRgba64 m = gpufx_qt_multiply(d, 65535u - s.a);
    s.r = (s.r + m.r) & 0xffffu;
    s.g = (s.g + m.g) & 0xffffu;
    s.b = (s.b + m.b) & 0xffffu;
    s.a = (s.a + m.a) & 0xffffu;
    return s;
}

// _mm_packus_epi32 then _mm_packus_epi16 of a rounded float
GPUFX_DEV gpufx_u32 gpufx_qt_pack_float(float v)
{
    int i = gpufx_f2i_rn(v);
    i = gpufx_clamp(i, 0, 65535);
    if (i > 32767) {
        return 0; // negative as a signed 16-bit value
    }
    return gpufx_u32(gpufx_min(i, 255));
}

/* convertARGBFromRGBA64PM_sse4<RGBA>: stores a group of 4 pixels, or a
 * scalar tail of fewer pixels (qConvertRgba64ToRgb32_sse4). */
GPUFX_DEV gpufx_u32 gpufx_qt_store(const GpufxRgba64 *v, int count, int k)
{
    // Constant indices keep the group in registers
    const GpufxRgba64 p = k == 0 ? v[0] : (k == 1 ? v[1] : (k == 2 ? v[2] : v[3]));
    if (count < 4) {
        if (p.a == 0) {
            return 0;
        }
        gpufx_u32 r = p.r;
        gpufx_u32 g = p.g;
        gpufx_u32 b = p.b;
        if (p.a != 65535u) {
            const float via = gpufx_fdiv_rn(65535.0f, gpufx_i2f_rn(int(p.a)));
            r = gpufx_u32(gpufx_clamp(gpufx_f2i_rn(gpufx_fmul_rn(gpufx_i2f_rn(int(r)), via)), 0, 65535));
            g = gpufx_u32(gpufx_clamp(gpufx_f2i_rn(gpufx_fmul_rn(gpufx_i2f_rn(int(g)), via)), 0, 65535));
            b = gpufx_u32(gpufx_clamp(gpufx_f2i_rn(gpufx_fmul_rn(gpufx_i2f_rn(int(b)), via)), 0, 65535));
        }
        return gpufx_pack(gpufx_div257(r), gpufx_div257(g), gpufx_div257(b), gpufx_div257(p.a));
    }
    const bool transparent = (v[0].a | v[1].a | v[2].a | v[3].a) == 0;
    const bool opaque = (v[0].a & v[1].a & v[2].a & v[3].a) == 65535u;
    if (transparent) {
        return 0;
    }
    if (opaque) {
        return gpufx_pack(gpufx_div257(p.r), gpufx_div257(p.g), gpufx_div257(p.b), gpufx_div257(p.a));
    }
    if (p.a == 0) {
        return 0;
    }
    const float ia = gpufx_fdiv_rn(255.0f, gpufx_i2f_rn(int(p.a)));
    return gpufx_pack(gpufx_qt_pack_float(gpufx_fmul_rn(gpufx_i2f_rn(int(p.r)), ia)), gpufx_qt_pack_float(gpufx_fmul_rn(gpufx_i2f_rn(int(p.g)), ia)),
                      gpufx_qt_pack_float(gpufx_fmul_rn(gpufx_i2f_rn(int(p.b)), ia)), gpufx_div257(p.a));
}

GPUFX_DEV gpufx_u32 gpufx_qt_coverage(const int *__restrict__ spans, int first, int count, int x)
{
    int lo = 0;
    int hi = count - 1;
    while (lo <= hi) {
        const int mid = (lo + hi) >> 1;
        const int *s = spans + 3 * (first + mid);
        if (x < s[0]) {
            hi = mid - 1;
        } else if (x >= s[0] + s[1]) {
            lo = mid + 1;
        } else {
            return gpufx_u32(s[2]);
        }
    }
    return 0;
}

// fetchTransformedBilinear_pixelBounds<BlendTransformedBilinear>
GPUFX_DEV void gpufx_qt_bounds(int max, int &v1, int &v2)
{
    if (v1 < 0) {
        v2 = v1 = 0;
    } else if (v1 >= max) {
        v2 = v1 = max;
    } else {
        v2 = v1 + 1;
    }
}

GPUFX_KERNEL void gpufx_qt_draw_image(const gpufx_u32 *__restrict__ src, int sw, int sh, const gpufx_u32 *__restrict__ under, gpufx_u32 background,
                                      gpufx_u32 *__restrict__ dst, int dw, int dh, const int *__restrict__ rows, const int *__restrict__ spans,
                                      const int *__restrict__ runs, int mode, int fdx, int fdy, int opacity)
{
    const int x = GPUFX_GLOBAL_X;
    const int y = GPUFX_GLOBAL_Y;
    if (x >= dw || y >= dh) {
        return;
    }
    const int *row = rows + 4 * y;
    const int *run = 0;
    for (int r = 0; r < row[3]; ++r) {
        const int *candidate = runs + 4 * (row[2] + r);
        if (x >= candidate[0] && x < candidate[0] + candidate[1]) {
            run = candidate;
            break;
        }
    }
    const size_t index = size_t(y) * size_t(dw) + size_t(x);
    if (!run) {
        dst[index] = under ? under[index] : background;
        return;
    }
    const int start = run[0];
    const int length = run[1];
    const int i = x - start;
    const int group = i & ~3;
    const int count = gpufx_min(4, length - group);

    // Source pixels of the group in 16-bit premultiplied. Entries past the end
    // of the run repeat its last pixel and are not used.
    GpufxRgba64 source[4];
    if (mode == GPUFX_QT_BILINEAR) {
        gpufx_u32 tl[4];
        gpufx_u32 tr[4];
        gpufx_u32 bl[4];
        gpufx_u32 br[4];
        gpufx_u32 distx[4];
        gpufx_u32 disty[4];
        GPUFX_UNROLL
        for (int k = 0; k < 4; ++k) {
            const int j = gpufx_min(group + k, length - 1);
            // Qt adds fdx for every pixel of the run, the host checked that it stays in range
            const int fx = int(gpufx_i64(run[2]) + gpufx_i64(j) * fdx);
            const int fy = int(gpufx_i64(run[3]) + gpufx_i64(j) * fdy);
            int x1 = fx >> 16;
            int x2;
            int y1 = fy >> 16;
            int y2;
            gpufx_qt_bounds(sw - 1, x1, x2);
            gpufx_qt_bounds(sh - 1, y1, y2);
            const gpufx_u32 *s1 = src + size_t(y1) * size_t(sw);
            const gpufx_u32 *s2 = src + size_t(y2) * size_t(sw);
            tl[k] = s1[x1];
            tr[k] = s1[x2];
            bl[k] = s2[x1];
            br[k] = s2[x2];
            distx[k] = gpufx_u32(fx) & 0xffffu;
            disty[k] = gpufx_u32(fy) & 0xffffu;
        }
        // Qt converts the left and right texels of two output pixels as one group of 4
        GpufxRgba64 ctl[4];
        GpufxRgba64 ctr[4];
        GpufxRgba64 cbl[4];
        GpufxRgba64 cbr[4];
        GPUFX_UNROLL
        for (int k = 0; k < 4; k += 2) {
            const bool full = count - k >= 2;
            const gpufx_u32 top[4] = {tl[k], tr[k], tl[k + 1], tr[k + 1]};
            const gpufx_u32 bottom[4] = {bl[k], br[k], bl[k + 1], br[k + 1]};
            GpufxRgba64 ct[4];
            GpufxRgba64 cb[4];
            gpufx_qt_convert(top, ct, full);
            gpufx_qt_convert(bottom, cb, full);
            ctl[k] = ct[0];
            ctr[k] = ct[1];
            ctl[k + 1] = ct[2];
            ctr[k + 1] = ct[3];
            cbl[k] = cb[0];
            cbr[k] = cb[1];
            cbl[k + 1] = cb[2];
            cbr[k + 1] = cb[3];
        }
        GPUFX_UNROLL
        for (int k = 0; k < 4; ++k) {
            source[k] = gpufx_qt_interpolate(ctl[k], ctr[k], cbl[k], cbr[k], distx[k], disty[k]);
        }
    } else {
        gpufx_u32 texels[4];
        GPUFX_UNROLL
        for (int k = 0; k < 4; ++k) {
            const int j = gpufx_min(group + k, length - 1);
            int px;
            int py;
            if (mode == GPUFX_QT_NEAREST) {
                px = gpufx_clamp(int(gpufx_i64(run[2]) + gpufx_i64(j) * fdx) >> 16, 0, sw - 1);
                py = gpufx_clamp(int(gpufx_i64(run[3]) + gpufx_i64(j) * fdy) >> 16, 0, sh - 1);
            } else {
                px = run[2] + j;
                py = run[3];
            }
            texels[k] = src[size_t(py) * size_t(sw) + size_t(px)];
        }
        gpufx_qt_convert(texels, source, count == 4);
    }

    // The destination content: the image below or the background
    gpufx_u32 targets[4];
    GPUFX_UNROLL
    for (int k = 0; k < 4; ++k) {
        targets[k] = under ? under[size_t(y) * size_t(dw) + size_t(start + gpufx_min(group + k, length - 1))] : background;
    }
    GpufxRgba64 dest[4];
    gpufx_qt_convert(targets, dest, count == 4);

    GpufxRgba64 blended[4];
    GPUFX_UNROLL
    for (int k = 0; k < 4; ++k) {
        const gpufx_u32 coverage = gpufx_qt_coverage(spans, row[0], row[1], start + gpufx_min(group + k, length - 1));
        blended[k] = gpufx_qt_source_over(source[k], dest[k], (coverage * gpufx_u32(opacity)) >> 8);
    }
    dst[index] = gpufx_qt_store(blended, count, i - group);
}

/* mlt_image_is_opaque() of an RGBA image: transparent is set to 1 when any
 * pixel has an alpha below 255 (every thread that finds one writes the same
 * value). */
GPUFX_KERNEL void gpufx_alpha_check(const gpufx_u32 *__restrict__ src, int n, int *__restrict__ transparent)
{
    const int i = GPUFX_GLOBAL_X;
    if (i < n && (src[i] >> 24) != 255u) {
        *transparent = 1;
    }
}

/* ==========================================================================
 * Image format conversions that libswscale does for MLT, with the flags MLT
 * picks (mlt_get_sws_flags()): 4:2:0 YUV decoded by the avformat producer to
 * RGBA, and RGBA to packed 4:2:2 YUV (YUYV) for the consumer. This is the
 * arithmetic of libswscale's C code (LGPL-2.1-or-later, FFmpeg developers),
 * which its x86 SIMD code reproduces exactly for these paths. The
 * coefficients and sampling positions come from gpufx/src/sws_port.cpp.
 * ========================================================================== */

// av_clip_uintp2(a, 30)
GPUFX_DEV int gpufx_clip_uintp2_30(int a)
{
    if (a & ~((1 << 30) - 1)) {
        return (~a) >> 31 & ((1 << 30) - 1);
    }
    return a;
}

/* 4:2:0 YUV (yuv420p, optionally with an alpha plane) to RGBA with SWS_POINT,
 * SWS_ACCURATE_RND and SWS_FULL_CHR_H_INT. Each plane goes through the 15-bit
 * intermediate (hScale8To15 with a single tap), then
 * yuv2rgb_full_1_c_template() writes the pixels. columns and rows give the
 * chroma sample of every pixel column and row. The alpha plane, if any, has
 * the pitch of the luma plane. */
GPUFX_KERNEL void gpufx_yuv420_to_rgba(const gpufx_u8 *__restrict__ luma, const gpufx_u8 *__restrict__ cb, const gpufx_u8 *__restrict__ cr,
                                       const gpufx_u8 *__restrict__ alpha, int lumaPitch, int chromaPitch, const int *__restrict__ columns,
                                       const int *__restrict__ rows, gpufx_u32 *__restrict__ dst, int w, int h, int yOffset, int yCoeff, int v2r,
                                       int v2g, int u2g, int u2b)
{
    const int x = GPUFX_GLOBAL_X;
    const int y = GPUFX_GLOBAL_Y;
    if (x >= w || y >= h) {
        return;
    }
    const size_t pixel = size_t(y) * size_t(lumaPitch) + size_t(x);
    const size_t chroma = size_t(rows[y]) * size_t(chromaPitch) + size_t(columns[x]);
    int Y = int(luma[pixel]) << 7;
    int U = int(cb[chroma]) << 7;
    int V = int(cr[chroma]) << 7;

    // yuv2rgb_full_1_c_template() without vertical chroma interpolation
    Y *= 4;
    U = (U - (128 << 7)) * 4;
    V = (V - (128 << 7)) * 4;
    // yuv2rgb_write_full()
    Y -= yOffset;
    Y *= yCoeff;
    Y += 1 << 21;
    int R = int(gpufx_u32(Y) + gpufx_u32(V) * gpufx_u32(v2r));
    int G = int(gpufx_u32(Y) + gpufx_u32(V) * gpufx_u32(v2g) + gpufx_u32(U) * gpufx_u32(u2g));
    int B = int(gpufx_u32(Y) + gpufx_u32(U) * gpufx_u32(u2b));
    if ((R | G | B) & 0xC0000000) {
        R = gpufx_clip_uintp2_30(R);
        G = gpufx_clip_uintp2_30(G);
        B = gpufx_clip_uintp2_30(B);
    }
    // (a << 7 + 64) >> 7 for the alpha plane
    const gpufx_u32 A = alpha ? gpufx_u32(alpha[pixel]) : 255u;
    dst[size_t(y) * size_t(w) + size_t(x)] = gpufx_pack(gpufx_u32(R >> 22), gpufx_u32(G >> 22), gpufx_u32(B >> 22), A);
}

/* rgb32ToY_c() for RGBA followed by hScale16To15 with a single tap: 15-bit
 * luma. ry, gy and by come from input_rgb2yuv_table. */
GPUFX_DEV int gpufx_sws_rgba_luma(gpufx_u32 px, int ry, int gy, int by)
{
    const int r = int(px & 0xFFu);
    const int g = int(px & 0xFF00u);
    const int b = int((px & 0xFF0000u) >> 16);
    const gpufx_u32 rnd = (32u << 22) + (1u << 16);
    const int y14 = int((gpufx_u32((ry << 8) * r + gy * g + (by << 8) * b) + rnd) >> 17);
    return gpufx_min(y14 * 2, (1 << 15) - 1);
}

/* rgb32ToUV_half_c() for RGBA (the chroma of two neighbouring pixels)
 * followed by hScale16To15 with a single tap: 15-bit chroma. */
GPUFX_DEV int gpufx_sws_rgba_chroma(gpufx_u32 px0, gpufx_u32 px1, int cr, int cg, int cb)
{
    const gpufx_u32 g = (px0 & 0xFF00FF00u) + (px1 & 0xFF00FF00u);
    const gpufx_u32 rb = px0 + px1 - g;
    const int b = int((rb & 0x1FF0000u) >> 16);
    const int gg = int(g & 0x1FF00u);
    const int r = int(rb & 0x1FFu);
    const gpufx_u32 rnd = (256u << 23) + (1u << 17);
    const int c14 = int((gpufx_u32(cr * 256 * r + cg * gg + cb * 256 * b) + rnd) >> 18);
    return gpufx_min(c14 * 2, (1 << 15) - 1);
}

/* RGBA to YUYV 4:2:2 with SWS_BICUBIC and SWS_ACCURATE_RND: the 15-bit
 * values of gpufx_sws_rgba_luma() and gpufx_sws_rgba_chroma(), converted to
 * full range by lumRangeToJpeg_c() and chrRangeToJpeg_c() of FFmpeg 7.1 and
 * older when fullRange is set (FFmpeg 8.0 rounds that step differently),
 * written by yuv2422_1_c_template(). When alpha is not null the alpha
 * channel is also copied there, one byte per pixel, as MLT's avcolour_space
 * does. One thread per pair of pixels, the width must be even. */
GPUFX_KERNEL void gpufx_rgba_to_yuyv422(const gpufx_u32 *__restrict__ src, gpufx_u8 *__restrict__ dst, gpufx_u8 *__restrict__ alpha, int w, int h,
                                        int ry, int gy, int by, int ru, int gu, int bu, int rv, int gv, int bv, int fullRange)
{
    const int pair = GPUFX_GLOBAL_X;
    const int y = GPUFX_GLOBAL_Y;
    if (2 * pair >= w || y >= h) {
        return;
    }
    const size_t index = size_t(y) * size_t(w) + size_t(2 * pair);
    const gpufx_u32 p0 = src[index];
    const gpufx_u32 p1 = src[index + 1];
    int y0 = gpufx_sws_rgba_luma(p0, ry, gy, by);
    int y1 = gpufx_sws_rgba_luma(p1, ry, gy, by);
    int u = gpufx_sws_rgba_chroma(p0, p1, ru, gu, bu);
    int v = gpufx_sws_rgba_chroma(p0, p1, rv, gv, bv);
    if (fullRange) {
        y0 = (gpufx_min(y0, 30189) * 19077 - 39057361) >> 14;
        y1 = (gpufx_min(y1, 30189) * 19077 - 39057361) >> 14;
        u = (gpufx_min(u, 30775) * 4663 - 9289992) >> 12;
        v = (gpufx_min(v, 30775) * 4663 - 9289992) >> 12;
    }
    int Y1 = (y0 + 64) >> 7;
    int Y2 = (y1 + 64) >> 7;
    int U = (u + 64) >> 7;
    int V = (v + 64) >> 7;
    if ((Y1 | Y2 | U | V) & 0x100) {
        Y1 = gpufx_clamp(Y1, 0, 255);
        Y2 = gpufx_clamp(Y2, 0, 255);
        U = gpufx_clamp(U, 0, 255);
        V = gpufx_clamp(V, 0, 255);
    }
    gpufx_u8 *out = dst + index * 2;
    out[0] = gpufx_u8(Y1);
    out[1] = gpufx_u8(U);
    out[2] = gpufx_u8(Y2);
    out[3] = gpufx_u8(V);
    if (alpha) {
        alpha[index] = gpufx_u8(p0 >> 24);
        alpha[index + 1] = gpufx_u8(p1 >> 24);
    }
}

/* Packed 4:2:2 YUV (YUYV) to RGBA, as MLT's avcolour_space converts it with
 * the flags of gpufx_yuv420_to_rgba(): the same arithmetic, with the chroma of
 * the pixel's own row. columns gives the chroma sample of every pixel column.
 * The alpha channel comes from alphaSrc, an RGBA image of the same size, as
 * avcolour_space merges back the alpha plane it took out of the RGBA image. */
GPUFX_KERNEL void gpufx_yuyv422_to_rgba(const gpufx_u8 *__restrict__ src, const int *__restrict__ columns, const gpufx_u32 *__restrict__ alphaSrc,
                                        gpufx_u32 *__restrict__ dst, int w, int h, int yOffset, int yCoeff, int v2r, int v2g, int u2g, int u2b)
{
    const int x = GPUFX_GLOBAL_X;
    const int y = GPUFX_GLOBAL_Y;
    if (x >= w || y >= h) {
        return;
    }
    const size_t pixel = size_t(y) * size_t(w) + size_t(x);
    const size_t chroma = (size_t(y) * size_t(w) + 2 * size_t(columns[x])) * 2;
    int Y = int(src[pixel * 2]) << 7;
    int U = int(src[chroma + 1]) << 7;
    int V = int(src[chroma + 3]) << 7;

    // yuv2rgb_full_1_c_template() without vertical chroma interpolation
    Y *= 4;
    U = (U - (128 << 7)) * 4;
    V = (V - (128 << 7)) * 4;
    // yuv2rgb_write_full()
    Y -= yOffset;
    Y *= yCoeff;
    Y += 1 << 21;
    int R = int(gpufx_u32(Y) + gpufx_u32(V) * gpufx_u32(v2r));
    int G = int(gpufx_u32(Y) + gpufx_u32(V) * gpufx_u32(v2g) + gpufx_u32(U) * gpufx_u32(u2g));
    int B = int(gpufx_u32(Y) + gpufx_u32(U) * gpufx_u32(u2b));
    if ((R | G | B) & 0xC0000000) {
        R = gpufx_clip_uintp2_30(R);
        G = gpufx_clip_uintp2_30(G);
        B = gpufx_clip_uintp2_30(B);
    }
    dst[pixel] = gpufx_pack(gpufx_u32(R >> 22), gpufx_u32(G >> 22), gpufx_u32(B >> 22), alphaSrc[pixel] >> 24);
}

/* An RGBA image placed at (offsetX, offsetY) in a larger one filled with
 * fill, as MLT's resize filter pads images (resize_image()). */
GPUFX_KERNEL void gpufx_pad_rgba(const gpufx_u32 *__restrict__ src, int srcWidth, int srcHeight, gpufx_u32 *__restrict__ dst, int w, int h, int offsetX,
                                 int offsetY, gpufx_u32 fill)
{
    const int x = GPUFX_GLOBAL_X;
    const int y = GPUFX_GLOBAL_Y;
    if (x >= w || y >= h) {
        return;
    }
    const int sx = x - offsetX;
    const int sy = y - offsetY;
    const bool inside = sx >= 0 && sx < srcWidth && sy >= 0 && sy < srcHeight;
    dst[size_t(y) * size_t(w) + size_t(x)] = inside ? src[size_t(sy) * size_t(srcWidth) + size_t(sx)] : fill;
}

/* ==========================================================================
 * RGBA scaled to RGBA as MLT's rescale filter does it with libswscale (its
 * swscale scaler: SWS_FULL_CHR_H_INP, SWS_FULL_CHR_H_INT and
 * SWS_ACCURATE_RND): converted to 15-bit YUV and alpha while scaled
 * horizontally, then scaled vertically and converted back. The filters come
 * from gpufx/src/sws_port.cpp (initFilter()); as in libswscale, the taps of a
 * filter past the last input sample have no weight.
 * ========================================================================== */

/* rgb32ToY_c() for RGBA: 14-bit luma */
GPUFX_DEV int gpufx_sws_luma14(gpufx_u32 px, int ry, int gy, int by)
{
    const int r = int(px & 0xFFu);
    const int g = int(px & 0xFF00u);
    const int b = int((px & 0xFF0000u) >> 16);
    return int((gpufx_u32((ry << 8) * r + gy * g + (by << 8) * b) + (32u << 22) + (1u << 16)) >> 17);
}

/* rgb32ToUV_c() for RGBA: 14-bit chroma, U or V */
GPUFX_DEV int gpufx_sws_chroma14(gpufx_u32 px, int cr, int cg, int cb)
{
    const int r = int(px & 0xFFu);
    const int g = int(px & 0xFF00u);
    const int b = int((px & 0xFF0000u) >> 16);
    return int((gpufx_u32(cr * 256 * r + cg * g + cb * 256 * b) + (256u << 22) + (1u << 16)) >> 17);
}

/* The horizontal pass (rgb32ToY_c(), rgb32ToUV_c(), rgbaToA_c(), then
 * hScale16To15_c()): planes receives the 15-bit Y, U, V and alpha planes,
 * dstWidth x srcHeight each. Luma and alpha go through the luma filter,
 * chroma through the chroma filter. */
GPUFX_KERNEL void gpufx_sws_scale_rgba_h(const gpufx_u32 *__restrict__ src, int srcWidth, int srcHeight, gpufx_i16 *__restrict__ planes, int dstWidth,
                                         const gpufx_i16 *__restrict__ lumFilter, const int *__restrict__ lumPos, int lumSize,
                                         const gpufx_i16 *__restrict__ chrFilter, const int *__restrict__ chrPos, int chrSize, int ry, int gy, int by,
                                         int ru, int gu, int bu, int rv, int gv, int bv)
{
    const int x = GPUFX_GLOBAL_X;
    const int y = GPUFX_GLOBAL_Y;
    if (x >= dstWidth || y >= srcHeight) {
        return;
    }
    const gpufx_u32 *line = src + size_t(y) * size_t(srcWidth);
    int sumY = 0;
    int sumA = 0;
    for (int j = 0; j < lumSize; ++j) {
        const gpufx_u32 px = line[gpufx_min(lumPos[x] + j, srcWidth - 1)];
        const int c = lumFilter[x * lumSize + j];
        const int a = int(px >> 24);
        sumY += gpufx_sws_luma14(px, ry, gy, by) * c;
        sumA += ((a << 6) | (a >> 2)) * c;
    }
    int sumU = 0;
    int sumV = 0;
    for (int j = 0; j < chrSize; ++j) {
        const gpufx_u32 px = line[gpufx_min(chrPos[x] + j, srcWidth - 1)];
        const int c = chrFilter[x * chrSize + j];
        sumU += gpufx_sws_chroma14(px, ru, gu, bu) * c;
        sumV += gpufx_sws_chroma14(px, rv, gv, bv) * c;
    }
    const size_t plane = size_t(dstWidth) * size_t(srcHeight);
    const size_t index = size_t(y) * size_t(dstWidth) + size_t(x);
    planes[index] = gpufx_i16(gpufx_min(sumY >> 13, (1 << 15) - 1));
    planes[plane + index] = gpufx_i16(gpufx_min(sumU >> 13, (1 << 15) - 1));
    planes[2 * plane + index] = gpufx_i16(gpufx_min(sumV >> 13, (1 << 15) - 1));
    planes[3 * plane + index] = gpufx_i16(gpufx_min(sumA >> 13, (1 << 15) - 1));
}

/* A vertical filter of two taps that packed_vscale() gives to the 2-tap
 * output function */
GPUFX_DEV bool gpufx_sws_two_taps(const gpufx_i16 *f)
{
    const gpufx_u32 f0 = gpufx_u16(f[0]);
    const gpufx_u32 f1 = gpufx_u16(f[1]);
    return f1 + f0 == 4096 && f1 <= 4096u;
}

/* The vertical pass and the output (packed_vscale(), then
 * yuv2rgb_full_1_c_template() for single taps, yuv2rgb_full_2_c_template()
 * for two taps or yuv2rgb_full_X_c_template(), and yuv2rgb_write_full()):
 * planes of gpufx_sws_scale_rgba_h(), width x srcHeight, to dst, width x
 * dstHeight. The filters have one tap each, or more taps each. */
GPUFX_KERNEL void gpufx_sws_scale_rgba_v(const gpufx_i16 *__restrict__ planes, int width, int srcHeight, gpufx_u32 *__restrict__ dst, int dstHeight,
                                         const gpufx_i16 *__restrict__ lumFilter, const int *__restrict__ lumPos, int lumSize,
                                         const gpufx_i16 *__restrict__ chrFilter, const int *__restrict__ chrPos, int chrSize, int yOffset, int yCoeff,
                                         int v2r, int v2g, int u2g, int u2b)
{
    const int x = GPUFX_GLOBAL_X;
    const int y = GPUFX_GLOBAL_Y;
    if (x >= width || y >= dstHeight) {
        return;
    }
    const size_t plane = size_t(width) * size_t(srcHeight);
    const gpufx_i16 *lumaPlane = planes;
    const gpufx_i16 *uPlane = planes + plane;
    const gpufx_i16 *vPlane = planes + 2 * plane;
    const gpufx_i16 *alphaPlane = planes + 3 * plane;
    const gpufx_i16 *lf = lumFilter + y * lumSize;
    const gpufx_i16 *cf = chrFilter + y * chrSize;
    int Y;
    int U;
    int V;
    int A;
    if (lumSize == 1 && chrSize == 1) {
        const size_t l = size_t(lumPos[y]) * size_t(width) + size_t(x);
        const size_t c = size_t(chrPos[y]) * size_t(width) + size_t(x);
        Y = lumaPlane[l] * 4;
        U = (uPlane[c] - (128 << 7)) * 4;
        V = (vPlane[c] - (128 << 7)) * 4;
        A = (alphaPlane[l] + 64) >> 7;
    } else if (lumSize == 2 && chrSize == 2 && gpufx_sws_two_taps(lf) && gpufx_sws_two_taps(cf)) {
        const int yalpha = gpufx_u16(lf[1]);
        const int yalpha1 = 4096 - yalpha;
        const int uvalpha = gpufx_u16(cf[1]);
        const int uvalpha1 = 4096 - uvalpha;
        const size_t l0 = size_t(lumPos[y]) * size_t(width) + size_t(x);
        const size_t l1 = size_t(gpufx_min(lumPos[y] + 1, srcHeight - 1)) * size_t(width) + size_t(x);
        const size_t c0 = size_t(chrPos[y]) * size_t(width) + size_t(x);
        const size_t c1 = size_t(gpufx_min(chrPos[y] + 1, srcHeight - 1)) * size_t(width) + size_t(x);
        Y = (lumaPlane[l0] * yalpha1 + lumaPlane[l1] * yalpha) >> 10;
        U = (uPlane[c0] * uvalpha1 + uPlane[c1] * uvalpha - (128 << 19)) >> 10;
        V = (vPlane[c0] * uvalpha1 + vPlane[c1] * uvalpha - (128 << 19)) >> 10;
        A = (alphaPlane[l0] * yalpha1 + alphaPlane[l1] * yalpha + (1 << 18)) >> 19;
    } else {
        Y = 1 << 9;
        U = (1 << 9) - (128 << 19);
        V = (1 << 9) - (128 << 19);
        A = 1 << 18;
        for (int j = 0; j < lumSize; ++j) {
            const size_t l = size_t(gpufx_min(lumPos[y] + j, srcHeight - 1)) * size_t(width) + size_t(x);
            Y += lumaPlane[l] * lf[j];
            A += alphaPlane[l] * lf[j];
        }
        for (int j = 0; j < chrSize; ++j) {
            const size_t c = size_t(gpufx_min(chrPos[y] + j, srcHeight - 1)) * size_t(width) + size_t(x);
            U += uPlane[c] * cf[j];
            V += vPlane[c] * cf[j];
        }
        Y >>= 10;
        U >>= 10;
        V >>= 10;
        A >>= 19;
    }
    if (A & 0x100) {
        A = gpufx_clamp(A, 0, 255);
    }
    // yuv2rgb_write_full()
    Y -= yOffset;
    Y *= yCoeff;
    Y += 1 << 21;
    int R = int(gpufx_u32(Y) + gpufx_u32(V) * gpufx_u32(v2r));
    int G = int(gpufx_u32(Y) + gpufx_u32(V) * gpufx_u32(v2g) + gpufx_u32(U) * gpufx_u32(u2g));
    int B = int(gpufx_u32(Y) + gpufx_u32(U) * gpufx_u32(u2b));
    if ((R | G | B) & 0xC0000000) {
        R = gpufx_clip_uintp2_30(R);
        G = gpufx_clip_uintp2_30(G);
        B = gpufx_clip_uintp2_30(B);
    }
    dst[size_t(y) * size_t(width) + size_t(x)] = gpufx_pack(gpufx_u32(R >> 22), gpufx_u32(G >> 22), gpufx_u32(B >> 22), gpufx_u32(A) & 0xFFu);
}
