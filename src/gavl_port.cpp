/*
    SPDX-FileCopyrightText: Members of the Gmerlin project
    SPDX-FileCopyrightText: 2026 ThioJoe
    SPDX-License-Identifier: GPL-2.0-or-later
*/

#include "gavl_port.h"

#include <cfloat>
#include <cmath>
#include <cstddef>

namespace gpufx {
namespace gavl {

namespace {

// gavl_scale_mode_t values the bilinear mode can turn into for small images
enum Mode { Nearest, Bilinear, Bspline };

int modeTaps(Mode mode)
{
    return mode == Nearest ? 1 : mode == Bilinear ? 2 : 4;
}

// scale_kernels.c
float weight(Mode mode, double t)
{
    switch (mode) {
    case Nearest:
        if (t < -0.5) {
            return 0.0;
        }
        if (t <= 0.5) {
            return 1.0;
        }
        return 0.0;
    case Bilinear:
        if (t <= -1.0) {
            return 0.0;
        }
        if (t < 0.0) {
            return (t + 1.0);
        }
        if (t < 1.0) {
            return (1.0 - t);
        }
        return 0.0;
    case Bspline:
        if (t < 0.0) {
            t = -t;
        }
        if (t < 1.0) {
            return (4.0 + (t * t * (-6.0 + (t * 3.0)))) / 6.0;
        }
        if (t < 2.0) {
            t = 2.0 - t;
            return t * t * t / 6.0;
        }
        return 0.0;
    }
    return 0.0;
}

// scale_table.c
void shiftUp(float *fac, int num, int shift)
{
    for (int j = 0; j < shift; j++) {
        fac[shift] += fac[j];
    }
    for (int j = 0; j < num - shift; j++) {
        fac[j] = fac[j + shift];
    }
    for (int j = num - shift; j < num; j++) {
        fac[j] = 0.0;
    }
}

void shiftDown(float *fac, int num, int shift)
{
    for (int j = num - shift; j < num; j++) {
        fac[num - shift - 1] += fac[j];
    }
    for (int j = num - 1; j >= shift; j--) {
        fac[j] = fac[j - shift];
    }
    for (int j = 0; j < shift; j++) {
        fac[j] = 0.0;
    }
}

/** gavl_video_scale_table_init() with gavl's default options
 *  (GAVL_DOWNSCALE_FILTER_WIDE, downscale_blur 1.0), followed by
 *  gavl_video_scale_table_init_int() with the 14 bits of the SIMD code.
 *  mode may change for images smaller than the filter. */
void tableInit(ScaleTable &tab, Mode &mode, double src_off, double src_size, int dst_size, int src_width)
{
    const double scale_factor = (double)(dst_size) / src_size;
    tab.taps = modeTaps(mode);
    float widen_factor;
    if (scale_factor < 1.0) {
        widen_factor = ceil(1.0 / scale_factor);
        tab.taps *= (int)(widen_factor);
    } else {
        widen_factor = 1.0;
    }
    if (tab.taps > src_width) {
        switch (src_width) {
        case 1:
            mode = Nearest;
            break;
        case 2:
        case 3:
            mode = Bilinear;
            break;
        default:
            mode = Bspline;
            break;
        }
        tab.taps = modeTaps(mode);
    }
    const int taps = tab.taps;
    tab.index.assign(size_t(dst_size), 0);
    std::vector<float> f(size_t(dst_size) * size_t(taps), 0.0f);
    for (int i = 0; i < dst_size; i++) {
        const double src_index_f = ((double)((double)i)) / scale_factor + src_off;
        const int src_index_nearest = (src_index_f >= 0.0) ? (int)(src_index_f + 0.5) : (int)(src_index_f - 0.5);
        int &index = tab.index[size_t(i)];
        index = src_index_nearest - taps / 2;
        if (taps == 1) {
            if (index < 0) {
                index = 0;
            }
            if (index > src_width - 1) {
                index = src_width - 1;
            }
            continue;
        }
        double t = (src_index_f - index) / widen_factor;
        for (int j = 0; j < taps; j++) {
            f[size_t(i) * size_t(taps) + size_t(j)] = weight(mode, t);
            t -= 1.0 / widen_factor;
        }
    }
    // shift_borders()
    for (int i = 0; i < dst_size; i++) {
        float *fac = &f[size_t(i) * size_t(taps)];
        int &index = tab.index[size_t(i)];
        if (index < 0) {
            shiftUp(fac, taps, -index);
            index = 0;
        }
        if (index + taps > src_width) {
            shiftDown(fac, taps, index + taps - src_width);
            index = src_width - taps;
        }
    }
    // normalize_table()
    for (int i = 0; i < dst_size; i++) {
        float sum = 0.0;
        for (int j = 0; j < taps; j++) {
            sum += f[size_t(i) * size_t(taps) + size_t(j)];
        }
        sum += FLT_EPSILON;
        for (int j = 0; j < taps; j++) {
            f[size_t(i) * size_t(taps) + size_t(j)] /= sum;
        }
    }
    // gavl_video_scale_table_init_int()
    const int fac_max_i = (1 << 14);
    const float fac_max_f = (float)(fac_max_i);
    tab.factors.assign(f.size(), 0);
    int index = 0;
    int fac_i_norm = 0;
    for (int i = 0; i < dst_size; i++) {
        int min_index = index;
        int max_index = index;
        int sum_i = 0;
        float sum_f = 0.0;
        for (int j = 0; j < taps; j++) {
            tab.factors[size_t(index)] = (int)(fac_max_f * f[size_t(index)] + 0.5);
            sum_i += tab.factors[size_t(index)];
            sum_f += f[size_t(index)];
            if (j) {
                if (tab.factors[size_t(index)] > tab.factors[size_t(max_index)]) {
                    max_index = index;
                }
                if (tab.factors[size_t(index)] < tab.factors[size_t(min_index)]) {
                    min_index = index;
                }
            }
            index++;
        }
        if (!i) {
            fac_i_norm = (int)(sum_f * fac_max_i + 0.5);
        }
        if (sum_i > fac_i_norm) {
            tab.factors[size_t(max_index)] -= (sum_i - fac_i_norm);
        } else if (sum_i < fac_i_norm) {
            tab.factors[size_t(min_index)] += (fac_i_norm - sum_i);
        }
    }
}

// get_scale_offset() of scale_context.c for progressive RGBA
float scaleOffset(float scale_fac)
{
    const float chroma_scale_fac = (float)1 / (float)1;
    scale_fac *= chroma_scale_fac;
    const float src_dist = (0.5 + 0.0f) / 1;
    const float dst_dist = (0.5 + 0.0f) / (1 * scale_fac);
    return dst_dist - src_dist;
}

} // namespace

bool scale0tiltPlan(const Scale0TiltParams &p, int w, int h, Scale0TiltPlan &plan)
{
    plan = Scale0TiltPlan();
    // update_scaler() of scale0tilt.c
    float dst_x, dst_y, dst_w, dst_h;
    float src_x, src_y, src_w, src_h;
    src_x = w * p.cl;
    src_y = h * p.ct;
    src_w = w * (1.0 - p.cl - p.cr);
    src_h = h * (1.0 - p.ct - p.cb);
    dst_x = w * p.cl * p.sx + p.tx * w;
    dst_y = h * p.ct * p.sy + p.ty * h;
    dst_w = w * (1.0 - p.cl - p.cr) * p.sx;
    dst_h = h * (1.0 - p.ct - p.cb) * p.sy;
    const double EPSILON = 1e-6;
    if ((dst_w < EPSILON) || (dst_h < EPSILON) || (src_w < EPSILON) || (src_h < EPSILON)) {
        return true;
    }
    if (dst_x + dst_w > w) {
        src_w = src_w * ((w - dst_x) / dst_w);
        dst_w = w - dst_x;
    }
    if (dst_y + dst_h > h) {
        src_h = src_h * ((h - dst_y) / dst_h);
        dst_h = h - dst_y;
    }
    if (dst_x < 0) {
        src_x = src_x - dst_x * (src_w / dst_w);
        src_w = src_w * ((dst_w + dst_x) / dst_w);
        dst_w = dst_w + dst_x;
        dst_x = 0;
    }
    if (dst_y < 0) {
        src_y = src_y - dst_y * (src_h / dst_h);
        src_h = src_h * ((dst_h + dst_y) / dst_h);
        dst_h = dst_h + dst_y;
        dst_y = 0;
    }
    if ((dst_w < EPSILON) || (dst_h < EPSILON) || (src_w < EPSILON) || (src_h < EPSILON)) {
        return true;
    }
    // NaN parameters
    if (!(std::isfinite(dst_x) && std::isfinite(dst_y) && std::isfinite(dst_w) && std::isfinite(dst_h) && std::isfinite(src_x) && std::isfinite(src_y) &&
          std::isfinite(src_w) && std::isfinite(src_h)) ||
        !(dst_w < 1e8f && dst_h < 1e8f)) {
        return false;
    }
    const double srcX = src_x;
    const double srcY = src_y;
    const double srcW = src_w;
    const double srcH = src_h;
    const int dstX = lroundf(dst_x);
    const int dstY = lroundf(dst_y);
    const int dstW = lroundf(dst_w);
    const int dstH = lroundf(dst_h);
    // gavl draws dstW pixels per line from dstX on, also past the end of the
    // line
    if (dstW <= 0 || dstH <= 0 || dstX < 0 || dstY < 0 || dstX + dstW > w || dstY + dstH > h) {
        return false;
    }
    plan.dstX = dstX;
    plan.dstY = dstY;
    plan.dstWidth = dstW;
    plan.dstHeight = dstH;

    // gavl_video_scale_context_init(), plane 0 of progressive RGBA
    const double scaleFactorX = (float)dstW / srcW;
    const double scaleFactorY = (float)dstH / srcH;
    float offset_x = scaleOffset(scaleFactorX);
    float offset_y = scaleOffset(scaleFactorY);
    const double EPS = 1e-4;
    const bool scaleX = (fabs(srcW - dstW) > EPS) || (fabs(srcX) > EPS) || (fabs(offset_x) > EPS);
    const bool scaleY = (fabs(srcH - dstH) > EPS) || (fabs(srcY) > EPS) || (fabs(offset_y) > EPS);
    offset_x += srcX;
    offset_y += srcY;
    // The SSE2 code of the vertical pass computes the first bytes of a line
    // up to the next 16-byte boundary of the input with its C code, from the
    // start of the input line: MLT's images are 16-byte aligned, scale0tilt
    // copies those with lines of other lengths to aligned ones with padded
    // lines. Vertical passes over the intermediate image start aligned too.
    if (!scaleX && !scaleY) {
        plan.kind = Scale0TiltPlan::Copy;
        return true;
    }
    Mode modeH = Bilinear;
    Mode modeV = Bilinear;
    const int dstBytes = dstW * 4;
    if (scaleX && scaleY) {
        tableInit(plan.horizontal, modeH, offset_x, srcW, dstW, w);
        tableInit(plan.vertical, modeV, offset_y, srcH, dstH, h);
        if (modeH == Nearest || modeV == Nearest) {
            return false;
        }
        const int srcX0 = plan.horizontal.index.front();
        const int srcWidth = plan.horizontal.index.back() + plan.horizontal.taps - srcX0;
        const int srcY0 = plan.vertical.index.front();
        const int srcHeight = plan.vertical.index.back() + plan.vertical.taps - srcY0;
        // The order with the smaller intermediate image
        if (srcHeight * dstW < dstH * srcWidth) {
            plan.kind = Scale0TiltPlan::XThenY;
            plan.bufferWidth = dstW;
            plan.bufferHeight = srcHeight;
            plan.firstRow = srcY0;
            for (int &index : plan.vertical.index) {
                index -= srcY0;
            }
            plan.lead = 16;
            // The SSE2 code overruns lines shorter than its first C bytes
            if (plan.vertical.taps != 2 && plan.lead > dstBytes) {
                return false;
            }
        } else {
            plan.kind = Scale0TiltPlan::YThenX;
            plan.bufferWidth = srcWidth;
            plan.bufferHeight = dstH;
            plan.firstColumn = srcX0;
            for (int &index : plan.horizontal.index) {
                index -= srcX0;
            }
            // Overruns only reach the intermediate image's padding here
            plan.lead = 16 - (4 * srcX0) % 16;
        }
    } else if (scaleX) {
        tableInit(plan.horizontal, modeH, offset_x, srcW, dstW, w);
        if (modeH == Nearest) {
            return false;
        }
        plan.kind = Scale0TiltPlan::ScaleX;
    } else {
        tableInit(plan.vertical, modeV, offset_y, srcH, dstH, h);
        if (modeV == Nearest) {
            return false;
        }
        plan.kind = Scale0TiltPlan::ScaleY;
        plan.lead = 16;
        if (plan.vertical.taps != 2 && plan.lead > dstBytes) {
            return false;
        }
    }
    return true;
}

} // namespace gavl
} // namespace gpufx
