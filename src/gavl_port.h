/*
    SPDX-FileCopyrightText: Members of the Gmerlin project
    SPDX-FileCopyrightText: 2026 ThioJoe
    SPDX-License-Identifier: GPL-2.0-or-later
*/

/*
 * The setup of gavl's video scaler (gavl 2.0.0) as frei0r's scale0tilt plugin
 * uses it: the rectangles scale0tilt computes, the filters gavl's scaler
 * builds for them with its default options (bilinear, widened when scaling
 * down) and the order of its passes, ported so that the GPU kernels scale
 * with the same filters and positions. The scaling itself is in
 * gpufx_kernels.cu (gpufx_gavl_*).
 */

#pragma once

#include <vector>

namespace gpufx {
namespace gavl {

/** One pass of gavl's scaler: for every output sample, the first input
 *  sample and taps 14-bit integer coefficients. */
struct ScaleTable
{
    int taps = 0;
    std::vector<int> index;
    std::vector<int> factors;
};

struct Scale0TiltPlan
{
    enum Kind {
        Empty,  // nothing drawn, the output stays transparent
        Copy,   // the image copied without scaling
        ScaleX, // horizontal pass only
        ScaleY, // vertical pass only
        XThenY, // horizontal pass into a buffer, then the vertical one
        YThenX  // vertical pass into a buffer, then the horizontal one
    };
    Kind kind = Empty;
    // The rectangle of the output that is drawn
    int dstX = 0;
    int dstY = 0;
    int dstWidth = 0;
    int dstHeight = 0;
    ScaleTable horizontal;
    ScaleTable vertical;
    // The intermediate image of two passes, and where the first pass starts
    // in the input (XThenY: its first row, YThenX: its first column)
    int bufferWidth = 0;
    int bufferHeight = 0;
    int firstRow = 0;
    int firstColumn = 0;
    // Vertical passes over the input: the number of bytes of a line that
    // gavl's SSE2 code computes with its C code before its first aligned block
    int lead = 16;
};

/** The parameters of a scale0tilt instance (its fields, after
 *  f0r_set_param_value()) */
struct Scale0TiltParams
{
    double cl = 0.0, ct = 0.0, cr = 0.0, cb = 0.0;
    double sx = 1.0, sy = 1.0;
    double tx = 0.0, ty = 0.0;
};

/** What scale0tilt's f0r_update() does to a width x height image. Returns
 *  false for the cases the GPU does not reproduce: images smaller than the
 *  filters, a rectangle that gavl draws past the end of the image's lines,
 *  and outputs narrower than 4 pixels, which gavl's SSE2 code overruns. */
bool scale0tiltPlan(const Scale0TiltParams &params, int width, int height, Scale0TiltPlan &plan);

} // namespace gavl
} // namespace gpufx
