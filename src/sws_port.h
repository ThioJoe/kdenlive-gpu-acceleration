/*
    SPDX-FileCopyrightText: FFmpeg developers
    SPDX-FileCopyrightText: 2026 Kdenlive contributors
    SPDX-License-Identifier: LGPL-2.1-or-later
*/

/*
 * The parts of libswscale's setup that decide the arithmetic of the
 * conversions MLT does between YUV and RGBA, ported so that the GPU kernels
 * get the same coefficients and sampling positions as the CPU.
 *
 * The code paths were compared between FFmpeg 6.1, 7.1 and 8.0: the parts
 * used here do not differ.
 */

#pragma once

#include <vector>

namespace gpufx {
namespace sws {

// SWS_CS_* values
enum Colorspace { CsItu709 = 1, CsFcc = 4, CsItu601 = 5, CsSmpte240m = 7, CsBt2020 = 9, CsDefault = 5 };

/** The libswscale coefficient table that MLT's mlt_set_luma_transfer() picks
 *  for an MLT colorspace (mlt_colorspace value). */
int colorspaceTable(int mltColorspace);

/** yuv2rgb_* coefficients of SwsContext, set by ff_yuv2rgb_c_init_tables()
 *  for neutral brightness, contrast and saturation. */
struct YuvToRgb
{
    int yOffset;
    int yCoeff;
    int v2r;
    int v2g;
    int u2g;
    int u2b;
};
YuvToRgb yuvToRgb(int table, bool fullRange);

/** input_rgb2yuv_table of SwsContext, set by fill_rgb2yuv_table(). */
struct RgbToYuv
{
    int ry, gy, by;
    int ru, gu, bu;
    int rv, gv, bv;
};
RgbToYuv rgbToYuv(int table);

/** get_local_pos(): chroma sample position for the filter setup. */
int localPosition(int chromaSubsample, int position);

/** The source sample used for every destination sample by an SWS_POINT
 *  scaler from srcSize to dstSize samples (initFilter(), including its border
 *  handling). srcPos and dstPos come from localPosition(). */
void pointSamples(int srcSize, int dstSize, int srcPos, int dstPos, std::vector<int> &samples);

/** The chroma samples used by libswscale to convert 4:2:0 YUV of the given
 *  size to RGB with SWS_POINT and full chroma interpolation, as MLT's
 *  avformat producer does: the chroma column of every pixel column and the
 *  chroma row of every pixel row. With fields set, the image is converted
 *  as two separate fields (MLT's field-aware conversion of interlaced
 *  frames, the height must be a multiple of 4). */
void chroma420Samples(int width, int height, bool fields, std::vector<int> &columns, std::vector<int> &rows);

} // namespace sws
} // namespace gpufx
