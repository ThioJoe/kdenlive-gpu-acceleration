/*
    SPDX-FileCopyrightText: 2026 ThioJoe
    SPDX-License-Identifier: GPL-3.0-only
*/

/*
 * The kernels exported by gpufx_kernels.cu. The module resolves every entry
 * when it loads the PTX, and the test emulator registers every entry, so a new
 * kernel only has to be added here and in gpufx_kernels.cu.
 */

#pragma once

#define GPUFX_KERNEL_LIST(X)                                                                                                                                   \
    X(gpufx_dropshadow_alpha)                                                                                                                                  \
    X(gpufx_expblur_rows)                                                                                                                                      \
    X(gpufx_expblur_cols)                                                                                                                                      \
    X(gpufx_dropshadow_composite)                                                                                                                              \
    X(gpufx_over_self)                                                                                                                                         \
    X(gpufx_box_blur_h)                                                                                                                                        \
    X(gpufx_box_blur_v)                                                                                                                                        \
    X(gpufx_rect_blur_h)                                                                                                                                       \
    X(gpufx_rect_blur_v)                                                                                                                                       \
    X(gpufx_lut_rgba)                                                                                                                                          \
    X(gpufx_square_blur_h)                                                                                                                                     \
    X(gpufx_square_blur_v)                                                                                                                                     \
    X(gpufx_softglow_highlights)                                                                                                                               \
    X(gpufx_unsharp_h)                                                                                                                                         \
    X(gpufx_unsharp_v)                                                                                                                                         \
    X(gpufx_qt_draw_image)                                                                                                                                     \
    X(gpufx_alpha_check)                                                                                                                                       \
    X(gpufx_yuv420_to_rgba)                                                                                                                                    \
    X(gpufx_rgba_to_yuyv422)                                                                                                                                   \
    X(gpufx_yuyv422_to_rgba)                                                                                                                                   \
    X(gpufx_pad_rgba)                                                                                                                                          \
    X(gpufx_sws_scale_rgba_h)                                                                                                                                  \
    X(gpufx_sws_scale_rgba_v)                                                                                                                                  \
    X(gpufx_rgba_to_yuv420)                                                                                                                                    \
    X(gpufx_luma_dissolve_rgba)                                                                                                                                \
    X(gpufx_luma_wipe_rgba)                                                                                                                                    \
    X(gpufx_luma_yuyv)                                                                                                                                         \
    X(gpufx_gavl_scale_x)                                                                                                                                      \
    X(gpufx_gavl_scale_y)                                                                                                                                      \
    X(gpufx_gavl_copy)
