/*
    SPDX-FileCopyrightText: 2026 ThioJoe
    SPDX-License-Identifier: GPL-3.0-only
*/

#pragma once

#include "gpu_runtime.h"
#include "qt_raster.h"

#include <cstdint>
#include <vector>

namespace gpufx {

/**
 * QPainter::drawImage(0, 0, image) of an RGBA8888 image onto an RGBA8888
 * image, with a transform, an opacity, the alpha blend composition mode and
 * optionally smooth painting (SmoothPixmapTransform and Antialiasing), drawn
 * on the GPU the way Qt's raster engine draws it (gpufx_qt_draw_image).
 */
class QtImageDrawing
{
public:
    /** Decides Qt's code path and computes what its rasterizer produces for
     *  a sourceWidth x sourceHeight image drawn with matrix onto a
     *  targetWidth x targetHeight image. Returns false for the paths that
     *  are not reproduced (the CPU has to draw). */
    bool prepare(int sourceWidth, int sourceHeight, int targetWidth, int targetHeight, const qt::Transform &matrix, double opacity, bool smooth);

    /** Draws source onto the target: under holds the target content before
     *  drawing (targetWidth x targetHeight), or is 0 for a target filled
     *  with background. out receives the result, it must not overlap the
     *  inputs. */
    CUresult draw(GpuScope &scope, CUdeviceptr source, CUdeviceptr under, uint32_t background, CUdeviceptr out) const;

private:
    int m_sourceWidth = 0;
    int m_sourceHeight = 0;
    int m_targetWidth = 0;
    int m_targetHeight = 0;
    int m_mode = 0;
    int m_fdx = 0;
    int m_fdy = 0;
    int m_opacity = 256;
    std::vector<int> m_table; // rows, then spans, then runs
    int m_spansOffset = 0;
    int m_runsOffset = 0;
};

} // namespace gpufx
