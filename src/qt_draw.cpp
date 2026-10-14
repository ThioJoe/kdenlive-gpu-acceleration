/*
    SPDX-FileCopyrightText: 2026 Kdenlive contributors
    SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#include "qt_draw.h"

#include <algorithm>
#include <climits>
#include <cmath>

namespace gpufx {

namespace {

// Fixed point of qdrawhelper.cpp
constexpr int kFixedScale = 1 << 16;
constexpr int kHalfPoint = 1 << 15;

// Modes of gpufx_qt_draw_image
enum DrawMode { Untransformed = 0, Nearest = 1, Bilinear = 2 };

} // namespace

bool QtImageDrawing::prepare(int sw, int sh, int dw, int dh, const qt::Transform &matrix, double opacity, bool smooth)
{
    if (sw <= 0 || sh <= 0 || dw <= 0 || dh <= 0) {
        return false;
    }
    m_sourceWidth = sw;
    m_sourceHeight = sh;
    m_targetWidth = dw;
    m_targetHeight = dh;
    // QPainter::setOpacity() and QRasterPaintEngine::opacityChanged()
    const double clamped = std::max(0.0, std::min(1.0, opacity));
    m_opacity = int(clamped * 256);

    std::vector<qt::Span> spans;
    std::vector<qt::Run> runs;
    const qt::TransformationType type = matrix.type();
    // Source position of the first pixel of each run
    std::vector<int> starts;

    if (type <= qt::TxTranslate) {
        // QRasterPaintEngine::drawImage(QPointF): untransformed texture at the rounded position
        const double ptX = matrix.dx();
        const double ptY = matrix.dy();
        qt::fillRectSpans(qt::round(ptX), qt::round(ptY), sw, sh, dw, dh, spans);
        qt::spanRuns(spans, 512, false, runs);
        // blend_untransformed_generic_rgb64(): data->dx is -pt.x()
        const int xoff = -qt::round(ptX);
        const int yoff = -qt::round(ptY);
        for (const qt::Run &run : runs) {
            starts.push_back(xoff + run.x);
            starts.push_back(yoff + run.y);
        }
        m_mode = Untransformed;
        m_fdx = 0;
        m_fdy = 0;
    } else {
        // drawImage(QRectF(0, 0, sw, sh), image, QRectF(0, 0, sw, sh))
        if (type <= qt::TxScale && !smooth && sw == 1 && sh == 1) {
            return false; // drawn as a solid color
        }
        // QSpanData::setupMatrix()
        qt::Transform delta;
        delta.translate(1.0 / 65536, 1.0 / 65536);
        const qt::Transform inv = (delta * matrix).inverted();
        const double m11 = inv.m11();
        const double m12 = inv.m12();
        const double m21 = inv.m21();
        const double m22 = inv.m22();
        const double dx = inv.dx();
        const double dy = inv.dy();
        const double f1 = m11 * m11 + m21 * m21;
        const double f2 = m12 * m12 + m22 * m22;
        const bool fastMatrix =
            inv.isAffine() && f1 < 1e4 && f2 < 1e4 && f1 > (1.0 / 65536) && f2 > (1.0 / 65536) && std::fabs(dx) < 1e4 && std::fabs(dy) < 1e4;
        if (!fastMatrix) {
            return false;
        }
        const bool noShear = qt::scaleForTransform(matrix);
        if (!smooth && type == qt::TxScale) {
            double rx, ry, rw, rh;
            matrix.mapRect(0, 0, sw, sh, rx, ry, rw, rh);
            const int x1 = qt::round(rx);
            const int y1 = qt::round(ry);
            const int x2 = qt::round(rx + rw);
            const int y2 = qt::round(ry + rh);
            qt::fillRectSpans(x1, y1, x2 - x1, y2 - y1, dw, dh, spans);
            qt::spanRuns(spans, 512, true, runs);
        } else if (noShear || type == qt::TxScale) {
            const qt::PointF a = matrix.map({0, (0 + sh) * 0.5});
            const qt::PointF b = matrix.map({(sw + sw) * 0.5, (0 + sh) * 0.5});
            const double width = noShear ? double(sh) / sw : std::fabs((matrix.m22() * sh) / (matrix.m11() * sw));
            qt::rasterizeLine(a, b, width, smooth, {0, 0, dw - 1, dh - 1}, spans);
            qt::spanRuns(spans, 256, true, runs);
        } else {
            return false; // sheared: drawn as a path
        }
        m_mode = smooth ? Bilinear : Nearest;
        m_fdx = int(m11 * kFixedScale);
        m_fdy = int(m12 * kFixedScale);
        for (const qt::Run &run : runs) {
            const double cx = run.x + 0.5;
            const double cy = run.y + 0.5;
            // canUseFastMatrixPath()
            double fx = (m21 * cy + m11 * cx + dx) * kFixedScale;
            double fy = (m22 * cy + m12 * cx + dy) * kFixedScale;
            double minc = std::min(fx, fy);
            double maxc = std::max(fx, fy);
            fx += std::trunc(m11 * kFixedScale) * run.len;
            fy += std::trunc(m12 * kFixedScale) * run.len;
            minc = std::min(minc, std::min(fx, fy));
            maxc = std::max(maxc, std::max(fx, fy));
            if (!(minc >= double(INT_MIN) && maxc <= double(INT_MAX))) {
                return false;
            }
            int startX = int((m21 * cy + m11 * cx + dx) * kFixedScale);
            int startY = int((m22 * cy + m12 * cx + dy) * kFixedScale);
            if (smooth) {
                startX -= kHalfPoint;
                startY -= kHalfPoint;
            }
            starts.push_back(startX);
            starts.push_back(startY);
        }
    }

    // Spans and runs come row by row: per row, first span, span count, first run, run count
    m_table.assign(size_t(4) * size_t(dh), 0);
    m_spansOffset = int(m_table.size());
    for (size_t i = 0; i < spans.size(); ++i) {
        const qt::Span &span = spans[i];
        int *row = &m_table[size_t(4) * size_t(span.y)];
        if (row[1] == 0) {
            row[0] = int(i);
        }
        ++row[1];
        m_table.push_back(span.x);
        m_table.push_back(span.len);
        m_table.push_back(span.coverage);
    }
    m_runsOffset = int(m_table.size());
    for (size_t i = 0; i < runs.size(); ++i) {
        const qt::Run &run = runs[i];
        int *row = &m_table[size_t(4) * size_t(run.y)];
        if (row[3] == 0) {
            row[2] = int(i);
        }
        ++row[3];
        m_table.push_back(run.x);
        m_table.push_back(run.len);
        m_table.push_back(starts[2 * i]);
        m_table.push_back(starts[2 * i + 1]);
    }
    return true;
}

CUresult QtImageDrawing::draw(GpuScope &scope, CUdeviceptr source, CUdeviceptr under, uint32_t background, CUdeviceptr out) const
{
    DeviceBuffer tables;
    const size_t bytes = m_table.size() * sizeof(int);
    CUresult res = tables.allocate(bytes);
    if (res == CUDA_SUCCESS) {
        res = scope.upload(tables.ptr(), m_table.data(), bytes);
    }
    if (res == CUDA_SUCCESS) {
        const CUdeviceptr rows = tables.ptr();
        const CUdeviceptr spans = rows + CUdeviceptr(m_spansOffset) * sizeof(int);
        const CUdeviceptr runs = rows + CUdeviceptr(m_runsOffset) * sizeof(int);
        res = scope.launch2D(KernelId::gpufx_qt_draw_image, m_targetWidth, m_targetHeight, 32, 4, source, m_sourceWidth, m_sourceHeight, under, background, out,
                             m_targetWidth, m_targetHeight, rows, spans, runs, m_mode, m_fdx, m_fdy, m_opacity);
    }
    return res;
}

} // namespace gpufx
