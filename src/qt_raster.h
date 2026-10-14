/*
    SPDX-FileCopyrightText: 2016 The Qt Company Ltd.
    SPDX-FileCopyrightText: 2026 Kdenlive contributors
    SPDX-License-Identifier: LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only
*/

/*
 * The parts of Qt's raster paint engine that decide where QPainter draws a
 * transformed image, so that the GPU can draw the same pixels: QTransform
 * (with its cached transformation type, which selects the code paths),
 * QRasterizer::rasterizeLine() (the antialiased edges) and the way the
 * resulting spans are cut into the runs that Qt fetches, blends and stores
 * together. Adapted from qtransform.cpp, qrasterizer.cpp, qdrawhelper.cpp and
 * qpaintengine_raster.cpp (Qt 6), the arithmetic is kept as is.
 */

#pragma once

#include <vector>

namespace gpufx {
namespace qt {

enum TransformationType { TxNone = 0x00, TxTranslate = 0x01, TxScale = 0x02, TxRotate = 0x04, TxShear = 0x08, TxProject = 0x10 };

struct PointF
{
    double x = 0;
    double y = 0;
};

/** QTransform */
class Transform
{
public:
    Transform();
    Transform(double h11, double h12, double h13, double h21, double h22, double h23, double h31, double h32, double h33);

    Transform &translate(double dx, double dy);
    Transform &scale(double sx, double sy);
    /** Rotation around the Z axis, in degrees */
    Transform &rotate(double degrees);
    Transform operator*(const Transform &m) const;
    Transform inverted(bool *invertible = nullptr) const;
    TransformationType type() const;
    PointF map(PointF p) const;
    /** mapRect() of QRectF(x, y, w, h) */
    void mapRect(double x, double y, double w, double h, double &rx, double &ry, double &rw, double &rh) const;
    bool isAffine() const { return inlineType() < TxProject; }

    double m11() const { return m[0][0]; }
    double m12() const { return m[0][1]; }
    double m13() const { return m[0][2]; }
    double m21() const { return m[1][0]; }
    double m22() const { return m[1][1]; }
    double m23() const { return m[1][2]; }
    double m33() const { return m[2][2]; }
    double dx() const { return m[2][0]; }
    double dy() const { return m[2][1]; }

private:
    TransformationType inlineType() const;
    void doMap(double x, double y, double &nx, double &ny) const;
    Transform adjoint() const;
    double determinant() const;
    Transform &multiply(double num);

    double m[3][3];
    mutable int m_type;
    mutable int m_dirty;
};

/** qt_scaleForTransform(): true when the transform has no shear */
bool scaleForTransform(const Transform &transform);

/** qRound() */
int round(double d);

struct Span
{
    int x;
    int len;
    int y;
    int coverage;
};

struct Rect
{
    int left;
    int top;
    int right; // inclusive, like QRect::right()
    int bottom;
};

/** QRasterizer::rasterizeLine(a, b, width) without square cap, appends the
 *  spans in the order QSpanBuffer receives them (empty ones are dropped). */
void rasterizeLine(PointF a, PointF b, double width, bool antialiased, const Rect &clip, std::vector<Span> &spans);

/** One fetch, blend and store call of Qt's span handling: x .. x + len - 1 on row y. */
struct Run
{
    int x;
    int y;
    int len;
};

/** The calls made for spans coming in batches of batchSize (QSpanBuffer:
 *  256, fillRect_normalized(): 512): QT_THREAD_PARALLEL_FILLS splits every
 *  batch in segments, handleSpans() merges adjacent spans of a segment and
 *  cuts them in BufferSize pieces. When merge is false every span is handled
 *  on its own (blend_untransformed_generic_rgb64()). */
void spanRuns(const std::vector<Span> &spans, int batchSize, bool merge, std::vector<Run> &runs);

/** QRasterPaintEngine::drawImage(): the spans of fillRect_normalized(), one
 *  per row of the rectangle clipped to the device (width x height). */
void fillRectSpans(int x, int y, int w, int h, int deviceWidth, int deviceHeight, std::vector<Span> &spans);

} // namespace qt
} // namespace gpufx
