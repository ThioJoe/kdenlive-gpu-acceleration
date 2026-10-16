/*
    SPDX-FileCopyrightText: 2016 The Qt Company Ltd.
    SPDX-FileCopyrightText: 2026 ThioJoe
    SPDX-License-Identifier: LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only
*/

#include "qt_raster.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>

namespace gpufx {
namespace qt {

namespace {

bool fuzzyIsNull(double d)
{
    return std::fabs(d) <= 0.000000000001;
}

bool fuzzyCompare(double p1, double p2)
{
    return std::fabs(p1 - p2) * 1000000000000. <= std::min(std::fabs(p1), std::fabs(p2));
}

} // namespace

int round(double d)
{
    return int(d + std::copysign(0.5, d));
}

/* ==========================================================================
 * QTransform
 * ========================================================================== */

Transform::Transform()
    : m{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}
    , m_type(TxNone)
    , m_dirty(TxNone)
{
}

Transform::Transform(double h11, double h12, double h13, double h21, double h22, double h23, double h31, double h32, double h33)
    : m{{h11, h12, h13}, {h21, h22, h23}, {h31, h32, h33}}
    , m_type(TxNone)
    , m_dirty(TxProject)
{
}

TransformationType Transform::inlineType() const
{
    if (m_dirty == TxNone) {
        return TransformationType(m_type);
    }
    return type();
}

TransformationType Transform::type() const
{
    if (m_dirty == TxNone || m_dirty < m_type) {
        return TransformationType(m_type);
    }
    switch (TransformationType(m_dirty)) {
    case TxProject:
        if (!fuzzyIsNull(m[0][2]) || !fuzzyIsNull(m[1][2]) || !fuzzyIsNull(m[2][2] - 1)) {
            m_type = TxProject;
            break;
        }
        [[fallthrough]];
    case TxShear:
    case TxRotate:
        if (!fuzzyIsNull(m[0][1]) || !fuzzyIsNull(m[1][0])) {
            const double dot = m[0][0] * m[1][0] + m[0][1] * m[1][1];
            if (fuzzyIsNull(dot)) {
                m_type = TxRotate;
            } else {
                m_type = TxShear;
            }
            break;
        }
        [[fallthrough]];
    case TxScale:
        if (!fuzzyIsNull(m[0][0] - 1) || !fuzzyIsNull(m[1][1] - 1)) {
            m_type = TxScale;
            break;
        }
        [[fallthrough]];
    case TxTranslate:
        if (!fuzzyIsNull(m[2][0]) || !fuzzyIsNull(m[2][1])) {
            m_type = TxTranslate;
            break;
        }
        [[fallthrough]];
    case TxNone:
        m_type = TxNone;
        break;
    }
    m_dirty = TxNone;
    return TransformationType(m_type);
}

Transform &Transform::translate(double dx, double dy)
{
    if (dx == 0 && dy == 0) {
        return *this;
    }
    switch (inlineType()) {
    case TxNone:
        m[2][0] = dx;
        m[2][1] = dy;
        break;
    case TxTranslate:
        m[2][0] += dx;
        m[2][1] += dy;
        break;
    case TxScale:
        m[2][0] += dx * m[0][0];
        m[2][1] += dy * m[1][1];
        break;
    case TxProject:
        m[2][2] += dx * m[0][2] + dy * m[1][2];
        [[fallthrough]];
    case TxShear:
    case TxRotate:
        m[2][0] += dx * m[0][0] + dy * m[1][0];
        m[2][1] += dy * m[1][1] + dx * m[0][1];
        break;
    }
    if (m_dirty < TxTranslate) {
        m_dirty = TxTranslate;
    }
    return *this;
}

Transform &Transform::scale(double sx, double sy)
{
    if (sx == 1 && sy == 1) {
        return *this;
    }
    switch (inlineType()) {
    case TxNone:
    case TxTranslate:
        m[0][0] = sx;
        m[1][1] = sy;
        break;
    case TxProject:
        m[0][2] *= sx;
        m[1][2] *= sy;
        [[fallthrough]];
    case TxRotate:
    case TxShear:
        m[0][1] *= sx;
        m[1][0] *= sy;
        [[fallthrough]];
    case TxScale:
        m[0][0] *= sx;
        m[1][1] *= sy;
        break;
    }
    if (m_dirty < TxScale) {
        m_dirty = TxScale;
    }
    return *this;
}

Transform &Transform::rotate(double a)
{
    double sina = 0;
    double cosa = 0;
    if (a == 90. || a == -270.) {
        sina = 1.;
    } else if (a == 270. || a == -90.) {
        sina = -1.;
    } else if (a == 180.) {
        cosa = -1.;
    } else {
        // qDegreesToRadians(), M_PI as defined by qmath.h
        const double b = a * (3.14159265358979323846 / 180);
        sina = std::sin(b);
        cosa = std::cos(b);
    }
    switch (inlineType()) {
    case TxNone:
    case TxTranslate:
        m[0][0] = cosa;
        m[0][1] = sina;
        m[1][0] = -sina;
        m[1][1] = cosa;
        break;
    case TxScale: {
        const double tm11 = cosa * m[0][0];
        const double tm12 = sina * m[1][1];
        const double tm21 = -sina * m[0][0];
        const double tm22 = cosa * m[1][1];
        m[0][0] = tm11;
        m[0][1] = tm12;
        m[1][0] = tm21;
        m[1][1] = tm22;
        break;
    }
    case TxProject: {
        const double tm13 = cosa * m[0][2] + sina * m[1][2];
        const double tm23 = -sina * m[0][2] + cosa * m[1][2];
        m[0][2] = tm13;
        m[1][2] = tm23;
        [[fallthrough]];
    }
    case TxRotate:
    case TxShear: {
        const double tm11 = cosa * m[0][0] + sina * m[1][0];
        const double tm12 = cosa * m[0][1] + sina * m[1][1];
        const double tm21 = -sina * m[0][0] + cosa * m[1][0];
        const double tm22 = -sina * m[0][1] + cosa * m[1][1];
        m[0][0] = tm11;
        m[0][1] = tm12;
        m[1][0] = tm21;
        m[1][1] = tm22;
        break;
    }
    }
    if (m_dirty < TxRotate) {
        m_dirty = TxRotate;
    }
    return *this;
}

Transform Transform::operator*(const Transform &o) const
{
    const TransformationType otherType = o.inlineType();
    if (otherType == TxNone) {
        return *this;
    }
    const TransformationType thisType = inlineType();
    if (thisType == TxNone) {
        return o;
    }
    Transform t;
    const TransformationType type = std::max(thisType, otherType);
    switch (type) {
    case TxNone:
        break;
    case TxTranslate:
        t.m[2][0] = m[2][0] + o.m[2][0];
        t.m[2][1] = m[2][1] + o.m[2][1];
        break;
    case TxScale: {
        const double m11 = m[0][0] * o.m[0][0];
        const double m22 = m[1][1] * o.m[1][1];
        const double m31 = m[2][0] * o.m[0][0] + o.m[2][0];
        const double m32 = m[2][1] * o.m[1][1] + o.m[2][1];
        t.m[0][0] = m11;
        t.m[1][1] = m22;
        t.m[2][0] = m31;
        t.m[2][1] = m32;
        break;
    }
    case TxRotate:
    case TxShear: {
        const double m11 = m[0][0] * o.m[0][0] + m[0][1] * o.m[1][0];
        const double m12 = m[0][0] * o.m[0][1] + m[0][1] * o.m[1][1];
        const double m21 = m[1][0] * o.m[0][0] + m[1][1] * o.m[1][0];
        const double m22 = m[1][0] * o.m[0][1] + m[1][1] * o.m[1][1];
        const double m31 = m[2][0] * o.m[0][0] + m[2][1] * o.m[1][0] + o.m[2][0];
        const double m32 = m[2][0] * o.m[0][1] + m[2][1] * o.m[1][1] + o.m[2][1];
        t.m[0][0] = m11;
        t.m[0][1] = m12;
        t.m[1][0] = m21;
        t.m[1][1] = m22;
        t.m[2][0] = m31;
        t.m[2][1] = m32;
        break;
    }
    case TxProject: {
        const double m11 = m[0][0] * o.m[0][0] + m[0][1] * o.m[1][0] + m[0][2] * o.m[2][0];
        const double m12 = m[0][0] * o.m[0][1] + m[0][1] * o.m[1][1] + m[0][2] * o.m[2][1];
        const double m13 = m[0][0] * o.m[0][2] + m[0][1] * o.m[1][2] + m[0][2] * o.m[2][2];
        const double m21 = m[1][0] * o.m[0][0] + m[1][1] * o.m[1][0] + m[1][2] * o.m[2][0];
        const double m22 = m[1][0] * o.m[0][1] + m[1][1] * o.m[1][1] + m[1][2] * o.m[2][1];
        const double m23 = m[1][0] * o.m[0][2] + m[1][1] * o.m[1][2] + m[1][2] * o.m[2][2];
        const double m31 = m[2][0] * o.m[0][0] + m[2][1] * o.m[1][0] + m[2][2] * o.m[2][0];
        const double m32 = m[2][0] * o.m[0][1] + m[2][1] * o.m[1][1] + m[2][2] * o.m[2][1];
        const double m33 = m[2][0] * o.m[0][2] + m[2][1] * o.m[1][2] + m[2][2] * o.m[2][2];
        t.m[0][0] = m11;
        t.m[0][1] = m12;
        t.m[0][2] = m13;
        t.m[1][0] = m21;
        t.m[1][1] = m22;
        t.m[1][2] = m23;
        t.m[2][0] = m31;
        t.m[2][1] = m32;
        t.m[2][2] = m33;
        break;
    }
    }
    t.m_dirty = type;
    t.m_type = type;
    return t;
}

Transform Transform::adjoint() const
{
    const double h11 = m[1][1] * m[2][2] - m[1][2] * m[2][1];
    const double h21 = m[1][2] * m[2][0] - m[1][0] * m[2][2];
    const double h31 = m[1][0] * m[2][1] - m[1][1] * m[2][0];
    const double h12 = m[0][2] * m[2][1] - m[0][1] * m[2][2];
    const double h22 = m[0][0] * m[2][2] - m[0][2] * m[2][0];
    const double h32 = m[0][1] * m[2][0] - m[0][0] * m[2][1];
    const double h13 = m[0][1] * m[1][2] - m[0][2] * m[1][1];
    const double h23 = m[0][2] * m[1][0] - m[0][0] * m[1][2];
    const double h33 = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    return Transform(h11, h12, h13, h21, h22, h23, h31, h32, h33);
}

double Transform::determinant() const
{
    return m[0][0] * (m[2][2] * m[1][1] - m[2][1] * m[1][2]) - m[1][0] * (m[2][2] * m[0][1] - m[2][1] * m[0][2]) +
           m[2][0] * (m[1][2] * m[0][1] - m[1][1] * m[0][2]);
}

Transform &Transform::multiply(double num)
{
    if (num == 1.) {
        return *this;
    }
    for (auto &row : m) {
        for (double &value : row) {
            value *= num;
        }
    }
    if (m_dirty < TxScale) {
        m_dirty = TxScale;
    }
    return *this;
}

Transform Transform::inverted(bool *invertible) const
{
    Transform invert;
    bool inv = true;
    switch (inlineType()) {
    case TxNone:
        break;
    case TxTranslate:
        invert.m[2][0] = -m[2][0];
        invert.m[2][1] = -m[2][1];
        break;
    case TxScale:
        inv = !fuzzyIsNull(m[0][0]);
        inv &= !fuzzyIsNull(m[1][1]);
        if (inv) {
            invert.m[0][0] = 1. / m[0][0];
            invert.m[1][1] = 1. / m[1][1];
            invert.m[2][0] = -m[2][0] * invert.m[0][0];
            invert.m[2][1] = -m[2][1] * invert.m[1][1];
        }
        break;
    default: {
        const double det = determinant();
        inv = !fuzzyIsNull(det);
        if (inv) {
            // adjoint() / det
            invert = adjoint();
            if (det != 0) {
                invert.multiply(1 / det);
            }
        }
        break;
    }
    }
    if (invertible) {
        *invertible = inv;
    }
    if (inv) {
        // inverting doesn't change the type
        invert.m_type = m_type;
        invert.m_dirty = m_dirty;
    }
    return invert;
}

void Transform::doMap(double x, double y, double &nx, double &ny) const
{
    const TransformationType t = inlineType();
    switch (t) {
    case TxNone:
        nx = x;
        ny = y;
        return;
    case TxTranslate:
        nx = x + m[2][0];
        ny = y + m[2][1];
        return;
    case TxScale:
        nx = m[0][0] * x + m[2][0];
        ny = m[1][1] * y + m[2][1];
        return;
    case TxRotate:
    case TxShear:
    case TxProject:
        nx = m[0][0] * x + m[1][0] * y + m[2][0];
        ny = m[0][1] * x + m[1][1] * y + m[2][1];
        if (t == TxProject) {
            double w = (m[0][2] * x + m[1][2] * y + m[2][2]);
            if (w < 0.000001) {
                w = 0.000001;
            }
            w = 1. / w;
            nx *= w;
            ny *= w;
        }
        return;
    }
}

PointF Transform::map(PointF p) const
{
    PointF r;
    doMap(p.x, p.y, r.x, r.y);
    return r;
}

void Transform::mapRect(double x, double y, double w, double h, double &rx, double &ry, double &rw, double &rh) const
{
    const TransformationType t = inlineType();
    if (t <= TxTranslate) {
        rx = x + m[2][0];
        ry = y + m[2][1];
        rw = w;
        rh = h;
        return;
    }
    if (t <= TxScale) {
        rx = m[0][0] * x + m[2][0];
        ry = m[1][1] * y + m[2][1];
        rw = m[0][0] * w;
        rh = m[1][1] * h;
        if (rw < 0) {
            rw = -rw;
            rx -= rw;
        }
        if (rh < 0) {
            rh = -rh;
            ry -= rh;
        }
        return;
    }
    double px = 0;
    double py = 0;
    doMap(x, y, px, py);
    double xmin = px;
    double ymin = py;
    double xmax = px;
    double ymax = py;
    doMap(x + w, y, px, py);
    xmin = std::min(xmin, px);
    ymin = std::min(ymin, py);
    xmax = std::max(xmax, px);
    ymax = std::max(ymax, py);
    doMap(x + w, y + h, px, py);
    xmin = std::min(xmin, px);
    ymin = std::min(ymin, py);
    xmax = std::max(xmax, px);
    ymax = std::max(ymax, py);
    doMap(x, y + h, px, py);
    xmin = std::min(xmin, px);
    ymin = std::min(ymin, py);
    xmax = std::max(xmax, px);
    ymax = std::max(ymax, py);
    rx = xmin;
    ry = ymin;
    rw = xmax - xmin;
    rh = ymax - ymin;
}

bool scaleForTransform(const Transform &transform)
{
    const TransformationType type = transform.type();
    if (type <= TxTranslate) {
        return true;
    } else if (type == TxScale) {
        return fuzzyCompare(std::fabs(transform.m11()), std::fabs(transform.m22()));
    }
    // rotate then scale: compare columns
    const double xScale1 = transform.m11() * transform.m11() + transform.m21() * transform.m21();
    const double yScale1 = transform.m12() * transform.m12() + transform.m22() * transform.m22();
    // scale then rotate: compare rows
    const double xScale2 = transform.m11() * transform.m11() + transform.m12() * transform.m12();
    const double yScale2 = transform.m21() * transform.m21() + transform.m22() * transform.m22();
    if (std::fabs(xScale1 - yScale1) > std::fabs(xScale2 - yScale2)) {
        return type == TxRotate && fuzzyCompare(xScale1, yScale1);
    }
    return type == TxRotate && fuzzyCompare(xScale2, yScale2);
}

/* ==========================================================================
 * QRasterizer::rasterizeLine()
 * ========================================================================== */

namespace {

typedef int64_t QScFixed;

constexpr QScFixed QScFixedFactor = 65536;

inline QScFixed intToFixed(int i)
{
    return QScFixed(i) * (1 << 16);
}

inline int fixedToInt(QScFixed i)
{
    return int(i >> 16);
}

inline QScFixed fixedMultiply(QScFixed x, QScFixed y)
{
    return QScFixed((static_cast<long long>(x) * static_cast<long long>(y)) >> 16);
}

inline QScFixed fixedFastMultiply(QScFixed x, QScFixed y)
{
    return (x * y) >> 16;
}

/** QSpanBuffer::addSpan() */
inline void addSpan(std::vector<Span> &spans, int x, int len, int y, int coverage)
{
    if (!coverage || !len) {
        return;
    }
    spans.push_back({x, len, y, coverage});
}

QScFixed intersectPixelFP(int x, QScFixed top, QScFixed bottom, QScFixed leftIntersectX, QScFixed rightIntersectX, QScFixed slope, QScFixed invSlope)
{
    const QScFixed leftX = intToFixed(x);
    const QScFixed rightX = intToFixed(x) + QScFixedFactor;

    QScFixed leftIntersectY = 0;
    QScFixed rightIntersectY = 0;
    auto computeIntersectY = [&]() {
        if (slope > 0) {
            leftIntersectY = top + fixedMultiply(leftX - leftIntersectX, invSlope);
            rightIntersectY = leftIntersectY + invSlope;
        } else {
            leftIntersectY = top + fixedMultiply(leftX - rightIntersectX, invSlope);
            rightIntersectY = leftIntersectY + invSlope;
        }
    };

    if (leftIntersectX >= leftX && rightIntersectX <= rightX) {
        return fixedMultiply(bottom - top, leftIntersectX - leftX + ((rightIntersectX - leftIntersectX) >> 1));
    } else if (leftIntersectX >= rightX) {
        return bottom - top;
    } else if (leftIntersectX >= leftX) {
        computeIntersectY();
        if (slope > 0) {
            return (bottom - top) - fixedFastMultiply((rightX - leftIntersectX) >> 1, rightIntersectY - top);
        } else {
            return (bottom - top) - fixedFastMultiply((rightX - leftIntersectX) >> 1, bottom - rightIntersectY);
        }
    } else if (rightIntersectX <= leftX) {
        return 0;
    } else if (rightIntersectX <= rightX) {
        computeIntersectY();
        if (slope > 0) {
            return fixedFastMultiply((rightIntersectX - leftX) >> 1, bottom - leftIntersectY);
        } else {
            return fixedFastMultiply((rightIntersectX - leftX) >> 1, leftIntersectY - top);
        }
    } else {
        computeIntersectY();
        if (slope > 0) {
            return (bottom - rightIntersectY) + ((rightIntersectY - leftIntersectY) >> 1);
        } else {
            return (rightIntersectY - top) + ((leftIntersectY - rightIntersectY) >> 1);
        }
    }
}

inline bool q26Dot6Compare(double p1, double p2)
{
    return int((p2 - p1) * 64.) == 0;
}

inline PointF snapTo26Dot6Grid(PointF p)
{
    return {std::floor(p.x * 64) * (1 / double(64)), std::floor(p.y * 64) * (1 / double(64))};
}

inline double qSafeDivide(double x, double y)
{
    if (y == 0) {
        return x > 0 ? 1e20 : -1e20;
    }
    return x / y;
}

inline QScFixed qSafeFloatToQScFixed(double x)
{
    const double tmp = x * QScFixedFactor;
    if (tmp > double(INT_MAX) * QScFixedFactor) {
        return QScFixed(INT_MAX) * QScFixedFactor;
    } else if (tmp < double(INT_MIN) * QScFixedFactor) {
        return QScFixed(INT_MIN) * QScFixedFactor;
    }
    return QScFixed(tmp);
}

// returns true if the whole line gets clipped away
bool qClipLine(PointF *pt1, PointF *pt2, double xmin, double xmax, double ymin, double ymax)
{
    double &x1 = pt1->x;
    double &y1 = pt1->y;
    double &x2 = pt2->x;
    double &y2 = pt2->y;

    if (!std::isfinite(x1) || !std::isfinite(y1) || !std::isfinite(x2) || !std::isfinite(y2)) {
        return true;
    }
    if (x1 < xmin) {
        if (x2 <= xmin) {
            return true;
        }
        y1 += (y2 - y1) / (x2 - x1) * (xmin - x1);
        x1 = xmin;
    } else if (x1 > xmax) {
        if (x2 >= xmax) {
            return true;
        }
        y1 += (y2 - y1) / (x2 - x1) * (xmax - x1);
        x1 = xmax;
    }
    if (x2 < xmin) {
        y2 += (y2 - y1) / (x2 - x1) * (xmin - x2);
        x2 = xmin;
    } else if (x2 > xmax) {
        y2 += (y2 - y1) / (x2 - x1) * (xmax - x2);
        x2 = xmax;
    }
    if (y1 < ymin) {
        if (y2 <= ymin) {
            return true;
        }
        x1 += (x2 - x1) / (y2 - y1) * (ymin - y1);
        y1 = ymin;
    } else if (y1 > ymax) {
        if (y2 >= ymax) {
            return true;
        }
        x1 += (x2 - x1) / (y2 - y1) * (ymax - y1);
        y1 = ymax;
    }
    if (y2 < ymin) {
        x2 += (x2 - x1) / (y2 - y1) * (ymin - y2);
        y2 = ymin;
    } else if (y2 > ymax) {
        x2 += (x2 - x1) / (y2 - y1) * (ymax - y2);
        y2 = ymax;
    }
    return false;
}

// QPointF::operator==()
bool pointsEqual(PointF p1, PointF p2)
{
    const bool xEqual = (!p1.x || !p2.x) ? fuzzyIsNull(p1.x - p2.x) : fuzzyCompare(p1.x, p2.x);
    const bool yEqual = (!p1.y || !p2.y) ? fuzzyIsNull(p1.y - p2.y) : fuzzyCompare(p1.y, p2.y);
    return xEqual && yEqual;
}

template <typename T> inline T bound(T min, T value, T max)
{
    return std::max(min, std::min(max, value));
}

} // namespace

void rasterizeLine(PointF a, PointF b, double width, bool antialiased, const Rect &clipRect, std::vector<Span> &buffer)
{
    if (pointsEqual(a, b) || !(width > 0.0) || clipRect.right < clipRect.left || clipRect.bottom < clipRect.top) {
        return;
    }

    PointF pa = a;
    PointF pb = b;

    // QPointF offs = QPointF(qAbs(b.y() - a.y()), qAbs(b.x() - a.x())) * width * 0.5
    const double offsX = std::fabs(b.y - a.y) * width * 0.5;
    const double offsY = std::fabs(b.x - a.x) * width * 0.5;
    // QRectF clip(clipRect.topLeft() - offs, clipRect.bottomRight() + QPoint(1, 1) + offs)
    const double clipX = clipRect.left - offsX;
    const double clipY = clipRect.top - offsY;
    const double clipW = (clipRect.right + 1 + offsX) - clipX;
    const double clipH = (clipRect.bottom + 1 + offsY) - clipY;
    if (qClipLine(&pa, &pb, clipX, clipX + clipW, clipY, clipY + clipH)) {
        return;
    }

    {
        // old delta
        const double d0x = a.x - b.x;
        const double d0y = a.y - b.y;
        const double w0 = d0x * d0x + d0y * d0y;
        // new delta
        const double dx = pa.x - pb.x;
        const double dy = pa.y - pb.y;
        const double w = dx * dx + dy * dy;
        if (w == 0) {
            return;
        }
        // adjust width which is given relative to |b - a|
        width *= std::sqrt(w0 / w);
    }

    if (q26Dot6Compare(pa.y, pb.y)) {
        const double x = (pa.x + pb.x) * 0.5f;
        const double dx = std::fabs(pb.x - pa.x) * 0.5f;
        const double y = pa.y;
        const double dy = width * dx;
        pa = {x, y - dy};
        pb = {x, y + dy};
        width = 1 / width;
    }

    if (q26Dot6Compare(pa.x, pb.x)) {
        if (pa.y > pb.y) {
            std::swap(pa, pb);
        }
        const double dy = pb.y - pa.y;
        const double halfWidth = 0.5f * width * dy;

        double left = pa.x - halfWidth;
        double right = pa.x + halfWidth;

        left = bound(double(clipRect.left), left, double(clipRect.right + 1));
        right = bound(double(clipRect.left), right, double(clipRect.right + 1));

        pa.y = bound(double(clipRect.top), pa.y, double(clipRect.bottom + 1));
        pb.y = bound(double(clipRect.top), pb.y, double(clipRect.bottom + 1));

        if (q26Dot6Compare(left, right) || q26Dot6Compare(pa.y, pb.y)) {
            return;
        }

        if (antialiased) {
            const int iLeft = int(left);
            const int iRight = int(right);
            const QScFixed leftWidth = intToFixed(iLeft + 1) - qSafeFloatToQScFixed(left);
            const QScFixed rightWidth = qSafeFloatToQScFixed(right) - intToFixed(iRight);

            QScFixed coverage[3];
            int x[3];
            int len[3];

            int n = 1;
            if (iLeft == iRight) {
                if (leftWidth == QScFixedFactor) {
                    coverage[0] = rightWidth * 255;
                } else {
                    coverage[0] = (rightWidth + leftWidth - QScFixedFactor) * 255;
                }
                x[0] = iLeft;
                len[0] = 1;
            } else {
                coverage[0] = leftWidth * 255;
                x[0] = iLeft;
                len[0] = 1;
                if (leftWidth == QScFixedFactor) {
                    len[0] = iRight - iLeft;
                } else if (iRight - iLeft > 1) {
                    coverage[1] = intToFixed(255);
                    x[1] = iLeft + 1;
                    len[1] = iRight - iLeft - 1;
                    ++n;
                }
                if (rightWidth) {
                    coverage[n] = rightWidth * 255;
                    x[n] = iRight;
                    len[n] = 1;
                    ++n;
                }
            }

            const QScFixed iTopFP = intToFixed(int(pa.y));
            const QScFixed iBottomFP = intToFixed(int(pb.y));
            const QScFixed yPa = qSafeFloatToQScFixed(pa.y);
            const QScFixed yPb = qSafeFloatToQScFixed(pb.y);
            for (QScFixed yFP = iTopFP; yFP <= iBottomFP; yFP += QScFixedFactor) {
                const QScFixed rowHeight = std::min(yFP + QScFixedFactor, yPb) - std::max(yFP, yPa);
                const int y = fixedToInt(yFP);
                if (y > clipRect.bottom) {
                    break;
                }
                for (int i = 0; i < n; ++i) {
                    addSpan(buffer, x[i], len[i], y, fixedToInt(fixedMultiply(rowHeight, coverage[i])));
                }
            }
        } else { // aliased
            const int iTop = int(pa.y + 0.5f);
            const int iBottom = pb.y < 0.5f ? -1 : int(pb.y - 0.5f);
            const int iLeft = int(left + 0.5f);
            const int iRight = right < 0.5f ? -1 : int(right - 0.5f);

            const int iWidth = iRight - iLeft + 1;
            for (int y = iTop; y <= iBottom; ++y) {
                addSpan(buffer, iLeft, iWidth, y, 255);
            }
        }
    } else {
        if (pa.y > pb.y) {
            std::swap(pa, pb);
        }

        // QPointF delta = pb - pa; delta *= 0.5f * width;
        const double scaleDelta = 0.5f * width;
        const double deltaX = (pb.x - pa.x) * scaleDelta;
        const double deltaY = (pb.y - pa.y) * scaleDelta;
        const PointF perp = {deltaY, -deltaX};

        PointF top;
        PointF left;
        PointF right;
        PointF bottom;

        if (pa.x < pb.x) {
            top = {pa.x + perp.x, pa.y + perp.y};
            left = {pa.x - perp.x, pa.y - perp.y};
            right = {pb.x + perp.x, pb.y + perp.y};
            bottom = {pb.x - perp.x, pb.y - perp.y};
        } else {
            top = {pa.x - perp.x, pa.y - perp.y};
            left = {pb.x - perp.x, pb.y - perp.y};
            right = {pa.x + perp.x, pa.y + perp.y};
            bottom = {pb.x + perp.x, pb.y + perp.y};
        }

        top = snapTo26Dot6Grid(top);
        bottom = snapTo26Dot6Grid(bottom);
        left = snapTo26Dot6Grid(left);
        right = snapTo26Dot6Grid(right);

        const double topBound = bound(double(clipRect.top), top.y, double(clipRect.bottom));
        const double bottomBound = bound(double(clipRect.top), bottom.y, double(clipRect.bottom));

        const PointF topLeftEdge = {left.x - top.x, left.y - top.y};
        const PointF topRightEdge = {right.x - top.x, right.y - top.y};
        const PointF bottomLeftEdge = {bottom.x - left.x, bottom.y - left.y};
        const PointF bottomRightEdge = {bottom.x - right.x, bottom.y - right.y};

        const double topLeftSlope = qSafeDivide(topLeftEdge.x, topLeftEdge.y);
        const double bottomLeftSlope = qSafeDivide(bottomLeftEdge.x, bottomLeftEdge.y);

        const double topRightSlope = qSafeDivide(topRightEdge.x, topRightEdge.y);
        const double bottomRightSlope = qSafeDivide(bottomRightEdge.x, bottomRightEdge.y);

        const QScFixed topLeftSlopeFP = qSafeFloatToQScFixed(topLeftSlope);
        const QScFixed topRightSlopeFP = qSafeFloatToQScFixed(topRightSlope);

        const QScFixed bottomLeftSlopeFP = qSafeFloatToQScFixed(bottomLeftSlope);
        const QScFixed bottomRightSlopeFP = qSafeFloatToQScFixed(bottomRightSlope);

        const QScFixed invTopLeftSlopeFP = qSafeFloatToQScFixed(qSafeDivide(1, topLeftSlope));
        const QScFixed invTopRightSlopeFP = qSafeFloatToQScFixed(qSafeDivide(1, topRightSlope));

        const QScFixed invBottomLeftSlopeFP = qSafeFloatToQScFixed(qSafeDivide(1, bottomLeftSlope));
        const QScFixed invBottomRightSlopeFP = qSafeFloatToQScFixed(qSafeDivide(1, bottomRightSlope));

        if (antialiased) {
            const QScFixed iTopFP = intToFixed(int(topBound));
            const QScFixed iLeftFP = intToFixed(int(left.y));
            const QScFixed iRightFP = intToFixed(int(right.y));
            const QScFixed iBottomFP = intToFixed(int(bottomBound));

            QScFixed leftIntersectAf = qSafeFloatToQScFixed(top.x + (int(topBound) - top.y) * topLeftSlope);
            QScFixed rightIntersectAf = qSafeFloatToQScFixed(top.x + (int(topBound) - top.y) * topRightSlope);
            QScFixed leftIntersectBf = 0;
            QScFixed rightIntersectBf = 0;

            if (iLeftFP < iTopFP) {
                leftIntersectBf = qSafeFloatToQScFixed(left.x + (int(topBound) - left.y) * bottomLeftSlope);
            }
            if (iRightFP < iTopFP) {
                rightIntersectBf = qSafeFloatToQScFixed(right.x + (int(topBound) - right.y) * bottomRightSlope);
            }

            QScFixed rowTop, rowBottomLeft, rowBottomRight, rowTopLeft, rowTopRight, rowBottom;
            QScFixed topLeftIntersectAf, topLeftIntersectBf, topRightIntersectAf, topRightIntersectBf;
            QScFixed bottomLeftIntersectAf, bottomLeftIntersectBf, bottomRightIntersectAf, bottomRightIntersectBf;

            int leftMin, leftMax, rightMin, rightMax;

            const QScFixed yTopFP = qSafeFloatToQScFixed(top.y);
            const QScFixed yLeftFP = qSafeFloatToQScFixed(left.y);
            const QScFixed yRightFP = qSafeFloatToQScFixed(right.y);
            const QScFixed yBottomFP = qSafeFloatToQScFixed(bottom.y);

            rowTop = std::max(iTopFP, yTopFP);
            topLeftIntersectAf = leftIntersectAf + fixedMultiply(topLeftSlopeFP, rowTop - iTopFP);
            topRightIntersectAf = rightIntersectAf + fixedMultiply(topRightSlopeFP, rowTop - iTopFP);

            QScFixed yFP = iTopFP;
            while (yFP <= iBottomFP) {
                rowBottomLeft = std::min(yFP + QScFixedFactor, yLeftFP);
                rowBottomRight = std::min(yFP + QScFixedFactor, yRightFP);
                rowTopLeft = std::max(yFP, yLeftFP);
                rowTopRight = std::max(yFP, yRightFP);
                rowBottom = std::min(yFP + QScFixedFactor, yBottomFP);

                if (yFP == iLeftFP) {
                    const int y = fixedToInt(yFP);
                    leftIntersectBf = qSafeFloatToQScFixed(left.x + (y - left.y) * bottomLeftSlope);
                    topLeftIntersectBf = leftIntersectBf + fixedMultiply(bottomLeftSlopeFP, rowTopLeft - yFP);
                    bottomLeftIntersectAf = leftIntersectAf + fixedMultiply(topLeftSlopeFP, rowBottomLeft - yFP);
                } else {
                    topLeftIntersectBf = leftIntersectBf;
                    bottomLeftIntersectAf = leftIntersectAf + topLeftSlopeFP;
                }

                if (yFP == iRightFP) {
                    const int y = fixedToInt(yFP);
                    rightIntersectBf = qSafeFloatToQScFixed(right.x + (y - right.y) * bottomRightSlope);
                    topRightIntersectBf = rightIntersectBf + fixedMultiply(bottomRightSlopeFP, rowTopRight - yFP);
                    bottomRightIntersectAf = rightIntersectAf + fixedMultiply(topRightSlopeFP, rowBottomRight - yFP);
                } else {
                    topRightIntersectBf = rightIntersectBf;
                    bottomRightIntersectAf = rightIntersectAf + topRightSlopeFP;
                }

                if (yFP == iBottomFP) {
                    bottomLeftIntersectBf = leftIntersectBf + fixedMultiply(bottomLeftSlopeFP, rowBottom - yFP);
                    bottomRightIntersectBf = rightIntersectBf + fixedMultiply(bottomRightSlopeFP, rowBottom - yFP);
                } else {
                    bottomLeftIntersectBf = leftIntersectBf + bottomLeftSlopeFP;
                    bottomRightIntersectBf = rightIntersectBf + bottomRightSlopeFP;
                }

                if (yFP < iLeftFP) {
                    leftMin = fixedToInt(bottomLeftIntersectAf);
                    leftMax = fixedToInt(topLeftIntersectAf);
                } else if (yFP == iLeftFP) {
                    leftMin = fixedToInt(std::max(bottomLeftIntersectAf, topLeftIntersectBf));
                    leftMax = fixedToInt(std::max(topLeftIntersectAf, bottomLeftIntersectBf));
                } else {
                    leftMin = fixedToInt(topLeftIntersectBf);
                    leftMax = fixedToInt(bottomLeftIntersectBf);
                }

                leftMin = bound(clipRect.left, leftMin, clipRect.right);
                leftMax = bound(clipRect.left, leftMax, clipRect.right);

                if (yFP < iRightFP) {
                    rightMin = fixedToInt(topRightIntersectAf);
                    rightMax = fixedToInt(bottomRightIntersectAf);
                } else if (yFP == iRightFP) {
                    rightMin = fixedToInt(std::min(topRightIntersectAf, bottomRightIntersectBf));
                    rightMax = fixedToInt(std::min(bottomRightIntersectAf, topRightIntersectBf));
                } else {
                    rightMin = fixedToInt(bottomRightIntersectBf);
                    rightMax = fixedToInt(topRightIntersectBf);
                }

                rightMin = bound(clipRect.left, rightMin, clipRect.right);
                rightMax = bound(clipRect.left, rightMax, clipRect.right);

                if (leftMax > rightMax) {
                    leftMax = rightMax;
                }
                if (rightMin < leftMin) {
                    rightMin = leftMin;
                }

                const QScFixed rowHeight = rowBottom - rowTop;

                int x = leftMin;
                while (x <= leftMax) {
                    QScFixed excluded = 0;

                    if (yFP <= iLeftFP && rowBottomLeft > rowTop) {
                        excluded += intersectPixelFP(x, rowTop, rowBottomLeft, bottomLeftIntersectAf, topLeftIntersectAf, topLeftSlopeFP, invTopLeftSlopeFP);
                    }
                    if (yFP >= iLeftFP && rowBottom > rowTopLeft) {
                        excluded +=
                            intersectPixelFP(x, rowTopLeft, rowBottom, topLeftIntersectBf, bottomLeftIntersectBf, bottomLeftSlopeFP, invBottomLeftSlopeFP);
                    }
                    if (x >= rightMin) {
                        if (yFP <= iRightFP && rowBottomRight > rowTop) {
                            excluded += (rowBottomRight - rowTop) - intersectPixelFP(x, rowTop, rowBottomRight, topRightIntersectAf, bottomRightIntersectAf,
                                                                                     topRightSlopeFP, invTopRightSlopeFP);
                        }
                        if (yFP >= iRightFP && rowBottom > rowTopRight) {
                            excluded += (rowBottom - rowTopRight) - intersectPixelFP(x, rowTopRight, rowBottom, bottomRightIntersectBf, topRightIntersectBf,
                                                                                     bottomRightSlopeFP, invBottomRightSlopeFP);
                        }
                    }

                    const QScFixed coverage = rowHeight - excluded;
                    addSpan(buffer, x, 1, fixedToInt(yFP), fixedToInt(255 * coverage));
                    ++x;
                }
                if (x < rightMin) {
                    addSpan(buffer, x, rightMin - x, fixedToInt(yFP), fixedToInt(255 * rowHeight));
                    x = rightMin;
                }
                while (x <= rightMax) {
                    QScFixed excluded = 0;
                    if (yFP <= iRightFP && rowBottomRight > rowTop) {
                        excluded += (rowBottomRight - rowTop) - intersectPixelFP(x, rowTop, rowBottomRight, topRightIntersectAf, bottomRightIntersectAf,
                                                                                 topRightSlopeFP, invTopRightSlopeFP);
                    }
                    if (yFP >= iRightFP && rowBottom > rowTopRight) {
                        excluded += (rowBottom - rowTopRight) - intersectPixelFP(x, rowTopRight, rowBottom, bottomRightIntersectBf, topRightIntersectBf,
                                                                                 bottomRightSlopeFP, invBottomRightSlopeFP);
                    }

                    const QScFixed coverage = rowHeight - excluded;
                    addSpan(buffer, x, 1, fixedToInt(yFP), fixedToInt(255 * coverage));
                    ++x;
                }

                leftIntersectAf += topLeftSlopeFP;
                leftIntersectBf += bottomLeftSlopeFP;
                rightIntersectAf += topRightSlopeFP;
                rightIntersectBf += bottomRightSlopeFP;
                topLeftIntersectAf = leftIntersectAf;
                topRightIntersectAf = rightIntersectAf;

                yFP += QScFixedFactor;
                rowTop = yFP;
            }
        } else { // aliased
            const int iTop = int(top.y + 0.5f);
            const int iLeft = left.y < 0.5f ? -1 : int(left.y - 0.5f);
            int iRight = right.y < 0.5f ? -1 : int(right.y - 0.5f);
            int iBottom = bottom.y < 0.5f ? -1 : int(bottom.y - 0.5f);
            int iMiddle = std::min(iLeft, iRight);
            int iLeftSegment = iLeft;

            QScFixed leftIntersectAf = qSafeFloatToQScFixed(top.x + 0.5f + (iTop + 0.5f - top.y) * topLeftSlope);
            QScFixed leftIntersectBf = qSafeFloatToQScFixed(left.x + 0.5f + (iLeft + 1.5f - left.y) * bottomLeftSlope);
            QScFixed rightIntersectAf = qSafeFloatToQScFixed(top.x - 0.5f + (iTop + 0.5f - top.y) * topRightSlope);
            QScFixed rightIntersectBf = qSafeFloatToQScFixed(right.x - 0.5f + (iRight + 1.5f - right.y) * bottomRightSlope);

            int ny;
            int y = iTop;
            auto segment = [&](int &next, QScFixed &li, QScFixed &ri, QScFixed ls, QScFixed rs) {
                ny = std::min(next + 1, clipRect.top);
                if (y < ny) {
                    li += ls * (ny - y);
                    ri += rs * (ny - y);
                    y = ny;
                }
                if (next > clipRect.bottom) {
                    next = clipRect.bottom;
                }
                for (; y <= next; ++y) {
                    const int x1 = std::max(fixedToInt(li), clipRect.left);
                    const int x2 = std::min(fixedToInt(ri), clipRect.right);
                    if (x2 >= x1) {
                        addSpan(buffer, x1, x2 - x1 + 1, y, 255);
                    }
                    li += ls;
                    ri += rs;
                }
            };
            segment(iMiddle, leftIntersectAf, rightIntersectAf, topLeftSlopeFP, topRightSlopeFP);
            segment(iRight, leftIntersectBf, rightIntersectAf, bottomLeftSlopeFP, topRightSlopeFP);
            segment(iLeftSegment, leftIntersectAf, rightIntersectBf, topLeftSlopeFP, bottomRightSlopeFP);
            segment(iBottom, leftIntersectBf, rightIntersectBf, bottomLeftSlopeFP, bottomRightSlopeFP);
        }
    }
}

/* ==========================================================================
 * Span handling
 * ========================================================================== */

namespace {

constexpr int kBufferSize = 2048; // BufferSize in qdrawhelper_p.h

void addRuns(int x, int y, int length, std::vector<Run> &runs)
{
    while (length) {
        const int l = std::min(kBufferSize, length);
        runs.push_back({x, y, l});
        x += l;
        length -= l;
    }
}

/** The work of one QT_THREAD_PARALLEL_FILLS segment in handleSpans() */
void handleSegment(const Span *spans, int cStart, int cEnd, bool merge, std::vector<Run> &runs)
{
    for (int c = cStart; c < cEnd;) {
        if (!spans[c].len) {
            ++c;
            continue;
        }
        const int x = spans[c].x;
        const int y = spans[c].y;
        int right = x + spans[c].len;
        int next = c + 1;
        if (merge) {
            // compute length of adjacent spans
            for (; next < cEnd && spans[next].y == y && spans[next].x == right; ++next) {
                right += spans[next].len;
            }
        }
        addRuns(x, y, right - x, runs);
        c = next;
    }
}

} // namespace

void spanRuns(const std::vector<Span> &spans, int batchSize, bool merge, std::vector<Run> &runs)
{
    for (size_t start = 0; start < spans.size(); start += size_t(batchSize)) {
        const int count = int(std::min(spans.size() - start, size_t(batchSize)));
        const Span *batch = spans.data() + start;
        const int segments = (count + 32) / 64;
        if (segments > 1) {
            int c = 0;
            for (int i = 0; i < segments; ++i) {
                const int cn = (count - c) / (segments - i);
                handleSegment(batch, c, c + cn, merge, runs);
                c += cn;
            }
        } else {
            handleSegment(batch, 0, count, merge, runs);
        }
    }
}

void fillRectSpans(int x, int y, int w, int h, int deviceWidth, int deviceHeight, std::vector<Span> &spans)
{
    const int x1 = std::max(x, 0);
    const int x2 = std::min(x + w, deviceWidth);
    const int y1 = std::max(y, 0);
    const int y2 = std::min(y + h, deviceHeight);
    if (x2 <= x1 || y2 <= y1) {
        return;
    }
    for (int row = y1; row < y2; ++row) {
        spans.push_back({x1, x2 - x1, row, 255});
    }
}

} // namespace qt
} // namespace gpufx
