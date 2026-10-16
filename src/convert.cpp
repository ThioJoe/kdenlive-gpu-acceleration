/*
    SPDX-FileCopyrightText: 2026 ThioJoe
    SPDX-License-Identifier: GPL-3.0-only
*/

#include "convert.h"
#include "sws_port.h"

#include <vector>

namespace gpufx {

CUresult convertYuv420ToRgba(GpuScope &scope, const Yuv420Image &image, DeviceBuffer &out)
{
    const int w = image.width;
    const int h = image.height;
    if (w <= 0 || h <= 0 || (w & 1) || (h & 1) || (image.fields && (h & 3))) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const size_t lumaBytes = size_t(w) * size_t(h);
    const size_t chromaBytes = size_t(w / 2) * size_t(h / 2);
    const size_t yuvBytes = lumaBytes + 2 * chromaBytes;
    const size_t alphaBytes = image.alpha ? lumaBytes : 0;

    std::vector<int> columns;
    std::vector<int> rows;
    sws::chroma420Samples(w, h, image.fields, columns, rows);
    const size_t tableBytes = (columns.size() + rows.size()) * sizeof(int);
    std::vector<int> tables(columns);
    tables.insert(tables.end(), rows.begin(), rows.end());
    const sws::YuvToRgb k = sws::yuvToRgb(sws::colorspaceTable(image.colorspace), image.fullRange);

    // One upload buffer: tables, YUV planes and the alpha plane
    DeviceBuffer in;
    const size_t tablesOffset = 0;
    const size_t yuvOffset = (tableBytes + 255) & ~size_t(255);
    const size_t alphaOffset = yuvOffset + ((yuvBytes + 255) & ~size_t(255));
    CUresult res = in.allocate(alphaOffset + alphaBytes);
    if (res == CUDA_SUCCESS) {
        res = scope.upload(in.ptr() + tablesOffset, tables.data(), tableBytes);
    }
    if (res == CUDA_SUCCESS) {
        res = scope.upload(in.ptr() + yuvOffset, image.data, yuvBytes);
    }
    if (res == CUDA_SUCCESS && image.alpha) {
        res = scope.upload(in.ptr() + alphaOffset, image.alpha, alphaBytes);
    }
    if (res == CUDA_SUCCESS) {
        res = out.allocate(lumaBytes * 4);
    }
    if (res == CUDA_SUCCESS) {
        const CUdeviceptr columnTable = in.ptr() + tablesOffset;
        const CUdeviceptr rowTable = columnTable + columns.size() * sizeof(int);
        const CUdeviceptr luma = in.ptr() + yuvOffset;
        const CUdeviceptr cb = luma + lumaBytes;
        const CUdeviceptr cr = cb + chromaBytes;
        const CUdeviceptr alpha = image.alpha ? in.ptr() + alphaOffset : 0;
        res = scope.launch2D(KernelId::gpufx_yuv420_to_rgba, w, h, 32, 8, luma, cb, cr, alpha, w, w / 2, columnTable, rowTable, out.ptr(), w, h, k.yOffset,
                             k.yCoeff, k.v2r, k.v2g, k.u2g, k.u2b);
    }
    return res;
}

CUresult convertRgbaToYuv422(GpuScope &scope, CUdeviceptr rgba, int width, int height, int colorspace, bool fullRange, uint8_t *yuv422, uint8_t *alpha)
{
    if (width <= 0 || height <= 0 || (width & 1)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const size_t pixels = size_t(width) * size_t(height);
    const size_t yuvBytes = pixels * 2;
    const size_t alphaOffset = (yuvBytes + 255) & ~size_t(255);
    const sws::RgbToYuv k = sws::rgbToYuv(sws::colorspaceTable(colorspace));
    DeviceBuffer out;
    CUresult res = out.allocate(alphaOffset + (alpha ? pixels : 0));
    if (res == CUDA_SUCCESS) {
        const CUdeviceptr alphaPlane = alpha ? out.ptr() + alphaOffset : 0;
        res = scope.launch2D(KernelId::gpufx_rgba_to_yuyv422, width / 2, height, 32, 8, rgba, out.ptr(), alphaPlane, width, height, k.ry, k.gy, k.by, k.ru,
                             k.gu, k.bu, k.rv, k.gv, k.bv, fullRange ? 1 : 0);
    }
    if (res == CUDA_SUCCESS) {
        res = scope.download(yuv422, out.ptr(), yuvBytes);
    }
    if (res == CUDA_SUCCESS && alpha) {
        res = scope.download(alpha, out.ptr() + alphaOffset, pixels);
    }
    if (res == CUDA_SUCCESS) {
        res = scope.synchronize();
    }
    return res;
}

} // namespace gpufx
