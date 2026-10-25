/*
    SPDX-FileCopyrightText: 2026 ThioJoe
    SPDX-License-Identifier: GPL-3.0-only
*/

#include "convert.h"
#include "sws_port.h"

#include <algorithm>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
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

bool convertsRgbaToYuv420(int width, int height)
{
    // libswscale writes past the chroma planes of narrower images
    sws::ScaleFilter filter;
    return width >= 8 && height >= 2 && !(width & 1) && !(height & 1) && sws::scaleFilter(sws::ScaleBicubic, true, height, height / 2, filter);
}

CUresult convertRgbaToYuv420(GpuScope &scope, CUdeviceptr rgba, int width, int height, int colorspace, bool tenBit, uint8_t *yuv, uint8_t *alpha)
{
    // The vertical filter of the chroma, which libswscale subsamples with its
    // scaler
    sws::ScaleFilter filter;
    if (width < 8 || height < 2 || (width & 1) || (height & 1) || !sws::scaleFilter(sws::ScaleBicubic, true, height, height / 2, filter)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const size_t pixels = size_t(width) * size_t(height);
    const size_t yuvBytes = pixels * 3 / 2 * (tenBit ? 2 : 1);
    const size_t coefficientBytes = filter.coefficients.size() * sizeof(int16_t);
    const size_t positionsOffset = (coefficientBytes + 255) & ~size_t(255);
    const size_t yuvOffset = (positionsOffset + filter.positions.size() * sizeof(int32_t) + 255) & ~size_t(255);
    const size_t alphaOffset = (yuvOffset + yuvBytes + 255) & ~size_t(255);
    const sws::RgbToYuv k = sws::rgbToYuv(sws::colorspaceTable(colorspace));
    DeviceBuffer work;
    CUresult res = work.allocate(alphaOffset + (alpha ? pixels : 0));
    if (res == CUDA_SUCCESS) {
        res = scope.upload(work.ptr(), filter.coefficients.data(), coefficientBytes);
    }
    if (res == CUDA_SUCCESS) {
        res = scope.upload(work.ptr() + positionsOffset, filter.positions.data(), filter.positions.size() * sizeof(int32_t));
    }
    if (res == CUDA_SUCCESS) {
        const CUdeviceptr alphaPlane = alpha ? work.ptr() + alphaOffset : 0;
        res = scope.launch2D(KernelId::gpufx_rgba_to_yuv420, width / 2, height / 2, 32, 8, rgba, work.ptr() + yuvOffset, alphaPlane, width, height,
                             tenBit ? 1 : 0, work.ptr(), work.ptr() + positionsOffset, filter.size, k.ry, k.gy, k.by, k.ru, k.gu, k.bu, k.rv, k.gv, k.bv);
    }
    if (res == CUDA_SUCCESS) {
        res = scope.download(yuv, work.ptr() + yuvOffset, yuvBytes);
    }
    if (res == CUDA_SUCCESS && alpha) {
        res = scope.download(alpha, work.ptr() + alphaOffset, pixels);
    }
    if (res == CUDA_SUCCESS) {
        res = scope.synchronize();
    }
    return res;
}

CUresult convertRgbaThroughYuv422(GpuScope &scope, CUdeviceptr rgba, int width, int height, int colorspace, int backColorspace, bool backFullRange,
                                  DeviceBuffer &out)
{
    if (width <= 0 || height <= 0 || (width & 1)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const size_t pixels = size_t(width) * size_t(height);
    std::vector<int> columns;
    sws::chroma422Columns(width, columns);
    const size_t tableBytes = columns.size() * sizeof(int);
    const sws::RgbToYuv to = sws::rgbToYuv(sws::colorspaceTable(colorspace));
    const sws::YuvToRgb back = sws::yuvToRgb(sws::colorspaceTable(backColorspace), backFullRange);

    // One buffer: the column table and the YUYV image
    DeviceBuffer work;
    const size_t yuvOffset = (tableBytes + 255) & ~size_t(255);
    CUresult res = work.allocate(yuvOffset + pixels * 2);
    if (res == CUDA_SUCCESS) {
        res = scope.upload(work.ptr(), columns.data(), tableBytes);
    }
    if (res == CUDA_SUCCESS) {
        const CUdeviceptr noAlpha = 0;
        res = scope.launch2D(KernelId::gpufx_rgba_to_yuyv422, width / 2, height, 32, 8, rgba, work.ptr() + yuvOffset, noAlpha, width, height, to.ry, to.gy,
                             to.by, to.ru, to.gu, to.bu, to.rv, to.gv, to.bv, 0);
    }
    if (res == CUDA_SUCCESS) {
        res = out.allocate(pixels * 4);
    }
    if (res == CUDA_SUCCESS) {
        res = scope.launch2D(KernelId::gpufx_yuyv422_to_rgba, width, height, 32, 8, work.ptr() + yuvOffset, work.ptr(), rgba, out.ptr(), width, height,
                             back.yOffset, back.yCoeff, back.v2r, back.v2g, back.u2g, back.u2b);
    }
    return res;
}

namespace {

/** The filters of a scaling, packed for a single upload */
struct ScaleTables
{
    std::vector<uint8_t> data;
    int sizes[4] = {};
    size_t coefficients[4] = {};
    size_t positions[4] = {};
};

/** The filters of a scaling by scaleRgba(): luma and chroma horizontally,
 *  then vertically. Null for the scalings that are not ported. The longer
 *  filters take a while to compute (Lanczos), so the last few are kept. */
std::shared_ptr<const ScaleTables> scaleTables(int width, int height, int dstWidth, int dstHeight, sws::ScaleMethod lumaMethod,
                                               sws::ScaleMethod chromaMethod)
{
    struct Entry
    {
        int key[6];
        std::shared_ptr<const ScaleTables> tables;
    };
    static std::mutex mutex;
    static std::deque<Entry> cache; // the most recent first
    const int key[6] = {width, height, dstWidth, dstHeight, int(lumaMethod), int(chromaMethod)};
    {
        std::lock_guard<std::mutex> lock(mutex);
        for (auto it = cache.begin(); it != cache.end(); ++it) {
            if (std::equal(key, key + 6, it->key)) {
                const Entry entry = *it;
                cache.erase(it);
                cache.push_front(entry);
                return entry.tables;
            }
        }
    }

    sws::ScaleFilter filters[4];
    std::shared_ptr<ScaleTables> tables;
    // A single luma tap with two chroma taps, which yuv2rgb_full_1_c_template()
    // rounds differently from FFmpeg 8.0 on, is left out
    if (sws::scaleFilter(lumaMethod, false, width, dstWidth, filters[0]) && sws::scaleFilter(chromaMethod, false, width, dstWidth, filters[1])
        && sws::scaleFilter(lumaMethod, true, height, dstHeight, filters[2]) && sws::scaleFilter(chromaMethod, true, height, dstHeight, filters[3])
        && (filters[2].size != 1 || filters[3].size == 1)) {
        tables = std::make_shared<ScaleTables>();
        std::vector<uint8_t> &data = tables->data;
        auto append = [&data](const void *bytes, size_t count) {
            const size_t offset = data.size();
            data.resize((offset + count + 255) & ~size_t(255));
            memcpy(data.data() + offset, bytes, count);
            return offset;
        };
        for (int i = 0; i < 4; ++i) {
            tables->sizes[i] = filters[i].size;
            tables->coefficients[i] = append(filters[i].coefficients.data(), filters[i].coefficients.size() * sizeof(int16_t));
            tables->positions[i] = append(filters[i].positions.data(), filters[i].positions.size() * sizeof(int32_t));
        }
    }
    std::lock_guard<std::mutex> lock(mutex);
    cache.push_front(Entry{{key[0], key[1], key[2], key[3], key[4], key[5]}, tables});
    if (cache.size() > 8) {
        cache.pop_back();
    }
    return tables;
}

} // namespace

CUresult scaleRgba(GpuScope &scope, CUdeviceptr rgba, int width, int height, int dstWidth, int dstHeight, sws::ScaleMethod lumaMethod,
                   sws::ScaleMethod chromaMethod, int srcColorspace, int dstColorspace, DeviceBuffer &out)
{
    if (width <= 0 || height <= 0 || dstWidth <= 0 || dstHeight <= 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const std::shared_ptr<const ScaleTables> tables = scaleTables(width, height, dstWidth, dstHeight, lumaMethod, chromaMethod);
    if (!tables) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    const sws::RgbToYuv to = sws::rgbToYuv(sws::colorspaceTable(dstColorspace));
    const sws::YuvToRgb back = sws::yuvToRgb(sws::colorspaceTable(srcColorspace), false);

    // One buffer: the filters, then the planes between the two passes
    const size_t planesOffset = tables->data.size();
    const size_t planesBytes = size_t(dstWidth) * size_t(height) * 4 * sizeof(int16_t);
    DeviceBuffer work;
    CUresult res = work.allocate(planesOffset + planesBytes);
    if (res == CUDA_SUCCESS) {
        res = scope.upload(work.ptr(), tables->data.data(), tables->data.size());
    }
    const CUdeviceptr planes = work.ptr() + planesOffset;
    const CUdeviceptr base = work.ptr();
    if (res == CUDA_SUCCESS) {
        res = scope.launch2D(KernelId::gpufx_sws_scale_rgba_h, dstWidth, height, 32, 8, rgba, width, height, planes, dstWidth, base + tables->coefficients[0],
                             base + tables->positions[0], tables->sizes[0], base + tables->coefficients[1], base + tables->positions[1], tables->sizes[1],
                             to.ry, to.gy, to.by, to.ru, to.gu, to.bu, to.rv, to.gv, to.bv);
    }
    if (res == CUDA_SUCCESS) {
        res = out.allocate(size_t(dstWidth) * size_t(dstHeight) * 4);
    }
    if (res == CUDA_SUCCESS) {
        res = scope.launch2D(KernelId::gpufx_sws_scale_rgba_v, dstWidth, dstHeight, 32, 8, planes, dstWidth, height, out.ptr(), dstHeight,
                             base + tables->coefficients[2], base + tables->positions[2], tables->sizes[2], base + tables->coefficients[3],
                             base + tables->positions[3], tables->sizes[3], back.yOffset, back.yCoeff, back.v2r, back.v2g, back.u2g, back.u2b);
    }
    return res;
}

} // namespace gpufx
