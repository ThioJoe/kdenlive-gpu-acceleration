/*
    SPDX-FileCopyrightText: 2026 ThioJoe
    SPDX-License-Identifier: GPL-3.0-only
*/

/*
 * Checks the gpufx module against the CPU filters it accelerates.
 *
 * Every test renders the same input twice, once with the module turned off
 * for the filters (the original CPU code) and once through the GPU path, and
 * compares the pixels. By default the GPU is the emulated driver built with
 * the tests (mockcuda), which also lets the tests count transfers between
 * host and device and inject failures. Pointing GPUFX_CUDA_LIBRARY at the real
 * driver (libcuda.so.1 / nvcuda.dll) runs the same comparisons on an actual
 * NVIDIA GPU; the checks that need the emulator are then skipped.
 *
 * Usage: gpufx_test [--no-gpu | --settings-off]
 *   --no-gpu: expect CUDA to be unavailable (run with MOCKCUDA_NO_DEVICE=1)
 *   --settings-off: start with GPU processing turned off in the settings file,
 *   and only check that the module then leaves MLT alone
 */

#include "mockcuda/mockcuda.h"
#include "settings_page_api.h"

#include <framework/mlt.h>

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <process.h>
#include <windows.h>
#define getpid _getpid
#else
#include <dlfcn.h>
#include <unistd.h>
#endif

namespace {

int g_failures = 0;
int g_checks = 0;
/** The module, as loaded by MLT or by the tests */
void *g_module = nullptr;
/** The settings file the module reads (GPUFX_CONFIG_FILE) */
std::string g_settingsFile;

void check(bool condition, const char *format, ...)
{
    ++g_checks;
    if (condition) {
        return;
    }
    ++g_failures;
    va_list args;
    va_start(args, format);
    fprintf(stderr, "FAIL: ");
    vfprintf(stderr, format, args);
    fprintf(stderr, "\n");
    va_end(args);
}

void *openLibrary(const char *path)
{
#if defined(_WIN32)
    return reinterpret_cast<void *>(LoadLibraryA(path));
#else
    return dlopen(path, RTLD_NOW | RTLD_GLOBAL);
#endif
}

void *findSymbol(void *library, const char *name)
{
#if defined(_WIN32)
    return reinterpret_cast<void *>(GetProcAddress(reinterpret_cast<HMODULE>(library), name));
#else
    return dlsym(library, name);
#endif
}

void setEnvironment(const char *name, const char *value)
{
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

void writeText(const std::string &path, const char *text)
{
    if (FILE *file = fopen(path.c_str(), "wb")) {
        fputs(text, file);
        fclose(file);
    }
}

/** The handle of the module MLT loaded from its module folder */
void *findLoadedModule(mlt_repository repository)
{
    auto properties = reinterpret_cast<mlt_properties>(repository);
    for (int i = 0; i < mlt_properties_count(properties); ++i) {
        const char *name = mlt_properties_get_name(properties, i);
        if (name && (strstr(name, "libmltgpufx.so") || strstr(name, "libmltgpufx.dll"))) {
            return mlt_properties_get_data_at(properties, i, nullptr);
        }
    }
    return nullptr;
}

struct Mock
{
    MockCudaGetStats getStats = nullptr;
    MockCudaResetStats resetStats = nullptr;
    MockCudaFailAfter failAfter = nullptr;
    bool active() const { return getStats != nullptr; }
    MockCudaStats stats() const
    {
        MockCudaStats s{};
        if (getStats) {
            getStats(&s);
        }
        return s;
    }
    void reset() const
    {
        if (resetStats) {
            resetStats();
        }
    }
} g_mock;

struct Image
{
    int width = 0;
    int height = 0;
    std::vector<uint8_t> pixels;
};

Image makeImage(int width, int height, int pattern)
{
    Image image;
    image.width = width;
    image.height = height;
    image.pixels.resize(size_t(width) * size_t(height) * 4);
    uint32_t seed = 12345u + uint32_t(pattern) * 7919u;
    auto random = [&seed]() {
        seed = seed * 1664525u + 1013904223u;
        return seed >> 8;
    };
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            uint8_t *p = &image.pixels[(size_t(y) * size_t(width) + size_t(x)) * 4];
            p[0] = uint8_t(x * 7 + y * 3);
            p[1] = uint8_t(x * 2 + y * 5);
            p[2] = uint8_t(x ^ y);
            int alpha = 255;
            switch (pattern) {
            case 0: { // opaque rectangle with a semi transparent border on a transparent background
                const bool inside = x > width / 4 && x < 3 * width / 4 && y > height / 4 && y < 3 * height / 4;
                alpha = inside ? 255 : 0;
                if (x == width / 4 + 1 || y == height / 4 + 1) {
                    alpha = 128;
                }
                break;
            }
            case 1: // random alpha, including many 0 and 255
                alpha = int(random() & 255);
                if ((random() & 3) == 0) {
                    alpha = 0;
                } else if ((random() & 3) == 0) {
                    alpha = 255;
                }
                p[0] = uint8_t(random());
                p[1] = uint8_t(random());
                p[2] = uint8_t(random());
                break;
            case 2: { // soft disc
                const double dx = x - width * 0.4;
                const double dy = y - height * 0.6;
                const double d = std::sqrt(dx * dx + dy * dy);
                alpha = std::clamp(int(255 - (d - height * 0.2) * 10), 0, 255);
                break;
            }
            case 3: // stripes touching the borders
                alpha = (x < 3 || y < 2 || x > width - 4) ? 200 : (((x / 7 + y / 5) % 3 == 0) ? 255 : 0);
                break;
            default: // opaque
                alpha = 255;
                break;
            }
            p[3] = uint8_t(alpha);
        }
    }
    return image;
}

struct FilterSpec
{
    const char *service;
    std::vector<std::pair<const char *, std::string>> properties;
};

bool hasFilter(const char *service)
{
    return mlt_properties_get_data(mlt_repository_filters(mlt_factory_repository()), service, nullptr) != nullptr;
}

bool hasFilters(const std::vector<FilterSpec> &specs)
{
    for (const FilterSpec &spec : specs) {
        if (!hasFilter(spec.service)) {
            printf("SKIP: MLT has no %s filter\n", spec.service);
            return false;
        }
    }
    return true;
}

std::vector<mlt_filter> createFilters(mlt_profile profile, const std::vector<FilterSpec> &specs, bool cpu)
{
    std::vector<mlt_filter> filters;
    for (const FilterSpec &spec : specs) {
        mlt_filter filter = mlt_factory_filter(profile, spec.service, nullptr);
        if (!filter) {
            fprintf(stderr, "cannot create filter %s\n", spec.service);
            exit(2);
        }
        for (const auto &property : spec.properties) {
            mlt_properties_set(MLT_FILTER_PROPERTIES(filter), property.first, property.second.c_str());
        }
        mlt_properties_set_int(MLT_FILTER_PROPERTIES(filter), "_gpufx.disable", cpu ? 1 : 0);
        filters.push_back(filter);
    }
    return filters;
}

typedef std::vector<std::pair<const char *, std::string>> Properties;

/** Runs the filters (in attach order) over a frame holding the image.
 *  frameProperties are set on the frame, and the output is requested at
 *  requestWidth x requestHeight (default: the input size). */
Image render(mlt_profile profile, const std::vector<FilterSpec> &specs, const Image &input, bool cpu, mlt_position position = 0,
             const Properties &frameProperties = {}, int requestWidth = 0, int requestHeight = 0)
{
    std::vector<mlt_filter> filters = createFilters(profile, specs, cpu);
    mlt_frame frame = mlt_frame_init(nullptr);
    mlt_properties properties = MLT_FRAME_PROPERTIES(frame);
    const int size = input.width * input.height * 4;
    auto *buffer = static_cast<uint8_t *>(mlt_pool_alloc(size));
    memcpy(buffer, input.pixels.data(), size_t(size));
    mlt_frame_set_image(frame, buffer, size, mlt_pool_release);
    mlt_properties_set_int(properties, "format", mlt_image_rgba);
    mlt_properties_set_int(properties, "width", input.width);
    mlt_properties_set_int(properties, "height", input.height);
    mlt_properties_set_int(properties, "meta.media.width", input.width);
    mlt_properties_set_int(properties, "meta.media.height", input.height);
    for (const auto &property : frameProperties) {
        mlt_properties_set(properties, property.first, property.second.c_str());
    }
    mlt_frame_set_position(frame, position);
    for (mlt_filter filter : filters) {
        mlt_filter_process(filter, frame);
    }
    uint8_t *image = nullptr;
    mlt_image_format format = mlt_image_rgba;
    int width = requestWidth > 0 ? requestWidth : input.width;
    int height = requestHeight > 0 ? requestHeight : input.height;
    const int error = mlt_frame_get_image(frame, &image, &format, &width, &height, 0);
    Image output;
    if (!error && image && format == mlt_image_rgba) {
        output.width = width;
        output.height = height;
        output.pixels.assign(image, image + size_t(width) * size_t(height) * 4);
    } else {
        fprintf(stderr, "get_image failed: error %d format %d\n", error, int(format));
    }
    mlt_frame_close(frame);
    for (mlt_filter filter : filters) {
        mlt_filter_close(filter);
    }
    return output;
}

struct Difference
{
    int maxChannel = 0; // largest difference on any channel
    int maxAlpha = 0;
    double maxPremultiplied = 0; // largest difference of premultiplied color
    long pixels = 0;             // pixels with any difference
    bool sizeMismatch = false;
};

Difference compare(const Image &a, const Image &b)
{
    Difference d;
    if (a.width != b.width || a.height != b.height || a.pixels.size() != b.pixels.size() || a.pixels.empty()) {
        d.sizeMismatch = true;
        return d;
    }
    for (size_t i = 0; i < a.pixels.size(); i += 4) {
        bool differs = false;
        for (int c = 0; c < 4; ++c) {
            const int diff = std::abs(int(a.pixels[i + c]) - int(b.pixels[i + c]));
            if (diff) {
                differs = true;
            }
            if (c == 3) {
                d.maxAlpha = std::max(d.maxAlpha, diff);
            } else {
                d.maxChannel = std::max(d.maxChannel, diff);
                const double pa = a.pixels[i + c] * a.pixels[i + 3] / 255.0;
                const double pb = b.pixels[i + c] * b.pixels[i + 3] / 255.0;
                d.maxPremultiplied = std::max(d.maxPremultiplied, std::fabs(pa - pb));
            }
        }
        if (differs) {
            ++d.pixels;
        }
    }
    return d;
}

std::string describe(const std::vector<FilterSpec> &specs)
{
    std::string text;
    for (const FilterSpec &spec : specs) {
        if (!text.empty()) {
            text += " -> ";
        }
        text += spec.service;
        for (const auto &property : spec.properties) {
            text += std::string(" ") + property.first + "=" + property.second;
        }
    }
    return text;
}

/** Renders on CPU and GPU and checks the results. exact: identical pixels,
 *  otherwise colors may differ by 1 (Qt un-premultiplies with an approximate
 *  reciprocal) and alpha must be identical. */
void expectSame(mlt_profile profile, const std::vector<FilterSpec> &specs, const Image &input, bool exact, mlt_position position = 0,
                const Properties &frameProperties = {}, int requestWidth = 0, int requestHeight = 0)
{
    if (!hasFilters(specs)) {
        return;
    }
    const Image cpu = render(profile, specs, input, true, position, frameProperties, requestWidth, requestHeight);
    const Image gpu = render(profile, specs, input, false, position, frameProperties, requestWidth, requestHeight);
    const Difference d = compare(cpu, gpu);
    std::string what = describe(specs) + " on " + std::to_string(input.width) + "x" + std::to_string(input.height);
    for (const auto &property : frameProperties) {
        what += std::string(", ") + property.first + "=" + property.second;
    }
    if (d.sizeMismatch) {
        check(false, "%s: output size mismatch", what.c_str());
        return;
    }
    if (exact) {
        check(d.pixels == 0, "%s: %ld pixels differ (max channel %d, alpha %d)", what.c_str(), d.pixels, d.maxChannel, d.maxAlpha);
    } else {
        check(d.maxAlpha == 0 && d.maxPremultiplied <= 1.0, "%s: alpha diff %d, premultiplied diff %.2f, channel diff %d (%ld pixels)", what.c_str(),
              d.maxAlpha, d.maxPremultiplied, d.maxChannel, d.pixels);
    }
}

void testStatus(mlt_profile profile, bool expectGpu)
{
    mlt_filter status = mlt_factory_filter(profile, "gpufx", nullptr);
    check(status != nullptr, "the gpufx status service is missing");
    if (!status) {
        return;
    }
    mlt_properties properties = MLT_FILTER_PROPERTIES(status);
    const int available = mlt_properties_get_int(properties, "available");
    printf("gpufx status: available=%d device='%s' status='%s' effects='%s'\n", available, mlt_properties_get(properties, "device"),
           mlt_properties_get(properties, "status"), mlt_properties_get(properties, "effects"));
    check(available == (expectGpu ? 1 : 0), "status: available is %d", available);
    const char *effects = mlt_properties_get(properties, "effects");
    check(effects && strstr(effects, "dropshadow"), "status: dropshadow is not listed");
    mlt_filter_close(status);
}

void testDropShadow(mlt_profile profile)
{
    struct Case
    {
        int width, height, pattern;
        const char *radius, *x, *y, *color;
    } cases[] = {
        {64, 48, 0, "1", "8", "8", "#b4636363"},        {64, 48, 0, "0", "0", "0", "#b4636363"},         {64, 48, 1, "0", "0", "0", "#b4636363"},
        {64, 48, 0, "3.9", "2", "-3", "#ff000000"},     {64, 48, 0, "4", "8", "8", "#80ff0000"},         {65, 47, 2, "10", "-5.5", "2.5", "#b4636363"},
        {65, 47, 1, "10", "8", "8", "#b4636363"},       {200, 120, 2, "25", "8.4", "-7.6", "#ff123456"}, {200, 120, 3, "7.5", "0", "0", "#c0000000"},
        {31, 17, 3, "100", "100", "-100", "#b4636363"}, {64, 48, 1, "0.5", "0.5", "-0.5", "#b4636363"},  {64, 48, 1, "2", "2.5", "-2.5", "#01ffffff"},
        {64, 48, 4, "6", "5", "5", "#b4636363"},        {1, 1, 0, "5", "1", "1", "#ff00ff00"},           {2, 3, 1, "4", "-1", "1", "#ff00ff00"},
        {48, 64, 3, "0", "3", "0", "#40ffffff"},        {48, 64, 1, "33", "-60", "70", "#ffffffff"},
    };
    for (const Case &c : cases) {
        const Image input = makeImage(c.width, c.height, c.pattern);
        expectSame(profile, {{"dropshadow", {{"radius", c.radius}, {"x", c.x}, {"y", c.y}, {"color", c.color}}}}, input, false);
    }
    // Animated parameters are evaluated at the frame position
    const Image input = makeImage(80, 60, 2);
    for (mlt_position position : {0, 5, 10}) {
        expectSame(profile, {{"dropshadow", {{"radius", "0=1;10=20"}, {"x", "0=-10;10=10"}, {"y", "0=3;10=-3"}, {"out", "10"}}}}, input, false, position);
    }
}

void testBoxBlur(mlt_profile profile)
{
    const Image input = makeImage(160, 90, 1);
    for (const char *h : {"0", "10", "50", "3", "200"}) {
        for (const char *v : {"0", "20", "7"}) {
            for (const char *preserve : {"0", "1"}) {
                expectSame(profile, {{"box_blur", {{"hradius", h}, {"vradius", v}, {"preserve_alpha", preserve}}}}, input, true);
            }
        }
    }
    expectSame(profile, {{"box_blur", {{"hradius", "33"}, {"vradius", "21"}}}}, makeImage(97, 53, 2), true);
}

void testRectBlur(mlt_profile profile)
{
    const Image input = makeImage(160, 90, 1);
    for (const char *blur : {"0", "1", "5", "40"}) {
        for (const char *hori : {"1", "3"}) {
            expectSame(profile, {{"boxblur", {{"blur", blur}, {"hori", hori}, {"vert", "2"}}}}, input, true);
        }
    }
    // Old style start parameter
    expectSame(profile, {{"boxblur", {{"start", "4"}}}}, makeImage(77, 41, 3), true);
}

void testColor(mlt_profile profile)
{
    const Image input = makeImage(96, 54, 1);
    // Lightweight effects next to a heavy one run on the GPU
    for (const char *level : {"0.5", "1", "1.7", "0"}) {
        for (const char *alpha : {"1", "0.3", "-1"}) {
            expectSame(profile, {{"brightness", {{"level", level}, {"alpha", alpha}}}, {"box_blur", {{"hradius", "20"}, {"vradius", "20"}}}}, input, true);
        }
    }
    expectSame(profile, {{"box_blur", {{"hradius", "20"}, {"vradius", "0"}}}, {"brightness", {{"start", "0.2"}, {"end", "1.4"}, {"out", "20"}}}}, input, true,
               7);
    expectSame(profile,
               {{"lift_gamma_gain",
                 {{"lift_r", "0.1"},
                  {"lift_g", "0"},
                  {"lift_b", "0.2"},
                  {"gamma_r", "1.2"},
                  {"gamma_g", "0.8"},
                  {"gamma_b", "1"},
                  {"gain_r", "1.1"},
                  {"gain_g", "0.9"},
                  {"gain_b", "1.3"}}},
                {"boxblur", {{"blur", "3"}}}},
               input, true);
}

void testFrei0r(mlt_profile profile)
{
    const Image input = makeImage(160, 90, 1);
    const Image portrait = makeImage(53, 97, 2);
    // Square Blur: the kernel size is relative to the largest side
    for (const char *kernel : {"0", "0.004", "0.05", "0.3", "1", "3"}) {
        expectSame(profile, {{"frei0r.squareblur", {{"Kernel size", kernel}}}}, input, true);
    }
    expectSame(profile, {{"frei0r.squareblur", {{"0", "0.2"}}}}, portrait, true);
    for (const char *blur : {"0", "0.01", "0.2", "1", "10"}) {
        expectSame(profile, {{"frei0r.glow", {{"Blur", blur}}}}, input, true);
    }
    expectSame(profile, {{"frei0r.glow", {{"Blur", "2"}}}}, portrait, true);
    for (const char *blend : {"0", "0.5", "1"}) {
        for (const char *blur : {"0", "0.1", "0.5"}) {
            expectSame(profile, {{"frei0r.softglow", {{"blurblend", blend}, {"blur", blur}, {"brightness", "0.6"}, {"sharpness", "0.7"}}}}, input, true);
        }
    }
    // Plugin defaults
    expectSame(profile, {{"frei0r.softglow", {}}}, portrait, true);
    for (const char *amount : {"0", "0.3", "0.5", "1"}) {
        for (const char *size : {"0", "0.3", "0.5", "1"}) {
            expectSame(profile, {{"frei0r.sharpness", {{"Amount", amount}, {"Size", size}}}}, input, true);
        }
    }
    expectSame(profile, {{"frei0r.sharpness", {{"0", "0.9"}, {"1", "0.2"}}}}, makeImage(31, 7, 3), true);
    // Animated parameters, mixed with other effects
    expectSame(profile,
               {{"frei0r.glow", {{"Blur", "0=0.1;10=0.9"}, {"out", "10"}}},
                {"dropshadow", {{"radius", "4"}}},
                {"frei0r.sharpness", {{"Amount", "0.7"}, {"Size", "0=0;10=1"}, {"out", "10"}}},
                {"frei0r.softglow", {{"blur", "0.05"}}}},
               input, false, 5);
    // Scale and Tilt: scaled up (2-tap filters), down by up to 2 (4 taps) and
    // further (wider filters), moved partly out of the image, cropped
    const Image wide = makeImage(160, 90, 2);
    for (const Image *image : {&input, &wide, &portrait}) {
        for (const char *scale : {"0.5", "0.75", "0.37", "0.26", "0.1", "1.3"}) {
            for (const char *tilt : {"0.5", "0.45", "0.71"}) {
                expectSame(profile, {{"frei0r.scale0tilt", {{"4", scale}, {"5", scale}, {"6", tilt}, {"7", "0.5"}}}}, *image, true);
            }
        }
        expectSame(profile, {{"frei0r.scale0tilt", {{"Scale X", "0.3"}, {"Scale Y", "0.8"}, {"Tilt X", "0.37"}, {"Tilt Y", "0.62"}}}}, *image, true);
        expectSame(profile,
                   {{"frei0r.scale0tilt", {{"0", "0.1"}, {"1", "0.2"}, {"2", "0.15"}, {"3", "0.05"}, {"4", "0.45"}, {"5", "0.55"}, {"6", "0.3"}, {"7", "0.6"}}}},
                   *image, true);
        expectSame(profile, {{"frei0r.scale0tilt", {{"4", "0.2"}, {"5", "0.6"}, {"6", "0.1"}, {"7", "0.95"}}}}, *image, true);
    }
    // Plugin defaults (a copy), nothing left in the image
    expectSame(profile, {{"frei0r.scale0tilt", {}}}, input, true);
    expectSame(profile, {{"frei0r.scale0tilt", {{"6", "1.2"}}}}, input, true);
    expectSame(profile, {{"frei0r.scale0tilt", {{"4", "0.3"}, {"5", "0.004"}}}}, makeImage(31, 7, 3), true);
    if (g_mock.active() && hasFilter("frei0r.scale0tilt")) {
        // gavl draws results narrower than 4 pixels with values from memory
        // it never wrote: left to the CPU, whose output then varies
        g_mock.reset();
        render(profile, {{"frei0r.scale0tilt", {{"4", "0.005"}, {"5", "0.3"}}}}, input, false);
        check(g_mock.stats().launches == 0, "scale0tilt drew a result narrower than 4 pixels on the GPU");
    }
    // Kdenlive presets animate the scale and the tilt, next to other effects
    expectSame(profile,
               {{"frei0r.scale0tilt", {{"4", "0=0.4;10=0.5"}, {"5", "0=0.4;10=0.5"}, {"6", "0=0.55;10=0.5"}, {"7", "0=0.6;10=0.5"}, {"out", "10"}}},
                {"brightness", {{"alpha", "0=0;10=1"}, {"out", "10"}}},
                {"dropshadow", {{"radius", "4"}}}},
               wide, false, 3);
    if (g_mock.active() && hasFilter("frei0r.squareblur")) {
        // MLT slices the frame when "threads" is set: left to the CPU. The
        // slices share one plugin instance, so the CPU result is not even
        // repeatable for this plugin, nothing to compare.
        g_mock.reset();
        render(profile, {{"frei0r.squareblur", {{"Kernel size", "0.1"}, {"threads", "0"}}}}, input, false);
        check(mlt_slices_count_normal() == 1 || g_mock.stats().launches == 0, "a sliced frei0r filter ran on the GPU");
    }
}

/** Kdenlive's Transform: MLT's qtblend filter, drawn by QPainter on the CPU */
void testTransform(mlt_profile profile)
{
    if (!hasFilter("qtblend")) {
        printf("SKIP: MLT has no qtblend filter\n");
        return;
    }
    const Properties smooth = {{"consumer.rescale", "bilinear"}};
    const Properties nearest = {{"consumer.rescale", "nearest"}};
    struct Case
    {
        int width, height, pattern;
        const char *rect, *rotation;
        bool distort;
    } cases[] = {
        {160, 90, 1, "20 10 80 45 1", nullptr, false},         // picture in picture
        {160, 90, 2, "-30.5 12.25 200 120 1", nullptr, false}, // zoom from a fractional position
        {160, 90, 1, "10 5 160 90 0.6", nullptr, false},       // moved only, with opacity
        {120, 70, 3, "15 8 100 60 1", "25", false},            // rotation
        {160, 90, 0, "0 0 160 90 1", "-90", false},            // quarter turn
        {97, 61, 4, "5.5 3.5 120 70 0.8", "180", false},       // half turn
        {160, 90, 1, "30 20 50 70 1", nullptr, true},          // distort
        {160, 90, 2, "0% 10% 50% 50% 1", nullptr, false},      // percentages
        {200, 100, 1, "-400 -200 960 540 1", "3", false},      // large zoom, slightly rotated
        {40, 30, 3, "60.3 40.7 33.1 21.9 0.93", "-47", false}, // small
    };
    for (const Case &c : cases) {
        Properties properties = {{"rect", c.rect}};
        if (c.rotation) {
            properties.push_back({"rotation", c.rotation});
        }
        if (c.distort) {
            properties.push_back({"distort", "1"});
        }
        const Image input = makeImage(c.width, c.height, c.pattern);
        for (const Properties *frameProperties : {&smooth, &nearest}) {
            // The output has the profile size
            expectSame(profile, {{"qtblend", properties}}, input, false, 0, *frameProperties, profile->width, profile->height);
        }
    }
    const Image input = makeImage(160, 90, 1);
    // Animated, rotation around the center or an anchor point
    expectSame(profile, {{"qtblend", {{"rect", "0=0 0 160 90 1;10=40 20 80 45 0.5"}, {"rotation", "0=0;10=30"}, {"rotate_center", "1"}, {"out", "10"}}}}, input,
               false, 4, smooth, 160, 90);
    expectSame(profile, {{"qtblend", {{"rect", "10 10 100 60 1"}, {"rotation", "-15"}, {"rotate_anchor", "0.2 0.8"}}}}, input, false, 0, smooth, 160, 90);
    // Two transforms: the lower one follows the consumer scaling set by the upper one
    expectSame(profile, {{"qtblend", {{"rect", "10 10 120 70 1"}}}, {"qtblend", {{"rect", "-20 5 140 80 0.9"}, {"rotation", "12"}}}}, input, false, 0, smooth,
               160, 90);
    // Preview at half resolution: consumer scaling
    expectSame(profile, {{"qtblend", {{"rect", "30 15 100 60 1"}, {"rotation", "8"}}}}, makeImage(80, 45, 2), false, 0, smooth, 80, 45);
    // With a background color
    expectSame(profile, {{"qtblend", {{"rect", "30 15 100 60 0.7"}, {"background_color", "0x80402010"}}}}, input, false, 0, smooth, 160, 90);

    if (g_mock.active()) {
        // Zoom with a shadow: one upload, one download (MLT 7.38 downscales on the CPU, hence the zoom)
        g_mock.reset();
        const std::vector<FilterSpec> chain = {{"qtblend", {{"rect", "-20 -10 200 112.5 1"}, {"rotation", "5"}}}, {"dropshadow", {{"radius", "4"}}}};
        expectSame(profile, chain, input, false, 0, smooth, 160, 90);
        g_mock.reset();
        render(profile, chain, input, false, 0, smooth, 160, 90);
        check(g_mock.stats().uploads == 2 && g_mock.stats().downloads == 1,
              "transform and shadow: %llu uploads (the frame and the transform tables) and %llu downloads", (unsigned long long)g_mock.stats().uploads,
              (unsigned long long)g_mock.stats().downloads);

        // Left to the CPU: other composition modes, rotation with distortion
        for (const Properties &properties :
             std::vector<Properties>{{{"rect", "20 10 100 60 1"}, {"compositing", "13"}}, {{"rect", "20 10 100 30 1"}, {"distort", "1"}, {"rotation", "20"}}}) {
            g_mock.reset();
            expectSame(profile, {{"qtblend", properties}}, input, false, 0, smooth, 160, 90);
            check(g_mock.stats().launches == 0, "%s ran on the GPU", describe({{"qtblend", properties}}).c_str());
        }
    }
}

/* A producer of a fixed RGBA image, standing for a clip */
int patternGetImage(mlt_frame frame, uint8_t **image, mlt_image_format *format, int *width, int *height, int)
{
    auto *pattern = static_cast<const Image *>(mlt_properties_get_data(MLT_FRAME_PROPERTIES(frame), "_pattern", nullptr));
    const int size = pattern->width * pattern->height * 4;
    auto *buffer = static_cast<uint8_t *>(mlt_pool_alloc(size));
    memcpy(buffer, pattern->pixels.data(), size_t(size));
    mlt_frame_set_image(frame, buffer, size, mlt_pool_release);
    *image = buffer;
    *format = mlt_image_rgba;
    *width = pattern->width;
    *height = pattern->height;
    return 0;
}

int patternGetFrame(mlt_producer producer, mlt_frame_ptr frame, int)
{
    *frame = mlt_frame_init(MLT_PRODUCER_SERVICE(producer));
    mlt_properties properties = MLT_FRAME_PROPERTIES(*frame);
    auto *pattern = static_cast<const Image *>(mlt_properties_get_data(MLT_PRODUCER_PROPERTIES(producer), "_pattern", nullptr));
    mlt_properties_set_data(properties, "_pattern", const_cast<Image *>(pattern), 0, nullptr, nullptr);
    mlt_properties_set_int(properties, "format", mlt_properties_get_int(MLT_PRODUCER_PROPERTIES(producer), "_format"));
    mlt_properties_set_int(properties, "width", pattern->width);
    mlt_properties_set_int(properties, "height", pattern->height);
    mlt_properties_set_int(properties, "meta.media.width", pattern->width);
    mlt_properties_set_int(properties, "meta.media.height", pattern->height);
    mlt_frame_set_aspect_ratio(*frame, 1.0);
    mlt_frame_set_position(*frame, mlt_producer_position(producer));
    mlt_frame_push_get_image(*frame, patternGetImage);
    mlt_producer_prepare_next(producer);
    return 0;
}

void patternClose(mlt_producer producer)
{
    // mlt_producer_close() only frees producers created by mlt_producer_new() through this
    producer->close = nullptr;
    mlt_producer_close(producer);
    free(producer);
}

/** A producer of the image. Its frames announce declaredFormat before their
 *  image is fetched, as video producers announce their decoded format. */
mlt_producer patternProducer(mlt_profile profile, const Image *pattern, mlt_image_format declaredFormat = mlt_image_rgba)
{
    mlt_producer producer = mlt_producer_new(profile);
    producer->get_frame = patternGetFrame;
    producer->close = reinterpret_cast<mlt_destructor>(patternClose);
    producer->close_object = producer;
    mlt_properties_set_data(MLT_PRODUCER_PROPERTIES(producer), "_pattern", const_cast<Image *>(pattern), 0, nullptr, nullptr);
    mlt_properties_set_int(MLT_PRODUCER_PROPERTIES(producer), "_format", declaredFormat);
    mlt_properties_set_position(MLT_PRODUCER_PROPERTIES(producer), "length", 100);
    mlt_properties_set_position(MLT_PRODUCER_PROPERTIES(producer), "out", 99);
    return producer;
}

struct Track
{
    const Image *image;
    std::vector<FilterSpec> filters;
    Properties transition; // qtblend transition properties onto track 0
};

/** Renders a timeline like Kdenlive's: one track per entry, the first one at
 *  the bottom, each other track composited onto it by a qtblend transition.
 *  With nested, each track is a tractor of two playlists, the clip on the
 *  first one, as in Kdenlive's projects. */
Image renderTimeline(mlt_profile profile, const std::vector<Track> &tracks, bool cpu, const Properties &frameProperties, mlt_position position = 0,
                     bool nested = false)
{
    mlt_tractor tractor = mlt_tractor_new();
    mlt_multitrack multitrack = mlt_tractor_multitrack(tractor);
    mlt_field field = mlt_tractor_field(tractor);
    std::vector<mlt_producer> producers;
    std::vector<mlt_filter> filters;
    std::vector<mlt_transition> transitions;
    std::vector<mlt_playlist> playlists;
    std::vector<mlt_tractor> trackTractors;
    for (size_t i = 0; i < tracks.size(); ++i) {
        mlt_producer producer = patternProducer(profile, tracks[i].image);
        for (mlt_filter filter : createFilters(profile, tracks[i].filters, cpu)) {
            mlt_producer_attach(producer, filter);
            filters.push_back(filter);
        }
        mlt_producer track = producer;
        if (nested) {
            mlt_playlist clips = mlt_playlist_new(profile);
            mlt_playlist empty = mlt_playlist_new(profile);
            mlt_playlist_append(clips, producer);
            mlt_tractor trackTractor = mlt_tractor_new();
            mlt_multitrack_connect(mlt_tractor_multitrack(trackTractor), MLT_PLAYLIST_PRODUCER(clips), 0);
            mlt_multitrack_connect(mlt_tractor_multitrack(trackTractor), MLT_PLAYLIST_PRODUCER(empty), 1);
            playlists.push_back(clips);
            playlists.push_back(empty);
            trackTractors.push_back(trackTractor);
            track = MLT_TRACTOR_PRODUCER(trackTractor);
        }
        mlt_multitrack_connect(multitrack, track, int(i));
        producers.push_back(producer);
        if (i > 0) {
            mlt_transition transition = mlt_factory_transition(profile, "qtblend", nullptr);
            mlt_properties properties = MLT_TRANSITION_PROPERTIES(transition);
            mlt_properties_set_int(properties, "always_active", 1);
            for (const auto &property : tracks[i].transition) {
                mlt_properties_set(properties, property.first, property.second.c_str());
            }
            mlt_properties_set_int(properties, "_gpufx.disable", cpu ? 1 : 0);
            mlt_field_plant_transition(field, transition, 0, int(i));
            transitions.push_back(transition);
        }
    }
    mlt_producer_seek(MLT_TRACTOR_PRODUCER(tractor), position);
    mlt_frame frame = nullptr;
    mlt_service_get_frame(MLT_TRACTOR_SERVICE(tractor), &frame, 0);
    Image output;
    if (frame) {
        for (const auto &property : frameProperties) {
            mlt_properties_set(MLT_FRAME_PROPERTIES(frame), property.first, property.second.c_str());
        }
        uint8_t *image = nullptr;
        mlt_image_format format = mlt_image_rgba;
        int width = profile->width;
        int height = profile->height;
        if (!mlt_frame_get_image(frame, &image, &format, &width, &height, 0) && image && format == mlt_image_rgba) {
            output.width = width;
            output.height = height;
            output.pixels.assign(image, image + size_t(width) * size_t(height) * 4);
        } else {
            fprintf(stderr, "timeline: get_image failed, format %d\n", int(format));
        }
        mlt_frame_close(frame);
    }
    for (mlt_transition transition : transitions) {
        mlt_transition_close(transition);
    }
    mlt_tractor_close(tractor);
    for (mlt_tractor trackTractor : trackTractors) {
        mlt_tractor_close(trackTractor);
    }
    for (mlt_playlist playlist : playlists) {
        mlt_playlist_close(playlist);
    }
    for (mlt_filter filter : filters) {
        mlt_filter_close(filter);
    }
    for (mlt_producer producer : producers) {
        mlt_producer_close(producer);
    }
    return output;
}

void expectSameTimeline(mlt_profile profile, const std::vector<Track> &tracks, const Properties &frameProperties, const char *what, mlt_position position = 0,
                        bool nested = false)
{
    const Image cpu = renderTimeline(profile, tracks, true, frameProperties, position, nested);
    const Image gpu = renderTimeline(profile, tracks, false, frameProperties, position, nested);
    const Difference d = compare(cpu, gpu);
    if (d.sizeMismatch) {
        check(false, "%s: output size mismatch", what);
        return;
    }
    check(d.maxAlpha == 0 && d.maxPremultiplied <= 1.0, "%s: alpha diff %d, premultiplied diff %.2f, channel diff %d (%ld pixels)", what, d.maxAlpha,
          d.maxPremultiplied, d.maxChannel, d.pixels);
}

/** Track compositing: MLT's qtblend transition */
void testCompositing(mlt_profile profile)
{
    if (!hasFilter("qtblend") || !mlt_properties_get_data(mlt_repository_transitions(mlt_factory_repository()), "qtblend", nullptr)) {
        printf("SKIP: MLT has no qtblend transition\n");
        return;
    }
    const Properties smooth = {{"consumer.rescale", "bilinear"}};
    const Properties nearest = {{"consumer.rescale", "nearest"}};
    const Image background = makeImage(160, 90, 4);
    const Image clip1 = makeImage(160, 90, 2);
    const Image clip2 = makeImage(120, 80, 1);
    const Image clip3 = makeImage(160, 90, 0);
    // Picture in picture on two tracks, Kdenlive style compositing (no rectangle)
    const std::vector<Track> pip = {
        {&background, {}, {}},
        {&clip1, {{"qtblend", {{"rect", "10 5 80 45 1"}}}}, {}},
        {&clip2, {{"qtblend", {{"rect", "70 30 80 50 0.8"}, {"rotation", "12"}}}, {"dropshadow", {{"radius", "3"}}}}, {}},
    };
    for (const Properties *frameProperties : {&smooth, &nearest}) {
        expectSameTimeline(profile, pip, *frameProperties, frameProperties == &smooth ? "picture in picture, smooth" : "picture in picture, nearest");
    }
    expectSameTimeline(profile, pip, smooth, "picture in picture, tracks in tractors", 0, true);
    // Transitions with their own geometry, opacity and rotation
    expectSameTimeline(profile,
                       {{&background, {}, {}},
                        {&clip3, {{"box_blur", {{"hradius", "2"}, {"vradius", "2"}}}}, {{"rect", "20 10 100 60 0.7"}}},
                        {&clip2, {{"brightness", {{"level", "0.8"}}}}, {{"rect", "-10 -5 120 70 1"}, {"rotation", "-8"}, {"rotate_center", "1"}}}},
                       smooth, "transition geometry");
    // An opaque top track hides the tracks below
    expectSameTimeline(profile, {{&background, {}, {}}, {&clip2, {}, {}}, {&clip1, {{"box_blur", {{"hradius", "3"}, {"vradius", "1"}}}}, {}}}, smooth,
                       "opaque top track");

    if (g_mock.active()) {
        // The whole timeline stays on the GPU: each clip is uploaded once, the
        // result downloaded once (besides the results of the opacity checks)
        g_mock.reset();
        const Image result = renderTimeline(profile, pip, false, smooth);
        const MockCudaStats stats = g_mock.stats();
        const uint64_t imageBytes = uint64_t(background.pixels.size() + clip1.pixels.size() + clip2.pixels.size());
        const uint64_t frameBytes = uint64_t(result.pixels.size());
        check(!result.pixels.empty() && stats.downloadedBytes < frameBytes + 64 && stats.uploadedBytes < imageBytes + imageBytes / 2,
              "timeline: %llu bytes downloaded for a %llu bytes frame, %llu bytes uploaded for %llu bytes of images", (unsigned long long)stats.downloadedBytes,
              (unsigned long long)frameBytes, (unsigned long long)stats.uploadedBytes, (unsigned long long)imageBytes);
        check(stats.misuse == 0, "timeline: %llu invalid CUDA calls", (unsigned long long)stats.misuse);

        // The same when each track is a tractor: the images of the clips'
        // effects stay on the GPU across them
        g_mock.reset();
        const Image nestedResult = renderTimeline(profile, pip, false, smooth, 0, true);
        const MockCudaStats nestedStats = g_mock.stats();
        check(!nestedResult.pixels.empty() && nestedStats.downloadedBytes < frameBytes + 64 && nestedStats.uploadedBytes < imageBytes + imageBytes / 2,
              "timeline with tracks in tractors: %llu bytes downloaded for a %llu bytes frame, %llu bytes uploaded for %llu bytes of images",
              (unsigned long long)nestedStats.downloadedBytes, (unsigned long long)frameBytes, (unsigned long long)nestedStats.uploadedBytes,
              (unsigned long long)imageBytes);
        check(nestedStats.misuse == 0, "timeline with tracks in tractors: %llu invalid CUDA calls", (unsigned long long)nestedStats.misuse);
    }
}

std::string tempPath(const std::string &name);

/** Renders frames of MLT's luma transition from a clip on track 0 (the
 *  bottom) to one on track 1, with filters on both clips, at positions of a
 *  transition of 10 frames. With converters, the clips have an avcolor_space
 *  filter, which lets MLT convert their images as the loader's does. With a
 *  background, the tractor is a track composited onto it by a qtblend
 *  transition, as in Kdenlive's projects. The images are asked for in RGBA,
 *  or in YUV 4:2:2 as consumers do: the YUYV image followed by its alpha
 *  plane, if any. */
std::vector<Image> renderLuma(mlt_profile profile, const Image &bottom, const Image &top, const std::vector<FilterSpec> &bottomFilters,
                              const std::vector<FilterSpec> &topFilters, const Properties &lumaProperties, bool cpu, const std::vector<mlt_position> &positions,
                              bool converters = true, const Image *background = nullptr, mlt_image_format requestFormat = mlt_image_rgba)
{
    mlt_tractor tractor = mlt_tractor_new();
    mlt_multitrack multitrack = mlt_tractor_multitrack(tractor);
    mlt_field field = mlt_tractor_field(tractor);
    std::vector<mlt_producer> producers;
    std::vector<mlt_filter> filters;
    const Image *images[2] = {&bottom, &top};
    const std::vector<FilterSpec> *specs[2] = {&bottomFilters, &topFilters};
    for (int i = 0; i < 2; ++i) {
        mlt_producer producer = patternProducer(profile, images[i]);
        if (converters) {
            mlt_filter converter = mlt_factory_filter(profile, "avcolor_space", nullptr);
            mlt_producer_attach(producer, converter);
            filters.push_back(converter);
        }
        for (mlt_filter filter : createFilters(profile, *specs[i], cpu)) {
            mlt_producer_attach(producer, filter);
            filters.push_back(filter);
        }
        mlt_multitrack_connect(multitrack, producer, i);
        producers.push_back(producer);
    }
    mlt_transition transition = mlt_factory_transition(profile, "luma", nullptr);
    mlt_properties properties = MLT_TRANSITION_PROPERTIES(transition);
    for (const auto &property : lumaProperties) {
        mlt_properties_set(properties, property.first, property.second.c_str());
    }
    mlt_properties_set_int(properties, "_gpufx.disable", cpu ? 1 : 0);
    mlt_transition_set_in_and_out(transition, 0, 9);
    mlt_field_plant_transition(field, transition, 0, 1);
    mlt_tractor outer = nullptr;
    mlt_transition compositing = nullptr;
    if (background) {
        outer = mlt_tractor_new();
        mlt_producer producer = patternProducer(profile, background);
        if (converters) {
            mlt_filter converter = mlt_factory_filter(profile, "avcolor_space", nullptr);
            mlt_producer_attach(producer, converter);
            filters.push_back(converter);
        }
        producers.push_back(producer);
        mlt_multitrack_connect(mlt_tractor_multitrack(outer), producer, 0);
        mlt_multitrack_connect(mlt_tractor_multitrack(outer), MLT_TRACTOR_PRODUCER(tractor), 1);
        compositing = mlt_factory_transition(profile, "qtblend", nullptr);
        mlt_properties_set_int(MLT_TRANSITION_PROPERTIES(compositing), "always_active", 1);
        mlt_properties_set_int(MLT_TRANSITION_PROPERTIES(compositing), "_gpufx.disable", cpu ? 1 : 0);
        mlt_field_plant_transition(mlt_tractor_field(outer), compositing, 0, 1);
    }
    mlt_tractor output = outer ? outer : tractor;
    std::vector<Image> outputs;
    for (mlt_position position : positions) {
        mlt_producer_seek(MLT_TRACTOR_PRODUCER(output), position);
        mlt_frame frame = nullptr;
        mlt_service_get_frame(MLT_TRACTOR_SERVICE(output), &frame, 0);
        Image output;
        if (frame) {
            uint8_t *image = nullptr;
            mlt_image_format format = requestFormat;
            int width = profile->width;
            int height = profile->height;
            if (!mlt_frame_get_image(frame, &image, &format, &width, &height, 0) && image && format == requestFormat) {
                output.width = width;
                output.height = height;
                const size_t pixels = size_t(width) * size_t(height);
                output.pixels.assign(image, image + pixels * (format == mlt_image_rgba ? 4 : 2));
                if (format == mlt_image_yuv422) {
                    if (const uint8_t *alpha = mlt_frame_get_alpha(frame)) {
                        output.pixels.insert(output.pixels.end(), alpha, alpha + pixels);
                    }
                    output.pixels.resize((output.pixels.size() + 3) / 4 * 4);
                }
            } else {
                fprintf(stderr, "luma: get_image failed, format %d\n", int(format));
            }
            mlt_frame_close(frame);
        }
        outputs.push_back(output);
    }
    if (outer) {
        mlt_transition_close(compositing);
        mlt_tractor_close(outer);
    }
    mlt_transition_close(transition);
    mlt_tractor_close(tractor);
    for (mlt_filter filter : filters) {
        mlt_filter_close(filter);
    }
    for (mlt_producer producer : producers) {
        mlt_producer_close(producer);
    }
    return outputs;
}

/** Kdenlive's dissolves, mixes and wipes: MLT's luma transition */
void testLuma(mlt_profile profile)
{
    if (!mlt_properties_get_data(mlt_repository_transitions(mlt_factory_repository()), "luma", nullptr) || !hasFilter("avcolor_space")) {
        printf("SKIP: MLT has no luma transition or avcolor_space filter\n");
        return;
    }
    // A luma map: a 16-bit PGM with a diagonal gradient
    const std::string map = tempPath("gpufx_luma_map.pgm");
    {
        const int w = 64;
        const int h = 36;
        std::string pgm = "P5\n" + std::to_string(w) + " " + std::to_string(h) + "\n65535\n";
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const int v = (x * 65535 / (w - 1) + y * 65535 / (h - 1)) / 2 ^ ((x * y) & 0x3f);
                pgm += char(v >> 8);
                pgm += char(v & 0xff);
            }
        }
        FILE *file = fopen(map.c_str(), "wb");
        if (!file) {
            check(false, "cannot write %s", map.c_str());
            return;
        }
        fwrite(pgm.data(), 1, pgm.size(), file);
        fclose(file);
    }
    const std::vector<mlt_position> positions = {0, 1, 3, 5, 8, 9};
    const std::vector<FilterSpec> none;
    const std::vector<FilterSpec> blur = {{"box_blur", {{"hradius", "20"}, {"vradius", "15"}}}};
    struct Case
    {
        const char *what;
        Properties properties;
        int bottomPattern;
        int topPattern;
        int width;
        bool converters;
    };
    const std::string resource = map;
    const std::vector<Case> cases = {
        {"dissolve with transparency", {{"alpha_over", "1"}, {"resource", ""}}, 1, 3, 0, true},
        {"dissolve with transparency, opaque", {{"alpha_over", "1"}}, 4, 4, 0, true},
        {"dissolve with transparency, reverse", {{"alpha_over", "1"}, {"reverse", "1"}}, 2, 0, 0, true},
        {"dissolve", {{"fix_background_alpha", "1"}, {"resource", ""}}, 1, 3, 0, true},
        {"dissolve, opaque", {{"fix_background_alpha", "1"}}, 4, 4, 0, true},
        {"dissolve, invert", {{"fix_background_alpha", "1"}, {"invert", "1"}}, 0, 2, 0, true},
        {"dissolve, width not a multiple of 8", {{"fix_background_alpha", "1"}}, 1, 2, 166, true},
        {"dissolve, narrow", {{"fix_background_alpha", "1"}}, 3, 1, 6, true},
        {"dissolve without background alpha fix", {}, 1, 3, 0, true},
        {"wipe", {{"resource", resource}, {"progressive", "1"}}, 4, 4, 0, true},
        {"wipe, soft, translucent", {{"resource", resource}, {"progressive", "1"}, {"softness", "0.3"}}, 1, 3, 0, true},
        {"wipe with transparency, soft, translucent", {{"resource", resource}, {"progressive", "1"}, {"softness", "0.2"}, {"alpha_over", "1"}}, 2, 0, 0,
         true},
        {"wipe, reverse, soft", {{"resource", resource}, {"progressive", "1"}, {"softness", "0.5"}, {"reverse", "1"}}, 4, 2, 0, true},
    };
    const int profileWidth = profile->width;
    const int profileColorspace = profile->colorspace;
    for (int colorspace : {profileColorspace, profileColorspace == 709 ? 601 : 709}) {
        profile->colorspace = colorspace;
        for (const Case &c : cases) {
            profile->width = c.width ? c.width : profileWidth;
            const Image bottom = makeImage(profile->width, profile->height, c.bottomPattern);
            const Image top = makeImage(profile->width, profile->height, c.topPattern);
            // The clips have an effect on the GPU, the transition finds their
            // images there
            for (mlt_image_format request : {mlt_image_rgba, mlt_image_yuv422}) {
                const std::vector<Image> cpu = renderLuma(profile, bottom, top, blur, blur, c.properties, true, positions, c.converters, nullptr, request);
                const std::vector<Image> gpu = renderLuma(profile, bottom, top, blur, blur, c.properties, false, positions, c.converters, nullptr, request);
                for (size_t i = 0; i < positions.size(); ++i) {
                    const Difference d = compare(cpu[i], gpu[i]);
                    check(!cpu[i].pixels.empty() && !d.sizeMismatch && d.pixels == 0, "luma, %s, colorspace %d, %s, frame %d: %ld pixels differ, channel diff %d",
                          c.what, colorspace, mlt_image_format_name(request), int(positions[i]), d.pixels, d.maxChannel);
                }
            }
        }
    }
    profile->width = profileWidth;
    profile->colorspace = profileColorspace;

    if (g_mock.active() && hasFilter("box_blur")) {
        // Images in host memory are blended in RGBA on the CPU, in YUV on the
        // GPU, which also saves the conversions
        for (size_t index : {size_t(0), size_t(3), size_t(10)}) {
            const Image bottom = makeImage(profile->width, profile->height, 1);
            const Image top = makeImage(profile->width, profile->height, 2);
            g_mock.reset();
            renderLuma(profile, bottom, top, none, none, cases[index].properties, false, {4});
            check((g_mock.stats().launches > 0) == (index == 3), "luma, %s, images in host memory: %llu kernel launches", cases[index].what,
                  (unsigned long long)g_mock.stats().launches);
            // Kdenlive's projects set fix_background_alpha, which the YUV paths need
            Properties properties = cases[index].properties;
            properties.push_back({"fix_background_alpha", "1"});
            // (the first frame of a wipe loads the luma map on the CPU)
            g_mock.reset();
            renderLuma(profile, bottom, top, none, none, properties, false, {3, 4}, true, nullptr, mlt_image_yuv422);
            check(g_mock.stats().launches > 0, "luma, %s, images in host memory, YUV 4:2:2: blended on the CPU", cases[index].what);
        }

        // Clips whose images are on the GPU stay there through the
        // transition: one download for the result
        const Image bottom = makeImage(profile->width, profile->height, 1);
        const Image top = makeImage(profile->width, profile->height, 2);
        for (size_t index : {size_t(0), size_t(3), size_t(10)}) {
            const Case &c = cases[index];
            // A wipe's first frame loads the luma map on the CPU
            const uint64_t downloads = index == 10 ? 3 : 2;
            const std::vector<mlt_position> two = {2, 4};
            const std::vector<Image> cpu = renderLuma(profile, bottom, top, blur, blur, c.properties, true, two);
            g_mock.reset();
            const std::vector<Image> gpu = renderLuma(profile, bottom, top, blur, blur, c.properties, false, two);
            const MockCudaStats stats = g_mock.stats();
            check(compare(cpu[1], gpu[1]).pixels == 0, "luma, %s, after GPU effects: the result differs", c.what);
            check(stats.downloads == downloads, "luma, %s, after GPU effects: %llu downloads for 2 frames", c.what, (unsigned long long)stats.downloads);
            check(stats.misuse == 0, "luma, %s: %llu invalid CUDA calls", c.what, (unsigned long long)stats.misuse);

            // The same in a track composited onto a background, as in
            // Kdenlive's projects: the transition's result stays on the GPU
            // for the compositing
            const Image background = makeImage(profile->width, profile->height, 4);
            const std::vector<Image> cpuTrack = renderLuma(profile, bottom, top, blur, blur, c.properties, true, two, true, &background);
            g_mock.reset();
            const std::vector<Image> gpuTrack = renderLuma(profile, bottom, top, blur, blur, c.properties, false, two, true, &background);
            const MockCudaStats trackStats = g_mock.stats();
            for (size_t i = 0; i < two.size(); ++i) {
                const Difference d = compare(cpuTrack[i], gpuTrack[i]);
                check(!cpuTrack[i].pixels.empty() && !d.sizeMismatch && d.pixels == 0, "luma, %s, in a composited track, frame %d: %ld pixels differ", c.what,
                      int(two[i]), d.pixels);
            }
            // Besides the results, the compositing downloads its opacity checks
            check(trackStats.downloads == downloads + 2, "luma, %s, in a composited track: %llu downloads for 2 frames", c.what,
                  (unsigned long long)trackStats.downloads);

            // A consumer asks for YUV 4:2:2, and the compositing asks the track
            // for it first: the transition's YUV result stays on the GPU, one
            // download per frame
            const std::vector<Image> cpuYuv = renderLuma(profile, bottom, top, blur, blur, c.properties, true, two, true, &background, mlt_image_yuv422);
            g_mock.reset();
            const std::vector<Image> gpuYuv = renderLuma(profile, bottom, top, blur, blur, c.properties, false, two, true, &background, mlt_image_yuv422);
            const MockCudaStats yuvStats = g_mock.stats();
            for (size_t i = 0; i < two.size(); ++i) {
                const Difference d = compare(cpuYuv[i], gpuYuv[i]);
                check(!cpuYuv[i].pixels.empty() && !d.sizeMismatch && d.pixels == 0, "luma, %s, in a composited track, YUV 4:2:2, frame %d: %ld pixels differ",
                      c.what, int(two[i]), d.pixels);
            }
            const uint64_t frameBytes = uint64_t(profile->width) * uint64_t(profile->height) * 3;
            check(index == 10 || yuvStats.downloadedBytes < 2 * frameBytes + 64, "luma, %s, in a composited track, YUV 4:2:2: %llu bytes downloaded for 2 frames",
                  c.what, (unsigned long long)yuvStats.downloadedBytes);
            check(yuvStats.misuse == 0, "luma, %s, in a composited track, YUV 4:2:2: %llu invalid CUDA calls", c.what, (unsigned long long)yuvStats.misuse);
        }
    }
    remove(map.c_str());
}

void testChains(mlt_profile profile)
{
    const Image input = makeImage(120, 80, 2);
    const std::vector<FilterSpec> chain = {
        {"brightness", {{"level", "0.8"}}},
        {"box_blur", {{"hradius", "15"}, {"vradius", "15"}}},
        {"dropshadow", {{"radius", "6"}, {"x", "4"}, {"y", "6"}}},
        {"lift_gamma_gain", {{"gain_g", "1.2"}}},
        {"boxblur", {{"blur", "2"}}},
    };
    expectSame(profile, chain, input, false);
    if (g_mock.active() && hasFilters(chain)) {
        // The whole chain runs on the GPU with a single upload and download
        g_mock.reset();
        const Image result = render(profile, chain, input, false);
        const MockCudaStats stats = g_mock.stats();
        // Only small lookup tables are uploaded besides the frame
        const uint64_t frameBytes = uint64_t(input.width) * uint64_t(input.height) * 4;
        check(stats.uploadedBytes < 2 * frameBytes && stats.downloads == 1, "chain: %llu bytes uploaded and %llu downloads, expected one frame each way",
              (unsigned long long)stats.uploadedBytes, (unsigned long long)stats.downloads);
        check(stats.misuse == 0, "chain: %llu invalid CUDA calls", (unsigned long long)stats.misuse);
        check(!result.pixels.empty(), "chain: no output");

        // A frame goes to and from the GPU through page-locked memory (not
        // with MOCKCUDA_NO_PINNED), and threads wait for the GPU on an event
        const Image large = makeImage(400, 300, 2);
        const std::vector<FilterSpec> shadow = {{"dropshadow", {{"radius", "6"}}}};
        expectSame(profile, shadow, large, false);
        g_mock.reset();
        render(profile, shadow, large, false);
        const MockCudaStats largeStats = g_mock.stats();
        const char *noPinned = getenv("MOCKCUDA_NO_PINNED");
        const uint64_t pinned = noPinned && noPinned[0] && strcmp(noPinned, "0") != 0 ? 0 : 1;
        check(largeStats.downloads == 1 && largeStats.pinnedUploads == pinned && largeStats.pinnedDownloads == pinned,
              "page-locked copies: %llu of %llu uploads and %llu of %llu downloads, expected %llu each", (unsigned long long)largeStats.pinnedUploads,
              (unsigned long long)largeStats.uploads, (unsigned long long)largeStats.pinnedDownloads, (unsigned long long)largeStats.downloads,
              (unsigned long long)pinned);
        check(largeStats.eventWaits > 0, "no thread waited for the GPU on an event");

        // A lightweight effect on its own stays on the CPU
        g_mock.reset();
        render(profile, {{"brightness", {{"level", "0.5"}}}}, input, false);
        check(g_mock.stats().uploads == 0, "a single brightness filter was uploaded to the GPU");

        // Effects with nothing to do stay transparent between GPU effects
        g_mock.reset();
        expectSame(profile,
                   {{"dropshadow", {{"radius", "5"}}},
                    {"box_blur", {{"hradius", "0"}, {"vradius", "0"}}},
                    {"brightness", {{"level", "1"}}},
                    {"boxblur", {{"blur", "2"}}}},
                   input, false);
        g_mock.reset();
        render(profile,
               {{"dropshadow", {{"radius", "5"}}},
                {"box_blur", {{"hradius", "0"}, {"vradius", "0"}}},
                {"brightness", {{"level", "1"}}},
                {"boxblur", {{"blur", "2"}}}},
               input, false);
        check(g_mock.stats().uploads == 1 && g_mock.stats().downloads == 1, "pass-through filters broke the GPU chain (%llu uploads, %llu downloads)",
              (unsigned long long)g_mock.stats().uploads, (unsigned long long)g_mock.stats().downloads);
    }
    // A CPU-only filter in the middle splits the chain, the result is the same
    expectSame(
        profile,
        {{"dropshadow", {{"radius", "8"}}}, {"gradientmap", {{"stop.1", "#ff000000 0.0"}, {"stop.2", "#ffffffff 1.0"}}}, {"box_blur", {{"hradius", "10"}}}},
        input, false);
}

/** Through a real producer: format conversions, and filters attached to a service. */
void testProducer(mlt_profile profile)
{
    if (!hasFilters({{"qtcrop", {}}, {"dropshadow", {}}, {"brightness", {}}})) {
        return;
    }
    auto renderProducer = [&](bool cpu, mlt_image_format requested) {
        mlt_producer producer = mlt_factory_producer(profile, nullptr, "color:#a0306090");
        if (!producer) {
            return Image();
        }
        std::vector<mlt_filter> filters = createFilters(profile,
                                                        {{"qtcrop", {{"rect", "10 10 100 60"}, {"radius", "0.2"}}},
                                                         {"dropshadow", {{"radius", "9"}, {"x", "6"}, {"y", "-4"}}},
                                                         {"brightness", {{"level", "0.7"}}}},
                                                        cpu);
        for (mlt_filter filter : filters) {
            mlt_producer_attach(producer, filter);
        }
        mlt_frame frame = nullptr;
        mlt_service_get_frame(MLT_PRODUCER_SERVICE(producer), &frame, 0);
        Image output;
        if (frame) {
            uint8_t *image = nullptr;
            mlt_image_format format = requested;
            int width = profile->width;
            int height = profile->height;
            if (!mlt_frame_get_image(frame, &image, &format, &width, &height, 0) && image) {
                output.width = width;
                output.height = height;
                output.pixels.assign(image, image + size_t(width) * size_t(height) * (format == mlt_image_rgba ? 4 : 2));
            }
            mlt_frame_close(frame);
        }
        for (mlt_filter filter : filters) {
            mlt_filter_close(filter);
        }
        mlt_producer_close(producer);
        return output;
    };
    for (mlt_image_format format : {mlt_image_rgba, mlt_image_yuv422}) {
        const Image cpu = renderProducer(true, format);
        const Image gpu = renderProducer(false, format);
        check(!cpu.pixels.empty() && cpu.pixels.size() == gpu.pixels.size(), "producer test: no image for format %d", int(format));
        if (cpu.pixels.size() == gpu.pixels.size()) {
            int maxDiff = 0;
            for (size_t i = 0; i < cpu.pixels.size(); ++i) {
                maxDiff = std::max(maxDiff, std::abs(int(cpu.pixels[i]) - int(gpu.pixels[i])));
            }
            check(maxDiff <= 1, "producer test (format %d): max difference %d", int(format), maxDiff);
        }
    }
}

/* ---------------------------------------------------------------------------
 * Format conversions
 * ------------------------------------------------------------------------- */

/** An image returned by mlt_frame_get_image(), with what the format
 *  conversions leave on the frame. */
struct Output
{
    int width = 0;
    int height = 0;
    int format = mlt_image_none;
    std::vector<uint8_t> data;
    std::vector<uint8_t> alpha;
    int colorspace = 0;
    int fullRange = 0;
};

Output fetchOutput(mlt_frame frame, mlt_image_format format, int width, int height)
{
    Output output;
    uint8_t *image = nullptr;
    if (mlt_frame_get_image(frame, &image, &format, &width, &height, 0) || !image) {
        fprintf(stderr, "get_image failed, format %d\n", int(format));
        return output;
    }
    output.width = width;
    output.height = height;
    output.format = format;
    const size_t pixels = size_t(width) * size_t(height);
    const size_t bytes = format == mlt_image_rgba ? pixels * 4 : format == mlt_image_yuv422 ? pixels * 2 : format == mlt_image_yuv420p ? pixels * 3 / 2
                                                                                   : format == mlt_image_yuv420p10 ? pixels * 3
                                                                                                                   : 0;
    output.data.assign(image, image + bytes);
    int alphaSize = 0;
    uint8_t *alpha = mlt_frame_get_alpha_size(frame, &alphaSize);
    if (alpha && size_t(alphaSize) >= pixels) {
        output.alpha.assign(alpha, alpha + pixels);
    }
    output.colorspace = mlt_properties_get_int(MLT_FRAME_PROPERTIES(frame), "colorspace");
    output.fullRange = mlt_properties_get_int(MLT_FRAME_PROPERTIES(frame), "full_range");
    return output;
}

/** tolerance: largest difference allowed per byte, for results drawn the
 *  way Qt does, whose un-premultiplication is approximate */
void expectSameOutput(const Output &cpu, const Output &gpu, const std::string &what, int tolerance = 0)
{
    const bool sameShape = !cpu.data.empty() && cpu.format == gpu.format && cpu.width == gpu.width && cpu.height == gpu.height && cpu.data.size() == gpu.data.size();
    check(sameShape, "%s: format %d %dx%d on the CPU, %d %dx%d on the GPU", what.c_str(), cpu.format, cpu.width, cpu.height, gpu.format, gpu.width, gpu.height);
    if (!sameShape) {
        return;
    }
    size_t differences = 0;
    size_t first = 0;
    int maxDiff = 0;
    for (size_t i = 0; i < cpu.data.size(); ++i) {
        const int diff = std::abs(int(cpu.data[i]) - int(gpu.data[i]));
        if (diff) {
            first = differences++ ? first : i;
            maxDiff = std::max(maxDiff, diff);
        }
    }
    const size_t bytesPerPixel = cpu.data.size() / std::max<size_t>(size_t(cpu.width) * size_t(cpu.height), 1);
    const size_t pixel = first / std::max<size_t>(bytesPerPixel, 1);
    check(maxDiff <= tolerance, "%s: %zu bytes differ (max %d), the first at %zu,%zu byte %zu: %d on the CPU, %d on the GPU", what.c_str(), differences, maxDiff,
          pixel % size_t(std::max(cpu.width, 1)), pixel / size_t(std::max(cpu.width, 1)), first % std::max<size_t>(bytesPerPixel, 1),
          cpu.data.empty() ? 0 : cpu.data[first], gpu.data.empty() ? 0 : gpu.data[first]);
    check(cpu.alpha == gpu.alpha, "%s: the alpha planes differ (%zu and %zu bytes)", what.c_str(), cpu.alpha.size(), gpu.alpha.size());
    check(cpu.colorspace == gpu.colorspace && cpu.fullRange == gpu.fullRange, "%s: colorspace %d / %d, full range %d / %d", what.c_str(), cpu.colorspace,
          gpu.colorspace, cpu.fullRange, gpu.fullRange);
}

Output withoutLastRows(Output output, int rows)
{
    if (rows > 0 && output.width > 0 && output.height > rows) {
        const size_t rowBytes = output.data.size() / size_t(output.height);
        output.height -= rows;
        output.data.resize(rowBytes * size_t(output.height));
        if (!output.alpha.empty()) {
            output.alpha.resize(size_t(output.width) * size_t(output.height));
        }
    }
    return output;
}

/** Writes a YUV4MPEG2 file of frames with random 4:2:0 content, which the
 *  avformat producer decodes as yuv420p. parameters holds stream parameters
 *  such as the interlacing ("Ip", "It"). */
bool writeY4m(const std::string &path, int width, int height, int frames, const char *parameters)
{
    FILE *file = fopen(path.c_str(), "wb");
    if (!file) {
        return false;
    }
    fprintf(file, "YUV4MPEG2 W%d H%d F25:1 A1:1 C420jpeg %s\n", width, height, parameters);
    std::vector<uint8_t> data(size_t(width) * size_t(height) + 2 * size_t((width + 1) / 2) * size_t((height + 1) / 2));
    uint32_t seed = uint32_t(width) * 7919u + uint32_t(height);
    for (int i = 0; i < frames; ++i) {
        for (uint8_t &value : data) {
            seed = seed * 1664525u + 1013904223u;
            value = uint8_t(seed >> 24);
        }
        fputs("FRAME\n", file);
        fwrite(data.data(), 1, data.size(), file);
    }
    fclose(file);
    return true;
}

std::string tempPath(const std::string &name)
{
    return (std::filesystem::temp_directory_path() / name).string();
}

/** The image of the avformat producer at position, with filters attached to
 *  the producer. */
Output decode(mlt_profile profile, const std::string &path, bool cpu, mlt_position position, const Properties &producerProperties,
              const std::vector<FilterSpec> &specs = {})
{
    Output output;
    mlt_producer producer = mlt_factory_producer(profile, "avformat", path.c_str());
    if (!producer) {
        fprintf(stderr, "cannot open %s\n", path.c_str());
        return output;
    }
    mlt_properties properties = MLT_PRODUCER_PROPERTIES(producer);
    mlt_properties_set_int(properties, "_gpufx.disable", cpu ? 1 : 0);
    for (const auto &property : producerProperties) {
        mlt_properties_set(properties, property.first, property.second.c_str());
    }
    std::vector<mlt_filter> filters = createFilters(profile, specs, cpu);
    for (mlt_filter filter : filters) {
        mlt_producer_attach(producer, filter);
    }
    mlt_producer_seek(producer, position);
    mlt_frame frame = nullptr;
    mlt_service_get_frame(MLT_PRODUCER_SERVICE(producer), &frame, 0);
    if (frame) {
        output = fetchOutput(frame, mlt_image_rgba, profile->width, profile->height);
        mlt_frame_close(frame);
    }
    for (mlt_filter filter : filters) {
        mlt_filter_close(filter);
    }
    mlt_producer_close(producer);
    return output;
}

void testDecoding(mlt_profile profile)
{
    if (!mlt_properties_get_data(mlt_repository_producers(mlt_factory_repository()), "avformat", nullptr)) {
        printf("SKIP: MLT has no avformat producer\n");
        return;
    }
    struct Case
    {
        const char *name;
        int width, height;
        const char *parameters;
        Properties producer;
        bool onGpu;
        // Rows at the bottom whose values MLT does not reproduce
        int unreliableRows = 0;
    };
    const Case cases[] = {
        {"720p", 1280, 720, "Ip", {}, true}, // BT.709 by default
        {"pal", 720, 576, "Ip", {}, true},   // BT.601 by default
        {"interlaced", 720, 576, "It", {}, true},
        // The producer converts the fields of interlaced frames separately,
        // which needs a height that is a multiple of 4 on the GPU. With an odd
        // field height it reads one chroma row past the planes for the last
        // rows of the second field.
        {"interlaced 486 lines", 720, 486, "Ib", {}, false, 3},
        {"full range", 640, 360, "Ip XCOLORRANGE=FULL", {}, true},
        {"full range interlaced", 640, 360, "It XCOLORRANGE=FULL", {}, true},
        {"odd width", 33, 20, "Ip", {}, false},
        {"bt2020", 640, 360, "Ip", {{"force_colorspace", "2020"}}, true},
        {"smpte240m", 640, 360, "Ip", {{"force_colorspace", "240"}}, true},
        {"bt601 hd", 1280, 720, "Ip", {{"force_colorspace", "601"}}, true},
        {"forced full range", 320, 180, "Ip", {{"force_full_range", "1"}}, true},
    };
    for (const Case &c : cases) {
        const std::string path = tempPath(std::string("gpufx_test_") + std::to_string(c.width) + "x" + std::to_string(c.height) + ".y4m");
        if (!writeY4m(path, c.width, c.height, 3, c.parameters)) {
            check(false, "decoding: cannot write %s", path.c_str());
            continue;
        }
        g_mock.reset();
        const Output gpu = decode(profile, path, false, 1, c.producer);
        const MockCudaStats stats = g_mock.stats();
        const Output cpu = decode(profile, path, true, 1, c.producer);
        expectSameOutput(withoutLastRows(cpu, c.unreliableRows), withoutLastRows(gpu, c.unreliableRows), std::string("decoding ") + c.name);
        if (g_mock.active()) {
            const uint64_t yuvBytes = uint64_t(c.width) * uint64_t(c.height) * 3 / 2;
            if (c.onGpu) {
                check(stats.launches == 1 && stats.uploadedBytes < yuvBytes + 16384 && stats.downloadedBytes == uint64_t(c.width) * uint64_t(c.height) * 4,
                      "decoding %s: %llu launches, %llu bytes uploaded, %llu downloaded", c.name, (unsigned long long)stats.launches,
                      (unsigned long long)stats.uploadedBytes, (unsigned long long)stats.downloadedBytes);
            } else {
                check(stats.launches == 0, "decoding %s: converted on the GPU", c.name);
            }
        }
        remove(path.c_str());
    }

    // The decoded image goes straight into a GPU effect, the RGBA image only
    // exists on the GPU between them
    const std::string path = tempPath("gpufx_test_chain.y4m");
    if (writeY4m(path, 320, 180, 2, "Ip")) {
        const std::vector<FilterSpec> chain = {{"brightness", {{"level", "0.8"}}}, {"box_blur", {{"hradius", "5"}, {"vradius", "3"}}}};
        if (hasFilters(chain)) {
            g_mock.reset();
            const Output gpu = decode(profile, path, false, 0, {}, chain);
            const MockCudaStats stats = g_mock.stats();
            const Output cpu = decode(profile, path, true, 0, {}, chain);
            expectSameOutput(cpu, gpu, "decoding into GPU effects");
            if (g_mock.active()) {
                check(stats.downloads == 1 && stats.uploadedBytes < 320 * 180 * 2,
                      "decoding into GPU effects: %llu uploads (%llu bytes), %llu downloads", (unsigned long long)stats.uploads,
                      (unsigned long long)stats.uploadedBytes, (unsigned long long)stats.downloads);
            }
        }
        remove(path.c_str());
    }
}

/** The image of a video clip as Kdenlive makes it: an MLT chain with the
 *  normalizers MLT attaches, and the filters on the chain. */
Output decodeChain(mlt_profile profile, const std::string &path, bool cpu, const std::vector<FilterSpec> &specs, const Properties &frameProperties = {})
{
    Output output;
    mlt_chain chain = mlt_chain_init(profile);
    mlt_producer source = chain ? mlt_factory_producer(profile, "avformat", path.c_str()) : nullptr;
    if (!source) {
        fprintf(stderr, "cannot open %s\n", path.c_str());
        mlt_chain_close(chain);
        return output;
    }
    mlt_properties_set_int(MLT_PRODUCER_PROPERTIES(source), "_gpufx.disable", cpu ? 1 : 0);
    mlt_chain_set_source(chain, source);
    mlt_producer_close(source);
    mlt_chain_attach_normalizers(chain);
    std::vector<mlt_filter> filters = createFilters(profile, specs, cpu);
    for (mlt_filter filter : filters) {
        mlt_service_attach(MLT_CHAIN_SERVICE(chain), filter);
    }
    mlt_frame frame = nullptr;
    mlt_service_get_frame(MLT_CHAIN_SERVICE(chain), &frame, 0);
    if (frame) {
        for (const auto &property : frameProperties) {
            mlt_properties_set(MLT_FRAME_PROPERTIES(frame), property.first, property.second.c_str());
        }
        output = fetchOutput(frame, mlt_image_rgba, profile->width, profile->height);
        mlt_frame_close(frame);
    }
    for (mlt_filter filter : filters) {
        mlt_filter_close(filter);
    }
    mlt_chain_close(chain);
    return output;
}

/** MLT's normalizing filters between a decoded video and its effects */
void testNormalizers(mlt_profile profile)
{
    mlt_chain probe = mlt_chain_init(profile);
    mlt_producer color = probe ? mlt_factory_producer(profile, "color", "black") : nullptr;
    int links = 0;
    if (color) {
        mlt_chain_set_source(probe, color);
        mlt_producer_close(color);
        mlt_chain_attach_normalizers(probe);
        links = mlt_chain_link_count(probe);
    }
    mlt_chain_close(probe);
    // An effect whose GPU result is exact, which also blurs the padding in
    const std::vector<FilterSpec> blur = {{"box_blur", {{"hradius", "4"}, {"vradius", "3"}}}};
    if (!links || !mlt_properties_get_data(mlt_repository_producers(mlt_factory_repository()), "avformat", nullptr) || !hasFilters(blur)) {
        printf("SKIP: no avformat producer, normalizers or box_blur filter\n");
        return;
    }
    struct Case
    {
        const char *name;
        int width, height;
        const char *parameters;
        Properties frame;
        // The decoded image goes into the effect on the GPU
        bool staysOnGpu;
        // A profile colorspace other than that of the video
        bool otherColorspace;
        // The profile's height when not 0
        int profileHeight;
    };
    const Case cases[] = {
        {"same size", 160, 90, "Ip", {}, true, false, 0},
        // resize pads the narrower image on the GPU
        {"pillarbox", 120, 90, "Ip", {}, true, false, 0},
        {"letterbox", 160, 72, "Ip", {}, true, false, 0},
        // rescale scales with the arithmetic of libswscale on the GPU
        {"scaled", 320, 180, "Ip", {}, true, false, 0},
        {"scaled, nearest", 96, 54, "Ip", {{"consumer.rescale", "nearest"}}, true, false, 0},
        {"scaled, bilinear", 96, 54, "Ip", {{"consumer.rescale", "bilinear"}}, true, false, 0},
        {"scaled, bicubic", 250, 140, "Ip", {{"consumer.rescale", "bicubic"}}, true, false, 0},
        {"scaled, Lanczos", 320, 180, "Ip", {{"consumer.rescale", "hyper"}}, true, false, 0},
        {"scaled, other colorspace", 320, 180, "Ip", {{"consumer.rescale", "bicubic"}}, true, true, 0},
        // fieldorder shifts the image by a line on the CPU. The height is a
        // multiple of 4: MLT's CPU decoding of the fields of an interlaced
        // 4:2:0 frame reads past the chroma otherwise.
        {"field order", 160, 88, "It", {{"consumer.top_field_first", "0"}}, false, false, 88},
    };
    const int profileColorspace = profile->colorspace;
    const int profileHeight = profile->height;
    for (const Case &c : cases) {
        profile->colorspace = c.otherColorspace ? (profileColorspace == 709 ? 601 : 709) : profileColorspace;
        profile->height = c.profileHeight ? c.profileHeight : profileHeight;
        const std::string path = tempPath(std::string("gpufx_normalize_") + std::to_string(c.width) + "x" + std::to_string(c.height) + ".y4m");
        if (!writeY4m(path, c.width, c.height, 1, c.parameters)) {
            check(false, "normalizers: cannot write %s", path.c_str());
            continue;
        }
        const std::string what = std::string("normalizers, ") + c.name;
        g_mock.reset();
        const Output gpu = decodeChain(profile, path, false, blur, c.frame);
        const MockCudaStats stats = g_mock.stats();
        const Output cpu = decodeChain(profile, path, true, blur, c.frame);
        expectSameOutput(cpu, gpu, what);
        check(gpu.width == profile->width && gpu.height == profile->height, "%s: %dx%d", what.c_str(), gpu.width, gpu.height);
        if (g_mock.active() && c.staysOnGpu) {
            // The frame, and the filters of a scaling
            const uint64_t yuvBytes = uint64_t(c.width) * uint64_t(c.height) * 3 / 2;
            check(stats.downloads == 1 && stats.uploadedBytes < yuvBytes + 32768, "%s: %llu downloads, %llu bytes uploaded", what.c_str(),
                  (unsigned long long)stats.downloads, (unsigned long long)stats.uploadedBytes);
        }
        // Without effects, nothing changes
        expectSameOutput(decodeChain(profile, path, true, {}, c.frame), decodeChain(profile, path, false, {}, c.frame), what + " without effects");
        remove(path.c_str());
    }
    profile->colorspace = profileColorspace;
    profile->height = profileHeight;
}

/** The properties of a frame, apart from data and the module's own */
std::map<std::string, std::string> frameValues(mlt_frame frame)
{
    std::map<std::string, std::string> values;
    mlt_properties properties = MLT_FRAME_PROPERTIES(frame);
    for (int i = 0; i < mlt_properties_count(properties); ++i) {
        const char *name = mlt_properties_get_name(properties, i);
        // pos.<id> carries the unique id of a filter
        if (name && strncmp(name, "_gpufx", 6) && strncmp(name, "pos.", 4) && !mlt_properties_get_data_at(properties, i, nullptr)) {
            const char *value = mlt_properties_get_value(properties, i);
            values[name] = value ? value : "(null)";
        }
    }
    return values;
}

/** Frames of a still producer at positions, through filters */
struct StillRun
{
    std::vector<Output> outputs;
    std::vector<std::map<std::string, std::string>> values;
    std::vector<std::vector<uint8_t>> alphas;
    std::vector<uint64_t> uploads;
};

StillRun renderStill(mlt_profile profile, const char *service, const char *resource, const Properties &producerProperties,
                     const std::vector<FilterSpec> &specs, bool cpu, const std::vector<int> &positions, bool loader)
{
    StillRun run;
    // MLT's loader adds its normalizing filters, as for a project's producers
    const std::string name = std::string(service) + ":" + (resource ? resource : "");
    mlt_producer producer = loader ? mlt_factory_producer(profile, nullptr, name.c_str()) : mlt_factory_producer(profile, service, resource);
    if (!producer) {
        return run;
    }
    for (const auto &property : producerProperties) {
        mlt_properties_set(MLT_PRODUCER_PROPERTIES(producer), property.first, property.second.c_str());
    }
    mlt_properties_set_int(MLT_PRODUCER_PROPERTIES(producer), "_gpufx.disable", cpu ? 1 : 0);
    // The image conversions of a Kdenlive clip
    mlt_filter converter = loader ? nullptr : mlt_factory_filter(profile, "avcolor_space", nullptr);
    if (converter) {
        mlt_producer_attach(producer, converter);
    }
    std::vector<mlt_filter> filters = createFilters(profile, specs, cpu);
    for (mlt_filter filter : filters) {
        mlt_producer_attach(producer, filter);
    }
    for (int position : positions) {
        mlt_producer_seek(producer, position);
        mlt_frame frame = nullptr;
        g_mock.reset();
        if (mlt_service_get_frame(MLT_PRODUCER_SERVICE(producer), &frame, 0) || !frame) {
            break;
        }
        run.outputs.push_back(fetchOutput(frame, mlt_image_rgba, profile->width, profile->height));
        run.values.push_back(frameValues(frame));
        // The alpha plane a producer sets beside an RGBA image, which
        // mlt_frame_get_alpha_size() does not return
        int alphaSize = 0;
        auto *alpha = static_cast<uint8_t *>(mlt_properties_get_data(MLT_FRAME_PROPERTIES(frame), "alpha", &alphaSize));
        run.alphas.emplace_back(alpha, alpha ? alpha + std::max(alphaSize, 0) : alpha);
        run.uploads.push_back(g_mock.stats().uploads);
        mlt_frame_close(frame);
    }
    for (mlt_filter filter : filters) {
        mlt_filter_close(filter);
    }
    mlt_filter_close(converter);
    mlt_producer_close(producer);
    return run;
}

/** Still images, titles and color clips: kept on the GPU once two frames
 *  gave the same picture, with the same output and frame properties */
void testStills(mlt_profile profile)
{
    const std::vector<FilterSpec> blur = {{"box_blur", {{"hradius", "20"}, {"vradius", "15"}}}};
    if (!hasFilters(blur)) {
        return;
    }
    // A picture for qimage, in a format Qt always reads. 16:9, so that it is
    // scaled to an even width: MLT's conversion of an odd width to YUV 4:2:2
    // (by avdeinterlace) leaves part of the last line unwritten.
    const std::string ppm = tempPath("gpufx_still.ppm");
    if (FILE *file = fopen(ppm.c_str(), "wb")) {
        fprintf(file, "P6\n%d %d\n255\n", 128, 72);
        for (int y = 0; y < 72; ++y) {
            for (int x = 0; x < 128; ++x) {
                const unsigned char pixel[3] = {uint8_t(x * 2), uint8_t(y * 3), uint8_t((x * y) & 255)};
                fwrite(pixel, 1, 3, file);
            }
        }
        fclose(file);
    }
    const std::string title = "<kdenlivetitle width=\"160\" height=\"90\" out=\"100\"><item type=\"QGraphicsRectItem\" z-index=\"0\">"
                              "<position x=\"20\" y=\"10\"><transform>1,0,0,0,1,0,0,0,1</transform></position>"
                              "<content brushcolor=\"200,40,40,255\" pencolor=\"0,0,0,255\" penwidth=\"0\" rect=\"0,0,60,30\"/></item>"
                              "<startviewport rect=\"0,0,160,90\"/><endviewport rect=\"%1\"/><background color=\"0,0,0,0\"/></kdenlivetitle>";
    auto titleXml = [&title](const char *end) {
        std::string xml = title;
        xml.replace(xml.find("%1"), 2, end);
        return xml;
    };
    struct Case
    {
        const char *name;
        const char *service;
        std::string resource;
        Properties properties;
        // Whether the picture is kept on the GPU
        bool kept;
        // Created by MLT's loader
        bool loader;
    };
    const Case cases[] = {
        {"color", "color", "0x3366cc80", {}, true, false},
        // The first frames of each period of 3 frames come from the producer
        {"qimage", "qimage", ppm, {{"ttl", "3"}}, true, false},
        {"title", "kdenlivetitle", "", {{"xmldata", titleXml("0,0,160,90")}}, true, false},
        {"animated title", "kdenlivetitle", "", {{"xmldata", titleXml("40,20,80,45")}}, false, false},
        // Through the deinterlace filter of the loader, when it is xine's
        {"color from the loader", "color", "0x3366cc80", {}, true, true},
        {"qimage from the loader", "qimage", ppm, {{"ttl", "3"}}, true, true},
    };
    const std::vector<int> positions = {0, 1, 2, 3, 4, 5, 6, 1, 2, 7};
    for (const Case &c : cases) {
        const std::string what = std::string("stills, ") + c.name;
        if (!mlt_properties_get_data(mlt_repository_producers(mlt_factory_repository()), c.service, nullptr)) {
            printf("SKIP: %s: no %s producer\n", what.c_str(), c.service);
            continue;
        }
        const char *resource = c.resource.empty() ? nullptr : c.resource.c_str();
        const StillRun cpu = renderStill(profile, c.service, resource, c.properties, blur, true, positions, c.loader);
        const StillRun gpu = renderStill(profile, c.service, resource, c.properties, blur, false, positions, c.loader);
        check(cpu.outputs.size() == positions.size() && gpu.outputs.size() == positions.size(), "%s: %zu and %zu frames", what.c_str(),
              cpu.outputs.size(), gpu.outputs.size());
        if (cpu.outputs.size() != positions.size() || gpu.outputs.size() != positions.size()) {
            continue;
        }
        int kept = 0;
        for (size_t i = 0; i < positions.size(); ++i) {
            const std::string frame = what + ", frame " + std::to_string(positions[i]);
            expectSameOutput(cpu.outputs[i], gpu.outputs[i], frame);
            for (const auto &value : cpu.values[i]) {
                auto it = gpu.values[i].find(value.first);
                check(it != gpu.values[i].end() && it->second == value.second, "%s: property %s is %s on the CPU, %s on the GPU", frame.c_str(),
                      value.first.c_str(), value.second.c_str(), it == gpu.values[i].end() ? "not set" : it->second.c_str());
            }
            for (const auto &value : gpu.values[i]) {
                check(cpu.values[i].count(value.first), "%s: property %s set on the GPU only (%s)", frame.c_str(), value.first.c_str(), value.second.c_str());
            }
            check(cpu.alphas[i] == gpu.alphas[i], "%s: the alpha planes differ (%zu and %zu bytes)", frame.c_str(), cpu.alphas[i].size(),
                  gpu.alphas[i].size());
            kept += gpu.uploads[i] == 0;
        }
        // avdeinterlace, the loader's other choice, converts every image to YUV
        if (g_mock.active() && (!c.loader || hasFilter("deinterlace"))) {
            // The first call to the producer also sets properties of the
            // producer (color, title) or of the next frames (qimage), so its
            // picture is kept from the third frame on. qimage shows a new
            // picture every 3 frames (ttl): the last frame of the second
            // period and the second visits of the first come from the GPU.
            const int expected = !c.kept ? 0 : !strcmp(c.service, "qimage") ? 3 : int(positions.size()) - 3;
            check(kept == expected, "%s: %d frames without upload, expected %d", what.c_str(), kept, expected);
        }
    }
    remove(ppm.c_str());
}

/** Filters on a producer, whose image is requested in format. MLT's
 *  avcolour_space converts it, as when the loader normalizes a producer. */
Output renderFormat(mlt_profile profile, const std::vector<FilterSpec> &specs, const Image &input, bool cpu, mlt_image_format format,
                    const Properties &frameProperties)
{
    Output output;
    mlt_producer producer = patternProducer(profile, &input);
    mlt_filter converter = mlt_factory_filter(profile, "avcolor_space", nullptr);
    if (converter) {
        mlt_producer_attach(producer, converter);
    }
    std::vector<mlt_filter> filters = createFilters(profile, specs, cpu);
    for (mlt_filter filter : filters) {
        mlt_producer_attach(producer, filter);
    }
    mlt_frame frame = nullptr;
    mlt_service_get_frame(MLT_PRODUCER_SERVICE(producer), &frame, 0);
    if (frame) {
        for (const auto &property : frameProperties) {
            mlt_properties_set(MLT_FRAME_PROPERTIES(frame), property.first, property.second.c_str());
        }
        output = fetchOutput(frame, format, input.width, input.height);
        mlt_frame_close(frame);
    }
    for (mlt_filter filter : filters) {
        mlt_filter_close(filter);
    }
    mlt_filter_close(converter);
    mlt_producer_close(producer);
    return output;
}

void testYuvOutput(mlt_profile profile)
{
    if (!hasFilter("avcolor_space") || !hasFilter("box_blur")) {
        printf("SKIP: no avcolor_space or box_blur filter\n");
        return;
    }
    const std::vector<FilterSpec> blur = {{"box_blur", {{"hradius", "6"}, {"vradius", "4"}}}};
    const int colorspace = profile->colorspace;
    struct Case
    {
        int width, height, pattern, colorspace;
        Properties frame;
        bool onGpu;
    };
    const Case cases[] = {
        {160, 90, 1, 709, {}, true},
        {160, 90, 2, 601, {}, true},
        {96, 54, 3, 2020, {}, true},
        {96, 54, 0, 240, {}, true},
        // Full range output stays with the CPU conversion, which differs
        // between FFmpeg versions
        {160, 90, 4, 709, {{"consumer.color_range", "full"}}, false},
        {160, 90, 1, 601, {{"consumer.color_range", "pc"}}, false},
        {160, 90, 2, 709, {{"consumer.color_range", "mpeg"}}, true},
        // Odd widths stay with the CPU conversion
        {97, 54, 2, 709, {}, false},
    };
    for (const Case &c : cases) {
        profile->colorspace = c.colorspace;
        const Image input = makeImage(c.width, c.height, c.pattern);
        g_mock.reset();
        const Output gpu = renderFormat(profile, blur, input, false, mlt_image_yuv422, c.frame);
        const MockCudaStats stats = g_mock.stats();
        const Output cpu = renderFormat(profile, blur, input, true, mlt_image_yuv422, c.frame);
        std::string what = "yuv422 output " + std::to_string(c.width) + "x" + std::to_string(c.height) + " colorspace " + std::to_string(c.colorspace);
        for (const auto &property : c.frame) {
            what += std::string(", ") + property.first + "=" + property.second;
        }
        expectSameOutput(cpu, gpu, what);
        if (g_mock.active()) {
            const uint64_t pixels = uint64_t(c.width) * uint64_t(c.height);
            // YUV 4:2:2 and the alpha plane instead of RGBA
            const uint64_t expected = c.onGpu ? pixels * 3 : pixels * 4;
            check(stats.downloadedBytes == expected, "%s: %llu bytes downloaded, expected %llu", what.c_str(), (unsigned long long)stats.downloadedBytes,
                  (unsigned long long)expected);
        }
    }

    // Planar 4:2:0, which a consumer with mlt_image_format=yuv420p or
    // yuv420p10 asks for
    struct Case420
    {
        int width, height, pattern, colorspace;
        Properties frame;
        bool onGpu;
    };
    const Case420 cases420[] = {
        {160, 90, 1, 709, {}, true},
        {96, 54, 3, 601, {}, true},
        {98, 56, 0, 2020, {}, true},
        {160, 90, 2, 240, {}, true},
        {160, 90, 4, 709, {{"consumer.color_range", "full"}}, false},
        // Odd heights stay with the CPU conversion
        {96, 55, 2, 709, {}, false},
    };
    for (mlt_image_format format : {mlt_image_yuv420p, mlt_image_yuv420p10}) {
        for (const Case420 &c : cases420) {
            profile->colorspace = c.colorspace;
            const Image input = makeImage(c.width, c.height, c.pattern);
            g_mock.reset();
            const Output gpu = renderFormat(profile, blur, input, false, format, c.frame);
            const MockCudaStats stats = g_mock.stats();
            const Output cpu = renderFormat(profile, blur, input, true, format, c.frame);
            std::string what = std::string(mlt_image_format_name(format)) + " output " + std::to_string(c.width) + "x" + std::to_string(c.height) +
                               " colorspace " + std::to_string(c.colorspace);
            for (const auto &property : c.frame) {
                what += std::string(", ") + property.first + "=" + property.second;
            }
            expectSameOutput(cpu, gpu, what);
            if (g_mock.active()) {
                // The YUV planes and the alpha plane, or RGBA
                const uint64_t pixels = uint64_t(c.width) * uint64_t(c.height);
                const uint64_t expected = !c.onGpu ? pixels * 4 : format == mlt_image_yuv420p ? pixels * 5 / 2 : pixels * 4;
                check(stats.downloadedBytes == expected && stats.downloads == (c.onGpu ? 2u : 1u), "%s: %llu bytes downloaded in %llu downloads, expected %llu",
                      what.c_str(), (unsigned long long)stats.downloadedBytes, (unsigned long long)stats.downloads, (unsigned long long)expected);
            }
        }
    }
    profile->colorspace = colorspace;

    // From track compositing
    if (hasFilter("qtblend") || mlt_properties_get_data(mlt_repository_transitions(mlt_factory_repository()), "qtblend", nullptr)) {
        const Image bottom = makeImage(profile->width, profile->height, 1);
        const Image top = makeImage(profile->width, profile->height, 3);
        // rect: the transition's, none for Kdenlive's track compositing
        auto renderYuv = [&](bool cpu, const std::vector<FilterSpec> &topFilters, const char *rect, mlt_image_format topFormat) {
            Output output;
            mlt_tractor tractor = mlt_tractor_new();
            mlt_multitrack multitrack = mlt_tractor_multitrack(tractor);
            mlt_field field = mlt_tractor_field(tractor);
            mlt_producer producers[2] = {patternProducer(profile, &bottom), patternProducer(profile, &top, topFormat)};
            // Both clips get the converter, as MLT's loader attaches it
            mlt_filter converters[2] = {mlt_factory_filter(profile, "avcolor_space", nullptr), mlt_factory_filter(profile, "avcolor_space", nullptr)};
            std::vector<mlt_filter> filters = createFilters(profile, topFilters, cpu);
            mlt_producer_attach(producers[0], converters[0]);
            mlt_producer_attach(producers[1], converters[1]);
            for (mlt_filter filter : filters) {
                mlt_producer_attach(producers[1], filter);
            }
            mlt_multitrack_connect(multitrack, producers[0], 0);
            mlt_multitrack_connect(multitrack, producers[1], 1);
            mlt_transition transition = mlt_factory_transition(profile, "qtblend", nullptr);
            mlt_properties_set_int(MLT_TRANSITION_PROPERTIES(transition), "always_active", 1);
            if (rect) {
                mlt_properties_set(MLT_TRANSITION_PROPERTIES(transition), "rect", rect);
            }
            mlt_properties_set_int(MLT_TRANSITION_PROPERTIES(transition), "_gpufx.disable", cpu ? 1 : 0);
            mlt_field_plant_transition(field, transition, 0, 1);
            mlt_frame frame = nullptr;
            mlt_service_get_frame(MLT_TRACTOR_SERVICE(tractor), &frame, 0);
            if (frame) {
                output = fetchOutput(frame, mlt_image_yuv422, profile->width, profile->height);
                mlt_frame_close(frame);
            }
            mlt_transition_close(transition);
            mlt_tractor_close(tractor);
            for (mlt_filter filter : filters) {
                mlt_filter_close(filter);
            }
            mlt_filter_close(converters[0]);
            mlt_filter_close(converters[1]);
            mlt_producer_close(producers[0]);
            mlt_producer_close(producers[1]);
            return output;
        };
        // Moved and faded, without scaling, which MLT 7.38 does with QImage::scaled() on the CPU
        const char *const moved = "12 -7 160 90 0.6";
        g_mock.reset();
        const Output gpu = renderYuv(false, {}, moved, mlt_image_rgba);
        const MockCudaStats stats = g_mock.stats();
        const Output cpu = renderYuv(true, {}, moved, mlt_image_rgba);
        expectSameOutput(cpu, gpu, "yuv422 output of compositing", 1);
        if (g_mock.active()) {
            const uint64_t pixels = uint64_t(profile->width) * uint64_t(profile->height);
            check(stats.downloadedBytes == pixels * 3, "yuv422 output of compositing: %llu bytes downloaded", (unsigned long long)stats.downloadedBytes);
        }
        // Effects on a video clip on the top track, composited without a
        // rectangle: the transition first asks for the top image in YUV,
        // which the effects must deliver for real rather than leave on the GPU
        struct TopCase
        {
            const char *what;
            std::vector<FilterSpec> filters;
        };
        const TopCase topCases[] = {
            {"Transform", {{"qtblend", {{"rect", "20 10 100 56 1"}}}}},
            {"Transform and drop shadow", {{"qtblend", {{"rect", "20 10 100 56 1"}, {"rotation", "8"}}}, {"dropshadow", {{"radius", "3"}}}}},
            {"opaque blur", {{"box_blur", {{"hradius", "3"}, {"vradius", "2"}}}}},
        };
        for (const TopCase &c : topCases) {
            if (!hasFilters(c.filters)) {
                continue;
            }
            const std::string what = std::string("yuv422 output of compositing, ") + c.what + " on the top clip";
            g_mock.reset();
            const Output gpuTop = renderYuv(false, c.filters, nullptr, mlt_image_yuv420p);
            const uint64_t launches = g_mock.stats().launches;
            const uint64_t downloaded = g_mock.stats().downloadedBytes;
            expectSameOutput(renderYuv(true, c.filters, nullptr, mlt_image_yuv420p), gpuTop, what, 1);
            if (g_mock.active()) {
                check(launches > 0, "%s: nothing ran on the GPU", what.c_str());
                // The top image goes to YUV and back to RGBA on the GPU
                const uint64_t pixels = uint64_t(profile->width) * uint64_t(profile->height);
                check(downloaded == pixels * 3, "%s: %llu bytes downloaded, expected only the YUV output", what.c_str(), (unsigned long long)downloaded);
            }
        }
    }
}

void testGlobalSwitch(mlt_profile profile)
{
    if (!g_mock.active() || !hasFilter("dropshadow")) {
        return;
    }
    mlt_properties_set(mlt_global_properties(), "gpufx.enabled", "0");
    g_mock.reset();
    expectSame(profile, {{"dropshadow", {{"radius", "5"}}}}, makeImage(40, 30, 0), false);
    check(g_mock.stats().launches == 0, "gpufx.enabled=0 did not turn GPU processing off");
    mlt_properties_set(mlt_global_properties(), "gpufx.enabled", "1");
    g_mock.reset();
    render(profile, {{"dropshadow", {{"radius", "5"}}}}, makeImage(40, 30, 0), false);
    check(g_mock.stats().launches > 0, "gpufx.enabled=1 did not turn GPU processing back on");
    // Back to the settings file
    mlt_properties_set(mlt_global_properties(), "gpufx.enabled", "");
}

struct ModuleStatus
{
    int enabled = -1;
    int hooked = -1;
    std::string effects;
    std::string transitions;
    std::string conversions;
};

ModuleStatus moduleStatus(mlt_profile profile)
{
    ModuleStatus result;
    if (mlt_filter status = mlt_factory_filter(profile, "gpufx", "noinit")) {
        mlt_properties properties = MLT_FILTER_PROPERTIES(status);
        result.enabled = mlt_properties_get_int(properties, "enabled");
        result.hooked = mlt_properties_get_int(properties, "hooked");
        result.effects = std::string(",") + mlt_properties_get(properties, "effects") + ",";
        result.transitions = std::string(",") + mlt_properties_get(properties, "transitions") + ",";
        result.conversions = std::string(",") + mlt_properties_get(properties, "conversions") + ",";
        mlt_filter_close(status);
    }
    return result;
}

/** The settings the settings page saves in Kdenlive's settings file */
void testSettingsFile(mlt_profile profile, bool expectGpu)
{
    using GetHost = const gpufx_settings_host *(*)();
    auto getHost = g_module ? reinterpret_cast<GetHost>(findSymbol(g_module, "gpufx_get_settings_host")) : nullptr;
    check(getHost != nullptr, "the module does not export gpufx_get_settings_host()");
    const gpufx_settings_host *host = getHost ? getHost() : nullptr;
    if (!host) {
        return;
    }
    check(host->version == GPUFX_SETTINGS_API_VERSION && !strcmp(host->group, "gpufx"), "settings host: version %d, group %s", host->version, host->group);
    const std::string services = std::string(",") + host->services + ",";
    for (const char *service : {",effect:dropshadow,", ",effect:qtblend,", ",transition:qtblend,", ",transition:luma,", ",conversion:avformat,",
                                ",conversion:avcolor_space,", ",conversion:normalize,", ",conversion:stills,"}) {
        check(services.find(service) != std::string::npos, "settings host: %s is not in the services %s", service, host->services);
    }
    char text[1024];
    host->describe_gpu(text, sizeof(text));
    printf("describe_gpu: %s", text);
    check(strstr(text, expectGpu ? "available=1\n" : "available=0\n") && strstr(text, "\nstatus=") && strstr(text, "\nmemory_mb="),
          "describe_gpu: unexpected description");

    ModuleStatus status = moduleStatus(profile);
    check(status.enabled == 1 && status.hooked == 1, "settings: the module starts with enabled=%d hooked=%d", status.enabled, status.hooked);

    writeText(g_settingsFile, "[gpufx]\nenabled=false\n");
    host->reload_settings();
    status = moduleStatus(profile);
    check(status.enabled == 0, "settings: enabled=false did not turn GPU processing off");
    if (g_mock.active() && hasFilter("dropshadow")) {
        g_mock.reset();
        expectSame(profile, {{"dropshadow", {{"radius", "5"}}}}, makeImage(40, 30, 0), false);
        check(g_mock.stats().launches == 0, "settings: the GPU was used with enabled=false");
    }

    // KConfig's format: other groups, nested groups, options, CRLF line ends
    writeText(g_settingsFile, "# comment\n[other]\nenabled=false\neffect:frei0r.glow=false\n[gpufx][nested]\nenabled=false\n"
                              "[gpufx]\r\nenabled = true\r\neffect:dropshadow=false\ntransition:qtblend=off\neffect:box_blur[$i]=false\n"
                              "effect:boxblur[de]=false\neffect:brightness[$d]=false\nconversion:avformat=0\neffect:frei0r.sharpness=true\n");
    host->reload_settings();
    status = moduleStatus(profile);
    check(status.enabled == 1, "settings: enabled = true was not read");
    check(status.effects.find(",dropshadow,") == std::string::npos && status.effects.find(",box_blur,") == std::string::npos,
          "settings: dropshadow and box_blur are still on the GPU: %s", status.effects.c_str());
    check(status.effects.find(",qtblend,") != std::string::npos && status.effects.find(",boxblur,") != std::string::npos
              && status.effects.find(",brightness,") != std::string::npos && status.effects.find(",frei0r.glow,") != std::string::npos
              && status.effects.find(",frei0r.sharpness,") != std::string::npos,
          "settings: effects missing from the GPU: %s", status.effects.c_str());
    check(status.transitions == ",luma,", "settings: transition:qtblend=off was not read: %s", status.transitions.c_str());
    check(status.conversions == ",avcolor_space,normalize,stills,", "settings: conversion:avformat=0 was not read: %s", status.conversions.c_str());
    if (g_mock.active() && hasFilter("dropshadow") && hasFilter("boxblur")) {
        g_mock.reset();
        expectSame(profile, {{"dropshadow", {{"radius", "5"}}}}, makeImage(40, 30, 0), false);
        check(g_mock.stats().launches == 0, "settings: dropshadow ran on the GPU");
        g_mock.reset();
        render(profile, {{"boxblur", {{"blur", "3"}}}}, makeImage(40, 30, 0), false);
        check(g_mock.stats().launches > 0, "settings: boxblur did not run on the GPU");
    }

    // No file: the defaults
    remove(g_settingsFile.c_str());
    host->reload_settings();
    status = moduleStatus(profile);
    check(status.enabled == 1 && status.effects.find(",dropshadow,") != std::string::npos && status.transitions == ",qtblend,luma,"
              && status.conversions == ",avformat,avcolor_space,normalize,stills,",
          "settings: the defaults did not come back without a file");

#if !defined(_WIN32)
    // Without GPUFX_CONFIG_FILE: Kdenlive's file in XDG_CONFIG_HOME, named
    // after the package Kdenlive's PACKAGE_TYPE names, else kdenlive-appimagerc
    // when the module is in an extracted AppImage. Each file keeps one effect
    // on the CPU, which tells which one was read.
    const std::string config = g_settingsFile + ".d";
    std::filesystem::create_directories(config);
    writeText(config + "/kdenliverc", "[gpufx]\neffect:dropshadow=false\n");
    writeText(config + "/kdenlive-appimagerc", "[gpufx]\neffect:box_blur=false\n");
    writeText(config + "/kdenlive-flatpakrc", "[gpufx]\neffect:boxblur=false\n");
    const char *xdg = getenv("XDG_CONFIG_HOME");
    const std::string savedXdg = xdg ? xdg : "";
    const char *packageType = getenv("PACKAGE_TYPE");
    const std::string savedPackageType = packageType ? packageType : "";
    unsetenv("GPUFX_CONFIG_FILE");
    setenv("XDG_CONFIG_HOME", config.c_str(), 1);
    auto offEffect = [&]() {
        const std::string effects = moduleStatus(profile).effects;
        for (const char *effect : {"dropshadow", "box_blur", "boxblur"}) {
            if (effects.find(std::string(",") + effect + ",") == std::string::npos) {
                return std::string(effect);
            }
        }
        return std::string("none");
    };
    setenv("PACKAGE_TYPE", "AppImage", 1);
    host->reload_settings();
    std::string off = offEffect();
    check(off == "box_blur", "settings: kdenlive-appimagerc not read with PACKAGE_TYPE=AppImage (%s off)", off.c_str());
    setenv("PACKAGE_TYPE", "flatpak", 1);
    host->reload_settings();
    off = offEffect();
    check(off == "boxblur", "settings: kdenlive-flatpakrc not read with PACKAGE_TYPE=flatpak (%s off)", off.c_str());
    unsetenv("PACKAGE_TYPE");
    Dl_info info{};
    const bool inAppImage = dladdr(reinterpret_cast<void *>(getHost), &info) && info.dli_fname
                            && std::filesystem::exists(std::filesystem::path(info.dli_fname).parent_path() / "../../../AppRun");
    host->reload_settings();
    off = offEffect();
    check(off == (inAppImage ? "box_blur" : "dropshadow"), "settings: %s not read without PACKAGE_TYPE (%s off)",
          inAppImage ? "kdenlive-appimagerc" : "kdenliverc", off.c_str());

    std::filesystem::remove_all(config);
    if (packageType) {
        setenv("PACKAGE_TYPE", savedPackageType.c_str(), 1);
    }
    if (xdg) {
        setenv("XDG_CONFIG_HOME", savedXdg.c_str(), 1);
    } else {
        unsetenv("XDG_CONFIG_HOME");
    }
    setenv("GPUFX_CONFIG_FILE", g_settingsFile.c_str(), 1);
    host->reload_settings();
#endif
}

/** GPU processing turned off in the settings when the module was loaded */
void testSettingsOff(mlt_profile profile)
{
    const ModuleStatus status = moduleStatus(profile);
    check(status.enabled == 0 && status.hooked == 0, "settings off: enabled=%d hooked=%d", status.enabled, status.hooked);
    if (g_mock.active() && hasFilter("dropshadow")) {
        g_mock.reset();
        expectSame(profile, {{"dropshadow", {{"radius", "5"}}}}, makeImage(40, 30, 0), false);
        check(g_mock.stats().launches == 0, "settings off: the GPU was used");
    }
}

/** Must run last: a launch failure turns the GPU off for the session. */
void testFailures(mlt_profile profile)
{
    if (!g_mock.active() || !hasFilter("dropshadow")) {
        return;
    }
    const Image input = makeImage(64, 48, 2);
    const std::vector<FilterSpec> chain = {{"box_blur", {{"hradius", "15"}, {"vradius", "15"}}}, {"dropshadow", {{"radius", "6"}}}};

    // Out of memory is recoverable: the frame falls back to the CPU
    g_mock.failAfter(MOCKCUDA_ALLOC, 1, 2);
    expectSame(profile, chain, input, false);
    g_mock.reset();
    render(profile, chain, input, false);
    check(g_mock.stats().launches > 0, "the GPU was not used anymore after an out of memory error");

    // A failing kernel in the middle of the chain: the output is still right
    g_mock.failAfter(MOCKCUDA_LAUNCH, 2, 719);
    expectSame(profile, chain, input, false);
    g_mock.reset();
    expectSame(profile, chain, input, false);
    check(g_mock.stats().launches == 0, "the GPU was still used after a launch failure");
    mlt_filter status = mlt_factory_filter(profile, "gpufx", "noinit");
    check(status && mlt_properties_get_int(MLT_FILTER_PROPERTIES(status), "available") == 0, "status still reports the GPU after a failure");
    if (status) {
        printf("after failure: %s\n", mlt_properties_get(MLT_FILTER_PROPERTIES(status), "status"));
        mlt_filter_close(status);
    }
}

} // namespace

int main(int argc, char **argv)
{
    bool expectGpu = true;
    bool settingsOff = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--no-gpu")) {
            expectGpu = false;
        } else if (!strcmp(argv[i], "--settings-off")) {
            settingsOff = true;
        }
    }
#if !defined(_WIN32)
    if (!getenv("DISPLAY") && !getenv("WAYLAND_DISPLAY")) {
        setenv("QT_QPA_PLATFORM", "offscreen", 0);
    }
    setenv("GPUFX_CUDA_LIBRARY", GPUFX_TEST_CUDA_LIBRARY, 0);
#else
    if (!getenv("GPUFX_CUDA_LIBRARY")) {
        _putenv_s("GPUFX_CUDA_LIBRARY", GPUFX_TEST_CUDA_LIBRARY);
    }
#endif

    // A settings file of the tests' own instead of Kdenlive's
    g_settingsFile = (std::filesystem::temp_directory_path() / ("gpufx_test_" + std::to_string(getpid()) + "rc")).string();
    setEnvironment("GPUFX_CONFIG_FILE", g_settingsFile.c_str());
    if (settingsOff) {
        writeText(g_settingsFile, "[gpufx]\nenabled=false\n");
    } else {
        remove(g_settingsFile.c_str());
    }

    // MLT on Windows looks for its modules next to the executable and ignores
    // MLT_REPOSITORY unless it is passed here
    mlt_repository repository = mlt_factory_init(getenv("MLT_REPOSITORY"));
    if (!repository) {
        fprintf(stderr, "cannot initialize MLT\n");
        return 2;
    }
    if (getenv("GPUFX_TEST_VERBOSE")) {
        mlt_log_set_level(MLT_LOG_DEBUG);
    }
    // Register the module from the build tree, unless MLT already loaded it
    if (!mlt_properties_get_data(mlt_repository_filters(repository), "gpufx", nullptr)) {
        g_module = openLibrary(GPUFX_TEST_MODULE);
        auto registerModule = g_module ? reinterpret_cast<void (*)(mlt_repository)>(findSymbol(g_module, "mlt_register")) : nullptr;
        if (!registerModule) {
            fprintf(stderr, "cannot load %s\n", GPUFX_TEST_MODULE);
            return 2;
        }
        registerModule(repository);
    } else {
        g_module = findLoadedModule(repository);
    }
    if (void *driver = openLibrary(getenv("GPUFX_CUDA_LIBRARY"))) {
        g_mock.getStats = reinterpret_cast<MockCudaGetStats>(findSymbol(driver, "mockcuda_get_stats"));
        g_mock.resetStats = reinterpret_cast<MockCudaResetStats>(findSymbol(driver, "mockcuda_reset_stats"));
        g_mock.failAfter = reinterpret_cast<MockCudaFailAfter>(findSymbol(driver, "mockcuda_fail_after"));
    }
    if (!expectGpu) {
        g_mock = Mock();
    }

    mlt_profile profile = mlt_profile_init(nullptr);
    profile->width = 160;
    profile->height = 90;
    profile->sample_aspect_num = 1;
    profile->sample_aspect_den = 1;
    profile->display_aspect_num = 16;
    profile->display_aspect_den = 9;

    if (settingsOff) {
        testSettingsOff(profile);
        printf("%d checks, %d failures\n", g_checks, g_failures);
        remove(g_settingsFile.c_str());
        mlt_profile_close(profile);
        mlt_factory_close();
        return g_failures ? 1 : 0;
    }

    testStatus(profile, expectGpu);
    testDropShadow(profile);
    testBoxBlur(profile);
    testRectBlur(profile);
    testColor(profile);
    testFrei0r(profile);
    testTransform(profile);
    testCompositing(profile);
    testLuma(profile);
    testChains(profile);
    testProducer(profile);
    testDecoding(profile);
    testNormalizers(profile);
    testStills(profile);
    testYuvOutput(profile);
    testGlobalSwitch(profile);
    testSettingsFile(profile, expectGpu);
    testFailures(profile);

    if (g_mock.active()) {
        const MockCudaStats stats = g_mock.stats();
        check(stats.misuse == 0, "%llu invalid CUDA calls", (unsigned long long)stats.misuse);
    }
    printf("%d checks, %d failures\n", g_checks, g_failures);
    mlt_profile_close(profile);
    mlt_factory_close();
    return g_failures ? 1 : 0;
}
