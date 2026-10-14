/*
    SPDX-FileCopyrightText: 2026 Kdenlive contributors
    SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

/*
 * gpufx: GPU (CUDA) acceleration of existing MLT filters.
 *
 * The module does not add new effects. It listens to the MLT factory and,
 * when a filter it knows is created (e.g. "dropshadow"), routes that filter's
 * frames through the GPU when possible. Projects keep using the regular
 * service names and parameters, and render the same with or without the
 * module, with or without an NVIDIA GPU.
 *
 * It also takes over two image conversions MLT does with libswscale: the
 * decoded 4:2:0 YUV video of avformat producers to RGBA, and RGBA results to
 * the packed 4:2:2 YUV that consumers ask for (MLT's avcolour_space).
 *
 * Runtime switches:
 *  - environment MLT_GPUFX=0: do not touch any filter
 *  - MLT global property "gpufx.enabled" (0/1): turns GPU processing off/on
 *    at runtime, overrides MLT_GPUFX
 *  - MLT_GPUFX_DISABLE=dropshadow,box_blur: keep these effects on the CPU
 *    (avformat and avcolor_space stand for the two conversions)
 *  - MLT_GPUFX_DEBUG=1: log every frame processed on the GPU
 *  - GPUFX_CUDA_DEVICE=n: use the n-th CUDA device
 *
 * The "gpufx" filter service reports the GPU status in its properties when
 * created (available, status, device, effects, ...), it does not process
 * frames.
 */

#include "bridge.h"
#include "gpufx_log.h"

#include <framework/mlt.h>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace gpufx {

namespace {

const char *const kGlobalEnabled = "gpufx.enabled";

bool isFalse(const char *value)
{
    return value && (!strcmp(value, "0") || !strcmp(value, "false") || !strcmp(value, "off") || !strcmp(value, "no"));
}

struct Settings
{
    bool wrapFilters = true;
    bool defaultEnabled = true;
    bool debug = false;
    std::vector<std::string> disabledEffects;
};

const Settings &settings()
{
    static const Settings s = [] {
        Settings result;
        const char *env = getenv("MLT_GPUFX");
        result.wrapFilters = !isFalse(env);
        result.defaultEnabled = result.wrapFilters;
        const char *debug = getenv("MLT_GPUFX_DEBUG");
        result.debug = debug && debug[0] && !isFalse(debug);
        if (const char *disabled = getenv("MLT_GPUFX_DISABLE")) {
            std::string list(disabled);
            size_t start = 0;
            while (start <= list.size()) {
                size_t end = list.find_first_of(",; ", start);
                if (end == std::string::npos) {
                    end = list.size();
                }
                if (end > start) {
                    result.disabledEffects.push_back(list.substr(start, end - start));
                }
                start = end + 1;
            }
        }
        return result;
    }();
    return s;
}

void vlog(int level, const char *format, va_list args)
{
    char message[1024];
    vsnprintf(message, sizeof(message), format, args);
    mlt_log(nullptr, level, "[gpufx] %s\n", message);
}

void onFilterCreated(mlt_properties, void *, mlt_event_data eventData)
{
    auto *data = static_cast<mlt_factory_event_data *>(mlt_event_data_to_object(eventData));
    if (!data || !data->name || !data->service) {
        return;
    }
    mlt_filter filter = *static_cast<mlt_filter *>(data->service);
    const EffectDescriptor *effect = filter ? findEffect(data->name) : nullptr;
    if (effect) {
        wrapFilter(filter, *effect);
    }
}

void onTransitionCreated(mlt_properties, void *, mlt_event_data eventData)
{
    auto *data = static_cast<mlt_factory_event_data *>(mlt_event_data_to_object(eventData));
    if (!data || !data->name || !data->service) {
        return;
    }
    mlt_transition transition = *static_cast<mlt_transition *>(data->service);
    const TransitionDescriptor *descriptor = transition ? findTransition(data->name) : nullptr;
    if (descriptor) {
        wrapTransition(transition, *descriptor);
    }
}

void onProducerCreated(mlt_properties, void *, mlt_event_data eventData)
{
    auto *data = static_cast<mlt_factory_event_data *>(mlt_event_data_to_object(eventData));
    if (!data || !data->name || !data->service) {
        return;
    }
    mlt_producer producer = *static_cast<mlt_producer *>(data->service);
    if (producer && (!strcmp(data->name, "avformat") || !strcmp(data->name, "avformat-novalidate"))) {
        wrapProducer(producer);
    }
}

std::string conversionList()
{
    std::string list;
    for (const char *service : {"avformat", "avcolor_space"}) {
        if (!serviceEnabled(service)) {
            continue;
        }
        if (!list.empty()) {
            list += ',';
        }
        list += service;
    }
    return list;
}

std::string transitionList()
{
    std::string list;
    for (const TransitionDescriptor *transition : transitionRegistry()) {
        if (!serviceEnabled(transition->service)) {
            continue;
        }
        if (!list.empty()) {
            list += ',';
        }
        list += transition->service;
    }
    return list;
}

std::string effectList()
{
    std::string list;
    for (const EffectDescriptor *effect : effectRegistry()) {
        if (!effectEnabled(*effect)) {
            continue;
        }
        if (!list.empty()) {
            list += ',';
        }
        list += effect->service;
    }
    return list;
}

/** "gpufx" service: a filter that does nothing but report the GPU status.
 *  Creating it initializes CUDA unless the argument is "noinit". */
void *statusServiceInit(mlt_profile, mlt_service_type, const char *, const void *arg)
{
    mlt_filter filter = mlt_filter_new();
    if (!filter) {
        return nullptr;
    }
    filter->process = nullptr;
    mlt_properties properties = MLT_FILTER_PROPERTIES(filter);
    Runtime &runtime = Runtime::instance();
    const bool probe = !(arg && !strcmp(static_cast<const char *>(arg), "noinit"));
    const bool available = probe ? runtime.ensureInitialized() : runtime.usable();
    const DeviceInfo device = runtime.device();
    mlt_properties_set(properties, "api", "CUDA");
    mlt_properties_set_int(properties, "available", available);
    mlt_properties_set_int(properties, "enabled", moduleEnabled());
    mlt_properties_set_int(properties, "hooked", settings().wrapFilters);
    mlt_properties_set(properties, "status", runtime.status().c_str());
    mlt_properties_set(properties, "device", device.name.c_str());
    char text[64];
    snprintf(text, sizeof(text), "%d.%d", device.computeMajor, device.computeMinor);
    mlt_properties_set(properties, "compute_capability", device.ordinal >= 0 ? text : "");
    snprintf(text, sizeof(text), "%d.%d", device.driverVersion / 1000, (device.driverVersion % 1000) / 10);
    mlt_properties_set(properties, "cuda_driver", device.driverVersion > 0 ? text : "");
    mlt_properties_set_int(properties, "memory_mb", int(device.totalMemory >> 20));
    mlt_properties_set(properties, "effects", effectList().c_str());
    mlt_properties_set(properties, "transitions", transitionList().c_str());
    mlt_properties_set(properties, "conversions", conversionList().c_str());
    mlt_properties_set_int64(properties, "gpu_frames", int64_t(runtime.gpuFrames()));
    mlt_properties_set_int64(properties, "cpu_frames", int64_t(runtime.cpuFrames()));
    return filter;
}

/** The module must stay loaded while filters it wrapped exist. */
void pinModule()
{
#if defined(_WIN32)
    HMODULE module = nullptr;
    GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN, reinterpret_cast<LPCSTR>(&pinModule), &module);
#else
    Dl_info info;
    if (dladdr(reinterpret_cast<void *>(&pinModule), &info) && info.dli_fname) {
        dlopen(info.dli_fname, RTLD_NOW | RTLD_NOLOAD | RTLD_NODELETE);
    }
#endif
}

} // namespace

void logError(const char *format, ...)
{
    va_list args;
    va_start(args, format);
    vlog(MLT_LOG_ERROR, format, args);
    va_end(args);
}

void logWarning(const char *format, ...)
{
    va_list args;
    va_start(args, format);
    vlog(MLT_LOG_WARNING, format, args);
    va_end(args);
}

void logInfo(const char *format, ...)
{
    va_list args;
    va_start(args, format);
    vlog(MLT_LOG_INFO, format, args);
    va_end(args);
}

void logDebug(const char *format, ...)
{
    va_list args;
    va_start(args, format);
    vlog(settings().debug ? MLT_LOG_INFO : MLT_LOG_DEBUG, format, args);
    va_end(args);
}

bool debugEnabled()
{
    return settings().debug;
}

bool moduleEnabled()
{
    if (!settings().wrapFilters) {
        return false;
    }
    if (mlt_properties global = mlt_global_properties()) {
        const char *value = mlt_properties_get(global, kGlobalEnabled);
        if (value && value[0]) {
            return !isFalse(value);
        }
    }
    return settings().defaultEnabled;
}

bool serviceEnabled(const char *service)
{
    for (const std::string &name : settings().disabledEffects) {
        if (name == service) {
            return false;
        }
    }
    return true;
}

bool effectEnabled(const EffectDescriptor &effect)
{
    return serviceEnabled(effect.service);
}

const std::vector<const EffectDescriptor *> &effectRegistry()
{
    static const std::vector<const EffectDescriptor *> effects = {
        &dropShadowEffect(),       &boxBlurEffect(),    &rectBlurEffect(),       &brightnessEffect(),      &liftGammaGainEffect(),
        &frei0rSquareBlurEffect(), &frei0rGlowEffect(), &frei0rSoftGlowEffect(), &frei0rSharpnessEffect(), &transformEffect()};
    return effects;
}

const std::vector<const TransitionDescriptor *> &transitionRegistry()
{
    static const std::vector<const TransitionDescriptor *> transitions = {&compositeTransition()};
    return transitions;
}

const TransitionDescriptor *findTransition(const char *service)
{
    if (!service) {
        return nullptr;
    }
    for (const TransitionDescriptor *transition : transitionRegistry()) {
        if (!strcmp(transition->service, service)) {
            return transition;
        }
    }
    return nullptr;
}

const EffectDescriptor *findEffect(const char *service)
{
    if (!service) {
        return nullptr;
    }
    for (const EffectDescriptor *effect : effectRegistry()) {
        if (!strcmp(effect->service, service)) {
            return effect;
        }
    }
    return nullptr;
}

} // namespace gpufx

extern "C" {

#if defined(_WIN32)
__declspec(dllexport)
#else
__attribute__((visibility("default")))
#endif
void mlt_register(mlt_repository repository)
{
    using namespace gpufx;
    mlt_repository_register(repository, mlt_service_filter_type, "gpufx", statusServiceInit);
    if (!settings().wrapFilters) {
        logInfo("disabled by MLT_GPUFX");
        return;
    }
    pinModule();
    mlt_events_listen(mlt_factory_event_object(), nullptr, "filter-create-done", onFilterCreated);
    mlt_events_listen(mlt_factory_event_object(), nullptr, "transition-create-done", onTransitionCreated);
    mlt_events_listen(mlt_factory_event_object(), nullptr, "producer-create-done", onProducerCreated);
}
}
