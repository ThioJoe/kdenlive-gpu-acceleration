/*
    SPDX-FileCopyrightText: 2026 ThioJoe
    SPDX-License-Identifier: GPL-3.0-only
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
 * the packed 4:2:2 YUV that consumers ask for (MLT's avcolour_space). And it
 * lets images on the GPU through the normalizing filters MLT puts after the
 * producers (see normalizer.h), and keeps the pictures of still images,
 * titles and color clips there (see stills.h).
 *
 * Runtime switches:
 *  - environment MLT_GPUFX=0: do not touch any filter
 *  - the [gpufx] group of Kdenlive's settings file, saved by the settings page
 *    (see settings_page_api.h): GPU processing off, or some services kept on
 *    the CPU. Read when the module is loaded, and again in Kdenlive when the
 *    settings are applied.
 *  - MLT global property "gpufx.enabled" (0/1): turns GPU processing off/on
 *    at runtime, overrides the settings file
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
#include "gpu_runtime.h"
#include "gpufx_log.h"
#include "settings_file.h"

#include <framework/mlt.h>

#include <algorithm>
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

/** Whether the module listens to MLT's factory to wrap services */
bool g_hooked = false;

bool isFalse(const char *value)
{
    return value && (!strcmp(value, "0") || !strcmp(value, "false") || !strcmp(value, "off") || !strcmp(value, "no"));
}

struct Settings
{
    bool wrapFilters = true;
    bool debug = false;
    std::vector<std::string> disabledEffects;
};

const Settings &settings()
{
    static const Settings s = [] {
        Settings result;
        const char *env = getenv("MLT_GPUFX");
        result.wrapFilters = !isFalse(env);
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
    } else if (const NormalizerDescriptor *normalizer = filter ? findNormalizer(data->name) : nullptr) {
        wrapNormalizer(filter, *normalizer);
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
    } else if (const StillDescriptor *still = producer ? findStill(data->name) : nullptr) {
        wrapStill(producer, *still);
    }
}

std::string conversionList()
{
    std::string list;
    for (const char *service : {"avformat", "avcolor_space", "normalize", "stills"}) {
        if (!serviceEnabled("conversion", service)) {
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
        if (!serviceEnabled("transition", transition->service)) {
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

/** Everything the module can run on the GPU, as <kind>:<service> */
std::string serviceList()
{
    std::string list;
    for (const EffectDescriptor *effect : effectRegistry()) {
        list += std::string(list.empty() ? "" : ",") + "effect:" + effect->service;
    }
    for (const TransitionDescriptor *transition : transitionRegistry()) {
        list += std::string(",transition:") + transition->service;
    }
    return list + ",conversion:avformat,conversion:avcolor_space,conversion:normalize,conversion:stills";
}

/** One line of text for the description of the GPU */
std::string oneLine(std::string text)
{
    std::replace(text.begin(), text.end(), '\n', ' ');
    std::replace(text.begin(), text.end(), '\r', ' ');
    return text;
}

void describeGpu(char *text, size_t size)
{
    if (!text || !size) {
        return;
    }
    Runtime &runtime = Runtime::instance();
    const bool available = runtime.ensureInitialized();
    const DeviceInfo device = runtime.device();
    char line[64];
    std::string description = std::string("available=") + (available ? "1" : "0") + "\nstatus=" + oneLine(runtime.status()) + "\ndevice=" + oneLine(device.name);
    snprintf(line, sizeof(line), "\ncompute_capability=%d.%d", device.computeMajor, device.computeMinor);
    description += device.ordinal >= 0 ? line : "\ncompute_capability=";
    snprintf(line, sizeof(line), "\ncuda_driver=%d.%d", device.driverVersion / 1000, (device.driverVersion % 1000) / 10);
    description += device.driverVersion > 0 ? line : "\ncuda_driver=";
    snprintf(line, sizeof(line), "\nmemory_mb=%d\n", int(device.totalMemory >> 20));
    description += line;
    snprintf(text, size, "%s", description.c_str());
}

void reloadSettings()
{
    reloadFileSettings();
}

/** What the module offers to its settings page */
const gpufx_settings_host &settingsHost()
{
    static const std::string services = serviceList();
    static const gpufx_settings_host host = {GPUFX_SETTINGS_API_VERSION, kSettingsGroup, services.c_str(), describeGpu, reloadSettings};
    return host;
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
    mlt_properties_set_int(properties, "hooked", g_hooked);
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
    return fileSettings().enabled;
}

bool serviceEnabled(const char *kind, const char *service)
{
    for (const std::string &name : settings().disabledEffects) {
        if (name == service) {
            return false;
        }
    }
    const size_t kindLength = strlen(kind);
    for (const std::string &name : fileSettings().disabled) {
        if (name.size() > kindLength && !name.compare(0, kindLength, kind) && name[kindLength] == ':' && name.compare(kindLength + 1, std::string::npos, service) == 0) {
            return false;
        }
    }
    return true;
}

bool effectEnabled(const EffectDescriptor &effect)
{
    return serviceEnabled("effect", effect.service);
}

const std::vector<const EffectDescriptor *> &effectRegistry()
{
    static const std::vector<const EffectDescriptor *> effects = {
        &dropShadowEffect(),       &boxBlurEffect(),    &rectBlurEffect(),       &brightnessEffect(),      &liftGammaGainEffect(),
        &frei0rSquareBlurEffect(), &frei0rGlowEffect(), &frei0rSoftGlowEffect(), &frei0rSharpnessEffect(), &transformEffect(),
        &frei0rScale0TiltEffect()};
    return effects;
}

const std::vector<const TransitionDescriptor *> &transitionRegistry()
{
    static const std::vector<const TransitionDescriptor *> transitions = {&compositeTransition(), &lumaTransition()};
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
    reloadFileSettings();
    const bool page = settingsPageWanted();
    if (!fileSettings().enabled && !page) {
        // Renders and other processes leave MLT alone. Kdenlive keeps the
        // filters wrapped, so that the setting can be turned back on.
        logInfo("turned off in the settings");
        return;
    }
    pinModule();
    mlt_events_listen(mlt_factory_event_object(), nullptr, "filter-create-done", onFilterCreated);
    mlt_events_listen(mlt_factory_event_object(), nullptr, "transition-create-done", onTransitionCreated);
    mlt_events_listen(mlt_factory_event_object(), nullptr, "producer-create-done", onProducerCreated);
    g_hooked = true;
    if (page) {
        loadSettingsPage(settingsHost());
    }
}

/** The interface the settings page gets, for the tests */
#if defined(_WIN32)
__declspec(dllexport)
#else
__attribute__((visibility("default")))
#endif
const gpufx_settings_host *gpufx_get_settings_host()
{
    return &gpufx::settingsHost();
}
}
