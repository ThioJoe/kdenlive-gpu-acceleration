/*
    SPDX-FileCopyrightText: 2026 ThioJoe
    SPDX-License-Identifier: GPL-3.0-only
*/

/*
 * Interface between the gpufx MLT module and its page in Kdenlive's settings
 * dialog (settings/). The page is a separate library that needs Qt and KDE
 * Frameworks: the module loads it from the gpufx folder next to itself, only
 * in Kdenlive's main program, and calls its gpufx_settings_init().
 *
 * The page saves the settings in the [gpufx] group of Kdenlive's settings
 * file, which the module reads in every process when it starts:
 *   enabled=false            GPU processing off
 *   <kind>:<service>=false   keep that service on the CPU, kind being effect,
 *                            transition or conversion (see services below)
 */

#pragma once

#include <stddef.h>

#define GPUFX_SETTINGS_API_VERSION 1

#ifdef __cplusplus
extern "C" {
#endif

struct gpufx_settings_host
{
    /** GPUFX_SETTINGS_API_VERSION */
    int version;
    /** Group of the settings file the module reads */
    const char *group;
    /** What the module can run on the GPU, separated by commas, each as
     *  <kind>:<service>: effect:<MLT filter>, transition:<MLT transition>,
     *  conversion:avformat (decoded video to RGBA) and conversion:avcolor_space
     *  (RGBA to YUV for output). These are also the keys of the settings. */
    const char *services;
    /** Describes the GPU in text (NUL terminated, at most size bytes) as lines
     *  of key=value: available (0 or 1), status, device, compute_capability,
     *  cuda_driver and memory_mb. Initializes CUDA the first time, which can
     *  take a moment: not to be called from the GUI thread. */
    void (*describe_gpu)(char *text, size_t size);
    /** Reads the settings file again: after the page saved it, or changed the
     *  GPUFX_CONFIG_FILE environment variable that names it. */
    void (*reload_settings)(void);
};

/** Exported by the page as gpufx_settings_init. Returns 0 if it cannot run. */
typedef int (*gpufx_settings_init_function)(const struct gpufx_settings_host *host);

#ifdef __cplusplus
}
#endif
