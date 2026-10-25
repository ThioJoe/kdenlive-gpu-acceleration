/*
    SPDX-FileCopyrightText: 2026 ThioJoe
    SPDX-License-Identifier: GPL-3.0-only
*/

#pragma once

#include "settings_page_api.h"

#include <string>
#include <vector>

namespace gpufx {

/** The settings saved by the settings page (see settings_page_api.h) */
struct FileSettings
{
    bool enabled = true;
    /** Services kept on the CPU, as <kind>:<service> */
    std::vector<std::string> disabled;
};

/** Name of the group of the settings file */
extern const char *const kSettingsGroup;

/** The settings as last read by reloadFileSettings(). Safe to call from any
 *  thread. */
const FileSettings &fileSettings();

/** Reads the settings file: GPUFX_CONFIG_FILE, or else Kdenlive's settings
 *  file, kdenliverc in the user's configuration folder (%LOCALAPPDATA% on
 *  Windows, $XDG_CONFIG_HOME or ~/.config elsewhere), which Kdenlive's
 *  AppImage names kdenlive-appimagerc. A missing file or group means the
 *  defaults. */
void reloadFileSettings();

/** True when the settings page is to be loaded in this process: in
 *  Kdenlive's main program, unless GPUFX_SETTINGS_PAGE=0, or anywhere with
 *  GPUFX_SETTINGS_PAGE=1. */
bool settingsPageWanted();

/** Loads the settings page from the gpufx folder next to the module and
 *  hands it the host interface. */
void loadSettingsPage(const gpufx_settings_host &host);

} // namespace gpufx
