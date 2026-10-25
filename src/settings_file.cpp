/*
    SPDX-FileCopyrightText: 2026 ThioJoe
    SPDX-License-Identifier: GPL-3.0-only
*/

#include "settings_file.h"

#include "gpufx_log.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#endif

namespace gpufx {

const char *const kSettingsGroup = "gpufx";

namespace {

#if defined(_WIN32)
using NativePath = std::wstring;
#else
using NativePath = std::string;
#endif

std::string displayPath(const NativePath &path)
{
#if defined(_WIN32)
    const int size = WideCharToMultiByte(CP_UTF8, 0, path.c_str(), int(path.size()), nullptr, 0, nullptr, nullptr);
    std::string result(size_t(std::max(size, 0)), '\0');
    if (size > 0) {
        WideCharToMultiByte(CP_UTF8, 0, path.c_str(), int(path.size()), result.data(), size, nullptr, nullptr);
    }
    return result;
#else
    return path;
#endif
}

NativePath moduleDirectory();

#if !defined(_WIN32)
/** The name of Kdenlive's settings file: kdenlive-<package>rc for its
 *  AppImage, Flatpak and Snap packages, which set PACKAGE_TYPE */
std::string kdenliveFileName()
{
    if (const char *type = getenv("PACKAGE_TYPE")) {
        std::string package(type);
        std::transform(package.begin(), package.end(), package.begin(), [](unsigned char c) { return char(tolower(c)); });
        if (package == "appimage" || package == "flatpak" || package == "snap") {
            return "kdenlive-" + package + "rc";
        }
    }
    // Programs started from an extracted AppImage without its AppRun, like
    // its melt, use the file of the AppImage's Kdenlive: the module is in
    // <AppImage>/usr/lib/mlt-7
    const std::string directory = moduleDirectory();
    if (!directory.empty() && access((directory + "/../../../AppRun").c_str(), F_OK) == 0) {
        return "kdenlive-appimagerc";
    }
    return "kdenliverc";
}
#endif

NativePath settingsFilePath()
{
#if defined(_WIN32)
    const wchar_t *file = _wgetenv(L"GPUFX_CONFIG_FILE");
    if (file && *file) {
        return file;
    }
    // Where Qt's GenericConfigLocation, and so KConfig, puts kdenliverc
    const wchar_t *local = _wgetenv(L"LOCALAPPDATA");
    if (local && *local) {
        return NativePath(local) + L"\\kdenliverc";
    }
#else
    const char *file = getenv("GPUFX_CONFIG_FILE");
    if (file && *file) {
        return file;
    }
    const char *config = getenv("XDG_CONFIG_HOME");
    if (config && config[0] == '/') {
        return std::string(config) + "/" + kdenliveFileName();
    }
    const char *home = getenv("HOME");
    if (home && *home) {
        return std::string(home) + "/.config/" + kdenliveFileName();
    }
#endif
    return {};
}

bool readFile(const NativePath &path, std::string &content)
{
#if defined(_WIN32)
    FILE *file = _wfopen(path.c_str(), L"rb");
#else
    FILE *file = fopen(path.c_str(), "rb");
#endif
    if (!file) {
        return false;
    }
    char buffer[4096];
    size_t length;
    while ((length = fread(buffer, 1, sizeof(buffer), file)) > 0) {
        content.append(buffer, length);
    }
    fclose(file);
    return true;
}

std::string trimmed(const std::string &text)
{
    const char *const spaces = " \t\r";
    const size_t first = text.find_first_not_of(spaces);
    if (first == std::string::npos) {
        return {};
    }
    return text.substr(first, text.find_last_not_of(spaces) - first + 1);
}

/** Booleans as KConfig reads them */
bool parseBool(std::string value, bool &result)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return char(tolower(c)); });
    if (value == "true" || value == "on" || value == "yes" || value == "1") {
        result = true;
        return true;
    }
    if (value == "false" || value == "off" || value == "no" || value == "0") {
        result = false;
        return true;
    }
    return false;
}

/** Reads the group of the settings in a file of KConfig's format (INI like:
 *  [group] headers, key=value entries, # comments). */
FileSettings parseSettings(const std::string &content)
{
    const std::string header = std::string("[") + kSettingsGroup + "]";
    bool inGroup = false;
    std::map<std::string, bool> services;
    FileSettings settings;
    size_t start = 0;
    while (start < content.size()) {
        size_t end = content.find('\n', start);
        if (end == std::string::npos) {
            end = content.size();
        }
        const std::string line = trimmed(content.substr(start, end - start));
        start = end + 1;
        if (line.empty() || line[0] == '#') {
            continue;
        }
        if (line[0] == '[') {
            // Nested groups are written [parent][child]: not this group
            inGroup = line == header;
            continue;
        }
        const size_t equal = line.find('=');
        if (!inGroup || equal == std::string::npos) {
            continue;
        }
        std::string key = trimmed(line.substr(0, equal));
        const size_t bracket = key.find('[');
        if (bracket != std::string::npos) {
            // key[$i] is an immutable entry. Localized entries (key[de]) and
            // deleted ones (key[$d]) do not set the value.
            const std::string options = key.substr(bracket);
            if (options.size() < 2 || options[1] != '$' || options.find('d') != std::string::npos) {
                continue;
            }
            key = trimmed(key.substr(0, bracket));
        }
        bool value;
        if (!parseBool(trimmed(line.substr(equal + 1)), value)) {
            continue;
        }
        if (key == "enabled") {
            settings.enabled = value;
        } else if (key.find(':') != std::string::npos) {
            services[key] = value;
        }
    }
    for (const auto &service : services) {
        if (!service.second) {
            settings.disabled.push_back(service.first);
        }
    }
    return settings;
}

std::atomic<const FileSettings *> g_settings{nullptr};

bool isKdenlive()
{
#if defined(_WIN32)
    wchar_t path[4 * MAX_PATH];
    const DWORD length = GetModuleFileNameW(nullptr, path, DWORD(std::size(path)));
    if (length == 0 || length >= std::size(path)) {
        return false;
    }
    const wchar_t *name = path;
    for (const wchar_t *c = path; *c; ++c) {
        if (*c == L'\\' || *c == L'/') {
            name = c + 1;
        }
    }
    return !_wcsicmp(name, L"kdenlive.exe");
#else
    char path[4096];
    const ssize_t length = readlink("/proc/self/exe", path, sizeof(path) - 1);
    if (length <= 0) {
        return false;
    }
    path[length] = '\0';
    const char *slash = strrchr(path, '/');
    return !strcmp(slash ? slash + 1 : path, "kdenlive");
#endif
}

NativePath moduleDirectory()
{
#if defined(_WIN32)
    HMODULE module = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCWSTR>(&moduleDirectory), &module)) {
        return {};
    }
    wchar_t path[4 * MAX_PATH];
    const DWORD length = GetModuleFileNameW(module, path, DWORD(std::size(path)));
    if (length == 0 || length >= std::size(path)) {
        return {};
    }
    const NativePath file(path, length);
    const size_t slash = file.find_last_of(L"\\/");
#else
    Dl_info info;
    if (!dladdr(reinterpret_cast<void *>(&moduleDirectory), &info) || !info.dli_fname) {
        return {};
    }
    const NativePath file = info.dli_fname;
    const size_t slash = file.rfind('/');
#endif
    return slash == NativePath::npos ? NativePath() : file.substr(0, slash);
}

} // namespace

const FileSettings &fileSettings()
{
    static const FileSettings defaults;
    const FileSettings *settings = g_settings.load(std::memory_order_acquire);
    return settings ? *settings : defaults;
}

void reloadFileSettings()
{
    static std::mutex mutex;
    // Other threads may still be reading the previous settings: they are kept.
    // They only change when the user applies the settings page.
    static auto *const all = new std::vector<std::unique_ptr<FileSettings>>();
    const NativePath path = settingsFilePath();
    std::string content;
    const bool found = !path.empty() && readFile(path, content);
    auto settings = std::make_unique<FileSettings>(found ? parseSettings(content) : FileSettings());
    logInfo("settings %s %s: GPU processing %s, %d services kept on the CPU", found ? "from" : "not found, defaults instead of",
            path.empty() ? "(no path)" : displayPath(path).c_str(), settings->enabled ? "on" : "off", int(settings->disabled.size()));
    std::lock_guard<std::mutex> lock(mutex);
    g_settings.store(settings.get(), std::memory_order_release);
    all->push_back(std::move(settings));
}

bool settingsPageWanted()
{
    const char *mode = getenv("GPUFX_SETTINGS_PAGE");
    if (mode && !strcmp(mode, "0")) {
        return false;
    }
    if (mode && !strcmp(mode, "1")) {
        return true;
    }
    return isKdenlive();
}

void loadSettingsPage(const gpufx_settings_host &host)
{
    static bool loaded = false;
    if (loaded) {
        return;
    }
    loaded = true;
    const NativePath directory = moduleDirectory();
    if (directory.empty()) {
        logWarning("cannot find the module's folder: no settings page");
        return;
    }
#if defined(_WIN32)
    const NativePath path = directory + L"\\gpufx\\gpufx-settings.dll";
    if (GetFileAttributesW(path.c_str()) == INVALID_FILE_ATTRIBUTES) {
        logInfo("no settings page: %s is missing", displayPath(path).c_str());
        return;
    }
    // The page uses Qt and KDE Frameworks, which Kdenlive has already loaded
    HMODULE library = LoadLibraryExW(path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!library) {
        // 126: a library it needs is missing, 127: a function is missing
        logWarning("cannot load the settings page %s (error %lu)", displayPath(path).c_str(), GetLastError());
        return;
    }
    auto init = reinterpret_cast<gpufx_settings_init_function>(reinterpret_cast<void (*)()>(GetProcAddress(library, "gpufx_settings_init")));
#else
    const NativePath path = directory + "/gpufx/gpufx-settings.so";
    if (access(path.c_str(), F_OK) != 0) {
        logInfo("no settings page: %s is missing", path.c_str());
        return;
    }
    void *library = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);
    if (!library) {
        logWarning("cannot load the settings page: %s", dlerror());
        return;
    }
    auto init = reinterpret_cast<gpufx_settings_init_function>(dlsym(library, "gpufx_settings_init"));
#endif
    if (!init) {
        logWarning("%s has no gpufx_settings_init()", displayPath(path).c_str());
        return;
    }
    if (!init(&host)) {
        logWarning("the settings page %s did not start", displayPath(path).c_str());
        return;
    }
    logInfo("settings page loaded from %s", displayPath(path).c_str());
}

} // namespace gpufx
