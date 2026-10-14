/*
    SPDX-FileCopyrightText: 2026 Kdenlive contributors
    SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#pragma once

#if defined(__GNUC__) || defined(__clang__)
#define GPUFX_PRINTF(fmt, args) __attribute__((format(printf, fmt, args)))
#else
#define GPUFX_PRINTF(fmt, args)
#endif

namespace gpufx {

/** Messages go through mlt_log() with a "[gpufx]" prefix. Debug messages are
 *  shown at info level when MLT_GPUFX_DEBUG is set. */
void logError(const char *format, ...) GPUFX_PRINTF(1, 2);
void logWarning(const char *format, ...) GPUFX_PRINTF(1, 2);
void logInfo(const char *format, ...) GPUFX_PRINTF(1, 2);
void logDebug(const char *format, ...) GPUFX_PRINTF(1, 2);
bool debugEnabled();

} // namespace gpufx
