/*
    SPDX-FileCopyrightText: 2026 ThioJoe
    SPDX-License-Identifier: GPL-3.0-only
*/

#pragma once

#include <framework/mlt.h>

namespace gpufx {

// MLT 7.42 turned the frame's convert_image into a list of converters, which
// frame->convert_image dispatches to. Both are handled at runtime, so the
// module works with the MLT it is loaded into whatever headers it was built
// with (Kdenlive 26.08 ships a 7.41 development snapshot).

inline bool hasConvertImage(mlt_frame frame)
{
    auto list = static_cast<mlt_deque>(mlt_properties_get_data(MLT_FRAME_PROPERTIES(frame), "_convert_image_callbacks", nullptr));
    if (list) {
        return mlt_deque_count(list) > 0;
    }
    // Older MLT: the converter itself. Newer MLT without any converter: the
    // dispatcher, which then leaves the image as it is.
    return frame->convert_image != nullptr;
}

inline int convertImage(mlt_frame frame, uint8_t **image, mlt_image_format *format, mlt_image_format output)
{
    return frame->convert_image ? frame->convert_image(frame, image, format, output) : 1;
}

} // namespace gpufx
