/*
    SPDX-FileCopyrightText: 2026 Kdenlive contributors
    SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#pragma once

#include "effect.h"

#include <framework/mlt.h>

namespace gpufx {

/** Replaces filter->process so that frames of this filter go through the bridge. */
void wrapFilter(mlt_filter filter, const EffectDescriptor &effect);
/** The same for a transition */
void wrapTransition(mlt_transition transition, const TransitionDescriptor &descriptor);
/** Routes the images of an avformat producer through the bridge, which
 *  converts the decoded pictures to RGBA on the GPU. */
void wrapProducer(mlt_producer producer);

/** Module wide switches (environment, MLT global properties, per service). */
bool moduleEnabled();
bool serviceEnabled(const char *service);
bool effectEnabled(const EffectDescriptor &effect);

} // namespace gpufx
