/*
    SPDX-FileCopyrightText: 2026 ThioJoe
    SPDX-License-Identifier: GPL-3.0-only
*/

#pragma once

#include "effect.h"
#include "normalizer.h"

#include <framework/mlt.h>

namespace gpufx {

/** Replaces filter->process so that frames of this filter go through the bridge. */
void wrapFilter(mlt_filter filter, const EffectDescriptor &effect);
/** The same for a transition */
void wrapTransition(mlt_transition transition, const TransitionDescriptor &descriptor);
/** Routes the images of an avformat producer through the bridge, which
 *  converts the decoded pictures to RGBA on the GPU. */
void wrapProducer(mlt_producer producer);
/** The same for one of MLT's normalizing filters, which lets GPU images
 *  through (see normalizer.h). */
void wrapNormalizer(mlt_filter filter, const NormalizerDescriptor &descriptor);

/** Module wide switches (environment, settings, MLT global properties, per
 *  service). kind is "effect", "transition" or "conversion" (avformat for the
 *  decoding to RGBA, avcolor_space for the conversion to YUV, normalize for
 *  MLT's normalizing filters). */
bool moduleEnabled();
bool serviceEnabled(const char *kind, const char *service);
bool effectEnabled(const EffectDescriptor &effect);

} // namespace gpufx
