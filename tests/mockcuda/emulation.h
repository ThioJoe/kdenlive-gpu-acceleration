/*
    SPDX-FileCopyrightText: 2026 Kdenlive contributors
    SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#pragma once

#include <cstddef>
#include <type_traits>
#include <utility>

/*
 * Registry of the kernels compiled for the CPU. cuLaunchKernel() in the mock
 * driver finds a kernel by name and calls its trampoline once per emulated
 * thread, with the same parameter array the module passes to the real driver.
 */

using GpufxTrampoline = void (*)(void **params);

void gpufx_emu_register(const char *name, GpufxTrampoline trampoline);

template <typename Signature> struct GpufxEmuInvoker;

template <typename... Args> struct GpufxEmuInvoker<void (*)(Args...)>
{
    template <void (*Kernel)(Args...), size_t... I> static void call(void **params, std::index_sequence<I...>)
    {
        Kernel(*static_cast<std::remove_reference_t<Args> *>(params[I])...);
    }

    template <void (*Kernel)(Args...)> static void trampoline(void **params) { call<Kernel>(params, std::index_sequence_for<Args...>{}); }
};

#define GPUFX_EMU_REGISTER(name) static const bool name##_registered = (gpufx_emu_register(#name, &GpufxEmuInvoker<decltype(&name)>::trampoline<&name>), true);
