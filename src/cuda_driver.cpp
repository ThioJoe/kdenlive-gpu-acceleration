/*
    SPDX-FileCopyrightText: 2026 Kdenlive contributors
    SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#include "cuda_driver.h"

#include <cstdio>
#include <cstdlib>

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

void *openLibrary(const char *name)
{
#if defined(_WIN32)
    return reinterpret_cast<void *>(LoadLibraryA(name));
#else
    return dlopen(name, RTLD_NOW | RTLD_LOCAL);
#endif
}

void *findSymbol(void *library, const char *name)
{
#if defined(_WIN32)
    return reinterpret_cast<void *>(GetProcAddress(reinterpret_cast<HMODULE>(library), name));
#else
    return dlsym(library, name);
#endif
}

template <typename T> bool resolve(void *library, T &target, const char *name)
{
    target = reinterpret_cast<T>(findSymbol(library, name));
    return target != nullptr;
}

} // namespace

bool CudaDriver::load(std::string &error)
{
    if (m_library) {
        return true;
    }
    const char *override = getenv("GPUFX_CUDA_LIBRARY");
#if defined(_WIN32)
    const char *defaultName = "nvcuda.dll";
#else
    const char *defaultName = "libcuda.so.1";
#endif
    const char *name = (override && override[0]) ? override : defaultName;
    void *library = openLibrary(name);
    if (!library) {
        error = std::string("NVIDIA CUDA driver not found (") + name + ")";
        return false;
    }

    bool ok = true;
    ok &= resolve(library, Init, "cuInit");
    ok &= resolve(library, DriverGetVersion, "cuDriverGetVersion");
    ok &= resolve(library, DeviceGetCount, "cuDeviceGetCount");
    ok &= resolve(library, DeviceGet, "cuDeviceGet");
    ok &= resolve(library, DeviceGetName, "cuDeviceGetName");
    ok &= resolve(library, DeviceGetAttribute, "cuDeviceGetAttribute");
    ok &= resolve(library, DeviceTotalMem, "cuDeviceTotalMem_v2");
    ok &= resolve(library, DevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain");
    ok &= resolve(library, CtxPushCurrent, "cuCtxPushCurrent_v2");
    ok &= resolve(library, CtxPopCurrent, "cuCtxPopCurrent_v2");
    ok &= resolve(library, ModuleLoadDataEx, "cuModuleLoadDataEx");
    ok &= resolve(library, ModuleGetFunction, "cuModuleGetFunction");
    ok &= resolve(library, MemAlloc, "cuMemAlloc_v2");
    ok &= resolve(library, MemFree, "cuMemFree_v2");
    ok &= resolve(library, MemGetInfo, "cuMemGetInfo_v2");
    ok &= resolve(library, MemcpyHtoDAsync, "cuMemcpyHtoDAsync_v2");
    ok &= resolve(library, MemcpyDtoHAsync, "cuMemcpyDtoHAsync_v2");
    ok &= resolve(library, MemcpyDtoDAsync, "cuMemcpyDtoDAsync_v2");
    ok &= resolve(library, MemsetD8Async, "cuMemsetD8Async");
    ok &= resolve(library, StreamCreate, "cuStreamCreate");
    ok &= resolve(library, StreamSynchronize, "cuStreamSynchronize");
    ok &= resolve(library, LaunchKernel, "cuLaunchKernel");
    resolve(library, GetErrorName, "cuGetErrorName");
    resolve(library, GetErrorString, "cuGetErrorString");

    if (!ok) {
        error = std::string("The NVIDIA driver (") + name + ") is too old: some CUDA functions are missing";
        // Keep the library loaded, unloading a driver can be unsafe.
        return false;
    }
    m_library = library;
    return true;
}

std::string CudaDriver::errorText(CUresult result) const
{
    const char *name = nullptr;
    const char *text = nullptr;
    if (GetErrorName) {
        GetErrorName(result, &name);
    }
    if (GetErrorString) {
        GetErrorString(result, &text);
    }
    char buffer[64];
    snprintf(buffer, sizeof(buffer), "CUDA error %d", result);
    std::string message = name ? name : buffer;
    if (text) {
        message += std::string(" (") + text + ")";
    }
    return message;
}

} // namespace gpufx
