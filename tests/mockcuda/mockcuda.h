/*
    SPDX-FileCopyrightText: 2026 ThioJoe
    SPDX-License-Identifier: GPL-3.0-only
*/

#pragma once

#include <cstddef>
#include <cstdint>

#if defined(_WIN32)
#define MOCKCUDA_EXPORT __declspec(dllexport)
#else
#define MOCKCUDA_EXPORT __attribute__((visibility("default")))
#endif

/** Counters of the emulated driver, read by the tests with mockcuda_get_stats() */
struct MockCudaStats
{
    uint64_t uploads;
    uint64_t downloads;
    uint64_t uploadedBytes;
    uint64_t downloadedBytes;
    uint64_t launches;
    uint64_t allocations;
    uint64_t frees;
    uint64_t streams;
    uint64_t liveAllocations;
    uint64_t misuse; // API calls the real driver would reject
    uint64_t pinnedUploads;   // from page-locked memory
    uint64_t pinnedDownloads; // to page-locked memory
    uint64_t hostAllocations; // of page-locked memory
    uint64_t eventWaits;
};

enum MockCudaOp { MOCKCUDA_ALLOC, MOCKCUDA_UPLOAD, MOCKCUDA_DOWNLOAD, MOCKCUDA_LAUNCH, MOCKCUDA_OP_COUNT };

typedef void (*MockCudaGetStats)(MockCudaStats *stats);
typedef void (*MockCudaResetStats)();
/** Makes the count-th next operation of that kind fail with error */
typedef void (*MockCudaFailAfter)(MockCudaOp op, int count, int error);
