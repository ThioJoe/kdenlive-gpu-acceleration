/*
    SPDX-FileCopyrightText: 2026 Kdenlive contributors
    SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

// The CUDA kernels, built as plain C++ for the emulated driver.

#define GPUFX_EMULATION 1
#include "../../kernels/gpufx_kernels.cu"

#include "../../kernels/gpufx_kernel_list.h"
#include "emulation.h"

GPUFX_KERNEL_LIST(GPUFX_EMU_REGISTER)
