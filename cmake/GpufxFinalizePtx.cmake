# SPDX-FileCopyrightText: 2026 ThioJoe
# SPDX-License-Identifier: GPL-3.0-only

# Turns the PTX written by nvcc into the file shipped with the sources:
#  - adds a license header (PTX accepts // comments)
#  - declares PTX ISA 6.0: the kernels only use basic instructions, and an old
#    ISA version lets the driver of older installations (CUDA 9 / R384 and
#    newer) load them, while nvcc writes the version of its own release
#
# Usage: cmake -DINPUT=raw.ptx -DOUTPUT=gpufx_kernels.ptx -P GpufxFinalizePtx.cmake

if(NOT INPUT OR NOT OUTPUT)
    message(FATAL_ERROR "GpufxFinalizePtx.cmake needs INPUT and OUTPUT")
endif()

file(READ "${INPUT}" ptx)
string(REGEX REPLACE "\n\\.version [0-9]+\\.[0-9]+\n" "\n.version 6.0\n" ptx "${ptx}")
if(NOT ptx MATCHES "\n\\.version 6\\.0\n")
    message(FATAL_ERROR "No .version directive found in ${INPUT}")
endif()
# Drop the compiler banner, it contains build specific information
string(REGEX REPLACE "^(//[^\n]*\n)+" "" ptx "${ptx}")

set(header "//\n// SPDX-FileCopyrightText: 2026 ThioJoe\n// SPDX-License-Identifier: GPL-3.0-only\n//\n// Generated from gpufx_kernels.cu, do not edit.\n// Rebuild with: cmake --build <builddir> --target gpufx_update_ptx\n//\n")
file(WRITE "${OUTPUT}" "${header}${ptx}")
