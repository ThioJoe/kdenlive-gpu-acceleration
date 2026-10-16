# SPDX-FileCopyrightText: 2026 ThioJoe
# SPDX-License-Identifier: GPL-3.0-only

# Writes a C++ source defining gpufx_ptx[], the NUL terminated content of the
# PTX file, and gpufx_ptx_size.
#
# Usage: cmake -DINPUT=gpufx_kernels.ptx -DOUTPUT=gpufx_ptx.cpp -P GpufxEmbedPtx.cmake

if(NOT INPUT OR NOT OUTPUT)
    message(FATAL_ERROR "GpufxEmbedPtx.cmake needs INPUT and OUTPUT")
endif()

file(READ "${INPUT}" content HEX)
string(LENGTH "${content}" length)
math(EXPR size "${length} / 2")
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," bytes "${content}")
# Break the array into lines of 32 bytes
string(REGEX REPLACE "((0x[0-9a-f][0-9a-f],){32})" "\\1\n" bytes "${bytes}")

file(WRITE "${OUTPUT}" "// Generated from ${INPUT}, do not edit.\n#include <cstddef>\n\nextern const unsigned char gpufx_ptx[] = {\n${bytes}0x00};\nextern const size_t gpufx_ptx_size = ${size};\n")
