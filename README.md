<!--
    SPDX-FileCopyrightText: 2026 ThioJoe
    SPDX-License-Identifier: GPL-3.0-only
-->

# GPU acceleration for Kdenlive

`gpufx` is an MLT module that runs some existing MLT filters on NVIDIA GPUs
with CUDA. It does not add new effects: projects keep using the regular
effects and parameters, and render the same with or without the module, with
or without an NVIDIA GPU.

Kdenlive itself is not modified. It's just 2 `.dll` files (`.so` files on Linux) that are added within the Kdenlive program directory, and is loaded automatically.

The module is one library that goes into
MLT's module folder of an installed Kdenlive (`lib\mlt` on Windows,
`usr/lib/mlt-7` in the Linux AppImage), and it
is used by every process that loads MLT from there: Kdenlive itself (monitors,
timeline preview) and the render processes. A second library adds a page to
Kdenlive's settings dialog for it (see [Settings](#settings)).

## Installing on Windows

The [Windows workflow](.github/workflows/windows.yml) builds the module and
its settings page for Kdenlive's Windows standalone build and uploads them as
an artifact, laid out like Kdenlive's folder. Open the latest run under *Actions*, download
`gpufx-kdenlive-<version>-windows` and unzip it, then drag its
`kdenlive-<version>_standalone` folder into the folder that contains your
Kdenlive folder and let Windows merge them. It adds
`lib\mlt\libmltgpufx.dll` and `lib\mlt\gpufx\gpufx-settings.dll`. To
uninstall, delete `libmltgpufx.dll` and the `gpufx` folder.

Or let [Tools/Install-Gpufx.ps1](Tools/Install-Gpufx.ps1) download the latest
release for your Kdenlive version and copy both files, given the Kdenlive folder:

```
powershell -NoProfile -ExecutionPolicy Bypass -File .\Tools\Install-Gpufx.ps1 "C:\path\to\kdenlive-26.08.2_standalone"
```

## Installing on Linux

The module is built for Kdenlive's Linux AppImage. An AppImage cannot be
changed, so the module goes into an extracted copy of it, in a folder of the
user's: no root is needed.

[Tools/install-gpufx.sh](Tools/install-gpufx.sh) does it all. Given the
AppImage, it extracts it into a folder next to it (`kdenlive-26.08.2` for
`kdenlive-26.08.2-x86_64.AppImage`), downloads the latest release for that
Kdenlive version and adds both files. Kdenlive is then started with the
`AppRun` file of that folder.

```
bash Tools/install-gpufx.sh ~/Applications/kdenlive-26.08.2-x86_64.AppImage
```

It also takes the folder of an AppImage extracted before. `--dest` chooses
the folder to extract to, `--package` installs a zip downloaded from the
workflow's artifacts instead of a release, and `--help` lists the other
options. It never asks anything: on an error it exits with 1 and prints the
reason.

By hand: the [Linux workflow](.github/workflows/linux.yml) uploads
`gpufx-kdenlive-<version>-linux`, which holds a `usr` folder. Extract the
AppImage with `./kdenlive-<version>-x86_64.AppImage --appimage-extract`
(which creates a `squashfs-root` folder, to rename as you like), copy the
package's `usr` folder into it, and start Kdenlive with its `AppRun`. It adds
`usr/lib/mlt-7/libmltgpufx.so` and `usr/lib/mlt-7/gpufx/gpufx-settings.so`.
To uninstall, delete `libmltgpufx.so` and the `gpufx` folder.

The programs of the extracted folder also work when started directly, such
as `usr/bin/melt` for renders from the command line: MLT loads its modules,
gpufx included, from the folder next to the program. Without a display (over
SSH, for example), set `QT_QPA_PLATFORM=offscreen`: MLT's Qt module, which
Transform, Drop Shadow and track compositing need, does not load otherwise,
and melt renders without those effects.

The packages of Linux distributions and the Flatpak are not covered. With
Kdenlive from a distribution, the module can be built from source against
the distribution's MLT (see [Building](#building)), and installing it into
MLT's module folder needs root.

## Accelerated effects

| Kdenlive effect                     | MLT service          | Notes |
|-------------------------------------|----------------------|-------|
| Drop Shadow                         | `dropshadow`         | Matches the CPU filter (Qt's `QGraphicsDropShadowEffect`): same alpha, colors within one level, 8-bit only |
| BoxBlur                             | `box_blur`           | Identical output, 8-bit only |
| Box Blur                            | `boxblur`            | Identical output |
| Intensity (Brightness), Fade in/out | `brightness`         | Lightweight, see below |
| Lift/gamma/gain                     | `lift_gamma_gain`    | Lightweight, see below |
| Square Blur                         | `frei0r.squareblur`  | Identical output |
| Glow                                | `frei0r.glow`        | Identical output |
| Soft Glow                           | `frei0r.softglow`    | Identical output |
| Sharpen                             | `frei0r.sharpness`   | Identical output |
| Transform                           | `qtblend`            | Matches the CPU filter (QPainter): same alpha, colors within one level. Alpha blend compositing and 8-bit only, see below |
| Track compositing                   | `qtblend` transition | Matches the CPU transition: same alpha, colors within one level, see below |
| Dissolve, Luma (wipes), same-track mixes | `luma` transition | Identical output, see below |
| Video decoding to RGBA              | `avformat` producer  | Identical output, 8-bit 4:2:0 video, see below |
| Output conversion to YUV            | `avcolor_space`      | Identical output, see below |
| Clip size and field order           | MLT's normalizing filters | Identical output, see below |
| Still images, titles, color clips   | `qimage`, `kdenlivetitle`, `color` producers | Identical output, see below |

Any mix of accelerated and CPU-only effects works: each accelerated effect is
moved to the GPU for its own processing, and the others stay on the CPU.

When several accelerated effects follow each other in an effect stack, the
frame stays on the GPU between them: it is uploaded once before the first one
and downloaded once after the last one.

*Lightweight* effects are cheap on the CPU, so on their own copying the frame
to the GPU and back would cost more than it saves. They only run on the GPU
when the frame is there anyway for a neighbouring accelerated effect (for
example a fade applied next to a drop shadow).

Frames in other formats than 8-bit RGBA (for example 16-bit `rgba64`
processing), unusual parameter values, and any CUDA error fall back to the
original CPU filter for that frame. frei0r effects always work in 8-bit RGBA;
they stay on the CPU when their MLT `threads` property splits the frame in
slices. An error that leaves the GPU in an unknown state turns GPU processing
off for the rest of the session.

### Transform

The Transform effect (MLT's `qtblend`) draws the clip with QPainter. The GPU
version reproduces Qt's raster engine: the transform with its rounding, the
antialiased edges (Qt's rasterizer is ported in `src/qt_raster.cpp`), the
16-bit bilinear sampling and blending, and the conversions back to 8 bits.
The only difference left comes from the approximate reciprocal that Qt uses
to un-premultiply colors (an SSE instruction whose exact results vary between
processors): some colors can differ by one level.

These cases stay on the CPU: composition modes other than *Alpha blend*,
rotation combined with *Distort*, and with MLT 7.38, scaling a clip down
(that version scales with `QImage::scaled()`). The geometry follows MLT 7.38
to 7.44; with a newer MLT the effect stays on the CPU until the module is
updated, unless `MLT_GPUFX_QTBLEND=force` is set.

### Track compositing

Kdenlive composites every video track onto the tracks below with MLT's
`qtblend` transition, which draws with QPainter like the Transform effect and
is reproduced the same way. The GPU version follows MLT's shortcuts too: a
top track without transparency replaces the tracks below without blending
(the opacity check runs on the GPU when the image is there).

Copying both images to the GPU and the result back costs about as much as
compositing them on the CPU, so a transition only runs on the GPU when one of
its images is already there (a clip with accelerated effects, or the result
of the transition below), or when the GPU can also convert the result to the
format the caller asks for (see below). With two composited tracks or more,
the whole stack is drawn on the GPU: every clip image is uploaded once and
the final frame is downloaded once. This includes Kdenlive's tracks, each of
which MLT wraps in a tractor of its own: the result of a clip's accelerated
effects is passed through the track's tractor on the GPU.

Frames are copied to and from the GPU through page-locked memory, which the
GPU reads and writes on its own: a downloaded frame is written straight into
the memory MLT then keeps (see `GPUFX_PINNED_MB` below). A thread waiting for
the GPU sleeps instead of keeping a CPU core busy.

The same cases as for Transform stay on the CPU (blend modes, 10-bit,
rotation with *Distort*, scaling down with MLT 7.38). MLT 7.41 only exists as
development versions (Kdenlive 26.08 ships one), which do not all size the
images the same way: when that makes a difference, the CPU composites.

### Dissolves and wipes

Kdenlive's dissolves, same-track mixes and wipes use MLT's `luma`
transition, which takes one of several code paths depending on its settings.
The GPU version follows the same paths with the same arithmetic:

* Without *Use transparency* (Kdenlive's default), MLT converts both images
  to YUV 4:2:2, blends them there in fixed point (with the SSE2 code of
  MLT's x86-64 builds, whose rounding differs from its C code) and converts
  the result back to RGBA. All of it runs on the GPU, so these dissolves run
  there even when both clips are in host memory.
* With *Use transparency*, and for wipes along a luma map, the images are
  blended in RGBA. Like track compositing, this only runs on the GPU when an
  image is there already.

These stay on the CPU: interlaced wipes, *Invert* on a wipe, video clips as
luma maps, dissolves with *Make padding transparent* turned off and without
*Use transparency*, and with MLT 7.41 development versions, wipes without
*Use transparency* (MLT 7.42 changed how they blend transparent images). The
first frame of a wipe is blended on the CPU, which loads the luma map.

### Image conversions

Effects and compositing work on RGBA images, while video is decoded as YUV
and encoded as YUV. MLT converts between them on the CPU with libswscale,
which takes more time than most effects: measured on one core of an x86-64
server processor, about 30 ms for a 1080p frame from YUV to RGBA, and 10 ms
back to YUV.

* **Video decoding to RGBA**: MLT's avformat producer converts every decoded
  frame to the format asked for. gpufx wraps the avformat producers: when
  RGBA is asked for and the video is 8-bit 4:2:0 YUV (`yuv420p`,
  `yuvj420p`, `yuva420p`, the format of nearly all H.264, HEVC and VP9
  video), it takes the frame as decoded, uploads it (1.5 bytes per pixel
  instead of 4) and converts it on the GPU. When the next stage is an
  accelerated effect, the RGBA image stays on the GPU. Other formats, odd
  sizes, interlaced frames whose height is not a multiple of 4, and hardware
  decoding (`hwaccel`) keep the CPU conversion.
* **Output conversion to YUV**: MLT's avformat consumer (rendering to
  `yuv420p` codecs) and Kdenlive's monitor ask for packed YUV 4:2:2. When the
  final effect or compositing ran on the GPU and MLT would convert with its
  `avcolor_space` filter, the GPU converts the result and only the YUV image
  (and its alpha channel) is downloaded. This also lets compositing and
  lightweight effects run on the GPU when nothing else of the frame is there.
  A render profile can ask for planar 4:2:0 instead, the format of most
  codecs, with `mlt_image_format=yuv420p` among its parameters
  (`mlt_image_format=yuv420p10` for 10-bit codecs, with `pix_fmt=p010le` or
  `yuv420p10le`): the GPU then converts to that format, and the encoder only
  copies it, instead of converting 4:2:2 on its own thread. The result is
  what MLT gives with that setting on the CPU, which subsamples the colors
  straight from RGBA rather than through 4:2:2.
* **Round trip through YUV**: compositing without a rectangle (Kdenlive's
  track compositing) first asks for the top track's image in the output
  format, and converts a YUV image with an alpha channel straight back to
  RGBA. An image converted from RGBA always has one. When the top track's
  last effect ran on the GPU, the image is converted to YUV and back there,
  with the same arithmetic, and stays on the GPU.
* **Clip size and field order**: MLT puts normalizing filters between each
  clip's producer and its effects (`color_transform`, `fieldorder`, `crop`,
  `rescale` and `resize`, and `deinterlace` for color clips and titles). As
  CPU filters, they needed every decoded frame in
  memory, only for it to go back to the GPU for the clip's effects or the
  compositing. gpufx wraps them: when they leave the image as it is, the most
  common case, it stays on the GPU. Scaling (`rescale`, with libswscale's
  nearest, bilinear, bicubic and Lanczos methods, the ones Kdenlive offers)
  and padding to the profile's aspect ratio (`resize`) run on the GPU.
  Changing the field order, deinterlacing, cropping and the other scaling
  methods stay on the CPU.
* **Still images, titles and color clips**: these producers make the same
  picture for every frame, but copy it into a new image each time, which the
  GPU then uploads. When the next stage is on the GPU, gpufx keeps the
  picture there: once two frames asking for the same thing (the same size,
  frame and producer properties apart from the position) got the same
  picture with the same changes to the frame, the next ones get a copy of it
  on the GPU and the same frame changes, without the producer. An image
  sequence changes picture every `ttl` frames, so each picture is kept
  separately. Animated titles (typewriter, moving view) are left alone. The
  pictures take at most 1/8 of the GPU memory (1 GiB at most), the least
  recently used going first.

The conversions and the scaling reproduce libswscale with the flags MLT uses: the
coefficients, chroma sampling positions and rounding are ported in
`src/sws_port.cpp`, and the output is identical for every colorspace MLT
handles (BT.601, BT.709, BT.2020, SMPTE 240M), limited and full range video,
and MLT's field by field conversion of interlaced video. This was checked
against libswscale of FFmpeg 6.1, 7.1 and 8.0, with and without its SIMD
code (the scaling and the 4:2:0 output against 7.1 and 8.0 with their SIMD
code, after comparing the code they use in 6.1, 7.1 and 8.0). Images
narrower than 8 pixels keep the CPU conversion to 4:2:0: for those,
libswscale writes past the end of the chroma planes. One exception: FFmpeg 8.0 rounds the conversion to full range YUV
differently from older versions, so a full range output (a consumer with
`color_range=full`, not Kdenlive's default) keeps the CPU conversion.

## Requirements

* An NVIDIA GPU of the Maxwell generation or newer (compute capability 5.0)
* The NVIDIA driver. Nothing else: the CUDA toolkit is not needed to build or
  to run the module. The driver compiles the kernels for the installed GPU
  the first time they are used and caches the result. On Linux, the module
  uses the driver's `libcuda.so.1`, which NVIDIA's driver installs (with
  either of its kernel modules); the nouveau driver has no CUDA.

Without an NVIDIA driver the module does nothing and every effect runs on the
CPU as usual.

## Settings

Kdenlive's settings (*Settings > Configure Kdenlive*) get a *GPU Acceleration*
page. It shows the GPU the module uses, turns GPU processing on or off, and
keeps chosen effects, compositing or conversions on the CPU. Changes apply to
the monitor right away and to renders started afterwards.

The page is a library of its own, `gpufx/gpufx-settings.dll` (or `.so`) next
to the module, because it needs Qt and KDE Frameworks while the module only needs
MLT. The module loads it only in Kdenlive's main program, which has those
libraries loaded already: render processes and other programs using MLT never
load it, and MLT does not look into the `gpufx` folder. When Kdenlive's
settings dialog (a `KConfigDialog`) opens, the page adds itself with
`KConfigDialog::addPage()`, with its own settings object, so the dialog's OK,
Apply, Cancel and Defaults buttons handle it like Kdenlive's own pages.

The settings are saved in Kdenlive's settings file (`kdenliverc` in
`%LOCALAPPDATA%` on Windows, in `~/.config` elsewhere, where the AppImage
names it `kdenlive-appimagerc`), in a `[gpufx]` group:

| Entry                       | Effect |
|-----------------------------|--------|
| `enabled=false`             | GPU processing off |
| `effect:<service>=false`    | Keep the effect with this MLT service on the CPU, e.g. `effect:dropshadow` |
| `transition:qtblend=false`  | Keep track compositing on the CPU (`transition:luma` for dissolves and wipes) |
| `conversion:avformat=false` | Keep the decoding of video to RGBA on the CPU (`conversion:avcolor_space` for the output conversion to YUV, `conversion:normalize` for MLT's normalizing filters, `conversion:stills` for the pictures of still images, titles and color clips) |

Every process using the module reads them when it starts, and Kdenlive again
when the settings are applied. With GPU processing off, render processes
leave MLT alone entirely.

The module also reads these environment variables:

| Variable                  | Effect |
|---------------------------|--------|
| `MLT_GPUFX=0`             | Do not touch any filter, and no settings page |
| `GPUFX_CONFIG_FILE`       | The settings file to read instead of Kdenlive's. The settings page sets it to Kdenlive's actual settings file, and the processes Kdenlive starts inherit it. Without it, the module looks for `kdenliverc`, or on Linux for the name Kdenlive's packages use (`kdenlive-appimagerc` with `PACKAGE_TYPE=appimage`, or when the module is in an extracted AppImage) |
| `GPUFX_SETTINGS_PAGE=0`   | Do not load the settings page (`1` loads it in any program, for tests) |
| `MLT_GPUFX_DISABLE`       | Comma separated MLT services to keep on the CPU, e.g. `dropshadow,box_blur`. `avformat` and `avcolor_space` stand for the two image conversions, `normalize` for MLT's normalizing filters, `stills` for still images, titles and color clips |
| `MLT_GPUFX_DEBUG=1`       | Log every frame processed on the GPU (MLT info log level) |
| `MLT_GPUFX_QTBLEND=force` | Accelerate Transform with MLT versions newer than the module knows |
| `GPUFX_CUDA_DEVICE=n`     | Use the n-th CUDA device (default 0) |
| `GPUFX_POOL_MB=n`         | Size of the GPU memory kept for reuse (default: 1/8 of the GPU memory, at most 1 GiB) |
| `GPUFX_STILLS_MB=n`       | GPU memory for the pictures of still images, titles and color clips (default: 1/8 of the GPU memory, at most 1 GiB) |
| `GPUFX_PINNED_MB=n`       | At most this much page-locked memory for frames on their way to and from the GPU, which the GPU copies faster and without the CPU (default 1024, `0` uses MLT's memory) |

Applications linking MLT can also set the MLT global property
`gpufx.enabled` to `0` or `1` at any time, which overrides the settings file.
Creating the `gpufx` filter service returns a filter whose properties describe
the state of the module (`available`, `status`, `device`, `effects`,
`gpu_frames`, ...).

## Building

The module only depends on MLT. The settings page needs Qt 6 and KDE
Frameworks 6 (KConfig and KConfigWidgets): it is built when they are found,
unless `GPUFX_SETTINGS_PAGE` is `OFF`.

```
cmake -S . -B build
cmake --build build
cmake --install build --prefix <Kdenlive's folder>
```

It finds MLT through its CMake package. For an MLT without development files,
such as the one in Kdenlive's Windows standalone build, pass MLT's headers and
library: `-DMLT_INCLUDE_DIR=<MLT source>/src -DMLT_LIBRARY=<Kdenlive>/bin/libmlt-7.dll`.

The module is installed in MLT's module directory: `lib/mlt` on Windows,
elsewhere the one of the `mlt-framework-7` pkg-config file when it is inside
the install prefix, otherwise `<prefix>/<libdir>/mlt-7`, which MLT does not
look into. Set `GPUFX_MLT_MODULE_DIR` to choose the location.

On Windows the module must use the same C runtime as MLT, as they free each
other's memory: Kdenlive's builds use UCRT, so a MinGW toolchain targeting
UCRT (not the older `msvcrt.dll`), such as MSYS2's UCRT64 environment.

The Windows workflow does this for Kdenlive's Windows standalone build: it
compiles the module with MSYS2's UCRT toolchain against the `libmlt-7.dll` of
the official build, and the settings page with MSYS2's Qt and KDE Frameworks.
The page runs with Kdenlive's own copies of those libraries, which can be
older, so the workflow checks that every function it uses is in them. It runs
the tests inside that build with the emulated GPU, then starts Kdenlive itself
with a variant of the page that opens Kdenlive's settings dialog, shows the
page and takes a screenshot. It uploads both libraries in a
`kdenlive-<version>_standalone` folder that merges into an installed one, and
the screenshots of the settings page. Both workflows build for the Kdenlive
release in `.github/kdenlive-version`; run one by hand (*Actions > Run
workflow*) to build for another Kdenlive version.

The Linux workflow does the same for Kdenlive's AppImage. It builds the
module in AlmaLinux 9 with GCC 14 from gcc-toolset (PyPA's `manylinux_2_34`
image), as Kdenlive's AppImage is built, against the AppImage's `libmlt-7.so`, and checks that it needs no newer
glibc or libstdc++ than the AppImage's own files, so that it runs wherever the
AppImage does. The settings page needs KDE Frameworks, which AlmaLinux 9 lacks:
it is built in Debian 13, whose Qt and KDE Frameworks are older than the
AppImage's, under the same check, and the workflow checks that every function
it uses is in the AppImage's libraries. The tests run inside the extracted
AppImage, Kdenlive opens the page on a virtual screen, and
`Tools/install-gpufx.sh` installs the package into the AppImage. It uploads
both libraries in a `usr` folder that merges into an extracted AppImage.

The CUDA kernels in `kernels/gpufx_kernels.cu` are shipped compiled to PTX in
`kernels/gpufx_kernels.ptx`, so that no CUDA toolkit is needed. After changing
the kernels, regenerate the PTX with nvcc:

```
cmake --build <builddir> --target gpufx_update_ptx
```

or configure with `-DGPUFX_BUILD_PTX=ON` to compile the kernels during the
build.

## Tests

`tests/gpufx_test` renders every accelerated effect, alone and in chains, once
with the original CPU filter and once through the GPU path, and compares the
pixels. The conversions are compared on video files the test writes
(YUV4MPEG2, decoded by MLT's avformat producer). It also checks the number of transfers between memory and GPU, the
fallbacks on errors, the behaviour without a GPU, and the settings file.

`tests/gpufx_settings_test` opens a settings dialog made like Kdenlive's,
checks that the module loads the page and that the page adds itself, then
applies changes and checks the settings file and the module. Run from the
`bin` folder of Kdenlive's Windows build (or `usr/bin` of the extracted
AppImage, with `LD_LIBRARY_PATH` set to its `usr/lib`), it uses Kdenlive's Qt
and KDE Frameworks; given a file name, it saves a screenshot of the page.

By default the tests run against `mockcuda`, an emulated CUDA driver built
with the tests: it executes the same kernel source compiled as C++ on the CPU,
so the tests run on machines without an NVIDIA GPU. To run the same
comparisons on a real GPU, point the tests at the driver:

```
GPUFX_CUDA_LIBRARY=libcuda.so.1 ./gpufx_test      # Linux
set GPUFX_CUDA_LIBRARY=nvcuda.dll & gpufx_test.exe  # Windows
```

On Windows, MLT looks for its modules next to the executable that uses it:
set `MLT_REPOSITORY` to MLT's module folder (for Kdenlive's standalone build,
`lib\mlt`), the tests pass it on to MLT.

## Performance

On an NVIDIA RTX 5090 with Kdenlive 26.08.1 for Windows (MLT 7.41, Qt 6.11,
FFmpeg 8), rendering 4K 60 fps projects made of Lottie animations with
Transform effects over an image background: H.264 4:2:0 renders ran 1.5 to
2.2 times as fast, and renders to lossless RGB took about as long with 20 to
30% less CPU time (the drawing of the animations and the encoding take most
of the time there). Every frame was identical to the CPU render.

## How it works

* `src/module.cpp` registers the module. It listens to MLT's
  `filter-create-done`, `transition-create-done` and `producer-create-done`
  factory events and wraps the filters, transitions and producers it knows.
* `src/settings_file.cpp` reads the settings and loads the settings page,
  `settings/settingspage.cpp`. `src/settings_page_api.h` is the interface
  between them.
* `src/bridge.cpp` replaces the wrapped filter's `process()`: the original
  function still runs and pushes the CPU `get_image` on the frame, and the
  bridge slides its own `get_image` on top of it. For each frame the bridge
  reads the effect parameters, then either calls the CPU `get_image`
  unchanged or fetches the input image, runs the effect on the GPU and
  returns the result. Between two accelerated effects the image is passed on
  the GPU through frame properties instead of being downloaded.
* `src/gpu_runtime.cpp` loads the CUDA driver at runtime, creates the
  context, loads the PTX and keeps pools of GPU memory and streams.
* `src/effect_*.cpp` read the parameters of each effect, exactly like the CPU
  filter does, and launch the kernels. `src/qt_raster.cpp` holds the parts of
  Qt's paint engine needed to draw like QPainter.
* `src/convert.cpp` runs the image conversions and the scaling, with the
  libswscale setup ported in `src/sws_port.cpp`. `src/normalizers.cpp` holds
  what MLT's normalizing filters do to a frame, `src/stills.cpp` the pictures
  of still producers kept on the GPU.
* `kernels/gpufx_kernels.cu` contains the kernels. They reproduce the integer
  arithmetic of the CPU code (for the drop shadow, Qt's blur, scaling and
  compositing), which is why the output matches.

## Adding an effect

1. Write the kernels in `kernels/gpufx_kernels.cu` (no shared memory, no
   `pow`/`exp`: precompute on the host) and list them in
   `kernels/gpufx_kernel_list.h`. Regenerate the PTX.
2. Add an `EffectDescriptor` in a `src/effect_*.cpp` file: its `plan()`
   function reads the parameters for the frame the same way the CPU filter
   does and returns a job, or `Decision::Cpu` for anything it does not
   handle identically. Register it in `effectRegistry()` in `src/module.cpp`.
3. Add comparisons with the CPU filter to `tests/test_gpufx.cpp`.
