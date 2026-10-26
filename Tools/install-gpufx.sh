#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 ThioJoe
# SPDX-License-Identifier: GPL-3.0-only
#
# Downloads the GPU module from the ThioJoe/kdenlive-gpu-acceleration releases
# and installs it into Kdenlive's Linux AppImage, without root:
#
#   install-gpufx.sh <Kdenlive AppImage, or the folder of an extracted one>
#
# An AppImage cannot be changed, so it is first extracted into a folder next
# to it, named after it (kdenlive-26.08.2-x86_64.AppImage gives
# kdenlive-26.08.2). The AppImage itself is left as it is. The module goes to
#   <folder>/usr/lib/mlt-7/libmltgpufx.so
#   <folder>/usr/lib/mlt-7/gpufx/gpufx-settings.so
# and Kdenlive is then started with <folder>/AppRun.
#
# It takes the newest release with a build for the Kdenlive version of the
# folder (from its AppStream data, else from its name, unless --version is
# given), or the release that --tag names (for example 26.08.2_0.2.0).
#
# Options:
#   --version <x.y.z>   Kdenlive version
#   --tag <tag>         release to take instead of the newest
#   --package <path>    install this package (a zip from the releases or from
#                       the workflow's artifacts, or its unzipped folder)
#                       instead of downloading one
#   --dest <folder>     where to extract an AppImage, instead of next to it
#   --repo <owner/repo> repository of the releases
#
# It never waits for input, so it can run unattended: it exits with 0 when
# installed and 1 on any error, with the reason on the error stream.

set -euo pipefail

repo=ThioJoe/kdenlive-gpu-acceleration
version=
tag=
package=
dest=
target=

fail() {
    echo "Error: $*" >&2
    exit 1
}

while [ $# -gt 0 ]; do
    case "$1" in
        --version|--tag|--package|--dest|--repo)
            [ $# -ge 2 ] || fail "$1 needs a value."
            case "$1" in
                --version) version=$2 ;;
                --tag) tag=$2 ;;
                --package) package=$2 ;;
                --dest) dest=$2 ;;
                --repo) repo=$2 ;;
            esac
            shift 2
            ;;
        -h|--help)
            sed -n '4,31p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        -*)
            fail "Unknown option $1."
            ;;
        *)
            [ -z "$target" ] || fail "Only one AppImage or folder can be given."
            target=$1
            shift
            ;;
    esac
done

[ -n "$target" ] || fail "Give the Kdenlive AppImage, or the folder of an extracted one (the one containing AppRun and usr)."
[ -e "$target" ] || fail "'$target' does not exist."
[ -z "$package" ] || [ -e "$package" ] || fail "'$package' does not exist."
target=$(cd "$(dirname "$target")" && pwd)/$(basename "$target")

work=
cleanup() {
    if [ -n "$work" ]; then
        rm -rf "$work"
    fi
}
trap cleanup EXIT

# Downloads $1 into the file $2
fetch() {
    if command -v curl > /dev/null; then
        curl --fail --location --retry 3 --silent --show-error --output "$2" "$1"
    elif command -v wget > /dev/null; then
        wget --quiet --tries=3 --output-document="$2" "$1"
    else
        fail "Neither curl nor wget is installed."
    fi
}

is_kdenlive_folder() {
    [ -f "$1/usr/bin/kdenlive" ] && [ -d "$1/usr/lib/mlt-7" ]
}

# An ELF file with the mark of AppImages (AI and the type, 2) at offset 8
is_appimage() {
    local header
    header=$(head -c 11 "$1" | od -An -tx1 | tr -d ' \n')
    [ "${header:0:8}" = 7f454c46 ] && [ "${header:16:6}" = 414902 ]
}

if [ -d "$target" ]; then
    folder=${target%/}
    is_kdenlive_folder "$folder" || fail "'$folder' is not an extracted Kdenlive AppImage: it has no usr/bin/kdenlive and usr/lib/mlt-7."
    [ -z "$dest" ] || fail "--dest is only for an AppImage."
else
    # Extracted next to the AppImage, as kdenlive-26.08.2 for
    # kdenlive-26.08.2-x86_64.AppImage
    if [ -z "$dest" ]; then
        name=$(basename "$target")
        name=${name%.AppImage}
        name=${name%.appimage}
        name=${name%-x86_64}
        [ "$name" != "$(basename "$target")" ] || name=$name-extracted
        dest=$(dirname "$target")/$name
    fi
    mkdir -p "$(dirname "$dest")" || fail "Cannot create the folder of '$dest'."
    folder=$(cd "$(dirname "$dest")" && pwd)/$(basename "$dest")
    if [ -e "$folder" ]; then
        is_kdenlive_folder "$folder" || fail "'$folder' exists and is not an extracted Kdenlive AppImage: give another folder with --dest."
        echo "Using the AppImage already extracted in $folder"
    else
        # The AppImage extracts itself into squashfs-root in the current
        # folder, here a temporary folder next to the destination
        appimage=$target
        is_appimage "$appimage" || fail "'$target' is not an AppImage."
        extract=$(mktemp -d "$(dirname "$folder")/.gpufx-extract.XXXXXX" 2> /dev/null) || fail "Cannot write in $(dirname "$folder"): give another folder with --dest."
        work=$extract
        if [ ! -x "$appimage" ]; then
            if [ -O "$appimage" ]; then
                chmod u+x "$appimage"
            else
                cp "$appimage" "$extract/kdenlive.AppImage"
                chmod u+x "$extract/kdenlive.AppImage"
                appimage=$extract/kdenlive.AppImage
            fi
        fi
        echo "Extracting $target into $folder"
        (cd "$extract" && "$appimage" --appimage-extract > /dev/null) || fail "Cannot extract '$target'."
        is_kdenlive_folder "$extract/squashfs-root" || fail "'$target' is not Kdenlive's AppImage: it has no usr/bin/kdenlive and usr/lib/mlt-7."
        mv "$extract/squashfs-root" "$folder"
        rm -rf "$extract"
        work=
    fi
fi

# The build has to match the Kdenlive version: the newest release in
# Kdenlive's AppStream data, else the one in the name of the folder or AppImage
if [ -z "$version" ]; then
    for metainfo in "$folder"/usr/share/metainfo/org.kde.kdenlive.appdata.xml "$folder"/usr/share/metainfo/org.kde.kdenlive.metainfo.xml; do
        if [ -f "$metainfo" ]; then
            version=$(grep -o '<release [^>]*version="[0-9.]*"' "$metainfo" | head -n 1 | sed 's/.*version="\([0-9.]*\)"/\1/')
            [ -z "$version" ] || break
        fi
    done
fi
if [ -z "$version" ]; then
    for name in "$(basename "$target")" "$(basename "$folder")"; do
        if [[ $name =~ ^[Kk]denlive-([0-9]+(\.[0-9]+)+) ]]; then
            version=${BASH_REMATCH[1]}
            break
        fi
    done
fi
if [ -n "$version" ]; then
    [[ $version =~ ^[0-9]+(\.[0-9]+)+$ ]] || fail "'$version' is not a version number like 26.08.2."
    echo "Kdenlive $version"
elif [ -z "$package" ]; then
    fail "Cannot tell the Kdenlive version: pass --version, for example --version 26.08.2."
fi

work=$(mktemp -d "${TMPDIR:-/tmp}/gpufx-install.XXXXXX" 2> /dev/null) || fail "Cannot create a temporary folder."
if [ -n "$package" ] && [ -d "$package" ]; then
    zip=
    unpacked=$package
else
    if [ -n "$package" ]; then
        zip=$package
    else
        # The package for this Kdenlive version, named after the release's
        # version (gpufx-kdenlive-26.08.2_0.2.0-linux.zip) or, before 0.2.0,
        # after the Kdenlive version alone (gpufx-kdenlive-26.08.1-linux.zip)
        if [ -n "$tag" ]; then
            api=https://api.github.com/repos/$repo/releases/tags/$tag
        else
            api="https://api.github.com/repos/$repo/releases?per_page=100"
        fi
        fetch "$api" "$work/releases.json" || fail "Cannot get the releases of $repo from $api."
        pattern="/gpufx-kdenlive-${version//./\\.}(_[0-9]+(\\.[0-9]+)*)?-linux\\.zip\$"
        # Releases come newest first
        url=$(grep -o '"browser_download_url": *"[^"]*"' "$work/releases.json" | grep -o 'https://[^"]*' | grep -E "$pattern" | head -n 1 || true)
        if [ -z "$url" ]; then
            if [ -n "$tag" ]; then
                fail "Release $tag has no build for Kdenlive $version on Linux."
            fi
            fail "No release has a build for Kdenlive $version on Linux yet."
        fi
        zip=$work/${url##*/}
        echo "Downloading $url"
        fetch "$url" "$zip" || fail "Download failed."
    fi
    unpacked=$work/package
    mkdir "$unpacked"
    if command -v unzip > /dev/null; then
        unzip -q "$zip" -d "$unpacked" || fail "Cannot unzip $zip."
    elif command -v python3 > /dev/null; then
        python3 -c 'import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])' "$zip" "$unpacked" || fail "Cannot unzip $zip."
    elif command -v bsdtar > /dev/null; then
        bsdtar -xf "$zip" -C "$unpacked" || fail "Cannot unzip $zip."
    else
        fail "Cannot unzip $zip: install unzip (or python3)."
    fi
fi

source=$unpacked/usr/lib/mlt-7
[ -f "$source/libmltgpufx.so" ] || fail "The package does not contain usr/lib/mlt-7/libmltgpufx.so."

# Copied under a temporary name, then renamed: a Kdenlive running from this
# folder keeps the files it has loaded
mlt=$folder/usr/lib/mlt-7
mkdir -p "$mlt/gpufx"
install_file() {
    cp "$1" "$2.gpufx-new"
    chmod 755 "$2.gpufx-new"
    mv -f "$2.gpufx-new" "$2"
}
install_file "$source/libmltgpufx.so" "$mlt/libmltgpufx.so"
if [ -f "$source/gpufx/gpufx-settings.so" ]; then
    install_file "$source/gpufx/gpufx-settings.so" "$mlt/gpufx/gpufx-settings.so"
fi

echo "Installed $mlt/libmltgpufx.so and $mlt/gpufx/gpufx-settings.so"
echo "Start Kdenlive with $folder/AppRun"
# Only a hint: the module runs everything on the CPU without the driver
if ! { /sbin/ldconfig -p 2>/dev/null || ldconfig -p 2>/dev/null; } | grep -q 'libcuda\.so\.1'; then
    echo "Note: libcuda.so.1, the CUDA library of NVIDIA's driver, was not found: without it every effect runs on the CPU."
fi
exit 0
