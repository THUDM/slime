#!/bin/sh

set -eu

# Detect the shell from which the script was called
parent=$(ps -o comm $PPID |tail -1)
parent=${parent#-}  # remove the leading dash that login shells have
case "$parent" in
  # shells supported by `micromamba shell init`
  bash|fish|xonsh|zsh)
    shell=$parent
    ;;
  *)
    # use the login shell (basename of $SHELL) as a fallback
    shell=${SHELL##*/}
    ;;
esac

# Parsing arguments
if [ -t 0 ] ; then
  printf "Micromamba binary folder? [~/.local/bin] "
  read BIN_FOLDER
  printf "Init shell ($shell)? [Y/n] "
  read INIT_YES
  printf "Configure conda-forge? [Y/n] "
  read CONDA_FORGE_YES
fi

# Fallbacks
BIN_FOLDER="${BIN_FOLDER:-${HOME}/.local/bin}"
INIT_YES="${INIT_YES:-yes}"
CONDA_FORGE_YES="${CONDA_FORGE_YES:-yes}"

# Prefix location is relevant only if we want to call `micromamba shell init`
case "$INIT_YES" in
  y|Y|yes)
    if [ -t 0 ]; then
      printf "Prefix location? [~/micromamba] "
      read PREFIX_LOCATION
    fi
    ;;
esac
PREFIX_LOCATION="${PREFIX_LOCATION:-${HOME}/micromamba}"

# Computing artifact location
case "$(uname)" in
  Linux)
    PLATFORM="linux" ;;
  Darwin)
    PLATFORM="osx" ;;
  *NT*)
    PLATFORM="win" ;;
esac

ARCH="$(uname -m)"
case "$ARCH" in
  aarch64|ppc64le|arm64)
      ;;  # pass
  *)
    ARCH="64" ;;
esac

case "$PLATFORM-$ARCH" in
  linux-aarch64|linux-ppc64le|linux-64|osx-arm64|osx-64|win-64)
      ;;  # pass
  *)
    echo "Failed to detect your OS" >&2
    exit 1
    ;;
esac

if [ "${VERSION:-}" = "" ]; then
  RELEASE_URL="https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-${PLATFORM}-${ARCH}"
else
  RELEASE_URL="https://github.com/mamba-org/micromamba-releases/releases/download/${VERSION}/micromamba-${PLATFORM}-${ARCH}"
fi


# Downloading artifact
mkdir -p "${BIN_FOLDER}"
if hash curl >/dev/null 2>&1; then
  curl "${RELEASE_URL}" -o "${BIN_FOLDER}/micromamba" -fsSL --compressed ${CURL_OPTS:-}
elif hash wget >/dev/null 2>&1; then
  wget ${WGET_OPTS:-} -qO "${BIN_FOLDER}/micromamba" "${RELEASE_URL}"
else
  echo "Neither curl nor wget was found" >&2
  exit 1
fi
chmod +x "${BIN_FOLDER}/micromamba"


# Initializing shell
case "$INIT_YES" in
  y|Y|yes)
    case $("${BIN_FOLDER}/micromamba" --version) in
      1.*|0.*)
        shell_arg=-s
        prefix_arg=-p
        ;;
      *)
        shell_arg=--shell
        prefix_arg=--root-prefix
        ;;
    esac
    "${BIN_FOLDER}/micromamba" shell init $shell_arg "$shell" $prefix_arg "$PREFIX_LOCATION"

    echo "Please restart your shell to activate micromamba or run the following:\n"
    echo "  source ~/.bashrc (or ~/.zshrc, ~/.xonshrc, ~/.config/fish/config.fish, ...)"
    ;;
  *)
    echo "You can initialize your shell later by running:"
    echo "  micromamba shell init"
    ;;
esac


# Initializing conda-forge
case "$CONDA_FORGE_YES" in
  y|Y|yes)
    "${BIN_FOLDER}/micromamba" config append channels conda-forge
    "${BIN_FOLDER}/micromamba" config append channels nodefaults
    "${BIN_FOLDER}/micromamba" config set channel_priority strict
    ;;
esac