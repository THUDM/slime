#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/link_codex_skills.sh [--remove]

Link slime repository skills from .claude/skills into the local Codex skills directory.

Options:
  --remove   Remove previously created slime-* symlinks instead of creating them.
  -h, --help Show this help message.
USAGE
}

remove_mode=0
case "${1-}" in
  "") ;;
  --remove) remove_mode=1 ;;
  -h|--help) usage; exit 0 ;;
  *)
    usage >&2
    exit 1
    ;;
esac

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
source_dir="$repo_root/.claude/skills"
codex_home="${CODEX_HOME:-$HOME/.codex}"
target_dir="$codex_home/skills"
prefix="slime-"

if [[ ! -d "$source_dir" ]]; then
  echo "Missing source skills directory: $source_dir" >&2
  exit 1
fi

mkdir -p "$target_dir"

linked=0
removed=0

while IFS= read -r -d '' skill_dir; do
  skill_name="$(basename "$skill_dir")"
  link_path="$target_dir/${prefix}${skill_name}"

  if (( remove_mode )); then
    if [[ -L "$link_path" ]]; then
      rm -f "$link_path"
      printf 'removed %s\n' "$link_path"
      removed=$((removed + 1))
    fi
    continue
  fi

  ln -sfn "$skill_dir" "$link_path"
  printf 'linked %s -> %s\n' "$link_path" "$skill_dir"
  linked=$((linked + 1))
done < <(find "$source_dir" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)

if (( remove_mode )); then
  printf 'removed %d slime Codex skill link(s) from %s\n' "$removed" "$target_dir"
else
  printf 'linked %d slime Codex skill(s) into %s\n' "$linked" "$target_dir"
fi
