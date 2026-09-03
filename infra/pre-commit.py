#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     # Pin the shared checks and vendored guidance to one audited revision.
#     "marin-style @ git+https://github.com/marin-community/marin-style@5094279da60b47b9a8fa8effaf7f73cd13f1e96f",
# ]
# ///
"""Run the pinned Marin-style checks configured for Slime."""

from marin_style.precommit import main

if __name__ == "__main__":
    raise SystemExit(main())
