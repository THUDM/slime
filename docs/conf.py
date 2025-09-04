import os
import sys
from datetime import datetime
from pathlib import Path
import shutil

sys.path.insert(0, os.path.abspath("../.."))

__version__ = "0.0.1"

project = "slime"
copyright = f"2025-{datetime.now().year}, slime"
author = "slime Team"

version = __version__
release = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
    "sphinx_tabs.tabs",
    "myst_parser",
    "sphinx_copybutton",
    "sphinxcontrib.mermaid",
    "nbsphinx",
    "sphinx.ext.mathjax",
]

nbsphinx_allow_errors = True
nbsphinx_execute = "never"

autosectionlabel_prefix_document = True
nbsphinx_allow_directives = True


myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "colon_fence",
    "html_image",
    "linkify",
    "substitution",
]

myst_heading_anchors = 3

nbsphinx_kernel_name = "python3"
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc={'figure.dpi': 96}",
]


nb_render_priority = {
    "html": (
        "application/vnd.jupyter.widget-view+json",
        "application/javascript",
        "text/html",
        "image/svg+xml",
        "image/png",
        "image/jpeg",
        "text/markdown",
        "text/latex",
        "text/plain",
    )
}

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "colon_fence",
    "html_image",
    "linkify",
    "substitution",
]

myst_heading_anchors = 3
myst_ref_domains = ["std", "py"]

templates_path = ["_templates"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"

language = os.environ.get("SLIME_DOC_LANG", "en")

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

pygments_style = "sphinx"

html_theme = "sphinx_book_theme"
html_logo = "_static/image/logo.jpg"
html_favicon = "_static/image/logo.ico"
html_title = project
html_copy_source = True
html_last_updated_fmt = ""

html_theme_options = {
    "repository_url": "https://github.com/THUDM/slime",
    "repository_branch": "main",
    "show_navbar_depth": 3,
    "max_navbar_depth": 4,
    "collapse_navbar": True,
    "use_edit_page_button": True,
    "use_source_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_download_button": True,
    "use_sidenotes": True,
    "show_toc_level": 2,
}

html_context = {
    "display_github": True,
    "github_user": "sgl-project",
    "github_repo": "sgl-project.github.io",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

html_static_path = ["_static"]
html_css_files = ["css/custom_log.css"]
# Add custom javascript for language toggle (en <-> zh)
html_js_files = [
    "js/lang-toggle.js",
]


def _sync_examples(app):
    """Sync top-level examples into language-specific doc trees.

    Policy:
      - README.md -> English docs/en/_examples_synced/<example>/README.md
      - README_zh.md -> Chinese docs/zh/_examples_synced/<example>/README_zh.md
      - If a language-specific README missing, that example is simply skipped for that language.
    """
    docs_root = Path(__file__).resolve().parent
    src_dir = docs_root.parent / "examples"
    if not src_dir.exists():
        return

    lang_cfgs = {
        "en": {
            "dir": docs_root / "en",
            "readme_name": "README.md",
        },
        "zh": {
            "dir": docs_root / "zh",
            # primary preferred name; will fallback to README.md
            "readme_name": "README_zh.md",
        },
    }

    for lang, cfg in lang_cfgs.items():
        lang_dir = cfg["dir"]
        if not lang_dir.exists():
            continue
        out_dir = lang_dir / "_examples_synced"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        entries = []  # (example_name, readme_rel_path)
        for d in sorted(src_dir.iterdir()):
            if not d.is_dir():
                continue
            # language-specific selection with fallback for zh
            if lang == "zh":
                primary = d / cfg["readme_name"]  # README_zh.md
                fallback = d / "README.md"
                candidate = primary if primary.exists() else fallback
            else:
                candidate = d / cfg["readme_name"]
            if not candidate.exists():
                continue  # skip entirely if nothing suitable
            target_dir = out_dir / d.name
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, target_dir / "README.md")
            entries.append((d.name, f"_examples_synced/{d.name}/README.md"))


def setup(app):
    # ensure examples are synced before reading source files
    app.connect("builder-inited", _sync_examples)


myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "colon_fence",
]
myst_heading_anchors = 5

htmlhelp_basename = "slimedoc"

latex_elements = {}

latex_documents = [
    (master_doc, "slime.tex", "slime Documentation", "slime Team", "manual"),
]

man_pages = [(master_doc, "slime", "slime Documentation", [author], 1)]

texinfo_documents = [
    (
        master_doc,
        "slime",
        "slime Documentation",
        author,
        "slime",
        "One line description of project.",
        "Miscellaneous",
    ),
]

epub_title = project

epub_exclude_files = ["search.html"]

copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

autodoc_preserve_defaults = True
navigation_with_keys = False

autodoc_mock_imports = [
    "torch",
    "transformers",
    "triton",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3.12", None),
    "typing_extensions": ("https://typing-extensions.readthedocs.io/en/latest", None),
    "pillow": ("https://pillow.readthedocs.io/en/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}

html_theme = "sphinx_book_theme"


nbsphinx_prolog = """
.. raw:: html

    <style>
        .output_area.stderr, .output_area.stdout {
            color: #d3d3d3 !important; /* light gray */
        }
    </style>
"""
