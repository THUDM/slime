import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

__version__ = "0.4.0"

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

myst_heading_anchors = 5

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
    "path_to_docs": f"docs/{language}",
    "show_navbar_depth": 1,
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
    "github_user": "THUDM",
    "github_repo": "slime",
    "github_version": "main",
    "conf_py_path": f"/docs/{language}/",
}

html_static_path = ["_static"]
html_css_files = ["css/custom_log.css"]
# Add custom javascript for language toggle (en <-> zh)
html_js_files = [
    ("js/lang-toggle.js", {"data-layout": os.environ.get("SLIME_DOC_LAYOUT", "root")}),
]


def _sync_examples(app):
    """Copy example docs and assets into the current language's source tree.

    Chinese builds prefer README_zh.md and fall back to README.md.
    The generated copy is always named README.md to keep navigation consistent.
    """
    src_dir = Path(__file__).resolve().parent.parent / "examples"
    out_dir = Path(app.srcdir) / "_examples_synced"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for example_dir in sorted(src_dir.iterdir()):
        if not example_dir.is_dir():
            continue
        readme = example_dir / "README.md"
        if app.config.language == "zh" and (example_dir / "README_zh.md").exists():
            readme = example_dir / "README_zh.md"
        if not readme.exists():
            continue
        target_dir = out_dir / example_dir.name
        shutil.copytree(
            example_dir,
            target_dir,
            ignore=shutil.ignore_patterns("README.md", "README_zh.md", "__pycache__", ".git"),
        )
        content = readme.read_text().replace("](#", "](README.md#")
        (target_dir / "README.md").write_text(content)


def setup(app):
    # ensure examples are synced before reading source files
    app.connect("builder-inited", _sync_examples)
    app.connect("html-page-context", _example_source_context, priority=499)
    app.connect("html-page-context", _experiment_home)


def _experiment_home(app, pagename, templatename, context, doctree):
    if pagename == "index" and app.builder.format == "html":
        context["doc_layout"] = os.environ.get("SLIME_DOC_LAYOUT", "root")
        return "experiment.html"


def _example_source_context(app, pagename, templatename, context, doctree):
    """Point copied example pages back to their original repository source."""
    if not pagename.startswith("_examples_synced/"):
        return
    example_name = pagename.split("/")[1]
    source_dir = Path(__file__).resolve().parent.parent / "examples" / example_name
    readme_name = "README.md"
    if app.config.language == "zh" and (source_dir / "README_zh.md").exists():
        readme_name = "README_zh.md"
    context["original_readme"] = readme_name
    context["edit_page_provider_name"] = "GitHub"
    context["edit_page_url_template"] = (
        "{{ github_url }}/{{ github_user }}/{{ github_repo }}/edit/{{ github_version }}/"
        "examples/{{ file_name.split('/')[1] }}/{{ original_readme }}"
    )


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
        "An LLM post-training framework for RL scaling.",
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
    "torch": ("https://docs.pytorch.org/docs/stable", None),
}

nbsphinx_prolog = """
.. raw:: html

    <style>
        .output_area.stderr, .output_area.stdout {
            color: #d3d3d3 !important; /* light gray */
        }
    </style>
"""
