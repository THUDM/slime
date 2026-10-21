# Building the Documentation

Documentation lives in `docs/en/` and `docs/zh/`. Update both languages when changing shared behavior, and add new pages to the matching category's `index.rst`. `library.rst` links the top-level categories.

Example READMEs are copied into the documentation during the build. Chinese builds prefer `README_zh.md` and fall back to `README.md`. Edit the original files in `examples/`; `_examples_synced/` is generated and ignored by Git.

## Install Dependencies

Run from the repository root:

```bash
sudo apt-get update
sudo apt-get install -y pandoc
python3 -m venv .venv-docs
.venv-docs/bin/python -m pip install -r docs/requirements.txt
```

The build scripts use `sphinx-build` from your current environment, or fall back to the repository's `.venv-docs`. You do not need to activate that environment to build. If neither is available, the script prints the setup commands above. To run the Python/browser checks below in this environment, first run `source .venv-docs/bin/activate`.

## Build and Preview

Build both languages and serve them from a common landing page:

```bash
bash docs/build_all.sh
bash docs/serve.sh all
```

Open `http://localhost:8000`; it opens the saved/browser language, defaulting to English. Set `PORT` to change the preview port.

To build and serve one language:

```bash
bash docs/build.sh en
bash docs/serve.sh en
# Use zh for Chinese.
```

For strict validation, pass Sphinx options through the build script:

```bash
bash docs/build.sh en -W --keep-going
bash docs/build.sh zh -W --keep-going
```

The documentation workflow publishes English at the site root and Chinese under `zh/`. Local builds made by `build_all.sh` use `en/` and `zh/`; the language toggle supports both layouts.

## Interactive lab

Sphinx renders `index` with `_templates/experiment.html`; the rest of the documentation keeps the book theme. The homepage uses local assets and plain JavaScript:

- `_static/js/experiment-engine.js`: model profiles, state validation and shell/YAML/JSON generation, with no DOM or runtime dependencies.
- `_static/js/experiment-ui.js`: bilingual controls, history, local persistence, sharing, downloads and mobile preview.
- `_static/js/experiment-diagrams.js`: choice-specific architecture and algorithm illustrations.
- `_static/js/experiment-reader.js`: in-page reading of existing Sphinx guides, with lazy MathJax and Mermaid rendering.
- `_static/css/experiment.css`: responsive styling and reduced-motion behavior.

Model profiles refer to actual `scripts/models/` and `scripts/run-*.sh` sources. Keep profile dimensions, argument validation and explanatory guides aligned with backend changes. A diagram is explanatory; it must not imply an unsupported configuration or measured performance.

Run generator tests with Node 22 or newer:

```bash
node --test docs/tests/experiment.test.cjs
```

Browser tests check both languages, reversible choices, straw/partial scheduling, choice-specific diagrams, in-page derivations, exports, categorized navigation and mobile widths. They also build the actual root/zh production layout and check assets and language links at `https://thudm.github.io/slime/`. They start their own local server and need Chromium:

```bash
pip install -r docs/tests/requirements.txt
python -m playwright install chromium
SLIME_DOC_LAYOUT=prefix bash docs/build.sh en -W --keep-going
SLIME_DOC_LAYOUT=prefix bash docs/build.sh zh -W --keep-going
python docs/tests/test_browser.py
```

Both suites are registered in the `docs-lab` matrix in `.github/workflows/pr-test.yml.j2`; regenerate `.github/workflows/pr-test.yml` when changing that job. Publishing also checks the generator and builds both languages strictly. These checks do not execute GPU training or certify memory capacity.

---

文档源文件位于 `docs/en/` 和 `docs/zh/`。修改共同功能时，请同步更新两种语言，并将新页面加入对应分类的 `index.rst`；`library.rst` 只列顶层分类。

构建时会复制 `examples/` 中的 README。中文构建优先使用 `README_zh.md`，缺少时使用英文 README。请编辑原始文件，不要修改自动生成的 `_examples_synced/`。

在仓库根目录安装依赖后，运行 `bash docs/build_all.sh` 和 `bash docs/serve.sh all`，即可在 `http://localhost:8000` 预览两种语言。单独构建中文可使用 `bash docs/build.sh zh`；严格检查可追加 `-W --keep-going`。

推荐按上方命令将文档依赖安装到 `.venv-docs`。构建脚本优先使用当前环境的 `sphinx-build`，找不到时自动使用仓库中的 `.venv-docs`，无需手动激活；尚未安装时会给出具体安装命令。运行 Python / 浏览器检查前，使用 `source .venv-docs/bin/activate` 激活该环境。

首页由 `_templates/experiment.html` 渲染，配置生成与交互分别位于 `experiment-engine.js` 和 `experiment-ui.js`。运行上面的 Node 与浏览器检查，可覆盖配置依赖、撤销/重做、脚本下载、straw/partial、双语链接和手机配图面板，并按实际发布目录验证 `/slime/` 根路径；不涉及 GPU 训练实测。
