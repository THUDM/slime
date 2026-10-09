# Building the Documentation

Documentation lives in `docs/en/` and `docs/zh/`. Update both languages when changing shared behavior, and add new pages to each language's `index.rst`.

Example READMEs are copied into the documentation during the build. Chinese builds prefer `README_zh.md` and fall back to `README.md`. Edit the original files in `examples/`; `_examples_synced/` is generated and ignored by Git.

## Install Dependencies

Run from the repository root:

```bash
sudo apt-get update
sudo apt-get install -y pandoc
pip install -r docs/requirements.txt
```

## Build and Preview

Build both languages and serve them from a common landing page:

```bash
bash docs/build_all.sh
bash docs/serve.sh all
```

Open `http://localhost:8000` and select a language. Set `PORT` to change the preview port.

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

---

文档源文件位于 `docs/en/` 和 `docs/zh/`。修改共同功能时，请同步更新两种语言，并将新页面加入各自的 `index.rst`。

构建时会复制 `examples/` 中的 README。中文构建优先使用 `README_zh.md`，缺少时使用英文 README。请编辑原始文件，不要修改自动生成的 `_examples_synced/`。

在仓库根目录安装依赖后，运行 `bash docs/build_all.sh` 和 `bash docs/serve.sh all`，即可在 `http://localhost:8000` 预览两种语言。单独构建中文可使用 `bash docs/build.sh zh`；严格检查可追加 `-W --keep-going`。
