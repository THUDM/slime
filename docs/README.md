# slime Documentation

We recommend new contributors start from writing documentation, which helps you quickly understand slime codebase.
Most documentation files are located under the `docs/` folder.

## Docs Workflow

### Install Dependency

```bash
apt-get update && apt-get install -y pandoc parallel retry
pip install -r requirements.txt
```

### Update Documentation

You can update the documentation in the en and zh folders by adding Markdown or Jupyter Notebook files to the appropriate subdirectories. If you create new files, make sure to update index.rst (or any other relevant .rst files) accordingly.

## Build and Render

```bash
# build english version
bash ./build.sh en
bash ./serve.sh en

# build chinese version
bash ./build.sh zh
bash ./serve.sh zh
```

You can then visit `http://localhost:8000` to view the documentation.