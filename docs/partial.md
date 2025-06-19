partial rollout 开发过程中，sgl-router 需要手动重新安装

参考教程：[sglang/sgl-router/README.md at 971a0dfa32f7521c77c2eeb1180cc9a4fa0100aa · sgl-project/sglang](https://github.com/sgl-project/sglang/blob/971a0dfa32f7521c77c2eeb1180cc9a4fa0100aa/sgl-router/README.md)（用 Option A!!! B 应该是不行的）

#### Build Rust Project

```bash
cargo build
```

#### Build Python Binding

##### Build and Install Wheel

Build the wheel package:

```
pip install setuptools-rust wheel build
python -m build
```

上面的指令如果报错说 Rust 编译器版本太旧，不能构建 `icu_normalizer v2.0.0`，它需要 Rust 1.82 或更新版本，就手动升级一下

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
# 检查是否成功，应该看到 rustc 1.82.0 或更新版本。
rustc --version
```

然后重新执行上面的命令

Install the generated wheel:

```
pip install --force-reinstall dist/*.whl
```

