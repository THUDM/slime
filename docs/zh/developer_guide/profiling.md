# 性能分析 (Profiling)

在 slime 中，我们可以通过 SGLang 提供的 profiling 接口对 rollout 过程进行详细的性能分析。

## 1. 使 Rollout 进程进入等待状态 (Sleep Rollout)

为了更自由地进行压力测试和性能分析，我们通常需要让 slime 的 rollout 进程在初始化完成后进入等待状态，而不是立即开始生成。

你可以通过在启动参数中替换 `rollout_function_path` 来实现，而无需修改代码：

```bash
python train.py \
    --rollout-function-path slime.rollout.sleep_rollout.sleep \
    ... (其他参数)
```

该函数会让 rollout 进程进入无限循环等待状态，方便你手动发送请求或运行压测工具。

## 2. 获取 SGLang 引擎列表

SGLang 引擎（workers）注册在 router 上。你可以通过访问 router 的 `/workers` 接口来获取所有活跃引擎的列表。

通常 router 地址会在启动日志中打印：
```
Router launched at 127.0.0.1:3000
```

你可以使用 `curl` 查看 workers：
```bash
curl http://127.0.0.1:3000/workers
```

## 3. 使用自动化 Profiling 工具

为了简化对多个引擎同时进行 profiling 的操作，我们提供了一个自动化脚本 `tools/profile_rollout.py`。

### 启动 Profiling

默认情况下，该工具会对所有 worker 启动 profiling，并在执行 3 步后自动停止：

```bash
python tools/profile_rollout.py --router-url http://127.0.0.1:3000 --action start --num-steps 3
```

**常用参数说明：**
* `--router-url`: Router 的访问地址。
* `--num-steps`: 记录的步数，默认为 3。
* `--output-dir`: trace 文件保存目录。
* `--activities`: 监控活动，如 `GPU` `CPU`。
* `--profile-by-stage`: 是否按阶段（prefill/decode）分析。

### 手动停止 Profiling

如果你没有设置 `num_steps` 或想要提前停止：

```bash
python tools/profile_rollout.py --router-url http://127.0.0.1:3000 --action stop
```

## 4. 进行压力测试

在 Rollout 进程通过 `sleep_rollout` 处于等待状态时，你可以：
1. 使用 `tools/profile_rollout.py` 启动 profiling。
2. 使用压测工具（如 `sglang` 自带的 benchmark 工具）向 router 或直接向引擎发送请求。
3. 等待 profiling 完成（如果设置了 `num_steps`）或手动停止。
4. 在 `output_dir` 中获取 `.json` trace 文件，并使用 `chrome://tracing` 或 [Perfetto](https://ui.perfetto.dev/) 查看。
