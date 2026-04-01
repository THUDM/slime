# PD 分离

slime 支持 Prefill 和 Decode 的分离部署 (PD Disaggregation)。

可以通过设置 `--prefill-num-servers` 参数来指定用于 Prefill 的服务器数量。

我们推荐在多轮或 agentic RL 训练中开启 PD 分离。
