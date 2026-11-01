# PhysWord × Cosmos3-Edge

在 NVIDIA Cosmos Framework 的原生 Cosmos3-Edge 双 Transformer MoT 上，实现 RGB 历史观测与语言指令条件下的 Physical Word 预测及未来视频训练。

```text
History RGB + Instruction
  → frozen Edge SigLIP2 / projector
  → UND predicts four Physical Words
  → fresh Token Repacking
  → native UND + GEN Joint MoT
  → future video velocity prediction
```

保留原生 MoT Attention，不新增 Transformer，不使用 Final-z Injection、Action 或 Tactile Conditioning。词表由冻结的 task manifest 确定；当前机器人 pilot 使用左臂7类词表。默认使用经过真实权重一致性验证的完整前缀重算解码。

## 当前验证状态

- M0–M3 的模型、数据接口、损失、精确参数组、checkpoint及有限步验收入口已实现。
- 最新代码回归：**172 PASS**；数据/任务测试此前 **58 PASS**。
- 真实 Edge 默认 AR/repack T06：3个窗口、423项逐层比较零差异。
- 真实 PW/head-only Phase A 参数更新、跨进程恢复通过。
- **完整真实 Phase B 的梯度、全部参数更新、跨进程恢复仍等待空卡执行，不能视为已通过。**
- 未开展 M4 正式拟合；未证明实际任务学习效果或视频质量提升。

验收使用64×64诊断尺寸；它不代表480p训练或生成质量已验证。缓存增量AR、CFG、FSDP和gradient checkpointing不在当前已验收范围。

## 代码入口

| 功能 | 位置 |
|---|---|
| PW词表、embedding/head、预训练语义初始化 | [`physword/`](cosmos_framework/model/generator/physword/) |
| RGB prefix与时序、teacher forcing、A+B+C训练 | [`multimodal.py`](cosmos_framework/model/generator/physword/multimodal.py)、[`training.py`](cosmos_framework/model/generator/physword/training.py) |
| 预测PW、fresh repack、可见性诊断 | [`inference/physword.py`](cosmos_framework/inference/physword.py)、[`physword_diagnostics.py`](cosmos_framework/inference/physword_diagnostics.py) |
| 连续RF、预处理、参数组、严格恢复 | [`rf_test.py`](cosmos_framework/model/generator/physword/rf_test.py)、[`preprocessing.py`](cosmos_framework/model/generator/physword/preprocessing.py)、[`optimization.py`](cosmos_framework/model/generator/physword/optimization.py)、[`checkpoint.py`](cosmos_framework/model/generator/physword/checkpoint.py) |
| 真实标注数据处理与冻结实验协议 | [`tools/physword_data/`](tools/physword_data/)、[`experiments/physword/badminton_left/`](experiments/physword/badminton_left/) |
| 真实完整Phase B有限步验收 | [`physword_real_b_probe.py`](cosmos_framework/scripts/physword_real_b_probe.py) |
| 集群空卡候补 | [`physword_gpu_queue.py`](tools/physword_gpu_queue.py)，当前针对121–129节点，按需配置后运行 |

训练是显式的单设备B=1 opt-in接口；原生默认SFT配方不会自动调用新增A+B+C流程。

## 本地资源和测试

基础环境参考上游 [setup](docs/setup.md)。本轮验证使用 Python3.11、Torch2.10.0+cu128、Transformers4.57.1、Diffusers0.39.0及Flash3。真实Edge资源固定到revision `344d602b128d1bbdacb43b08d0a3626f46343e29`，本地loader核验LFS SHA256及tensor keys/shapes。

仓库保存源码、测试、标注协议及小型JSON/XML证据；模型权重、运行环境、解码后的RGB数据、训练checkpoint和本地后台进程不随Git发布。真实数据/模型命令中的 `/aifs/...` 是原集群路径，其他环境应替换为自己的已核验本地资源。

下面命令使用已有环境，执行同一有限步runner的Synthetic Tiny验收：

```bash
export I4_ATTN_BACKENDS=flash3 COSMOS_TRAINING=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 PYTHONPATH=.

python -m cosmos_framework.scripts.physword_real_b_probe \
  --tiny --output /path/to/new/tiny_acceptance

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  -c tools/physword_m01_pytest.ini \
  cosmos_framework/model/generator/physword/*_test.py \
  tools/physword_gpu_queue_test.py -q
```

真实验收入口必须显式指定本地 `--checkpoint` 和冻结数据 `--task-dir`。它只执行两个不同optimizer步骤及第二步的新进程重放，并设45分钟时限；没有正式训练循环。

## 文档

- [M3当前验收](docs/physword_m3_acceptance.md)
- [三项定向修复](docs/physword_m3_postfix_fixes.md)
- [真实Phase B验收与候补队列](docs/physword_m3_real_b_queue.md)
- [M1](docs/physword_m01_report.md)、[M2](docs/physword_m2_report.md)、[M3](docs/physword_m3_report.md)历史实现报告
- [初始实施计划](docs/physword_implementation_plan.md)、[原生架构审计](docs/physword_architecture_audit.md)、[可见性根因](docs/pw_visibility_root_cause.md)，均保留其原始基线语境

本项目基于 [NVIDIA Cosmos Framework](https://github.com/NVIDIA/cosmos-framework)，保留原始提交历史、版权声明和 [LICENSE](LICENSE)。
