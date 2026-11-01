# M3 真实 Phase B 验收候补队列

日期：2026-11-01。用户已明确选择“准备完整 M3 Phase B 验收入口，空卡后自动执行有限步验证”。工作分支 `physword/m3-real-b-queue`。本轮新增诊断入口和候补工具，不修改 MoT、Token Repacking、训练 Step 或默认训练配置。

## 当前任务进度

22:12 对 10.254.30.121–129 的 GPU 进程、实际训练日志和停止规则进行只读检查：

| 节点 | 当前任务情况 | 估计剩余 |
|---|---|---|
| **127** | PI05，约21.2k/30k steps | **3.5–6.7小时，当前最接近结束** |
| 121 | 9799/41480 | 约71小时 |
| 124 | 19069/79640 | 约101小时 |
| 128 | 4190/30000 | 约104小时 |
| 129 | 1710/30000 | 约112小时 |
| 123 | 同时运行两套训练 | 无可靠统一ETA |
| 125 | A/B持续运行，实际停止模式是 `until_manual_stop` | 无自然结束时间 |
| 122、126 | GPU holder / reserve 进程 | 无自然结束时间 |

ETA由当时日志速度估计，会变化。127当前训练结束后，原项目的队列可能接续新任务，因此“当前任务最早结束”不保证该卡能够被本项目取得。没有修改、停止这些任务或其他项目的队列。

证据：[progress_snapshot.json](physword_m3_real_b_queue_results/progress_snapshot.json)。实时GPU扫描是 `live_scan/status.json`，仅表示本轮扫描快照，不是正在运行的队列状态。

## 验收入口

`cosmos_framework/scripts/physword_real_b_probe.py`，默认 `--mode run`：

1. 启动 `PYTHONHASHSEED=1` 的 save 解释器。
2. 从本地严格加载真实 UND/GEN/SigLIP2/projector/Wan VAE及冻结的左臂V7数据协议。遵循 `full_pw.json` 的 Phase B 选组：UND24–27层、PW/head、GEN与UND→GEN K norm。复用原有完整参数名选择器；先在CPU确定名单，仅所选master保留FP32，CUDA迁移后重建实际optimizer并检查名称一致。
3. 真实 Train窗口0执行同一个A+B+C图，分别对video/CE/latent反传，记录每组完整参数名、数量、LR和梯度；检查PW与UND K/V、GEN梯度、head与末层UND FFN的预期路径、冻结视觉/VAE无梯度。
4. 对total loss反传，执行第一个optimizer/scheduler更新，逐参数核对更新与冻结不变，保存schema3完整checkpoint（含VAE、optimizer、RNG、sampler和协议）。
5. 在不中断进程中，对真实 Train窗口1执行第二步随机连续RF采样/noise及更新，记录oracle。
6. 启动 `PYTHONHASHSEED=2` 的独立resume解释器，加载step1，重做同一个step2。逐张量比较model、optimizer、VAE、scheduler、sampler的shape/dtype/SHA256，并比较loss、teacher logits、predicted IDs、velocity、timestep和各随机源后续draws。必须完全一致。

只有两个不同训练步骤；第二步在恢复进程中重做一次以作对照，共执行3次optimizer update，无epoch训练循环。每项loss的独立backward只作路径诊断，不额外step。

固定实际输入480×640→64×64、B=1、10Hz、5历史+16未来、PW K4/stride4、原生连续waver RF、480 shift策略、默认recompute、FP32 masters/BF16 autocast。不是480p训练，也不是学习效果/视频质量评测。EMA/scaler关闭并显式记录，FSDP/CFG/cache仍不在本次范围。

原有 `PhysWordTrainingStep`、语义初始化、预测PW条件、原生loss scale均直接复用。未来RGB仅作监督；无GT oracle视频条件、Action/Tactile、Final-z或新增Transformer。

## 候补规则

`tools/physword_gpu_queue.py` 只依赖标准库：

- 每60秒只读查询121–129，共72张卡。相同卡连续3次满足：无compute进程、used≤1GiB、free≥70GiB、utilization≤5%。查询失败清除连续计数。
- 多卡同时符合时优先127，其次125/123/121/124/128/129/122/126；只选**一张**卡。
- 本项目队列使用共享文件锁；提交前再查一次。worker完成CPU加载后、CUDA分配前再次检查其他进程及free≥70GiB，避免把低利用率占卡程序当空闲。
- 固定独立Git checkout与完整提交号；远端启动时检查commit及tracked文件无改动，GPU按UUID绑定，所有模型加载强制离线。
- 最多等待24小时；仅提交一次实际验收尝试，成功或失败后退出。验收内部45分钟限时，外层timeout47.5分钟、SSH50分钟；超时仅清理本次子进程。
- 不分配占卡buffer。它是空闲卡轮询候补，不提供集群调度器的资源预留或抢占优先权。其他队列仍可能先取得卡；如发生加载期间竞争，记录失败并保留诊断，不启动循环重训。

共享环境：`/aifs/lijiawang/physword/envs/m3-validation`，从已验收的 `/tmp/physword-m01-venv` 原样复制，未下载权重或依赖。已在127通过Torch2.10.0+cu128、Transformers4.57.1、Diffusers0.39.0、Flash3导入与CUDA可见性检查。

## 状态和结果

后台队列目录：`/aifs/lijiawang/physword/m3_real_b_queue_results/queue_20261101/`。

- `status.json`：实时WAITING/RUNNING/PASS/FAIL/EXPIRED/STOPPED及各节点GPU快照。
- `controller.json`：PID、固定checkout、commit、启动时间；`controller.log`为控制器输出。
- `remote.log`：被选节点的验收控制器输出。
- `run/result.json`：最终验收结果；`run/save.log`、`run/resume.log`是两个独立解释器日志。
- `run/validation/independent_gradients.json`、`step1.json`：三项梯度与真实参数更新。
- `run/validation/checkpoint/`：保留完整schema3状态。
- `run/validation/oracle.json`、`resumed.json`：逐张量fingerprint与下一步对照。

固定执行checkout：`/aifs/lijiawang/physword/m3_real_b_queue_results/checkout/`。后续主工作目录的改动不会改变已排队代码。

等待时可用下列命令撤销；若已RUNNING，现有有限步任务按自身时限结束：

```bash
touch /aifs/lijiawang/physword/m3_real_b_queue_results/queue_20261101/STOP
```

如需人工重新提交，使用新的输出目录；不要复用已有checkpoint目录。当前排队不自动开展M4正式训练，也不在等待时宣称真实B已通过。

## 本轮验证

新增入口用同一runner的 `--tiny` 完整执行save/resume，三项独立梯度、全部组更新、下一步恢复的所有tensor fingerprint和观测完全一致。另测试真实路径使用的CPU master选择/部分BF16冻结/CUDA后optimizer identity，避免只测试全FP32 Tiny而遗漏真实构建分支。

队列反例覆盖：0%利用率但有进程、显存不足、无效查询、连续3次观测、SSH失败清除计数、可用节点优先选择、范围外节点排除、UUID绑定和限时离线命令。最终完整回归 **172 PASS、0 FAIL、0 SKIP，74.60秒**；Ruff check/format通过。完整测试命令和最终JUnit在 [physword_m3_real_b_queue_results](physword_m3_real_b_queue_results/)。真实B仍为等待执行。

```bash
CUDA_VISIBLE_DEVICES=7 I4_ATTN_BACKENDS=flash3 COSMOS_TRAINING=0 \
 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 PYTHONPATH=. \
 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /aifs/lijiawang/physword/envs/m3-validation/bin/python \
 -m pytest -c tools/physword_m01_pytest.ini \
 cosmos_framework/model/generator/physword/{training,multimodal,visibility,rf,pre_m4,postfix,real_b_probe}_test.py \
 tools/physword_gpu_queue_test.py \
 cosmos_framework/model/generator/mot/checkpoint_mapping_test.py \
 cosmos_framework/scripts/physword_preflight_test.py \
 cosmos_framework/data/generator/sequence_packing/{sequence,packers}_test.py \
 cosmos_framework/model/generator/mot/include_gen_pathway_test.py \
 cosmos_framework/model/generator/algorithm/loss/flow_matching_test.py \
 -q --junitxml=docs/physword_m3_real_b_queue_results/regression_tests.xml
```
