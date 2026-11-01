# M3 定向修复后的验收

日期：2026-11-01。审查分支 `physword/m3-postfix-fixes`，代码提交 `e133aa7`。对照 `physword_m3_postfix_review.md` 与 `physword_m3_postfix_fixes.md`，检查实际源码并重新执行测试。本轮没有修改模型代码、下载权重或启动 M4；只新增验收报告、诊断脚本和结果。

## 1. 结论

**三项定向修复验收 PASS。M3 整体为 CONDITIONAL GO：最小训练实现已通过，完整真实 Phase B 仍未验收。**

| 验收层次 | 当前结论 |
|---|---|
| 双 Transformer MoT、预测 PW 重打包、逐层 K/V 可见性 | PASS；保持原实现 |
| Tiny 的 A+B+C、三项独立梯度、指定参数更新、跨进程续训 | PASS，本次重新执行 |
| 真实权重加载、真实历史 RGB、默认 recompute T06 | PASS，本次重新执行 |
| 真实 PW/head-only Phase A 更新、独立进程下一步恢复 | PASS，本次重新执行，64×64 诊断尺寸 |
| 完整真实 Phase B 的独立梯度、全部参数组更新、恢复 | BLOCKED，空闲显存不足，不能写成已通过 |
| 实际任务学习效果、生成质量改善 | NOT RUN，未开展拟合 |

**M4-A：CONDITIONAL GO。** 可以准备真实数据上的有界 PW/head-only 小样本实验；启动前人工核对所选窗口，固定预处理、词表和训练预算。解冻 UND 或扩大分辨率前，补对应配置的真实单步验收。

**M4-B：NO-GO。** 必须先在足够空闲显存下完成真实完整 A+B+C 三项独立 backward、一次全部指定参数更新及跨进程下一步恢复。这里的阻塞是验证和资源缺口，不是本轮发现了新的主图数学错误。

新证据统一位于 [physword_m3_acceptance_results](physword_m3_acceptance_results/)，机器可读汇总见 [summary.json](physword_m3_acceptance_results/summary.json)。历史报告的 PASS 不计入本次实测。

## 2. 三项修复的关闭证据

### 跨进程恢复：关闭

源码 `cosmos_framework/model/generator/physword/checkpoint.py:28` 的 `training_protocol()` 仅删除 Diffusers 内部 `_use_default_values`，保留 scheduler class、实际解析参数、采样分布、shift、loss weighting。没有排序词表、sampler 或其他有语义的列表。

`postfix_test.py:22` 在两个独立解释器中以不同 `PYTHONHASHSEED=1/2` 执行 Tiny 完整 A+B+C。保存 step 1 后，分别比较不中断 step 2 与新进程恢复 step 2：

- 三项 loss、全部模型参数 max abs = **0**。
- Optimizer、LR scheduler、EMA、GradScaler、sampler 状态完全一致。
- Python、NumPy、Torch CPU/CUDA 随机数一致。

本次还重新生成了真实 Edge checkpoint，再启动独立解释器恢复。真实 PW/head-only Phase A 的 logits、单步 velocity、下一真实窗口 loss、PW/head 更新均 **max abs = 0**；预测 IDs、scheduler、Torch RNG 相同，没有重新做语义初始化。真实验证没有覆盖 EMA/scaler 或完整 Phase B，不外推这些能力。

证据：[tiny_process_resume.json](physword_m3_acceptance_results/tiny_process_resume.json)、[real_process_resume.json](physword_m3_acceptance_results/real_process_resume.json)。实现入口：`scripts/physword_resume_probe.py:20`、`scripts/physword_real_probe.py:32`、`scripts/physword_real_resume_probe.py:27`。真实状态文件位于 `/aifs/lijiawang/physword/m3_acceptance_results/real_edge/checkpoint/`。

### 真实 T06：默认 recompute 路径关闭

`physword/config.py:40` 默认 `decode_mode="recompute"`；`inference/physword.py:37` 每步只用原始历史前缀、start 和此前预测词，执行完整因果 UND。最终消耗 w4，随后 fresh repack；没有 Direct KV Handoff。

`inference/physword_diagnostics.py:70` 不仅比较最终完整 flush，还比较**实际用于四次 argmax 的 prediction hidden states**。因此本次结果不是仅重复同形完整 forward 的自我比较。

本次真实 Edge、真实 Train 历史 RGB（64×64）、合成固定形状 GEN latent 的结果：

| Train window | Greedy IDs | 逐层比较项数 | 所有项最大绝对差 |
|---|---|---:|---:|
| 000000 | `[4,4,4,4]` | 141 | 0 |
| 000001 | `[0,4,3,3]` | 141 | 0 |
| 000100 | `[4,4,2,4]` | 141 | 0 |

合计 **423 项零差异**；覆盖28层 input/hidden/raw K/raw V/GEN-normalized K，加最终 prediction states。`atol=rtol=0.02` 没有放宽。证据：[real_t06.json](physword_m3_acceptance_results/real_t06.json)。

旧 cached 真实路径并未被宣称修复。本轮审阅了修复时的单行 BF16 Linear 诊断及代码，但**没有重跑真实 cached 诊断**；历史证据仍是已知不通过，生产验收只覆盖默认 recompute。Tiny 两种模式都重新通过。后续若启用 cached，必须重新验收，不能据默认路径 PASS 放行。

### 预处理恢复协议：关闭当前训练入口缺口

`physword/preprocessing.py:11` 定义可执行不可变配置；`training.py:152` 在训练步骤内同时处理历史/未来 RGB，`training.py:175` 使用其像素归一化，`training.py:138` 统一执行 raw VAE latent 的归一化。配置实际执行并写入 checkpoint，而非仅添加说明字段。

本次重跑验证以下漂移均拒绝 Resume：input/output size、插值、align_corners、antialias、round/floor、pixel scale/offset/device、latent mean/std；image/video processor 的归一化配置变化也拒绝。执行测试验证非 identity latent normalization 和原始输入尺寸检查。

真实诊断明确记录原图480×640→64×64、bilinear、align_corners=false、antialias=false、round、uint8、CUDA 上 `float32(rgb)/127.5-1`、published Wan mean/std、posterior mode。RF resolution=480 是 shift 策略，不代表本次是480p训练。

`checkpoint.py:185,245` 使用 schema3，旧 schema2 直接拒绝；不将旧文件当作精确 Resume。证据：[real_checkpoint_manifest.json](physword_m3_acceptance_results/real_checkpoint_manifest.json)、`model_tests.xml` 中 preprocessing/processor drift 用例。

范围边界：公开的旧 `prepare_history_video_data()`（`physword/multimodal.py:291`）仍由调用者提供编码适配器，并在第295行使用固定像素公式；它不自动加载 schema3 预处理。此次真实恢复的单步视频比较走显式配置后的 repack 路径。后续 M4 评测入口需显式共用已冻结配置，不能声称所有通用 RGB 推理入口已自动获得训练预处理一致性。

## 3. 训练图、损失和参数更新

```text
原始历史 RGB + instruction
  → 执行冻结预处理 → 冻结 SigLIP2 + projector → 共享 UND prefix

A 有梯度：prefix + start + 前3个有效 GT words
  → 原生 causal UND → prediction_positions=C..C+3
  → 最终 Z_TF [1,4,D] → V类 word head → masked CE
  → 同一 Z_TF 对齐独立未来帧视觉均值 → FP32 cosine

B 无梯度：原始 prefix → K=4 greedy recompute → predicted IDs

C 有梯度：prefix + start + predicted words + EOS/BOV
  → fresh embedding lookup / Token Repacking
  → 原生双 Transformer Joint MoT，GEN读取UND逐层K/V
  → history-only clean latents + noised future latents
  → 原生 RF Video Loss × 10，一次

total = video + lambda_word*CE + lambda_latent*cosine
```

关键位置：`training.py:57` 的 teacher shift、`:161` 的 A、`:170` 的 B/C、`:224` 的原生 loss、`:234` 的单次 scale；`losses.py:10,31,46` 的冻结 target、valid CE、FP32 cosine；`optimization.py:40` 的完整名称选组。

当前 GT wk 不出现在预测它的位置；invalid 前词反馈 neutral start。未来 RGB 仅用于视觉监督和 Video RF target/noising，不进入 B 的推理条件。C 使用 B 的 IDs，重新 lookup embedding，所以 no-grad rollout/离散 argmax 不切断 C→UND/PW 的连续梯度。最后 UND 层 K/V 可接收 video 梯度；同层后续 FFN/final norm/head 没有直接 video 梯度符合计算图。

相对定向修复前 `00508fc`，MoT attention/unified_mot 和 `sequence.py/packers.py` 未改；没有新增 Transformer、Final-z、Action/Tactile 输入。冻结视觉/projector/VAE 的回归检查通过。

本次 Synthetic native Tiny（V=16、D=128）的独立损失及参数组梯度范数：

| Loss | 数值 | PW embedding | Word head | UND | GEN | UND→GEN K norm |
|---|---:|---:|---:|---:|---:|---:|
| Video（已乘10） | 12.60267258 | 0.12960735 | 0 | 1.87047524 | 8.27421873 | 0.00893324 |
| Word CE | 2.60927892 | 2.18998981 | 5.83854723 | 9.34323109 | 0 | 0 |
| Latent cosine | 1.01446438 | 1.10371351 | 0 | 6.97057306 | 0 | 0 |

Video raw=1.26026726，total=16.22641563，当前 lambda_word=lambda_latent=1。T10 中 Tiny Phase B 的组更新参数张量数分别为1/1、1/1、17/17、29/29、2/2；冻结参数及 VAE 不变。LR 分别为1e-4、1e-4、1e-6、1e-5、1e-5。完整名称、参数量及每个张量的变化在 [gradients](physword_m3_acceptance_results/physword-m3-gradients.json)、[step-A](physword_m3_acceptance_results/physword-m3-step-A.json)、[step-B](physword_m3_acceptance_results/physword-m3-step-B.json)。

真实 V=7、D=2048 的 PW/head-only A：CE=2.39783931，latent=0.99464858，total=3.39248800。

| 参数 | 参数量 | LR | Grad norm | Update norm |
|---|---:|---:|---:|---:|
| `physword.embedding.weight` | 16,384 | 1e-4 | 304.3778076 | 0.0078358883 |
| `physword.word_head.weight` | 14,336 | 1e-4 | 56.7093277 | 0.0119733335 |

其余真实模型参数保持不变；本次真实更新不包含 UND/GEN 解冻。见 [real_edge.json](physword_m3_acceptance_results/real_edge.json)。

## 4. 本次 T01–T14 状态

PASS 均指本次当前代码实测；每项的适用范围如下，不能将 Tiny PASS 移到真实权重列。

| 测试 | Tiny / Synthetic | 真实权重，本次 |
|---|---|---|
| T01 PW-off | PASS | PASS：native packing 下移除 PW 模块差0；不是全引擎原始commit的完整端到端回归 |
| T02 Token presence | PASS，含不等长 packed samples、索引 | 实际真实 repack 已运行；完整多样本索引专项 NOT RUN |
| T03 shift / future泄漏 | PASS，改当前GT/未来目标不进入条件 | 专项 GT 反事实 NOT RUN；本次没有把历史真实T03记录算成重跑 |
| T04 attention参考 | PASS，补跑原生 dispatcher 前向及Q/K/V梯度对照 | 非完整真实网络专项；单层核测试使用随机输入 |
| T05 跨样本 | PASS：M1 varlen PW 样本隔离 | NOT RUN；多模态 B>1 不在支持范围 |
| T06 AR/repack | PASS，Tiny cached/recompute | PASS：默认 recompute，3窗口423项；真实cached本次 NOT RUN，历史已知FAIL |
| T07 PW intervention | PASS | PASS：RMS差0.0219114218，相对2.0893144%，重复floor0 |
| T08 整段suffix mask | PASS，包含start/4words/EOS/BOV | PASS：masked替词差0 |
| T09 三项独立梯度 | PASS | 完整真实B BLOCKED；真实A只做总loss backward，不替代独立三项梯度 |
| T10 optimizer更新 | PASS，A/B完整组 | PASS限PW/head-only A；完整B BLOCKED |
| T11 train/infer一致 | PASS，包括FP32 master/BF16 autocast | 完整真实训练模式专项 NOT RUN |
| T12 CFG/cache | NOT RUN，M5范围 | NOT RUN；T06的UND缓存诊断不等于联合CFG/cache验收 |
| T13 视觉/VAE防泄漏 | PASS | PASS限真实Wan history/full prefix比较，差0；其他GT反事实未单独重跑 |
| T14 checkpoint | PASS，完整ABC跨进程及累积梯度等恢复 | PASS限真实A跨进程；完整B BLOCKED |

RF另行验收：sigma **0.12345、0.3746、0.82731** 的 noising、timestep embedding、velocity target、video loss、5组关键梯度与原生参考均 **max abs=0**（Tiny）。再运行历史独立脚本，对原始 NVIDIA 与当前 native sampler 的40组配置比较，全部零差；T04随机单层前向及Q/K/V梯度满足原有 abs+relative 容差0.02。未将这些写成真实大模型三sigma梯度通过。

真实 Native/Matched RGB-prefix/Full PW 的未来 velocity RMS 分别为0.5397911668、1.0487759113、1.0487374067。完整suffix屏蔽后的替词差0、重复floor0说明可测影响来自PW可见通路；**不证明词义学会，也不证明视频质量提高**。旧0.003584属于历史不同配置，不作为当前指标。

## 5. 权重、数据和剩余门槛

本次真实 loader 重新读取并核验五个大资源的 SHA256，对应模型 revision `344d602b128d1bbdacb43b08d0a3626f46343e29`。原 LFS 指针已被真实 tokenizer、UND/GEN、vision、VAE 文件替代；真实加载和 VAE 前向成功。语义初始化在 base加载后、optimizer前运行，恢复时不重做；本次检查初始化实现并重跑 Tiny 行均值测试，真实 loader 执行初始化。历史逐行独立真实语义均值比较未另行重跑，不冒称增加了这一独立证据。

任务词表仍为左臂 `[align, approach, grasp, hold, lift, lower, rotate]`，head7类、embedding8行，V=7不是早期固定16类。词表、定义和effector hash进入恢复契约。

本次重新核验433个真实窗口：Train247/Test186，episode/lineage/media fingerprint跨split无交集，473个引用RGB文件存在；目标frame/source index/timestamp逐项一致，PW和视频未来范围均1.6s，10Hz、5历史+16未来、K4/stride4。模型时钟与真实采集时间分别保存，不强行把抖动的采集PTS写成10Hz。真实Wan5/21帧→2/6latent，history前缀差0。

Train/Test无效word分别24/988、16/744。边界不确定的已知词可能保留原始class ID，**valid mask为权威**；无需全部改成-100。CE跳过、teacher feedback用start；清晰RGB仍可提供latent监督。本次真实标注+Tiny A/B单步验证word有效3、latent有效4，没有因mask捏造GT。

仍需处理的项目：

1. **P1，真实完整B验收阻塞。** 本次开始每卡约10.11GiB空闲。先前完整B参数库存的FP32权重/梯度/Adam两份moment下界24.08GiB已超过它，尚未计冻结基座和激活。本轮未尝试注定超预算的完整B。最小关闭条件是足够空闲单卡、冻结配置、小尺寸真实三loss梯度、全部组更新、跨进程恢复，随后验证目标尺寸。
2. **P2，数据人工验收未完成。** `annotation_status.reviewed=false`，有效Train标签中hold占74.79%，approach只有一个重叠窗口目标，且无Val。小样本拟合前核对所选片段，固定train内调参策略；测试集不用于选超参。
3. **P2，M4入口边界。** 当前仍是单设备B=1 opt-in step，不是默认TOML训练器的完整SFT配方。M4 loop及评测必须使用同一schema3预处理；公共旧RGB helper不会自动实现这一点。`max_train_steps=0` 的实验规格不能当作已执行训练。
4. **P2，未验证能力。** 真实cached、CFG、FSDP、gradient checkpointing、目标分辨率性能/质量仍未验收。64×64诊断通过只证明此尺寸可执行，不推断预训练生成质量。

没有发现需要再次修改已验收 Token Repacking 或原生 MoT Attention 的证据。三项修复已关闭，剩余工作应集中在上述真实验证和M4实验入口，不需要为了通过验收重构主图。

## 6. 本次命令与产物

所有模型命令工作目录为 `/aifs/lijiawang/physword/cosmos-framework`。环境固定 `I4_ATTN_BACKENDS=flash3 COSMOS_TRAINING=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=.`；没有联网下载。

```bash
# 模型回归：158 PASS，0 FAIL，0 SKIP，42.45s
CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
 /tmp/physword-m01-venv/bin/python -m pytest -c tools/physword_m01_pytest.ini \
 cosmos_framework/model/generator/physword/{training,multimodal,visibility,rf,pre_m4,postfix}_test.py \
 cosmos_framework/model/generator/mot/checkpoint_mapping_test.py \
 cosmos_framework/scripts/physword_preflight_test.py \
 cosmos_framework/data/generator/sequence_packing/{sequence,packers}_test.py \
 cosmos_framework/model/generator/mot/include_gen_pathway_test.py \
 cosmos_framework/model/generator/algorithm/loss/flow_matching_test.py \
 -q --junitxml=docs/physword_m3_acceptance_results/model_tests.xml

# 主模型仓库的测试目录包含新增task_test：58 PASS，3.08s
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /tmp/physword-data-venv/bin/python -m pytest \
 -c tools/physword_data/pytest.ini --confcutdir=tests/physword_data tests/physword_data \
 -q --junitxml=docs/physword_m3_acceptance_results/data_tests.xml

# 原始sampler与attention前向/梯度参考；只新增审查脚本
CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=1 /tmp/physword-m01-venv/bin/python \
 docs/physword_m3_acceptance_results/independent_checks.py
/tmp/physword-data-venv/bin/python docs/physword_m3_acceptance_results/data_audit.py

CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=8 /tmp/physword-m01-venv/bin/python \
 -m cosmos_framework.scripts.physword_t06_probe \
 --checkpoint /aifs/lijiawang/physword/Cosmos3-Edge \
 --task-dir /aifs/lijiawang/physword/pre_m4_fixes_results/badminton_left_release \
 --output docs/physword_m3_acceptance_results/real_t06.json

CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=1 /tmp/physword-m01-venv/bin/python \
 -m cosmos_framework.scripts.physword_task_probe \
 --task-dir /aifs/lijiawang/physword/pre_m4_fixes_results/badminton_left_release \
 --output /aifs/lijiawang/physword/m3_acceptance_results/real_data_tiny

PYTHONHASHSEED=1 CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=8 /tmp/physword-m01-venv/bin/python \
 -m cosmos_framework.scripts.physword_real_probe \
 --checkpoint /aifs/lijiawang/physword/Cosmos3-Edge \
 --task-dir /aifs/lijiawang/physword/pre_m4_fixes_results/badminton_left_release \
 --output /aifs/lijiawang/physword/m3_acceptance_results/real_edge \
 --phase-a-step --checkpoint-roundtrip --resume-in-new-process
PYTHONHASHSEED=2 CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=8 /tmp/physword-m01-venv/bin/python \
 -m cosmos_framework.scripts.physword_real_resume_probe \
 --checkpoint /aifs/lijiawang/physword/Cosmos3-Edge \
 --task-dir /aifs/lijiawang/physword/pre_m4_fixes_results/badminton_left_release \
 --probe-output /aifs/lijiawang/physword/m3_acceptance_results/real_edge \
 --output docs/physword_m3_acceptance_results/real_process_resume.json
```

Tiny独立进程save/resume由pytest用不同hash seed自动启动，最终JSON已归档。上述产生checkpoint或JSON的命令复跑需选择新目录，避免覆盖证据。

本轮另在数据worktree `cosmos-framework-data@e89480a` 跑了其原有55项并通过；最终58项来自主模型仓库多出的3项task测试，不把两次执行相加。独立数据审查首次错误假设所有invalid必须为-100，已按实际mask契约修正审查脚本后重跑；未修改数据或模型去满足这一假设。

本次仅有已有requests版本/deprecation等警告，没有测试失败或skip。真实大checkpoint保留在仓库外的验收输出目录，小JSON/XML归档于报告旁。审查结束，不启动后续训练。
