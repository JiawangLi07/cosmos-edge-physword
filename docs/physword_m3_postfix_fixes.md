# M3 定向修复与复测

日期：2026-11-01。分支 `physword/m3-postfix-fixes`，起点 `00508fc2388857af71b7305a64fe57ed3d8abd6e`。依据 [M3 postfix review](physword_m3_postfix_review.md) 的 P1-R1/R2/R3，本轮只修复跨进程恢复、真实 T06 和预处理协议。

## 1. 本轮结果

- scheduler 内部无序元数据已从行为协议中排除，实际 scheduler 参数继续严格比较。
- 默认 K=4 解码改用 **`decode_mode="recompute"`**，每步完整因果 UND forward；三个真实窗口逐层与 Joint MoT 比较均为 **0 差异**，没有放宽现有容差。
- RGB resize、插值、量化、输入尺寸、像素及 VAE latent 归一化由同一可执行配置控制，并进入 checkpoint；额外固定了像素归一化执行设备。配置或视觉 processor 发生变更会拒绝 Resume。
- **158 项模型测试、58 项数据测试全部通过，0 skip**。新测试包含独立进程完整 Tiny A+B+C 的下一步恢复，不是只验证能 load 文件。

真实完整 Phase B 的显存阻塞保持原有状态；本轮没有启动 M4、正式拟合、多卡/FSDP，没有下载权重。缓存 AR 路径保留为显式诊断选项，其真实 T06 不作为已修复通过；默认 recompute 路径完成验收。

## 2. 跨进程恢复：修复与证据

位置：`cosmos_framework/model/generator/physword/checkpoint.py:28`，`training_protocol()`。

先前 Diffusers 的 `_use_default_values` 由 set 生成列表，随着 `PYTHONHASHSEED` 改变顺序。现在只排除这一个内部字段，保留完整已解析的 scheduler 参数并记录 scheduler class。没有对所有列表排序，因此词表顺序、层序、sampler order 的严格性保持。单测验证内部字段顺序不影响协议，而实际 shift 改变会影响协议并被拒绝。

新入口 `scripts/physword_resume_probe.py` 在两个独立 Python 解释器中分别使用 `PYTHONHASHSEED=1/2`：

1. 随机原生 Tiny 执行一次完整 A+B+C 更新，保存 checkpoint，然后不中断地执行下一次随机 RF/noise/样本更新并记录 oracle。
2. 新进程重建模型/processor/optimizer/scheduler/EMA/scaler/sampler，加载 checkpoint 后执行相同下一步。
3. 验证下一步三项 Loss、**全部模型参数和 optimizer state**、scheduler、EMA、scaler、Python/NumPy/Torch CPU/CUDA 随机数、sampler状态 exact 相同。

测试位于 `physword/postfix_test.py::test_independent_process_abc_resume`，最终全量 pytest 已重新执行。小型结果见 [tiny_process_resume_final.json](physword_m3_postfix_fixes_results/tiny_process_resume_final.json)。

另有真实 Edge PW/head-only Phase A 独立进程测试：`physword_real_probe.py --resume-in-new-process` 保存 step1 与不中断下一步 oracle；`physword_real_resume_probe.py` 在另一个解释器加载后，比较 logits、AR IDs、单步 velocity、下一真实窗口 CE/latent Loss、PW/head 参数更新、scheduler 和 Torch CPU/CUDA RNG。该测试不是完整真实 Phase B，也没有启用真实 EMA/scaler。

真实结果见 [real_process_resume_final.json](physword_m3_postfix_fixes_results/real_process_resume_final.json)。保存与恢复使用不同 hash seed，实际数据 windows SHA 在新进程重新核验，未重新初始化 PW 语义权重。

## 3. 真实 T06：定位、修复与边界

位置：`physword/config.py:40`，`inference/physword.py:37`，`inference/physword_diagnostics.py:70`。

原真实窗口 0 的 cached AR→full UND prediction hidden 差：max abs=1.61962890625、RMS=0.1574984342、相对RMS=7.1204945%。本轮做了受控诊断：

| 诊断 | 相对 RMS 差 | max abs |
|---|---:|---:|
| 原 cached decode | 0.0712049454 | 1.61962890625 |
| 仅补齐 decode attention 的 query 长度 | 0.0712049454 | 1.61962890625 |
| 仅将 UND 单行 Linear 输入复制到128行，计算后取首行 | **0** | **0** |
| 每步完整前缀重算 UND，不复用 KV | **0** | **0** |

复制 Linear 输入保持每行数学运算相同，但改变单行/多行 BF16 计算形状；上述干预定位到单 Token Linear 的数值路径，不能把它归为 PW position/off-by-one 或 repack 看不到 PW。未做底层 CUDA kernel profiler，因此不指定未经记录的具体 cuBLAS kernel 名称。诊断文件：[cache_diagnosis.json](physword_m3_postfix_fixes_results/cache_diagnosis.json)、[cache_diagnosis.py](physword_m3_postfix_fixes_results/cache_diagnosis.py)。复制行及 attention padding **只在诊断脚本里**，没有进入生产 MoT。

生产路径采用完整前缀重算：

```text
shared frozen RGB prefix + start
  → full causal UND → greedy w1
prefix + start + w1
  → full causal UND → greedy w2
prefix + start + w1,w2
  → full causal UND → greedy w3
prefix + start + w1,w2,w3
  → full causal UND → greedy w4
prefix + start + w1,w2,w3,w4
  → full causal final flush
  → 原有 fresh Token Repacking → Joint MoT
```

仍是四个词、greedy、no-grad B 分支；只使用前序预测词，不使用 GT/future。历史视觉编码不重复提取。默认不创建缓存；当 `capture_cache=True` 时，最后完整 flush 产生一次性的只读诊断快照，既不用于增量解码，也不传给 GEN。

`compare_ar_repack()` 对 recompute 取最后完整 flush 的逐层数据，与 joint UND 比较；还**独立比较真正用于四次 argmax 的 prediction hidden states**，避免仅比较两个同形 full forward 就冒称 AR 一致。既有 `atol=rtol=0.02` 未改。

真实 Edge + 真实历史RGB（显式64²），GEN使用合成固定形状 latents以隔离 VAE，在三个 Train 窗口执行原生 Joint Forward：

| Window | 预测 class IDs | 141项比较的最大误差 |
|---|---|---:|
| episode0001 / 000000 | [4,4,4,4] | **0** |
| episode0001 / 000001 | [0,4,3,3] | **0** |
| episode0001 / 000100 | [4,4,2,4] | **0** |

141项=28层×input hidden/层末hidden/raw K/raw V/GEN-normalized K，加最终 prediction hidden。合计423项比较0差，不仅是同一 `[lift]*4` 轨迹。结果：[real_t06.json](physword_m3_postfix_fixes_results/real_t06.json)。

`decode_mode="cached"` 仍可显式调用用于诊断和后续优化，当前不作为真实训练默认。默认 recompute 额外消耗 UND 算力；K=4 有界可用，尚未完成长期性能评测。没有 Direct KV Handoff 或 Final-z Injection。

## 4. 预处理协议：执行与校验统一

新增 `physword/preprocessing.py:11`，`PreprocessingConfig` 为不可变配置，规定：

- 原始 TCHW uint8 RGB 输入的 `input_size`，以及可选 `output_size`。
- bilinear/bicubic、align_corners、antialias、round/floor 量化；无crop、无随机augmentation。
- 像素公式 `float32(rgb)/pixel_scale + pixel_offset`，并固定执行设备 cpu/cuda。
- VAE raw posterior-mode latent；统一计算 `(raw-latent_mean)/latent_std`，每通道48项。

`PhysWordTrainingStep(..., preprocessing=...)` 现在要求显式配置，Step 内执行历史/未来相同 resize。历史RGB仍只进入Reasoner条件，未来RGB仍只进入监督。外置二次resize后传入时，原始输入尺寸不匹配会报错。`encode_video` adapter 的接口明确改为返回**未归一化 posterior-mode latent**，由 Step 统一归一化；所有当前调用点均已更新，避免重复归一化。

`training_protocol()` 保存上述配置，以及实际 image/video processor 的完整配置。原 checkpoint 中的 processor 文件哈希继续校验。真实 probe 原始输入480×640，实际输出64²，RF resolution策略仍为480/shift5，两者分开记录。

本轮验证以下变更全部拒绝恢复：原始输入尺寸、输出尺寸、interpolation、align_corners、antialias、quantization、pixel_scale、pixel_offset、pixel_device、latent_mean、latent_std；image和video processor 的 image_mean变化也分别拒绝。执行测试验证了像素公式和非identity latent normalization。

像素归一化的执行设备也会影响舍入：开发中将原GPU计算移至CPU时，真实VAE输出发生微小变化。最终真实配置固定 `pixel_device="cuda"` 后，Native/Matched/Full未来velocity RMS恢复为0.5397911668 / 1.0487759113 / 1.0487374067，与原审查一致；PW-off差0、干预相对差2.0893144%、重复floor0、完整suffix masked替词差0。最终证据：[real_edge_final.json](physword_m3_postfix_fixes_results/real_edge_final.json)。中间候选结果保留为历史诊断，不作为最终配置证据。

Checkpoint 升为 **schema3**。它同时严格保存/检查解码策略、预处理、RF、网络配置、词表、初始化、optimizer等现有完整状态。旧schema2缺少这些必要协议，明确拒绝直接精确Resume，不静默猜测或迁移；旧文件保留。若后续需要从旧状态恢复权重，应另行定义显式 warm-start，不能当精确续训。

## 5. 本轮测试、Loss 与更新

最终模型测试：**158 PASS / 0 FAIL / 0 SKIP**；数据：**58 PASS / 0 FAIL / 0 SKIP**。初版整合时发现两个调用点尚未补齐新的必填 preprocessing 参数，已更新并重新运行最终全套；`initial_tests.log`保留首次结果，最终只看`model_tests.xml/log`。

- RF 三个非整数sigma的noising、time embedding、video loss和关键梯度仍 exact；原生RF源码未修改。
- T01/T02/T03/T05/T06/T07/T08/T09/T10/T11/T13/T14 相关 Tiny 回归重跑；新增两种decode mode Tiny parity及“recompute不复用cache”调用检查。
- 三项独立梯度、实际A/B optimizer更新和frozen检查继续通过；Checkpoint同进程累积梯度/EMA/scaler完整恢复也重跑通过。
- T12 CFG/Joint cache、FSDP/AC、完整真实B以及480p正式训练没有扩大验证。

Synthetic原生Tiny（V16,D128）三项 Loss：video_raw=1.2602672577、video=12.6026725769、CE=2.6092789173、latent=1.0144643784、total=16.2264156342。独立video梯度 group norm：PW0.12960735、UND1.87047524、GEN8.27421873、UND→GEN K norm0.00893324、head0；CE/latent反传路径与原图一致。完整参数名、grad/update norm和冻结检查见归档的 `physword-m3-gradients.json`、`physword-m3-step-A.json`、`physword-m3-step-B.json`。

真实V7标注+真实RGB+随机Tiny A/B单步也重新执行，见 [real_data_tiny.json](physword_m3_postfix_fixes_results/real_data_tiny.json)。这是数据接口/更新验收，不是学习收益。

真实Edge的PW/head-only A：CE=2.3978393078，latent=0.9946485758，total=3.3924880028；table/head grad norms=304.3778076/56.7093277，update norms=0.0078358883/0.0119733335；其余UND/GEN/visual参数未变。真实VAE history前缀差仍为0。数据、词表、真实权重来源与原固定revision保持。

## 6. 改动位置

| 文件 | 主要改动 |
|---|---|
| `physword/checkpoint.py:28` | 稳定scheduler行为协议、保存预处理及visual processor、schema3 |
| `physword/preprocessing.py:11` | 可执行RGB/VAE配置与输入尺寸检查 |
| `physword/training.py:101` | 显式preprocessing参数，Step统一执行变换及latent归一化 |
| `physword/config.py:40` | 默认recompute，cached显式诊断模式 |
| `inference/physword.py:37` | K=4完整前缀重算与最终flush |
| `inference/physword_diagnostics.py:70` | 区分recompute快照与增量cache的观察布局，保留prediction states实测 |
| `physword/postfix_test.py` | 独立进程、预处理反例、processor漂移、scheduler、两种decode模式 |
| `scripts/physword_resume_probe.py` | 两进程Tiny完整A+B+C续训对照 |
| `scripts/physword_t06_probe.py` | 多窗口真实权重T06验收 |
| `scripts/physword_real_resume_probe.py` | 独立解释器真实PW/head下一步恢复 |
| `scripts/physword_real_probe.py` | 执行共享预处理，写独立进程oracle，保持有界真实A验收 |
| `scripts/physword_m3_probe.py`、`physword_task_probe.py` | 更新为显式预处理和raw VAE adapter接口 |
| `physword/pre_m4_test.py`、`training_test.py` | 更新相关构造调用，不放宽断言 |

`mot/attention.py`、`mot/unified_mot.py`、Token Repacking的`sequence.py/packers.py`、原版默认训练配置没有代码改动。

## 7. 可复现命令

工作目录为仓库。所有保存类测试重跑时选择新的输出目录。

```bash
export I4_ATTN_BACKENDS=flash3 COSMOS_TRAINING=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=. OMP_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=7 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /tmp/physword-m01-venv/bin/python -m pytest \
 -c tools/physword_m01_pytest.ini \
 cosmos_framework/model/generator/physword/{training,multimodal,visibility,rf,pre_m4,postfix}_test.py \
 cosmos_framework/model/generator/mot/checkpoint_mapping_test.py \
 cosmos_framework/scripts/physword_preflight_test.py \
 cosmos_framework/data/generator/sequence_packing/{sequence,packers}_test.py \
 cosmos_framework/model/generator/mot/include_gen_pathway_test.py \
 cosmos_framework/model/generator/algorithm/loss/flow_matching_test.py \
 -q --junitxml=docs/physword_m3_postfix_fixes_results/model_tests.xml

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /tmp/physword-data-venv/bin/python -m pytest \
 -c tools/physword_data/pytest.ini --confcutdir=tests/physword_data tests/physword_data \
 -q --junitxml=docs/physword_m3_postfix_fixes_results/data_tests.xml

PYTHONHASHSEED=1 CUDA_VISIBLE_DEVICES=7 /tmp/physword-m01-venv/bin/python \
 -m cosmos_framework.scripts.physword_resume_probe save \
 --output /aifs/lijiawang/physword/m3_postfix_fixes_results/tiny_resume_final
PYTHONHASHSEED=2 CUDA_VISIBLE_DEVICES=7 /tmp/physword-m01-venv/bin/python \
 -m cosmos_framework.scripts.physword_resume_probe resume \
 --output /aifs/lijiawang/physword/m3_postfix_fixes_results/tiny_resume_final

CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=8 /tmp/physword-m01-venv/bin/python \
 -m cosmos_framework.scripts.physword_t06_probe \
 --checkpoint /aifs/lijiawang/physword/Cosmos3-Edge \
 --task-dir /aifs/lijiawang/physword/pre_m4_fixes_results/badminton_left_release \
 --output docs/physword_m3_postfix_fixes_results/real_t06.json

PYTHONHASHSEED=1 CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=8 /tmp/physword-m01-venv/bin/python \
 -m cosmos_framework.scripts.physword_real_probe \
 --checkpoint /aifs/lijiawang/physword/Cosmos3-Edge \
 --task-dir /aifs/lijiawang/physword/pre_m4_fixes_results/badminton_left_release \
 --output /aifs/lijiawang/physword/m3_postfix_fixes_results/real_edge_final \
 --phase-a-step --checkpoint-roundtrip --resume-in-new-process
PYTHONHASHSEED=2 CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=8 /tmp/physword-m01-venv/bin/python \
 -m cosmos_framework.scripts.physword_real_resume_probe \
 --checkpoint /aifs/lijiawang/physword/Cosmos3-Edge \
 --task-dir /aifs/lijiawang/physword/pre_m4_fixes_results/badminton_left_release \
 --probe-output /aifs/lijiawang/physword/m3_postfix_fixes_results/real_edge_final \
 --output docs/physword_m3_postfix_fixes_results/real_process_resume_final.json
```

定向修复到此停止。完整真实Phase B仍需要足够空闲显存，按原审查补三项独立真实梯度、全部参数组更新及完整B恢复；本轮三项修复不替代这些门槛，也没有自动放行正式M4。
