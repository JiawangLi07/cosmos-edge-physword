# PhysWord × Cosmos3-Edge M3：损失、梯度与更新验收

2026-11-01，在 `physword/m0-m1-visibility` 分支基于 M2 `f2a1e1d` 增量实现。

**M3 单设备训练能力通过 Tiny/Synthetic 验收：118 passed，0 failure/error/skip。** 同一个 Phase B 步骤完成有梯度 Teacher Forcing、无梯度预测 PW Rollout、预测 PW 条件下的有梯度原生 Joint MoT Video Loss，并执行实际 AdamW 更新。Phase A、独立损失 Backward、冻结参数、完整 Suffix Mask-out、Checkpoint/Tokenizer/Optimizer 恢复均已实测。

**这不是预训练 Edge 或真实数据训练成果。** 主模型、SigLIP2、VAE 和 tokenizer 仍有五个唯一 LFS 指针文件，真实小 Batch Forward/Backward 无法执行。Tiny 主模型及视觉/VAE 权重随机，RGB 与 Word 标注明确为 Synthetic。没有开展数据拟合曲线、大规模训练、M4 消融或 M5 并行优化。

## 1. M2 前置验收与 T08

先执行原 M0–M2 的 93 项回归及新增的 Mask-out 检查，**94/94 通过后才进入训练验证**。[前置 JUnit](physword_m3_results/m2_gate_tests.xml)。最终合并测试再次覆盖全部这些检查。

在与 M2 一致的 BF16 Tiny、固定 RGB/Instruction/Noise/Sigma/Weights 下，仅替换预测序列的第二个 PW：

| 指标 | 实测 |
|---|---:|
| 原始未来 Velocity RMS | 0.9756876230 |
| PW 替换的未来 Velocity RMS 差异 | 0.00358431763 |
| 相对差异 `RMS(v_a-v_b)/RMS(v_a)` | **0.00367363237 = 0.367363%** |
| 相同输入重复 Forward 的 RMS numerical floor | **0** |
| Masked 条件下交换 PW 的 RMS 差异 | **0，逐元素相等** |
| Masked 重复 Forward 的 RMS floor | 0 |
| Visible 与 Masked 的 RMS 差异 | 0.00551506784 |

[完整指标](physword_m3_results/visibility.json)。这是当前硬件、kernel 与 eval 状态下的 floor，不承诺其他硬件也为零。

新增 [mask_physical_word_suffix():109](../cosmos_framework/inference/physword_diagnostics.py#L109) 是 **B=1 临时诊断 context manager**：每层保留 native UND causal 结果，仅为 GEN 使用现有 PyTorch SDPA 计算 `[C keys; GEN keys]`，排除 `start+w1+w2+w3+w4+EOS+BOV` 全部七个后缀位置，本例 UND-local `81..87`。保留 Query、Token 数、mRoPE、GQA 数学及权重；物理删除这些 keys，未用 K/V 置零冒充 mask。退出后恢复所有 dispatcher，正式训练不会使用此诊断入口。

M2 的 T03/T06/T11/T13 和 M1 visibility/intervention/sample isolation 全部保留通过。最新 [M2 layerwise parity](physword_m3_results/m2_parity.json) 的 Hidden、Raw K/V、GEN-normalized K、prediction hidden 最大误差仍全部为 0。 另补测了 M3 实际 FP32 master + BF16 autocast 下的 no-grad AR / grad-enabled Joint parity，以上逐层最大误差也全部为 0：[AMP parity](physword_m3_results/amp_parity.json)。

## 2. 代码位置与改动边界

| 新增/修改文件 | 主要职责 |
|---|---|
| [training.py:24](../cosmos_framework/model/generator/physword/training.py#L24) | `LossConfig`：可配置 λ_word、λ_latent、native video_loss_scale、原生 reduction 开关。 |
| [training.py:36](../cosmos_framework/model/generator/physword/training.py#L36) | `PhysWordBatch`：历史、指令与独立的未来 RGB/GT labels/valid masks；固定 B=1、K=4。 |
| [training.py:57](../cosmos_framework/model/generator/physword/training.py#L57) | `forward_teacher_words()`：有梯度错位输入及 prediction-position gather，仅调用 16 类 Word Head。 |
| [training.py:96](../cosmos_framework/model/generator/physword/training.py#L96) | `PhysWordTrainingStep`：同一步 A/B/C、VAE/noising、原生 RF loss 与三项加权汇总。 |
| [losses.py:11](../cosmos_framework/model/generator/physword/losses.py#L11) | `encode_future_targets()`：逐目标帧 native image processor→冻结 SigLIP2/projector→空间均值。 |
| [losses.py:28](../cosmos_framework/model/generator/physword/losses.py#L28) | `word_ce()`、`latent_cosine()`：独立 valid mask、FP32 loss、有效计数与 invalid target 处理。 |
| [optimization.py:13](../cosmos_framework/model/generator/physword/optimization.py#L13) | `PhaseConfig`、`build_phase_optimizer()`：精确参数全集，Phase A/B 冻结与 LR；不使用 substring。 |
| [optimization.py:135](../cosmos_framework/model/generator/physword/optimization.py#L135) | `snapshot_parameters()`、`optimizer_audit()`：每个完整参数名、numel、LR、grad norm、实际 delta，以及冻结参数逐元素检查。 |
| [checkpoint.py:17](../cosmos_framework/model/generator/physword/checkpoint.py#L17) | 私有 PW mapping；严格 base warm start、保存与恢复 model/optimizer/RNG/native processor/tokenizer/manifest。 |
| [conditioning.py:97](../cosmos_framework/model/generator/physword/conditioning.py#L97) | PW 共享 lookup 在 CUDA autocast 下输出计算 dtype，以支持 FP32 master 参数和 BF16 RoPE/QKV。未改变非 autocast M1/M2 行为。 |
| [flow_matching.py:13](../cosmos_framework/model/generator/algorithm/loss/flow_matching.py#L13) | 仅将 `RectifiedFlow` 的类型注解 import 放入 `TYPE_CHECKING`，修复直接导入 RF 暴露的已有循环依赖；**没有改变 loss 数学**。 |
| [physword_diagnostics.py:109](../cosmos_framework/inference/physword_diagnostics.py#L109) | T08 临时诊断 mask，生产 attention dispatcher 无变更。 |
| [training_test.py:46](../cosmos_framework/model/generator/physword/training_test.py#L46) | 17 项 M3 验收测试，包括参数化 Phase / corruption 情况。 |
| [physword_m3_probe.py:32](../cosmos_framework/scripts/physword_m3_probe.py#L32) | 可执行的单次 Synthetic Tiny 更新及持久化 checkpoint；没有训练循环参数。 |

M2 的 `packers.py`、`sequence.py`、`cosmos3_vfm_network.py`、`multimodal.py`、`inference/physword.py`，原生 `attention.py`、`unified_mot.py` 及默认 Edge SFT recipe 均与 M2 字节相同。[哈希核查](physword_m3_results/summary.json)。没有重构 Repacking、RGB Prefix 或 MoT Attention，也没有新增 Transformer、Final-z、Action/Tactile 条件。

所有 PW 可训练参数已在原 `net.physword` 构造阶段注册；M3 没有动态添加可训练模块。Optimizer 通过 `network.named_parameters()` 的完整名字建立组，后续 FSDP 注册仍属 M5，本版遇到非空 `parallel_dims` 明确拒绝。

## 3. 同一步 A+B+C 数据流

```text
已观测 RGB [5,3,H,W] + Instruction
  -> 原 M2 build_reasoner_prefix()，模板/视觉预处理一次
  -> C + start；历史视觉特征冻结，真实三轴 mRoPE 复用

A（grad）:
  C + start + GT(w1) + GT(w2) + GT(w3)
  -> 原 UND reasoner_forward(cache=None)
  -> 在 C,C+1,C+2,C+3 取最终 norm 后 Z_TF [1,4,D]
  -> 原 16 类 word_head -> logits [1,4,16] -> masked Word CE
  Z_TF + future-only pooled target [1,4,D] -> FP32 masked cosine loss

B（no_grad，临时 eval）:
  原始 C + start -> 原 M2 generate_physical_words() -> 四个 greedy PW IDs
  -> 恢复各子模块原先 mode；不计算训练 loss；不读取 A 的 GT suffix

C（grad，Phase B 才执行）:
  同一个 C + 预测 PW + EOS/BOV -> 原 M2 repack_physical_words()
  GEN: history-only VAE clean latents + noisy future latents
  -> 原生 two-way Joint MoT -> Video Velocity
  -> 原生 compute_flow_matching_loss() -> 一次 video_loss_scale

total = L_video + lambda_word*L_word + lambda_latent*L_latent
total.backward() -> 同一个 optimizer.step()
```

D 为已有 UND hidden size，目标 Edge 配置为 2048；本次 Tiny D=128。没有用 LM 全词表 logits。Teacher 输入到 `GT(w3)` 为止，预测 w4 的输入是 w3；GT(w4) 仅为 target。GT 无效的前序反馈用现有 `pw_start` embedding 作中性占位，不伪造一个有效动作类；仍按当前 valid mask 计算监督。

未来视觉目标读取独立 future 容器的 `[3,7,11,15]`，对应完整 clip 的 `[8,12,16,20]`。每帧单独走 **原生 image processor、SigLIP2 与 PatchMerger/Projector**，随后对 merged spatial tokens 作 FP32 mean。没有给目标编码加入历史/其他未来帧，未添加 target projection。Tiny 目标为 `[1,4,128]`，`requires_grad=False`。

Latent Loss 在 FP32 中先选择 valid、finite、非零 target，再计算 cosine；不会通过 `0*NaN` 污染结果。全 invalid 返回可微零和 count=0；有效 prediction 出现非有限值会报错。Word mask 与 latent mask 独立。Phase A 全无监督的 batch 在调用方应跳过 optimizer step，避免纯 weight decay 更新；测试验证零 loss/安全反馈及 backward，不将这样的 batch 算作学习。

VAE 被 eval/freeze，所有 encode/noising 均在 no_grad 内。分别编码五帧历史和完整 21 帧 clip；**条件始终取 history-only 的两帧 latent**，完整 clip 仅提供未来四帧 clean latent target。RF 使用传入的原生 `RectifiedFlow` scheduler/interpolation/time weight：

```text
xt_future = sigma*noise + (1-sigma)*x0_future
v_target = noise-x0
GEN condition_mask = [1,1,0,0,0,0]
```

本次固定原生 scheduler 第 500 项（shift=3），不是用任意 sigma 冒充 scheduler timestep。未传 timestep/noise 时，入口使用原生 sampler 和当前 RNG。维持 Edge `normalize_loss_by_active=False`：历史帧误差乘零，分母仍包含完整六帧。调用 native helper 得未缩放 loss 后仅乘一次 10，不再调用外层 `_compute_losses()` 叠加 scale。独立测试验证了插值、clean history、分母及 scale。

## 4. 三项 Loss、独立 Grad Norm 与参数更新

所有数值来自 **随机 Tiny + Synthetic RGB/GT，单步、未 clip**。λ_word=λ_latent=1、video_loss_scale=10、AdamW weight_decay=0.01；FP32 master 参数、BF16 autocast，避免 BF16 参数下小 LR 更新被舍入吞掉。

| Loss | Phase A | Phase B |
|---|---:|---:|
| Native Video RF，未缩放 | 0 | 1.2602672577 |
| L_video，已乘一次 10 | 0 | 12.6026725769 |
| L_word | 2.6092789173 | 2.6092789173 |
| L_latent | 1.0144643784 | 1.0144643784 |
| L_total | 3.6237432957 | **16.2264156342** |

每个 batch 有 4 个 valid word blocks、4 个 valid latent blocks，均来自 1 个样本。Phase B 有 1 个 video sample、3072 个有效未来 velocity elements；Phase A 不运行 B/C、VAE encode 或 video loss。

T09 对以下三个 loss **各自清空梯度、独立 backward**，不是用 total 的梯度推断路径：[全参数 Grad Norm](physword_m3_results/gradients.json)。组范数为参数 grad 的全局 L2。

| 独立 backward | PW Embedding | Word Head | UND | GEN | GEN 的 UND K Norm |
|---|---:|---:|---:|---:|---:|
| L_video | **0.12960735** | **0** | **1.87047524** | **8.27421873** | **0.00893324** |
| L_word | 2.18998981 | 5.83854723 | 9.34323109 | 0 | 0 |
| L_latent | 1.10371351 | 0 | 6.97057306 | 0 | 0 |

Video-only 明确验证最后一层 UND `k_proj/v_proj` 和 `k_norm_und_for_gen` 有梯度，PW Embedding 有梯度；Word Head、最后 UND FFN、最终 UND norm 无该分支梯度。后者由 CE/Latent 覆盖，符合原生 GEN 在每层读取当前 K/V 的实现。Argmax/预测 IDs 无梯度；没有 straight-through estimator。视觉塔、projector、VAE、原 base embeddings 和原 LM head 的梯度均为空。

Phase B 合并 loss 一步后的实际更新如下；Tiny 解冻 UND layers `[0,1]` 与最终 UND norm：

| Exact group | numel | LR | total Grad Norm | 参数变化 L2 | 改变 tensors / 本组 tensors |
|---|---:|---:|---:|---:|---:|
| `pw_embedding` | 2176 | 1e-4 | 2.34845257 | 0.0022626910 | 1/1 |
| `word_head` | 2048 | 1e-4 | 5.83854723 | 0.0045254845 | 1/1 |
| `und` | 230016 | 1e-6 | 10.79916950 | 0.0004524483 | 17/17 |
| `gen` | 329024 | 1e-5 | 8.27421873 | 0.0057264615 | 29/29 |
| `gen_und_k_norm` | 64 | 1e-5 | 0.00893324 | 0.0000800249 | 2/2 |

每个完整参数名、numel、grad norm、update norm、changed、组 LR 均在 [Phase B 参数审计](physword_m3_results/step-B.json) 中。Phase A 的三组也全部更新：PW table delta=0.0019595386、head=0.0045254845、UND=0.0004524811；[Phase A 全量审计](physword_m3_results/step-A.json)。所有冻结参数及独立 VAE 参数与更新前逐元素一致。

选参规则是显式模块路径收集后转为完整参数名，组之间判重，并核对 optimizer 参数 identity。UND 只收集配置层的普通 `q_proj/k_proj/v_proj/o_proj`、原 UND MLP/norm；GEN 收集显式 `_moe_gen` 模块及 `time_embedder/vae2llm/llm2vae`。`k_norm_und_for_gen` 独立组。测试确认选 UND `q_proj` 不会带上 `q_proj_moe_gen`，B0 可冻结所有 UND/PW 仅训练 GEN 与 K Norm。选参后再建 optimizer；改变 phase/层集合后，旧 step 拒绝继续使用，需重建 step/optimizer。

另有测试在整个 UND 冻结时，PW Embedding 仍获得有效反传；没有把整个 A forward 包在 no_grad 内。Phase A 不包含 GEN 与 K Norm optimizer groups。

## 5. T08/T09/T10/T14 与保存恢复

| 验收 | 结果 |
|---|---|
| T08 | 完整七-token suffix 在每层从 GEN keys 中移除；交换 PW 后 masked velocity exact equal；visible 干预高于重复运行 floor。 |
| T09 | Video/Word/Latent 三次独立 backward，各组及关键完整参数路径符合上表；冻结教师与 base 参数无梯度。 |
| T10 | Phase A/B 各执行一次 AdamW update；全部预期组更新、冻结参数不变，每个 trainable 参数恰好在一个组。 |
| T14 | 原生 tokenizer/processor、17 个私有 PW 映射、logits、预测 IDs、单步 velocity、losses 全部恢复一致；Adam moments、CPU/CUDA RNG、下一次更新后的全 state 一致。 |
| 严格恢复边界 | 错误 mapping、tokenizer 文件改动、不同 teacher provenance 被拒绝；base warm start 仅允许缺少两个 PW 参数，非 PW 缺失和 shape mismatch 报错。 |
| M0–M2 | 原 93 项回归全部通过，包含 T03/T06/T11/T13 和 M1 原生可见性/跨样本隔离；另运行原 RF loss 的 8 项测试。 |

[最终 JUnit](physword_m3_results/all_tests.xml)：**118 passed in 13.99s**。其中 M3 17 项、原 M0–M2 93 项、native flow loss 8 项，零 skip。[Roundtrip 数值记录](physword_m3_results/roundtrip.json)。

Checkpoint 保存 `state.pt`（network，包括 frozen vision；optimizer；CPU/CUDA RNG）以及 native tokenizer 文件、两个原生 processor config、chat template、manifest。Manifest 记录 17-token 映射、16 类顺序、phase/loss config、完整 optimizer 参数名列表、base/vision/VAE/data provenance 和文件 SHA256。恢复严格验证，network shape/key 不符直接拒绝；**resume 不允许丢 PW 参数**。`load_base_state()` 是单独的 warm-start 入口，只接受已经过原生名称映射的 state dict，不承担原 HF shards→Framework remapping。

PW IDs 继续是 M1 的**模型私有映射**，没有将 HF tokenizer 扩展到 2^30 行。T14 同时恢复原 HF 基础词表和独立 PW manifest；不声称通用 `AutoTokenizer.encode('<pw_start>')` 能生成该私有 ID。恢复时由调用方按固定基座 config 先构造同架构网络，再建 matching phase optimizer，之后载入。

独立 VAE 不复制到 bundle，以固定 revision/normalization 的外部冻结资源使用；真实运行需用可核验的文件 hash/revision 替换 Synthetic provenance。本轮两个持久化示例目录：

```text
/aifs/lijiawang/physword/m3_results/phase_a/checkpoint
/aifs/lijiawang/physword/m3_results/phase_b/checkpoint
```

Phase B `state.pt` 约 7.1 MB，未加入 Git；[manifest 样例](physword_m3_results/checkpoint_manifest.json) 已入 Git。原生 tokenizer roundtrip 使用本地 Native Processor Builder，无 remote code 或权重下载。

## 6. 运行命令与 API 用法

复用 M01/M2 的隔离环境，仅补装仓库 `uv.lock` 固定的 `diffusers==0.39.0`（5.6MB 软件包），没有改系统环境或下载模型。修复的 RF 循环 import 在这个直接训练入口及 probe 中均得到实际验证。Ruff check / format check 和 `git diff --check` 均通过；剩余 pytest warning 是继承环境的 requests/urllib3/chardet 版本提示。

```bash
PY=/tmp/physword-m01-venv/bin/python
RESULTS=/aifs/lijiawang/physword/m3_results

CUDA_VISIBLE_DEVICES=7 I4_ATTN_BACKENDS=flash3 COSMOS_TRAINING=0 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PY" -m pytest \
  -c tools/physword_m01_pytest.ini \
  cosmos_framework/model/generator/physword/training_test.py \
  cosmos_framework/model/generator/physword/multimodal_test.py \
  cosmos_framework/model/generator/physword/visibility_test.py \
  cosmos_framework/scripts/physword_preflight_test.py \
  cosmos_framework/data/generator/sequence_packing/sequence_test.py \
  cosmos_framework/data/generator/sequence_packing/packers_test.py \
  cosmos_framework/model/generator/mot/include_gen_pathway_test.py \
  cosmos_framework/model/generator/algorithm/loss/flow_matching_test.py \
  -q --junitxml="$RESULTS/all_tests.xml"

# 以下每条只执行一次 Synthetic Tiny 更新；重跑请选择新的输出目录。
CUDA_VISIBLE_DEVICES=7 I4_ATTN_BACKENDS=flash3 COSMOS_TRAINING=0 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 \
"$PY" -m cosmos_framework.scripts.physword_m3_probe --phase A --output "$RESULTS/phase_a"

CUDA_VISIBLE_DEVICES=7 I4_ATTN_BACKENDS=flash3 COSMOS_TRAINING=0 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 \
"$PY" -m cosmos_framework.scripts.physword_m3_probe --phase B --output "$RESULTS/phase_b"

"$PY" -m cosmos_framework.scripts.physword_preflight \
  --checkpoint /aifs/lijiawang/physword/Cosmos3-Edge \
  --output "$RESULTS/m0_preflight.json"
```

`COSMOS_TRAINING=0` 沿用隔离运行环境；训练入口本身明确 `net.train()`、启用 autograd 并实际 backward/step，不是 no-grad 模拟更新。

真实资源准备好后，可复用以下 API 组合；本轮未启动此真实数据流程。传入已加载且 normalization 正确的 VAE 回调、原生 RF，以及一个 CPU RGB `PhysWordBatch`：

```python
from cosmos_framework.model.generator.physword.optimization import PhaseConfig, build_phase_optimizer
from cosmos_framework.model.generator.physword.training import LossConfig, PhysWordTrainingStep

phase = PhaseConfig(phase="B", reasoner_layers=(24, 25, 26, 27))
losses = LossConfig(lambda_word=1.0, lambda_latent=1.0, video_loss_scale=10.0)
# 当前单设备实现需要 FP32 master + BF16 autocast；先确认显存预算。
net.float()
optimizer = build_phase_optimizer(net, phase)
train_step = PhysWordTrainingStep(net, processor, vae, encode_video,
                                 rectified_flow, special_tokens, phase.phase, losses)
optimizer.zero_grad(set_to_none=True)
result = train_step(batch)
# Phase A 全 invalid batch 应跳过更新；Phase B 此处有完整未来视频监督。
result["losses"]["total"].backward()
optimizer.step()
```

## 7. 剩余风险、资源阻塞与 M4 建议

1. **真实 Edge 尚不能验收。** [重跑 preflight](physword_m3_results/m0_preflight.json)：metadata compatible=true、files_complete=false；八条缺失组件引用去重后仍为两个 transformer shards、vision_encoder、VAE、tokenizer.json 共五个 LFS 指针。未擅自下载。真实 tokenizer/special IDs、真实 BF16 数值范围、全模型显存、语义质量均待补齐资源后验证。
2. **正式训练数据依赖未满足。** 需要 RGB episode、instruction、正确时间戳和四个阶段 Word GT/valid masks。应先按 episode 划分训练/验证再切窗口，避免重叠窗口泄漏。当前 Synthetic label `[2,1,1,3]` 只是接口验证；没有 Toy Loss 下降曲线或真实效果结论。
3. **训练入口范围。** 当前是可复用的单设备、B=1 训练 step/API 和 single-update probe，未接默认 `scripts.train` 的 TOML/FSDP launcher、DDP、EMA、AC、CFG、cache、采样服务或 DCP/export。它可以消费真实批次，但完整数据适配和生产 launcher 接入尚待后续。默认 recipe 原样保留。
4. **显存与模式。** FP32 master + 同时保留 A/C 计算图在真实模型上需要实测预算；本机当时各 GPU 约 10GB 空闲，不能由 Tiny 的占用推断真实 Edge 可训练。未自动启用 FSDP/AC 规避。B rollout 临时 eval，不使用 dropout；A/C 遵循 train mode。恢复 master 参数做独立推理时也需 BF16 autocast，不能直接让 Flash3 消费 FP32 QKV。
5. **教师与监督质量。** Latent pooling 丢失空间细节，projector 与 UND state 同维并不意味着先验对齐。M4 需监控有效 target 数、target 方差、cosine collapse、Word 准确率与各组梯度。外部 VAE 的实际 hash、normalization 和预处理版本必须固定；manifest provenance 不能代替资源加载时的实际核验。
6. **梯度解释。** Video Loss 适配 PW 消费路径，不给离散选择或独立 Word Head 直接梯度。CE/Latent 是保持 Reasoner 能力的必要分支；不能因为总梯度非零就声称学会有意义的 Physical Words。M3 的可见性、独立梯度和实际更新共同通过，但尚未证明视频质量收益。

建议下一轮 M4 先做以下小规模设置，**这里只给建议，未运行**：

| 项目 | 建议起点 |
|---|---|
| 数据 | 约 10 个已标注 RGB episodes，按 episode 8/2 划分；32–64 个训练窗口与固定验证窗口，先人工核对四个阶段 target。 |
| 输入 | 单视角 256×256，5 history + 16 future，10Hz，stride=4，K=4，target frames 8/12/16/20。 |
| Batch | 单设备 B=1，先 accumulation=1；当前 API 的 batch 不直接扩至 ragged batch。 |
| Phase A | PW table/head + 最后四层 UND + final norm；原 Edge metadata 为 28 层，故候选 `[24,25,26,27]`，加载后再核对。先 50 steps 检查，不自动续训。 |
| Phase B | 启用 A/B/C，GEN/K Norm 解冻，保留上述 UND/PW；重建 optimizer，先 25 steps 检查验证曲线与干预。 |
| LR / loss | PW 1e-4、UND 1e-6、GEN/K Norm 1e-5；λ_word=λ_latent=1，video scale=10 一次；记录 raw/scaled losses，按梯度与验证结果调参。 |
| 稳定性 | FP32 masters/BF16 compute，建议 finite-grad 检查与 global grad clip=1；本轮表格未 clip。全 invalid Phase A batch 跳过 step。 |
| 验证 | 固定 noise/sigma/prefix 的验证 RF loss、Word accuracy、latent cosine、per-group grad/update、PW 替换相对差异/重复 floor、完整 suffix mask-out。真实权重先重跑 M3 全验收，再计训练预算。 |

在原生 sampler 接入新 PW 条件前，验证结果只称单步 Velocity / RF 指标，不称完整生成视频质量；后续再增加固定样例的完整视频采样检查。多卡、CFG、Cache 和直接 KV Handoff 不属于这个小规模训练起点。

**M3 测试与交付到此完成，未启动 M4 或长时间训练。**
