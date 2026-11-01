# PhysWord × Cosmos3-Edge M2 实现与验收

2026-11-01。在当前 `physword/m0-m1-visibility` 分支增量实现，M1 起点 `5450c01973e6f6d80978511607c3d96051a9629d`；保留 M1 的人工 PW 通路与默认关闭行为。

**M2 的 Tiny 原生组件验收通过：93 passed，0 failed / error / skipped。其中新增 M2 19 项，原有 M0/M1 与上游回归 74 项。真实预训练 Edge 验收尚未完成，原因是权重和 tokenizer 仍为 LFS 指针。没有下载大型模型、运行优化器或启动训练。**

本轮产物是一个可调用的 B=1 RGB→PW AR→Joint MoT **单步视频 Velocity** 入口，未接入原有 Ray/Gradio 通用服务入口，也没有增加视频采样循环或解码 MP4。测试证明条件通路和对应表示的一致性，不证明随机 Tiny Model 具有物理语义预测能力。

## 实际改动与入口

| 文件 | 核心改动 |
|---|---|
| [multimodal.py:21](../cosmos_framework/model/generator/physword/multimodal.py#L21) | `PWTiming`、`RGBHistory`、`ReasonerCondition` 和隔离的 `FutureVisualTargets`。固定 5+16 帧 / 10Hz / K=4；校验视觉索引只属于历史 context。 |
| [multimodal.py:174](../cosmos_framework/model/generator/physword/multimodal.py#L174) | `build_reasoner_prefix()`：一次 native chat template / video processor，冻结 SigLIP2 + projector，调用既有 multimodal helper / `get_rope_index()`，保留三轴 mRoPE。 |
| [multimodal.py:242](../cosmos_framework/model/generator/physword/multimodal.py#L242) | `embed_reasoner_condition()`、可微 `replay_physical_words()`、只编码五帧历史的 `prepare_history_video_data()`。 |
| [conditioning.py:83](../cosmos_framework/model/generator/physword/conditioning.py#L83) | 在原 M1 `PhysicalWordEmbedding` 下注册 `word_head = Linear(D,16,bias=False)`；从既有 16 行 PW embedding 复制初始化，之后为独立参数。 |
| [config.py:8](../cosmos_framework/model/generator/physword/config.py#L8) | 更新说明，继续使用原有 enabled / token_id_base / embedding_seed；没有增加默认训练设置。 |
| [physword.py:23](../cosmos_framework/inference/physword.py#L23) | `PhysWordPrediction`、四步 greedy `generate_physical_words()`、`repack_physical_words()` 及高层 RGB 入口。 |
| [packers.py:358](../cosmos_framework/data/generator/sequence_packing/packers.py#L358) | 可选 `reasoner_condition` 分支；向 native builder 追加完整 UND span 和原始 mRoPE，随后原生构建 GEN、indexes、offsets、mask metadata。旧 M1 分支保留。 |
| [sequence.py:1393](../cosmos_framework/data/generator/sequence_packing/sequence.py#L1393) | `PackedSequence.reasoner_condition`，默认 `None`。 |
| [cosmos3_vfm_network.py:931](../cosmos_framework/model/generator/mot/cosmos3_vfm_network.py#L931) | `_encode_text()` 在 native text/PW lookup 后，将共享冻结视觉 features scatter 到 UND 视觉占位位置，再按原逻辑放入全局 buffer。 |
| [physword_diagnostics.py:14](../cosmos_framework/inference/physword_diagnostics.py#L14) | 临时观测 hooks 与 AR/repack 逐层比较；wrapper 原样调用原 dispatcher，退出后恢复，生产入口不使用它。 |
| [multimodal_test.py:50](../cosmos_framework/model/generator/physword/multimodal_test.py#L50) | T03 / T06 / T11 / T13 与端到端、干预、时间和失败边界测试。 |
| [physword_m2_probe.py:30](../cosmos_framework/scripts/physword_m2_probe.py#L30) | 固定 Seed 的 Tiny 原生 processor/SigLIP2/PatchMerger/Wan/MoT fixture 与可执行 probe。 |
| [edge_chat_template.jinja](../tests/data/physword/edge_chat_template.jinja) | 从已经存在的本地 Edge revision 复制的小型原生模板，避免测试依赖仓库外模型路径。 |

核查发现 **M1 实际只有 17 行 embedding，没有 Word Head**，所以本轮补上必要的线性读出，并未把文档的拟议 head 当作已有实现。没有实现 CE、latent loss 或语义初始化；当前 head 尚未训练。PW 私有模型 ID、16 类顺序和原 embedding 初始化完全沿用 M1，没有注册或扩容原 HF tokenizer / LM head。

`attention.py`、`unified_mot.py`、默认 `vision_sft_edge.toml` 相对于原始 `55f0554` 的字节内容均未改变，哈希在 [summary.json](physword_m2_results/summary.json)。没有新增 Transformer、Action/Tactile、Final-z Injection 或直接 KV Handoff。

## 完整调用链

```text
predict_physical_word_video_from_rgb(network, processor, history, instruction,
                                    encode_history, future_noise, timesteps, special_tokens)
  ├─ prepare_history_video_data()
  │    history RGB [1,3,5,H,W] -> caller 的已加载冻结 Wan VAE encoder
  │    -> 两帧历史 latents + 四帧 future noise -> GenerationDataClean
  └─ predict_physical_word_video()
       ├─ build_reasoner_prefix()
       │    native processor.apply_chat_template(... do_sample_frames=False)
       │    -> prepare_multimodal_reasoner_inputs()
       │    -> SigLIP2.get_image_features() -> PatchMerger/projector
       │    -> native get_rope_index() -> shared ReasonerCondition + pw_start
       ├─ generate_physical_words()
       │    native UND reasoner_forward + ReasonerKVCache
       │    -> Word Head argmax -> PW embedding feedback，固定四步
       │    -> 再消费 w4，使 cache 覆盖全部 PW；不预测 w5
       ├─ repack_physical_words()
       │    同一 Condition.with_words(...).with_boundaries(EOS,BOV)
       │    -> pack_input_sequence() -> PackedSequence.to_cuda()
       └─ network(packed)
            _encode_text(): base/PW embedding + 同一历史视觉 features
            原生双 Transformer / two_way MoT -> preds_vision
```

`generate_physical_words()` 接受已经包含 RGB + Instruction 的 typed prefix；不接受 GT words、future RGB 或 `FutureVisualTargets`。该入口复用已有 UND 参数、原生 reasoner forward 和 KV cache，而不是调用通用自由文本 `generate_reasoner_text()`。只有 Word Head 的 16 类 logits 参与 greedy，允许重复词，无文本重复惩罚。

AR 和 Joint 共用同一份视觉 features、context IDs 与 mRoPE。Condition **没有缓存普通文本/PW embedding**，所以 Joint 时重新 lookup 仍有梯度；冻结视觉 features 保持 detached。模板和 SigLIP2 各执行一次，测试显式统计了调用次数。输入已在上游重采样为 10Hz，这一版不实现视频解码 / 通用重采样器。

`encode_history` 是现有已加载 VAE 的适配回调，接收 `[1,3,5,H,W]`、范围 `[-1,1]` 的 RGB。真实运行应使用 checkpoint 自身 normalization；Tiny fixture 的 identity normalization 不能用于替代真实 Edge VAE 归一化。此入口不会隐式加载或下载 vision weights。

## Shapes、位置和 Attention

符号 `C` 为不含 pw_start 的 native context 长度，`R` 为 merged 视觉 tokens 数，`D` 为 UND hidden size。Tiny 使用两层真实 Edge MoT 类：D=128、Q heads=4、KV heads=2、head_dim=32；原生 SigLIP2 只缩小尺寸，WanVAE_ 只缩小宽度，保留原生因果时间压缩逻辑。

| Tensor | 一般形状 | 本次 Tiny |
|---|---|---|
| 历史 RGB | `[5,3,H,W]` uint8 | `[5,3,64,64]` |
| Projected visual features | `[R,D]` | `[20,128]`，每历史帧 4 tokens |
| AR prefill IDs / mRoPE | `[C+1]` / `[3,C+1]` | `[82]` / `[3,82]`，C=81 |
| 每步 feedback embedding / position | `[1,1,D]` / `[3,1,1]` | `[1,1,128]` / `[3,1,1]` |
| 输出 class IDs / token IDs | `[4]` / `[4]` | `[7,7,7,7]` / `[1073741832]×4`，随机未训练结果 |
| PW prediction hidden states | `[1,4,D]` | `[1,4,128]` |
| AR 完整 cache K / V，每层 | `[1,C+5,Hkv,head_dim]` | `[1,86,2,32]` |
| Repacked UND IDs / positions | `[C+7]` / `[3,C+7]` | `[88]` / `[3,88]`，含 EOS/BOV |
| 历史 / 全序列 VAE latents | `[1,48,2,H/16,W/16]` / `[1,48,6,H/16,W/16]` | `[1,48,2,4,4]` / `[1,48,6,4,4]` |
| 单步 Velocity | `[1,48,6,H/16,W/16]` | `[1,48,6,4,4]`，输出保留条件帧，比较未来用 `[:,:,2:]` |

本次 prediction indexes 是 `[81,82,83,84]`，实际输入的 PW token indexes 是 `[82,83,84,85]`，二者相差 1。`pw_start,w1..w4,EOS,BOV` 的三轴 mRoPE 标量为 `71..77`；**序列 index 81 不等于 mRoPE position 71**。历史视觉 prefix 的三轴 positions 存在差异，测试未关闭视觉位置编码。原生 packed positions 是 float32，但 UND 的每个位置与 AR int64 mRoPE 数值逐项精确相等。

UND `[context,start,w1..w4,EOS,BOV]` 是一个 native causal span。GEN 使用原生 two-way attention：读取全部 UND K/V 和 GEN span；UND 仅读取本 span 的因果前缀，不读取 GEN。GEN 的 UND K 采用既有 `k_norm_und_for_gen`，依旧 **RMSNorm(pre-RoPE K) 再 RoPE**；没有对 AR 的 post-RoPE cache K 直接归一化。

本次 unpadded sample length=112，split lengths=`[88,24]`，UND global indexes=`0..87`；GEN 六帧、每帧四个 patch tokens。先把 typed span 放入 builder，再构造全局 metadata，未在已完成打包的 tensor 后拼接 IDs。当前 B=1；多样本隔离沿用并复跑 M1 的两样本回归，新多模态 AR 不声称支持 B>1。

## 测试与数值证据

完整 JUnit：[all_tests.xml](physword_m2_results/all_tests.xml)。执行结果 **93 passed in 7.93s**；只有继承 Python 环境的 requests/urllib3/chardet 版本 warning。Probe 还有原有 Transformers 的 dtype deprecation / image processor size 提示，native video processor 实际输出了五帧，未用 stub 替换视觉处理。

| 验收 | 实际检查 | 结果 |
|---|---|---|
| T03 | 对每个 k 改当前 GT label，teacher replay 输入仍为 `start,w1,w2,w3`，当前预测及之前 hidden 不变；改变前序词影响后续；w4 不在该 replay 输入中。 | 四个 k 全部通过，禁止位置的 hidden 差异=0。 |
| T06 | 同一 prefix 精确核对 mRoPE；AR prefill+四次 feedback 与 joint 对齐至 w4；比较两层 block input/output hidden、Raw K/V、GEN-normalized K、四个最终 prediction hidden。 | 所有比较 max_abs=0、RMS=0，cache length=86，包含 w4。 |
| T11 | 同一 predicted IDs 从原始 prefix 重建 condition；IDs/indexes/positions/splits 精确一致，视觉 payload 复用；eval/no_grad 与 train/enable_grad 分别 forward，无 backward。 | Velocity max_abs=0，train 输出 requires_grad=True。Tiny dropout=0；未测试正式 trainer。 |
| T13 | 改 future RGB/labels 不改变 prefix features/AR；future target 参数被 AR API 拒绝；条件结构无 future targets；改历史 RGB 则改变 UND 视觉和 GEN velocity。 | 通过。 |
| T13 VAE | 实际 native WanVAE_，history-only 5 帧对 full 21 帧历史两帧 latent；对未来像素取反重做。temporal_window=4 和 68 两种路径（含尾部 padding）。 | 两种窗口历史 max_abs=0；未来 latents 确实改变。 |
| 时间 / 条件 mask | VAE callback 只收到五帧；前两帧为 condition，其余四帧为 noisy；校验 30Hz source→10Hz 已采样 metadata 和绝对 target timestamps。 | mask=`[1,1,0,0,0,0]`，noisy indexes=`[2,3,4,5]`。 |
| M2 PW 干预 | 固定历史 features、instruction、GEN latents、noise、sigma、weights，修改 greedy 预测序列的第二个 PW。 | 未来 Velocity RMS=`0.00358431763`，max_abs=`0.015625`。 |
| M2 UND 隔离 | 对整个 GEN latent 输入加 3。 | UND hidden max_abs=0。 |
| M1 回归 | PW-off、token presence、native visibility、pick→push 干预、两样本隔离及原上游 packer/network 测试。 | 74 项通过；M1 干预 RMS=`0.01337472256`，max_abs=`0.0390625`，与 M1 记录相同。 |
| 原始 baseline | 当前 PW-off tensor 与原始 worktree 保存的 baseline 比较。 | 逐元素完全相等，max_abs=0；SHA256=`c114f25d5ec0d6257547b25c001aba1c113f895cd2ad78d6199a68ddb0381b02`。 |

T06 容差设为 atol=rtol=0.02 以允许 BF16/kernel 差异，但本次观察值全部为 0。Raw K 指 **未采用 GEN RMSNorm 的、经过原生 RoPE 的 UND K**。诊断从 AR pre-RoPE projection 单独构造 GEN 所需 K，再与 joint dispatcher 实际收到的 normalized K 比较，不是比较两个未经使用的中间变量。两个路径的 EOS/BOV 之后无对应 AR 输入，不纳入 parity。

指标明细：[parity](physword_m2_results/parity.json)、[T11](physword_m2_results/t11.json)、[VAE-4](physword_m2_results/vae-4.json)、[VAE-68](physword_m2_results/vae-68.json)、[M2 intervention](physword_m2_results/intervention.json)、[RGB probe 完整 positions/metadata](physword_m2_results/rgb_probe.json)、[M1 regression](physword_m2_results/m1_regression.json)。

推理不消费未来 target。`FutureVisualTargets` 只定义监督数据边界，并未计算 future features/loss。T13 对整个未来目标容器改变后重做历史 prefix 与 AR，同时测试 AR 签名不接受它；它没有以隐藏字段进入 condition。历史源时间信息使用 native source timing helper；fixture 中 source hash 是已观测 RGB bytes 的内容哈希，不是某个不存在的容器文件哈希。

## 可复现命令

在仓库根目录，复用已验证的 `/tmp/physword-m01-venv`（Torch 2.10.0+cu128、Transformers 4.57.1、Flash3）。没有覆盖默认环境，也没有修改默认训练 recipe。

```bash
PY=/tmp/physword-m01-venv/bin/python
RESULTS=/aifs/lijiawang/physword/m2_results

CUDA_VISIBLE_DEVICES=7 I4_ATTN_BACKENDS=flash3 COSMOS_TRAINING=0 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PY" -m pytest \
  -c tools/physword_m01_pytest.ini \
  cosmos_framework/model/generator/physword/multimodal_test.py \
  cosmos_framework/model/generator/physword/visibility_test.py \
  cosmos_framework/scripts/physword_preflight_test.py \
  cosmos_framework/data/generator/sequence_packing/sequence_test.py \
  cosmos_framework/data/generator/sequence_packing/packers_test.py \
  cosmos_framework/model/generator/mot/include_gen_pathway_test.py \
  -q --junitxml="$RESULTS/all_tests.xml"

CUDA_VISIBLE_DEVICES=7 I4_ATTN_BACKENDS=flash3 COSMOS_TRAINING=0 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 \
"$PY" -m cosmos_framework.scripts.physword_m2_probe --output "$RESULTS/rgb_probe"

CUDA_VISIBLE_DEVICES=7 I4_ATTN_BACKENDS=flash3 COSMOS_TRAINING=0 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 \
"$PY" -m cosmos_framework.scripts.physword_probe --output "$RESULTS/m1_regression"

"$PY" -m cosmos_framework.scripts.physword_preflight \
  --checkpoint /aifs/lijiawang/physword/Cosmos3-Edge \
  --output "$RESULTS/m0_preflight.json"
```

上述完整 pytest 命令同时执行 T03/T06/T11/T13 和全部 M1 回归。`multimodal_test.py` 的函数名直接标注验收编号，可用 `-k t06` 等筛选诊断。沿用 M01 的隔离 pytest profile，避免无关 serving/training fixture 导入，不改变项目默认 pytest 配置。

已对全部新增/修改 Python 文件运行 Ruff check、format check（12 files），并运行 `git diff --check`。JSON/JUnit 入 Git；Velocity `.pt` 等原始输出留在 `$RESULTS`，没有将运行权重或 tensor 大文件提交源码。

## 未解决项、根因和 M3 最小任务

1. **真实 checkpoint 验证仍阻塞。** 重跑 preflight：metadata compatible=true、files_complete=false、real_edge_forward=blocked。revision=`344d602b128d1bbdacb43b08d0a3626f46343e29`。八条 component 缺失引用去重后为五个 LFS 文件：两个 transformer shards（5,000,040,072 / 1,739,335,512 bytes）、SigLIP2（978,739,880）、VAE（1,409,400,600）、tokenizer.json（17,077,500）。文件现存大小仅 133–135 bytes。[完整检查记录](physword_m2_results/m0_preflight.json)。真实 special IDs、tokenizer/template revision 兼容性及真实权重 parity 不能用 Tiny 成绩替代。
2. **PW 还没有语义能力。** 本轮新 head 从随机 PW embedding 复制，Tiny 主模型和视觉/VAE 同样随机；重复 class 7 是结构结果。实现了真实的现有 UND AR 计算，但没有训练过的 Physical Word predictor。
3. **当前约束明确。** 只支持 B=1、单视角、5+16/10Hz/K=4、48-channel Wan、unsharded、two_way、guidance=1 的独立 Python 入口。多视角/批 AR/CFG/服务 CLI/长视频 cache/export 未实现；没有为支持它们改写 native attention。Frozen condition tensor 内容需按只读约定使用，dataclass frozen 不阻止调用方原地修改 tensor。
4. **梯度边界。** Joint 的 PW lookup 和 UND 层保持可微，T11 验证可微 forward。AR 是 no_grad greedy，离散 argmax/IDs 阻断 Video Loss 到 AR 采样过程及 Word Head 的梯度；不能声称 Video Loss 会直接训练 Word Head。以后 head 需要 Word CE 监督，预测 IDs 下 Joint 的 Video Loss 可以更新其消费的 PW embedding/UND/GEN 参数。本轮没有计算这些 loss 或测试 backward。

下一轮最小 M3 清单，按依赖执行：

1. 在补齐并固定真实资源后，先复跑本报告四类 parity/leakage/干预和 PW-off 测试；保存 checkpoint、tokenizer、processor、VAE normalization 与词表映射 manifest。旧 M1 state_dict 没有 `physword.word_head.weight`，恢复时须显式初始化新增 head；已有参数的缺失不能静默忽略。
2. 添加小规模、固定时间对齐的数据适配：history condition 与 future-only targets 分离，targets 对应原 clip frames `8/12/16/20`（future 容器内 `3/7/11/15`）。接入已有数据来源，不扩展 action/tactile。
3. 在当前可微 replay 上只加入按预测位置错位的 Word CE 和冻结视觉教师 Future Visual Latent Loss；保留 T03 和显式梯度检查。确定 latent head/归一化维度后再实现，不能把 target feature 放进 Condition。
4. 接上 teacher/predicted PW 的 Joint Video Loss，做小批 backward/梯度归属与 PW 干预验证，明确离散采样不传梯度；随后才按独立 opt-in recipe 配置 Phase A/B 参数组与 checkpoint 恢复。

M3 的损失、优化器、Phase A/B 和正式训练本轮均未执行，到此停止。
