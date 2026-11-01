# PhysWord × Cosmos3-Edge：Pre-M4 实现审查

审查日期：2026-11-01。**结论：NO-GO——当前不应启动真实权重 M4 学习实验。**

M0–M3 已建立可验证的 Tiny 信息通路和单步训练计算图。本轮复跑的 29 项模型前向/协议检查、38 项数据测试均通过，未发现受测路径中的未来条件泄漏或 Video→UND 梯度断裂。但发现 M3 使用离散 timetable 而原版 Edge 使用连续 RF 的实际训练协议问题；真实资源、语义初始化、实验入口、恢复配置和评测协议也尚未闭环。不能把这些 Tiny 结果称为“真实 Edge 的 M0–M3 全部完成”。

本轮仅新增本文档；没有修改代码、下载权重、调用外部标注服务、执行 backward/optimizer step 或启动训练。T09/T10/T14 的更新证据来自已有提交中的测试产物，本轮核对了实现和原始记录，没有重做更新。

## 1. 审查对象与证据边界

| 对象 | 固定版本 / 范围 |
|---|---|
| 模型工作区，以下简称 `M/` | `/aifs/lijiawang/physword/cosmos-framework`，分支 `physword/m0-m1-visibility`，HEAD `ee6288f73f1ff3b4b6566bb6192360f029687c8c` |
| 模型基线 | `55f0554d0bf167603432041c794a8babbac759b9`；M1 `5450c01`，M2 `f2a1e1d`，M3 `ee6288f`；审查其 57 个变更文件、代码/测试/报告及 JSON/JUnit 证据 |
| 数据工作区，以下简称 `D/` | `/aifs/lijiawang/physword/cosmos-framework-data`，分支 `physword/data-pipeline`，审查基线 `826ee2e0e168d39016d21062da7e4cc7a495f198`，基于 M2；与 M3 尚未合并；另见 P1-08 的并行未提交改动 |
| 设计资料，以下简称 `P/` | `/aifs/lijiawang/physword/docs/` 下的 `physword_implementation_plan.md`、`physword_architecture_audit.md`、`pw_visibility_root_cause.md` |
| 论文 | 用户本地 [physical word.pdf](../../physical%20word.pdf)，24 页；§3.3–3.5，PDF pp.4–5；Appendix A.3，pp.14–15 |
| 论文 SHA-256 | `dd717e3e9c1cf9dec01b121c11ce48f25ecb275416660a34b6387cad3567e528` |
| Checkpoint 元数据目录 | `/aifs/lijiawang/physword/Cosmos3-Edge`，revision `344d602b128d1bbdacb43b08d0a3626f46343e29` |

下文 `文件:行` 对应上述模型 HEAD / 数据基线，不沿用早期设计文档的旧行号；P1-08 单独标出在途版本。开始审查时两个工作区均干净，但结束检查发现数据工作区有其它并行修改。因此又通过 `git show 826ee2e:<file>` 导出固定源码至临时目录，独立重跑基线的38项测试，避免把移动中的工作区当作可复现快照。最终版本边界记录时间为17:22（+08）。本次没有在线核验论文版本，以用户提供的 PDF 为准。

证据等级：**本轮复测**表示本轮实际执行；**既有 Tiny 实测**表示提交中有测试实现及通过记录，本轮审读；**静态确认**表示代码路径成立；**未验证**不能视为通过。真实预训练 Edge 的数值测试目前全部未验证。

## 2. 按严重程度列出的问题

### P0

**未发现已证实的 P0。** 在 B=1、unsharded、two-way、无 CFG/Joint KV cache 的受测路径中，未发现当前 GT 泄漏到自身预测位置、未来 RGB 进入 UND、GEN→PW 通路被切断、额外 Transformer 或 Action/Tactile 条件。此结论不覆盖尚无法加载的真实权重，也不覆盖 M5 路径。

### P1-01：训练 Sigma 协议与原版 Edge 不一致，不能接收一般连续 timestep

- 位置：[M/training.py:150](../cosmos_framework/model/generator/physword/training.py#L150)，`PhysWordTrainingStep.__call__`；原生对照 `M/cosmos_framework/model/generator/omni_mot_model.py:2403,2491` 的 `_get_train_noise_level_vision`；`M/.../diffusion/rectified_flow.py:160,167`。
- 原版显式要求 continuous RF：按 resolution/token 数确定 `shifts`，`sigma = sample_train_time(..., shifts=shifts)`，`timestep = sigma * num_train_timesteps`。
- M3 则将 `sample_train_time(1)` 的结果当作离散索引，调用 `get_discrete_timestamp()`，再通过 timetable 浮点相等查找 `get_sigmas()`。它既没有接入原生按分辨率取 shift 的逻辑，也不能接受一般连续 timestep。
- **本轮 CPU 复现**：`RectifiedFlow(None, shift=3, device="cpu")`，seed=17，`u=0.4342412353`。M3 得到 `t=796.4352417 / sigma=0.79643524`；按原生连续 shift 得到 `t=697.2094727 / sigma=0.69720948`。将后者交给 M3 使用的 `get_sigmas()`，返回 shape `[0]`、空 tensor；后续 `sigma.reshape(1,1,1,1,1)` 将无法执行。结果保存在 `/tmp/pre-m4-schedule.json`。
- 同 seed 数值不同本身不能证明边际分布必然错误：对称基础分布的时间反转可保持分布。这里确认的是**离散化、连续输入不兼容、原生 shift 协议未接入**。
- 现有 `synthetic_training()` 始终传 `noise_scheduler.timesteps[500:501]`，`training_test.py:350` 也只核对该固定离散点的插值/归约。因此 118 项通过并未覆盖正式训练的默认时间采样路径。官方 Edge 使用 logitnormal，Tiny probe 默认 uniform，二者也不能默认为同一配方。
- **修复要求**：使用连续 RF `(timestep,sigma)` 协议，显式绑定 Edge 的 distribution/shift/resolution；允许任意有效连续 timestep，避免 timetable 精确相等查找；补同噪声的原版/M3 对齐、非表内 timestep、默认采样和不同分辨率检查。无需改变 attention 或 Video Loss 数学。

### P1-02：预训练语义 Embedding 初始化尚未实现

- 位置：[M/conditioning.py:88](../cosmos_framework/model/generator/physword/conditioning.py#L88)，`PhysicalWordEmbedding.reset_parameters()`；`M/.../mot/cosmos3_vfm_network.py:552`。
- 实际为 seed 固定的 `Normal(0,0.02)` 初始化 17 行，再将其中 16 行复制到独立 Word Head。没有使用动作词的 tokenizer 分解、base input embedding 或 LM-head 语义向量。`load_base_state()` 允许两项 PW 参数缺失，但不会补语义初始化。
- 与论文 §3.3、计划 §3.2 的语义初始化有明确差距。Tiny 的随机初始化适合通路测试；非零干预、CE 梯度或随机 head 输出均不能证明已获得语义锚点。
- **修复要求**：资源补齐后增加一次性、可审计的预训练初始化；保存每词的 subword IDs、聚合规则、start 来源和 base hash；禁止 resume 时重置已训练 PW 参数。随机初始化只作为另行命名的对照。

### P1-03：真实模型资源仍不完整，M0 的真实基座门槛未完成

- 位置：[M/physword_preflight.py:117](../cosmos_framework/scripts/physword_preflight.py#L117)，`inspect_checkpoint()`；实际缺失路径见第 8 节。
- 本轮重跑结果：`metadata_compatible=true`、`files_complete=false`、`real_edge_forward_status=blocked`；8 个组件引用对应 **5 个唯一 LFS pointer 文件**。这些文件实际只有 133–135 bytes。
- 真实 tokenizer ID/embedding 行数及绑定关系、全部权重 shape/remap、SigLIP2/Projector 权重加载、Wan normalization、真实 FP32-master/BF16 数值和完整模型内存均未实测。
- 当前每张 GPU 的空闲量约 10,353 MiB。Tiny 可运行不意味着当前空闲量能容纳真实 Phase B 参数、梯度、Adam 状态和 A/C 两份激活；没有进行真实 OOM 试跑。
- **解除条件**：由后续获授权的资源准备工作补齐或定位完整本地 snapshot，核验 LFS OID/大小及加载结果，准备足够显存，然后先完成真实小样本基座和 M3 验收。本轮未下载。

### P1-04：已有可微 Step，但没有完整的真实 M4 单设备入口

- 位置：`M/.../physword/training.py:96`，`PhysWordTrainingStep`；`M/cosmos_framework/scripts/physword_m3_probe.py:32,62`，`synthetic_training()/main()`；`M/.../omni_mot_model.py:1000` 附近 `_pack_input_sequence()`；`D/.../physword/dataset.py:203`，`to_m3_batch()`。
- Step 可以计算 A+B+C；现有可执行 optimizer probe 固定随机 Tiny、Synthetic RGB、identity VAE normalization、synthetic tokenizer 和 special IDs。没有真实 Edge factory + 严格 warm-start + dataset + train/validation/resume 的统一入口。
- 原生 `OmniMoTModel.training_step()` 没有调用这个 Step；wrapper 只新增了手工 `physical_words` 参数，没有 M2 typed condition 透传。仅把默认训练配置的 PW 开关打开，会落入需要 manual words 的路径，不能自动得到 A+B+C。
- 数据代码尚在独立 M2 分支。`check_m3_bridge.py:35` 通过临时扩展 package path 验证真实数据能构造 M3 类型；它未执行真实数据的网络 A+B+C。其 `batch_size=2` DataLoader 示例也不是模型端支持 B=2 的证据。
- **修复要求**：增加一个明确的、默认不启动训练的单设备 M4 harness 即可，不要求改默认训练配置、不要求接 Ray/FSDP。它需整合数据分支、真实资源加载、正确 VAE normalization/CPU→GPU 转移、连续 RF、精确参数组、空监督处理、验证和保存恢复；先完成真实小 batch 验证，再给训练预算。

### P1-05：T14 没有覆盖影响行为的完整实验配置

- 位置：[M/checkpoint.py:41](../cosmos_framework/model/generator/physword/checkpoint.py#L41)，`save_checkpoint()/load_checkpoint()`；`M/.../physword/training_test.py:164`。
- Bundle 保存 network、optimizer、processor、私有 PW mapping、phase/loss config、CPU/单 GPU RNG 和调用方提供的 `revisions`，并校验文件 hash/参数 keys/shapes；这部分实现有效。
- 但未保存或核验完整 network 行为配置、RF sampler/shift/训练 timetable、外部 VAE 的实际资源及 normalization、数据 manifest/split 与 sampler 进度。`revisions` 仅要求非空，传相同字符串不保证外部 VAE/flow 确实相同。
- T14 使用同一个 `synthetic_training()` factory 重建网络/VAE/flow，并传固定 noise/timestep；因此其 logits/velocity/下一次更新相等只证明**同 factory 的 Tiny bundle**，未证明可恢复完整真实实验。改变 shape 不变的 RoPE/调度配置也不一定被现有 manifest 拒绝。
- **修复要求**：将上述行为配置和实际资源 fingerprint 纳入实验 manifest 并在加载时比对；无需把冻结 VAE 再复制一份。保存数据采样状态，补“变更 flow/normalization/config 应拒绝”和恢复后的默认随机下一步一致测试。

### P1-06：公平对照目前没有可执行的共同实验协议

- 位置：`M/.../physword/multimodal.py:182,159`，`build_reasoner_prefix()/with_boundaries()`；`M/.../inference/physword.py:77`，`repack_physical_words()`；`M/.../inference/physword_diagnostics.py:109`；计划 §13。
- 关闭 PW 的原生 baseline 已有 Tiny 回归，但它不包含新增的 UND RGB prefix。当前 builder 要求 PW enabled，repack 要求恰好四个词和边界；**Matched RGB-prefix / no-PW 对照尚无正式实验入口**。
- T08 是所有层的临时 suffix mask 诊断，不能直接当作已完成、同预算的 auxiliary-only 训练实验。它只验证换词不变性；尚未有 auxiliary-only video backward 的零 PW 梯度证据。不同 attention backend 的 visible/masked 差异也不能全部解释成语义贡献。
- **修复要求**：预先固定 Native、Matched RGB-prefix、Full PW、Auxiliary-only 的输入、分辨率、history、future、noise/time、更新预算和参数规则；实现最小可运行对照。M4 最小比较至少需要 Full/auxiliary-only，并保留 matched-prefix，避免把新增历史 RGB 条件的收益算给 PW。完整论文消融仍可放后续。

### P1-07：真实 Pilot 的语义和评测覆盖不足以验收 M4

- 位置：`D/tools/physword_data/pilot_annotations/*.json` 的 `provenance`；`D/.../physword/annotation.py:65` 的 `label_at()`；`D/.../physword/dataset.py:215`；[数据报告](../../cosmos-framework-data/docs/physword_data/pilot_report.md)。本轮重新统计实际 `data_pilot/badminton/windows.jsonl`。
- 444 个窗口均 `reviewed=false`。原始 207 Episode 已划分，但目前只有 3 条完成 Pilot，并非 207 条训练数据已就绪。
- 当前 Train 仅两条 Episode：433 windows、829 个有效单轨标签，覆盖 approach/move/pick/insert；Test 仅 `episode_0038`：11 windows、18 个有效标签，**全部 approach**，没有一个四词均有效的 Test window。不能据此衡量 pick/move/insert 泛化或完整四步轨迹。
- 完整双臂 `[2,4]` GT 已保留；M3 只消费兼容 `[1,4]` 单轨视图，并行动作被 mask。这符合首版“单轨”范围，但不是论文的双臂 blockwise mask，也不应称为已使用全部双臂监督。
- **解除条件**：人工抽检时间边界/动作语义，明确首版仍采用单轨兼容协议；从已固定的 Train/Test Episode 集合补足标注和评估覆盖。另从 Train 内划 validation，不能为提高结果交换既定 Test Episode 或用 Test 调参。无需为本轮审查擅自扩成双臂模型。

### P1-08：并行 scene_v2 数据词表已扩为25类，与当前16类 M3 不兼容

- **在途改动，非 `826ee2e` 已提交内容**：17:22时 `D/tools/physword_data/scene_v2/vocabulary.json` 的 `badminton_scene_v2` 含25词，在原16词后新增 rest/hold/grasp/lift/align/adjust/retract/lower/release；`D/.../physword/vocabulary.py:48` 的 `require_legacy_model_vocabulary()`、在途 `dataset.py:231,239` 的 `to_m3_batch()` 新增拒绝保护。
- 本轮只读调用该 guard，确认 scene_v2 被显式拒绝，错误为 `Scene vocabulary cannot feed the legacy 16-class M3 Word Head...`。这是正确的阻止静默错配措施，**不是25类模型已支持**。当前 M3 的 Word Head、Embedding table、CE 标签范围、AR反馈、manifest 均仍为16/17类。
- 已审计的 `data_pilot/badminton/windows.jsonl` 在该时点仍为原16类444 windows；SHA-256 为 `a4c20e2d06492ef37bb2b7419277971724549a17cca62d7250736cc65d9957d7`。第6节统计只适用于这份文件，不适用于可能后续生成的 scene_v2 标签。
- 在途 vocabulary JSON SHA-256 为 `64fb4d19646ed989d7c5b5903df3834ebc63bc7c66173bcd58764c5e4b5de066`；guard 文件 SHA-256 为 `c7d4ca1772748edc682209ac09f8916b0d028fa6d251d9dde7c4ab1944efcdbe`。新增代码/标签仍在编辑，未把旧38项通过记录用于宣称在途版本全面通过。
- **解除条件**：M4 固定使用哪个 vocabulary/hash、单轨还是双臂目标；本轮按用户指定的16类机制审计，不擅自扩 head 或强制把新词映射回旧词。若选择 scene_v2，需要独立完成模型/损失/AR/checkpoint/测试迁移再验收。该项属于集成阻塞，不是已发生的标签静默污染。

### P2：范围约束和测试缺口

| ID | 位置与证据 | 影响与所需检查 |
|---|---|---|
| P2-01 时间范围校验只保证“不超出” | `D/.../physword/common.py:96–99`，`PipelineConfig.__post_init__()` 只检查 `K*stride > future_frames` | 本轮构造 `PipelineConfig(future_frames=20)` 被接受，PW last=20，video last=24，违反 mentor 的同 horizon 要求。当前 5/16/4/4 数据实际一致，模型也会拒绝其它长度，故不是当前静默泄漏。允许改配置前应强制 `K*stride == future_frames`，或显式定义覆盖到末帧的非等距时间表。 |
| P2-02 T04 缺梯度参考，T05 未覆盖多样本 RGB AR | `M/.../physword/visibility_test.py:218–255` 全 test 为 no-grad；`multimodal.py:216` / `inference/physword.py:88` 限制 B=1 | T04 仅 BF16 forward 与 FP32 explicit reference；T09 并非 attention reference 梯度对齐。T05 仅 M1 varlen 手工词/GEN 视频隔离，不能外推到 M2 多样本 UND RGB。维持 B=1 可避免后者进入本轮风险范围。 |
| P2-03 Word 缺标连带关闭 latent supervision | `D/.../physword/dataset.py:215–222`，`valid_latent=valid_word.clone()` | 所有视频窗口 RGB 已有效，idle/并行动作/词语歧义却也丢失未来视觉监督。三条 Pilot 共 102 个全无效窗口，其中 Train 101 个；Phase A 的 dataset 默认未过滤。需明确是有意仅监督有效词 block，还是分开 visual-valid；空监督 batch 的 optimizer step 必须跳过，零梯度不保证 AdamW 的 decay/momentum 不更新。 |
| P2-04 模型时钟与源 PTS 字段混用 | `M/.../physword/multimodal.py:65,191`，`D/.../physword/dataset.py:126–152` | `RGBHistory.source_pts_seconds` 实际强制模型网格；bridge 明确传 resampled timestamps，真实采集时间仍存 manifest，未篡改原记录。但 payload 名称和 `1/source_fps` duration 与 10Hz 网格语义混杂。后续清楚拆分模型时间/采集时间，保持 AR/Repack 共用。 |
| P2-05 推理/系统优化边界 | `M/.../inference/physword.py:34,119,137`；`M/.../mot/parallelize_vfm_network.py:213` | 只有单步 velocity，未接多步 sampler/解码/CFG/Joint cache。FP32 master 恢复后推理需 BF16 autocast，API 本身不包 autocast。PW 新入口未注册 FSDP top forward hook，支持入口以 `parallel_dims is not None` fail fast。不能通过删 guard 宣称支持 FSDP。M5 功能不作为 M4 单设备 gate，但必须保持关闭。 |

## 3. 完整计算图与模块调用链

### 3.1 架构保留情况

本轮逐字节比较 `M/cosmos_framework/model/generator/mot/{attention.py,unified_mot.py}` 与原始 `55f0554`，**两者均完全相同**。新增部分是独立 PW table/head、typed condition、packing 输入、损失和控制流程；并未增加第三个 Transformer。

原 Framework `Nemotron3DenseVLTextForCausalLM / Nemotron3DenseVLTextModel` 仍管理同一组 MoT layers。每层 UND 使用原 `q/k/v_proj, o_proj, mlp`，GEN 使用 `*_moe_gen` 参数。这里 `moe_gen` 的命名不表示另加了一套 PhysWord backbone。独立 HF `modeling_cosmos3_edge.py` 没有被当作新的联合训练入口。

| 实际阶段 | 调用链与代码位置 | Tensor / autograd |
|---|---|---|
| 历史输入 | `D/.../dataset.py:203 to_m3_batch` → `M/.../training.py:128 build_reasoner_prefix` → native processor/chat template → `prepare_multimodal_reasoner_inputs` → native SigLIP2 + PatchMerger/Projector | RGB `[5,3,H,W]` uint8；冻结 features `[R,D]`；context IDs `[C]`，mRoPE `[3,C]` |
| A：Teacher forcing | `training.py:57 forward_teacher_words` → `embed_reasoner_condition` + 前三 GT embedding → `language_model.model.reasoner_forward(cache=None)` → native final UND norm → gather prediction positions → 16 类 head | `[1,C+4,D] → Z_TF[1,4,D] → logits[1,4,16]`，有梯度 |
| A：未来视觉目标 | `losses.py:11 encode_future_targets` → 独立逐帧 native image processor → `visual.get_image_features()` → spatial mean | future 容器索引 `[3,7,11,15]`；`G[1,4,D]` FP32、no-grad |
| A：损失 | `word_ce`、`latent_cosine` | 有效 CE mean；有效且 finite/nonzero target 的 FP32 cosine mean；不同计数分别记录 |
| B：预测 | `training.py:139 rollout_eval` → `inference/physword.py:35 generate_physical_words` → 同一 UND、native `ReasonerKVCache`、四次 16 类 greedy | fresh `C+start`；no-grad；`class_ids[4]`；不读 A 的 GT suffix |
| C：VAE/noising | `training.py:141` 冻结 callback 分别编码 history-only 和完整 clip → 取历史专用前缀 + future latents → RF 插值 | clean/noise/xt/vt `[1,48,6,H/16,W/16]`；历史两 latent clean；future 四 latent noisy |
| C：repack | `inference/physword.py:77` → `pack_input_sequence(reasoner_condition=...)` → `append_text_span` → native builder/finalize | `C+start+pred(w1..w4)+EOS+BOV`，所有 offsets 重新计算 |
| C：Joint | `Cosmos3VFMNetwork.forward` → `_encode_text:924` lookup/scatter RGB → `_encode_vision` → native packed MoT → `llm2vae` | 每层 GEN Attention 读取当层 UND K/V；有梯度 |
| C：Video Loss / 总 Loss | `compute_flow_matching_loss` → 一次 `video_loss_scale` → `video + λw*word + λz*latent` | `training.py:170–182`；不经过外层 Omni `_compute_losses` 再乘 scale |
| 更新 | 外部 caller `total.backward()/optimizer.step()` | M3 probe/测试实测；未接默认 trainer |

Edge metadata：`D=2048`、28 层、Q/KV heads=16/8、head_dim=128、VAE C=48。Tiny 实测：`D=128`、2 层、Q/KV heads=4/2、head_dim=32；SigLIP2 width=32/1 层，随机 native Wan，小 synthetic tokenizer。**Edge shape 是元数据确认，不是已加载真实参数的数值运行。**

在 256² RGB、21 帧时：VAE `[1,48,6,16,16]`；GEN 2×2 latent patches 共 `6*8*8=384`，未来 256 个 patches。UND 视觉 token 数必须按 processor 实际 grid 计算，不能假设总与原图 256² 相同。

### 3.2 Attention 与 K/V 路径

`unified_mot.py:830` 在 **RoPE 之前**给 GEN 使用的 UND K 加 `k_norm_und_for_gen`；UND 自注意力保持原始 UND K。`attention.py:202` 的 two-way 对 UND 做 causal，对 GEN 用同样本 `[UND;GEN]` K/V 做一次联合 softmax。

| Query → Key | 历史 RGB/Instruction `C` | PW 整个 suffix `P` | GEN `G` |
|---|---|---|---|
| C | causal | 禁止 | 禁止 |
| P | 允许 | causal | 禁止 |
| G | 允许 | 全部允许 | 原生 full |

跨 packed sample 由 native offsets 隔离。M2 单样本只形成一个 causal UND split；视觉占位仍属于 UND `text_indexes`，不被误放入 GEN vision indexes。`packers.py:582` 保留输入的三轴 positions，`cosmos3_vfm_network.py:931` 将冻结历史特征 scatter 回这些 UND 位置。

GEN 获得的是预测 PW 轨迹的 **逐层 contextual K/V**。`prediction_hidden_states` 是诊断输出，不作为 GEN soft-prefix；最终 post-norm `z_k` 只用于 CE/latent supervision。没有 Final-z Injection、AR KV Direct Handoff、Action/Tactile 条件。配置在 `cosmos3_vfm_network.py:181` 拒绝相关非 RGB 模式，输入在 `inference/physword.py:94` 再检查。

### 3.3 Shift、索引及当前目标泄漏

设 context 长度为 C：

| 预测 | 预测位置 | A 的该位置输入 | target | Joint 中完整预测词自身位置 |
|---|---:|---|---|---:|
| w1 | C | start | GT w1 / future frame 8 | C+1 |
| w2 | C+1 | GT w1 | GT w2 / frame 12 | C+2 |
| w3 | C+2 | GT w2 | GT w3 / frame 16 | C+3 |
| w4 | C+3 | GT w3 | GT w4 / frame 20 | C+4 |

`multimodal.py:136,140` 明确区分 `prediction_positions` 与 `token_positions`。`training.py:75` gather 的是前者；native `_impl_reasoner_forward:1985` 已经过最终 UND norm。CE 没有使用全 LM vocabulary，Word Head 确实是 `Linear(D,16,bias=False)`，17 行 input embedding 中只有 start 不属于预测类。

Teacher 中即使后方出现 GT w_k，causal mask 也禁止预测 w_k 的位置读取它。第 4 个 GT 不进 Teacher 输入。无效前序标签用 start embedding 占位，不把任意 class 0 当作有效 GT。M3 T03 对每个当前标签替换后的所有既往 logits 作 exact-equality 检查；本轮复测通过。

论文 §3.4 的完整定义含同一执行器内 block full、跨 block causal、跨执行器禁止。首版一个 visual token/单轨/block 退化为普通 causal token shift，在限定范围内合理；不具备原论文双臂同时推理能力。

### 3.4 冻结与梯度

`build_reasoner_prefix` 和 `encode_future_targets` 均 no-grad，并对 `net.language_model.visual` 调用 `eval().requires_grad_(False)`。该模块包含 `visual` 与 `projector` 两个子模块（`vision_siglip2.py:446–451`），并非只冻结 SigLIP2。外部 Wan 在 Step 构造时冻结。Future target 每帧独立提取、float 后 spatial mean；cosine 在选择有效 finite/nonzero targets 后计算，避免 `0*NaN`。

冻结视觉特征 detach 是预期行为。普通文本/PW embedding 没缓存为 detached tensor：C 中重新 lookup PW table，因此 Video Loss 可反传。B 的 argmax 无梯度；C 仅复用其离散 IDs，不复用 no-grad hidden/cache。

```text
L_video → llm2vae → GEN 各层
        → Attention(Q_gen, [K_und_for_gen; K_gen], [V_und; V_gen])
        → 当层 UND K/V projection、input norm
        → 较早 UND layers、PW embedding

L_word   → word_head → 最终 UND norm/全部可训练 UND → PW embedding
L_latent → 最终 UND norm/全部可训练 UND → PW embedding
```

最后一层 UND 的 Q/O/FFN、post-attention norm、最终 UND norm 不位于当次 GEN 读 K/V 之前，故不会从 video-only loss 获得直接梯度；独立 Word Head 也没有该梯度。这是原生结构的预期，不是梯度断裂。较早层的 UND attention/FFN 可以通过下一层 K/V 获得 Video Loss 梯度。28 层真实模型对应最后一层 index=27；目前实际独立 backward 证据仅为 2 层 Tiny。

## 4. Loss、Optimizer 与已执行的更新证据

M3 `optimization.py:40` 通过精确 module paths 解析 **完整参数名**，检查重复/不存在/交集，再设置全网 requires_grad；没有 substring 解冻。`optimizer_audit:139` 验证参数 identity、每个 trainable 恰好一组、LR、grad norm 和冻结参数不变。PW table/head 在网络构造 `cosmos3_vfm_network.py:460` 已注册，先于 optimizer。选中参数要求 FP32 master，计算使用 BF16 autocast。

Phase A：PW table/head、选定 UND layers、可选 final norm；GEN/K norm/视觉/VAE 冻结，仅 A。

Phase B：A+B+C；GEN projections/FFN/norms、time embedding、VAE IO projections 和 `k_norm_und_for_gen` 组启用；PW/UND 可配置。切阶段需要重建 optimizer/Step。没有 EMA/FSDP 参数被伪称纳入；这些路径未接入。

以下数值来自已提交 [gradients.json](physword_m3_results/gradients.json)、[step-A.json](physword_m3_results/step-A.json)、[step-B.json](physword_m3_results/step-B.json)，本轮没有重新执行 backward/step：

| 损失 | Phase A | Phase B |
|---|---:|---:|
| Word CE | 2.6092789173 | 2.6092789173 |
| Future latent cosine | 1.0144643784 | 1.0144643784 |
| Video raw RF | 0 | 1.2602672577 |
| Video，10× 后 | 0 | 12.6026725769 |
| Total，λw=λz=1 | 3.6237432957 | 16.2264156342 |

每步有效 word/latent 各 4，样本数各 1；Phase B future video 有效元素 3,072。默认 RF denominator 仍包含历史帧，但历史 loss 分子被 mask 为零；`normalize_loss_by_active=false` 与原版一致。`training.py:180` 只乘一次 10；原生 wrapper 的另一次 scale 在此未被调用。**这一损失/归约确认不抵消 P1-01 的采样协议缺陷。**

| 独立 backward | PW table grad norm | Word Head | UND | GEN | GEN→UND K norm |
|---|---:|---:|---:|---:|---:|
| Video only | 0.12960735 | 0 | 1.87047524 | 8.27421873 | 0.00893324 |
| Word only | 2.18998981 | 5.83854723 | 9.34323109 | 0 | 0 |
| Latent only | 1.10371351 | 0 | 6.97057306 | 0 | 0 |

Tiny Phase B 实际 optimizer step：

| 组 | numel | LR | total grad norm | 参数更新 L2 | 变化 tensor 数 |
|---|---:|---:|---:|---:|---:|
| `pw_embedding` | 2,176 | 1e-4 | 2.34845257 | 0.0022626910 | 1/1 |
| `word_head` | 2,048 | 1e-4 | 5.83854723 | 0.0045254845 | 1/1 |
| `und`，layers 0,1 + final norm | 230,016 | 1e-6 | 10.79916950 | 0.0004524483 | 17/17 |
| `gen` | 329,024 | 1e-5 | 8.27421873 | 0.0057264615 | 29/29 |
| `gen_und_k_norm` | 64 | 1e-5 | 0.00893324 | 0.0000800249 | 2/2 |

Phase A 的三组也实际更新，GEN/视觉/base embedding/外部 VAE 冻结验证通过。完整逐参数名字、梯度和变化量保存在上述 JSON，不以组总梯度替代逐参数审计。以上不是真实 Edge 的参数量或学习效果。

## 5. 训练/推理一致性与缓存

同一 prefix builder 只执行一次 native template/历史视觉编码；A/B/C 共用 context IDs、视觉 features、mRoPE。B 的新词位置使用 `prefix.next_position + step`，repack 从同一 cursor 追加四词和 EOS/BOV。没有 decode→clean→retokenize。

AR 在采样 w4 后仍执行一次反馈 forward（`inference/physword.py:61`），使 cache 包含最后一个已生成词。Tiny 例：C=81，AR 长度 86，joint UND=88，GEN=24，prediction positions 81..84、word positions 82..85；完整屏蔽区域 81..87，包含 start、四词、EOS、BOV。

T06 在每层比较 input/output hidden、raw post-RoPE K/V、GEN-normalized K，以及最终预测 states。GEN-normalized AR K 从 **pre-RoPE projection** 经实际 K norm 再 RoPE 推导，没有错误地对 cached post-RoPE K 做归一化。原生 BF16 和 M3 FP32-master/BF16-autocast 两种 Tiny parity 本轮均复测，误差全为 0。

T11 证明受测 dropout=0 的 Tiny 中，grad-enabled varlen 与 no-grad 单样本 dense 路径输出一致；不能将它扩展为所有 dropout、真实权重、CFG/缓存策略均已一致。

目前 Joint 入口每次直接 `net(packed)`，不传 memory state。AR KV 只存在 B 内，诊断 capture 可返回它，但未 handoff GEN；不存在训练使用 detached Joint KV 的路径。CFG/unconditional、denoising loop、Joint cache invalidation 未实现/未验证。不要直接把 PW 接入原 native cache 或服务分支后沿用这些结论。

## 6. 时间与数据协议

### 6.1 当前 horizon 满足 mentor 要求

窗口起点为 a，model FPS=10，最后历史时刻为 `t=(a+4)/10`：

| 数据 | 模型帧位置 | 相对最后历史时刻 |
|---|---|---|
| 历史 RGB | a+[0..4] | -0.4s..0 |
| 未来视频 RGB | a+[5..20] | +0.1s..+1.6s，共16帧 |
| PW / future visual targets | a+[8,12,16,20] | +0.4/+0.8/+1.2/+1.6s |
| VAE latent 右边界 | a+[0,4,8,12,16,20] | 历史2 + 未来4 latent |

因此 PW 和视频的未来 **终点均为 t+1.6s**；PW 是稀疏目标时刻，不要求每一帧都有一个 Word。不能把完整 clip 2.0s 的最后坐标误称为预测未来 2.0s，也不能把 K=4、stride=4 当作论文不可修改的要求。

本轮扫描实际 444 windows，`pw_target_frame_indices[-1] != future_frame_indices[-1]` 的数量为 0。`windows.py:33` 用目标图像的真实 `source_elapsed_s` 查阶段标签；模型时间和采集时间分别保存。`media.py:78` 使用 `searchsorted(..., side="right")-1` 向过去取帧，不取未来最近邻，不插值。当前实际最大帧龄等质量量来自 Pilot 报告，未重新解码全 207 条视频。

VAE 采用原生 causal Wan 4× temporal /16× spatial compression。history=5 与 full=21 均满足 `1+4m`；训练强制使用 history-only 编码结果构成干净 condition，即使未来编码存在问题也不会直接替换历史 condition。本轮 GPU/CPU Tiny Wan 都复测 temporal_window=4、68 的 5→2/21→6，以及改变未来后历史 prefix 不变，误差为 0。**没有使用真实 VAE 权重完成这些验证。**

改变初始配置时，须同时更新模型端硬限制、targets 索引、VAE边界、loss/noise/condition masks、AR长度和测试；只修改数据 YAML 当前会被 bridge/model 拒绝，不能通过“数据可配置”声称训练已支持其它 horizon。P2-01 的数据校验也需同步修正。

### 6.2 Split、标注与正式训练依赖

`split.py:46` 先按 original identity、lineage、源 RGB 内容 hash 合并组，seed=42 后作 Episode 186/21 Train/Test 划分，再在 `windows.py:24` 切重叠窗口。`check_leakage()` 检查各身份不会跨 split；本轮 38 个数据测试包含变换副本、窗口 split 篡改、时间采样和 poison action 字段隔离。

没有发现当前 split 发生 Episode 泄漏；未知 lineage 的重新编码/裁剪近重复仍不由 hash 自动识别。现有 Test 已用于本轮覆盖度审计，不应再拿它选择阈值、λ 或训练步数。

| 实际 Pilot | Train | Test |
|---|---:|---:|
| 已处理 Episode | 2 | 1 |
| windows | 433 | 11 |
| 至少1个有效单轨 Word 的 windows | 332 | 10 |
| 4个 Word 全有效的 windows | 82 | 0 |
| 有效单轨 Word targets | 829 | 18 |
| 类别计数 | approach 214 / move 343 / pick 152 / insert 120 | approach 18 |

完整两臂监督有 919/3552 个有效位置；单轨兼容视图有 847/1776 个有效位置，因并发/歧义等 mask，不能把二者混报。`to_m3_batch()` 没有把八词偷偷拼成四词，也未加入动作/状态条件。Future RGB 和标签仅进入 target 容器，`condition_inputs()` 仅白名单历史图像、指令和公开时钟。

这些标签来自完整序列的视觉审阅，尚非人工金标；论文 A.3 报告的标注质量不能转用于本 Pilot。当前只覆盖四类也不能据此对 16 类行为有效性作结论。

## 7. T01–T14 的真实执行状态

既有 JUnit：M0/M1 **74/74**（15:11:43 +08）、M2 **93/93**（15:38:58）、M3 **118/118**（16:18:15），各自 failure/error/skip=0；不是每个数字都代表新 PW 测试，M3 含前序 93、新 M3 17 和原生 flow loss 回归 8。本轮抽取不含 backward/step 的模型测试 **29/29**（17:12），数据工作区测试 **38/38**（17:11），另在不可变 `826ee2e` 源码快照上再次确认 **38/38**（17:22，2.81s），并复跑 CPU M3 bridge/Wan 检查。数据测试两次是相同基线测试集，不相加为76项独立覆盖。

| ID | 代码 / 既有证据 | 当前可认定的状态 | 不能外推的部分 |
|---|---|---|---|
| T01 PW-off | `visibility_test.py:143`；`physword_m01_results/baseline_comparison.json` | 既有原始 commit Tiny baseline 比较；本轮 off forward 回归通过 | 未做真实 Edge baseline；本轮未重新 checkout 原始 commit 跑一遍 |
| T02 presence/packing | `visibility_test.py:55`；`multimodal_test.py:50` | 既有 M1 不等长多样本索引；本轮 M2 AR/repack 元数据通过 | 新多模态路径限定 B=1 |
| T03 shift / target leakage | `multimodal_test.py:77`、`training_test.py:279` | 本轮 Tiny 复测通过，包括逐 GT 改动、未来 RGB 不改 AR/UND prefix、GT-only 不改 C velocity | 训练改变未来 x0 时 noisy future/velocity 可变，这是合法 flow 训练，不是条件泄漏 |
| T04 attention reference | `visibility_test.py:219` | 本轮 native BF16 与 FP32 explicit-mask **forward** 通过 | **梯度 reference 对齐未执行**；不能标记完整 T04 已通过 |
| T05 isolation | `visibility_test.py:201` | 本轮 M1 两个不等长 packed samples 干预隔离通过 | M2 多样本 UND RGB/AR 未实现，未验证 |
| T06 AR/repack | `multimodal_test.py:50`；`training_test.py:398`；`amp_parity.json` | 本轮 BF16 与 AMP 两种 Tiny 逐层结果通过，max abs 全0 | 未用真实28层参数，不含 CFG/Joint cache |
| T07 intervention | `visibility_test.py:158`；`multimodal_test.py:195` | 本轮手工 pick/push、预测第二词替换都有响应 | 通路成立，不代表语义有效或视频质量改善 |
| T08 full suffix | `training_test.py:46`；`visibility.json` | 本轮全7位置、所有层屏蔽，换词后 exact invariant | 诊断 eval；auxiliary-only 训练/梯度尚未验证 |
| T09 independent losses | `training_test.py:87`；`gradients.json` | **既有 Tiny 实测通过，本轮核对未重跑 backward** | 无真实 Edge backward，未覆盖全部可能冻结配置 |
| T10 actual update | `training_test.py:131`；`step-A/B.json` | **既有 Tiny A/B 各一步更新和冻结验证通过，本轮未 step** | 无真实参数更新，无学习曲线 |
| T11 train/infer parity | `multimodal_test.py:93` | 本轮同条件 grad/no-grad Tiny packed fields、velocity 通过，max abs=0 | 原生 CLI、完整 sampler、非零 dropout、真实权重尚未验证 |
| T12 CFG/cache | 计划 §11/§13；当前 `physword.py` 入口 | **未实现/未执行**，按 M5 范围排除 | AR cache flush 不等于 Joint cache/CFG 验收 |
| T13 visual/VAE leakage | `multimodal_test.py:118,151`；`training_test.py:319`；data bridge | 本轮原生 Tiny SigLIP/Projector 与 Tiny Wan 检查通过，history-prefix max abs=0 | 无真实 teacher/VAE 权重；CPU bridge 的 VAE 输入也是随机32²，不是 Pilot视频编码 |
| T14 roundtrip | `training_test.py:164,220,238`；`roundtrip.json` | **既有同 factory Tiny 实测通过，本轮审读**：17映射、logits/velocity0差、IDs相同、Adam moments/RNG/下一更新一致；损坏拒绝 | 私有 PW mapping 恢复，不是 HF tokenizer 新增17词；真实资源/配置/数据进度恢复未验证（P1-05） |

M0/M1 原始 Tiny baseline SHA-256：`c114f25d5ec0d6257547b25c001aba1c113f895cd2ad78d6199a68ddb0381b02`，既有 original/PW-off 比较 max abs=0。

本轮 T08 数值与提交记录逐项相等：

| 指标 | 数值 |
|---|---:|
| future velocity RMS | 0.9756876230 |
| PW intervention RMS difference | 0.003584317630 |
| 相对 RMS difference | 0.003673632365 = **0.367363%** |
| 相同输入重复 forward floor | 0 |
| 全 suffix masked 后换词 RMS difference | 0 |
| masked 重复 forward floor | 0 |
| visible/masked RMS difference | 0.005515067838 |

这证明在该 Tiny/seed/backend 上存在可消除的 PW→GEN 数值依赖，且差异高于本次重复运行 floor。head 未经过真实语义训练；不能由它推出“pick 比 push 更合理”、物理推理成功或视频质量收益。

## 8. 真实权重、Tokenizer、Checkpoint 的就绪状态

本轮只读 preflight 与之前五个 pointer 的结论相同。以下相对路径均在 `Cosmos3-Edge/`：

| 缺失实际内容的文件 | 当前 bytes | LFS 指针声明的应有 bytes |
|---|---:|---:|
| `transformer/diffusion_pytorch_model-00001-of-00002.safetensors` | 135 | 5,000,040,072 |
| `transformer/diffusion_pytorch_model-00002-of-00002.safetensors` | 135 | 1,739,335,512 |
| `vision_encoder/model.safetensors` | 134 | 978,739,880 |
| `vae/diffusion_pytorch_model.safetensors` | 135 | 1,409,400,600 |
| `tokenizer.json` | 133 | 17,077,500 |

合计约 9.145 GB（十进制）。文件存在不代表权重已下载；pointer 仅含 LFS version/OID/size。主模型含多个组件引用，因此 preflight 的缺失引用数为8，而唯一文件为5。本轮核查的是配置的这个 snapshot，未声称搜索了所有挂载盘和用户私有缓存。

真实 tokenizer 未加载，故 base special IDs、最大 ID、tokenizer 长度、base embedding/LM-head 是否绑定的运行时验证仍缺失。现有 PW 使用 `token_id_base=2**30` 的私有 namespace，`class_id=0..15` 对应 `base+1..base+16`，start=`base`；通过独立 lookup 和 manifest 恢复，不允许直接交给通用 HF tokenizer/generate。该选择符合低侵入 mapping 方案，但与最初“向 HF tokenizer 注册17词”的提议不同。

已有 Tiny checkpoint 存在于 `/aifs/lijiawang/physword/m3_results/phase_{a,b}/checkpoint/`。T14 的严格 network key/shape、processor hash、mapping/phase/teacher 字符串检查有效；它尚不能取代真实 snapshot 加载验收，也不能覆盖 P1-05 列出的行为配置恢复。

## 9. M4 启动前的最小关闭清单

按以下顺序完成，**本文不执行这些开发或训练动作**：

1. 修复 P1-01 连续 RF 协议，并将默认采样路径纳入测试；把 timestep/shift/normalization 纳入 manifest。当前固定点 Loss 测试继续保留。
2. 补齐或定位真实 snapshot，完成 tokenizer/special IDs/tied weights/shape/视觉塔/VAE normalization 的严格验证；准备足够单设备内存，先跑 PW-off 原始 baseline 和小样本 forward/backward/update/恢复验收。不要绕过现有 FSDP guard。
3. 完成预训练 PW 语义初始化及其保存/恢复；加载后确认17行/16类和逐词来源，已有 PW checkpoint 不重新初始化。
4. 合并或正式集成数据代码，提供单设备真实 harness，消费 `PhysWordBatch`，使用同一协议运行 A+B+C；显式跳过 Phase A 空监督步骤，确定 latent-valid 独立性；确认每个 batch 的实际分辨率/视觉 token budget。当前 Pilot 是480×640，不是报告示例的256²。
5. 固定经人工抽检的单轨标注协议、vocabulary ID/hash；在途25类 scene_v2 不能直接供给当前16类模型。补足 Train/Validation/Test 的语义覆盖。保留原 Test 划分，从 Train 内留验证集；不要用当前只有 approach 的 Test 宣称四步机制有效。
6. 实现最小 Matched RGB-prefix/Full/auxiliary-only 对照，统一预算、teacher、输入、noise/time和评测 seed；补完整 suffix mask 的梯度检查与 T04 reference 梯度证据。B=1 保持不变。
7. 扩展 checkpoint 实验 manifest/数据 RNG 恢复校验，先做恢复后默认下一步一致性；真实 M0–M3 gate 通过后再评估放行 M4。

关闭以上问题后的建议起点：单视角、B=1、10Hz、5 history+16 future、K=4/stride=4；统一 resize/crop 到预先固定的尺寸，记录 SigLIP token budget；先取少量已审阅 Episode/32–64个训练窗口和独立 validation。Phase A 只训练 PW+选定最后 UND 层；Phase B 继续 A+B+C、GEN/K norm；PW/UND/GEN LR 可从 `1e-4/1e-6/1e-5`、λw=λz=1、video scale10一次开始。明确选择并记录 AdamW betas/eps/decay，当前自定义 optimizer 默认值并不等同原生 SFT recipe。

每次只授权有限步数，例如 Phase A 先50步、Phase B 先25步，记录按 block/类的有效数、CE accuracy、latent cosine/方差、固定连续sigma下 RF loss、各组 grad/update 与干预/floor/mask-out。若只有单步 velocity 接口，结果仅称 RF/velocity 评测；完整生成视频指标需另接原生采样/解码后验证。CFG/cache/FSDP、Direct KV Handoff、Final-z、Action/Tactile 均不属于此最小放行配置。

## 10. 本轮验证命令及产物

以下模型命令选择的测试没有调用 backward 或 optimizer step；部分会构建 optimizer 并运行有梯度 forward，以检查模式 parity。所有新增运行产物位于 `/tmp`，没有覆盖仓库中既有 evidence。

```bash
cd /aifs/lijiawang/physword/cosmos-framework
CUDA_VISIBLE_DEVICES=7 I4_ATTN_BACKENDS=flash3 COSMOS_TRAINING=0 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /tmp/physword-m01-venv/bin/python -m pytest \
  -c tools/physword_m01_pytest.ini \
  cosmos_framework/model/generator/physword/multimodal_test.py \
  cosmos_framework/model/generator/physword/visibility_test.py::test_pw_off_forward_is_exact_baseline \
  cosmos_framework/model/generator/physword/visibility_test.py::test_intervention_changes_future_velocity_through_native_layers \
  cosmos_framework/model/generator/physword/visibility_test.py::test_pw_cannot_read_generator_and_earlier_pw_cannot_read_later_pw \
  cosmos_framework/model/generator/physword/visibility_test.py::test_cross_sample_isolation_on_real_varlen_pack \
  cosmos_framework/model/generator/physword/visibility_test.py::test_native_two_way_matches_explicit_visibility_reference \
  cosmos_framework/model/generator/physword/training_test.py::test_t08_full_suffix_mask_and_relative_intervention \
  cosmos_framework/model/generator/physword/training_test.py::test_t03_teacher_targets_do_not_enter_rollout_or_joint \
  cosmos_framework/model/generator/physword/training_test.py::test_future_targets_are_independent_native_spatial_means \
  cosmos_framework/model/generator/physword/training_test.py::test_video_native_reduction_scale_and_clean_history \
  cosmos_framework/model/generator/physword/training_test.py::test_t06_fp32_master_bf16_autocast_ar_joint_parity \
  --junitxml=/tmp/pre-m4-forward-tests.xml -q
# 29 passed，12.44s；一个既有 requests 依赖版本 warning。

/tmp/physword-m01-venv/bin/python -m cosmos_framework.scripts.physword_preflight \
  --checkpoint /aifs/lijiawang/physword/Cosmos3-Edge \
  --output /tmp/pre-m4-preflight.json

cd /aifs/lijiawang/physword/cosmos-framework-data
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
/tmp/physword-data-venv/bin/python -m pytest \
  -c tools/physword_data/pytest.ini --confcutdir=tests/physword_data \
  tests/physword_data -q --junitxml=/tmp/pre-m4-data-tests.xml
# 38 passed，2.82s。

/tmp/physword-m01-venv/bin/python -m tools.physword_data.check_m3_bridge \
  --manifest /aifs/lijiawang/physword/data_pilot/badminton/windows.jsonl \
  --m3-root /aifs/lijiawang/physword/cosmos-framework \
  --output /tmp/pre-m4-data-bridge.json --check-vae
# 真实样本构造 CPU PhysWordBatch + 随机 native Tiny Wan 因果检查通过；无模型训练。
```

并行数据修改出现后，以 `git show` 逐文件导出 `826ee2e` 的数据包/测试/pytest配置到 `/data/9qTymlC6XNreTg7J/tmp/pre-m4-data-source-qupfoigs`，在该目录执行同一数据 pytest 命令，JUnit 改为 `/tmp/pre-m4-data-pinned-tests.xml`，结果38 passed。版本/hash元数据保存于 `/tmp/pre-m4-data-snapshots.json`。

操作备注：最初尝试整仓 `git archive` 固定源码时，Git LFS filter 自动触发了仓库示例 MP4 媒体获取；发现后已终止该进程，改用不触发过滤器的逐文件 `git show`。这可能填充了仓库 LFS 媒体缓存，未下载模型权重，未改动工作区代码；五个模型/tokenizer pointer 的阻塞状态未因此解除。

P1-01 最小复现（CPU，无模型或训练）：

```python
import torch
from cosmos_framework.model.generator.diffusion.rectified_flow import RectifiedFlow

flow = RectifiedFlow(None, shift=3, device="cpu")
kw = {"device": "cpu", "dtype": torch.float32}
torch.manual_seed(17)
u = flow.sample_train_time(1)
t_m3 = flow.get_discrete_timestamp(u, kw)
sigma_m3 = flow.get_sigmas(t_m3, kw)
torch.manual_seed(17)
sigma_native = flow.sample_train_time(1, shifts=torch.tensor([3.0]))
t_native = sigma_native * 1000
print(t_m3, sigma_m3, t_native, sigma_native)
print(flow.get_sigmas(t_native, kw).shape)  # torch.Size([0])
```

既有证据可核验 SHA-256：

| `M/docs/physword_m3_results/` 文件 | SHA-256 |
|---|---|
| `all_tests.xml` | `149af357528e03589d80bd9a08088fe941e3cb78252071c8b897c9b7567359ce` |
| `gradients.json` | `97b9a49ef301f35f02116418d4e513f5a44e9fdb4a33ce948db87f6e6da03eaa` |
| `step-B.json` | `9d1de4c2fa31cf3c9c9a40e9b02d00744e8a770d46692e35eab94dbed1eca860` |
| `roundtrip.json` | `e2db3ffb6ce7beb005b7b27f73a203c7c09c8cf857a231b8bdfa9214caf9f2ab` |

**最终判定仍为 NO-GO，针对现在启动真实 M4 学习实验。** 原生双 Transformer 信息通路及 Tiny A+B+C 验证成果保留；不要求重做已通过的 Token Repacking、UND RGB 或 MoT attention。完成上述修复/资源和评测 gate 后可重新作放行审查。本轮审查到此停止。
