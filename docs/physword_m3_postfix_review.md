# PhysWord × Cosmos3-Edge：M3 修复后独立验收审查

审查日期：2026-11-01。仓库 `/aifs/lijiawang/physword/cosmos-framework`；分支 `physword/pre-m4-fixes`；HEAD `00508fc2388857af71b7305a64fe57ed3d8abd6e`。对照 M3 基线 `ee6288f`，以及原始 NVIDIA 提交 `55f0554`。

## 1. 验收结论

**M3 整体验收：NO-GO，尚不能标记为完整通过。** 这不是沿用旧审查结论：原 RF、真实权重、语义初始化、动态任务词表问题已经取得当前代码上的实测关闭证据；本次发现了新的真实 T06 失败和跨进程恢复缺陷。

| 层次 | 本轮结论 |
|---|---|
| 双 Transformer MoT 与 PW→GEN 架构 | 正确；未添加第三个 Transformer、Final-z、Direct KV、Action/Tactile 输入 |
| Tiny A+B+C、独立损失梯度、真实参数更新 | PASS，当前代码重跑 |
| 真实 UND/GEN/SigLIP2/Projector/VAE/Tokenizer 与语义初始化 | PASS，重新完整加载和校验 |
| 真实 PW/head-only Phase A 更新及同进程新实例恢复 | PASS；不等于真实完整 Phase B |
| 真实 AR→Repack 数值一致性 T06 | **FAIL**，既定容差下失败；无缓存 UND replay→joint 却完全一致 |
| 独立 Python 进程恢复 | **FAIL**，相同语义配置因内部列表顺序不同被拒绝 |
| 外置空间预处理变更的恢复保护 | **FAIL**，resize 变化没有进入实际 checkpoint 协议 |
| 完整真实 Phase B 更新/恢复 | **BLOCKED**，当前空闲显存不足；未实例化完整 B optimizer 做更新 |
| 实际任务学习效果/视频质量提升 | **NOT RUN**，没有拟合或质量实验 |

**M4-A：CONDITIONAL GO，仅指关闭下述条件后可开展有界小样本实验，当前不直接放行。** 先修复跨进程恢复与预处理契约，解决真实 T06 的验收失败，人工抽查所选训练片段。可从已验证的 PW/head-only、B=1 配置开始；解冻 UND 或提高尺寸前另做真实单步。

**M4-B：NO-GO。** 除上述修复外，还必须在有足够空闲显存的单卡上完成真实 A+B+C 的三项独立 backward、全部指定参数组更新及跨进程下一步恢复；不能用 meta 库存或真实 A 替代。

本轮没有修改模型、训练、数据 Pipeline 实现，没有下载资源，没有开启 M4。新增内容仅为本报告、审查脚本及证据。旧审查文档与未跟踪材料保留；没有切换分支、提交或覆盖模型代码。

## 2. 本次证据与历史证据的边界

阅读了修复报告、M2/M3 报告、两轮 Pre-M4 审查、Implementation Plan 的调用链/测试/里程碑、冻结实验 Manifest，并核对当前源码及相对 `ee6288f` 的修改。论文依据为本地 `physical word.pdf` §3.3–3.5、Appendix A.3；其“单视觉、单 effector”退化为每个 block 一个词，因此普通因果 shift 符合本轮约束；旧 16 类不是不可变要求。

所有下文写作“本次 PASS”的证据均来自新目录 [physword_m3_postfix_results](physword_m3_postfix_results/)，不是复制历史 PASS。仅旧测试产生的 `/tmp` 小型 JSON 在**本轮重新运行后**归档。汇总见 [summary.json](physword_m3_postfix_results/summary.json)。

本次既有测试：**140 模型 PASS / 58 数据 PASS / 0 skip**，耗时分别 22.29s、2.98s。另行独立检查发现的 FAIL 不包含在这 198 项既有 pytest 中；不能写成“198 项通过，全部验收通过”。

真实大 checkpoint 的新测试输出位于 `/aifs/lijiawang/physword/m3_postfix_review_results/real_edge/`，小型结果归档在报告旁；其 `checkpoint/state.pt` 为 7,711,326,599 bytes，是本次恢复测试生成的状态文件，不是下载。另有 `real_data_tiny/` 与 `cross_process_checkpoint/`。

## 3. 原五项阻塞的关闭状态

| 原问题 | 当前源码与本次证据 | 关闭状态 |
|---|---|---|
| 离散查表 RF 与连续训练不一致 | `physword/training.py:172`、`diffusion/rectified_flow.py:167`；三个非整数 sigma 的输入/embedding/loss/梯度均零差；额外 40 组原始/当前原生采样零差 | **已关闭：当前 Edge RGB-only 协议**。真实三 sigma 全梯度仍未执行 |
| 缺少预训练语义初始化 | `physword/initialization.py:12`、`loading.py:46`；真实每一个 input/head pooled row 独立比较 max abs=0，start=0 | **已关闭，真实权重验证** |
| 五个 LFS 指针 | `loading.py:22` 校验完整 SHA256；本次 loader 和真实 VAE 均成功 | **已关闭**，不是仅文件大小通过 |
| 评测/Baseline/恢复闭环 | 已有 episode holdout、三个 packing baseline、Tiny 与真实 A 同进程下一步恢复；新跨进程恢复 FAIL，预处理契约漏项；真实 B BLOCKED | **部分关闭，不可整体关闭** |
| 25 类标注与固定 16 类模型冲突 | `physword/config.py:32`、`conditioning.py:62`、`data/generator/physword/task.py:26`；实际左臂 V=7，模型 V+1/head V，真实数据接通 | **已关闭：本轮左臂单轨 task** |

## 4. 新发现与剩余问题，按严重程度

### P0：未证实新增的 Future Leakage、RF 数学错误或主图梯度断裂

本次没有发现支持这些 P0 指控的证据。完整真实 Phase B 仍有不可跳过的资源/验收阻塞，但它不是已证实的模型数学错误。以下 P1 足以阻止宣布 M3 全部验收完成。

### P1-R1：相同配置跨进程 Resume 被误拒绝（已复现）

- 位置：`cosmos_framework/model/generator/physword/checkpoint.py:28` 的 `training_protocol()`，尤其第 36 行 `dict(flow.noise_scheduler.config)`；第 24 行 `_canonical()`；`load_checkpoint():234–253`。
- 原因：Diffusers scheduler config 中 `_use_default_values` 是由 set 转 list 的内部元数据；`sort_keys=True` 只排序字典键，不排序该列表。不同 `PYTHONHASHSEED` 下，同样配置生成的列表顺序不同，严格 experiment 比较报错。
- 外部源码证据：当前环境 `diffusers/configuration_utils.py:731` 为 `list(set(new_kwargs.keys()) - set(init_kwargs))`；其自身序列化在第 652 行删除这一内部字段。
- 本次复现：同一个 `synthetic_training('A')`，同一 RF/data/phase/loss 配置，在独立进程 `PYTHONHASHSEED=1` 保存、`=2` 恢复，得到 **`ValueError: Checkpoint experiment differs; refusing a silent phase/mapping/teacher change`**。去掉该内部字段后两份语义配置完全相等，default-key 集合也完全相等。
- 证据：[cross_process_resume.py](physword_m3_postfix_results/cross_process_resume.py)、[cross_process_resume.json](physword_m3_postfix_results/cross_process_resume.json)。此 FAIL 是验收语义失败，脚本捕获异常后写 JSON，不能只看退出码。
- 影响：之前“fresh model”恢复是在同一个解释器内重建实例，无法证明重启进程可续训。真实 A 同进程 0 差依然有效，但完整恢复协议未闭环。真实跨进程恢复本次 **NOT RUN**，不冒称它已经实测失败；已证实缺陷位于两者共用的协议比较函数。
- 最小修复：生成稳定的 scheduler **行为配置**，删除或规范化明确无序的内部元数据，同时保留所有实际行为字段；增加不同 hash seed 的独立进程保存/恢复测试和真正 shift 改变的拒绝测试。不要排序所有列表，尤其不能排序词表、层序或 sampler order。

### P1-R2：真实 Edge 的缓存 AR 与 joint repack 未通过 T06（已复现）

- 入口：`cosmos_framework/inference/physword.py:35`，`generate_physical_words()` 的 cached prefill/decode；第 63 行最后一个词 flush 存在。
- 对照：`physword/multimodal.py:280` 的 `replay_physical_words()`；`inference/physword_diagnostics.py:70` 的既定 `atol=rtol=0.02` 比较；原生 `mot/unified_mot.py:946,1665,1908` 的 reasoner 路径。
- 条件：真实 V7 Edge 权重，真实 Train 窗口 0 的历史 RGB/指令，显式 64×64，合成固定 GEN latents（隔离 VAE），BF16 autocast，FP32 派生频率 buffer；与正式训练并非同一尺寸。
- 首次直接调用现有 comparator 在 `layer_6.raw_k` 失败。补充审查只收集全部失败，不放宽容差、不改生产计算。28 层 × 5 类比较加最终 hidden，共 **141 项中 94 项失败**。
- 第 0 层 input/hidden/raw K/raw V/normalized K 均 exact；第 6 层 raw K max abs=0.03125，RMS=0.00137254。最终 `[4,2048]` prediction hidden max abs=**1.61962890625**、RMS=**0.1574984342**、相对 RMS=**7.1204945%**，8192 个元素中 4539 超过现有数值容差。
- Word logits 最大绝对差 **0.390625**；该样本 AR、无缓存 replay、joint 的 argmax 都是 `[4,4,4,4]`。同样 argmax 不是 hidden/KV parity 的替代验收。
- **定位证据：同一预测轨迹的无缓存 UND replay 与 joint 的 prediction hidden max abs=0。** 差异集中在缓存增量 AR 与全序列计算之间；当前证据不支持把错误归到 repacking 或宣称 GEN 看不到 PW。BF16 增量执行的累计误差是待检验解释，不能据此自动豁免 7.12% 差异。
- 证据：[real_structural_initial.json](physword_m3_postfix_results/real_structural_initial.json)、[real_structural.json](physword_m3_postfix_results/real_structural.json)、[real_structural.py](physword_m3_postfix_results/real_structural.py)。后者仍将 overall T06 写为 FAIL。
- 最小下一步：固定同一前缀和词轨迹，对 cached AR、每步完整无缓存 UND、joint 做逐 token/逐层比较，区分 kernel/精度与 cache/position 错误；补更高精度参考及多个真实窗口，不以重复 `[lift]*4` 校准准确率。可先验证不使用 cache 的 K=4 greedy 作为低侵入候选，再决定修复；保持原 MoT 数学结构。未经证据不要单纯调大容差。

### P1-R3：实际空间预处理未进入恢复契约（已复现）

- 位置：`scripts/physword_real_probe.py:168–172` 外部执行 bilinear 64×64 resize、round、uint8；`real_checkpoint_roundtrip():36–38` 构造 experiment 却没有记录这些规则。`physword/checkpoint.py:28–45` 也没有该配置项。
- `rf.resolution='480'` 是采样 shift 策略，不是实际输入 480p 的证明。processor 文件哈希覆盖原生 processor，不覆盖调用前的 `F.interpolate`。
- 本次 Tiny 复现：同一源数据、相同 checkpoint/experiment，恢复后外部尺寸由 64² 改为 96²，load 和两次 forward 均接受；总 loss 由 **3.6237432957** 变为 **3.5446515083**。这不是要禁止有意 fine-tune 尺寸，而是当前“严格续训拒绝行为漂移”的承诺未覆盖实际变换。
- 证据：[preprocessing_resume.py](physword_m3_postfix_results/preprocessing_resume.py)、[preprocessing_resume.json](physword_m3_postfix_results/preprocessing_resume.json)。同一变换下恢复的零差结果不受此结论否定。
- 最小修复：冻结并保存 RGB resize/crop/interpolation/rounding/range、teacher preprocessing、VAE encode/normalization、实际训练空间尺寸与 RF resolution 的独立字段；训练入口据此构造数据变换并在 Resume 校验。实际文件内容 hash 应在入口重新核验，不能只信 caller 传来的旧字符串。增加变更尺寸或归一化后拒绝恢复的反例。

### P1-R4：完整真实 Phase B 更新和恢复缺少必需实测

- `physword/optimization.py:40–124` 的真实 Phase B 选组正确，但当前 loader 的冻结基座是 BF16；所选可训练参数必须按该函数要求为 FP32 masters。真实 probe 第 280 行之后只构造 Phase A，不能冒称已覆盖 B。
- 本次在**新生成真实 checkpoint 的 tensor shapes**上重新核对全部历史选组完整名称、无重复、参数量：PW=16,384，head=14,336，UND=201,345,024，GEN=1,414,921,408，GEN-und K norm=3,584；合计 **1,616,300,736**。选择器源码未变，此检查依然只是库存核对。
- 仅选中参数的 FP32 weight+gradient+两份 Adam moment 即 **25,860,811,776 bytes = 24.0848 GiB**，尚未包含冻结基座、VAE、激活和 workspace。历史 28.2464 GiB 估算还包含冻结参数；本轮使用更保守的 24.0848 GiB 下界，也已超过实测每卡约 **10.11 GiB** 空闲显存。
- 本次没有启动会确定超预算的完整 B backward，没有抢占别人的 GPU 进程，也没有用 FSDP 绕过。证据：[real_phase_b_inventory.json](physword_m3_postfix_results/real_phase_b_inventory.json)。
- 最小关闭条件：取得足够空闲单卡显存后，在冻结后的真实训练配置下做三项独立 backward、一次 A+B+C optimizer update，以及独立进程下一次随机 RF/noise/数据更新对照。先小尺寸，再验证目标尺寸；不据 64² 推断 480p 可训练。

### P2：范围及实验准备

1. `annotation_status.reviewed=false`；Train 有效词中 hold 占 **74.7925%**，approach 只有 **1** 个重叠窗口目标。人工核对目标片段/边界并分离 train 内调参集；不能把 hold majority 的 CE 下降或复制重叠窗口当作推理进步。
2. 两个 episode 的 pilot holdout 已成立，但没有 Val。没有证据表明本轮使用 Test 拟合或选词；本轮真实 probes 只读取 Train 0/1 和另一个含 mask 的 Train 样本。代码无法证明所有仓库外历史操作，不能声称全程零调参污染。
3. `PhysWordTrainingStep` 是单设备 B=1 opt-in API，尚不是默认 TOML trainer 的完整可运行 SFT 配方；JSON 的 `max_train_steps=0` 表明它们是冻结实验规格。正式 loop、验证/停止预算、数据 sampler 与配置联动属于 M4 最小入口工作，不要直接运行默认 Edge recipe 以为会调用 A+B+C。
4. FSDP/AC、Joint CFG/cache 不在当前 PW 支持范围。`training.py:121` 与 `inference/physword.py:40` 对已分片模型 fail fast；本轮没有验证多 rank hooks。原生 per-layer reasoner 有 FSDP 注册不意味着新增顶层训练 API 已自动兼容。
5. 原生全 inference 测试 collection 本次仍因 Python 3.11 没有 `typing.override` 而 **BLOCKED**（`inference/args.py:11`）；纯 key mapper 测试及真实 strict loading 通过，不能替代全 engine 测试。

## 5. A+B+C 调用链、Tensor 与梯度

```text
TrainingWindowDataset / to_m3_batch
  history uint8 [5,3,H,W] + instruction
  future uint8 [16,3,H,W]、word_labels/valid_word/valid_latent [1,4]（target only）
    │
    ├─ build_reasoner_prefix @no_grad
    │    native processor/chat template → SigLIP2 → PatchMerger/Projector
    │    visual_features [R,2048] frozen；IDs [C+1]、mRoPE [3,C+1]
    │    suffix 的第一个位置为 <pw_start>
    │
    ├─ A forward_teacher_words （有梯度）
    │    prefix + valid GT w1,w2,w3；无效前词反馈 neutral start
    │    reasoner_forward(cache=None) → final UND norm
    │    prediction_positions=C..C+3 → Z_TF [1,4,2048]
    │    V7 word_head → logits [1,4,7] → valid CE
    │    独立 future frames 3,7,11,15 → frozen visual/projector
    │    → spatial mean [1,4,2048] → FP32 cosine
    │
    ├─ B rollout_eval + generate_physical_words @no_grad
    │    原始 prefix；K=4 greedy；private IDs；最后 w4 也 flush
    │    不传 GT/future；argmax 不参与反传
    │
    └─ C repack_physical_words （有梯度）
         原始历史 prefix + start + predicted w1..w4 + EOS/BOV
         PackedSequence UND [C+7] / GEN latent [1,48,6,h,w]
         新 lookup embedding → native Joint MoT 每层 UND K/V
         GEN history 两个 clean latents；future 四个 noised latents
         preds_vision → native compute_flow_matching_loss × 10（一次）

L = L_video + lambda_word*L_word + lambda_latent*L_latent
```

源码定位：`physword/training.py:57,138–225`；`physword/multimodal.py:140,199,271`；`inference/physword.py:35,77`；`sequence_packing/packers.py:582,1118`；`mot/cosmos3_vfm_network.py:925`；`mot/unified_mot.py:833`；`mot/attention.py:202`。

可见性：同样本 UND 因果读取前缀和前词，不能读取 GEN；GEN 读取同样本 UND K/V 与 GEN。传给 GEN 的是逐层 contextual representations，**不是最终 post-norm Z 的直接注入**。`k_norm_und_for_gen` 在 RoPE 前规范化原生 UND K；此数学没有被改写，`attention.py` 和 `unified_mot.py` 相对 `ee6288f` 无差异。

A 中使用 prediction positions，不是 GT token positions；当前 GT wk 位于下一输入位置或根本不在 A 输入中。真实 GT 逐项交换 T03 的前缀预测 hidden 差均为 0；Tiny future/GT counterfactual 与 B/C 条件一致测试重跑通过。`encode_future_targets():10` 的 no-grad 和 target detach 合理；它调用的 `vision_siglip2.py:457` 包含 projector，然后 mean pooling。`latent_cosine():46` 先排除 invalid/nonfinite/零范数 target 再计算 FP32 cosine，不用 NaN×0。

C 重新 lookup PW table，所以梯度能从 Video Loss 经 GEN attention 回到 UND K/V 和 PW embedding；B 的 IDs 不可导没有切断 C 的连续路径。最后 UND 层的 K/V、其输入 norm 和对应 GEN K norm 可获 video 梯度；最后层 UND Q/O/MLP/final norm/Word Head 没有直接 video 梯度是图的正常结构。CE/latent 可训练这些被选中的 UND 部分。冻结参数的 forward 没有被整体 no-grad 包裹，Tiny 已验证 frozen UND 仍向 PW table 传播梯度。

## 6. 连续 RF 的本次数值审查

原生 recipe：`configs/base/experiment/sft/models/edge_model_config.py:101–120`：video **waver**；shift 256→3、480→5、720→10；uniform weight；active normalization=false；loss scale=10；discrete=false。PW step 的 resolution 是明确策略参数，64²诊断仍声明使用480策略，不假装为正式480p。

- `sample_continuous_train_time():167` 复用 `sample_train_time()`；t=Nσ，N=1000。
- `training.py:172–196` 显式 sigma 原样用于 noising；仅显式 timesteps 输入才做 t/N。双输入严格校验一致性与有限/范围；没有调用 `get_sigmas()` 查表。
- `vision_sigmas()/noise_vision():181–194` 得 sigma_eff=σ(1−history_mask)。xσ=(1−sigma_eff)x0+sigma_eff ε，target=ε−x0。history target 保留原值但 loss mask 掉。
- `compute_flow_matching_loss():52,143–165` 保留按全部元素 mean 的原生分母和时间权重。PW 在第223行应用一次 scale=10；没有额外重叠 scale。
- `Cosmos3VFMNetwork._embed_packed_timesteps():944` 使用现有 time embedder 和 timestep_scale；没有离散 embedding table。
- `loading.py:103` 只转 parameters 为 BF16；本次实测 `rotary_emb.inv_freq`、`original_inv_freq`、`time_embedder._timestep_frequencies` 均 FP32。真实 input/head 初始化来源为已经严格加载的权重。
- 发布 scheduler 的 `prediction_type=flow_prediction`、`use_flow_sigmas=true`、N=1000；原生 recipe 使用 UniPC，训练的连续 t/σ 和速度语义一致。没有执行完整 UniPC 视频 rollout，不能声称生成质量或多步误差通过。

| sigma | noised latent max abs | time input/embedding max abs | target/loss max abs | 5组关键梯度 max abs |
|---:|---:|---:|---:|---:|
| 0.12345 | 0 | 0 / 0 | 0 / 0 | 全0 |
| 0.37460 | 0 | 0 / 0 | 0 / 0 | 全0 |
| 0.82731 | 0 | 0 / 0 | 0 / 0 | 全0 |

上表为随机原生 Tiny、合成 RGB/小 VAE，`atol=rtol=0`；梯度对照为 PW table、UND layer0 K/V、GEN llm2vae、time MLP。见三个 [RF JSON](physword_m3_postfix_results/physword-rf-0.12345.json)。既有 sampler 测试是3分布×3分辨率，9组 exact。

本次另从 `55f0554` 和当前文件 AST 提取完整原生 `_get_train_noise_level_vision()` 执行：4分布（waver/uniform/logitnormal/ltx2）×2策略（none/diffusion_forcing）×5配置（video/image/mixed/dynamic/lidar-only）=40组，seed20261101，t/σ max abs全0。它覆盖共享 helper 的其他采样分支；不宣称执行了这些模态的完整训练、CP或所有 noising 路径。证据：[independent_checks.json](physword_m3_postfix_results/independent_checks.json)。

## 7. 损失、参数组与实际更新

### Tiny Synthetic：三项独立 backward

| Loss | 数值 | PW table grad norm | Head | UND | GEN | K-norm UND→GEN |
|---|---:|---:|---:|---:|---:|---:|
| Video（scale后） | 12.6026725769 | 0.12960735 | 0 | 1.87047524 | 8.27421873 | 0.00893324 |
| Word CE | 2.6092789173 | 2.18998981 | 5.83854723 | 9.34323109 | 0 | 0 |
| Latent | 1.0144643784 | 1.10371351 | 0 | 6.97057306 | 0 | 0 |

raw video=1.2602672577，total=16.2264156342，lambda_word=lambda_latent=1。完整每参数证据：[gradients.json](physword_m3_postfix_results/gradients.json)。

Phase B Tiny optimizer 真实更新：

| Group | 参数量 | LR | Total-loss grad norm | 改变tensor/组选中tensor |
|---|---:|---:|---:|---:|
| pw_embedding | 2,176 | 1e-4 | 2.34845257 | 1/1 |
| word_head | 2,048 | 1e-4 | 5.83854723 | 1/1 |
| und | 230,016 | 1e-6 | 10.79916950 | 17/17 |
| gen | 329,024 | 1e-5 | 8.27421873 | 29/29 |
| gen_und_k_norm | 64 | 1e-5 | 0.00893324 | 2/2 |

这里 V16/D128 是保留的 Synthetic fixture，不是实际任务规格。动态 V3 测试和真实标签 V7 Tiny 测试另外通过。Phase A 的三个组也全部实际更新；frozen network 和 VAE 均逐参数未变。完整名称、每 tensor grad/update norm 见 [step-A.json](physword_m3_postfix_results/step-A.json)、[step-B.json](physword_m3_postfix_results/step-B.json)。

参数收集由 `optimization.py:47` 使用 `get_submodule(exact_path)` 和完整名字，检查重复/遗漏/身份；不是 substring 选择。PW table/head 在 network 构造时注册，semantic init 在 optimizer 前，未发现 optimizer 漏收。

### 真实数据 + Tiny，及真实 Edge + 真实数据

本次重跑真实 V7 标签、真实 RGB 显式 resize64² 的 Tiny A/B：A total=2.7574362755，CE=1.7660483122，latent=0.9913879633；B video=13.1596384048，total=15.9170751572。word_count=3、latent_count=4，验证 mask 独立，冻结参数未变。证据：[real_data_tiny_summary.json](physword_m3_postfix_results/real_data_tiny_summary.json)。

真实 Edge PW/head-only A：CE=**2.3978393078**、latent=**0.9946485758**、total=**3.3924880028**，有效4/4。

| 完整参数名 | numel | LR | grad norm | update norm |
|---|---:|---:|---:|---:|
| `physword.embedding.weight` | 16,384 | 1e-4 | 304.3778076 | 0.0078358883 |
| `physword.word_head.weight` | 14,336 | 1e-4 | 56.7093277 | 0.0119733335 |

真实冻结 UND/GEN/vision 与 step 前 CPU snapshot 比较不变。这里只有 A 联合 CE+latent backward，不能把它填写为真实三项独立 T09 或完整 B T10。较大梯度应在后续学习实验监控，但单步有限并成功更新，不凭数值大就断言不稳定。

## 8. 词表、Mask、真实权重

任务词表为左臂有序 `[align, approach, grasp, hold, lift, lower, rotate]`，V=7；embedding `[8,2048]`，head `[7,2048]`，没有复用整个 LM vocabulary logits。`task.py:55–67` 只统计 Train valid 标签；真实所有行重算的词表、counts 与冻结 Manifest 一致。

ID 空间：class=0..6；PW embedding index=class+1；private ID=1073741824+index；start index=0；原 tokenizer subword IDs 独立。input/head 分别 FP32 mean pooling 自己的真实 rows；start来自 input `<|im_start|>` ID10。真实 tying=False。`initialization.py:19` 防止 optimizer 后/重复初始化；`load_checkpoint():264` 恢复 provenance，真实恢复使用 `initialize_pw=False`。

本次逐词逐 row 独立检查全为0差，含 approach `[13008,1664]`、grasp `[1870,11982]`、lift `[1108,4083]` 等多 subword；这比只看初始化统计更强。词义、effector、有序词表进入 fingerprint，值为 `3d74b2a6532a95090dc0292b30a8210ba074530188939a7d5d9eacaaf6381b07`。

原25类文件未修改；overlay 的 per-arm 字段重映射到V7，原值在 `source_*` 保留。无效/未知类不进入 CE；前一无效 GT 在 `training.py:65–68` 反馈 private start，不反馈 class0。未来 RGB 有效可独立参与 latent，即 word mask=false；invalid target先筛选，不强行填错标签。

资源固定于 `/aifs/lijiawang/physword/Cosmos3-Edge`，revision `344d602b128d1bbdacb43b08d0a3626f46343e29`。本次 loader 按 pinned git LFS OID 完整重算五文件 SHA256并加载：

| 文件 | 字节数 | SHA256前缀 | 当前 |
|---|---:|---|---|
| tokenizer.json | 17,077,500 | 4dc692a99dca | 真实文件 |
| transformer shard 1 | 5,000,040,072 | 116134e5492c | 真实权重 |
| transformer shard 2 | 1,739,335,512 | 3b4c0aafe270 | 真实权重 |
| vision_encoder/model.safetensors | 978,739,880 | 2180ad739ecc | 真实权重 |
| vae/diffusion_pytorch_model.safetensors | 1,409,400,600 | 230496cb59ff | 真实权重 |

UND/GEN 544 tensors、vision/projector 443 tensors严格加载；missing required/unexpected/shape mismatch均为空。RGB-only 明确不使用的5个 Action keys 在 [real_edge.json](physword_m3_postfix_results/real_edge.json) 的 `loading.explicitly_unused_action_keys` 列出；未以宽泛 strict=False 吞掉关键缺失。Tokenizer最大ID131071，input/head rows各131072。完整 SHA、subword 与初始化来源见同文件和 [real_structural.json](physword_m3_postfix_results/real_structural.json)。

## 9. 数据、时间和 Baseline

本次重新核验全部433行 Manifest：Train episode0001有247 windows，Test episode0017有186；按 episode/group/original_episode/lineage/media_fingerprint 检查无跨split复用。这个检查依据现有 lineage/fingerprint 元数据，不是本次重新哈希所有原视频像素。实际模型 probes读取真实RGB文件。

源 windows SHA仍为 `a870319f565b76c919711cbf4161f6313966f7460da9bbb7ef0e90df92431ca8`；release windows SHA仍为 `3f7e2a855f39c49387b64b020b2f2f4fffc6995dcfed732975ec09049b812e49`。全局split没有修改；Test是独立 pilot overlay从两个原全局Train episodes中hold out一个，不是原全局20个Test已经准备齐全。

实际为10Hz、history5/future16、K4/stride4。历史截止模型时间0.4s，targets0.8/1.2/1.6/2.0s；PW与视频未来范围都是1.6s。所有行 horizon浮点误差最大2.22e-15s；PW target的真实source timestamps与future RGB offsets3/7/11/15完全对应。

首窗口实际capture target times为16396.257333、16396.655840、16397.056376、16397.459085，不把capture抖动伪装成严格0.1s；model clock使用统一抽帧后的10Hz。VAE temporal compression=4，5帧→2 latent、21帧→6 latent；history `[0,1]`，future `[2,3,4,5]`，对应右边界frames8/12/16/20。

本次真实 published Wan VAE history-only与full前缀 max abs=0（tol1e-3），shape `[1,48,2,4,4]` / `[1,48,6,4,4]`；normalization采用发布 mean/std。未来RGB作为latent监督和video flow target，不传Reasoner条件；C的clean history取history-only编码。

Train masked24/988=2.42915%，Test16/744=2.15054%，合计40/1732=2.30947%；没有通过消灭mask改变GT。标注全部 `reviewed=false`。

三Baseline packing当前可运行：`inference/physword_baselines.py:11`。Matched与Full历史input IDs/mRoPE/visual features相同，matched没有start/private PW；Full增加start+4词，二者都有EOS/BOV，因此UND长度、后续position存在明确的5-token差。Native使用原caption协议，不能把它与Full差异全归因于PW，Matched对照不可省略。

真实固定weights/RGB/instruction/noise/sigma=0.3746：native future velocity RMS0.5397911668，matched1.0487759113，full1.0487374067；这些是输出幅度，不是视频质量分数。PW-off max abs=0。替换第二个PW的RMS差=0.02191142179，相对差=2.0893144%，重复forward floor=0；每层完整7-token suffix（start+4词+EOS/BOV）mask后替换差=0。mask诊断临时使用SDPA移除完整suffix，不修改生产MoT数学。

Tiny干预RMS差0.00358431763，相对0.3673632%，floor0、masked差0。两者都只证明条件通路，不证明动作语义正确。当前真实预测 `[lift,lift,lift,lift]`，没有学习有效性证据。没有执行正式三baseline同预算SFT或完整生成视频；21帧/10Hz/64²能够运行不等于官方常规范围生成质量已经验证，也不等于架构不支持。

## 10. T01–T14 完整状态矩阵

PASS/FAIL均指本次执行；历史报告仅作对照。真实数据Tiny单步不能代替真实权重测试。

| ID | Tiny/合成或小型原生模块，本次 | 真实权重，本次 | 说明 |
|---|---|---|---|
| T01 PW-off | PASS | PASS，max abs0 | 原生无private条件路径；未跑完整视频生成 |
| T02 IDs/packing | PASS，含不等长样本 | PASS，真实V7构建/forward | IDs与prediction/token positions分离 |
| T03 当前GT/未来泄漏 | PASS，当前GT交换、future边界、B/C条件反例 | PASS：真实4个GT逐项交换不影响自身及更早预测；真实future-RGB交换全套NOT RUN | 未发现GT自泄漏 |
| T04 explicit attention参考 | PASS：既有forward；新增BF16 Q/K/V backward PASS | NOT RUN：完整真实模型参考梯度 | 单层模块Q/K/V是随机输入，不称真实权重训练 |
| T05 跨样本隔离 | PASS：M1 varlen双样本 | NOT RUN | 多模态PW B>1当前不支持 |
| T06 AR/repack逐层 | PASS：既有BF16/AMP Tiny | **FAIL**，94/141比较超既定容差 | 真模型无缓存replay→joint预测hidden为0差 |
| T07 intervention | PASS，RMS/relative/floor齐全 | PASS，2.0893%相对差/floor0 | 条件敏感性，不是语义有效性 |
| T08 完整suffix mask | PASS | PASS，masked替词差0 | 7个token，每层 |
| T09 三项独立backward | PASS，合成Tiny逐参数 | NOT RUN：独立真实三loss；PASS：真实A联合backward | 真实标签Tiny A+B+C总loss也PASS |
| T10 optimizer更新 | PASS：Tiny A/B，真实标签Tiny A/B | PASS：PW/head-only A；BLOCKED：完整B | 冻结参数实际比较不变 |
| T11 train/infer mode | PASS，Tiny grad/no-grad | NOT RUN：独立真实grad-mode parity | T06无缓存no-grad replay一致不能替代T11 |
| T12 CFG/Joint cache | NOT RUN | NOT RUN | M5；不要与B的Reasoner KV cache混淆 |
| T13 VAE/未来边界 | PASS，Tiny真实因果VAE类 | PASS：真实Wan前缀0差；全套future-swap NOT RUN | 真实时序Manifest另PASS |
| T14 logits/AR/video/state恢复 | PASS：Tiny同进程、含scheduler/EMA/scaler/sampler/累积；**FAIL：跨进程与预处理guard** | PASS：真实A同进程新模型；真实跨进程NOT RUN；完整B恢复BLOCKED | 不再笼统填写T14 PASS |

新增T04数值：forward max abs0.0077903271；Q/K/V gradient max abs分别0.0107066631、0.0142135620、0.0247030258；RMS分别0.0011015281、0.0018204502、0.0026905874。使用 `abs_error <= 0.02 + 0.02*abs(reference)`；V max abs大于0.02仍满足相对项，不是偷偷忽略失败。

同进程 Tiny full A+B+C resume：下一随机RF/noise/data样本、Python/NumPy/Torch CPU/CUDA RNG、Cosine scheduler、启用的EMA/Scaler、VAE、全部network参数与下一loss/update exact；mid-accumulation gradients恢复exact。改变RF/data协议字符串被拒绝。这些PASS真实存在，但跨进程与外置预处理缺口见R1/R3。

真实A同进程恢复：Word logits、单步velocity、下一真实Train窗口loss和下一PW/head更新max abs全部0；AR IDs、CPU/CUDA RNG、scheduler相同；cursor1；不重复语义初始化。真实A未启用EMA/Scaler，没有用Tiny证明真实EMA。

## 11. 本次执行命令

工作目录均为本仓库。环境 Python3.11、Torch2.10.0+cu128、Transformers4.57.1、H100/Flash3。无网络下载。以下均为实际执行；审查脚本把异常记为FAIL时可能退出0，请同时读JSON。

```bash
export I4_ATTN_BACKENDS=flash3 COSMOS_TRAINING=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 PYTHONPATH=.
# COSMOS_TRAINING=0 不禁用显式 autograd。
CUDA_VISIBLE_DEVICES=7 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /tmp/physword-m01-venv/bin/python -m pytest \
 -c tools/physword_m01_pytest.ini \
 cosmos_framework/model/generator/physword/training_test.py \
 cosmos_framework/model/generator/physword/multimodal_test.py \
 cosmos_framework/model/generator/physword/visibility_test.py \
 cosmos_framework/model/generator/physword/rf_test.py \
 cosmos_framework/model/generator/physword/pre_m4_test.py \
 cosmos_framework/model/generator/mot/checkpoint_mapping_test.py \
 cosmos_framework/scripts/physword_preflight_test.py \
 cosmos_framework/data/generator/sequence_packing/sequence_test.py \
 cosmos_framework/data/generator/sequence_packing/packers_test.py \
 cosmos_framework/model/generator/mot/include_gen_pathway_test.py \
 cosmos_framework/model/generator/algorithm/loss/flow_matching_test.py \
 -q --junitxml=docs/physword_m3_postfix_results/model_tests.xml

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /tmp/physword-data-venv/bin/python -m pytest \
 -c tools/physword_data/pytest.ini --confcutdir=tests/physword_data tests/physword_data \
 -q --junitxml=docs/physword_m3_postfix_results/data_tests.xml

CUDA_VISIBLE_DEVICES=7 /tmp/physword-m01-venv/bin/python \
 docs/physword_m3_postfix_results/independent_checks.py

CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=8 /tmp/physword-m01-venv/bin/python \
 -m cosmos_framework.scripts.physword_real_probe \
 --checkpoint /aifs/lijiawang/physword/Cosmos3-Edge \
 --task-dir /aifs/lijiawang/physword/pre_m4_fixes_results/badminton_left_release \
 --output /aifs/lijiawang/physword/m3_postfix_review_results/real_edge \
 --phase-a-step --checkpoint-roundtrip

CUDA_VISIBLE_DEVICES=6 /tmp/physword-m01-venv/bin/python \
 -m cosmos_framework.scripts.physword_task_probe \
 --task-dir /aifs/lijiawang/physword/pre_m4_fixes_results/badminton_left_release \
 --output /aifs/lijiawang/physword/m3_postfix_review_results/real_data_tiny

CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=8 /tmp/physword-m01-venv/bin/python \
 docs/physword_m3_postfix_results/real_structural.py
# 首次原comparator直接失败；随后增加全部误差记录和无缓存replay对照重新执行。

CUDA_VISIBLE_DEVICES=6 /tmp/physword-m01-venv/bin/python \
 docs/physword_m3_postfix_results/preprocessing_resume.py

PYTHONHASHSEED=1 CUDA_VISIBLE_DEVICES=6 /tmp/physword-m01-venv/bin/python \
 docs/physword_m3_postfix_results/cross_process_resume.py save \
 /aifs/lijiawang/physword/m3_postfix_review_results/cross_process_checkpoint
PYTHONHASHSEED=2 CUDA_VISIBLE_DEVICES=6 /tmp/physword-m01-venv/bin/python \
 docs/physword_m3_postfix_results/cross_process_resume.py load \
 /aifs/lijiawang/physword/m3_postfix_review_results/cross_process_checkpoint

CUDA_VISIBLE_DEVICES=6 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /tmp/physword-m01-venv/bin/python -m pytest \
 -c tools/physword_m01_pytest.ini cosmos_framework/inference/model_test.py --collect-only -q
# BLOCKED: typing.override；没有执行自动获取其他checkpoint的测试。
```

重跑需要给保存类脚本换新的输出目录，避免碰到防覆盖保护。数据integrity检查对433行重新计算源/release hash、词频/mask、时序和lineage集合；结果见 [data_integrity.json](physword_m3_postfix_results/data_integrity.json)。资源SHA校验包含在本次真实 loader 内，没有只复用旧资源清单。

## 12. 最小后续工作与停止点

1. 修复稳定RF/checkpoint行为配置序列化，用真正独立进程恢复验收；保留拒绝真实配置漂移的严格性。
2. 闭合实际RGB/teacher/VAE预处理协议，区分RF shift策略和像素尺寸；让实际loop从冻结规格生成变换。
3. 专项解决真实cached AR↔full replay↔joint的T06失败，增加真实窗口、logit margin与精度参考；不改MoT结构，不自动放宽容差。
4. 以上通过后，人工核对少量训练窗口，以PW/head-only、B1、10Hz/5+16/K4/stride4、lambda各1、LR1e-4为M4-A候选；记录每类计数/准确率、CE/latent、预测分布与固定验证协议。解冻24–27层使用单独真实单步门槛，不能直接套用已验证head-only结果。
5. M4-B还需足够空闲显存与真实完整单步/跨进程续训。下一轮先补有界验收，再讨论正式步数；不得将合成loss下降或任意PW干预差作为收益证据。

**已停止于审查。没有修复上述生产代码，没有启动M4或长时间H100任务。**
