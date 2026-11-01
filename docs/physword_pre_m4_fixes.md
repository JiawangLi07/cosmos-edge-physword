# PhysWord × Cosmos3-Edge：Pre-M4 阻塞项修复与验收

2026-11-01，分支 **`physword/pre-m4-fixes`**。保留 M0–M3 基线 `ee6288f`，合入数据提交 `e89480a`，所有修复均位于独立模型分支；原数据 worktree 未修改。

**本轮完成代码修复，且真实 Edge 权重的加载、语义初始化、单步视频 Forward、PW/head 更新和 Checkpoint 恢复均已实测通过。没有启动 M4 正式训练。** 完整 Phase B 解冻后的真实 optimizer step 仍受当前空闲显存限制，不能标为通过。

## 1. 五项 NO-GO 问题的处理状态

| 问题 | 本轮处理 | 验证层次与边界 |
|---|---|---|
| 离散 RF 时间查表 | **修复**：共享原生连续采样与 history-mask noising，训练不再查离散 scheduler 表 | Tiny 三组指定 sigma 的数值/梯度差均0；与原始 NVIDIA 方法的9组采样对比差均0；真实 Edge 单步接受非整数 timestep |
| PW 预训练语义初始化 | **实现并完成真实权重验证**：input embedding 与 LM-head 分别 mean pooling，记录真实 tying/start/subwords | 真实 V=7、D=2048；不是 Mock 初始化；恢复时不再次初始化 |
| 五个 LFS 指针 | **已补齐**：五个真实文件逐一通过官方 SHA256 | UND/GEN 544 tensors、SigLIP2+Projector 443 tensors 严格加载；VAE 实际 encode；Tokenizer 最大ID与矩阵对齐 |
| 评测/Baseline/Resume | **最小闭环完成**：独立 episode pilot holdout、三种 baseline 配置/Forward、完整单卡 checkpoint API | Tiny 完整 A+B+C 中断恢复通过；真实 PW/head-only Phase A 恢复通过；正式 baseline SFT/完整视频生成实验未执行 |
| 25类数据 vs 16类模型 | **修复**：按用户确认的左臂单轨，训练标签生成冻结的7词 manifest；模型 V/V+1 自动匹配 | 真实数据进入 Tiny A+B+C 和真实 Phase A；原始25类标注及右臂原始数据保留 |

证据目录：[physword_pre_m4_fixes_results](physword_pre_m4_fixes_results/summary.json)。模型测试 **140 PASS**，数据测试 **58 PASS**；本轮均重新执行，无 skip。M0 状态文字调整后另复跑其3项测试通过，不重复计入总数。

## 2. 修改文件、函数与核心 Diff

以下代码路径相对仓库根目录。

| 文件/位置 | 主要改动 |
|---|---|
| `cosmos_framework/model/generator/diffusion/rectified_flow.py:167` | `sample_continuous_train_time()`：原生连续 sigma/shift、t=Nσ；`vision_sigmas()`/`noise_vision()` 统一历史 mask 与原生线性插值 |
| `cosmos_framework/model/generator/omni_mot_model.py:2477,2640` | 原生 Omni 的采样尾部与 vision-mask 逻辑委托给同一 RF 内核；没有修改默认训练配置、损失数学或 attention |
| `cosmos_framework/model/generator/physword/training.py:57,100` | TF 标签按实际V校验；Phase A 可不装配 VAE；Phase B 连续时间，支持显式 `sigmas=`；保留原 A+B+C 编排 |
| `cosmos_framework/model/generator/physword/config.py:30` | 配置词表、词义、ID、effector；fingerprint 包含顺序/词义/effector；保留 legacy 默认用于回归 |
| `cosmos_framework/model/generator/physword/conditioning.py:58` | V+1 embedding、V-way head；未知 private ID 显式报错；已初始化/恢复后禁止 reset |
| `cosmos_framework/model/generator/physword/initialization.py:12` | `initialize_physical_words()`：真实普通 subword input/head rows 的 FP32 mean；真实 weight-tying 检查与 provenance |
| `cosmos_framework/model/generator/physword/loading.py:21,46` | 固定 revision 的本地资源 SHA256、严格 UND/GEN/vision 装配、Tokenizer 检查、加载后初始化；支持 resume 不初始化 PW |
| `cosmos_framework/model/generator/mot/checkpoint_mapping.py` | 从原 inference loader 提取原有纯 key/index 映射；原 inference 继续重导出这些接口，训练装配不导入完整 inference engine |
| `cosmos_framework/model/generator/mot/cosmos3_vfm_network.py:176,467` | 保存完整 PW config；`init_weights()` 在语义初始化/恢复之后明确拒绝重新初始化 |
| `cosmos_framework/model/generator/physword/multimodal.py`、`cosmos_framework/inference/physword.py` | 动态V/fingerprint、显式无PW RGB-prefix 类型；原 Full PW repacking 保留 |
| `cosmos_framework/inference/physword_baselines.py:11` | Native / Matched RGB-prefix / Full PW 的可执行 packed-condition builder |
| `cosmos_framework/data/generator/physword/task.py:26` | 只读源标注、冻结训练词表、任务 split overlay、单轨标签映射、独立 latent mask、三种实验配置 |
| `cosmos_framework/data/generator/physword/dataset.py:236` | `to_m3_batch(..., pw_config=...)` 严格校验任务 fingerprint；分别传 Word/Latent mask |
| `cosmos_framework/data/generator/physword/common.py:98` | PW horizon 必须**等于** future video horizon |
| `cosmos_framework/data/generator/physword/sampling.py:6` | 单设备/worker=0 可恢复 sampler：epoch、order、cursor、generator state |
| `cosmos_framework/model/generator/physword/checkpoint.py:28,138,214` | schema2 完整行为配置、随机状态、optimizer/scheduler/EMA/scaler、累积梯度、数据进度；严格 resume |
| `cosmos_framework/scripts/physword_{task,real}_probe.py` | 真实标注+Tiny 单步、真实 Edge 初始化/Forward/更新/恢复的有界验收入口 |
| `tools/physword_resources.py`、`physword_preflight.py` | 仅五个资源的 pinned/hash 校验与下载工具；preflight 明确区分资源可用与运行时兼容 |

核心时间协议变化：

```python
# 修复前：u 被当成下降的离散 scheduler table 的索引
u = flow.sample_train_time(1)
t = flow.get_discrete_timestamp(u, ...)
sigma = flow.get_sigmas(t, ...)

# 修复后：原生 Omni 与 PW 共用同一连续采样实现
t, sigma = flow.sample_continuous_train_time(..., shifts=resolved_native_shift)
# t = sigma * num_train_timesteps
xt, vt = flow.noise_vision(epsilon, x0, sigma, history_condition_masks)
```

保留已有 `PhysWordTrainingStep` 作为 opt-in A/B/C forward/loss API，不新建 trainer loop。VFM 使用原生 RF/原生 `compute_flow_matching_loss()`/原生 `TimestepEmbedder`。`mot/attention.py`、`mot/unified_mot.py` 相对 `ee6288f` **完全未变**，见 summary 中 hash/比较结果。没有新增 Transformer、Final-z、Action/Tactile Conditioning、Direct KV Handoff。

## 3. 连续 RF：修复前后数值

原生 Edge video 分布是 **waver**；原始配置 `edge_model_config.py` 的 resolution shift 为256→3、480→5、720→10。原 probe 的 uniform 是 Synthetic fixture，不能当作正式 Edge 默认。

`rf_test.py:20` 从基线 `55f0554` 提取未改写的 `_get_train_noise_level_vision()` 方法，fixture 仅提供 self/config。固定 seed=20261101、B=8，waver/logitnormal/uniform × 三种 resolution，共9组；修复后 sigma 和 timestep **max abs=0，atol=rtol=0**。

对比上一轮实际反例中的 waver 结果：

| resolution / shift | 修复前 sigma max abs / RMS 差 | 修复后 |
|---|---:|---:|
| 256 / 3 | 0.699844241 / 0.288823605 | 0 / 0 |
| 480 / 5 | 0.604793429 / 0.241626993 | 0 / 0 |
| 720 / 10 | 0.445881248 / 0.171895057 | 0 / 0 |

固定 sigma 的完整 PW training forward/独立 native-reference regression（随机 Tiny 权重；本轮执行）：

| sigma | 修复前 | Noised latent 差 | Timestep embedding 差 | Target velocity 差 | Video loss 差 | 关键 gradient max abs |
|---:|---|---:|---:|---:|---:|---:|
| 0.12345 | 查表为空后 reshape 失败 | 0 | 0 | 0 | 0 | 0 |
| 0.37460 | 同上 | 0 | 0 | 0 | 0 | 0 |
| 0.82731 | 同上 | 0 | 0 | 0 | 0 | 0 |

容差均 **atol=rtol=0**，梯度覆盖 PW embedding、UND layer0 K/V、GEN output projection、time MLP。原始记录：[rf-0.12345.json](physword_pre_m4_fixes_results/rf-0.12345.json)、[rf-0.3746.json](physword_pre_m4_fixes_results/rf-0.3746.json)、[rf-0.82731.json](physword_pre_m4_fixes_results/rf-0.82731.json)。

采样得到的 sigma 原样用于加噪，不做 `sigma*N/N` 的 FP32往返；显式传 `timesteps=` 时按 t/N 定义 sigma；同时传 sigma/t 必须一致。固定 sigma 测试使用显式 `sigmas=`，避免把往返误差隐藏在参考中。

保持：`x_sigma=(1-sigma)*x0+sigma*epsilon`，`target=epsilon-x0`，历史有效 sigma=0，condition mask `[1,1,0,0,0,0]`；默认 loss 分母含全部6帧、历史仅在分子被mask，uniform weighting，video scale=10 **一次**。推理 scheduler 的 `flow_prediction` 与该 velocity 定义一致；完整采样/CFG/Joint cache 未在本轮扩展。

真实装配还修正了一个精度隐患：只将**参数**转为 BF16，保留派生 rotary/timestep buffer 为原生 FP32；否则整模块 `.bfloat16()` 会不可逆地舍入频率。Time MLP 仍按原生方式 FP32。`fp32_derived_buffers=True` 写入 network config，恢复检查能拒绝不同的 buffer 构造策略。本报告真实结果均来自修正后的 `real_edge_release`。

## 4. 真实资源与语义初始化

固定官方来源：`nvidia/Cosmos3-Edge@344d602b128d1bbdacb43b08d0a3626f46343e29`。先检查本地/cache/已有 checkpoint 与资源空间；已有其他机器人 checkpoint 未冒充 Edge。CPU 可用内存约1.7TiB、AIFS可用约5.5TB，满足必要资源装配。

仅补齐以下文件，没有下载示例视频/数据集。下载先 stage，复用约2.32GB已下载前缀，通过 range 分段补齐，校验完整大小/SHA256后替换指针。官方 API metadata 与 git LFS OID 一致。

| 文件 | 真实字节数 | SHA256 前缀 |
|---|---:|---|
| transformer shard1 | 5,000,040,072 | `116134e5492c` |
| transformer shard2 | 1,739,335,512 | `3b4c0aafe270` |
| vision_encoder/model.safetensors | 978,739,880 | `2180ad739ecc` |
| vae/diffusion_pytorch_model.safetensors | 1,409,400,600 | `230496cb59ff` |
| tokenizer.json | 17,077,500 | `4dc692a99dca` |

完整 hash 与最终校验见 [resources.json](physword_pre_m4_fixes_results/resources.json)，官方记录见 [official_resources.json](physword_pre_m4_fixes_results/official_resources.json)。最新 preflight：files_complete=True、metadata_compatible=True、missing=0。preflight 不声称已完成 tensor 检查；下面是另行执行的真实加载证据。

- 严格加载 UND/GEN **544** 个 tensor、原生 vision bundle **443** 个 tensor（包括 Projector），required missing / unexpected / shape mismatch 均为空。
- Checkpoint 中5个 Action head tensor 明确列为 RGB-only 未使用，不静默忽略关键模型参数。
- Tokenizer max ID **131071**；input embedding rows 与 LM-head rows 均 **131072**。
- 处理公开 HF config 到原生 Nemotron config 的字段映射：`rms_norm_eps→layer_norm_epsilon`、`hidden_act→mlp_hidden_act`、真实 mRoPE/rope_theta；直接将只读 alias 当构造参数的初次失败已修复，再经实际加载验证。
- 真实 published Wan VAE 严格加载并编码：5帧→`[1,48,2,4,4]`，21帧→`[1,48,6,4,4]`。采用发布配置的 latents_mean/std，history-only 与 full prefix 差 **0**。

初始化真实值：

| Class | Word | 原 Tokenizer subword IDs | PW embedding index | private ID |
|---:|---|---|---:|---:|
| 0 | align | [24181] | 1 | 1073741825 |
| 1 | approach | [13008,1664] | 2 | 1073741826 |
| 2 | grasp | [1870,11982] | 3 | 1073741827 |
| 3 | hold | [8681] | 4 | 1073741828 |
| 4 | lift | [1108,4083] | 5 | 1073741829 |
| 5 | lower | [40576] | 6 | 1073741830 |
| 6 | rotate | [48449] | 7 | 1073741831 |

Start 的 embedding index=0、private ID=1073741824，来源为真实 `<|im_start|>` input row **10**。上述 base Tokenizer IDs、class IDs、private IDs 三个空间分开保存。Input 与 Head 分别从自身权重做 FP32算术平均；实测 **weight tying=False**，没有依赖 config 猜测。

| 参数 | 随机占位 std / RMS | 真实语义初始化 std / RMS |
|---|---:|---:|
| PW embedding | 0.019981826 / 0.019981788 | 0.017382273 / 0.017383805 |
| Word head | 0.019951401 / 0.019951535 | 0.020384418 / 0.020383727 |

本地 loader 在构造/注册参数、加载 base 后初始化，再由 caller 建 optimizer；初始化函数拒绝已建 optimizer 或已初始化的对象，`init_weights` 和 warm-start 也不能覆盖已初始化/恢复的模型。Resume 只加载 PW tensor+provenance，真实恢复实测未重复初始化。完整统计：[real_edge.json](physword_pre_m4_fixes_results/real_edge.json)。

## 5. 最终 Task Vocabulary、Mask 与时间协议

按用户确认采用**左臂单轨**。冻结文件位于 [experiments/physword/badminton_left/vocabulary.json](../experiments/physword/badminton_left/vocabulary.json)；模型表为 **`[8,2048]`**，Word Head 为 **`[7,2048]`**。

词表从 **Train 的有效左臂标签**产生，按字典序固定；Test 标签不决定 head 类别。词表 fingerprint：`3d74b2a6532a95090dc0292b30a8210ba074530188939a7d5d9eacaaf6381b07`，包含词义与 effector。

| Word | Train valid labels | Train频率 | Test valid labels |
|---|---:|---:|---:|
| align | 34 | 3.5270% | 15 |
| approach | 1 | 0.1037% | 2 |
| grasp | 72 | 7.4689% | 35 |
| hold | 721 | 74.7925% | 620 |
| lift | 36 | 3.7344% | 28 |
| lower | 64 | 6.6390% | 0 |
| rotate | 36 | 3.7344% | 28 |

Train 247 windows，988 labels，masked24（**2.42915%**），完全 word-valid 223 windows；Test 186 windows，744 labels，masked16（**2.15054%**），完全有效175 windows。左臂总mask40/1732=**2.30947%**；不同于此前双臂合并的3.81062%，没有通过填错标签清零。

Word CE 只使用 valid labels；未来RGB完整时 latent mask 可保持 True，即使该 block 的 Word 被mask。真实数据+Tiny 的验收样本是 word_count=3、latent_count=4。前一词无效时反馈明确的 neutral start，禁止把 class0/任意假GT当有效词；未知 task 标签按记录的 eval policy mask CE，不改成正常动作类。原始25类文件不动；overlay 的 per-arm 字段也映射到同一V7空间，原始字段以 `source_*` 保存溯源。

全局源 split 未改。本轮在尚未进行真实模型拟合的两个已标注 episode 中，建立独立 pilot overlay：episode_0001→Train，episode_0017→Test。它是明确记录的 pilot holdout，**不是已有全局20个Test episode已经完成标注**；没有 Val，不能反复用 Test 调参。源 windows SHA256 保持 `a870319f...31ca8`，原数据 worktree clean。

实际时间参数严格为10Hz、history5、future16、K4、stride4；PW与video未来范围均 **1.6s**。第一窗口PW target frames `[8,12,16,20]`，对应未来容器 `[3,7,11,15]` 和 full VAE右边界 `[2,3,4,5]`；历史latent `[0,1]`。Model clock 与选择到的真实 capture timestamp 的区别保留在原metadata中，label取真实帧，不把未来帧送入 Reasoner。

正式可消费的派生数据：`/aifs/lijiawang/physword/pre_m4_fixes_results/badminton_left_release/`。其windows SHA256为 `3f7e2a855f39c49387b64b020b2f2f4fffc6995dcfed732975ec09049b812e49`，见 [protocol.json](../experiments/physword/badminton_left/protocol.json)。原来的25类数据桥接仍会明确拒绝误接，必须提供正确 task config。

## 6. A+B+C、Loss、梯度与真实更新

数据流保持：

```text
History RGB + Instruction → frozen native SigLIP2/Projector → shared prefix
A: prefix + start + previous valid GT words → UND final states [1,4,D]
   → V-way CE + FP32 cosine to independent frozen future-image targets
B: original prefix → no-grad K=4 greedy → Predicted IDs
C: original prefix + Predicted IDs → fresh repack → native Joint MoT
   GEN = clean history + noisy future → native flow loss × native scale
L_total = L_video + lambda_word*L_word + lambda_latent*L_latent
```

A 的GT不复用到C；B不接收未来GT；C重新 lookup PW embedding，不通过 argmax 求导。GEN每层读取原生 UND K/V。视觉编码器、Projector、VAE冻结；最后 UND 层 FFN/Q/O/final norm 没有直接 video 梯度符合图，并非断裂。

Legacy Synthetic Tiny 的独立 backward（本轮重跑）：

| Loss | 数值 | PW table grad norm | Word head | UND | GEN | UND-for-GEN K norm |
|---|---:|---:|---:|---:|---:|---:|
| Video，已scale | 12.60267258 | 0.12960735 | 0 | 1.87047524 | 8.27421873 | 0.00893324 |
| Word CE | 2.60927892 | 2.18998981 | 5.83854723 | 9.34323109 | 0 | 0 |
| Latent | 1.01446438 | 1.10371351 | 0 | 6.97057306 | 0 | 0 |

raw video=1.26026726，total=16.22641563。Tiny Phase B 每组完整参数名、LR、每 tensor Grad Norm 与 update norm 在 [step-B.json](physword_pre_m4_fixes_results/step-B.json)；所有预期可训练 tensor实际改变、冻结参数不变。不是只有非零总梯度。

真实标注+真实RGB（显式resize为64²）配随机原生 Tiny：A total=2.75743628（CE1.76604831、latent0.99138796）；B total=15.91707516（video13.15963840、CE/latent同前）。这是数据接入和更新测试，不是学习效果。

真实预训练 Edge，同样只做64²诊断：

| 项目 | 真实实测 |
|---|---:|
| Phase A CE / latent / total | 2.39783931 / 0.99464858 / 3.39248800 |
| PW table 参数量 / Word head 参数量 | 16384 / 14336 |
| LR | 均1e-4 |
| 更新范围 | PW table + Word head；UND/GEN/视觉全部冻结 |
| 冻结参数检查 | PASS，逐参数与step前snapshot比较 |

真实完整参数名/Grad Norm/实际差异见 real_edge.json 的 `phase_a_update.groups`。本轮真实更新没有解冻 UND/GEN，不能外推为完整 Phase B optimizer step已通过。

## 7. Baseline、可见性与 Checkpoint 恢复

三种配置：[native.json](../experiments/physword/badminton_left/native.json)、[matched_rgb.json](../experiments/physword/badminton_left/matched_rgb.json)、[full_pw.json](../experiments/physword/badminton_left/full_pw.json)。相同 snapshot、GEN历史、future、noise、sigma、seed、分辨率/评测split；Matched与Full的历史RGB IDs/features/positions严格相同，Matched中没有 start/PW private ID。配置 `max_train_steps=0`，未触发正式训练。

真实单步 baseline 均 forward通过；PW-off vs 原始无PW embedding路径 **max abs=0**。Full模型换第二个预测词：

- velocity RMS difference **0.02191142179**，相对差 **0.02089314413（2.089314%）**。
- 相同输入重复 numerical floor **0**。
- 屏蔽每层完整7-token suffix后替词差 **0**。
- 当前 AR 结果 `[lift,lift,lift,lift]`，仅证明真实路径可执行，**没有证明任务语义预测正确或视频质量提升**。

Legacy Tiny 干预仍是 RMS差0.00358431763、相对0.367363%、floor0、完整suffix masked差0。不能把真实模型与Tiny之间的差值大小当训练收益。

Checkpoint schema2 保存/检查：完整network state/config、PW映射/词义/effector/hash、初始化来源、processor/tokenizer hashes、optimizer、LR scheduler、EMA/Scaler（启用时）、global step、accumulation phase及未提交梯度、Python/NumPy/Torch CPU/本设备CUDA RNG、显式data cursor或sampler状态、RF与数据/实验manifest。EMA decay/常用LR callable也进入配置兼容检查。

Tiny 完整 A+B+C 中断恢复：fresh RF time/noise、Python/NumPy/Torch/CUDA RNG、样本顺序、Cosine scheduler、启用的EMA/Scaler、下一步loss及全部参数均 exact一致；另外测试累积1个microbatch保存梯度后继续更新，差0。修改RF shift或data split hash会拒绝恢复。

**真实 Edge 恢复实测**：保存step1后建立新模型，`initialize_pw=False`，恢复全部参数/optimizer/scheduler/数据cursor，比较不中断与恢复后的下一真实窗口：

| 项目 | 结果 |
|---|---|
| Word logits、single-step velocity | max abs=0 |
| AR predicted IDs | 相同 |
| 下一步 CE/latent/total loss | max abs=0 |
| 下一步 PW/head 参数更新 | max abs=0 |
| Torch CPU/CUDA RNG、LR scheduler | 相同 |
| cursor / accumulation | 1 / 0，下一窗口为真实Train窗口1 |
| Resume语义初始化 | 未重复执行 |

真实checkpoint为 **7,711,326,535 bytes**，位于 `/aifs/lijiawang/physword/pre_m4_fixes_results/real_edge_release/checkpoint/`；仅小型manifest纳入Git：[real_checkpoint_manifest.json](physword_pre_m4_fixes_results/real_checkpoint_manifest.json)。真实A使用BF16，无EMA/Scaler；其启用恢复由Tiny测试覆盖。旧schema1 diagnostic bundle不冒充完整resume，当前loader明确拒绝；不静默把16类旧PW重排到7类。

## 8. 本轮测试状态矩阵

| 项目 | Tiny/Synthetic | 真实标注数据 | 真实Edge权重 |
|---|---|---|---|
| RF指定连续sigma数值/梯度 | **PASS**，三sigma+9采样配置 | **PASS**，Tiny实际RGB/标签采用非整数t | **PASS**，真实单步σ=0.3746；三sigma全梯度对照 NOT RUN |
| Semantic initialization | **PASS**，tied/untied、多subword、顺序保护 | **PASS**，左臂manifest接入 | **PASS**，真实input/head rows与tying |
| Dynamic vocabulary / masks | **PASS**，V=3与非法标签 | **PASS**，V=7、3 CE/4 latent单步 | **PASS**，V=7真实A；独立Loss梯度全套仍为Tiny证据 |
| T01 off regression | **PASS** | — | **PASS**，off max abs0 |
| T02 presence/packing | **PASS**，含不等长samples | **PASS**，任务映射 | **PASS**，真实Full及Matched条件可运行 |
| T03 GT/future leakage | **PASS**，全部既有反例重跑 | **PASS**，只读target边界/单轨映射 | **NOT RUN** 独立真实GT交换测试，不冒称通过 |
| T04 reference attention | **PASS** forward；reference backward **NOT RUN** | — | 独立reference **NOT RUN** |
| T05 cross sample | **PASS** M1 varlen；M2多样本 **NOT RUN** | B=1 | B=1 |
| T06 AR/repack per-layer parity | **PASS** BF16及AMP | 模型边界同一builder | 独立逐层真实对照 **NOT RUN** |
| T07 intervention | **PASS** | 真实RGB进入Tiny | **PASS**，真实差/floor已记录 |
| T08 full suffix mask | **PASS** | — | **PASS**，masked替词差0 |
| T09 independent losses | **PASS**，Video/CE/latent分别backward | **PASS**，Tiny A+B+C总loss | 真实A联合backward **PASS**；真实独立Video/CE/latent **NOT RUN** |
| T10 actual update | **PASS** Phase A/B全部指定组 | **PASS**，Tiny真实标注 | **PASS** PW/head；完整Phase B **BLOCKED** 显存 |
| T11 train/infer condition parity | **PASS** | 同typed builder | 独立真实grad/no-grad parity **NOT RUN** |
| T12 CFG/Joint cache | **NOT RUN**，M5范围 | — | **NOT RUN** |
| T13 VAE/future leakage | **PASS** | 时间/帧边界 **PASS** | **PASS** VAE真实history prefix差0；全部future-swap反例 NOT RUN |
| T14 / full resume | **PASS**，含EMA/Scaler/累积状态 | **PASS**，保存data cursor、真实下一窗口 | **PASS** 真实A logits/AR/video/下一更新恢复；完整B更新恢复 BLOCKED |

另有原 `inference/model_test.py` 测试尝试因当前Python3.11缺 `typing.override` 在collection时 **BLOCKED**。没有为此修改无关全引擎兼容代码；提取的纯key/index映射与原始NVIDIA实现的独立测试通过，真实544+443 tensor加载也通过。该BLOCKED不能写成原生全inference测试通过。

## 9. 剩余限制与下一轮 Astra 复审重点

1. **完整真实 Phase B 更新：BLOCKED。** 原生 exact-name selector 在真实checkpoint形状上的库存：PW16384、head14336、UND201345024、GEN1414921408、K-norm3584，共1,616,300,736 trainable parameters。仅混合精度参数+FP32梯度+Adam状态，连同冻结基座，最低 **28.2464GiB**，尚不含激活、VAE和workspace；当前每卡仅约10.11GiB空闲。该库存用meta参数计算，**不是实际optimizer更新**。详见 [real_phase_b_memory.json](physword_pre_m4_fixes_results/real_phase_b_memory.json)。需要单卡释放足够显存后先做真实B单步；不以未经验证的FSDP/多卡绕过。
2. 真实A已证明PW/head-only更新及恢复；解冻所选UND层、480p/完整时间配置的显存尚需有界单步确认。真实140测试中的多项精细结构/独立梯度仍以Tiny证据为主，矩阵已逐项说明。
3. 当前只有2个已标注episode、无独立Val，标签尚未人工review；hold占Train有效词74.79%，approach仅1个重叠窗口标签。pilot holdout是最小接口闭环，不是统计可信的泛化结论。下一轮先人工核对少量目标片段/边界，不强填mask。
4. 三种baseline的配置、条件和真实单步forward已具备；正式同预算SFT、完整视频生成/质量评测尚未做。64² smoke不等于480p质量；10Hz/21帧也不能凭本轮结果宣称已获官方预训练质量保证。
5. M3仍是单设备B=1 API；默认TOML trainer接入、Joint CFG/cache、FSDP/AC保持原范围限制，没有提前宣称支持。T04 gradient reference、完整真实T03/T06/T11反例可在复审时补测。

建议 Astra 下一轮优先复核：连续采样/helper是否与原始分布/shift一致；buffer FP32策略；真实subword/tying；task manifest与旧source标签空间；schema2的严格恢复和实际下一更新；真实B显存下界及未执行项目。**本报告提交修复证据，不替代 Astra 的独立 GO/NO-GO 判定，不自动启动 M4。**

## 10. 可复现命令

工作目录均为本仓库。实际模型回归：

```bash
CUDA_VISIBLE_DEVICES=7 I4_ATTN_BACKENDS=flash3 COSMOS_TRAINING=0 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /tmp/physword-m01-venv/bin/python -m pytest \
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
  -q --junitxml=docs/physword_pre_m4_fixes_results/model_tests.xml
```

数据回归：

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /tmp/physword-data-venv/bin/python -m pytest \
  -c tools/physword_data/pytest.ini --confcutdir=tests/physword_data \
  tests/physword_data -q --junitxml=docs/physword_pre_m4_fixes_results/data_tests.xml
```

冻结任务（要求新输出目录，保留源数据）：

```bash
PYTHONPATH=. /tmp/physword-data-venv/bin/python -m cosmos_framework.data.generator.physword.task \
  --windows /aifs/lijiawang/physword/data_pilot/badminton/scene_v2/windows.jsonl \
  --vocabulary tools/physword_data/scene_v2/vocabulary.json --task badminton --effector left \
  --test-episode badminton__episode_0017 \
  --output /aifs/lijiawang/physword/pre_m4_fixes_results/badminton_left_release
```

真实Edge有界验收（已执行；包含真实A中断恢复比较）：

```bash
CUDA_VISIBLE_DEVICES=6 I4_ATTN_BACKENDS=flash3 COSMOS_TRAINING=0 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8 PYTHONPATH=. \
/tmp/physword-m01-venv/bin/python -m cosmos_framework.scripts.physword_real_probe \
  --checkpoint /aifs/lijiawang/physword/Cosmos3-Edge \
  --task-dir /aifs/lijiawang/physword/pre_m4_fixes_results/badminton_left_release \
  --output /aifs/lijiawang/physword/pre_m4_fixes_results/real_edge_release \
  --phase-a-step --checkpoint-roundtrip
```

真实数据+Tiny（已执行，使用最终release数据）：

```bash
CUDA_VISIBLE_DEVICES=7 I4_ATTN_BACKENDS=flash3 COSMOS_TRAINING=0 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 PYTHONPATH=. \
/tmp/physword-m01-venv/bin/python -m cosmos_framework.scripts.physword_task_probe \
  --task-dir /aifs/lijiawang/physword/pre_m4_fixes_results/badminton_left_release \
  --output /aifs/lijiawang/physword/pre_m4_fixes_results/real_data_tiny_release
```

资源复核（已执行；需要补齐时显式追加`--download`，仅处理五个资源，支持curl断点续传）：

```bash
/tmp/physword-m01-venv/bin/python tools/physword_resources.py \
  --checkpoint /aifs/lijiawang/physword/Cosmos3-Edge \
  --metadata docs/physword_pre_m4_fixes_results/official_resources.json \
  --output docs/physword_pre_m4_fixes_results/resources.json
```

环境：Torch2.10.0+cu128、Transformers4.57.1、原生Flash3/H100。测试中的 `COSMOS_TRAINING=0` 不关闭 autograd；T09/T10及真实A均显式训练/更新。Diffusers对发布VAE的 `clip_output=False` 元数据给出不识别警告；未设为True，没有启用额外clip，实际key/shape/encode/prefix检验通过。Processor的无效可选crop参数警告没有被当作恢复通过证据，恢复是否一致由上述实际0差比较判定。

**已停止于修复与轻量验收。未开始正式数据拟合、长时间H100训练、多卡任务或M4论文消融。**
