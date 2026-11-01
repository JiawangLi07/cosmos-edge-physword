# PhysWord × Cosmos3-Edge：实际架构审计

> 历史基线文档（原始代码 `55f0554`）。当前实现与验收状态见 [PhysWord 项目说明](../README_PHYSWORD.md) 和 [M3 验收](physword_m3_acceptance.md)。文中的本地论文/权重路径是审计来源，不随源码仓库发布。

日期：2026-11-01。代码版本 `55f0554d0bf167603432041c794a8babbac759b9`。本次仅设计。

输入资料：[V3 交接文档](../Physword%20x%20cosmos-edge.md)、[本地论文 PDF](../physical%20word.pdf)。PDF 首页标识为 arXiv:2610.26854v3，正文日期 2026-10-22；在线书目信息可由 [arXiv](https://arxiv.org/abs/2610.26854) 核对。设计以用户提供的 PDF 和本地代码为准。

## 1. PhysWord 中实际迁移的机制

论文 §3.3–3.5，PDF 第 4–5 页定义的是：**用预测位置的 contextual output state `z_k` 同时预测动作词标签并对齐未来视觉 latent**。它不是仅对动作名 embedding 做分类，也不是令 `z_k = embedding(w_k)`。

- 16 类原子词：approach、move、pick、place、flip、rotate、toggle、open、close、push、pull、insert、press、click、strike、handover。
- 原论文同时包含视触觉、左右执行器；完整 block 内 full attention、同执行器跨 block causal、跨执行器禁止访问。
- 每个 block 的输入来自上一时间 block；第一个 block 使用 start token。当前 block 的输出监督下一目标时刻，不能输入同一时刻 GT 词再监督它。
- 视觉 target 为冻结 Qwen3-VL 视觉编码器对未来图像的 spatial tokens pooling；未来图像仅作为 target。
- 原论文 temporal branch 会提供预测特征给 reasoning/action 分支；用户所需的“UND 不读取 GEN 视频，GEN 读取 PW”是本项目明确的适配设计，不能宣称完全复刻论文 Mask。

本项目保留“词语义锚点 + 未来视觉 latent 对齐 + 跨未来 block AR”的机制；采用 Edge 现有 SigLIP2 视觉编码器替代论文编码器，不引入论文 Qwen3-VL/Cosmos-Predict2.5，也不引入 action/tactile 分支。

第一版为**一个视觉 Word / 时间 block / 一个语义轨迹**。这可退化为普通 token causal attention。若数据包含同时发生的双臂动作，需要另行确定 per-effector 标签和 block Mask；不能把两组并发词悄悄合并成一个标签并称作论文原实现。

## 2. Framework 联合模型与独立 HF 模型不是同一个入口

| 路径 | 本地入口 / 类 | 本项目用途 |
|---|---|---|
| Framework 联合 MoT | [edge_model_config.py:153](../cosmos_framework/configs/base/experiment/sft/models/edge_model_config.py:153) 选择 `Nemotron3DenseVLTextForCausalLM` | 主要改动路径 |
| 双路径 Transformer | [unified_mot.py:2833](../cosmos_framework/model/generator/mot/unified_mot.py:2833)，`Nemotron3DenseVLTextModel`、`MoTDecoderLayer` | UND 和 GEN 共用层迭代，但参数分开 |
| 视频网络包装 | [cosmos3_vfm_network.py:238](../cosmos_framework/model/generator/mot/cosmos3_vfm_network.py:238) | embedding / latent IO、packing、位置编码、输出解码 |
| 独立 HF Edge Reasoner | [modeling_cosmos3_edge.py:742](../cosmos_framework/model/generator/reasoner/cosmos3_edge/modeling_cosmos3_edge.py:742) | 独立 HF 兼容实现；仅改此文件不会改变上述 Framework 联合路径 |

`Nemotron3DenseVLMoTConfig` 将旧 Nemotron 配置中交替 attention/MLP 的 56 个 block 合并解释为 28 个有效 decoder layers，见 [unified_mot.py:473](../cosmos_framework/model/generator/mot/unified_mot.py:473)。本地 HF metadata 也声明 28 层。不能按 56 层再配对一次，也不能另外实例化第三、第四个 Transformer。

## 3. 参数和 Tensor Shapes

以下值来自本地 [config.json](../Cosmos3-Edge/config.json)、[transformer/config.json](../Cosmos3-Edge/transformer/config.json)、[vae/config.json](../Cosmos3-Edge/vae/config.json)，并与 Framework 构造代码交叉核对。

| 项目 | Edge 值 |
|---|---|
| UND / GEN hidden width | 2048 / 2048 |
| 有效 MoT layers | 28 |
| Q heads / KV heads / head dim | 16 / 8 / 128，GQA ratio=2 |
| UND / GEN FFN intermediate | 9216，ReLU² |
| base LM vocabulary rows | 131072；实际 tokenizer 长度尚未运行核验 |
| UND Q/K norm | identity |
| GEN Q/K norm | RMSNorm |
| GEN 访问 UND 的额外 K norm | `k_norm_und_for_gen`，开启 |
| RoPE theta / sections | 1e8 / [24,20,20]，half-dim sections 合计 64 |
| GEN VAE channels | 48 |
| VAE spatial / temporal compression | 16 / 4 |
| GEN latent spatial patch | 2×2，patch dim=48×2×2=192 |
| Edge 视觉 encoder width | 1152，patch 16，spatial merge 2×2 |
| 视觉 projector | 4608 → 11520 → 2048 |
| GEN base FPS / text-media temporal margin | 24 / 15000 |

令单样本 UND 长度为 `U_i`，GEN patch 数为 `G_i`；`N_U=ΣU_i`、`N_G=ΣG_i`。实际 kernel 还可能附加 padding segment，下面是 real-token shape。

```text
PackedSequence.text_ids       [N_U] int64
PackedSequence.text_indexes   [N_U] int64，指向全 packed buffer
PackedSequence.position_ids   [3, N_U+N_G]
packed input embedding        [N_U+N_G, 2048]
UND / GEN hidden streams      [N_U,2048] / [N_G,2048]
Q_und / Q_gen                 [N_U,16,128] / [N_G,16,128]
K,V_und / K,V_gen             [N_U,8,128] / [N_G,8,128]
attention kernel input        [1,N_stream,H,D]，offsets 描述实际 batch
AR reasoner hidden            [B,T,2048]
AR ReasonerKVCache per layer  K,V: [B,T_cached,8,128]
network last_hidden_state     [N_U+N_G,2048]（CP gather 后）
```

代码：[unified_mot.py:711](../cosmos_framework/model/generator/mot/unified_mot.py:711)、[946](../cosmos_framework/model/generator/mot/unified_mot.py:946)、[cosmos3_vfm_network.py:1859](../cosmos_framework/model/generator/mot/cosmos3_vfm_network.py:1859)。

给定 `T_pix=1+4m`、尺寸可整除 32 的视频：

```text
x0, x_sigma: [1,48,1+(T_pix-1)/4,H/16,W/16]
patchified:  [T_lat * (H/32) * (W/32),192]
vae2llm:     [...,192] → [...,2048]
llm2vae:     [...,2048] → [...,192]
```

非整除的 latent H/W 会在 `patchify_and_pack_latents()` pad 到 patch 的倍数，并在反向 unpatch 时裁回。见 [cosmos3_vfm_network.py:612](../cosmos_framework/model/generator/mot/cosmos3_vfm_network.py:612)。不要把 48-channel VAE latent 直接当作与 2048 维 PW state 同空间的目标。

## 4. 两个 Transformer 怎样交互

每层包含两套独立参数：

```text
UND: input_layernorm → q/k/v_proj → UND causal attention → o_proj
     → residual → post_attention_layernorm → mlp → residual

GEN: input_layernorm_moe_gen → q/k/v_proj_moe_gen
     → Attention(Q_gen, [K_und_for_gen;K_gen], [V_und;V_gen])
     → o_proj_moe_gen → residual → post_attention_layernorm_moe_gen
     → mlp_moe_gen → residual
```

`moe_gen` 是这个双路径实现的参数命名；Edge 此处为 dense FFN，不是引入第三个专家 Transformer。见 [unified_mot.py:1334](../cosmos_framework/model/generator/mot/unified_mot.py:1334)。

这一交互是**同层的联合 softmax**。不能替换成单独做一遍 cross-attention 再与 self-attention 输出直接相加，认为与预训练算子相同。

基础可见性，行是 Query，列是 Key：

| | UND context / PW | GEN clean history / noisy future |
|---|---|---|
| UND | 同一样本内 causal | 禁止 |
| GEN | 同一样本内全部 | 同一样本内全部（two-way） |

条件历史 VAE tokens 仍属于 GEN；它们只是从 flow loss 中排除，**不会因此变成 UND 可读的视觉输入**。路由见 [cosmos3_vfm_network.py:1706](../cosmos_framework/model/generator/mot/cosmos3_vfm_network.py:1706)。这正是必须给 UND 单独提供现有 SigLIP2 历史特征的原因。

## 5. “PW hidden state”必须明确是哪一个 state

设 `p_k` 是预测 `w_k` 的位置，输入为 start 或 `w_{k-1}`；`z_k = norm(h^28[p_k])`。PhysWord CE 和 latent loss 监督这个 `z_k`。

但原 MoT 第 `ℓ` 层 GEN 读取的是：

```text
K_PW^ℓ = RoPE(RMSNorm_gen-key(W_K_und^ℓ · LN_und^ℓ(h^(ℓ-1)[p_k])))
V_PW^ℓ = W_V_und^ℓ · LN_und^ℓ(h^(ℓ-1)[p_k])
```

它读取预测轨迹的**逐层 contextual representations**，并非把最终 `z_k` 直接注入 Generator。当前 MoT 也不把最终 UND hidden state 回送到 GEN 第 1 层。

后果：

1. 同一 prefix / IDs / 位置 / causal mask / 权重下，AR 与 repack 的 UND states 在数学上可比较；不同 kernel 的浮点误差仍需容差检验。
2. 单纯文本 repack、丢 RGB 或换 chat template / mRoPE，不满足等价条件。
3. `L_video` 可以更新生成侧读取的 UND K/V 通路，但不会通过本次 joint forward 的 GEN 输出反传到最后一层 UND FFN、最终 UND norm 或 word head。这些模块没有到 GEN 输出的计算边。
4. 若验收要求是“Generator 必须直接拿到最终 `z_k`”，需要额外的 soft-prefix 回填或经过验证的 per-layer handoff；不能用普通 token repack 冒充。

## 6. 两种视觉表示与 preprocessing

独立 Reasoner 使用 [reasoner_multimodal_utils.py:38](../cosmos_framework/model/generator/reasoner/nemotron_3_dense_vl/reasoner_multimodal_utils.py:38)：

```text
历史 RGB → native processor → SigLIP2 → PatchMerger
         → [N_merged_patches,2048]
         → masked_scatter 到 image/video placeholder 位置
         → get_rope_index → [3,B,T] + position_deltas
```

Edge 无 Qwen deepstack；helper 的 deepstack 返回空列表。SigLIP2 逐帧编码，时间信息由 token 顺序、timestamp 和 mRoPE 提供。`_ensure_vision_tower()` 延迟加载该现有视觉塔，见 [unified_mot.py:2983](../cosmos_framework/model/generator/mot/unified_mot.py:2983)。

注意这段加载代码只调用 `.eval()`，并不等于 `requires_grad_(False)`。未来视觉目标必须显式冻结 encoder **和 projector**。初始化也不能在新训练参数已被 FSDP / optimizer 收集后才悄悄添加。

Native processor 特别说明 checkpoint 的 patch 排列与某些 transformers-main 实现不同，见 [cosmos3_edge_processing.py:11](../cosmos_framework/data/generator/processors/cosmos3_edge_processing.py:11)。方案应复用仓库 native processor，不任意换成新版本 AutoProcessor。

GEN 使用 Wan2.2 causal VAE 和自己的 latent 3D positions。`get_3d_mrope_ids_text_tokens` 默认将文字三轴设为相同递增整数；这不等于 Reasoner 多模态 grid，见 [mrope.py:40](../cosmos_framework/data/generator/sequence_packing/mrope.py:40)。

## 7. 训练和推理还有哪些真实差异

| 方面 | 当前代码 | PW 设计要求 |
|---|---|---|
| 文本来源 | train: dataset `text_token_ids`；infer: caption template | 用共享的 typed prefix builder 生成同一协议 |
| PW CE | pack 有一般文本 labels；网络可输出 CE；视频 trainer 没有消费 | 单独选择 PW prediction positions 和 labels |
| hidden state 暴露 | network 返回，`denoise` 丢弃 | 只返回需要的 `[B,K,2048]`，避免持有全部序列 |
| inference KV | 首个 joint step 建 cache，后续复用，保存时 detach | 训练关闭；推理 PW prefix 变化必须失效 |
| Attention backend | training 默认 varlen；单样本 no-grad 可 dense | 数值 parity 测试必须覆盖两条实际路径 |
| FSDP | block `reasoner_forward`、top `generate_reasoner_text` 特别注册 | 新 PW forward 入口也需要注册 |
| checkpoint | 原 LM / GEN 权重、tokenizer revision 由不同配置提供 | 固定 manifest、补齐新 PW 参数、严格校验 missing keys |

官方 Edge SFT tokenizer revision 写死为 `2a00e87...`，而本地元数据快照是 `344d602...`；未运行权重/词表兼容测试，不能将二者默认为相同。现有 `.safetensors` 和 `tokenizer.json` 是 LFS 指针，首个执行里程碑必须解决，但本阶段不下载大权重。

## 8. 审计后的架构选择

主方案：**统一 RGB-conditioned UND prefix + 预测 PW IDs repack + 原双 Transformer joint MoT**。

保持原双塔、原每层 QKV/FFN、GEN→UND K norm 和正常 two-way kernel。新增的是数据接口、少量词表参数/word head、显式 PW metadata、损失和控制流程。

下一步细化与验收顺序见 [Implementation Plan](physword_implementation_plan.md)。本次不执行其开发里程碑。
