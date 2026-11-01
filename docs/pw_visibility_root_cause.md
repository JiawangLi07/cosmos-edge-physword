# PW → Video Generator 可见性根因审计

> 历史基线文档（原始代码 `55f0554`）。当前实现与验收状态见 [PhysWord 项目说明](../README_PHYSWORD.md) 和 [M3 验收](physword_m3_acceptance.md)。文中的本地论文/权重路径是审计来源，不随源码仓库发布。

审计日期：2026-11-01。仅架构审计；没有修改模型实现，没有启动训练。

固定源码：`cosmos-framework` commit `55f0554d0bf167603432041c794a8babbac759b9`。模型元数据：本地 `Cosmos3-Edge` commit `344d602b128d1bbdacb43b08d0a3626f46343e29`。以下行号均对应这份检出。本文的“确认”指源代码证据；神经网络数值实验尚未执行。

相关交付：[架构审计](physword_architecture_audit.md)、[Implementation Plan](physword_implementation_plan.md)。

## 1. 结论

**当前仓库没有把自主预测的 Physical Words 作为一种视频条件接起来。普通视频路径不生成 PW，独立 Reasoner 路径不继续生成视频；可选 upsampling 只将清理后的文本重新用作 caption。基础 two-way attention 并未禁止 GEN 读取 UND。**

因此，mentor 指出的现象在接口和数据流层面成立，但若将其解释为“Cosmos3-Edge 的共享 Attention 不支持 GEN→Reasoner”，则与当前代码不符。

另一个关键缺口是：**独立视觉 Reasoner 的输入嵌入和 mRoPE，没有进入视频联合 Forward 的 UND 输入构造。** 即使追加 PW IDs，默认视频路径也只会重算“caption + PW”的文本条件表征，不能直接认定这些表征等价于基于历史 RGB 生成 PW 时的隐状态。

## 2. 三条实际推理路径

### 2.1 普通视频生成：没有 PW AR 阶段

```text
OmniInference.generate_batch
  → OmniMoTModel.generate_samples_from_batch
  → _maybe_apply_prompt_upsampling（upsample_task=None 时原样返回）
  → _prepare_inference_data
      → _get_inference_text_tokens → _tokenize_captions
  → velocity_fn → _get_velocity
      → _pack_input_sequence → pack_input_sequence
  → denoise → Cosmos3VFMNetwork.forward
      → _encode_text（embedding lookup）
      → _encode_vision（VAE latent → GEN embedding）
      → build_packed_sequence → language_model → Shared Attention
```

证据：

- [inference.py:2464](../cosmos_framework/inference/inference.py:2464)：视频调用入口。
- [omni_mot_model.py:4065](../cosmos_framework/model/generator/omni_mot_model.py:4065)：视频采样；[4239](../cosmos_framework/model/generator/omni_mot_model.py:4239) 附近取得条件 tokens。
- [omni_mot_model.py:2898](../cosmos_framework/model/generator/omni_mot_model.py:2898)：条件和负条件都来自 captions。
- [packers.py:535](../cosmos_framework/data/generator/sequence_packing/packers.py:535)：将 `input_text_indexes` 交给 `pack_text_tokens`。
- [cosmos3_vfm_network.py:885](../cosmos_framework/model/generator/mot/cosmos3_vfm_network.py:885)：只执行 `embed_tokens(packed_seq.text_ids)`，并 scatter 到 `text_indexes`；没有 AR，也没有视觉占位符替换。

### 2.2 独立 Reasoner：生成后返回文本，没有接入视频

```text
generate_batch
  if model_mode.is_reasoner:
    return _generate_reasoner_batch
      → model.generate_reasoner_text
      → net.generate_reasoner_text
      → Nemotron3DenseVLTextForCausalLM.generate_reasoner_text
      → _impl_generate_reasoner_text
          → 多模态 prefill
          → reasoner_forward + layer-wise ReasonerKVCache
          → lm_head → sample next token
      → IDs 解码成文本 → reasoner_text.txt / SampleOutput
```

- [inference.py:2218](../cosmos_framework/inference/inference.py:2218)：Reasoner 分支直接 return，且拒绝在同一 batch 混用 Reasoner / 非 Reasoner 模式。
- [inference.py:3096](../cosmos_framework/inference/inference.py:3096)：输出 `reasoner_text`。
- [unified_mot.py:2113](../cosmos_framework/model/generator/mot/unified_mot.py:2113)：AR helper 被 `@torch.no_grad()` 装饰。
- [unified_mot.py:2458](../cosmos_framework/model/generator/mot/unified_mot.py:2458)：最终只返回 token IDs；没有返回 layer-wise cache 或 hidden states 给视频生成器。

AR 能力存在；“视频任务会自动调用它并使用其推理 token”则不成立。

### 2.3 Prompt upsampling：存在有限的文本桥，不能说“一概没有传递”

```text
generate_samples_from_batch(upsample_task=...)
  → _maybe_apply_prompt_upsampling
  → upsample_captions → generate_reasoner_text
  → clean_response：可能删除 thinking/reasoning blocks
  → cleaned text 写回 ai_caption
  → 重新套 caption chat template、分词、pack
  → 视频生成
```

证据：[omni_mot_model.py:6808](../cosmos_framework/model/generator/omni_mot_model.py:6808)、[7750](../cosmos_framework/model/generator/omni_mot_model.py:7750)、[upsampler/prompts.py:1105](../cosmos_framework/model/generator/upsampler/prompts.py:1105)。

准确判断：清理后保留的生成文本会影响视频；被删掉的 `<think>` 等内容不会进入 caption。原始 token IDs、原始多模态上下文 hidden states 和 AR KV 没有交接。I2V upsampling 传的是提取的 anchor image，并不自动形成所需的多帧历史 PW 推理流程。

## 3. Attention：具备基础访问能力，但不能创造未打包的 token

`two_way_attention()` 在 [attention.py:202](../cosmos_framework/model/generator/mot/attention.py:202)：

- UND Query：只取 causal UND K/V，进行因果 attention。
- GEN Query：取该样本 `get_all_seq*()` 的 K/V，即 UND + GEN。
- 多样本使用 varlen offsets 隔离；单样本无梯度快捷路径使用去 padding 的 KV。

`three_way_attention()` 在 [attention.py:399](../cosmos_framework/model/generator/mot/attention.py:399)：UND causal、GEN self-attention、GEN→UND attention 分别计算，后两者通过 LSE 合并；这也不是第三个 Transformer。

实际 dispatch 优先级见 [attention.py:764](../cosmos_framework/model/generator/mot/attention.py:764)：multi-control → multiview → three-way → two-way。标准 [vision_sft_edge.toml:17](../examples/toml/sft_config/vision_sft_edge.toml:17) 选择 two-way。没有用户的训练启动配置，不能据此宣称任何未来自定义配置都走相同路径。

Edge 还有专用 UND K 归一化：

```text
UND 自注意力：K_und 原始投影 → RoPE
GEN 读 UND：K_und → k_norm_und_for_gen → RoPE
```

实现：[unified_mot.py:607](../cosmos_framework/model/generator/mot/unified_mot.py:607)、[830](../cosmos_framework/model/generator/mot/unified_mot.py:830)。AR cache 只有自己的原始 UND K，直接复用时缺少这一路处理。

## 4. 训练侧与输出侧的缺口

| 发现 | 实际代码 | 意义 |
|---|---|---|
| 视频训练从已有 `text_token_ids` 读取文本 | [omni_mot_model.py:2245](../cosmos_framework/model/generator/omni_mot_model.py:2245) | 没有在线预测 PW 的步骤 |
| 官方白名单只有 GEN 和相关投影 / K norm | [vision_sft_edge.toml:42](../examples/toml/sft_config/vision_sft_edge.toml:42) | 新 PW embedding / word head / UND 参数不会自然被选入 |
| 白名单遗漏会设置 `requires_grad=False` | [optimizer.py:303](../cosmos_framework/utils/generator/optimizer.py:303) | 不是仅仅不给学习率，而是实质冻结 |
| 优化器仅收集 `model.net` 子树 | [optimizer.py:270](../cosmos_framework/utils/generator/optimizer.py:270) | 新 head 放在外层 model 下可能完全不被优化 |
| network 可返回 `last_hidden_state` / 可选 `ce_preds` | [cosmos3_vfm_network.py:1889](../cosmos_framework/model/generator/mot/cosmos3_vfm_network.py:1889) | 具备扩展接口，但还不等于训练到这些输出 |
| `denoise()` 只保留各模态预测与 `lbl_metadata_*` | [omni_mot_model.py:6990](../cosmos_framework/model/generator/omni_mot_model.py:6990) | `last_hidden_state` 和 `ce_preds` 在此被过滤 |
| `_compute_losses()` 汇总 flow matching / load balancing | [omni_mot_model.py:1902](../cosmos_framework/model/generator/omni_mot_model.py:1902) | 当前没有 Word CE 和 Future Visual Latent Loss |
| 推理文本 KV cache detach | [inference_text_kv_memory.py:39](../cosmos_framework/model/generator/mot/inference_text_kv_memory.py:39) | 不能直接拿它做训练可微条件通路 |

## 5. A–E 分类结果

| 类别 | 本地代码结论 |
|---|---|
| A：没有生成 | 普通视频路径成立；未定义 PhysWord 词表 / 专用解码流程。独立 Reasoner 有通用 AR 能力。 |
| B：没有传递 | 独立 Reasoner 到视频没有桥；upsampling 有 cleaned-caption 桥，但不交接原始推理表征。 |
| C：Mask 禁止 | 标准 two-way 不是根因。新增 token 必须属于同一样本 UND split，特殊路径仍需实测。 |
| D：未适配参数 | 官方视频 SFT 确实冻结 UND，缺少 PW 损失和参数组。 |
| E：训练推理不一致 | 当前二者都没有 PW 条件协议。训练读预分词 IDs，推理重分词 caption，直接只改一侧会造成新的不一致。 |

## 6. 最小修复边界

1. 使用现有 Nemotron UND 权重，新增固定 K 步的 PW 解码入口；返回结构化 IDs / 位置 / prefix 信息，避免文本清理和再次套模板。
2. 在打包 API 中显式支持 UND 视觉历史特征和其 mRoPE；联合 Forward 进行视觉占位符替换。
3. 把预测 PW 放入同一 causal UND split，重算联合 MoT；原 two-way kernel 保留。
4. 新增预测位置索引与损失输出；连接 `denoise` / loss 汇总 / optimizer / checkpoint。
5. 首版禁用 PW 路径的推理 KV 复用，正确性验证后复用**首次联合 Forward**产生的 UND KV；无需首先做 AR cache handoff。

更详细的数据结构、代码改动表和里程碑见 [Implementation Plan](physword_implementation_plan.md)。

## 7. 本次验证边界

已执行：固定版本与 Git 状态检查、源码调用链审阅、AST 定位、模型 JSON shape 核对、论文 PDF 解析与逐节相关机制核对。可复核的静态证据见 [audit_evidence.json](audit_evidence.json)。

未执行：带实际权重的 forward、GPU attention 测试、PW intervention、反向梯度测试、训练和视频指标评测。当前 shell 的 Python 无 `torch` / `transformers`，本地模型权重及 `tokenizer.json` 仍是 LFS 指针。因此本文不声称数值可见性、训练可用性或性能收益已经被验证；这些是实施计划的验收门槛。
