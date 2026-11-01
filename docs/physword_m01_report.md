# PhysWord M0/M1 实现与验证记录

日期：2026-11-01。分支：`physword/m0-m1-visibility`。基线 commit：`55f0554d0bf167603432041c794a8babbac759b9`。

## 已交付结果

**M1 的人工 Physical Word → UND PackedSequence → 原生 Two-way MoT → Video Velocity 通路已运行验证。M0 的真实大模型验证受本地 LFS 权重缺失限制；未下载大型模型，使用原始 commit 的 Tiny Edge 模型完成可执行基座验证。**

共 **74 项测试通过，0 失败、0 跳过**：20 项 M0/M1 新测试，54 项相关原有回归。测试使用实际 `Nemotron3DenseVLTextForCausalLM`、`Cosmos3VFMNetwork` 和未修改的 `two_way_attention()`，GPU kernel 为仓库支持的 FlashAttention 3。没有 mock/替换生产 Attention。

| 验收 | 结果 |
|---|---|
| PW-off 对原始 commit 回归 | baseline velocity 逐元素完全相等，`max_abs_diff=0` |
| 新 embedding 不改变共享权重 | PW-on/off 所有原有参数逐元素相等 |
| Token Presence | 人工 IDs / global indexes / 不等长 sample offsets / UND 路由通过 |
| Attention Visibility | 原生 two-way 与显式 visibility reference 对齐；GEN 可以读取 UND/PW |
| PW 不能读取 GEN | 改变视频 latents，全部 UND 最终 states 完全不变 |
| PW causal 顺序 | 替换后面的词，不影响前面的 context / PW states |
| PW Intervention | `pick → push`：future velocity RMS 差异 **0.013374722562730312**，最大差异 **0.0390625** |
| 跨样本隔离 | 改变样本 B 的词和视频，样本 A 的 states/velocity 完全不变 |
| 原默认行为 | 原 attention、双 Transformer 实现和 `vision_sft_edge.toml` 与基线逐字节一致 |

结果索引：[summary.json](physword_m01_results/summary.json)、[完整 JUnit](physword_m01_results/all_tests.xml)、[原始基线](physword_m01_results/original_baseline.json)、[M1 干预与 trace](physword_m01_results/m1_probe.json)、[跨 commit 对比](physword_m01_results/baseline_comparison.json)。

这组随机权重数值响应证明信息通路可用，不代表动作词已具有正确物理语义。没有执行训练、辅助分类、AR prediction 或最终 z 注入。

## M0：资源与兼容性

本地模型 snapshot：`344d602b128d1bbdacb43b08d0a3626f46343e29`。离线检查确认 metadata 中 UND/GEN width、层数、GQA、VAE channels / temporal compression 和 latent patch width 相容。实际权重 shapes 与 tokenizer ID 兼容性尚不能验证，因为以下文件都是 LFS 指针：

| 唯一缺失文件 | 指针标记的真实大小（bytes） |
|---|---:|
| `transformer/diffusion_pytorch_model-00001-of-00002.safetensors` | 5,000,040,072 |
| `transformer/diffusion_pytorch_model-00002-of-00002.safetensors` | 1,739,335,512 |
| `vision_encoder/model.safetensors` | 978,739,880 |
| `vae/diffusion_pytorch_model.safetensors` | 1,409,400,600 |
| `tokenizer.json` | 17,077,500 |

Reasoner 和 Generator 的索引共享部分 shard，故检查清单中会出现重复引用，不应重复下载。`text_tokenizer/tokenizer.json` 也仍是 LFS 指针；根目录 tokenizer 尚未可加载。官方 Edge SFT 还固定了旧 tokenizer revision，仍需在选择真实权重时核对。

检查器 [physword_preflight.py:17](../cosmos_framework/scripts/physword_preflight.py:17) 使用标准库识别 missing / LFS pointer / 非法 JSON / 截断 safetensors header 和 data ranges，**不会调用任何模型下载 API**。可直接用系统 Python 运行。完整输出：[m0_preflight.json](physword_m01_results/m0_preflight.json)。

当前已有 Python 环境是 torch 2.7 / transformers 5.x，与这份仓库不匹配。新建 `/tmp/physword-m01-venv`，复用原环境的基础包并局部覆盖依赖；没有修改原有 Conda 环境。主要运行版本：

```text
Python 3.11
torch 2.10.0+cu128
torchvision 0.25.0+cu128
transformers 4.57.1
huggingface-hub 0.36.0
flash-attn-3-nv 1.0.3+cu128.torch210
triton 3.6.0
numpy 1.26.0
pyarrow 20.0.0
pandas 2.2.3
ruff 0.12.7
```

下载的只有运行依赖，没有下载 Cosmos / SigLIP2 / VAE 模型权重。测试期间启用 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`。

## 实现范围与代码位置

1. [physword/config.py](../cosmos_framework/model/generator/physword/config.py)：`PhysWordConfig`，默认 `enabled=False`；支持 dict / structured OmegaConf。
2. [conditioning.py:39](../cosmos_framework/model/generator/physword/conditioning.py:39)：16 类 visual words 与 start 的独立 mapping；`PhysicalWordEmbedding` 仅有 17 行；`physical_word_trace()` 输出 global indexes / sample offsets。
3. [packers.py](../cosmos_framework/data/generator/sequence_packing/packers.py)：在原生 caption pack 计算边界之前追加 `start + manual words`。原生 EOS/BOV、mRoPE、sample offsets 和 UND routing 继续由现有 packer 生成。
4. [cosmos3_vfm_network.py](../cosmos_framework/model/generator/mot/cosmos3_vfm_network.py)：仅开启 PW 时建立小 embedding；`_encode_text` 对私有 IDs 使用独立行，其余 IDs 仍用原 embedding。关闭时走原 lookup，不新增参数。
5. [model_config.py](../cosmos_framework/configs/base/defaults/model_config.py) 与 [omni_mot_model.py](../cosmos_framework/model/generator/omni_mot_model.py)：默认关闭字段；构建网络和 `_pack_input_sequence(..., physical_words=...)` 的薄透传。
6. [physword_probe.py:32](../cosmos_framework/scripts/physword_probe.py:32)：可对原始 commit 和新分支运行的固定 seed Tiny forward；保存 baseline / pick / push velocity tensors。
7. [visibility_test.py](../cosmos_framework/model/generator/physword/visibility_test.py)、[preflight_test.py](../cosmos_framework/scripts/physword_preflight_test.py)、[测试 profile](../tools/physword_m01_pytest.ini)：新增测试和隔离的收集配置。

独立 ID namespace 从 `1<<30` 开始：start=`1073741824`，approach=`1073741825`，pick=`1073741827`，push=`1073741834`。这些是 **Framework 私有 IDs，不是 HF Tokenizer IDs**；原 embedding table 不扩容。私有 IDs 被安全替换后才调用原 embedding，再在这些位置使用 PW 小表，避免越界。

启用模式仅支持 RGB + 单 caption、原 two-way、无 action/sound/其他传感器、无全词表 CE。非法模式直接报错。没有添加 Word Head、训练 loss、额外 Transformer 或 Final-z Injection。

## 最小调用协议

```python
pw_config = {"enabled": True}
network_config = Cosmos3VFMNetworkConfig(
    vlm_config=language_model.config,
    vision_gen=True,
    action_gen=False,
    sound_gen=False,
    joint_attn_implementation="two_way",
    physword=pw_config,
    # 其余参数与所用 checkpoint / tiny fixture 匹配
)
packed = pack_input_sequence(
    sequence_plans=plans,
    input_text_indexes=caption_ids,
    gen_data_clean=latent_data,
    input_timesteps=timesteps_cpu,
    special_tokens=special_tokens,
    latent_patch_size=2,
    physword=pw_config,
    physical_words=[["approach", "pick", "move", "place"]],
)
packed.to_cuda()
out = network(packed)
velocity = out["preds_vision"]
```

`physical_words` 每个 packed sample 一组。关闭 PW 时忽略人工序列，并保留原输入和 packing。当前接通的是低层 `pack_input_sequence → Cosmos3VFMNetwork.forward` 及对应 model wrapper；尚未给高层 `generate_samples_from_batch`、CFG 或多步缓存增加手工词数据协议。请使用本轮 probe 验证入口，不要把打开开关等同于完整产品推理流程已经完成。

## 固定输入与 shape

Tiny 模型使用现有 Edge 两路径类：2 层、hidden 128、Q heads 4、KV heads 2、head dim 32、mRoPE `[6,5,5]`、GEN QK norm 与 `k_norm_und_for_gen` 均开启，48-channel video latents 和 2×2 patch。

合成 RGB 为固定 seed 的 `[1,3,32,32] uint8`；仅用于固定 GEN 历史条件。RGB→latent 使用可复现的 pooling/repeat fixture，**没有冒充真实 Wan VAE 或 SigLIP2**。instruction 是固定的 synthetic IDs `[3,4,5]`，因为真实 tokenizer 尚缺失。

```text
seed=20261101
sigma=0.5, timestep=500
video input/output [1,48,2,4,4]
第 0 个 latent frame：clean history
第 1 个 latent frame：future/noisy，干预差异只计算这里

PW-on sample:
[caption×3][start][approach][pick/push][move][place][EOS][BOV][GEN×8]
UND length=10, GEN length=8, sample length=18
PW global indexes=3,4,5,6,7
```

两样本不等长测试 caption lengths=3/4，对应 sample boundaries `[0,18,37]`。显式 attention reference 测试另外覆盖真实 padding segment 与 varlen offsets。

原始 baseline velocity SHA256：

```text
c114f25d5ec0d6257547b25c001aba1c113f895cd2ad78d6199a68ddb0381b02
```

## 实际执行命令

全部在仓库根目录运行，`PY` 指向本轮隔离环境。测试只使用一张 GPU 的短时 tiny forward，不运行训练。

```bash
PY=/tmp/physword-m01-venv/bin/python
RESULTS=/aifs/lijiawang/physword/m01_results

python3 -m cosmos_framework.scripts.physword_preflight \
  --checkpoint /aifs/lijiawang/physword/Cosmos3-Edge \
  --output "$RESULTS/m0_preflight.json"

GIT_LFS_SKIP_SMUDGE=1 git worktree add --detach \
  /tmp/physword-m01-baseline 55f0554d0bf167603432041c794a8babbac759b9

CUDA_VISIBLE_DEVICES=7 I4_ATTN_BACKENDS=flash3 COSMOS_TRAINING=0 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 \
PYTHONPATH=/tmp/physword-m01-baseline \
"$PY" cosmos_framework/scripts/physword_probe.py \
  --baseline-only --output "$RESULTS/original_baseline"

CUDA_VISIBLE_DEVICES=7 I4_ATTN_BACKENDS=flash3 COSMOS_TRAINING=0 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 \
"$PY" -m cosmos_framework.scripts.physword_probe --output "$RESULTS/m1_probe"

CUDA_VISIBLE_DEVICES=7 I4_ATTN_BACKENDS=flash3 COSMOS_TRAINING=0 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
OMP_NUM_THREADS=1 "$PY" -m pytest -c tools/physword_m01_pytest.ini \
  cosmos_framework/model/generator/physword/visibility_test.py \
  cosmos_framework/scripts/physword_preflight_test.py \
  cosmos_framework/data/generator/sequence_packing/sequence_test.py \
  cosmos_framework/data/generator/sequence_packing/packers_test.py \
  cosmos_framework/model/generator/mot/include_gen_pathway_test.py \
  -q --junitxml="$RESULTS/all_tests.xml"
```

原始 worktree 已存在时不重复创建。原始基线命令必须让 `PYTHONPATH` 的原始仓库优先生效；本轮实际执行时工作目录也切到原始 worktree。probe 脚本来自当前分支，但 `--baseline-only` 只构造原有类，不导入新增 PW 模块，不传新参数。

已运行 `git diff --check`、Ruff check 与 format check。`omni_mot_model.py` 保留了原有 import 顺序，其已有 I001 单独忽略；未为此重排整段原始 imports。测试中唯一剩余 warning 来自继承环境的 requests/urllib3/chardet 版本提示，不影响离线 forward；生产环境宜使用独立完整依赖集。

原始输出 tensors 在 `/aifs/lijiawang/physword/m01_results/`。Git 内保留 JSON 摘要和 JUnit，避免把运行产物权重写入源码。

## 尚未解决与 M2 最小任务

1. **补齐并固定真实资源**：选择与权重匹配的 tokenizer / processor revision，获取上表资源后验证实际 tensor shapes / special IDs，再用真实 Edge 复验 baseline 与人工 PW 干预。本轮没有下载授权外的大模型。
2. **统一 RGB-conditioned UND prefix**：将现有冻结 SigLIP2 历史特征和准确 mRoPE 同时用于 Reasoner 和 joint forward；当前 M1 只完成文本 UND + 人工 PW，历史 RGB 仍仅在 GEN 条件中。
3. **加入固定 K 步 PW AR 并 repack**：使用同一 UND 参数，输出 IDs / prediction positions；测试 input-target shift、历史输入无未来泄漏，以及 AR / repack 的逐层 parity。M1 私有 ID 不能直接交给原 HF generate，需要共享 embedding helper / 专用词预测入口。
4. **只连接必要的高层推理数据流**：把 predicted/manual PW 从视频入口传到 packer，先 guidance=1、缓存关闭；实现 checkpoint 中 PW mapping/table 的严格恢复规则。完成后再评估 CFG / KV cache。

完整 Phase A/B loss、optimizer 和训练属于后续 M3，本轮没有提前实施。**M0 的 fallback 与 M1 已完成，到此停止。**
