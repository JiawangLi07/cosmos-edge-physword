# PhysWord RGB 数据预处理与双臂阶段标注

最终任务范围：**只处理、标注 badminton**；不纳入 top。实现位于独立分支 `physword/data-pipeline`，Worktree 为 `/aifs/lijiawang/physword/cosmos-framework-data`。不改核心 Transformer、MoT Attention、训练 Loss，不启动模型训练。

实现依据：项目实现计划第 8、9、13 节；本机 `physical word.pdf` 第 14–15 页 Appendix A.3。采用完整 Episode 阶段标注、左右臂独立时间轴、未来真实观测对齐。16 类顺序与模型端 `ATOMIC_ACTIONS` 一致，`idle/unknown` 只作为数据状态并 Mask。

## 已交付数据与结果

- 主输出：`/aifs/lijiawang/physword/data_pilot/badminton/`。
- 全部 207 条原始 Episode 的划分：`split_manifest.json`，186 Train / 21 Test，seed=42。
- Pilot：`episode_0001 / episode_0017 / episode_0038`，共 504 帧、444 个窗口、48 个左右臂阶段。
- 每个窗口保存左右臂各 4 个 PW target：`pw_arm_class_targets[2,4]`、`pw_arm_valid_mask[2,4]`，`pw_effectors=["left","right"]`。
- HTML 抽检：主输出的 `index.html`；每条 Episode 提供全序列接触表、随机历史/未来关键帧、左右臂时间轴、带 WebVTT 阶段字幕的视频。
- 测试和 native M3/Wan 验证结果见 [Pilot 报告](../../docs/physword_data/pilot_report.md)。

标签由本轮对完整 10Hz 图像序列的视觉审阅生成，未调用外部计费 API，未做人工金标评测。`provenance.reviewed=false`。置信度为保守的主观分数，不表示经过校准的概率。

## 安装与执行

从本 Worktree 根目录执行。预处理只需要 CPU、ffmpeg/ffprobe，无需加载模型权重。

```bash
cd /aifs/lijiawang/physword/cosmos-framework-data
python3 -m venv /tmp/physword-data-venv
/tmp/physword-data-venv/bin/pip install -r tools/physword_data/requirements.txt
ffmpeg -version
ffprobe -version
```

本机已经安装了上述独立环境。完整重现小规模 Pilot，输出目录必须是新目录：

```bash
PHYSWORD_PYTHON=/tmp/physword-data-venv/bin/python \
  bash tools/physword_data/run_pilot.sh /aifs/lijiawang/physword/data_pilot/my_badminton_pilot
```

分步执行（`PW_PY` / `PW_OUT` 为任务专用变量）：

```bash
PW_PY=/tmp/physword-data-venv/bin/python
PW_OUT=/aifs/lijiawang/physword/data_pilot/my_badminton_pilot
"$PW_PY" -m cosmos_framework.data.generator.physword.cli audit \
  --config tools/physword_data/badminton_pilot.yaml --output "$PW_OUT"
"$PW_PY" -m cosmos_framework.data.generator.physword.cli prepare \
  --config tools/physword_data/badminton_pilot.yaml --output "$PW_OUT" \
  --episode badminton__episode_0001 --episode badminton__episode_0017
"$PW_PY" -m cosmos_framework.data.generator.physword.cli annotate \
  --config tools/physword_data/badminton_annotation_import.yaml --output "$PW_OUT"
"$PW_PY" -m cosmos_framework.data.generator.physword.cli build \
  --config tools/physword_data/badminton_pilot.yaml --output "$PW_OUT"
"$PW_PY" -m cosmos_framework.data.generator.physword.cli qa --output "$PW_OUT"
"$PW_PY" -m cosmos_framework.data.generator.physword.cli visualize \
  --config tools/physword_data/badminton_pilot.yaml --output "$PW_OUT"
"$PW_PY" -m tools.physword_data.source_audit \
  --inventory "$PW_OUT/inventory.json" --output "$PW_OUT/source_audit.json"
```

`audit` 先划分全部原始 Episode，然后才有 `prepare/build`。默认完整 ffprobe 3 条 Episode 的所有相机；`audit --probe-limit 0` 可全量探测。`source_audit` 对全部原始 Manifest / 时间戳日志进行轻量审查，明确区分“Manifest 与时钟行数一致”和“视频已解码验证”。

`prepare` 可重复执行，校验已保存帧/视频 SHA-256，恢复不完整或损坏输出；输入/config 变化时拒绝覆盖，使用新输出根目录。`--limit 0` 可全量预处理（本轮未执行全量），不触发模型标注或训练。`build/qa/visualize` 可在更新标签后重新运行。

本机查看带字幕的报告：

```bash
python3 -m http.server 8765 --bind 127.0.0.1 \
  --directory /aifs/lijiawang/physword/data_pilot/badminton
# 在有浏览器访问能力的机器打开 http://127.0.0.1:8765/；远程环境可使用端口转发。
```

## 数据发现与 Adapter

`real_robot` 根据实际 `*/manifest.json` 枚举：

```text
/aifs/lijiawang/episode-badminton/episode_badminton/episode_NNNN/
  manifest.json
  main/color_take001.mp4
  main/timestamps_take001.jsonl
  wrist/...
  wrist2/...
  robot_states.h5                  # 不读入模型条件
```

首版使用 `main` 原始 640×480 RGB。另两相机只用于时钟/同步检查，不拼接为额外模型输入。容器声明 30 FPS，实际采集时钟存在不规则间隔。唯一 ID 保留目录名与采集 Manifest 编号；`original_episode_id/lineage_id` 来自原始采集身份，而非目录数字。

原任务元数据只有 `badminton`。按用户授权，查看完整视频后归纳任务指令为 `Pick up the shuttlecock and insert it into the tube.`，保存 `instruction_source=visual_inference` 和证据。该目标指令不证明每段短视频完成了抓取/插入；例如截断的 `episode_0038` 并未标注发生了 pick。

其它布局使用 `manifest` Adapter：显式提供 `episode_id / original_episode_id / lineage_id / dataset_source / task_id / source_kind / source_ref / camera`，视频路径和可选外部时间戳写在 `cameras`。相对路径以配置/Manifest 所在目录为基准。变换副本必须提供相同 lineage；不从文件名猜任务或血缘。

`libero_hdf5` Adapter 仍作为通用接口保留，支持明确的 RGB dataset key（默认 `obs/agentview_rgb`）、Timestamp key、任务属性/覆盖映射；没有 Timestamp 时必须显式声明 `source_fps`，并记录为重建时钟。当前本机 40 个 LIBERO 文件共 2,000 demos 只有动作和初始状态，没有 RGB，因此不能转换成真实 RGB Episode；已用带 RGB/时间戳的 Synthetic HDF5 验证该 Adapter。最终 badminton 配置不包含 LIBERO/top。

## 时间与视频契约

优先级：逐帧采集 `mono` > 显式外部时间戳 > 容器 PTS。机器人时间戳要求 `i=1..N` 连续，逐帧对应解码顺序，时钟有限且严格递增。`ffprobe` 保存每个真实解码帧的 PTS；不使用整数帧间隔抽样。

模型时钟以选定相机的第一帧为零点，间隔 `1/sampling_fps`。每个网格点只选择 **不晚于该网格点的最近真实观测**。重复取到同一源帧时保留相同源索引/时间，不声称得到新观测；不生成插值帧。过旧观测（默认 >150ms）标记无效，包含该帧的窗口整体丢弃。

`metadata.json` 同时保存原始绝对采集时间、原始容器 PTS、归一化源时间、10Hz 网格时间、Frame Mapping、帧龄、源 FPS、图像/视频 SHA-256、所有相机检查。`rgb_10hz.mp4` 为可视化 CFR 版本；训练读取无损 RGB PNG，因此可重现实际采样帧。非 10 FPS 配置会使用相应视频文件名。

阶段与 PW 监督按 **目标 RGB 的 `source_elapsed_s`** 查询。播放/模型时间单独保存，不用四舍五入后的帧号代替真实时间。WebVTT 通过 Mapping 把源时间阶段映射回播放时间。

所有区间为 `[start_time,end_time)`。内部边界默认 ±50ms Mask，低于 0.6 置信度 Mask；这两个阈值均可配置。初始参数在 YAML / `PipelineConfig` 中，验证 Wan 历史和总长度均满足 `1+4m`，不能静默向上补齐；末尾不足窗口明确丢弃。默认局部 History `[0..4]`、Future `[5..20]`、PW `[8,12,16,20]`；每条记录的索引均是 Episode **全局**索引。

## 左右臂标注与 Schema

严格 Schema：[atomic_stages.schema.json](../../cosmos_framework/data/generator/physword/schemas/atomic_stages.schema.json)。Prompt：[annotation_prompt.txt](../../cosmos_framework/data/generator/physword/annotation_prompt.txt)。

badminton 标注明确 `effectors=["left","right"]`，两条轨迹分别完整覆盖 Episode。每条 Stage 必须指定单个 effector；不以 `both` 合并，不以共享 Word 替代两个预测目标。静止持物阶段使用 `idle`；遮挡或语义不明使用 `unknown`，均不增加 Word Head 类别。验证器拒绝同臂重叠、轨迹缺口、错误类别、无序/越界/非有限时间，以及未声明的 effector。

当前左右定义为 `effector_coordinate=selected_camera_image_sides`：main 视野左侧/右侧的手臂。没有硬件标定文件，未把视觉侧别冒充已标定的机器人关节命名；若需要硬件 anatomical left/right，应在 Adapter 中显式映射并更新坐标约定。

```json
{
  "start_time": 10.5,
  "end_time": 11.4,
  "action": "pick",
  "confidence": 0.82,
  "effector": "right",
  "ambiguous": false,
  "candidates": [],
  "evidence": "Shuttlecock is lifted clear of the table by image-right fingertips."
}
```

窗口字段：

| 字段 | 单样本形状/含义 |
|---|---|
| `history_rgb` | uint8 `[5,3,H,W]` |
| `future_rgb` | uint8 `[16,3,H,W]`，只供训练 Target |
| `instruction`, `episode_id` | 任务指令、稳定身份 |
| `pw_effectors` | `["left","right"]`，以下两行的轴顺序 |
| `pw_arm_class_targets` | int64 `[2,4]`，两臂各 4 个独立 Word GT |
| `pw_arm_valid_mask` | bool `[2,4]`，分别 Mask idle/unknown/边界/低置信度 |
| `pw_class_targets`, `pw_valid_mask` | `[4]`，旧 M3 单序列兼容视图，见下文 |
| `pw_target_timestamps` | float64 `[4]`，两臂共享目标 RGB 的真实源相对时间 |
| `pw_target_source_timestamps` | 原始采集绝对时间 |
| `pw_target_resampled_timestamps` | 模型/播放网格时间 |
| `*_frame_indices`, `*_source_frame_indices` | Episode 全局模型索引 / 源解码索引 |
| `metadata_path`, `video_path`, `feature_cache_key` | 可追溯路径及 teacher cache 输入身份 |

`pw_class_targets` 兼容视图只有单个有效动作且另一臂 idle 时才可有效。双臂同时有有效/不确定非 idle 阶段时，此视图 Mask；**两臂原始标签和各自有效性保持完整**。即便两臂同为 move，也不无声压成一个“正确” Word。无效类 ID 用 0 占位且 Mask=false，不增加第 17 类。

## VLM Adapter、缓存和 Pilot 门槛

`AnnotationAdapter` 支持：

- `pending`：全 Episode unknown，不伪造标签，不调用模型。
- `json_file`：导入完整 Schema JSON，本轮 Pilot 使用此方式保存本轮视觉模型审阅结果。
- `command`：显式 argv 本地 VLM/网关进程；stdin 为 JSON 请求，stdout 为严格标注 JSON，无 shell 展开。
- `http_json`：明确配置的 HTTPS JSON 标注网关，必须显式 `--allow-network`。此通用协议不是某个厂商的原生视频 API；网关需能访问请求中完整视频/帧路径（共享文件系统或自行上传素材）。

请求包含 model、parameters、Prompt、Schema、完整 Episode 视频及**所有**10Hz 帧和时间映射、任务指令、effector 约定。禁止只凭 instruction 生成动作顺序。自定义网关负责适配所选模型的视频输入格式；不隐式下载权重、不默认访问供应商 API。

示例 adapter 配置（须自行提供本地入口，命令的联网/费用策略也由该入口负责）：

```yaml
annotation:
  kind: command
  command: ["/path/to/local/vlm-python", "/path/to/annotate_complete_episode.py"]
  model: your-local-video-vlm
  parameters: {temperature: 0, max_output_tokens: 8192}
  effectors: [left, right]
  effector_coordinate: selected_camera_image_sides
  max_attempts: 3
  timeout_seconds: 120
  retry_backoff_seconds: 1
```

缓存 Key 绑定素材指纹、完整帧路径/时间映射、指令、模型/参数、Prompt、Schema；文件导入还绑定标注文件 SHA-256。只缓存通过验证的响应；失败状态保存次数/错误类型，可重复运行恢复。API Key 只能通过 `api_key_env` 指定环境变量，禁止 literal `api_key`。不把密钥/响应异常正文写入状态日志。

默认 `annotate` 最多 3 条。超过 3 条必须提供人工填写的 `--approved-pilot review.json`，含 reviewer、pilot_report、approved_for_batch=true；HTTPS 还需要 `--allow-network`。本轮没有批准或执行批量 VLM 标注。审阅决定模板见 [pilot_review.json](pilot_review.json)。

## Dataset / DataLoader 与 M3 交接

```python
from cosmos_framework.data.generator.physword.dataset import (
    TrainingWindowDataset,
    make_dataloader,
    condition_inputs,
    to_m3_batch,
    m3_target_payload,
    feature_cache_identity,
)

manifest = "/aifs/lijiawang/physword/data_pilot/badminton/windows.jsonl"
dataset = TrainingWindowDataset(manifest, split="train")
sample = dataset[0]
loader = make_dataloader(manifest, split="train", batch_size=2, seed=42)
batch = next(iter(loader))
# batch['pw_arm_class_targets']: [B,2,4]; batch['pw_arm_valid_mask']: [B,2,4]
# condition_inputs(sample) 仅返回历史 RGB、指令和公开模型时钟。
```

实际并行 M3 当前类型名是 `PhysWordBatch`，包含 `RGBHistory / instruction / future_rgb / word_labels / valid_word / valid_latent`；并无已定义的 `PhysWordTargets`。`to_m3_batch(sample)` 已验证构造真实类型，使用兼容单序列标签和 Mask。`build_condition` 只使用 `RGBHistory` 与任务指令调用现有 `build_reasoner_prefix`，不接收 future/GT PW。

**左右臂各 4 个 Word 的完整 GT 已交付，但当前 M3 模型接口只消费一条 4-Word 序列。** 完整使用 `[2,4]` 监督需要 M3/M4 显式采用两臂目标接口；本数据分支不修改 Reasoner 架构、Word Head 或 Loss，也不把 8 个词擅自扁平拼接到旧接口。

`RGBHistory` 现有 `source_pts_seconds` 校验器实际要求模型网格；桥接时仅此字段传入 resampled timestamps，真实采集 PTS 仍留在 Dataset/Manifest，不修改或伪造原始记录。旧 M2/M3 的 5/16/4/4 限制在桥接层显式失败；独立预处理可以采用其它满足 Wan 因果长度约束的配置。

`m3_target_payload` 提供目标图像（Future 相对索引 `[3,7,11,15]`），SigLIP2 / projector 编码和 latent loss 完全由 M3 负责。`feature_cache_identity` 另外绑定 teacher SHA-256、processor config、pooling，避免不同 teacher/预处理复用缓存。

可选 CPU native 检查，在具备已有 Cosmos 依赖的环境中运行（不启动训练）：

```bash
/tmp/physword-m01-venv/bin/python -m tools.physword_data.check_m3_bridge \
  --manifest /aifs/lijiawang/physword/data_pilot/badminton/windows.jsonl \
  --m3-root /aifs/lijiawang/physword/cosmos-framework \
  --output /aifs/lijiawang/physword/data_pilot/badminton/m3_bridge.json --check-vae
```

`--m3-root` 只读扩展模型 PhysWord 包搜索路径，方便两个独立 Worktree 合并前检查；常规集成后无需该参数。Wan 检查使用原生类的随机小模型，验证 5/21→2/6 latent，以及改变 Future 不影响历史 prefix；不等同于真实 checkpoint 或学习效果验证。

## Synthetic Fixture 与测试

```bash
PW_PY=/tmp/physword-data-venv/bin/python
"$PW_PY" -m cosmos_framework.data.generator.physword.cli fixture --output /tmp/pw_fixture_source
"$PW_PY" -m cosmos_framework.data.generator.physword.cli audit \
  --config /tmp/pw_fixture_source/config.json --output /tmp/pw_fixture_processed
"$PW_PY" -m cosmos_framework.data.generator.physword.cli prepare \
  --config /tmp/pw_fixture_source/config.json --output /tmp/pw_fixture_processed --limit 0
"$PW_PY" -m cosmos_framework.data.generator.physword.cli fixture-labels --output /tmp/pw_fixture_processed
"$PW_PY" -m cosmos_framework.data.generator.physword.cli build \
  --config /tmp/pw_fixture_source/config.json --output /tmp/pw_fixture_processed
"$PW_PY" -m cosmos_framework.data.generator.physword.cli qa --output /tmp/pw_fixture_processed
PYTHONPATH=. "$PW_PY" -m pytest -c tools/physword_data/pytest.ini \
  --confcutdir=tests/physword_data tests/physword_data -q
```

Fixture 明确标记 synthetic_fixture，采用不规则时间戳、RGB 图像和故意存在的 poison actions 字段，验证该字段不会进入模型条件；`fixture-labels` 对真实数据会拒绝执行。

质量报告检查 FPS/PTS/采集时间、PNG 解码/分辨率/校验和、视频帧数/时间网格、原始 Episode 与副本防泄漏、窗口与划分一致、16 类词表、双臂完整覆盖/重叠、PW 真实时间对齐、逐臂无效 Mask、History/Future 边界。按任务、类别、手臂统计有效/无效量，并同时给出重叠窗口 target 计数与去重后帧统计。

输出包含 provenance 和显式 Mask 原因；质量检查通过只说明结构/对齐正确。近重复裁剪/重编码缺少 lineage 时不能仅凭像素推断身份，导入前必须补齐来源。M4 使用本数据前应完成人工语义抽检、侧别映射确认，并增加足够的训练任务/类别覆盖。
