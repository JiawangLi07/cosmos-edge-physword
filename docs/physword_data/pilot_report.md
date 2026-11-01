> 历史 v1 报告。按用户最新要求重新标注的场景词表版本，请看 [scene_v2_report.md](scene_v2_report.md)。

# PhysWord 数据 Pipeline Pilot 报告

日期：2026-11-01。分支：`physword/data-pipeline`；Worktree：`/aifs/lijiawang/physword/cosmos-framework-data`；基于 `f2a1e1d`。本轮最终范围只含 **badminton**，top 不进入最终 Manifest、标注或训练窗口。预处理代码仍保留通用 LIBERO/视频/HDF5 接口。

## 实际数据审查

原始根目录为 `/aifs/lijiawang/episode-badminton/episode_badminton`，207 个采集 Manifest，621 个 RGB 相机文件引用：main / wrist / wrist2 各 207。已逐份检查 621 份 JSONL 的 `i=1..N`、行数、有限且严格递增的 mono 时钟，共 526,192 行原始帧记录，无日志格式或时钟顺序失败。全部 writer 声明 30 FPS；全库最大的相邻采集间隔为 0.302460s，因此不能用固定整数帧间隔替代时间采样。全库时间戳记录通过不表示全库视频都已解码。

完整 ffprobe 审查覆盖 inventory 的 episode_0001/0002/0003 三路相机，以及 Pilot 三条 Episode 三路相机；重合 episode_0001 只计一次。Pilot main 全部顺序解码并保存 PNG，原始分辨率为 640×480。以下 `实际 FPS` 是帧数/单调时钟跨度计算的均值，不能用来还原每帧时间。

| Episode | 源 main 帧数 | 采集时钟跨度 | 容器时长 | 声明 / 实际 FPS | 输出 10Hz 帧 | 完整窗口 |
|---|---:|---:|---:|---:|---:|---:|
| badminton__episode_0001 | 798 | 26.660s | 26.600s | 30.0 / 29.895 | 267 | 247 |
| badminton__episode_0017 | 616 | 20.576s | 20.533s | 30.0 / 29.889 | 206 | 186 |
| badminton__episode_0038 | 92 | 3.085s | 3.067s | 30.0 / 29.502 | 31 | 11 |

源 Manifest 的采集编号与目录编号不一定一致，例如 badminton 目录 episode_0001 的原采集编号为 2，episode_0038 为 48。保存两者，Original ID 由采集身份生成，避免重组副本跨 Split。现有 OpenWAM 转换和 action loader 含动作/状态，旧处理也有“全部训练”划分，不能直接当作本实验的 90/10 RGB 输入。

`tasks.parquet` 只有文本 `badminton`。按用户授权，完整查看 episode_0001/0017 及另外几段后编写任务指令 `Pick up the shuttlecock and insert it into the tube.`，记录 visual_inference 和证据；不把该目标序列当作每段视频实际发生事件的依据。

前期还审查了本机 LIBERO：40 个 HDF5、2,000 个 demo，只有 actions/initial_state，没有 RGB；最终配置按最新指示仅 badminton。没有虚构 LIBERO RGB Episode，也没有继续标注 top。

## 时间、相机和因果边界

真实采集 `mono` 优先于容器 PTS；两者均完整保留。Pilot main 的最大 PTS/采集时钟累计偏差分别约 106ms、87ms、54ms。统一网格采用向过去选择最近真实观测，不插值；每帧保存 Source Frame、Source Timestamp、Source PTS、Resampled Timestamp、帧龄和文件 SHA-256。

Pilot 中每个入选帧的最大帧龄不超过 63ms，150ms 默认有效性阈值下无 stale frame、无解码失败。wrist/wrist2 使用同一 mono 时钟估计同步；episode_0038 的 wrist2 比 main 早结束约 503ms，最近帧误差最大约 503ms，不能盲目同帧号拼接。首版只使用 main，另外两路不作为模型输入；完整同步量在 metadata/quality_report。

默认 5 帧历史 + 16 帧未来，共 21 帧；PW Episode 全局索引为窗口起点 + `[8,12,16,20]`。标签使用这些目标图像的实际采集相对时间，绝对采集时间与网格时间分别保存。Stage 采用半开区间，内部边界 ±50ms、低置信度和歧义均 Mask。长度不足/末尾窗口显式丢弃，无静默补帧。

原生 Wan `WanVAE_` CPU 随机小模型实测 5/21 帧分别产生 2/6 个时间 latent；temporal_window=4 与 68 均验证历史单独编码等于完整 clip 的历史前缀，最大差为 0；改 Future 后历史前缀最大差也为 0。仅验证原生结构和因果性，不声称验证真实 checkpoint 的生成质量。

## Episode 划分与 Pilot 数量

在窗口化前对全部 207 个 Original Episode 固定 seed=42、90/10 划分，实际为 **186 Train / 21 Test**。按已知 task=badminton 分层；同采集身份、相同 lineage 或字节相同 RGB 源经并查集合并。未知血缘的裁剪/重编码不能自动判定，需要导入方提供 lineage；不声称已做视觉近重复检索。

Pilot：3 条真实 Episode、**504 帧**、**444 个完整窗口**。其中 Train 433 个窗口，Test 11 个窗口；episode_0038 是按全库既定划分抽到的 Test，不以此调整阈值。重现目录 `badminton_repro` 得到相同划分、窗口数和全部标签分布。

另有 10 条独立 synthetic fixture：330 帧、130 个窗口、9/1 Episode 划分，用于 VFR/非整数抽样、边界、poison action 字段不进入 Condition 等测试。Synthetic 数据不混入 badminton。

## 左右臂分别标注

每只手臂均完整覆盖整段 Episode，所有 Stage 指定单个 `effector=left/right`，没有 `both` 压缩。3 条共 **48 个 Stage**：episode_0001 左 11 / 右 12，episode_0017 左 9 / 右 12，episode_0038 左 3 / 右 1。没有动作时记录独立 idle；不确定接触和动作类别记录独立 unknown/歧义，保留候选。静止持物不被编造为新的抓取/放置。

侧别明确为 `effector_coordinate=selected_camera_image_sides`（main 画面左侧/右侧），没有机器人硬件映射标定的证据。下游若使用硬件左右关节身份，应确认/配置映射。两条时间轴、标签和 Mask 均保留，不因当前 M3 的接口限制丢弃。

标注来源为本轮对完整 10Hz 观察序列的视觉模型审阅，`provider=codex_visual_pilot`，`model=conversation_visual_review`，`reviewed=false`。置信度未校准，没有人工语义准确率评测；未调用外部计费 VLM，也没有把样例当作已验收金标。

| 手臂 | PW target 数（含重叠窗口） | 有效 target | approach | move | pick | insert |
|---|---:|---:|---:|---:|---:|---:|
| left | 1776 | 179 | 32 | 63 | 84 | 0 |
| right | 1776 | 740 | 200 | 352 | 68 | 120 |

两臂合计 3,552 个监督位置，其中 **919 个有效**。其余分别为 idle 1,436、歧义 1,123、边界不确定 74；idle 是已标注的非 head 状态，不等于标注丢失。其它 12 类没有观察到足够明确事件，未补造标签来凑齐类别。

当前 M3 实际类型名是 `PhysWordBatch`，只支持 4 个 Word 的单序列接口。因此另保留 `pw_class_targets / pw_valid_mask` 兼容视图：另一臂 idle 且只有一个有效动作时可用；并行动作或不确定并行语义 Mask。此视图有效 847/1,776 个 target，342/444 个窗口至少一个有效 Word，Mask 率 52.31%。完整双臂 `[2,4]` GT 的 919 个有效监督仍在数据中；完整消费它们需要 M3/M4 显式接入双臂目标，不能将它们无声扁平化后交给旧接口。

## 验证、耗时与边界

- **38 项离线测试通过**：包括真实 VFR 容器 PTS、RGB LIBERO HDF5、左右臂同时不同词不丢失、修改右臂不影响左臂标签、逐臂覆盖、Mask、无 future/GT 进入 Condition、原始 Episode/变换副本防泄漏、窗口篡改 Split 检出、缓存损坏恢复和标注重试。
- Ruff 对新增包、工具和测试通过；Shell 入口 `bash -n` 通过。
- 原生 M3 类型 CPU 接入通过：history `[5,3,480,640]`，future `[16,3,480,640]`，legacy labels `[1,4]`；DataLoader history `[2,5,3,480,640]`，双臂 target `[2,2,4]`。
- 全库清单/划分审查约 6.3s；3 条真实视频探测、解码、PNG/10Hz MP4 输出约 **41.1s**；JSON 标注导入约 0.05s；窗口构建约 0.15s；完整 Pilot QA 约 4.4s；可视化约 5.9s；38 测试约 2.9s；native M3 / Wan CPU 验证分别约 4.8s / 5.2s。视觉审阅为本轮交互过程，没有独立 VLM API 调用延迟测量，不将这部分虚报为批处理吞吐。

未启动训练，未修改 Transformer、Reasoner、MoT、训练 Loss 或主工作区的 M3 开发文件。所有动作、Proprioception、Tactile 均未进入 Condition；Future RGB 和两臂 PW GT 仅在目标侧。

Pilot 的结构/对齐检查通过，语义质量仍需人工抽检；有较多遮挡/接触歧义且仅覆盖 4 类，**不自动批准批量标注**。下一步依赖：人工确认两臂时间边界与 main 画面/硬件臂映射、确认任务指令适用于剩余 Episode、配置本地 VLM 或经授权的完整视频网关/模型/凭证，再明确批准 Pilot。若扩至其它来源，须先补 RGB 和可靠时钟；不使用 Test 调参。

## 文件位置

- [完整执行说明与 API](../../tools/physword_data/README.md)
- [严格阶段 Schema](../../cosmos_framework/data/generator/physword/schemas/atomic_stages.schema.json)
- [VLM Prompt](../../cosmos_framework/data/generator/physword/annotation_prompt.txt)
- [左/右臂 Pilot JSON](../../tools/physword_data/pilot_annotations/badminton__episode_0001.json)
- [质量统计 JSON](results/quality_report.json)
- [全 207 条 Split Manifest](results/split_manifest.json)
- [含双臂标签的窗口样例](results/training_window_example.json)
- [测试 XML](results/tests.xml)、[原生 M3 检查](results/m3_bridge.json)、[Wan 因果检查](results/native_wan_prefix.json)
- 主数据、全部 PNG/MP4、原始时间映射、完整时钟审查、HTML 报告：`/aifs/lijiawang/physword/data_pilot/badminton/`。
- 可视化入口：`/aifs/lijiawang/physword/data_pilot/badminton/index.html`；其中字幕、左右时间轴、历史/未来关键帧及全序列接触表可直接抽检。
