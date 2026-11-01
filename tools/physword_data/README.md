# Badminton 双臂场景原语标注 v2

当前版本为 `badminton_scene_v2`。按最新要求，重新审阅 3 条 badminton Pilot 的完整 main 10Hz 序列，并补查按采集时钟同步的 main / wrist / wrist2 图像；每只手臂独立标注，所有标签均为一个英文单词，**没有 `idle` 或 `unknown` 标注**。

旧版 16 类流程和历史结果保存在 [README_v1.md](README_v1.md) 与 [v1 Pilot 报告](../../docs/physword_data/pilot_report.md)。新版扩展的是数据词表，不声称与论文原始 16 类设置相同；没有修改模型、Loss，也没有启动训练。

## 查看新版

原来的网页入口已更新。服务器若仍在运行，刷新 `http://localhost:8765/` 即可；之前打开的 Episode 页面也会转到新版。历史网页保留在 `index_v1.html`。

```bash
/usr/bin/python3 -m http.server 8765 --bind 127.0.0.1 \
  --directory /aifs/lijiawang/physword/data_pilot/badminton
```

远程 VS Code 在 Ports 面板转发 8765，然后浏览器打开上述地址。页面显示当前左右臂词语、独立阶段表、可点击跳转的阶段开始时间、字幕和未来关键帧。关键帧词语旁的 `*` 表示该监督位置在边界带内被 Mask，动作标签本身仍保留。

## 结果与词表

主输出：`/aifs/lijiawang/physword/data_pilot/badminton/scene_v2/`。

- 3 条 Episode：`episode_0001 / episode_0017 / episode_0038`。
- 504 个 10Hz RGB 帧、444 个完整窗口、45 个阶段，两臂各自连续覆盖整段 Episode。
- 全部 207 条原始 Episode 的 186 Train / 21 Test 划分保持原样；Pilot 为 433 Train / 11 Test 窗口。
- 双臂监督共 3,552 个位置，3,415 个有效（96.14%）；137 个位置仅因阶段边界不确定被 Mask。
- 该比例是监督覆盖率，不是人工标注准确率；置信度仍为主观评分，`reviewed=false`。

词表共有 25 类，原有 16 类 ID 保留，新增以下 9 类：

| ID | 单词 | 场景定义 |
|---:|---|---|
| 16 | `rest` | 空手保持原位或准备姿态 |
| 17 | `hold` | 持续抓持、支撑已经拿起的物体，如持筒等待另一只手操作 |
| 18 | `grasp` | 手指围拢物体的抓握阶段；不等同于已成功提起 |
| 19 | `lift` | 物体离开桌面的上提阶段 |
| 20 | `align` | 调整手、物体与筒口之间的位置或方向 |
| 21 | `adjust` | 可见的局部手指、握姿或物体微调；不推断隐藏的按压事件 |
| 22 | `retract` | 空手从操作位置撤回 |
| 23 | `lower` | 持物下降；不推断已经放下或松手 |
| 24 | `release` | 手指张开并与原先抓持的物体分离 |

完整映射与逐词定义：[vocabulary.json](scene_v2/vocabulary.json)。实际观察到 13 类：`approach / move / rotate / insert / rest / hold / grasp / lift / align / adjust / retract / lower / release`。

这次重新划分了阶段和部分边界，并区分左臂持筒与右臂操作，没有将历史 `unknown` 一律机械替换成同一个词。短 Episode 0038 只标到抓握阶段，不添加未观察到的提起或插入。静止持物使用 `hold`，空手静止使用 `rest`，两者是新版词表的有效类别。

侧别约定仍为 `selected_camera_image_sides`：main 画面左/右手臂。需要与硬件臂名称对应时，应单独确认映射。

## 重现重新标注结果

在独立数据 Worktree 中执行，使用新输出目录。无需重新抽帧，不调用计费 API。

```bash
cd /aifs/lijiawang/physword/cosmos-framework-data
/tmp/physword-data-venv/bin/python -m tools.physword_data.relabel_scene \
  --source /aifs/lijiawang/physword/data_pilot/badminton \
  --output /aifs/lijiawang/physword/data_pilot/badminton/scene_v2_repeat \
  --publish
```

`--publish` 在质量检查成功后更新原网页入口，同时归档 v1 抽检文件；省略则只生成新版本。输出目录已有内容时拒绝覆盖。词表、标注 JSON、Prompt 与严格 Schema 位于 [scene_v2/](scene_v2)。

重新标注使用已有真实时间 Mapping 和不可变 RGB 媒体，通过链接复用 PNG/MP4，原始视频、v1 阶段标签、v1 窗口和 Episode 划分均保留。`badminton_scene_v2_repro` 已独立重建并获得相同词表、标签分布和窗口数量。

若需从零生成 10Hz 媒体，先按 [v1 基础预处理命令](README_v1.md) 执行 audit / prepare，再运行本命令。数据只包括 badminton，未处理 top。

## Dataset 与训练接口

新版记录携带 `pw_vocabulary_id`、`pw_vocabulary`（有序类别列表）、`pw_vocabulary_hash`，防止类别 ID 被不同词表误读。

```python
from cosmos_framework.data.generator.physword.dataset import TrainingWindowDataset, make_dataloader

manifest = "/aifs/lijiawang/physword/data_pilot/badminton/scene_v2/windows.jsonl"
dataset = TrainingWindowDataset(manifest, split="train", require_valid_arm_pw=True)
sample = dataset[0]
batch = next(iter(make_dataloader(manifest, batch_size=2, seed=42, require_valid_arm_pw=True)))
# batch['pw_arm_class_targets']: [B,2,4], int64
# batch['pw_arm_valid_mask']: [B,2,4], bool
# pw_effectors: ['left','right']; 两臂共享四个目标 RGB 的真实时间戳。
```

左右臂 GT 才是本版本的监督接口。旧的 `pw_class_targets / pw_valid_mask` 单序列视图会因双臂并行被 Mask；不能用其“有效窗口数=0”判断新版没有标签。使用 `require_valid_arm_pw`，不要用只检查旧单序列的 `require_valid_pw` 过滤本版本。

当前 `to_m3_batch / m3_target_payload` 是原 16 类桥接接口，遇到新版词表会明确报错。完整消费新版需要 M3/M4 显式配置 25 类和独立双臂目标；数据分支不会修改核心模型，也不会自动把新增类别投影回 16 类或把两臂压成一个词。

Condition 仍只包含 main 历史 RGB 与任务指令，腕部相机仅供离线标注审阅。未来 RGB、标签和双臂 GT 不进入 Condition。SigLIP2 未来目标编码仍由 M3 负责。

## 验证与报告

```bash
PYTHONPATH=. /tmp/physword-data-venv/bin/python -m pytest \
  -c tools/physword_data/pytest.ini --confcutdir=tests/physword_data tests/physword_data -q
```

50 项测试通过，包含旧 16 类回归、单词/禁用标签验证、扩展词表 ID 隔离、双臂完整覆盖与独立性、真实时间映射、边界 Mask、混合词表拒绝和旧 Word Head 防误用。扩展 Dataset/DataLoader 实测输出 `[2,2,4]` 标签与 Mask。

[新版质量报告](../../docs/physword_data/scene_v2_report.md) · [严格 JSON Schema](scene_v2/atomic_stages.schema.json) · [新版 Prompt](scene_v2/prompt.txt) · [Episode 0001 标注](scene_v2/annotations/badminton__episode_0001.json)
