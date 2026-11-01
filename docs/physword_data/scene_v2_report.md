> 已执行 episode_0038 排除；清理与短视频筛查详见 [报告](episode_0038_removal_report.md)。

# Badminton 场景原语重新标注 v2

用户最新要求：允许 16 类之外、符合场景的单词原语，不保留 idle/unknown 标注。最终范围仍仅为 badminton 当前 2 条 Pilot，左右臂独立标注；本轮不批量标注全部 206 条，也不启动训练。

本版本为 `badminton_scene_v2`，共 **25 类**：保留原 16 类及 ID，增加 `rest / hold / grasp / lift / align / adjust / retract / lower / release`。这些是有效类别；不是新增两个无效状态后继续 Mask。词表和语义定义显式记录在每份标注与输出 `vocabulary.json` 中。

## 重新审阅和阶段变化

重新检查完整 main 10Hz 序列，并按原始 mono 时间同步读取 main/wrist/wrist2 的 2Hz 全序列概览，共 95 个三视角时间点。辅助视角只能解释已有观测的握持、筒口对齐和手指运动，不能依据指令补造事件；移除的截断录制不再参与本版本。

本次是新的阶段判断，不是词名替换：持续持筒的左臂改为 hold，右臂根据时间上下文区分 approach、grasp、lift、move、align、insert、release、retract、adjust；空手保持原位为 rest。episode_0001 末尾只标 lower/hold，没有声称已松手放置。

每只手臂连续覆盖整段 Episode，共 **40 个 Stage**：episode_0001 左 8 / 右 13，episode_0017 左 6 / 右 13。全部 action 均为一个小写英文单词，**idle=0、unknown=0**。侧别约定为 main 画面左右；未声称已验证机器人硬件臂映射。

标注保留置信度和 evidence，`provider=codex_visual_pilot`、`model=conversation_multiview_review_v2`、`reviewed=false`。仍未经过独立人工金标评测，不能把监督覆盖率当作语义准确率；本版本也不等同于论文原始 16 类实验设置。

## 处理数量与可用监督

RGB 媒体、时间 Mapping 与原始 Episode 划分均复用，不重新采样。2 条 Episode 仍是 **473 帧、433 窗口**，其中 Train 433 / Test 0。从既定 Split 中仅移除被排除 Episode，其余归属不变；现在为 186 Train / 20 Test。Pilot 没有 Test 窗口，不能用于 Test 指标评估。

| 手臂 | Target 位置数（含重叠窗口） | 有效数 | 覆盖率 | 边界 Mask |
|---|---:|---:|---:|---:|
| left | 1732 | 1692 | 97.69% | 40 |
| right | 1732 | 1640 | 94.69% | 92 |

合计 **3,332 / 3,464 个有效监督（96.19%）**，剩余 132 个仅因 ±50ms 阶段边界容差被 Mask；这些位置仍有具名单词标签。433/433 个窗口至少有一个有效双臂 Target，312 个窗口的双臂 8 个 Target 全部有效。不存在用 unknown/idle 占位的标签。

原先大量 Mask 的减少主要来自更贴合场景的词表，尤其持物状态成为 hold 类；这并不意味着自动获得了更高准确率。少量 release/insert 精确边界仍需人工复核。

## 接口与防误用

DataLoader 输出 `pw_arm_class_targets / pw_arm_valid_mask=[B,2,4]`。每条窗口携带 `pw_vocabulary_id / pw_vocabulary / pw_vocabulary_hash`，验证器拒绝非单词标签、禁用词、混合类别映射、逐臂缺口和同臂重叠。

新版本双臂可同时为 hold 与 insert，不能压成单个动作。因此旧单序列兼容视图被 Mask；单序列有效窗口数不是本版本的质量指标。两臂有效监督保持完整。

现有 16 类 M3 桥接 `to_m3_batch / m3_target_payload` 对新版显式拒绝，避免新增 ID 被错误解释。完整使用本版需要 M3/M4 显式接入 25 类、独立双臂目标。未修改 Transformer、Reasoner、Word Head、Loss，未启动训练。

Condition 仍只读取 main 历史 RGB、任务指令与公开时间信息。三视角仅用于离线标注；未来 RGB 和两臂标签始终在 Target 侧。

## 验证与访问

- **55 项测试通过**：包含原 38 项回归及新词表、单词格式、禁用标签、边界保持词名、类别映射隔离和旧桥接防误用验证。
- 实际 Dataset/DataLoader 读取新版 RGB，得到标签/Mask `[2,2,4]`；Condition 中无 future 或标签字段；两个旧桥接均按预期拒绝新词表。
- 独立输出 `badminton_scene_v2_repro` 的所有任务/手臂标签统计、窗口数、Split 和词表哈希与主版本一致。
- 2 条网页的 HTTP 跳转、视频/图片/字幕资源与旧版归档检查通过；V8 中编译并运行实际页面脚本，验证双臂初始标签、timeupdate 与点击阶段跳转回调（模拟 DOM，不等同于完整浏览器渲染测试）。
- 网络/API 调用 0，训练启动 0。旧 v1 保留样本仍可对照，0038 已在两版中移除，历史网页保存为 `index_v1.html` 与逐 Episode `review_v1/`。

主数据：`/aifs/lijiawang/physword/data_pilot/badminton/scene_v2/`。
原网页 `http://localhost:8765/` 已更新为本版；旧 Episode URL 会转到新版，用户刷新即可查看。页面提供当前左右臂标签、两条时间轴、可点击跳转的阶段表、字幕、历史/未来关键帧和完整接触表。

[执行说明](../../tools/physword_data/README.md) · [完整词表](../../tools/physword_data/scene_v2/vocabulary.json) · [质量 JSON](scene_v2_results/quality_report.json) · [窗口样例](scene_v2_results/training_window_example.json) · [测试 XML](scene_v2_results/tests.xml) · [Dataset 验证](scene_v2_results/dataset_contract.json)
