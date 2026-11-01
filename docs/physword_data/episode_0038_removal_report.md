# episode_0038 清理与短视频筛查

日期：2026-11-01。范围为当前机器已发现的 badminton 原始副本、对齐数据、LeRobot/OpenWAM 数据和 PhysWord Pilot；在独立 `physword/data-pipeline` 分支执行，未启动训练或修改模型。

## 结论

`episode_0038` 是原始采集提前中断，不是抽检 HTML 截短。原始 Manifest 的采集时长 **3.114671 秒**，main 相机 92 帧、首末真实时间戳跨度 **3.084570 秒**；Manifest 标记 `incomplete`，错误为 `wrist2 相机采集超过 0.5 秒没有新帧`。main MP4 在名义 30 FPS 下的容器时长约 3.067 秒，10Hz 版本约 3.1 秒，均与截断来源一致。

已按原始采集身份与来源映射排除该 Episode，而非只删除抽检链接。原始目录编号 `0038` 对应采集编号 **48**，在旧 LeRobot 数据中对应 **episode_index=37**；不能按转换后的文件名 `038` 直接删除。

全量扫描清理前 **207 条 Episode、621 份相机时间戳日志**，以采集单调时钟和 main 首末帧跨度筛查 `<5s`；同时保留三相机各自跨度和共同覆盖范围。仅发现以下两条，其他相机日志也没有额外的 `<5s` Episode：

| 原始目录 | 采集编号 | 采集时长 | main 帧数 | main 时间戳跨度 | 处理 |
|---|---:|---:|---:|---:|---|
| episode_0038 | 48 | 3.114671s | 92 | 3.084570s | 已从活动数据集移除 |
| episode_0042 | 53 | 3.657115s | 107 | 3.599648s | 已筛出、保留待处理 |

`0042` 同样为 `incomplete`，记录同一 wrist2 超时错误；其 wrist/wrist2 跨度为 3.633466/3.070585 秒。本次仅对用户指定的 `0038` 执行移除，未自动删除 `0042`。这次是时长筛查，未将其余录制自动认定为语义完整或标注合格。

完整短视频证据见 [short_episodes.json](curation_results/short_episodes.json)。清理后原始数据剩 **206 条**，其中仍有上述 1 条短于 5 秒。

## 实际移除范围

数据根目录为 `/aifs/lijiawang/episode-badminton/`：

| 数据集/副本 | 清理前 Episode 数 | 清理后 Episode 数 | 说明 |
|---|---:|---:|---|
| episode_badminton | 207 | 206 | 原始三相机、日志与 Manifest |
| episode_badminton_91 | 91 | 90 | 原始硬链接副本 |
| aligned_native_v2 | 91 | 90 | JSON/NPZ 与聚合审计统计 |
| aligned_native_v2_146 | 146 | 145 | JSON/NPZ 与聚合审计统计 |
| lerobot_next_state_v3 | 91 | 90 | 72,436 → 72,361 行，Train 81 / Val 9 |
| lerobot_next_state_v3_146 | 146 | 145 | 125,434 → 125,359 行 |
| openwam_eef80 | 146 | 145 | 125,434 → 125,359 行 |
| archive/openwam_eef80_91 | 91 | 90 | 72,436 → 72,361 行 |

另从 `/aifs/lijiawang/physword/data_pilot/` 下 **6 个 PhysWord 版本**移除：`badminton/scene_v2`、`badminton`、`badminton_scene_v2_repro`、`badminton_repro`、`real_v1`、`real`。涉及原始 Episode 清单、Split、窗口、标注、缓存、处理媒体、抽检页面与统计；已同步删除代码仓库中的该 Episode 标注样例和 metadata 快照。旧 `real` 清单中的 top 同号 Episode 属于不同采集，不在本次移除范围。

OpenWAM 工作目录中指向 EEF 数据的符号链接自动指向清理后的活动目录。未修改远端数据、已训练权重或部署模型；历史报告可以保留此次清理的文字记录，不代表仍有活动样本。

## 保留数据的一致性

LeRobot 转换数据重新生成连续 Episode/行索引、Episode 偏移、Split 范围、来源映射与汇总。每个保留样本的原始数值列与隔离前逐列比对，除 `episode_index`、全局 `index` 外完全相同；视频按实际文件身份复用并核对 SHA256，避免索引前移后符号链接指错视频。

统计量在原先统计人群上重算：`lerobot_next_state_v3` 仍只用 Train，其余三版保持原有全量统计方式。EEF 归一化的有效槽位重新计算 min/max/mean/std/q01/q99，保留原有旋转及非活动槽位的恒等策略。现有全量统计方式在后续另行划分测试集时仍需单独评估，不将它改称 Train-only。

PhysWord 幸存样本的 Train/Test 归属保持不变，当前全量 badminton 为 **186 Train / 20 Test**。新配置显式继承清理后的 canonical Split，不因少一个 Episode 重新随机分配。两个原始根目录新增 `_episode_exclusions.json`，按采集身份和源视频 SHA256 防止 `0038` 重新下载、改名后被同一 Adapter 收入。

当前新版 Pilot 为 **2 条 Episode、473 帧、433 个窗口**，全部为 Train；没有可用于评估的 Test 窗口。双臂共有 3,332/3,464 个有效监督，剩余 132 个位置因阶段边界容差被 Mask。保留 Episode 为 `0001`、`0017`。

## 验证与恢复

- 55 项数据测试通过，包括硬链接备份隔离、重新编号后仍可排除来源、LeRobot 连续索引/Train-only 统计、Split 继承和真实时钟时长筛查。
- 独立验证器检查 2 套原始副本、2 套对齐数据、4 套 LeRobot、7 个 PhysWord inventory 根目录，核对来源排除策略、连续索引、窗口路径、保留数值、视频和来源哈希、统计量及 EEF 归一化，结果 `passed`。
- HTTP 实测两条新版及旧版归档页面、全部引用的图片/视频/字幕可读取；被移除 Episode 的新旧 URL 均为 404。V8 中执行实际网页脚本，检查双臂显示、timeupdate 和阶段按钮跳转；使用模拟 DOM，不等同于完整浏览器渲染测试。
- 重新执行配置审查，发现 206 条 Episode，且其 Split 归属与 canonical Manifest 完全一致。

隔离备份位于 `/aifs/lijiawang/physword/data_quarantine/episode_0038_20261101/`，不属于训练数据根目录；`QUARANTINED_DO_NOT_TRAIN.txt` 标识为仅供恢复。`journal.json` 记录移动/替换操作，`removed/` 保存移出的原始 Episode 和旧转换目录，`before/` 保存修改前文件。没有不可恢复地销毁原始观测；需要回滚时按日志检查依赖，不能只拷回一个转换文件。

主要结果：[移除摘要](curation_results/removal_summary.json) · [独立验证](curation_results/verification_report.json) · [重新审查](curation_results/pipeline_reaudit.json) · [测试 XML](curation_results/tests.xml) · [更新后的质量报告](scene_v2_results/quality_report.json)。完整前后时长扫描和转换索引映射保存在隔离目录。

## 只读复核命令

```bash
cd /aifs/lijiawang/physword/cosmos-framework-data
/tmp/physword-data-venv/bin/python - <<'PY'
from tools.physword_data.curation import scan_short_episodes
report = scan_short_episodes('/aifs/lijiawang/episode-badminton/episode_badminton', 5.0)
print('Episodes:', report['episodes_scanned'])
for row in report['short_episodes']:
    print(row['source_name'], row['recording_duration_s'], row['status'], row['recording_errors'])
PY
/tmp/physword-data-venv/bin/python -m tools.physword_data.verify_curation \
  --quarantine /aifs/lijiawang/physword/data_quarantine/episode_0038_20261101 \
  --pilot-root /aifs/lijiawang/physword/data_pilot
```

上述验证不修改活动数据，只刷新隔离目录中的验证报告。`curation.stage_lerobot_removal()` 和 `remove_episode --apply` 是针对已审查本地布局的维护工具：需要先依据来源关系生成四套 staging 和 `lerobot_filter_plan.json`，再交换活动目录；不可作为未知数据集的通用自动删除器。本次操作日志与计划均已保存，不需重复执行清理。

抽检服务器无需重启；刷新 `http://localhost:8765/` 即可看到当前两条 Episode。
