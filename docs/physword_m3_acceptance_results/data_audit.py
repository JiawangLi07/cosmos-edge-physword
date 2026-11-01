"""Read-only frozen release integrity and timing acceptance."""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path('/aifs/lijiawang/physword/pre_m4_fixes_results/badminton_left_release')
protocol = json.loads((ROOT / 'protocol.json').read_text())
vocab = json.loads((ROOT / 'vocabulary.json').read_text())
raw = (ROOT / 'windows.jsonl').read_bytes()
digest = hashlib.sha256(raw).hexdigest()
assert digest == protocol['windows_sha256']
assert hashlib.sha256(Path(protocol['source_windows']).read_bytes()).hexdigest() == protocol['source_windows_sha256']
rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
counts = Counter()
labels = defaultdict(Counter)
masks = Counter()
invalid_stored_ids = Counter()
owners = {key: defaultdict(set) for key in ('episode_id', 'original_episode_id', 'lineage_id', 'media_fingerprint')}
paths = set()
max_horizon_error = 0.0
for row in rows:
    split = row['split']
    counts[split] += 1
    for key, mapping in owners.items():
        mapping[row[key]].add(split)
    assert row['selected_effector'] == 'left'
    assert row['model_fps'] == 10
    assert row['task_vocabulary_hash'] == protocol['model_vocabulary_hash']
    assert row['pw_vocabulary'] == vocab['words']
    history, future, targets = [row[key] for key in ('history_frame_indices', 'future_frame_indices', 'pw_target_frame_indices')]
    assert len(history) == 5 and len(future) == 16 and len(targets) == 4
    assert targets == [history[-1] + 4 * i for i in range(1, 5)]
    assert targets[-1] == future[-1]
    horizon_error = abs(row['future_timestamps'][-1] - row['history_timestamps'][-1] - 1.6)
    max_horizon_error = max(max_horizon_error, horizon_error)
    assert horizon_error < 1e-12
    for i, frame in enumerate(targets):
        offset = frame - future[0]
        assert row['pw_target_resampled_timestamps'][i] == row['future_timestamps'][offset]
        assert row['pw_target_source_timestamps'][i] == row['future_source_timestamps'][offset]
        assert row['pw_target_source_frame_indices'][i] == row['future_source_frame_indices'][offset]
    for cls, valid in zip(row['pw_class_targets'], row['pw_valid_mask'], strict=True):
        if valid:
            assert 0 <= cls < len(vocab['words'])
            labels[split][vocab['words'][cls]] += 1
        else:
            masks[split] += 1
            # Boundary-uncertain known words retain their original class for
            # provenance. valid_word controls CE and neutral TF feedback.
            # Only unknown vocabulary entries need the -100 sentinel.
            assert cls == -100 or 0 <= cls < len(vocab['words'])
            invalid_stored_ids[cls] += 1
    paths.update((ROOT / x).resolve() for key in ('history_frame_paths', 'future_frame_paths') for x in row[key])
assert all(len(splits) == 1 for mapping in owners.values() for splits in mapping.values())
assert all(path.is_file() for path in paths)
assert dict(counts) == protocol['window_counts']
assert dict(labels) == protocol['label_counts']
result = {
    'status': 'PASS',
    'windows_sha256': digest,
    'windows': dict(counts),
    'labels': {k: dict(v) for k, v in labels.items()},
    'invalid_word_counts': dict(masks),
    'invalid_stored_class_ids_ignored_by_mask': dict(invalid_stored_ids),
    'distinct_rgb_files_exist': len(paths),
    'horizon_max_abs_seconds': max_horizon_error,
    'episode_lineage_content_split_disjoint': True,
    'all_target_rgb_source_indexes_and_timestamps_equal': True,
    'annotations_reviewed': all(row['annotation_status'].get('reviewed', False) for row in rows),
    'validation_split': protocol['validation_split'],
}
(Path(__file__).parent / 'data_integrity.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
