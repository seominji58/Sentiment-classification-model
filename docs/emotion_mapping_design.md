# Emotion Mapping Design Overview

## Summary
- Added `scripts/build_label_mapping.py` to aggregate training/validation splits and produce the five target emotions (`happy`, `sad`, `angry`, `peaceful`, `unrest`).
- Regenerated `configs/config.json` so each original SV label code maps directly to the target emotions, and persisted supporting metadata under `emotion_hierarchy`.
- Documented rule logic for major/minor category heuristics and keyword-based overrides to align service UX with the five-brand emotion themes.

## Pipeline Changes
1. **Rule Script** (`scripts/build_label_mapping.py`)
   - Counts `감정_대분류` occurrences per code and selects the dominant major emotion.
   - Applies minor-category ratios (threshold 0.4) to reroute:
     - `기쁨` → `peaceful` when calm-related minors dominate.
     - `당황` → `sad` when shame/孤立 minors dominate.
     - `상처` → `unrest` when unfairness minors dominate.
   - Writes both `label_mapping` (code → final emotion) and `emotion_hierarchy` metadata into `configs/config.json`.
2. **Config Output** (`configs/config.json`)
   - `label_mapping` distribution after regeneration: `sad=376`, `unrest=225`, `angry=149`, `happy=90`, `peaceful=60`.
   - `emotion_hierarchy` stores:
     - `code_to_major`: code → dominant native major emotion.
     - `major_base_map`: baseline major → final emotion mapping.
     - `minor_rules`: thresholds and minor emotion lists driving overrides.
     - `keyword_rules`: user keyword triggers for service-side adjustments.

## Service Integration Notes
- During inference, apply the pipeline mapping first, then optionally boost/override using `keyword_rules` when user-entered keywords match.
- Keep `scripts/build_label_mapping.py` in sync with any future CSV schema changes or new major/minor categories.
- Re-run the script whenever splits are updated so `label_mapping` stays consistent.

## Emotion Theme Reference
| emotion | icon | palette | theme name | primary UI usage |
|---------|------|---------|------------|------------------|
| happy | 😊 | Gold (#FFD700) | 행복 테마 | 캘린더, 리포트, 글귀 카드 |
| sad | 😢 | Blue (#4A90E2) | 슬픔 테마 | 감정 버튼, 필터링 UI |
| angry | 😡 | Red (#E74C3C) | 화남 테마 | 경고 메시지, 삭제 버튼 |
| peaceful | 😌 | Green (#27AE60) | 평온 테마 | 성공 메시지, 완료 상태 |
| unrest | 🫨 | Orange (#F39C12) | 불안 테마 | 알림, 주의 메시지 |

## Next Steps
- Optionally expose `emotion_hierarchy.keyword_rules` to frontend through an API so UI and model share consistent triggers.
- Add unit tests around `build_label_mapping.py` if the mapping logic evolves beyond threshold tuning.
