# Emotion Mapping Design Overview

## Document Purpose
This note explains how the project consolidates raw 감정 코드 into the five production emotions and how that mapping feeds downstream services. It acts as a hand-off reference for anyone updating label rules, regenerating `config.json`, or syncing the UX emotion themes with model outputs.

## Summary
- Added `scripts/build_label_mapping.py` to aggregate training/validation splits and produce the five target emotions (`happy`, `sad`, `angry`, `peaceful`, `unrest`).
- Regenerated `configs/config.json` so each original SV label code maps directly to the target emotions, and persisted supporting metadata under `emotion_hierarchy`.
- Documented rule logic for major/minor category heuristics and keyword-based overrides to align service UX with the five-brand emotion themes.

## Implementation Steps
1. **Aggregate Observations**
   - `scripts/build_label_mapping.py` loads `processed/splits/train.csv` and `valid.csv`.
   - For each SV 라벨 코드, it counts `감정_대분류` occurrences and picks the dominant major 감정.
2. **Apply Heuristic Overrides**
   - Minor-category ratios (threshold `0.4`) reroute specific patterns:
     - `기쁨` → `peaceful` when calm-related minors dominate.
     - `당황` → `sad` when shame/고립 minors dominate.
     - `상처` → `unrest` when unfairness minors dominate.
   - Keyword hints (예: `행복`, `불안`) are persisted for inference-time overrides.
3. **Persist Mapping**
   - The script writes both `label_mapping` (code → 최종 감정) and `emotion_hierarchy` metadata back into `configs/config.json`.
   - Resulting distribution (latest run): `sad=376`, `unrest=225`, `angry=149`, `happy=90`, `peaceful=60`.

## Config Structure Highlights
- `code_to_major`: original code → dominant native major emotion.
- `major_base_map`: baseline major → final five-emotion mapping.
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
- Maintain a changelog of mapping revisions (date, rationale, threshold adjustments) so experiments remain auditable.
- Evaluate whether minor-category thresholds require per-emotion tuning as new 상담 데이터가 추가될 때마다 분포를 점검합니다.
