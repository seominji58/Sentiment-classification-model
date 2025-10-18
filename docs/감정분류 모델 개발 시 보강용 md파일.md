# 감정 분류 모델 성능 보강 가이드

현재 파이프라인(Transformer 기반 다중 감정 분류)은 슬픔/불안 라벨이 과도하게 많은 데이터 특성상 소수 감정(행복, 평온 등)의 성능이 상대적으로 낮아질 수 있습니다. 이 문서는 성능 저하 요인을 분석하고, 실전에서 적용할 수 있는 개선 전략을 "데이터 → 학습 → 평가 및 사후 처리" 단계별로 정리한 보강 가이드입니다.

## 1. 데이터 단계: 불균형 완화
- **라벨 분포 확인**: `processed/splits/train.csv`와 `valid.csv`에서 `value_counts()`로 비율을 점검하고, 소수 감정 라벨 수를 파악합니다.
- **업샘플링(Up-sampling)**: 행복/평온 등 소수 라벨을 배치별로 복제해 빈도를 맞춥니다. 단순 복제 외에도 `EDA`, 번역·역번역, 동의어 치환 등으로 문장을 변형해 데이터 다양성을 확보합니다.
- **다운샘플링(Down-sampling)**: 슬픔/불안 샘플에서 랜덤으로 일부만 사용해 분포를 줄입니다. 데이터가 충분히 많은 데에만 권장합니다.
- **Stratified split 재검토**: 검증/테스트 데이터도 학습과 동일한 분포를 갖도록 재분할해 평가 안정성을 높입니다.

## 2. 학습 단계: 설정 조정
- **클래스 가중치**: `src/transformer.py`에서 기본 제공하는 inverse-frequency 가중치를 활성화합니다 (`configs/config.json` → `transformer.use_class_weights: true`). 이 설정은 이미 기본값으로 적용돼 있으며, 슬픔/불안 위주로 수렴하던 손실 함수를 보정합니다.
- **하이퍼파라미터 튜닝**:
  - `num_train_epochs`, `learning_rate`, `warmup_ratio` 등을 조정해 소수 감정 학습을 충분히 진행합니다.
  - `per_device_train_batch_size`를 상황에 맞게 확대하면 step 수가 줄어 학습 시간이 단축되지만, 메모리 한계를 고려합니다.
- **사전학습 모델 교체**: `transformer.model_name`을 `klue/roberta-large`, `snunlp/KR-ELECTRA`, `beomi/KcELECTRA` 등 다른 한국어 모델로 변경해 표현력을 강화합니다.
- **정규화 전략**: `weight_decay`, 드롭아웃, 데이터 셔플 시드 등을 조절해 과적합을 방지합니다.
- **실험 관리**: 에폭별 `metrics.json`을 버전별로 보관하거나 MLflow/Weights & Biases 같은 실험 추적 도구를 도입해 비교·재현성을 높입니다.

## 3. 평가 및 사후 처리 단계
- **클래스별 지표 모니터링**: macro-F1 외에도 각 감정별 F1, 정밀도, 재현율을 산출해 소수 감정 개선 여부를 확인합니다.
- **임계값(Threshold) 조정**:
  - 소프트맥스 확률이 일정 값 이상일 때만 해당 감정으로 판정하도록 클래스별 임계값을 설정합니다.
  - 행복/평온은 임계값을 낮춰 탐지를 쉽게 하고, 슬픔/불안은 임계값을 높여 편향을 완화합니다.
  - 임계값을 조정한 뒤에도 어떤 클래스도 기준을 넘지 못하면 최고 확률 클래스를 선택하도록 안전 장치를 둡니다.
- **후순위 보정**: 상위 2개 감정의 확률 차이가 작고 소수 감정이 후순위라면 키워드 규칙(`configs/config.json`의 `keyword_rules`)을 참고해 감정을 교정합니다.
- **확률 보정(Calibration)**: Platt Scaling, Isotonic Regression 등을 적용해 출력 확률의 신뢰도를 높이면, 임계값/룰 기반 판단을 더 정확하게 할 수 있습니다.

## 4. 권장 워크플로우
1. 데이터 분포 파악 → 필요 시 업/다운샘플링 또는 증강 적용.
2. `transformer.use_class_weights`가 활성화된 상태로 다양한 하이퍼파라미터·모델을 실험.
3. 학습 시 `transformer.sampling_strategy`를 `balanced`로 두어 배치마다 균형 잡힌 샘플링이 이뤄지도록 하고, 로그(`trainer_state.json`)로 정상 작동 여부를 확인합니다.
4. 학습 후 `metrics.json`과 클래스별 F1을 확인하고, `transformer.default_threshold` 및 `probability_thresholds`로 소프트맥스 임계값을 조정한 뒤 필요하면 룰 기반 보정과 결합합니다.
5. 효과적인 조합을 문서화하고, 반복 실험 시 동일 절차를 재사용.

### 최신 설정 체크리스트
- **균형 샘플링**: `configs/config.json` → `transformer.sampling_strategy=balanced` 로 두면 `WeightedRandomSampler`가 활성화되어 행복/평온 같은 소수 클래스가 매 배치에 충분히 포함됩니다.
- **손실 가중치**: `transformer.use_class_weights=true`로 inverse-frequency 가중치를 유지해 다수 클래스 편향을 추가로 완화합니다.
- **임계값 후처리**: `transformer.default_threshold`(기본 0.3)와 `probability_thresholds`(행복/평온 0.2, 슬픔/불안 0.45 등)을 조절해 소프트맥스 확률을 기준으로 판정을 보정합니다. 추론 코드에서도 동일 설정을 로드해야 일관성이 유지됩니다.
- **평가 지표 저장**: 학습 종료 시 `processed/models/transformer/metrics.json`에 `eval_loss`가 함께 기록되고, `processed/models/transformer/final/probability_thresholds.json`으로 임계값이 저장되므로 과적합 점검 및 추론 연동이 쉬워졌습니다.
- **실행 예시**: `set PYTHONPATH=. && envs\python.exe scripts\train_transformer.py --config configs\config.json` 또는 `envs\python.exe -m scripts.train_transformer --config configs\config.json`을 사용해 학습을 실행합니다.

## 추가 팁
- **테스트 데이터 분리**: 최종 성능 확인용 데이터셋은 학습/튜닝에 사용하지 않고 별도 보관합니다.
- **오류 분석**: 소수 감정에서 오분류된 문장을 모아 공통 패턴(문장 길이, 표현 방식 등)을 분석하고, 전처리/사후 처리 규칙 개선에 반영합니다.
- **협업/버전 관리**: 개선 전략과 실험 로그를 `docs/` 폴더에 지속 기록해 팀 간 지식 공유를 촉진합니다.

데이터 불균형 문제는 단일 기법으로 완전히 해결되기 어렵습니다. 위 단계들을 조합해 반복적으로 실험하고, 각 전략이 미치는 영향을 체계적으로 기록하면 다중 감정 분류 모델의 안정성과 정확도를 꾸준히 향상시킬 수 있습니다.
