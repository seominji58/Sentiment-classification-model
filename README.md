# Sentiment Classification Model

Transformer 기반 한국어 감정 분류 시스템으로, 원시 발화 데이터를 정제·전처리한 뒤 다중 감정(분노, 슬픔, 불안, 행복, 평온 등)을 분류하도록 학습됩니다. 대규모 클래스 불균형을 완화하기 위해 inverse-frequency 손실 가중치를 적용하며, 학습 전 과정을 스크립트 형태로 자동화했습니다.

## 아키텍처 개요
- **데이터 준비**: `processed/splits/*.csv`에 학습/검증용 문장과 라벨이 저장되며, `configs/config.json`의 `label_mapping` 및 `emotion_hierarchy`가 라벨 계층 정보를 제공합니다.
- **모델 계층**: Hugging Face `AutoModelForSequenceClassification`을 기반으로 하여 사전학습 모델(`klue/bert-base` 기본값)을 로드하고, `src/transformer.py`에서 토크나이저/데이터세트를 구성한 뒤 `Trainer`로 파인튜닝합니다.
- **클래스 불균형 대응**: 학습 시 라벨 분포를 계산해 inverse-frequency 가중치를 산출하고 `CrossEntropyLoss`에 주입합니다(`transformer.use_class_weights` 옵션). 이를 통해 슬픔·불안에 치우친 데이터에서도 행복·평온 등 소수 클래스에 대한 감도를 유지합니다.
- **산출물 관리**: 체크포인트와 최종 모델(`model.safetensors`, `tokenizer.json` 등)은 `processed/models/transformer/`에 저장되며, 평가 지표는 `metrics.json`으로 기록됩니다.

## 주요 디렉터리
- `configs/`: 학습 하이퍼파라미터, 라벨 매핑, 규칙 기반 보조 정보.
- `scripts/`: 파이프라인 엔트리 포인트(`train_transformer.py`, `run_all.py` 등).
- `src/`: 공통 유틸리티(`utils.py`)와 모델 래퍼(`transformer.py`).
- `processed/`: 전처리 결과, 데이터 분할, 모델 산출물(버전 관리 제외).
- `tests/`: 핵심 컴포넌트 단위 테스트.

## 학습 파이프라인
1. `scripts/train_transformer.py`가 설정 파일을 로드하고 시드를 고정합니다.
2. 학습/검증 CSV를 `Dataset` 객체로 변환하고 토큰화합니다.
3. 클래스 가중치가 활성화된 `WeightedTrainer`로 모델을 파인튜닝하며, 에폭마다 평가와 체크포인트 저장을 수행합니다.
4. 최종 모델과 토크나이저를 내보내고, `metrics.json`에 핵심 지표(`eval_accuracy`, `eval_macro_f1`, `train_runtime` 등)를 기록합니다.

## 사용 방법
```bash
# 가상환경 활성화 후 학습 실행
python -m scripts.train_transformer

# 하이퍼파라미터 혹은 클래스 가중치 사용 여부는 configs/config.json 수정
```

모델 구조 변경(예: `transformer.model_name`을 `klue/roberta-large`로 변경), 배치 크기, 에폭 수 조정을 통해 원하는 성능/시간 균형을 맞출 수 있습니다. 학습 이후에는 `processed/models/transformer/metrics.json` 및 소수 클래스 문장에 대한 추론 결과를 검토하여 개선 효과를 검증합니다.
