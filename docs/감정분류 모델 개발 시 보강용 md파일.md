# ⚠️ Pipeline Tips Addendum (for Cursor AI)

본 문서는 `PIPELINE_GUIDE.md`를 보완하는 **누락되기 쉬운 디테일/가이드** 모음입니다.  
Cursor에게 그대로 붙여넣어 “이 기준을 충족하는 코드만 생성/수정”하게 하세요.

---

## 1) 데이터 계약(Data Contract)

**입력 CSV 스키마(최소):**

- 컬럼: `text`, `label` (추가 컬럼은 무시 가능)
    
- 인코딩: `utf-8` 또는 `utf-8-sig`
    
- `text`: 문자열, 길이 ≥ 1 (공백만인 행 제거)
    
- `label`: 문자열 (최종 5클래스로 매핑됨)
    

**불변 규칙:**

- 전처리는 **훈련/검증 나누기 이전**에만 수행 (데이터 누수 금지).
    
- 라벨 매핑은 **단일 진실원** `configs/config.json` 또는 `processed/label_mapping.json` 에서 **로드만** 함.
    
- 어떤 단계든 **필수 컬럼/비어있음** 검사 실패 시 즉시 `RuntimeError`로 중단.
    

---

## 2) 한국어 텍스트 정규화 권장안

전처리(`preprocess.py`)에 아래를 옵션으로 제공:

- 유니코드 정규화: NFKC
    
- 공백 정리: 다중 공백 → 단일 공백
    
- 양끝 공백 제거, 제어문자 제거
    
- 특수 이모티콘/이모지 보존 여부 옵션(`keep_emoji: true`)
    
- 길이 필터링(예: 2자 미만 제거)
    
- 중복 문장 제거(정규화 적용 후 기준)
    
- (선택) 반복 문자 축약: `ㅋㅋㅋㅋ → ㅋㅋ`, `ㅎㅎㅎㅎ → ㅎㅎ`
    
- (선택) 영문/숫자 표준화: 전각→반각
    

> **주의:** 토큰화 전처리가 모델과 상충하지 않도록 **Transformer 파이프라인에서는 최소 전처리** 원칙.

---

## 3) 라벨 매핑(6→5) 디테일

**`configs/config.json` 예시(확장):**

`{   "seed": 42,   "labels": ["행복","평온","불안","분노","슬픔"],   "label_mapping": {     "기쁨": "행복",     "중립": "평온",     "평온": "평온",     "불안": "불안",     "당황": "불안",     "분노": "분노",     "상처": "슬픔",     "슬픔": "슬픔"   },   "split": { "valid_ratio": 0.1, "stratified": true, "min_per_class_valid": 1 },   "baseline": {     "tfidf": { "analyzer": "char_wb", "ngram_range": [3, 5], "min_df": 2, "max_features": 200000 },     "svc": { "C": 2.0, "class_weight": null }   },   "transformer": {     "model_name": "klue/roberta-base",     "epochs": 3,     "batch_size": 16,     "learning_rate": 3e-5,     "max_length": 256,     "early_stopping_patience": 2,     "warmup_ratio": 0.06,     "fp16": true   } }`

**검증:**

- 매핑 후 `final_df['label'].nunique() == 5` 아니면 즉시 중단.
    
- 각 클래스 훈련/검증 모두 최소 1개 보장 (`min_per_class_valid`).
    

---

## 4) Split 정책 (누수/희귀 클래스)

- Stratified 90/10 기본.
    
- 클래스 수가 적어 stratify 불가 시:
    
    - 기본: 중단하고 사용자에게 `--allow-random-split` 옵션 안내.
        
    - 옵션 허용 시에만 Random Split로 폴백 + 로그에 “폴백” 명시.
        
- `processed/splits/train.csv`, `valid.csv` 저장 직후 클래스 분포를 `metrics.json`에 기록.
    

---

## 5) 베이스라인 세부 팁

- **Vectorizer:** `char_wb 3~5그램`이 한국어 짧은 문장에 강함.
    
- **Classifier:** LinearSVC (빠르고 강건).
    
    - 클래스 심각 불균형이면 언더/오버샘플링으로 대응 (imblearn 옵션으로 분리).
        
- **평가:** macro-F1 필수, confusion matrix 저장.
    
- **산출:** `processed/models/baseline/{model.joblib, metrics.json}`.
    

---

## 6) Transformer 학습 체크리스트

- **모델:** `klue/roberta-base` (한국어 강함).
    
    - 대안: `snunlp/KR-ELECTRA`, `kykim/roberta-*`.
        
- **설정:**
    
    - `id2label/label2id`를 config 기반으로 단일 생성 후 모델에 주입.
        
    - `load_best_model_at_end=True`
        
    - `metric_for_best_model="f1"` (macro F1)
        
    - `EarlyStoppingCallback(patience=config.transformer.early_stopping_patience)`
        
- **데이터셋:**
    
    - `datasets.Dataset.from_pandas → tokenize → (선택) save_to_disk`
        
- **저장:**
    
    - `processed/models/transformer/checkpoints/*`
        
    - `processed/models/transformer/final/*`
        
    - `metrics.json` 갱신
        

---

## 7) 이상치 탐지(어노테이션 점검)

- `anomaly.py`:
    
    - `TfidfVectorizer(analyzer="char", ngram_range=(3,5))`
        
    - 코사인 유사도 Top-K(예: 20)에서 다수 라벨과 상이한 표본을 CSV로 저장:
        
    - `processed/anomaly_samples.csv` (컬럼: text, label, nearest_labels, disagreement_score)
        

---

## 8) 실험 재현 & 로깅

- **Seed 고정:** `numpy/random/torch` 모두 42.
    
- **metrics.json:** 단계별로 누적 업데이트.
    

예시:

`{   "data": {"n_total": 39208, "n_train": 35287, "n_valid": 3921,            "dist_train": {"행복": 7000, ...}, "dist_valid": {...}},   "baseline": {"accuracy": 0.78, "macro_f1": 0.74},   "transformer": {"accuracy": 0.83, "macro_f1": 0.80} }`

- **스모크 테스트:** 최초 실행 시 `--sample 1000` 지원.
    

---

## 9) 성능 기준 (acceptance)

- `python scripts/run_all.py` 한 번으로 끝까지 통과.
    
- `processed/` 산출물 모두 존재 + 파일 크기 > 0.
    
- Transformer macro-F1 ≥ Baseline macro-F1 (개선 실패 시 경고 + 원인 로그).
    
- 재시작 후 동일 결과 재현 가능.
    

---

## 10) 실패/복구 플레이북

- `AssertionError: text/label 누락` → 전처리 단계 재검사, 컬럼명 매핑 추가.
    
- `Stratify 오류` → 희귀 클래스(샘플 수) 출력 후 옵션 폴백 안내.
    
- `OOM/메모리` → 배치 축소, max_length 축소, fp16=True, num_workers=0.
    
- `토크나이저 경고` → 특수문자/이모지 보존 옵션 확인. 과도한 정규화 제한.
    

---

## 11) Git/환경 위생

- `.gitignore`에 `processed/`, `.ipynb_checkpoints/`, `__pycache__/` 포함.
    
- `.gitattributes`로 EOL 통일 (`* text=auto`).
    
- `requirements.txt` 고정, 필요 시 pip-tools로 lock.
    

---

## 12) Cursor 지시 프롬프트 예시 (복붙)

- Create files skeleton
    
- Create all files & dirs defined in PIPELINE_GUIDE.md and this ADDENDUM.
    
- Put minimal, runnable code stubs with TODO markers.
    
- Add requirements.txt.
    
- Implement data_io, preprocess with hard acceptance checks.
    
- Implement load/save with encoding utf-8-sig fallback.
    
- Add assert and file-size check on save.
    
- Implement label_map using config-only.
    
    - No hard-coded mapping. Fail if mapping missing or results ≠ 5 classes.
        
- Implement split with stratified default and guarded fallback.
    
    - If stratify fails, stop unless `--allow-random-split` is passed.
        
- Implement baseline and transformer training.
    
    - Baseline: TF-IDF(char_wb 3–5) + LinearSVC, macro-F1 to metrics.json.
        
    - Transformer: KLUE/RoBERTa-base, EarlyStopping, best model save.
        
- Add smoke-run CLI (`--sample N`, `--allow-random-split`, `--out processed`, `--config configs/config.json`).
    

---

## 13) 흔한 함정 (Do / Don’t)

- ❌ Don’t: 라벨 매핑을 코드에서 재정의(하드코딩).
    
- ❌ Don’t: 전처리 후에 split (누수).
    
- ✅ Do: 저장 직후 파일 크기 확인.
    
- ✅ Do: macro-F1 기준으로 모델 버전 선택.
    
- ✅ Do: 실패 시 멈추고 원인 로그 남기기.
    

---

## 14) 최종 산출물 체크리스트

- `processed/cleaned.csv`
    
- `processed/label_mapping.json`
    
- `processed/splits/train.csv`, `processed/splits/valid.csv`
    
- `processed/anomaly_samples.csv`
    
- `processed/models/baseline/model.joblib`, `processed/models/baseline/metrics.json`
    
- `processed/models/transformer/final/*`, `processed/models/transformer/metrics.json`
    

---

필요하면 위 ADDENDUM에 맞춰 **샘플 `config.json`**,  
`run_all.py`의 **CLI 인자 목록**(`--src`, `--config`, `--out`, `--allow-random-split`, `--sample`)도 추가 가능.