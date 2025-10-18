"""Transformer-based fine-tuning pipeline."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from . import utils


def _import_transformers():
    try:
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:  # pragma: no cover - dependency guard
        utils.fail(
            "transformers 또는 datasets 라이브러리가 설치되어 있지 않습니다. "
            "pip install transformers datasets torch"
        )
    return Dataset, AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments


def train(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    *,
    text_column: str = "text",
    label_column: str = "label",
    config_path: str = "configs/config.json",
    checkpoint_dir: str = "processed/models/transformer/checkpoints",
    final_dir: str = "processed/models/transformer/final",
    metrics_path: str = "processed/models/transformer/metrics.json",
) -> Dict[str, float]:
    utils.ensure_columns(train_df, [text_column, label_column])
    utils.ensure_columns(valid_df, [text_column, label_column])

    config = utils.Config.load(config_path)
    tfm_cfg = config.get("transformer", {})

    try:
        import torch
        from torch.utils.data import DataLoader, WeightedRandomSampler
    except ImportError as exc:  # pragma: no cover - dependency guard
        utils.fail(
            "torch 또는 torch.utils.data 모듈을 불러올 수 없습니다. pip install torch"
        )

    (
        Dataset,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    ) = _import_transformers()

    label_space = sorted(train_df[label_column].unique())
    label2id = {label: idx for idx, label in enumerate(label_space)}
    id2label = {idx: label for label, idx in label2id.items()}
    num_labels = len(label_space)

    if num_labels < 2:
        utils.fail("Transformer 학습을 위해서는 최소 2개 이상의 라벨이 필요합니다.")

    train_encoded = train_df.assign(label_id=train_df[label_column].map(label2id))
    valid_encoded = valid_df.assign(label_id=valid_df[label_column].map(label2id))

    train_dataset = Dataset.from_pandas(
        train_encoded[[text_column, "label_id"]], preserve_index=False
    ).rename_column("label_id", "labels")
    valid_dataset = Dataset.from_pandas(
        valid_encoded[[text_column, "label_id"]], preserve_index=False
    ).rename_column("label_id", "labels")

    use_class_weights = bool(tfm_cfg.get("use_class_weights", True))
    sampling_strategy: Optional[str] = tfm_cfg.get("sampling_strategy", "balanced")
    if sampling_strategy not in {"balanced", "none"}:
        utils.fail(
            "transformer.sampling_strategy는 'balanced' 또는 'none' 이어야 합니다."
        )
    class_weights_tensor = None

    if use_class_weights:
        label_ids = train_encoded["label_id"].to_numpy()
        class_counts = np.bincount(label_ids, minlength=num_labels)
        if (class_counts == 0).any():
            utils.fail("라벨 분포를 계산할 수 없습니다. train 데이터에 모든 라벨이 포함되어 있는지 확인하세요.")

        # inverse frequency weighting: total/(num_labels * count)
        total = class_counts.sum()
        weights = total / (num_labels * class_counts.astype(np.float64))
        class_weights_tensor = torch.tensor(weights, dtype=torch.float32)

    default_threshold = tfm_cfg.get("default_threshold")
    threshold_overrides = tfm_cfg.get("probability_thresholds")
    thresholds_array: Optional[np.ndarray] = None

    if default_threshold is not None or threshold_overrides:
        if default_threshold is None:
            default_value = 0.0
        else:
            default_value = float(default_threshold)
        if not 0.0 <= default_value < 1.0:
            utils.fail("transformer.default_threshold는 0 이상 1 미만의 값이어야 합니다.")

        thresholds_array = np.full(num_labels, default_value, dtype=np.float64)

        if threshold_overrides:
            for label, value in threshold_overrides.items():
                if label not in label2id:
                    utils.fail(
                        f"probability_thresholds에 정의된 라벨 '{label}' 이(가) 학습 데이터 라벨에 없습니다."
                    )
                threshold_value = float(value)
                if not 0.0 <= threshold_value < 1.0:
                    utils.fail(
                        "probability_thresholds 값은 0 이상 1 미만이어야 합니다."
                    )
                thresholds_array[label2id[label]] = threshold_value

    tokenizer = AutoTokenizer.from_pretrained(tfm_cfg.get("model_name", "klue/bert-base"))

    max_length = int(tfm_cfg.get("max_length", 128))

    def tokenize_fn(examples):
        return tokenizer(
            examples[text_column],
            truncation=True,
            padding=False,
            max_length=max_length,
        )

    remove_columns = [text_column]
    train_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=remove_columns)
    valid_dataset = valid_dataset.map(tokenize_fn, batched=True, remove_columns=remove_columns)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        tfm_cfg.get("model_name", "klue/bert-base"),
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    )

    output_dir = Path(checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 일부 구형 transformers 버전에서는 TrainingArguments 인자가 제한적이므로
    # 시그니처를 기준으로 허용된 키만 전달한다.
    import inspect

    desired_kwargs = {
        "output_dir": str(output_dir),
        "evaluation_strategy": tfm_cfg.get("evaluation_strategy", "epoch"),
        "save_strategy": tfm_cfg.get("save_strategy", "epoch"),
        "learning_rate": float(tfm_cfg.get("learning_rate", 5e-5)),
        "per_device_train_batch_size": int(tfm_cfg.get("per_device_train_batch_size", 16)),
        "per_device_eval_batch_size": int(tfm_cfg.get("per_device_eval_batch_size", 32)),
        "num_train_epochs": float(tfm_cfg.get("num_train_epochs", 3)),
        "weight_decay": float(tfm_cfg.get("weight_decay", 0.01)),
        "warmup_ratio": float(tfm_cfg.get("warmup_ratio", 0.1)),
        "logging_steps": int(tfm_cfg.get("logging_steps", 20)),
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "report_to": ["none"],
        "save_total_limit": 2,
    }

    signature = inspect.signature(TrainingArguments.__init__)
    valid_params = set(signature.parameters)

    eval_strategy = desired_kwargs.get("evaluation_strategy", "no")
    save_strategy = desired_kwargs.get("save_strategy", "no")

    evaluation_supported = "evaluation_strategy" in valid_params
    save_supported = "save_strategy" in valid_params

    # 구버전 호환용 evaluate_during_training 매핑
    if not evaluation_supported:
        eval_strategy = desired_kwargs.pop("evaluation_strategy", "no")
        if "evaluate_during_training" in valid_params:
            desired_kwargs["evaluate_during_training"] = eval_strategy != "no"
    if not save_supported:
        save_strategy = desired_kwargs.pop("save_strategy", "no")

    final_eval_strategy = desired_kwargs.get("evaluation_strategy", eval_strategy)
    final_save_strategy = desired_kwargs.get("save_strategy", save_strategy)

    # load_best_model_at_end는 평가/저장 전략 일치가 필요하다.
    load_best = desired_kwargs.get("load_best_model_at_end")
    if load_best and (
        (not evaluation_supported)
        or (not save_supported)
        or (final_eval_strategy not in {"epoch", "steps"})
        or (final_save_strategy not in {"epoch", "steps"})
        or (final_eval_strategy != final_save_strategy)
    ):
        desired_kwargs["load_best_model_at_end"] = False
        desired_kwargs.pop("metric_for_best_model", None)
        desired_kwargs.pop("greater_is_better", None)

    for key in list(desired_kwargs.keys()):
        if key not in valid_params:
            desired_kwargs.pop(key)

    training_args = TrainingArguments(**desired_kwargs)

    from sklearn.metrics import accuracy_score, f1_score

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]

        if thresholds_array is not None:
            shifted = logits - logits.max(axis=-1, keepdims=True)
            probs = np.exp(shifted)
            probs /= probs.sum(axis=-1, keepdims=True)

            chosen = []
            for prob_vector in probs:
                mask = prob_vector >= thresholds_array
                if np.any(mask):
                    candidate_indices = np.where(mask)[0]
                    best_idx = candidate_indices[np.argmax(prob_vector[mask])]
                    chosen.append(best_idx)
                else:
                    chosen.append(int(np.argmax(prob_vector)))
            predictions = np.array(chosen, dtype=np.int64)
        else:
            predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, predictions),
            "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        }

    trainer_kwargs: Dict[str, object] = {}
    if sampling_strategy == "balanced":
        trainer_kwargs["sampling_strategy"] = sampling_strategy

    class SamplingTrainerMixin:
        def __init__(self, *args, sampling_strategy: str = "none", **kwargs):
            self.sampling_strategy = sampling_strategy
            super().__init__(*args, **kwargs)

        def _build_balanced_sampler(self) -> WeightedRandomSampler:
            if self.train_dataset is None:
                raise ValueError("train_dataset이 설정되지 않았습니다.")

            labels = self.train_dataset["labels"]
            labels_array = np.asarray(labels, dtype=np.int64)
            if labels_array.size == 0:
                raise ValueError("train_dataset에 레이블이 비어 있습니다.")

            class_counts = np.bincount(
                labels_array, minlength=self.model.config.num_labels
            ).astype(np.float64)
            class_counts[class_counts == 0] = 1.0
            sample_weights = 1.0 / class_counts[labels_array]
            return WeightedRandomSampler(
                sample_weights.tolist(),
                len(sample_weights),
                replacement=True,
            )

        def get_train_dataloader(self):
            if self.sampling_strategy != "balanced":
                return super().get_train_dataloader()

            sampler = self._build_balanced_sampler()
            return DataLoader(
                self.train_dataset,
                batch_size=self.args.train_batch_size,
                sampler=sampler,
                collate_fn=self.data_collator,
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

    if class_weights_tensor is not None:
        class WeightedTrainer(SamplingTrainerMixin, Trainer):
            def __init__(
                self,
                *args,
                class_weights: "torch.Tensor",
                sampling_strategy: str = "none",
                **kwargs,
            ):
                self.class_weights = class_weights.clone().detach()
                super().__init__(
                    *args,
                    sampling_strategy=sampling_strategy,
                    **kwargs,
                )
                if self.class_weights.dim() != 1:
                    raise ValueError("class_weights는 1차원 텐서여야 합니다.")

            def compute_loss(self, model, inputs, return_outputs=False, **_):
                labels = inputs["labels"]
                outputs = model(**inputs)
                if hasattr(outputs, "logits"):
                    logits = outputs.logits
                elif isinstance(outputs, dict):
                    logits = outputs["logits"]
                else:
                    logits = outputs[0]
                weights = self.class_weights.to(logits.device)
                loss_fct = torch.nn.CrossEntropyLoss(weight=weights)
                loss = loss_fct(logits.view(-1, weights.size(0)), labels.view(-1))
                return (loss, outputs) if return_outputs else loss

        trainer_cls = WeightedTrainer
        trainer_kwargs["class_weights"] = class_weights_tensor
    else:
        class BalancedTrainer(SamplingTrainerMixin, Trainer):
            pass

        trainer_cls = BalancedTrainer if sampling_strategy == "balanced" else Trainer

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        **trainer_kwargs,
    )

    train_result = trainer.train()
    eval_metrics = trainer.evaluate()

    final_path = Path(final_dir)
    final_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    metrics = {
        "train_runtime": float(train_result.metrics.get("train_runtime", 0.0)),
        "train_samples_per_second": float(train_result.metrics.get("train_samples_per_second", 0.0)),
        "eval_accuracy": float(eval_metrics.get("eval_accuracy", 0.0)),
        "eval_macro_f1": float(eval_metrics.get("eval_macro_f1", 0.0)),
    }

    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    utils.log_step(
        "transformer",
        eval_macro_f1=round(metrics["eval_macro_f1"], 4),
        eval_accuracy=round(metrics["eval_accuracy"], 4),
    )
    utils.mark_ok("transformer")
    return metrics


__all__ = ["train"]
