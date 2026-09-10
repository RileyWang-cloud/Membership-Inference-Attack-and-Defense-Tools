from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[1]
UTIL_DIR = Path(__file__).resolve().parent / "utils_secmia"
for path in (REPO_ROOT, UTIL_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from Attack.shadow_based import AttackInput, AttackOutput, BaseAttack, EvaluationResult


_SECMI_UTILS = importlib.import_module("Attack.utils_secmia.secmia")


def _safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    try:
        if len(np.unique(y_true)) < 2:
            return None
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return None


def _tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float) -> float:
    try:
        fpr, tpr, _ = roc_curve(y_true, y_score)
    except ValueError:
        return 0.0
    idx = int(np.argmin(np.abs(fpr - target_fpr)))
    return float(tpr[idx])


def _to_numpy_1d(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.reshape(-1)
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().reshape(-1)
    return np.asarray(value).reshape(-1)


def _to_tensor_4d_or_5d(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.detach().cpu()
    return torch.as_tensor(value)


def _load_flagfile_defaults() -> Dict[str, Any]:
    return {
        "T": 1000,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "ch": 128,
        "ch_mult": [1, 2, 2, 2],
        "attn": [1],
        "num_res_blocks": 2,
        "dropout": 0.1,
        "k": 3,
        "num_t_groups": 5,
        "sparsity": 0.3,
        "parallel": True,
        "dataset": "cifar10",
        "data_root": "./datasets",
        "model_type": "ddpm",
        "ckpt_name": "",
        "output_save_dir": "./experiment_results/recent",
        "t_sec": 100,
        "m": 10,
    }


def _coerce_flag_value(key: str, raw: str) -> Any:
    bool_keys = {"parallel"}
    int_keys = {"T", "ch", "num_res_blocks", "k", "num_t_groups", "t_sec", "m"}
    float_keys = {"beta_1", "beta_T", "dropout", "sparsity"}
    list_int_keys = {"ch_mult", "attn"}

    if key in bool_keys:
        return raw.lower() in {"true", "1", "yes"}
    if key in int_keys:
        return int(raw)
    if key in float_keys:
        return float(raw)
    if key in list_int_keys:
        return [int(part) for part in raw.split(",") if part]
    return raw


def _parse_flagfile(flag_path: Optional[str], overrides: Dict[str, Any]) -> SimpleNamespace:
    values = _load_flagfile_defaults()
    if flag_path and os.path.exists(flag_path):
        repeated_list_keys: set[str] = set()
        with open(flag_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#") or not line.startswith("--"):
                    continue
                body = line[2:]
                if "=" in body:
                    key, raw = body.split("=", 1)
                else:
                    key, raw = body, "true"
                parsed = _coerce_flag_value(key, raw)
                # absl DEFINE_multi_integer accepts repeated --ch_mult lines.
                if key in {"ch_mult", "attn"}:
                    if key not in repeated_list_keys:
                        values[key] = []
                        repeated_list_keys.add(key)
                    values[key].extend(parsed)
                else:
                    values[key] = parsed
    values.update({k: v for k, v in overrides.items() if v is not None})
    return SimpleNamespace(**values)


class SecMIAAttack(BaseAttack):
    """
    Unified wrapper for the SecMIA diffusion-model attack.

    Supported modes
    ---------------
    1. Direct stat attack:
       - no `fit()` needed
       - `infer()` computes per-sample reconstruction-error style scores

    2. NNS attack:
       - `fit()` trains the NNS attack head from `shadow_data`
       - `infer()` applies the trained attack head to query samples

    Supported `AttackInput` patterns
    --------------------------------
    A. Fully precomputed query signals:
       signals = {
           "internal_diffusions": Tensor/ndarray,
           "internal_denoise": Tensor/ndarray,
       }

    B. Precomputed query scores:
       signals = {
           "stat_scores": Tensor/ndarray,
       }

    C. Raw query samples:
       - provide `target_model` plus `samples`
       - or provide `config`/`metadata` with `model_dir`, `ckpt_name`, `flag_path`

    D. Benchmark-style member/non-member loading from dataset split:
       config or metadata should contain:
           `data_root`, `dataset`

    E. NNS shadow training data:
       shadow_data = {
           "member_diffusions": ...,
           "member_internal_samples": ...,
           "nonmember_diffusions": ...,
           "nonmember_internal_samples": ...,
       }
       or raw shadow samples + shadow model/model path.
    """

    def __init__(
        self,
        attack_variant: str = "stat",
        t_sec: int = 100,
        m: int = 10,
        batch_size: int = 128,
        model_type: str = "ddpm",
        k: int = 3,
        num_t_groups: int = 5,
        sparsity: float = 0.3,
        output_save_dir: str = "./experiment_results/recent",
        nns_train_portion: float = 0.2,
        compute_fid: bool = False,
        fid_num_images: int = 100,
        fid_batch_size: Optional[int] = None,
        fid_cache: Optional[str] = None,
        fid_use_torch: bool = False,
        device: Optional[str] = None,
    ) -> None:
        self.attack_variant = attack_variant
        self.t_sec = t_sec
        self.m = m
        self.batch_size = batch_size
        self.model_type = model_type
        self.k = k
        self.num_t_groups = num_t_groups
        self.sparsity = sparsity
        self.output_save_dir = output_save_dir
        self.nns_train_portion = nns_train_portion
        self.compute_fid = compute_fid
        self.fid_num_images = fid_num_images
        self.fid_batch_size = fid_batch_size
        self.fid_cache = fid_cache
        self.fid_use_torch = fid_use_torch
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

        self.runtime_config: Dict[str, Any] = {}
        self.nns_model: Optional[torch.nn.Module] = None
        self.is_fitted = False

    def fit(self, attack_input: AttackInput) -> "SecMIAAttack":
        self.runtime_config = self._merge_config(attack_input.config)
        if self.runtime_config["attack_variant"] != "nns":
            self.is_fitted = True
            return self

        shadow_t_results = self._resolve_shadow_t_results(attack_input)
        _, _, model = _SECMI_UTILS.nns_attack(
            shadow_t_results,
            train_portion=float(self.runtime_config["nns_train_portion"]),
            device=self.device,
        )
        self.nns_model = model
        self.is_fitted = True
        return self

    def infer(self, attack_input: AttackInput) -> AttackOutput:
        self.runtime_config = self._merge_config(attack_input.config)
        variant = self.runtime_config["attack_variant"]

        if variant == "stat":
            scores, extra = self._infer_stat_scores(attack_input)
            metadata = {
                "attack_name": "secmia",
                "attack_variant": "stat",
                "score_direction": "higher_is_member",
                "t_sec": self.runtime_config["t_sec"],
                "m": self.runtime_config["m"],
            }
            fid = self._maybe_compute_fid(attack_input)
            if fid is not None:
                metadata["fid"] = fid
            return AttackOutput(
                membership_scores=scores,
                membership_preds=None,
                intermediate_outputs=extra,
                metadata=metadata,
            )

        if variant == "nns":
            if not self.is_fitted or self.nns_model is None:
                raise RuntimeError("SecMIAAttack with attack_variant='nns' must be fitted before infer().")
            features, extra = self._resolve_query_nns_features(attack_input)
            with torch.no_grad():
                logits = self.nns_model(features.to(self.device)).reshape(-1)
            scores = logits.detach().cpu().numpy().astype(np.float32)
            preds = (scores >= 0.5).astype(np.int64)
            metadata = {
                "attack_name": "secmia",
                "attack_variant": "nns",
                "score_direction": "higher_is_member",
                "pred_threshold": 0.5,
                "t_sec": self.runtime_config["t_sec"],
                "m": self.runtime_config["m"],
            }
            fid = self._maybe_compute_fid(attack_input)
            if fid is not None:
                metadata["fid"] = fid
            return AttackOutput(
                membership_scores=scores,
                membership_preds=preds,
                intermediate_outputs=extra,
                metadata=metadata,
            )

        raise ValueError(f"Unsupported SecMIA variant: {variant}")

    def evaluate(self, attack_output: AttackOutput, attack_input: AttackInput) -> EvaluationResult:
        y_true = _to_numpy_1d(attack_input.membership_labels).astype(np.int64)
        y_score = _to_numpy_1d(attack_output.membership_scores).astype(np.float64)

        if attack_output.membership_preds is not None:
            y_pred = _to_numpy_1d(attack_output.membership_preds).astype(np.int64)
            threshold = float(attack_output.metadata.get("pred_threshold", 0.5))
        else:
            fpr, tpr, thresholds = roc_curve(y_true, y_score)
            accs = []
            for threshold_item in thresholds:
                pred = (y_score >= threshold_item).astype(np.int64)
                accs.append(accuracy_score(y_true, pred))
            best_idx = int(np.argmax(accs))
            threshold = float(thresholds[best_idx])
            y_pred = (y_score >= threshold).astype(np.int64)

        return EvaluationResult(
            accuracy=float(accuracy_score(y_true, y_pred)),
            auroc=_safe_auroc(y_true, y_score),
            tpr_at_fpr={
                "1%": _tpr_at_fpr(y_true, y_score, 0.01),
                "0.1%": _tpr_at_fpr(y_true, y_score, 0.001),
            },
            extra_metrics={"selected_threshold": threshold},
        )

    def _merge_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        merged = {
            "attack_variant": self.attack_variant,
            "t_sec": self.t_sec,
            "m": self.m,
            "batch_size": self.batch_size,
            "model_type": self.model_type,
            "k": self.k,
            "num_t_groups": self.num_t_groups,
            "sparsity": self.sparsity,
            "output_save_dir": self.output_save_dir,
            "nns_train_portion": self.nns_train_portion,
            "compute_fid": self.compute_fid,
            "fid_num_images": self.fid_num_images,
            "fid_batch_size": self.fid_batch_size,
            "fid_cache": self.fid_cache,
            "fid_use_torch": self.fid_use_torch,
        }
        merged.update(config)
        return merged

    def _maybe_compute_fid(self, attack_input: AttackInput) -> Optional[float]:
        """Compute optional generation-quality FID for the attacked DDPM."""
        if not bool(self.runtime_config.get("compute_fid", False)):
            return None
        if str(self.runtime_config.get("model_type", self.model_type)).lower() not in {"ddpm", "smcd"}:
            raise ValueError("FID evaluation supports model_type='ddpm' and 'smcd' only.")
        model = self._load_model_from_config(
            attack_input.target_model, self.runtime_config, attack_input.metadata,
        )
        flags_obj = self._resolve_flags(self.runtime_config, attack_input.metadata)
        return float(_SECMI_UTILS.calculate_fid(
            model, flags_obj,
            num_images=int(self.runtime_config["fid_num_images"]),
            batch_size=self.runtime_config.get("fid_batch_size"),
            fid_cache=self.runtime_config.get("fid_cache"),
            use_torch=bool(self.runtime_config.get("fid_use_torch", False)),
            device=self.device,
        ))

    def _infer_stat_scores(self, attack_input: AttackInput) -> Tuple[np.ndarray, Dict[str, Any]]:
        signals = attack_input.signals or {}
        if "stat_scores" in signals:
            raw_scores = _to_numpy_1d(signals["stat_scores"]).astype(np.float32)
            return raw_scores, {"raw_scores": raw_scores}

        if "internal_diffusions" in signals and "internal_denoise" in signals:
            diffusions = _to_tensor_4d_or_5d(signals["internal_diffusions"])
            denoise = _to_tensor_4d_or_5d(signals["internal_denoise"])
            raw_scores = self._stat_scores_from_internal(diffusions, denoise)
            return raw_scores, {
                "internal_diffusions": diffusions,
                "internal_denoise": denoise,
                "raw_scores": raw_scores,
            }

        partition = self._resolve_member_nonmember_partition(
            attack_input,
            prefer_benchmark_split=attack_input.samples is None,
        )
        if partition is not None:
            member_t = self._resolve_t_results_from_samples(
                attack_input=attack_input,
                samples=partition["member_samples"],
                model=partition["model"],
                flags_obj=partition["flags_obj"],
            )
            nonmember_t = self._resolve_t_results_from_samples(
                attack_input=attack_input,
                samples=partition["nonmember_samples"],
                model=partition["model"],
                flags_obj=partition["flags_obj"],
            )
            member_raw = self._stat_scores_from_internal(
                member_t["internal_diffusions"],
                member_t["internal_denoise"],
            )
            nonmember_raw = self._stat_scores_from_internal(
                nonmember_t["internal_diffusions"],
                nonmember_t["internal_denoise"],
            )
            combined = np.concatenate([member_raw, nonmember_raw], axis=0)
            return combined, {
                "member_scores": member_raw,
                "nonmember_scores": nonmember_raw,
                "member_internal_diffusions": member_t["internal_diffusions"],
                "member_internal_denoise": member_t["internal_denoise"],
                "nonmember_internal_diffusions": nonmember_t["internal_diffusions"],
                "nonmember_internal_denoise": nonmember_t["internal_denoise"],
                "raw_scores": combined,
            }

        t_results = self._resolve_t_results_for_query(attack_input)
        raw_scores = self._stat_scores_from_internal(
            t_results["internal_diffusions"],
            t_results["internal_denoise"],
        )
        membership_scores = -raw_scores
        return membership_scores, {
            "internal_diffusions": t_results["internal_diffusions"],
            "internal_denoise": t_results["internal_denoise"],
            "raw_scores": raw_scores,
        }

    def _resolve_query_nns_features(self, attack_input: AttackInput) -> Tuple[torch.Tensor, Dict[str, Any]]:
        signals = attack_input.signals or {}
        if "nns_features" in signals:
            features = torch.as_tensor(signals["nns_features"], dtype=torch.float32)
            return features, {"nns_features": features}

        if "internal_diffusions" in signals and "internal_denoise" in signals:
            features = self._nns_features_from_internal(
                _to_tensor_4d_or_5d(signals["internal_diffusions"]),
                _to_tensor_4d_or_5d(signals["internal_denoise"]),
            )
            return features, {"nns_features": features}

        partition = self._resolve_member_nonmember_partition(
            attack_input,
            prefer_benchmark_split=attack_input.samples is None,
        )
        if partition is not None:
            member_t = self._resolve_t_results_from_samples(
                attack_input=attack_input,
                samples=partition["member_samples"],
                model=partition["model"],
                flags_obj=partition["flags_obj"],
            )
            nonmember_t = self._resolve_t_results_from_samples(
                attack_input=attack_input,
                samples=partition["nonmember_samples"],
                model=partition["model"],
                flags_obj=partition["flags_obj"],
            )
            member_features = self._nns_features_from_internal(
                member_t["internal_diffusions"],
                member_t["internal_denoise"],
            )
            nonmember_features = self._nns_features_from_internal(
                nonmember_t["internal_diffusions"],
                nonmember_t["internal_denoise"],
            )
            return torch.cat([member_features, nonmember_features], dim=0), {
                "member_nns_features": member_features,
                "nonmember_nns_features": nonmember_features,
            }

        t_results = self._resolve_t_results_for_query(attack_input)
        features = self._nns_features_from_internal(
            t_results["internal_diffusions"],
            t_results["internal_denoise"],
        )
        return features, {
            "internal_diffusions": t_results["internal_diffusions"],
            "internal_denoise": t_results["internal_denoise"],
            "nns_features": features,
        }

    def _resolve_shadow_t_results(self, attack_input: AttackInput) -> Dict[str, torch.Tensor]:
        shadow_data = attack_input.shadow_data
        if shadow_data is None:
            raise ValueError("shadow_data is required for SecMIAAttack NNS training.")

        explicit_keys = {
            "member_diffusions",
            "member_internal_samples",
            "nonmember_diffusions",
            "nonmember_internal_samples",
        }
        if explicit_keys <= set(shadow_data.keys()):
            return {
                "member_diffusions": _to_tensor_4d_or_5d(shadow_data["member_diffusions"]),
                "member_internal_samples": _to_tensor_4d_or_5d(shadow_data["member_internal_samples"]),
                "nonmember_diffusions": _to_tensor_4d_or_5d(shadow_data["nonmember_diffusions"]),
                "nonmember_internal_samples": _to_tensor_4d_or_5d(shadow_data["nonmember_internal_samples"]),
            }

        shadow_model = shadow_data.get("shadow_model")
        if shadow_model is None:
            shadow_model = self._load_model_from_config(
                model=shadow_model,
                config={**self.runtime_config, **shadow_data},
                metadata={**attack_input.metadata, **shadow_data},
            )
        flags_obj = self._resolve_flags(
            config={**self.runtime_config, **shadow_data},
            metadata={**attack_input.metadata, **shadow_data},
        )

        if "member_samples" not in shadow_data or "nonmember_samples" not in shadow_data:
            raise ValueError(
                "SecMIA NNS shadow_data must provide either explicit intermediate tensors or "
                "'member_samples'/'nonmember_samples'."
            )

        member_t = self._resolve_t_results_from_samples(
            attack_input=attack_input,
            samples=shadow_data["member_samples"],
            model=shadow_model,
            flags_obj=flags_obj,
        )
        nonmember_t = self._resolve_t_results_from_samples(
            attack_input=attack_input,
            samples=shadow_data["nonmember_samples"],
            model=shadow_model,
            flags_obj=flags_obj,
        )
        return {
            "member_diffusions": member_t["internal_diffusions"],
            "member_internal_samples": member_t["internal_denoise"],
            "nonmember_diffusions": nonmember_t["internal_diffusions"],
            "nonmember_internal_samples": nonmember_t["internal_denoise"],
        }

    def _resolve_t_results_for_query(self, attack_input: AttackInput) -> Dict[str, torch.Tensor]:
        model = self._load_model_from_config(
            model=attack_input.target_model,
            config=self.runtime_config,
            metadata=attack_input.metadata,
        )
        flags_obj = self._resolve_flags(self.runtime_config, attack_input.metadata)
        return self._resolve_t_results_from_samples(
            attack_input=attack_input,
            samples=attack_input.samples,
            model=model,
            flags_obj=flags_obj,
        )

    def _resolve_t_results_from_samples(
        self,
        attack_input: AttackInput,
        samples: Any,
        model: torch.nn.Module,
        flags_obj: SimpleNamespace,
    ) -> Dict[str, torch.Tensor]:
        loader = self._build_loader(
            samples=samples,
            batch_size=int(self.runtime_config["batch_size"]),
        )
        return _SECMI_UTILS.get_intermediate_results(
            model,
            flags_obj,
            loader,
            int(self.runtime_config["t_sec"]),
            int(self.runtime_config["m"]),
        )

    def _resolve_member_nonmember_partition(
        self,
        attack_input: AttackInput,
        prefer_benchmark_split: bool,
    ) -> Optional[Dict[str, Any]]:
        signals = attack_input.signals or {}
        if "member_samples" in signals and "nonmember_samples" in signals:
            model = self._load_model_from_config(
                model=attack_input.target_model,
                config=self.runtime_config,
                metadata=attack_input.metadata,
            )
            flags_obj = self._resolve_flags(self.runtime_config, attack_input.metadata)
            return {
                "member_samples": signals["member_samples"],
                "nonmember_samples": signals["nonmember_samples"],
                "model": model,
                "flags_obj": flags_obj,
            }

        if attack_input.samples is None and prefer_benchmark_split:
            data_root = self.runtime_config.get("data_root") or attack_input.metadata.get("data_root")
            dataset = self.runtime_config.get("dataset") or attack_input.metadata.get("dataset")
            if data_root and dataset:
                model = self._load_model_from_config(
                    model=attack_input.target_model,
                    config=self.runtime_config,
                    metadata=attack_input.metadata,
                )
                flags_obj = self._resolve_flags(self.runtime_config, attack_input.metadata)
                _, _, member_loader, nonmember_loader = _SECMI_UTILS.load_member_data(
                    dataset_root=data_root,
                    dataset_name=dataset,
                    batch_size=int(self.runtime_config["batch_size"]),
                    shuffle=False,
                    randaugment=False,
                )
                return {
                    "member_samples": member_loader,
                    "nonmember_samples": nonmember_loader,
                    "model": model,
                    "flags_obj": flags_obj,
                }
        return None

    def _load_model_from_config(
        self,
        model: Optional[torch.nn.Module],
        config: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> torch.nn.Module:
        if model is not None:
            return model.to(self.device)

        model_dir = config.get("model_dir") or metadata.get("model_dir")
        ckpt_name = config.get("ckpt_name") or metadata.get("ckpt_name")
        if not model_dir or not ckpt_name:
            raise ValueError(
                "SecMIAAttack requires either `target_model` or both `model_dir` and `ckpt_name`."
            )

        flag_path = config.get("flag_path") or metadata.get("flag_path") or os.path.join(model_dir, "flagfile.txt")
        flags_obj = self._resolve_flags(config, metadata, flag_path=flag_path)
        ckpt_path = os.path.join(model_dir, ckpt_name)
        model_type = str(config.get("model_type") or metadata.get("model_type") or self.model_type).lower()

        if model_type == "ddpm":
            return _SECMI_UTILS.get_model(ckpt_path, flags_obj, WA=True).to(self.device)
        if model_type == "smcd":
            return _SECMI_UTILS.get_model_selective(ckpt_path, flags_obj).to(self.device)
        raise ValueError(f"Unsupported SecMIA model_type: {model_type}")

    def _resolve_flags(
        self,
        config: Dict[str, Any],
        metadata: Dict[str, Any],
        flag_path: Optional[str] = None,
    ) -> SimpleNamespace:
        if "flags_obj" in config:
            return config["flags_obj"]
        if "flags_obj" in metadata:
            return metadata["flags_obj"]

        flag_path = flag_path or config.get("flag_path") or metadata.get("flag_path")
        overrides = {
            "dataset": config.get("dataset") or metadata.get("dataset"),
            "data_root": config.get("data_root") or metadata.get("data_root"),
            "model_type": config.get("model_type") or metadata.get("model_type"),
            "ckpt_name": config.get("ckpt_name") or metadata.get("ckpt_name"),
            "output_save_dir": config.get("output_save_dir") or metadata.get("output_save_dir"),
            "t_sec": config.get("t_sec"),
            "m": config.get("m"),
            "k": config.get("k"),
            "num_t_groups": config.get("num_t_groups"),
            "sparsity": config.get("sparsity"),
            "T": config.get("T"),
            "beta_1": config.get("beta_1"),
            "beta_T": config.get("beta_T"),
            "ch": config.get("ch"),
            "ch_mult": config.get("ch_mult"),
            "attn": config.get("attn"),
            "num_res_blocks": config.get("num_res_blocks"),
            "dropout": config.get("dropout"),
            "parallel": config.get("parallel"),
        }
        return _parse_flagfile(flag_path, overrides)

    def _build_loader(self, samples: Any, batch_size: int) -> DataLoader:
        if isinstance(samples, DataLoader):
            return samples
        if torch.is_tensor(samples):
            tensor = samples.detach().cpu()
        elif isinstance(samples, np.ndarray):
            tensor = torch.from_numpy(samples)
        else:
            tensor = torch.as_tensor(samples)
        dataset = TensorDataset(tensor.float())
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def _stat_scores_from_internal(self, diffusions: torch.Tensor, denoise: torch.Tensor) -> np.ndarray:
        diffusions = diffusions.float()
        denoise = denoise.float()
        if diffusions.ndim == 4:
            scores = ((diffusions - denoise) ** 2).flatten(1).sum(dim=-1)
        elif diffusions.ndim == 5:
            num_timestep = diffusions.size(0)
            delta = ((diffusions - denoise) ** 2).permute(1, 0, 2, 3, 4).reshape(
                -1,
                num_timestep * 3,
                diffusions.size(-2),
                diffusions.size(-1),
            )
            scores = delta.flatten(1).sum(dim=-1)
        else:
            raise ValueError(f"Unsupported SecMIA internal tensor shape: {tuple(diffusions.shape)}")
        return (-scores).detach().cpu().numpy().astype(np.float32)

    def _nns_features_from_internal(self, diffusions: torch.Tensor, denoise: torch.Tensor) -> torch.Tensor:
        diffusions = diffusions.float()
        denoise = denoise.float()
        if diffusions.ndim == 4:
            return (diffusions - denoise).abs().float()
        if diffusions.ndim == 5:
            num_timestep = diffusions.size(0)
            return ((diffusions - denoise).abs() ** 2).permute(1, 0, 2, 3, 4).reshape(
                -1,
                num_timestep * 3,
                diffusions.size(-2),
                diffusions.size(-1),
            )
        raise ValueError(f"Unsupported SecMIA internal tensor shape: {tuple(diffusions.shape)}")


__all__ = ["AttackInput", "AttackOutput", "EvaluationResult", "SecMIAAttack"]
