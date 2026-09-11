from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from diffusers import DDPMScheduler
from sklearn import preprocessing


REPO_ROOT = Path(__file__).resolve().parents[1]
UTIL_DIR = Path(__file__).resolve().parent / "utils_gsamia"
for path in (REPO_ROOT, UTIL_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from Attack.shadow_based import AttackInput, AttackOutput, BaseAttack


_GSAMIA_UTILS = importlib.import_module("Attack.utils_gsamia.gsamia")


def _to_numpy_2d(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        array = value
    elif torch.is_tensor(value):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix, got shape {array.shape}.")
    return array.astype(np.float32)


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
        "model_type": "ddpm",
        "dataset": "cifar10",
        "data_root": "./datasets",
        "attack_method": 1,
        "sampling_frequency": 10,
        "prediction_type": "epsilon",
        "output_name": "./gradient_mia/interface",
        "use_cached_features": True,
    }


def _coerce_flag_value(key: str, raw: str) -> Any:
    bool_keys = {"parallel"}
    int_keys = {
        "T",
        "ch",
        "num_res_blocks",
        "k",
        "num_t_groups",
        "attack_method",
        "sampling_frequency",
    }
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


class GSAMIAAttack(BaseAttack):
    """
    Unified wrapper for the gradient-signal attack (GSAMIA).

    Supported training input (`attack_input.shadow_data`)
    ----------------------------------------------------
    1. Explicit shadow features:
       {
           "member_gradient_features": ndarray/Tensor,
           "nonmember_gradient_features": ndarray/Tensor,
       }

    2. Saved feature paths:
       {
           "shadow_model_member_path": ["...pt", ...] or "...pt",
           "shadow_model_non_member_path": ["...pt", ...] or "...pt",
       }

    3. Shadow model + shadow member/non-member samples:
       {
           "shadow_model": nn.Module,  # optional if model_dir/ckpt_name is given
           "member_samples": ...,
           "nonmember_samples": ...,
       }

    Supported query input
    ---------------------
    - signals["gradient_features"]
    - signals["member_gradient_features"] + signals["nonmember_gradient_features"]
    - saved feature paths in `signals`
    - raw target samples + `target_model`
    - benchmark mode: no `samples`, but `config`/`metadata` provides `data_root`, `dataset`
    """

    def __init__(
        self,
        attack_method: int = 1,
        sampling_frequency: int = 5,
        prediction_type: str = "epsilon",
        batch_size: int = 16,
        model_type: str = "ddpm",
        output_name: str = "./gradient_mia/interface",
        xgb_n_estimators: int = 200,
        use_cached_features: bool = True,
        timestep_chunk_size: int = 1,
        compute_fid: bool = False,
        fid_num_images: int = 100,
        fid_batch_size: Optional[int] = None,
        fid_cache: Optional[str] = None,
        fid_use_torch: bool = False,
        device: Optional[str] = None,
    ) -> None:
        self.attack_method = attack_method
        self.sampling_frequency = sampling_frequency
        self.prediction_type = prediction_type
        self.batch_size = batch_size
        self.model_type = model_type
        self.output_name = output_name
        self.xgb_n_estimators = xgb_n_estimators
        self.use_cached_features = use_cached_features
        self.timestep_chunk_size = timestep_chunk_size
        self.compute_fid = compute_fid
        self.fid_num_images = fid_num_images
        self.fid_batch_size = fid_batch_size
        self.fid_cache = fid_cache
        self.fid_use_torch = fid_use_torch
        self.device = device if device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu")

        self.runtime_config: Dict[str, Any] = {}
        self.attack_model: Optional[Any] = None
        self.is_fitted = False

    def fit(self, attack_input: AttackInput) -> "GSAMIAAttack":
        self.runtime_config = self._merge_config(attack_input.config)
        shadow_member, shadow_nonmember = self._resolve_shadow_features(attack_input)
        shadow_x, shadow_y = self._preprocess_partition(shadow_member, shadow_nonmember)

        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError("GSAMIAAttack.fit() requires the `xgboost` package.") from exc

        xgb = XGBClassifier(n_estimators=int(self.runtime_config["xgb_n_estimators"]))
        xgb.fit(shadow_x, shadow_y)
        self.attack_model = xgb
        self.is_fitted = True
        return self

    def infer(self, attack_input: AttackInput) -> AttackOutput:
        if not self.is_fitted or self.attack_model is None:
            raise RuntimeError("GSAMIAAttack must be fitted before infer().")

        self.runtime_config = self._merge_config(attack_input.config)
        features, explicit_partition, extra = self._resolve_query_features(attack_input)
        if explicit_partition is not None:
            query_x, _ = self._preprocess_partition(
                explicit_partition[0],
                explicit_partition[1],
            )
        else:
            query_x = preprocessing.scale(features).astype(np.float32)

        scores = self.attack_model.predict_proba(query_x)[:, 1].astype(np.float32)
        preds = (scores >= 0.5).astype(np.int64)

        return AttackOutput(
            membership_scores=scores,
            membership_preds=preds,
            intermediate_outputs=extra,
            metadata={
                "attack_name": "gsamia",
                "attack_method": int(self.runtime_config["attack_method"]),
                "sampling_frequency": int(self.runtime_config["sampling_frequency"]),
                "score_direction": "higher_is_member",
                "pred_threshold": 0.5,
                **self._fid_metadata(attack_input),
            },
        )

    def _merge_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        merged = {
            "attack_method": self.attack_method,
            "sampling_frequency": self.sampling_frequency,
            "prediction_type": self.prediction_type,
            "batch_size": self.batch_size,
            "model_type": self.model_type,
            "output_name": self.output_name,
            "xgb_n_estimators": self.xgb_n_estimators,
            "use_cached_features": self.use_cached_features,
            "timestep_chunk_size": self.timestep_chunk_size,
            "compute_fid": self.compute_fid,
            "fid_num_images": self.fid_num_images,
            "fid_batch_size": self.fid_batch_size,
            "fid_cache": self.fid_cache,
            "fid_use_torch": self.fid_use_torch,
        }
        merged.update(config)
        return merged

    def _fid_metadata(self, attack_input: AttackInput) -> Dict[str, float]:
        """Compute optional generation-quality FID for the attacked DDPM."""
        if not bool(self.runtime_config.get("compute_fid", False)):
            return {}
        if str(self.runtime_config.get("model_type", self.model_type)).lower() not in {"ddpm", "smcd"}:
            raise ValueError("FID evaluation supports model_type='ddpm' and 'smcd' only.")
        model = self._load_model_from_config(
            attack_input.target_model, self.runtime_config, attack_input.metadata,
        )
        flags_obj = self._resolve_flags(self.runtime_config, attack_input.metadata)
        fid = _GSAMIA_UTILS.calculate_fid(
            model, flags_obj,
            num_images=int(self.runtime_config["fid_num_images"]),
            batch_size=self.runtime_config.get("fid_batch_size"),
            fid_cache=self.runtime_config.get("fid_cache"),
            use_torch=bool(self.runtime_config.get("fid_use_torch", False)),
            device=self.device,
            timestep_chunk_size=int(self.runtime_config["timestep_chunk_size"]),
        )
        return {"fid": float(fid)}

    def _resolve_shadow_features(self, attack_input: AttackInput) -> Tuple[np.ndarray, np.ndarray]:
        shadow_data = attack_input.shadow_data
        if shadow_data is None:
            raise ValueError("GSAMIAAttack.fit() requires shadow_data.")
        partition = self._resolve_feature_partition_payload(
            payload=shadow_data,
            model=shadow_data.get("shadow_model"),
            fallback_metadata={**attack_input.metadata, **shadow_data},
        )
        if partition is None:
            raise ValueError(
                "Unable to resolve GSAMIA shadow features. Provide explicit shadow features, "
                "saved feature paths, or raw shadow samples."
            )
        return partition

    def _resolve_query_features(
        self,
        attack_input: AttackInput,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, np.ndarray]], Dict[str, Any]]:
        signals = attack_input.signals or {}
        explicit_partition = self._resolve_feature_partition_payload(
            payload=signals,
            model=attack_input.target_model,
            fallback_metadata=attack_input.metadata,
        )
        if explicit_partition is not None:
            member_features, nonmember_features = explicit_partition
            combined = np.concatenate([member_features, nonmember_features], axis=0)
            return combined, explicit_partition, {
                "member_gradient_features": member_features,
                "nonmember_gradient_features": nonmember_features,
            }

        if "gradient_features" in signals:
            features = _to_numpy_2d(signals["gradient_features"])
            return features, None, {"gradient_features": features}

        if attack_input.samples is not None:
            features = self._extract_query_features_from_samples(
                attack_input=attack_input,
                samples=attack_input.samples,
                model=attack_input.target_model,
                membership="query",
                metadata=attack_input.metadata,
            )
            return features, None, {"gradient_features": features}

        benchmark_partition = self._resolve_feature_partition_payload(
            payload={},
            model=attack_input.target_model,
            fallback_metadata=attack_input.metadata,
            allow_benchmark_split=True,
        )
        if benchmark_partition is not None:
            member_features, nonmember_features = benchmark_partition
            combined = np.concatenate([member_features, nonmember_features], axis=0)
            return combined, benchmark_partition, {
                "member_gradient_features": member_features,
                "nonmember_gradient_features": nonmember_features,
            }

        raise ValueError(
            "Unable to resolve GSAMIA query features. Provide signals, samples, or benchmark dataset settings."
        )

    def _resolve_feature_partition_payload(
        self,
        payload: Dict[str, Any],
        model: Optional[torch.nn.Module],
        fallback_metadata: Dict[str, Any],
        allow_benchmark_split: bool = False,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if "member_gradient_features" in payload and "nonmember_gradient_features" in payload:
            return (
                _to_numpy_2d(payload["member_gradient_features"]),
                _to_numpy_2d(payload["nonmember_gradient_features"]),
            )

        member_path_key = None
        nonmember_path_key = None
        if "shadow_model_member_path" in payload and "shadow_model_non_member_path" in payload:
            member_path_key = "shadow_model_member_path"
            nonmember_path_key = "shadow_model_non_member_path"
        elif "target_model_member_path" in payload and "target_model_non_member_path" in payload:
            member_path_key = "target_model_member_path"
            nonmember_path_key = "target_model_non_member_path"

        if member_path_key is not None and nonmember_path_key is not None:
            return self._load_feature_paths(
                payload[member_path_key],
                payload[nonmember_path_key],
            )

        if "member_samples" in payload and "nonmember_samples" in payload:
            member_features = self._extract_query_features_from_samples(
                attack_input=None,
                samples=payload["member_samples"],
                model=model,
                membership="mem",
                metadata={**fallback_metadata, **payload},
            )
            nonmember_features = self._extract_query_features_from_samples(
                attack_input=None,
                samples=payload["nonmember_samples"],
                model=model,
                membership="nonmem",
                metadata={**fallback_metadata, **payload},
            )
            return member_features, nonmember_features

        if allow_benchmark_split:
            data_root = self.runtime_config.get("data_root") or fallback_metadata.get("data_root")
            dataset = self.runtime_config.get("dataset") or fallback_metadata.get("dataset")
            if data_root and dataset:
                loaded_model = self._load_model_from_config(
                    model=model,
                    config=self.runtime_config,
                    metadata=fallback_metadata,
                )
                flags_obj = self._resolve_flags(self.runtime_config, fallback_metadata)
                member_loader, nonmember_loader = _GSAMIA_UTILS.get_dataset(
                    dataset_root=data_root,
                    dataset=dataset,
                    batch_size=int(self.runtime_config["batch_size"]),
                )
                member_features = self._extract_features_with_loader(
                    model=loaded_model,
                    dataloader=member_loader,
                    membership="mem",
                    metadata=fallback_metadata,
                    flags_obj=flags_obj,
                )
                nonmember_features = self._extract_features_with_loader(
                    model=loaded_model,
                    dataloader=nonmember_loader,
                    membership="nonmem",
                    metadata=fallback_metadata,
                    flags_obj=flags_obj,
                )
                return member_features, nonmember_features

        return None

    def _extract_query_features_from_samples(
        self,
        attack_input: Optional[AttackInput],
        samples: Any,
        model: Optional[torch.nn.Module],
        membership: str,
        metadata: Dict[str, Any],
    ) -> np.ndarray:
        loaded_model = self._load_model_from_config(
            model=model,
            config=self.runtime_config,
            metadata=metadata,
        )
        flags_obj = self._resolve_flags(self.runtime_config, metadata)
        dataloader = self._build_loader(samples, int(self.runtime_config["batch_size"]))
        return self._extract_features_with_loader(
            model=loaded_model,
            dataloader=dataloader,
            membership=membership,
            metadata=metadata,
            flags_obj=flags_obj,
        )

    def _extract_features_with_loader(
        self,
        model: torch.nn.Module,
        dataloader: Any,
        membership: str,
        metadata: Dict[str, Any],
        flags_obj: SimpleNamespace,
    ) -> np.ndarray:
        output_name = (
            metadata.get("output_name")
            or self.runtime_config.get("output_name")
            or self.output_name
        )
        ckpt_name = metadata.get("ckpt_name") or self.runtime_config.get("ckpt_name") or "model"
        model_name = Path(str(ckpt_name)).stem
        cache_path = Path(output_name) / membership / f"gradient_{model_name}.pt"
        use_cached_features = bool(self.runtime_config.get("use_cached_features", self.use_cached_features))

        if use_cached_features and cache_path.exists():
            cached = torch.load(cache_path, map_location=self.device)
            return _to_numpy_2d(cached)

        _GSAMIA_UTILS.args = SimpleNamespace(output_name=output_name)
        scheduler = DDPMScheduler(
            num_train_timesteps=int(flags_obj.T),
            beta_start=float(flags_obj.beta_1),
            beta_end=float(flags_obj.beta_T),
            prediction_type=str(self.runtime_config["prediction_type"]),
        )
        features = _GSAMIA_UTILS.extract_attack_features(
            model=model,
            model_name=model_name,
            dataloader=dataloader,
            noise_scheduler=scheduler,
            attack_method=int(self.runtime_config["attack_method"]),
            prediction_type=str(self.runtime_config["prediction_type"]),
            sampling_frequency=int(self.runtime_config["sampling_frequency"]),
            ddpm_num_steps=int(flags_obj.T),
            membership=membership,
            device=self.device,
        )
        return _to_numpy_2d(features)

    def _load_feature_paths(
        self,
        member_paths: Sequence[str] | str,
        nonmember_paths: Sequence[str] | str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(member_paths, str):
            member_paths = [member_paths]
        if isinstance(nonmember_paths, str):
            nonmember_paths = [nonmember_paths]

        member_tensors = []
        nonmember_tensors = []
        for member_path, nonmember_path in zip(member_paths, nonmember_paths):
            member_tensors.append(torch.load(member_path, map_location=self.device))
            nonmember_tensors.append(torch.load(nonmember_path, map_location=self.device))

        member = torch.cat(member_tensors, dim=0)
        nonmember = torch.cat(nonmember_tensors, dim=0)
        return _to_numpy_2d(member), _to_numpy_2d(nonmember)

    def _preprocess_partition(
        self,
        member_features: np.ndarray,
        nonmember_features: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        member_features = _to_numpy_2d(member_features)
        nonmember_features = _to_numpy_2d(nonmember_features)
        member_features = member_features[: nonmember_features.shape[0]]
        member_labels = np.ones(member_features.shape[0], dtype=np.int64)
        nonmember_labels = np.zeros(nonmember_features.shape[0], dtype=np.int64)
        x = np.vstack((member_features, nonmember_features)).astype(np.float32)
        y = np.concatenate((member_labels, nonmember_labels), axis=0)
        x = preprocessing.scale(x).astype(np.float32)
        return x, y

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
                "GSAMIAAttack requires either a loaded model or both `model_dir` and `ckpt_name`."
            )

        flag_path = config.get("flag_path") or metadata.get("flag_path") or os.path.join(model_dir, "flagfile.txt")
        flags_obj = self._resolve_flags(config, metadata, flag_path=flag_path)
        ckpt_path = os.path.join(model_dir, ckpt_name)
        model_type = str(config.get("model_type") or metadata.get("model_type") or self.model_type).lower()

        if model_type == "ddpm":
            return _GSAMIA_UTILS.get_model(ckpt_path, flags_obj, WA=True).to(self.device)
        if model_type == "etdm":
            return _GSAMIA_UTILS.get_model_grouped(ckpt_path, flags_obj).to(self.device)
        if model_type == "smcd":
            return _GSAMIA_UTILS.get_model_selective(ckpt_path, flags_obj).to(self.device)
        raise ValueError(f"Unsupported GSAMIA model_type: {model_type}")

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
            "model_type": config.get("model_type") or metadata.get("model_type"),
            "dataset": config.get("dataset") or metadata.get("dataset"),
            "data_root": config.get("data_root") or metadata.get("data_root"),
            "attack_method": config.get("attack_method"),
            "sampling_frequency": config.get("sampling_frequency"),
            "prediction_type": config.get("prediction_type"),
            "output_name": config.get("output_name") or metadata.get("output_name"),
            "use_cached_features": config.get("use_cached_features"),
            "T": config.get("T"),
            "beta_1": config.get("beta_1"),
            "beta_T": config.get("beta_T"),
            "ch": config.get("ch"),
            "ch_mult": config.get("ch_mult"),
            "attn": config.get("attn"),
            "num_res_blocks": config.get("num_res_blocks"),
            "dropout": config.get("dropout"),
            "k": config.get("k"),
            "num_t_groups": config.get("num_t_groups"),
            "sparsity": config.get("sparsity"),
            "parallel": config.get("parallel"),
        }
        return _parse_flagfile(flag_path, overrides)

    def _build_loader(self, samples: Any, batch_size: int) -> Any:
        if isinstance(samples, torch.utils.data.DataLoader):
            return samples
        tensor = torch.as_tensor(samples).float()
        dataset = torch.utils.data.TensorDataset(tensor)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)


__all__ = ["AttackInput", "AttackOutput", "GSAMIAAttack"]
