"""ADF: adaptive selective diffusion, a training-time MIA defense for DDPMs."""
from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from Defense.adf_diffusion import SelectiveDenoiser, SelectiveGaussianDiffusionTrainer, generate_t_to_group
from Defense.base import BaseDefense, DefenseEvaluationResult, DefenseInput, DefenseOutput


class _GroupedImages(Dataset):
    def __init__(self, source: Any, k: int, seed: int):
        self.source, self.k = source, k
        self.groups = torch.randint(k, (len(source),), generator=torch.Generator().manual_seed(seed))

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int):
        item = self.source[index]
        image = item[0] if isinstance(item, (tuple, list)) else item
        return image, self.groups[index]


class ADFDefense(BaseDefense):
    """Training-time ADF defense; the resulting inference model uses ``model_type='smcd'``.

    ``model_factory`` must construct the base UNet.  ``train_data`` may be a
    Dataset or DataLoader whose first item is an image tensor.  The method first
    estimates an adaptive group/timestep mask, then optimizes the masked grouped
    diffusion objective.
    """
    name = "adf"
    defense_family = "adaptive_selective_diffusion"
    defense_mode = "training_time"
    supported_model_types = ["diffusion", "smcd"]
    required_input_keys = ["model_factory", "train_data"]
    optional_input_keys = ["auxiliary_data", "eval_config", "metadata"]

    def __init__(self, *, batch_size: int = 128, diffusion_steps: int = 1000, training_steps: int = 1000, k: int = 3,
                 num_t_groups: int = 5, sparsity: float = 0.3, stage1_steps: int = 1,
                 learning_rate: float = 2e-4, beta_1: float = 1e-4, beta_T: float = 0.02,
                 grad_clip: float = 1.0, ema_decay: float = 0.9999, seed: int = 0,
                 save_dir: str = "Defense/artifacts/adf", checkpoint_name: str = "adf_smcd.pt",
                 device: Optional[str] = None) -> None:
        self.defaults = dict(batch_size=batch_size, diffusion_steps=diffusion_steps, training_steps=training_steps, k=k,
            num_t_groups=num_t_groups, sparsity=sparsity, stage1_steps=stage1_steps,
            learning_rate=learning_rate, beta_1=beta_1, beta_T=beta_T, grad_clip=grad_clip,
            ema_decay=ema_decay, seed=seed, save_dir=save_dir, checkpoint_name=checkpoint_name)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.defended_model: Optional[SelectiveDenoiser] = None
        self.checkpoint_path: Optional[Path] = None
        self.flagfile_path: Optional[Path] = None
        self.mask: Optional[torch.Tensor] = None
        self.train_seconds: Optional[float] = None
        self.config: Dict[str, Any] = {}

    def fit(self, defense_input: DefenseInput) -> "ADFDefense":
        if defense_input.model_factory is None or defense_input.train_data is None:
            raise ValueError("ADFDefense requires model_factory and train_data.")
        self.config = {**self.defaults, **defense_input.defense_config}
        c = self.config
        if bool(c.get("skip_training", False)):
            checkpoint_path = c.get("checkpoint_path") or defense_input.metadata.get("checkpoint_path")
            if not checkpoint_path:
                raise ValueError("ADF skip_training requires defense_config['checkpoint_path'].")
            self._load_checkpoint(Path(checkpoint_path))
            self.train_seconds = 0.0
            return self
        loader = self._loader(defense_input.train_data, int(c["batch_size"]), int(c["k"]), int(c["seed"]))
        t_to_group = generate_t_to_group(int(c["diffusion_steps"]), int(c["num_t_groups"]))
        trainer = SelectiveGaussianDiffusionTrainer(
            defense_input.model_factory, float(c["beta_1"]), float(c["beta_T"]), int(c["diffusion_steps"]),
            int(c["k"]), int(c["num_t_groups"]), t_to_group, float(c["sparsity"]),
        ).to(self.device)
        optimizer = torch.optim.Adam(trainer.parameters(), lr=float(c["learning_rate"]))
        start = time.perf_counter()
        self._learn_mask(trainer, loader, optimizer)
        ema_trainer = copy.deepcopy(trainer).eval()
        iterator = self._cycle(loader)
        trainer.train()
        for _ in range(int(c["training_steps"])):
            images, groups = next(iterator)
            images = self._normalize(images.to(self.device))
            groups = groups.to(self.device)
            optimizer.zero_grad(set_to_none=True)
            loss = trainer(images, groups)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainer.parameters(), float(c["grad_clip"]))
            optimizer.step()
            with torch.no_grad():
                for source, target in zip(trainer.parameters(), ema_trainer.parameters()):
                    target.mul_(float(c["ema_decay"])).add_(source, alpha=1.0 - float(c["ema_decay"]))
        self.train_seconds = time.perf_counter() - start
        self.mask = trainer.M.detach().cpu()
        self.defended_model = ema_trainer.as_denoiser().to(self.device).eval()
        self._save(trainer, ema_trainer, t_to_group)
        return self

    def infer(self, defense_input: DefenseInput) -> DefenseOutput:
        if self.defended_model is None or self.checkpoint_path is None:
            raise RuntimeError("ADFDefense must be fitted before infer().")
        return DefenseOutput(
            defended_model=self.defended_model, protected_predictor=self.defended_model,
            artifacts={"checkpoint_path": str(self.checkpoint_path), "flagfile_path": str(self.flagfile_path),
                       "mask_path": str(self.checkpoint_path.with_name("learned_M.pt"))},
            intermediate_outputs={"mask": self.mask},
            metadata={"defense_name": self.name, "defense_family": self.defense_family,
                      "defense_mode": self.defense_mode, "model_type": "smcd",
                      "model_save_path": str(self.checkpoint_path)},
        )

    def evaluate(self, defense_output: DefenseOutput, defense_input: DefenseInput) -> DefenseEvaluationResult:
        c = {**self.config, **(defense_input.eval_config or {})}
        utility: Dict[str, float] = {}
        privacy: Dict[str, float] = {}
        extra: Dict[str, Any] = {}
        if bool(c.get("compute_fid", False)):
            from Attack.utils_secmia import secmia as secmia_utils
            flags = _flags(c, self.flagfile_path)
            utility["fid"] = float(secmia_utils.calculate_fid(
                defense_output.defended_model, flags, num_images=int(c.get("fid_num_images", 100)),
                batch_size=c.get("fid_batch_size"), fid_cache=c.get("fid_cache"), device=str(self.device)))
        attack_results = self._run_attacks(defense_output.defended_model, c)
        extra["attack_results"] = attack_results
        for name, result in attack_results.items():
            if isinstance(result, dict) and "auroc" in result:
                prefix = name + "_"
                privacy[prefix + "auroc"] = result["auroc"]
                privacy[prefix + "accuracy"] = result["accuracy"]
                privacy[prefix + "tpr_at_1pct_fpr"] = result["tpr_at_1pct_fpr"]
                privacy[prefix + "tpr_at_0_1pct_fpr"] = result["tpr_at_0_1pct_fpr"]
        return DefenseEvaluationResult(utility_metrics=utility or None, privacy_metrics=privacy or None,
            efficiency_metrics={"train_time": float(self.train_seconds or 0.0)}, extra_metrics=extra)

    def _learn_mask(self, trainer: SelectiveGaussianDiffusionTrainer, loader: DataLoader,
                    optimizer: torch.optim.Optimizer) -> None:
        """Stage 1 from the reference: retain groups with the largest loss decrease."""
        c, iterator = self.config, self._cycle(loader)
        mask = trainer.M.detach().clone()
        drop_count = round(float(c["sparsity"]) * int(c["k"]))
        for timestep in range(int(c["diffusion_steps"])):
            reductions = torch.zeros(int(c["k"]), device=self.device)
            for group in range(int(c["k"])):
                images, groups = next(iterator)
                images, groups = self._normalize(images.to(self.device)), groups.to(self.device)
                selected = groups == group
                if not selected.any():
                    continue
                before = trainer.loss_at_timestep(images[selected], groups[selected], timestep)
                for _ in range(int(c["stage1_steps"])):
                    optimizer.zero_grad(set_to_none=True)
                    warmup_loss = trainer.loss_at_timestep(images[selected], groups[selected], timestep)
                    warmup_loss.backward(); optimizer.step()
                after = trainer.loss_at_timestep(images[selected], groups[selected], timestep)
                reductions[group] = (before - after).detach()
            mask[:, timestep] = 1
            if drop_count:
                mask[torch.argsort(reductions)[:drop_count], timestep] = 0
        trainer.M.copy_(mask)

    def _save(self, trainer: nn.Module, ema_trainer: nn.Module, t_to_group: list[int]) -> None:
        output_dir = Path(self.config["save_dir"]); output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = output_dir / self.config["checkpoint_name"]
        torch.save({"trainer": trainer.state_dict(), "ema_model": ema_trainer.state_dict(), "step": int(self.config["training_steps"]) - 1}, self.checkpoint_path)
        torch.save(self.mask, output_dir / "learned_M.pt")
        self.flagfile_path = output_dir / "flagfile.txt"
        self.flagfile_path.write_text("\n".join(f"--{key}={value}" for key, value in {
            "T": self.config["diffusion_steps"], "k": self.config["k"], "num_t_groups": self.config["num_t_groups"],
            "sparsity": self.config["sparsity"], "beta_1": self.config["beta_1"], "beta_T": self.config["beta_T"],
            "model_type": "smcd", "img_size": self.config.get("img_size", 32)}.items()), encoding="utf-8")

    def _load_checkpoint(self, checkpoint_path: Path) -> None:
        """Restore a previously saved ADF/SMCD checkpoint without retraining."""
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"ADF checkpoint was not found: {checkpoint_path}")
        from types import SimpleNamespace
        from Attack.utils_secmia.secmia import get_model_selective
        flags = SimpleNamespace(
            T=int(self.config["diffusion_steps"]), ch=int(self.config.get("ch", 128)),
            ch_mult=self.config.get("ch_mult", [1, 2, 2, 2]), attn=self.config.get("attn", [1]),
            num_res_blocks=int(self.config.get("num_res_blocks", 2)),
            dropout=float(self.config.get("dropout", 0.1)),
            num_t_groups=int(self.config["num_t_groups"]),
        )
        self.defended_model = get_model_selective(str(checkpoint_path), flags).to(self.device).eval()
        self.checkpoint_path = checkpoint_path
        self.flagfile_path = checkpoint_path.with_name("flagfile.txt")
        self.mask = getattr(self.defended_model, "M", None)

    def _run_attacks(self, model: nn.Module, config: Dict[str, Any]) -> Dict[str, Any]:
        # Attack execution is explicit: GSAMIA needs an independently trained shadow model.
        if not config.get("run_attacks", False): return {}
        from Attack.secmia import AttackInput as SecInput, SecMIAAttack
        data_root, dataset = config["data_root"], config.get("dataset", "CIFAR10")
        from Attack.utils_secmia.mia_evals.dataset_utils import load_member_data
        members, nonmembers, member_loader, nonmember_loader = load_member_data(
            data_root, dataset, batch_size=int(config.get("attack_batch_size", 128)))
        labels = torch.cat([torch.ones(len(members), dtype=torch.long), torch.zeros(len(nonmembers), dtype=torch.long)]).numpy()
        common = {"data_root": data_root, "dataset": dataset, "model_type": "smcd", "t_sec": int(config.get("t_sec", 100)), "m": int(config.get("m", 10)), "batch_size": int(config.get("attack_batch_size", 128))}
        results: Dict[str, Any] = {}
        sec = SecMIAAttack(model_type="smcd", t_sec=common["t_sec"], m=common["m"], batch_size=common["batch_size"])
        output = sec.run(SecInput(target_model=model, samples=None, membership_labels=labels, config=common))
        results["secmia"] = _attack_metrics(output.evaluation)
        # caller must supply an independently trained SMCD shadow model and its member/nonmember samples.
        if config.get("gsamia_shadow_model") is not None:
            from Attack.gsamia import AttackInput as GsaInput, GSAMIAAttack
            shadow = config["gsamia_shadow_model"]
            gsa_batch_size = int(config.get("gsamia_batch_size", 1))
            gsa_member_loader = torch.utils.data.DataLoader(members, batch_size=gsa_batch_size, shuffle=False)
            gsa_nonmember_loader = torch.utils.data.DataLoader(nonmembers, batch_size=gsa_batch_size, shuffle=False)
            gsa = GSAMIAAttack(model_type="smcd", batch_size=gsa_batch_size,
                sampling_frequency=int(config.get("gsamia_sampling_frequency", 5)),
                timestep_chunk_size=int(config.get("gsamia_timestep_chunk_size", 1)), use_cached_features=False)
            gsa_labels = torch.cat([torch.ones(len(gsa_member_loader), dtype=torch.long),
                                    torch.zeros(len(gsa_nonmember_loader), dtype=torch.long)]).numpy()
            output = gsa.run(GsaInput(target_model=model, samples=None, membership_labels=gsa_labels,
                shadow_data={"shadow_model": shadow, "member_samples": gsa_member_loader, "nonmember_samples": gsa_nonmember_loader}, config={**common, "batch_size": gsa_batch_size, "output_name": str(Path(self.config["save_dir"]) / "gsamia_features"), "use_cached_features": False,
                "sampling_frequency": int(config.get("gsamia_sampling_frequency", 5)),
                "timestep_chunk_size": int(config.get("gsamia_timestep_chunk_size", 1))}))
            results["gsamia"] = _attack_metrics(output.evaluation)
        else:
            results["gsamia"] = {"status": "skipped", "reason": "eval_config.gsamia_shadow_model is required"}
        return results

    @staticmethod
    def _loader(data: Any, batch_size: int, k: int, seed: int) -> DataLoader:
        source = data.dataset if isinstance(data, DataLoader) else data
        return DataLoader(_GroupedImages(source, k, seed), batch_size=batch_size, shuffle=True, drop_last=True)

    @staticmethod
    def _cycle(loader: DataLoader) -> Iterator[Any]:
        while True: yield from loader

    @staticmethod
    def _normalize(images: torch.Tensor) -> torch.Tensor:
        return images * 2.0 - 1.0 if images.detach().amin() >= 0 else images


def _flags(config: Dict[str, Any], flagfile: Optional[Path]):
    from types import SimpleNamespace
    return SimpleNamespace(T=int(config["diffusion_steps"]), beta_1=float(config["beta_1"]), beta_T=float(config["beta_T"]),
        img_size=int(config.get("img_size", 32)), mean_type="epsilon", var_type="fixedlarge", batch_size=int(config["batch_size"]))


def _attack_metrics(evaluation: Any) -> Dict[str, float]:
    return {"auroc": float(evaluation.auroc), "accuracy": float(evaluation.accuracy),
        "tpr_at_1pct_fpr": float(evaluation.tpr_at_fpr["1%"]), "tpr_at_0_1pct_fpr": float(evaluation.tpr_at_fpr["0.1%"])}
