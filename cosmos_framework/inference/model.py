# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import contextlib
import json
from pathlib import Path
from typing import Any

import attrs
import hydra
import omegaconf
import torch
import torch.distributed.checkpoint as dcp
import transformers
from torch.distributed.checkpoint.filesystem import FileSystemReader
from torch.distributed.checkpoint.hf_storage import (
    CUSTOM_METADATA_KEY,
    SAVED_OFFSETS_KEY,
    HuggingFaceStorageReader,
    _HFStorageInfo,
)
from torch.distributed.checkpoint.metadata import (
    STORAGE_TYPES,
    ChunkStorageMetadata,
    Metadata,
    StorageMeta,
    TensorProperties,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.planner import MetadataIndex
from torch.distributed.checkpoint.state_dict import get_model_state_dict
from typing_extensions import TYPE_CHECKING, assert_never

from cosmos_framework.configs.base.defaults.compile import CompileConfig
from cosmos_framework.configs.base.defaults.parallelism import ParallelismConfig
from cosmos_framework.configs.base.defaults.quantization import QuantizationConfig
from cosmos_framework.inference.common.args import CheckpointType
from cosmos_framework.inference.common.checkpoints import register_checkpoints
from cosmos_framework.inference.common.config import structure_config, undo_config_dict_replacements, unstructure_config
from cosmos_framework.inference.common.public_model_config import (
    build_public_model_config,
    model_config_uses_public_aliases,
    restore_model_config_from_public_model_config,
)
from cosmos_framework.utils import misc
from cosmos_framework.utils.flags import SMOKE
from cosmos_framework.utils.generator.quantization import (
    apply_modelopt_fp8_checkpoint_inplace,
    is_modelopt_fp8_checkpoint,
    plan_modelopt_fp8_targets,
)

if TYPE_CHECKING:
    from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel


# Resolve to the release-tree root so relative-path checkpoint config entries
# (e.g. `cosmos_framework/model/generator/vlm/qwen3_vl/configs/Qwen3-VL-8B-Instruct.json`) load
# correctly under contextlib.chdir(_ROOT_DIR) in __init__. In the original
# cosmos3 release the cosmos3 package lives at the tree root, so parents[1] is
# the release root. In the cosmos_training release the package lives at
# cosmos_framework/inference/, so the release root is one level higher (parents[2]).
try:
    import cosmos_framework.model.generator  # noqa: F401

    _ROOT_DIR = Path(__file__).parents[2].absolute()
except ImportError:
    _ROOT_DIR = Path(__file__).parents[1].absolute()


def _configure_inference_cudnn() -> None:
    """Match the cuDNN runtime used by native I4 inference.

    I4 inherits these flags from its trainer setup. Framework inference loads
    models directly, so configure them explicitly before constructing the model.
    """
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


from cosmos_framework.model.generator.mot.checkpoint_mapping import (
    _DIFFUSERS_DROP_KEY_RES as _DIFFUSERS_DROP_KEY_RES,
)
from cosmos_framework.model.generator.mot.checkpoint_mapping import (
    _DIFFUSERS_DROP_WEIGHT_PATH_RES as _DIFFUSERS_DROP_WEIGHT_PATH_RES,
)
from cosmos_framework.model.generator.mot.checkpoint_mapping import (
    _DIFFUSERS_KEY_MAPPING_RES as _DIFFUSERS_KEY_MAPPING_RES,
)
from cosmos_framework.model.generator.mot.checkpoint_mapping import (
    _DIFFUSERS_MODEL_INDEX as _DIFFUSERS_MODEL_INDEX,
)
from cosmos_framework.model.generator.mot.checkpoint_mapping import (
    _DIFFUSERS_NET_KEY_PREFIXES as _DIFFUSERS_NET_KEY_PREFIXES,
)
from cosmos_framework.model.generator.mot.checkpoint_mapping import (
    _DIFFUSERS_NET_KEYS as _DIFFUSERS_NET_KEYS,
)
from cosmos_framework.model.generator.mot.checkpoint_mapping import (
    _DIFFUSERS_ROOT_INDEX as _DIFFUSERS_ROOT_INDEX,
)
from cosmos_framework.model.generator.mot.checkpoint_mapping import (
    _apply_diffusers_key_mapping as _apply_diffusers_key_mapping,
)
from cosmos_framework.model.generator.mot.checkpoint_mapping import (
    _diffusers_files_to_keys as _diffusers_files_to_keys,
)
from cosmos_framework.model.generator.mot.checkpoint_mapping import (
    _diffusers_to_net_key as _diffusers_to_net_key,
)
from cosmos_framework.model.generator.mot.checkpoint_mapping import (
    _diffusers_weight_map as _diffusers_weight_map,
)
from cosmos_framework.model.generator.mot.checkpoint_mapping import (
    _is_diffusers_checkpoint as _is_diffusers_checkpoint,
)
from cosmos_framework.model.generator.mot.checkpoint_mapping import (
    _is_diffusers_model_weight_path as _is_diffusers_model_weight_path,
)
from cosmos_framework.model.generator.mot.checkpoint_mapping import (
    _is_loadable_diffusers_net_key as _is_loadable_diffusers_net_key,
)
from cosmos_framework.model.generator.mot.checkpoint_mapping import (
    _read_safetensors_index as _read_safetensors_index,
)
from cosmos_framework.model.generator.mot.checkpoint_mapping import (
    _should_drop_diffusers_key as _should_drop_diffusers_key,
)
from cosmos_framework.model.generator.mot.checkpoint_mapping import (
    _should_drop_diffusers_weight_path as _should_drop_diffusers_weight_path,
)

_LM_VISUAL_KEY_MARKER = "language_model.visual."


def _raise_on_missing_vision_keys(checkpoint_path: Path, state_dict: dict[str, Any]) -> None:
    """Curated error for root-safetensors checkpoints that lack the reasoner vision tower.

    Mirrors ``_DiffusersLoadPlanner``'s tolerant/curated missing-keys handling for
    the plain ``HuggingFaceStorageReader`` path: a model built with a visual tower
    (``include_visual=True``) cannot load from a checkpoint exported with
    ``--no-vit``; without this pre-check, ``dcp.load`` dies mid-load with a raw
    ``Missing key in checkpoint state_dict: model.net.language_model.visual...``.
    """
    wanted = sorted(key for key in state_dict if _LM_VISUAL_KEY_MARKER in key)
    if not wanted:
        return
    # Root index shares its filename with the diffusers layout ("model.safetensors.index.json").
    index_path = checkpoint_path / _DIFFUSERS_ROOT_INDEX
    if not index_path.exists():
        return
    if any(_LM_VISUAL_KEY_MARKER in key for key in _read_safetensors_index(index_path)):
        return
    raise ValueError(
        f"Checkpoint at {checkpoint_path} provides none of the {len(wanted)} reasoner vision-tower "
        f"tensor(s) the model config requires (first: {wanted[0]!r}). It was likely exported with "
        "--no-vit while its config still enables the visual tower. Fix: re-export with the default "
        "--vit (or use a full checkpoint); for generation-only use, re-export with --no-vit — new "
        "--no-vit exports write include_visual=false so the model loads without the tower."
    )


def _normalize_diffusers_target_key(name: str) -> str:
    return name.removeprefix("model.net.").replace("_orig_mod.", "").replace("_checkpoint_wrapped_module.", "")


def _validate_mixed_precision_load(
    quantization_config: QuantizationConfig, *, modelopt_checkpoint: bool, use_cuda_graphs: bool
) -> None:
    """Fail fast on unsupported mixed-precision-step combinations."""
    if not quantization_config.mixed_precision_enabled:
        return
    if not modelopt_checkpoint:
        raise ValueError(
            "mixed_precision_first_steps/last_steps require a ModelOpt FP8 checkpoint "
            "(hf_quant_config.json with quant_method='modelopt', quant_algo='FP8')."
        )
    if use_cuda_graphs:
        raise ValueError(
            "Mixed-precision diffusion steps are incompatible with use_cuda_graphs: "
            "per-step precision switching cannot be captured/replayed."
        )


class _MmapSafeReadMixin:
    """Materialize each safetensors slice into anonymous RAM before the H2D copy.

    ``HuggingFaceStorageReader._process_read_request`` copies tensors straight from the
    ``mmap``-backed safetensors slice onto the GPU, so the transfer handles a page fault
    per tensor. On Grace that dominates checkpoint load: a Cosmos3-Super reasoner load
    spends ~1200s here with zero disk I/O, the data already resident in page cache.

    The copy is applied conditionally -- see ``_materialize_enabled`` for why.
    """

    _materialize_cache = None

    @classmethod
    def _materialize_enabled(cls) -> bool:
        """Whether to stage tensors through anonymous host memory before the H2D copy.

        Defaults on for aarch64 and off elsewhere, overridable in either direction with
        COSMOS_MATERIALIZE_CHECKPOINT=1/0.

        The staging copy removes a slow file-backed mmap H2D path on Grace, but where that
        path is already fast the copy is pure overhead: on 8x A100 (Cosmos3-Super, 8 ranks)
        applying it unconditionally measured ~13.2% slower, because concurrent ranks contend
        for host memory bandwidth. The architecture is only a proxy for "is the mmap H2D path
        slow here", so it is used as a default rather than as a hard condition.
        """
        if cls._materialize_cache is None:
            import os
            import platform

            override = os.environ.get("COSMOS_MATERIALIZE_CHECKPOINT")
            if override is not None:
                cls._materialize_cache = override.strip().lower() not in (
                    "0",
                    "false",
                    "no",
                    "off",
                    "",
                )
            else:
                cls._materialize_cache = platform.machine().lower() in ("aarch64", "arm64")
        return cls._materialize_cache

    def _process_read_request(self, f, req, planner) -> None:  # noqa: D102
        slices = tuple(slice(offset, offset + length) for offset, length in zip(req.storage_offsets, req.lengths))
        tensor = f.get_slice(req.storage_index.fqn)[slices]
        target_tensor = planner.resolve_tensor(req).detach()

        if target_tensor.size() != tensor.size():
            raise AssertionError(f"req {req.storage_index} mismatch sizes {target_tensor.size()} vs {tensor.size()}")

        if target_tensor.is_cuda and self._materialize_enabled():
            # Materialise into anonymous host memory before the H2D copy.
            #
            # The base implementation copies straight from mmap-backed safetensors
            # storage into a CUDA tensor, so the transfer handles a page fault per
            # tensor. On Grace this dominates load time (Cosmos3-Super load: ~1200s
            # down to ~70-85s with this change, measured across two boots; zero disk
            # I/O either way -- the data is already in page cache).
            #
            # Pre-faulting in place is not sufficient: measured on a 4.61 GiB shard,
            # an mmap source still costs 9.00 ms/tensor after faults are pre-paid,
            # versus 1.61 ms/tensor from the heap. Pinning the staging buffer is not
            # necessary either -- with a heap source, pageable (1.61) and pinned
            # (1.69) are equivalent. A single clone() does both jobs: the memcpy
            # faults the pages sequentially and leaves the data resident and
            # anonymous. It is freed as soon as the H2D copy completes.
            tensor = tensor.contiguous().clone()

        target_tensor.copy_(tensor)
        planner.commit_tensor(req, target_tensor)


class _MmapSafeHuggingFaceStorageReader(_MmapSafeReadMixin, HuggingFaceStorageReader):
    """Plain HF safetensors reader with the mmap-H2D staging copy."""


class _DiffusersHuggingFaceStorageReader(_MmapSafeReadMixin, HuggingFaceStorageReader):
    """Hugging Face safetensors reader that follows diffusers' root weight map."""

    def __init__(self, checkpoint_path: Path) -> None:
        super().__init__(str(checkpoint_path))
        self.checkpoint_path = checkpoint_path
        self.files_to_keys = _diffusers_files_to_keys(_diffusers_weight_map(checkpoint_path))

    def read_metadata(self) -> Metadata:
        from safetensors import safe_open
        from safetensors.torch import _getdtype

        state_dict_metadata: dict[str, STORAGE_TYPES] = {}
        storage_data: dict[MetadataIndex, _HFStorageInfo] = {}

        for rel_path, diff_keys in sorted(self.files_to_keys.items()):
            shard_path = self.checkpoint_path / rel_path
            if not shard_path.exists():
                raise FileNotFoundError(f"Diffusers checkpoint shard not found: {shard_path}")

            with safe_open(str(shard_path), framework="pt") as f:
                shard_keys = set(f.keys())
                missing_keys = sorted(set(diff_keys) - shard_keys)
                if missing_keys:
                    raise KeyError(
                        f"Diffusers checkpoint shard {shard_path} is missing {len(missing_keys)} "
                        f"indexed tensor(s). First up to 10: {missing_keys[:10]}"
                    )

                extra_metadata = f.metadata()
                dcp_sharding_info: dict[str, Any] | None = None
                if extra_metadata and extra_metadata.get(CUSTOM_METADATA_KEY):
                    dcp_sharding_info = json.loads(extra_metadata[CUSTOM_METADATA_KEY])

                for diff_key in sorted(diff_keys):
                    tensor_slice = f.get_slice(diff_key)
                    shape = tensor_slice.get_shape()
                    dtype = _getdtype(tensor_slice.get_dtype())
                    offset = dcp_sharding_info[diff_key][SAVED_OFFSETS_KEY] if dcp_sharding_info else [0] * len(shape)
                    chunk = ChunkStorageMetadata(
                        offsets=torch.Size(offset),
                        sizes=torch.Size(shape),
                    )

                    if diff_key not in state_dict_metadata:
                        state_dict_metadata[diff_key] = TensorStorageMetadata(
                            properties=TensorProperties(dtype=dtype),
                            size=torch.Size(saved + start for saved, start in zip(shape, offset)),
                            chunks=[chunk],
                        )
                    else:
                        existing_metadata = state_dict_metadata[diff_key]
                        assert isinstance(existing_metadata, TensorStorageMetadata)
                        existing_metadata.chunks.append(chunk)
                        size = list(existing_metadata.size)
                        for i, dim_size in enumerate(size):
                            size[i] = max(dim_size, shape[i] + offset[i])
                        existing_metadata.size = torch.Size(size)

                    metadata_index = MetadataIndex(fqn=diff_key, offset=offset)
                    storage_data[metadata_index] = _HFStorageInfo(
                        relative_path=str(shard_path),
                        shape=torch.Size(shape),
                        dtype=dtype,
                    )

        metadata = Metadata(
            state_dict_metadata=state_dict_metadata,
            storage_data=storage_data,
        )
        storage_meta = metadata.storage_meta
        if storage_meta is None:
            storage_meta = StorageMeta()
            metadata.storage_meta = storage_meta
        storage_meta.load_id = self.load_id
        return metadata


class _DiffusersLoadPlanner(dcp.DefaultLoadPlanner):
    """Remap diffusers source keys onto the OmniMoTModel.net state dict for DCP load."""

    def __init__(
        self,
        checkpoint_path: Path,
        *,
        defer_modelopt_fp8_weights_loading: bool = False,
    ) -> None:
        """Initialize the planner.

        When ``defer_modelopt_fp8_weights_loading`` is enabled, ModelOpt FP8
        weight tensors are omitted from the DCP load so they can be installed
        directly as TorchAO weights after the remaining checkpoint is loaded.
        """
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.weight_map = _diffusers_weight_map(checkpoint_path)
        self.files_to_keys = _diffusers_files_to_keys(self.weight_map)
        self.has_vision_weights = any(rel_path.startswith("vision_encoder/") for rel_path in self.files_to_keys)
        self.defer_modelopt_fp8_weights_loading = defer_modelopt_fp8_weights_loading
        self.skipped_source_keys: set[str] = set()

    def set_up_planner(
        self,
        state_dict: dict[str, Any],
        metadata: Metadata | None = None,
        is_coordinator: bool = False,
    ) -> None:
        target_state_dict = self._normalize_target_state_dict(state_dict)
        if self.defer_modelopt_fp8_weights_loading:
            if metadata is None:
                raise ValueError("Checkpoint metadata is required to identify ModelOpt FP8 weights.")
            self.skipped_source_keys = {
                key
                for key, tensor_metadata in metadata.state_dict_metadata.items()
                if isinstance(tensor_metadata, TensorStorageMetadata)
                and tensor_metadata.properties.dtype == torch.float8_e4m3fn
                and key.endswith(".weight")
            }
        remapped_state_dict, loaded_keys, skipped_target_keys = self._build_remapped_state_dict(target_state_dict)

        missing_keys = set(target_state_dict) - loaded_keys - skipped_target_keys
        if not self.has_vision_weights:
            missing_keys = {key for key in missing_keys if not key.startswith("language_model.visual.")}
        # Task-specialized checkpoints (e.g. Text2Image, Image2Video) omit the
        # optional generative-modality projection heads (action, sound). They
        # are unused for those tasks, so tolerate their absence the same way
        # vision weights are tolerated when the checkpoint provides none of them.
        # LiDAR has no modality embedding of its own -- unlike action and sound, the network
        # defines only the two projections -- so the pair alone decides whether the checkpoint
        # carries the stream.
        for modality_prefixes in (
            ("action2llm.", "llm2action.", "action_modality_embed"),
            ("sound2llm.", "llm2sound.", "sound_modality_embed"),
            ("lidar2llm.", "llm2lidar."),
        ):
            if not any(key.startswith(modality_prefixes) for key in loaded_keys):
                missing_keys = {key for key in missing_keys if not key.startswith(modality_prefixes)}
        if missing_keys:
            sample = sorted(missing_keys)[:10]
            raise ValueError(
                f"Diffusers checkpoint at {self.checkpoint_path} did not provide {len(missing_keys)} "
                f"required model tensor(s). First up to 10: {sample}"
            )

        super().set_up_planner(
            state_dict=remapped_state_dict,
            metadata=metadata,
            is_coordinator=is_coordinator,
        )

    @staticmethod
    def _normalize_target_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
        target_state_dict: dict[str, Any] = {}
        for name, tensor in state_dict.items():
            net_key = _normalize_diffusers_target_key(name)
            if net_key in target_state_dict:
                raise KeyError(f"Multiple target model keys normalize to {net_key!r}.")
            target_state_dict[net_key] = tensor
        return target_state_dict

    def _build_remapped_state_dict(
        self, target_state_dict: dict[str, Any]
    ) -> tuple[dict[str, Any], set[str], set[str]]:
        """Build the state dict used to load diffusers weights into the model.

        Args:
            target_state_dict: Model state dict keyed by normalized Cosmos3 target names.

        Returns:
            A tuple containing:
            - A state dict keyed by diffusers checkpoint names whose values reference
              the corresponding target model tensors.
            - Target keys that will be populated by the DCP load.
            - Target keys whose ModelOpt FP8 weights were deferred for direct TorchAO
              installation.
        """
        remapped_state_dict: dict[str, Any] = {}
        loaded_keys: set[str] = set()
        skipped_target_keys: set[str] = set()
        # When the model is built without a visual tower (e.g. Cosmos3-Edge t2i with
        # include_visual disabled), its state dict has no `language_model.visual.*`
        # targets, so the checkpoint's vision_encoder weights have nowhere to go — skip
        # them rather than erroring on unmapped projector/visual keys.
        has_visual_target = any(key.startswith("language_model.visual.") for key in target_state_dict)
        for diff_key, rel_path in sorted(self.weight_map.items()):
            net_key = _diffusers_to_net_key(diff_key, rel_path)
            if diff_key in self.skipped_source_keys:
                if net_key in target_state_dict:
                    skipped_target_keys.add(net_key)
                continue
            if net_key is None:
                if _is_diffusers_model_weight_path(rel_path):
                    if rel_path.startswith("vision_encoder/") and not has_visual_target:
                        continue
                    raise KeyError(f"Diffusers model key {diff_key!r} from {rel_path!r} has no Cosmos3 mapping.")
                continue
            target_tensor = target_state_dict.get(net_key)
            if target_tensor is None:
                continue
            if net_key in loaded_keys:
                raise KeyError(f"Multiple diffusers keys map to target model key {net_key!r}.")
            remapped_state_dict[diff_key] = target_tensor
            loaded_keys.add(net_key)
        return remapped_state_dict, loaded_keys, skipped_target_keys


class Cosmos3OmniConfig(transformers.PretrainedConfig):
    model_type = "cosmos3_omni"

    def __init__(self, model: dict | None = None, **kwargs):
        self._use_public_model_config = False
        if model is not None and model_config_uses_public_aliases(model):
            model = restore_model_config_from_public_model_config(model)
            self._use_public_model_config = True
        if model is not None:
            model = undo_config_dict_replacements(model)
        self.model = model or {}

        super().__init__(**kwargs)

        self.auto_map = {
            "AutoConfig": "cosmos3.model.Cosmos3OmniConfig",
            "AutoModel": "cosmos3.model.Cosmos3OmniModel",
        }

    def to_dict(self) -> dict[str, Any]:
        output = super().to_dict()
        output.pop("_use_public_model_config", None)
        if self._use_public_model_config:
            output["model"] = build_public_model_config(self.model)
        return output

    @property
    def parallelism(self) -> dict:
        return self.model.get("config", {}).get("parallelism", {})

    @parallelism.setter
    def parallelism(self, value: dict | None):
        if value is None:
            return
        self.model.setdefault("config", {})["parallelism"] = unstructure_config(ParallelismConfig(**value))

    @property
    def compile(self) -> dict:
        return self.model.get("config", {}).get("compile", {})

    @compile.setter
    def compile(self, value: dict | None):
        if value is None:
            return
        self.model.setdefault("config", {})["compile"] = unstructure_config(CompileConfig(**value))

    @property
    def quantization(self) -> dict:
        return self.model.get("config", {}).get("quantization", {})

    @quantization.setter
    def quantization(self, value: dict | None):
        if value is None:
            return
        self.model.setdefault("config", {})["quantization"] = unstructure_config(QuantizationConfig(**value))


class Cosmos3OmniModel(transformers.PreTrainedModel):
    config_class = Cosmos3OmniConfig  # type: ignore

    def __init__(self, config: Cosmos3OmniConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        self.before_load_model()
        model_dict: "OmniMoTModel" = structure_config(config.model, omegaconf.DictConfig)

        # Disable training-only features
        model_dict.config.ema.enabled = False
        model_dict.config.activation_checkpointing.mode = "none"
        # Training's `dp_enabled` is unconditionally True so the FSDP2 wrap can install
        # the mixed-precision policy, and that wrap builds a device mesh, which needs an
        # initialized process group. This wrapper is only ever built for inference and
        # export -- including single-process runs with no torchrun rendezvous (the
        # checkpoint conversion scripts) -- so it must not inherit that from a training
        # config it was handed.
        model_dict.config.parallelism.enable_inference_mode = True
        if SMOKE:
            # Minimize model size for smoke test
            vlm_dict = model_dict.config.vlm_config.model_instance
            assert vlm_dict is not None
            with omegaconf.open_dict(vlm_dict.config):
                vlm_dict.config.text_config_overrides = {"num_hidden_layers": 2, "num_window_layers": 2}

        # The model loads some files by relative path 'cosmos3/...'
        with contextlib.chdir(_ROOT_DIR):
            self.model: "OmniMoTModel" = hydra.utils.instantiate(model_dict)
        self.after_load_model(self.model)

    @classmethod
    def from_pretrained_dcp(
        cls,
        checkpoint_path: Path,
        config: Cosmos3OmniConfig | None = None,
        parallelism_config: ParallelismConfig | None = None,
        compile_config: CompileConfig | None = None,
        quantization_config: QuantizationConfig | None = None,
    ):
        from cosmos_framework.inference.bimanual import load_bimanual_action_contract

        action_contract = load_bimanual_action_contract(checkpoint_path)
        if config is None:
            config = Cosmos3OmniConfig.from_pretrained(checkpoint_path)
        if parallelism_config is None:
            parallelism_config = ParallelismConfig()
        if compile_config is None:
            compile_config = CompileConfig()
        if quantization_config is None:
            quantization_config = QuantizationConfig()
        config.parallelism = attrs.asdict(parallelism_config)
        config.compile = attrs.asdict(compile_config)
        config.quantization = attrs.asdict(quantization_config)

        # ModelOpt FP8 checkpoints ship already-quantized E4M3 weights plus static
        # scales. The target linears are swapped to FP8 modules on the meta device
        # during `build_net` — before FSDP wrap and materialization — so both the
        # sharded and replicated paths work and peak memory follows the FP8 weights.
        modelopt_checkpoint = is_modelopt_fp8_checkpoint(checkpoint_path)
        if modelopt_checkpoint and quantization_config.method is not None:
            raise ValueError(
                "A ModelOpt FP8 checkpoint is already quantized; do not also request runtime quantization."
            )
        if modelopt_checkpoint and not _is_diffusers_checkpoint(checkpoint_path):
            raise ValueError(f"ModelOpt FP8 loading requires a diffusers-format checkpoint layout: {checkpoint_path}")
        _validate_mixed_precision_load(
            quantization_config,
            modelopt_checkpoint=modelopt_checkpoint,
            use_cuda_graphs=compile_config.use_cuda_graphs,
        )
        if modelopt_checkpoint:
            # Resolve which linears the checkpoint quantizes here, where the
            # diffusers key mapping lives, and hand the plain FQN list to the
            # model config so `build_net` can do the meta-device swap without
            # reaching back into the loader.
            config.quantization["modelopt_fp8_checkpoint_path"] = str(checkpoint_path)
            config.quantization["modelopt_fp8_target_fqns"] = plan_modelopt_fp8_targets(
                checkpoint_path,
                _diffusers_to_net_key,
                weight_map=_diffusers_weight_map(checkpoint_path),
            )

        # Published Diffusers bundles include the LiDAR VAE, while config.json
        # can retain its training-time object-store path and credential name.
        # Bind the runtime config to this snapshot before tokenizer construction.
        lidar_tokenizer = config.model.get("config", {}).get("lidar_tokenizer")
        bundled_lidar_vae = checkpoint_path / "lidar_vae" / "diffusion_pytorch_model.safetensors"
        if lidar_tokenizer is not None and bundled_lidar_vae.is_file():
            # HF snapshots symlink to extensionless blobs; preserve the filename
            # so the artifact loader selects the safetensors reader.
            lidar_tokenizer["vae_path"] = str(bundled_lidar_vae.absolute())
            lidar_tokenizer["bucket_name"] = ""
            lidar_tokenizer["object_store_credential_path_pretrained"] = None

        _configure_inference_cudnn()
        model = cls(config)
        model.model._bimanual_action_contract = action_contract
        # Sample argument validation reads the config, not the loaded model.
        setattr(model.model.config, "_bimanual_action_contract", action_contract)
        # Thread the local checkpoint dir to the reasoner LM (consumed by Edge's
        # lazy ``_ensure_vision_tower``): checkpoints that bundle a
        # ``vision_encoder/`` next to the weights then load it without any hub
        # access. Harmless when the dir has no bundle (hub fallback applies).
        language_model = getattr(getattr(model.model, "net", None), "language_model", None)
        if language_model is not None:
            language_model._local_checkpoint_dir = str(checkpoint_path)
        checkpoint_type = CheckpointType.from_path(checkpoint_path)
        match checkpoint_type:
            case CheckpointType.DCP:
                state_dict = get_model_state_dict(model.model)
                storage_reader = FileSystemReader(str(checkpoint_path))
            case CheckpointType.HF:
                if _is_diffusers_checkpoint(checkpoint_path):
                    state_dict = get_model_state_dict(model.model.net)
                    # Single-rank load must skip DCP's collective path: it gathers
                    # (pickles) the load plan across ranks, and _DiffusersLoadPlanner
                    # carries code objects that cannot be pickled ("cannot pickle
                    # code objects"). no_dist loads locally without collectives.
                    _dist = torch.distributed
                    no_dist = not (_dist.is_available() and _dist.is_initialized() and _dist.get_world_size() > 1)
                    dcp.load(
                        state_dict=state_dict,
                        storage_reader=_DiffusersHuggingFaceStorageReader(checkpoint_path),
                        planner=_DiffusersLoadPlanner(
                            checkpoint_path,
                            defer_modelopt_fp8_weights_loading=modelopt_checkpoint,
                        ),
                        no_dist=no_dist,
                    )
                    if modelopt_checkpoint:
                        # The FP8 weights were skipped by the planner above; install
                        # them straight from the checkpoint as TorchAO weights.
                        apply_modelopt_fp8_checkpoint_inplace(
                            model.model.net,
                            checkpoint_path,
                            key_mapper=_diffusers_to_net_key,
                            weight_map=_diffusers_weight_map(checkpoint_path),
                        )
                        # Mixed precision installs after FP8 weights are in
                        # place and the module graph is final (parallelize +
                        # materialize already ran inside build_net).
                        from cosmos_framework.utils.generator.mixed_precision import (
                            install_mixed_precision_runtime,
                        )

                        install_mixed_precision_runtime(model.model.net, quantization_config)
                    return model
                state_dict = get_model_state_dict(model)
                _raise_on_missing_vision_keys(checkpoint_path, state_dict)
                storage_reader = _MmapSafeHuggingFaceStorageReader(str(checkpoint_path))
            case _:
                assert_never(checkpoint_type)
        dcp.load(state_dict=state_dict, storage_reader=storage_reader)
        return model

    @classmethod
    def before_load_model(cls):
        # Disable duck shapes, which triggers recompile.
        misc.set_torch_compile_options(use_duck_shape=False)

        register_checkpoints()

    @classmethod
    def after_load_model(cls, model: "OmniMoTModel"):
        pass
