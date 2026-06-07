# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
from __future__ import annotations

import dataclasses
from dataclasses import replace
import hashlib
import json
from typing import TYPE_CHECKING, Literal, Optional
from anny.paths import PathLike, get_anny_cache_path
import torch
import inspect
if TYPE_CHECKING:
    from anny.models.rigged_model import RiggedModelWithLinearBlendShapes
from pathlib import Path
from typing import Callable
import importlib.metadata
ANNY_VERSION = importlib.metadata.version("anny")

# Increase this if there are any non-backwards-compatible changes to the data/metadata format
CURRENT_DATA_VERSION = 1

@dataclasses.dataclass(frozen=True)
class ModelMetadata:
    """Non-tensor configuration for an Anny model, serialized as JSON in the safetensors header."""
    model_type: Literal["tail", "procrustes"]
    bone_parents: list
    bone_labels: list
    local_change_labels: list
    pose_parameterization: str
    skinning_method: Optional[str]
    all_phenotypes: bool
    extrapolate_phenotypes: bool
    bone_orientation: Optional[str]
    version: int = CURRENT_DATA_VERSION 
    anny_version: str = ANNY_VERSION



@dataclasses.dataclass(frozen=True)
class ModelData:
    """Typed, immutable container for all data needed to construct any Anny model.

    Tensor fields are stored directly; non-tensor configuration lives in ``metadata``.
    Use :meth:`save_safetensors` / :meth:`load_safetensors` for portable serialization and
    :func:`model_from_model_data` to instantiate the correct model class.
    """
    metadata: ModelMetadata
    # Always present
    template_vertices: torch.Tensor
    faces: torch.Tensor
    blendshapes: torch.Tensor
    stacked_phenotype_blend_shapes_mask: torch.Tensor
    template_bone_heads: torch.Tensor
    bone_heads_blendshapes: torch.Tensor
    vertex_bone_weights: torch.Tensor
    vertex_bone_indices: torch.Tensor
    base_mesh_vertex_indices: torch.Tensor
    # Optional / topology-dependent
    texture_coordinates: Optional[torch.Tensor] = None
    face_texture_coordinate_indices: Optional[torch.Tensor] = None
    # Tail-based orientation (model_type == "tail")
    template_bone_tails: Optional[torch.Tensor] = None
    bone_tails_blendshapes: Optional[torch.Tensor] = None
    bone_rolls_rotmat: Optional[torch.Tensor] = None
    # Procrustes-based orientation (model_type == "procrustes")
    bone_nonzeroweight_mask: Optional[torch.Tensor] = None
    bone_vertex_indices: Optional[torch.Tensor] = None
    bone_vertex_weights: Optional[torch.Tensor] = None
    template_bone_vertices: Optional[torch.Tensor] = None
    reference_bone_orientations: Optional[torch.Tensor] = None

    @property
    def device(self) -> torch.device:
        return self.template_vertices.device

    def save_safetensors(self, path: PathLike) -> None:
        """Serialize to a safetensors file.  Non-None tensors are stored as-is; ``metadata``
        is packed as a JSON string in the safetensors header under the key ``"metadata"``."""
        import safetensors.torch
        tensors = {
            f.name: getattr(self, f.name).contiguous()
            for f in dataclasses.fields(self)
            if f.name != "metadata" and getattr(self, f.name) is not None
        }
        meta_json = json.dumps(dataclasses.asdict(self.metadata))
        safetensors.torch.save_file(tensors, path, metadata={"metadata": meta_json})

    @classmethod
    def load_safetensors(cls, path: PathLike) -> ModelData:
        """Deserialize from a safetensors file previously written by :meth:`save_safetensors`."""
        from safetensors import safe_open
        tensors = {}
        with safe_open(path, framework="pt") as f:
            meta_str = f.metadata().get("metadata", "{}")
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        meta_dict = json.loads(meta_str)
        meta_dict.setdefault("version", 1)
        # Backward compat: old files may not have base_mesh_vertex_indices
        if "base_mesh_vertex_indices" not in tensors:
            tensors["base_mesh_vertex_indices"] = torch.arange(len(tensors["template_vertices"]))
        return cls(metadata=ModelMetadata(**meta_dict), **tensors)


def model_from_model_data(data: ModelData) -> RiggedModelWithLinearBlendShapes:
    """Instantiate the correct concrete model class from a :class:`ModelData`."""
    from anny.models.phenotype import (
        RiggedModelWithPhenotypeParameters,
        RiggedModelWithProcrustesAndPhenotypeParameters,
    )
    if data.metadata.model_type == "procrustes":
        return RiggedModelWithProcrustesAndPhenotypeParameters.from_model_data(data)
    return RiggedModelWithPhenotypeParameters.from_model_data(data)


def _get_builder_metadata(f: Callable[..., ModelData], *args, **kwargs) -> dict[str, str | int | bool]:
    all_kwargs = {}
    def _to_valid(x):
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, (set, frozenset)):
            return sorted(x)
        return x
    for i, param in enumerate(inspect.signature(f).parameters.values()):
        if i < len(args):
            all_kwargs[param.name] = _to_valid(args[i])
            continue
        if kwargs.get(param.name) is not None:
            all_kwargs[param.name] = _to_valid(kwargs[param.name])
            continue
        if param.default is not param.empty:
            all_kwargs[param.name] = _to_valid(param.default)
            continue
        raise ValueError(f"Missing value for parameter {param.name} of builder function {f.__name__}")
    return all_kwargs


def cache_builder(f: Callable[..., ModelData]) -> Callable[..., ModelData]:
    """Decorator to add metadata about the model-building function and its arguments to the resulting ModelData."""
    def wrapper(*args, **kwargs) -> ModelData:
        cache_path = get_anny_cache_path()
        if cache_path is None:
            print("No cache directory specified, building model data without caching...")
            return f(*args, **kwargs)
            
        metadata = _get_builder_metadata(f, *args, **kwargs)
        hex = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:32]
        cache_path = Path(cache_path) / f"v{CURRENT_DATA_VERSION}" / f"{f.__name__}_{hex}.safetensors"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            print(f"Loading cached model data from {cache_path}")
            data = ModelData.load_safetensors(cache_path)
        else:
            print(f"No cached model data found at {cache_path}, building model data...")
            data = f(*args, **kwargs)
            data.save_safetensors(cache_path)
            print(f"Saved built model data to cache at {cache_path}")
        return data
    return wrapper