#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
import logging
import argparse
import os
import sys
import json
from math import prod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator, Sequence, SupportsIndex, cast
from transformers import AutoConfig, AutoTokenizer

import torch

if TYPE_CHECKING:
    from torch import Tensor

if 'NO_LOCAL_GGUF' not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent / 'gguf-py'))
import gguf
from gguf.constants import GGUFValueType

# reuse model definitions from the conversion/ package
from conversion import LazyTorchTensor, ModelBase, get_model_class, ModelType, get_model_architecture

logger = logging.getLogger("lora-to-gguf")


@dataclass
class PartialLoraTensor:
    A: Tensor | None = None
    B: Tensor | None = None


# magic to support tensor shape modifications and splitting
class LoraTorchTensor:
    _lora_A: Tensor  # (n_rank, row_size)
    _lora_B: Tensor  # (col_size, n_rank)
    _rank: int

    def __init__(self, A: Tensor, B: Tensor):
        if len(A.shape) != len(B.shape):
            raise ValueError(f"LoRA factors must have the same rank, got {tuple(A.shape)} and {tuple(B.shape)}")
        if A.shape[-2] != B.shape[-1]:
            raise ValueError(f"LoRA factors have incompatible ranks, got {tuple(A.shape)} and {tuple(B.shape)}")
        if A.dtype != B.dtype:
            A = A.to(torch.float32)
            B = B.to(torch.float32)
        self._lora_A = A
        self._lora_B = B
        self._rank = B.shape[-1]

    def get_lora_A_B(self) -> tuple[Tensor, Tensor]:
        return (self._lora_A, self._lora_B)

    def __getitem__(
        self,
        indices: (
            SupportsIndex
            | slice
            | tuple[SupportsIndex | slice | Tensor, ...]  # TODO: add ellipsis in the type signature
        ),
    ) -> LoraTorchTensor:
        shape = self.shape
        if isinstance(indices, SupportsIndex):
            if len(shape) > 2:
                return LoraTorchTensor(self._lora_A[indices], self._lora_B[indices])
            else:
                raise NotImplementedError  # can't return a vector
        elif isinstance(indices, slice):
            if len(shape) > 2:
                return LoraTorchTensor(self._lora_A[indices], self._lora_B[indices])
            else:
                return LoraTorchTensor(self._lora_A, self._lora_B[indices])
        elif isinstance(indices, tuple):
            assert len(indices) > 0
            if indices[-1] is Ellipsis:
                return self[indices[:-1]]
            # expand ellipsis
            indices = tuple(
                u
                for v in (
                    (
                        (slice(None, None) for _ in range(len(indices) - 1))
                        if i is Ellipsis
                        else (i,)
                    )
                    for i in indices
                )
                for u in v
            )

            if len(indices) < len(shape):
                indices = (*indices, *(slice(None, None) for _ in range(len(indices), len(shape))))

            # TODO: make sure this is correct
            indices_A = (
                *(
                    (
                        j.__index__() % self._lora_A.shape[i]
                        if isinstance(j, SupportsIndex)
                        else slice(None, None)
                    )
                    for i, j in enumerate(indices[:-2])
                ),
                slice(None, None),
                indices[-1],
            )
            indices_B = indices[:-1]
            return LoraTorchTensor(self._lora_A[indices_A], self._lora_B[indices_B])
        else:
            raise NotImplementedError  # unknown indice type

    @property
    def dtype(self) -> torch.dtype:
        assert self._lora_A.dtype == self._lora_B.dtype
        return self._lora_A.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        assert len(self._lora_A.shape) == len(self._lora_B.shape)
        return (*self._lora_B.shape[:-1], self._lora_A.shape[-1])

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def dim(self) -> int:
        return self.ndim

    def size(self, dim=None):
        assert dim is None
        return self.shape

    def contiguous(self) -> LoraTorchTensor:
        return LoraTorchTensor(
            self._lora_A.contiguous(),
            self._lora_B.contiguous(),
        )

    def reshape(self, *shape: int | tuple[int, ...]) -> LoraTorchTensor:
        if isinstance(shape[0], tuple):
            new_shape: tuple[int, ...] = shape[0]
        else:
            new_shape = cast(tuple[int, ...], shape)
        orig_shape = self.shape
        if len(new_shape) < 2:
            raise NotImplementedError  # can't become a vector

        # expand -1 in the shape
        if any(dim == -1 for dim in new_shape):
            n_elems = prod(orig_shape)
            n_new_elems = prod(dim if dim != -1 else 1 for dim in new_shape)
            assert n_elems % n_new_elems == 0
            new_shape = (*(dim if dim != -1 else n_elems // n_new_elems for dim in new_shape),)

        if new_shape[-1] != orig_shape[-1]:
            raise NotImplementedError  # can't reshape the row size trivially

        shape_A = (*(1 for _ in new_shape[:-2]), self._rank, orig_shape[-1])
        shape_B = (*new_shape[:-1], self._rank)
        return LoraTorchTensor(
            self._lora_A.reshape(shape_A),
            self._lora_B.reshape(shape_B),
        )

    def reshape_as(self, other: Tensor) -> LoraTorchTensor:
        return self.reshape(*other.shape)

    def view(self, *size: int) -> LoraTorchTensor:
        return self.reshape(*size)

    def permute(self, *dims: int) -> LoraTorchTensor:
        shape = self.shape
        dims = tuple(dim - len(shape) if dim >= 0 else dim for dim in dims)
        if dims[-1] == -1:
            # TODO: support higher dimensional A shapes bigger than 1
            assert all(dim == 1 for dim in self._lora_A.shape[:-2])
            return LoraTorchTensor(self._lora_A, self._lora_B.permute(*dims))
        if len(shape) == 2 and dims[-1] == -2 and dims[-2] == -1:
            return LoraTorchTensor(self._lora_B.permute(*dims), self._lora_A.permute(*dims))
        else:
            # TODO: compose the above two
            raise NotImplementedError

    def transpose(self, dim0: int, dim1: int) -> LoraTorchTensor:
        shape = self.shape
        dims = [i for i in range(len(shape))]
        dims[dim0], dims[dim1] = dims[dim1], dims[dim0]
        return self.permute(*dims)

    def swapaxes(self, axis0: int, axis1: int) -> LoraTorchTensor:
        return self.transpose(axis0, axis1)

    def split(self, split_size: int | Sequence[int], dim: int = 0) -> tuple[LoraTorchTensor, ...]:
        shape = self.shape
        ndim = len(shape)
        if dim < 0:
            dim += ndim
        if dim == ndim - 1:
            A_chunks = self._lora_A.split(split_size, dim=-1)
            return tuple(LoraTorchTensor(a, self._lora_B) for a in A_chunks)
        elif dim == ndim - 2:
            B_chunks = self._lora_B.split(split_size, dim=-2)
            return tuple(LoraTorchTensor(self._lora_A, b) for b in B_chunks)
        else:
            B_chunks = self._lora_B.split(split_size, dim=dim)
            if self._lora_A.shape[dim] == 1:
                return tuple(LoraTorchTensor(self._lora_A, b) for b in B_chunks)
            A_chunks = self._lora_A.split(split_size, dim=dim)
            return tuple(LoraTorchTensor(a, b) for a, b in zip(A_chunks, B_chunks))

    def to(self, *args, **kwargs):
        return LoraTorchTensor(self._lora_A.to(*args, **kwargs), self._lora_B.to(*args, **kwargs))

    def __mul__(self, other) -> LoraTorchTensor:
        # Only output-side multiplication for now
        # W = B @ A, so M_out * W == (M_out * B) @ A
        if not isinstance(other, (int, float)) and other.shape and other.shape[-1] != 1:
            raise NotImplementedError
        return LoraTorchTensor(self._lora_A, self._lora_B * other)

    def __rmul__(self, other) -> LoraTorchTensor:
        return self * other

    @classmethod
    def __torch_function__(cls, func: Callable, types, args=(), kwargs=None):
        del types  # unused

        if kwargs is None:
            kwargs = {}

        if func is torch.permute:
            assert len(args)
            return type(args[0]).permute(*args, **kwargs)
        elif func is torch.reshape:
            assert len(args)
            return type(args[0]).reshape(*args, **kwargs)
        elif func is torch.stack:
            assert len(args)
            assert isinstance(args[0], Sequence)
            dim = kwargs.get("dim", 0)
            assert dim == 0
            return LoraTorchTensor(
                torch.stack([a._lora_A for a in args[0]], dim),
                torch.stack([b._lora_B for b in args[0]], dim),
            )
        elif func is torch.cat:
            assert len(args)
            assert isinstance(args[0], Sequence)
            dim = kwargs.get("dim", 0)
            assert dim == 0
            if len(args[0][0].shape) > 2:
                return LoraTorchTensor(
                    torch.cat([a._lora_A for a in args[0]], dim),
                    torch.cat([b._lora_B for b in args[0]], dim),
                )
            elif all(torch.equal(args[0][0]._lora_A, t._lora_A) for t in args[0][1:]):
                return LoraTorchTensor(
                    args[0][0]._lora_A,
                    torch.cat([b._lora_B for b in args[0]], dim),
                )
            else:
                raise NotImplementedError
        elif func is torch.split:
            assert len(args) and len(args) >= 2
            tensor, split_size = args[0], args[1]
            dim = args[2] if len(args) > 2 else kwargs.get("dim", 0)
            return tensor.split(split_size, dim=dim)
        else:
            raise NotImplementedError


def get_base_tensor_name(lora_tensor_name: str) -> str:
    base_name = lora_tensor_name.replace("base_model.model.", "")
    base_name = base_name.replace(".lora_A.weight", ".weight")
    base_name = base_name.replace(".lora_B.weight", ".weight")
    # models produced by mergekit-extract-lora have token embeddings in the adapter
    base_name = base_name.replace(".lora_embedding_A", ".weight")
    base_name = base_name.replace(".lora_embedding_B", ".weight")
    return base_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a Hugging Face PEFT LoRA adapter to a GGUF file")
    parser.add_argument(
        "--outfile", type=Path,
        help="path to write to; default: based on input. {ftype} will be replaced by the outtype.",
    )
    parser.add_argument(
        "--outtype", type=str, choices=["f32", "f16", "bf16", "q8_0", "auto"], default="f32",
        help="output format - use f32 for float32, f16 for float16, bf16 for bfloat16, q8_0 for Q8_0, auto for the highest-fidelity 16-bit float type depending on the first loaded tensor type",
    )
    parser.add_argument(
        "--bigendian", action="store_true",
        help="model is executed on big endian machine",
    )
    parser.add_argument(
        "--no-lazy", action="store_true",
        help="use more RAM by computing all outputs before writing (use in case lazy evaluation is broken)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="increase output verbosity",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="only print out what will be done, without writing any new files",
    )
    parser.add_argument(
        "--base", type=Path,
        help="directory containing Hugging Face model config files (config.json, tokenizer.json) for the base model that the adapter is based on - only config is needed, actual model weights are not required. If base model is unspecified, it will be loaded from Hugging Face hub based on the adapter config",
    )
    parser.add_argument(
        "--base-model-id", type=str,
        help="the model ID of the base model, if it is not available locally or in the adapter config. If specified, it will ignore --base and load the base model config from the Hugging Face hub (Example: 'meta-llama/Llama-3.2-1B-Instruct')",
    )
    parser.add_argument(
        "--trust-remote-code", default=False, action="store_true",
        help="trust remote code in the model",
    )
    parser.add_argument(
        "lora_path", type=Path,
        help="directory containing Hugging Face PEFT LoRA config (adapter_model.json) and weights (adapter_model.safetensors or adapter_model.bin)",
    )

    return parser.parse_args()


def load_hparams_from_hf(hf_model_id: str, trust_remote_code: bool) -> tuple[dict[str, Any], Path | None]:
    from huggingface_hub import try_to_load_from_cache

    # normally, adapter does not come with base model config, we need to load it from AutoConfig
    config = AutoConfig.from_pretrained(hf_model_id, trust_remote_code=trust_remote_code)
    cache_dir = try_to_load_from_cache(hf_model_id, "config.json")
    cache_dir = Path(cache_dir).parent if isinstance(cache_dir, str) else None

    return config.to_dict(), cache_dir


if __name__ == '__main__':
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    ftype_map: dict[str, gguf.LlamaFileType] = {
        "f32": gguf.LlamaFileType.ALL_F32,
        "f16": gguf.LlamaFileType.MOSTLY_F16,
        "bf16": gguf.LlamaFileType.MOSTLY_BF16,
        "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
        "auto": gguf.LlamaFileType.GUESSED,
    }

    ftype = ftype_map[args.outtype]

    dir_base_model: Path | None = args.base
    dir_lora: Path = args.lora_path
    base_model_id: str | None = args.base_model_id
    lora_config = dir_lora / "adapter_config.json"
    input_model = dir_lora / "adapter_model.safetensors"

    if args.outfile is not None:
        fname_out = args.outfile
    else:
        # output in the same directory as the model by default
        fname_out = dir_lora

    if os.path.exists(input_model):
        # lazy import load_file only if lora is in safetensors format.
        from safetensors.torch import load_file

        lora_model = load_file(input_model, device="cpu")
    else:
        input_model = os.path.join(dir_lora, "adapter_model.bin")
        lora_model = torch.load(input_model, map_location="cpu", weights_only=True)

    # load LoRA config
    with open(lora_config, "r") as f:
        lparams: dict[str, Any] = json.load(f)

    # load base model
    if base_model_id is not None:
        logger.info(f"Loading base model from Hugging Face: {base_model_id}")
        hparams, dir_base_model = load_hparams_from_hf(base_model_id, args.trust_remote_code)
    elif dir_base_model is None:
        if "base_model_name_or_path" in lparams:
            model_id = lparams["base_model_name_or_path"]
            logger.info(f"Loading base model from Hugging Face: {model_id}")
            try:
                hparams, dir_base_model = load_hparams_from_hf(model_id, args.trust_remote_code)
            except OSError as e:
                logger.error(f"Failed to load base model config: {e}")
                logger.error("Please try downloading the base model and add its path to --base")
                sys.exit(1)
        else:
            logger.error("'base_model_name_or_path' is not found in adapter_config.json")
            logger.error("Base model config is required. Please download the base model and add its path to --base")
            sys.exit(1)
    else:
        logger.info(f"Loading base model: {dir_base_model.name}")
        hparams = ModelBase.load_hparams(dir_base_model, False)

    with torch.inference_mode():
        model_arch = get_model_architecture(hparams, ModelType.TEXT)
        try:
            model_class = get_model_class(model_arch)
            logger.info("Using model architecture: %s", model_arch)
        except NotImplementedError:
            logger.error(f"Model {model_arch} is not supported")
            sys.exit(1)

        class LoraModel(model_class):  # ty: ignore[unsupported-base]
            model_arch = model_class.model_arch

            lora_alpha: float

            def __init__(self, *args, dir_lora_model: Path, lora_alpha: float, **kwargs):

                super().__init__(*args, **kwargs)

                # LoRA adapters are loaded alongside the base model.
                self.gguf_writer.add_chat_template(None)
                self.dir_model_card = dir_lora_model
                self.lora_alpha = float(lora_alpha)

            def set_vocab(self):
                pass

            def set_type(self):
                self.gguf_writer.add_type(gguf.GGUFType.ADAPTER)
                self.gguf_writer.add_string(gguf.Keys.Adapter.TYPE, "lora")

            def set_gguf_parameters(self):
                logger.debug("GGUF KV: %s = %d", gguf.Keys.Adapter.LORA_ALPHA, self.lora_alpha)
                self.gguf_writer.add_float32(gguf.Keys.Adapter.LORA_ALPHA, self.lora_alpha)
                alora_invocation_tokens = lparams.get("alora_invocation_tokens")
                invocation_string = lparams.get("invocation_string")
                if invocation_string and not alora_invocation_tokens:
                    logger.debug("Tokenizing invocation_string -> alora_invocation_tokens")
                    base_model_path_or_id = hparams.get("_name_or_path")
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(base_model_path_or_id)
                    except ValueError:
                        logger.error("Unable to load tokenizer from %s", base_model_path_or_id)
                        raise
                    # NOTE: There's an off-by-one with the older aLoRAs where
                    # the invocation string includes the "<|start_of_turn|>"
                    # token, but the adapters themselves were trained to
                    # activate _after_ that first token, so we drop it here.
                    alora_invocation_tokens = tokenizer(invocation_string)["input_ids"][1:]  # ty: ignore[call-non-callable]
                if alora_invocation_tokens:
                    logger.debug("GGUF KV: %s = %s", gguf.Keys.Adapter.ALORA_INVOCATION_TOKENS, alora_invocation_tokens)
                    self.gguf_writer.add_key_value(
                        gguf.Keys.Adapter.ALORA_INVOCATION_TOKENS,
                        alora_invocation_tokens,
                        GGUFValueType.ARRAY,
                        GGUFValueType.UINT32,
                    )

            def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
                # Never add extra tensors (e.g. rope_freqs) for LoRA adapters
                return ()

            def _num_experts(self) -> int:
                for key in ("num_local_experts", "num_experts", "n_routed_experts"):
                    num_experts = self.hparams.get(key)
                    if num_experts is not None:
                        return int(num_experts)
                raise ValueError("PEFT MoE LoRA tensor found, but base model config has no expert count")

            def _convert_peft_moe_lora_tensor(self, name: str, tensor: Tensor) -> tuple[str, Tensor]:
                """Map PEFT MoE LoRA tensors to tensor-shaped expert LoRA pairs.

                Qwen 3.5 stores gate/up and down factors as separate 3-D expert tensors.
                DeepSeek-V4 stores the gate/up pair in one factor pair: its B factor
                is concatenated on the output dimension and its A factor is shared.
                Other PEFT MoE adapters may flatten the expert and rank dimensions.
                """
                is_gate_up = ".mlp.experts.base_layer.lora_" in name
                is_qwen35_gate_up = self.model_arch in (gguf.MODEL_ARCH.QWEN35, gguf.MODEL_ARCH.QWEN35MOE) and (
                    name.endswith(".mlp.experts.lora_A.weight") or name.endswith(".mlp.experts.lora_B.weight")
                )
                is_qwen35_down = self.model_arch in (gguf.MODEL_ARCH.QWEN35, gguf.MODEL_ARCH.QWEN35MOE) and (
                    name.endswith(".mlp.experts.lora_A_down.weight") or name.endswith(".mlp.experts.lora_B_down.weight")
                )
                is_dsv4_gate_up = self.model_arch == gguf.MODEL_ARCH.DEEPSEEK4 and (
                    name.endswith(".mlp.experts.lora_A.weight") or name.endswith(".mlp.experts.lora_B.weight")
                )
                is_dsv4_down = self.model_arch == gguf.MODEL_ARCH.DEEPSEEK4 and (
                    name.endswith(".mlp.experts.lora_A_down.weight") or name.endswith(".mlp.experts.lora_B_down.weight")
                )
                is_down = name.endswith(".mlp.experts.lora_A.weight") or name.endswith(".mlp.experts.lora_B.weight")
                if is_qwen35_gate_up or is_dsv4_gate_up:
                    is_gate_up = True
                    is_down = False
                elif is_qwen35_down or is_dsv4_down:
                    is_gate_up = False
                    is_down = True
                if not is_gate_up and not is_down:
                    return name, tensor

                num_experts = self._num_experts()
                is_lora_a = name.endswith((".lora_A.weight", ".lora_A_down.weight"))
                is_lora_b = name.endswith((".lora_B.weight", ".lora_B_down.weight"))
                if self.model_arch == gguf.MODEL_ARCH.DEEPSEEK4 and tensor.ndim != 3:
                    raise ValueError(
                        f"DeepSeek-V4 PEFT MoE tensor must be expert-major 3-D: {name} has shape {tuple(tensor.shape)}"
                    )
                if tensor.ndim == 3:
                    if tensor.shape[0] != num_experts:
                        raise ValueError(f"Unexpected PEFT MoE tensor shape for {name}: {tuple(tensor.shape)}")
                    tensor = tensor.contiguous()
                elif is_lora_a and tensor.ndim == 2 and tensor.shape[0] % num_experts == 0:
                    rank = tensor.shape[0] // num_experts
                    tensor = tensor.reshape(num_experts, rank, tensor.shape[1]).contiguous()
                elif is_lora_b and tensor.ndim == 2 and tensor.shape[1] % num_experts == 0:
                    rank = tensor.shape[1] // num_experts
                    tensor = tensor.reshape(tensor.shape[0], rank, num_experts).permute(2, 0, 1).contiguous()
                else:
                    factor = "lora_A" if is_lora_a else "lora_B" if is_lora_b else "LoRA"
                    raise ValueError(f"Unexpected PEFT MoE {factor} shape for {name}: {tuple(tensor.shape)}")

                if is_gate_up:
                    name = name.replace(".mlp.experts.base_layer.", ".mlp.experts.gate_up_proj.")
                    name = name.replace(".mlp.experts.lora_A.weight", ".mlp.experts.gate_up_proj.lora_A.weight")
                    name = name.replace(".mlp.experts.lora_B.weight", ".mlp.experts.gate_up_proj.lora_B.weight")
                elif is_qwen35_down or is_dsv4_down:
                    name = name.replace(".mlp.experts.lora_A_down.weight", ".mlp.experts.down_proj.lora_A.weight")
                    name = name.replace(".mlp.experts.lora_B_down.weight", ".mlp.experts.down_proj.lora_B.weight")
                elif name.endswith(".mlp.experts.lora_A.weight"):
                    name = name.replace(".mlp.experts.lora_A.weight", ".mlp.experts.down_proj.lora_A.weight")
                elif name.endswith(".mlp.experts.lora_B.weight"):
                    name = name.replace(".mlp.experts.lora_B.weight", ".mlp.experts.down_proj.lora_B.weight")
                return name, tensor

            def get_tensors(self) -> Iterator[tuple[str, Tensor]]:
                tensor_map: dict[str, PartialLoraTensor] = {}

                for name, tensor in lora_model.items():
                    name, tensor = self._convert_peft_moe_lora_tensor(name, tensor)
                    if self.lazy:
                        tensor = LazyTorchTensor.from_eager(tensor)
                    base_name = get_base_tensor_name(name)
                    # filter base name, ignore tensor transformations for now
                    data_gen = lambda g=tensor: g  # noqa: E731
                    if (titem := self.filter_tensors((base_name, data_gen))) is None:
                        continue
                    base_name, _ = titem
                    # note: mergekit-extract-lora also adds token embeddings to the adapter
                    is_lora_a = ".lora_A.weight" in name or ".lora_embedding_A" in name
                    is_lora_b = ".lora_B.weight" in name or ".lora_embedding_B" in name
                    if not is_lora_a and not is_lora_b:
                        if ".base_layer.weight" in name:
                            continue
                        # mergekit-extract-lora add these layernorm to the adapter, we need to keep them
                        if "_layernorm" in name or ".norm" in name:
                            yield (base_name, tensor)
                            continue
                        logger.error(f"Unexpected name '{name}': Not a lora_A or lora_B tensor")
                        if ".embed_tokens.weight" in name or ".lm_head.weight" in name:
                            logger.error("Embeddings is present in the adapter. This can be due to new tokens added during fine tuning")
                            logger.error("Please refer to https://github.com/ggml-org/llama.cpp/pull/9948")
                        sys.exit(1)

                    if base_name in tensor_map:
                        if is_lora_a:
                            tensor_map[base_name].A = tensor
                        else:
                            tensor_map[base_name].B = tensor
                    else:
                        if is_lora_a:
                            tensor_map[base_name] = PartialLoraTensor(A=tensor)
                        else:
                            tensor_map[base_name] = PartialLoraTensor(B=tensor)

                for name, tensor in tensor_map.items():
                    if tensor.A is None or tensor.B is None:
                        missing = "lora_A" if tensor.A is None else "lora_B"
                        raise ValueError(f"LoRA tensor pair for {name!r} is missing {missing}")
                    yield (name, cast(torch.Tensor, LoraTorchTensor(tensor.A, tensor.B)))

            def _dsv4_layer_suffix(self, name: str, bid: int | None) -> str | None:
                source = name.removeprefix("model.")
                parts = source.split(".", 2)
                if len(parts) != 3 or parts[0] != "layers" or not parts[1].isdecimal():
                    return None
                layer = int(parts[1])
                if bid != layer:
                    raise ValueError(f"Tensor {name!r} parsed bid {bid} but layer name has {layer}")
                return parts[2]

            def _dsv4_target_name(self, name: str, bid: int | None) -> str:
                suffix = self._dsv4_layer_suffix(name, bid)
                if suffix is None:
                    raise ValueError(f"Unsupported DeepSeek-V4 LoRA tensor {name!r}")
                target_keys: dict[str, gguf.MODEL_TENSOR] = {
                    "self_attn.q_a_proj.weight": gguf.MODEL_TENSOR.ATTN_Q_A,
                    "self_attn.q_b_proj.weight": gguf.MODEL_TENSOR.ATTN_Q_B,
                    "self_attn.kv_proj.weight": gguf.MODEL_TENSOR.ATTN_KV,
                    "self_attn.o_b_proj.weight": gguf.MODEL_TENSOR.ATTN_OUT_B,
                    "self_attn.compressor.kv_proj.weight": gguf.MODEL_TENSOR.ATTN_COMPRESSOR_WKV,
                    "self_attn.compressor.gate_proj.weight": gguf.MODEL_TENSOR.ATTN_COMPRESSOR_WGATE,
                    "mlp.shared_experts.gate_proj.weight": gguf.MODEL_TENSOR.FFN_GATE_SHEXP,
                    "mlp.shared_experts.up_proj.weight": gguf.MODEL_TENSOR.FFN_UP_SHEXP,
                    "mlp.shared_experts.down_proj.weight": gguf.MODEL_TENSOR.FFN_DOWN_SHEXP,
                    "mlp.experts.down_proj.weight": gguf.MODEL_TENSOR.FFN_DOWN_EXP,
                }
                key = target_keys.get(suffix)
                if key is None:
                    raise ValueError(f"Unsupported DeepSeek-V4 LoRA target {name!r}")
                assert bid is not None
                return self.format_tensor_name(key, bid)

            def _dsv4_expected_shape(self, suffix: str, bid: int) -> tuple[int, ...]:
                hidden = int(self.hparams["hidden_size"])
                intermediate = int(self.hparams["moe_intermediate_size"])
                experts = self._num_experts()
                shared = int(self.hparams["n_shared_experts"])
                head_dim = int(self.hparams["head_dim"])
                if suffix == "self_attn.q_a_proj.weight":
                    return int(self.hparams["q_lora_rank"]), hidden
                if suffix == "self_attn.q_b_proj.weight":
                    return int(self.hparams["num_attention_heads"]) * head_dim, int(self.hparams["q_lora_rank"])
                if suffix == "self_attn.kv_proj.weight":
                    return head_dim, hidden
                if suffix == "self_attn.o_b_proj.weight":
                    return hidden, int(self.hparams["o_groups"]) * int(self.hparams["o_lora_rank"])
                if suffix.startswith("self_attn.compressor."):
                    ratios = self.hparams["compress_ratios"]
                    ratio = int(ratios[bid])
                    if ratio not in (4, 128):
                        raise ValueError(f"DeepSeek-V4 compressor LoRA target is invalid at layer {bid}: ratio {ratio}")
                    width = (2 if ratio == 4 else 1) * head_dim
                    return width, hidden
                if suffix in ("mlp.shared_experts.gate_proj.weight", "mlp.shared_experts.up_proj.weight"):
                    return shared * intermediate, hidden
                if suffix == "mlp.shared_experts.down_proj.weight":
                    return hidden, shared * intermediate
                if suffix == "mlp.experts.gate_up_proj.weight":
                    return experts, 2 * intermediate, hidden
                if suffix == "mlp.experts.down_proj.weight":
                    return experts, hidden, intermediate
                raise ValueError(f"Unsupported DeepSeek-V4 LoRA target {suffix!r}")

            @staticmethod
            def _validate_lora_pair(name: str, lora_a: Tensor, lora_b: Tensor, expected: tuple[int, ...]) -> None:
                a_shape = tuple(int(dim) for dim in lora_a.shape)
                b_shape = tuple(int(dim) for dim in lora_b.shape)
                if len(expected) == 2:
                    valid = (
                        len(a_shape) == 2 and len(b_shape) == 2
                        and a_shape[1] == expected[1]
                        and b_shape[0] == expected[0]
                        and a_shape[0] == b_shape[1]
                    )
                else:
                    valid = (
                        len(a_shape) == 3 and len(b_shape) == 3
                        and a_shape[0] == expected[0] and b_shape[0] == expected[0]
                        and a_shape[2] == expected[2] and b_shape[1] == expected[1]
                        and a_shape[1] == b_shape[2]
                    )
                if not valid:
                    raise ValueError(
                        f"Incompatible LoRA factors for {name}: got A {a_shape}, B {b_shape}; "
                        f"expected a rank-compatible decomposition of {expected}"
                    )

            def _modify_dsv4_tensors(self, data_torch: Tensor, name: str, bid: int) -> Iterable[tuple[str, Tensor]]:
                suffix = self._dsv4_layer_suffix(name, bid)
                assert suffix is not None
                if not isinstance(data_torch, LoraTorchTensor):
                    raise TypeError(f"DeepSeek-V4 LoRA target {name!r} is not a factor pair")
                lora_a, lora_b = data_torch.get_lora_A_B()

                if suffix == "mlp.experts.gate_up_proj.weight":
                    expected = self._dsv4_expected_shape(suffix, bid)
                    self._validate_lora_pair(name, lora_a, lora_b, expected)
                    split_size = expected[1] // 2
                    gate_b, up_b = lora_b.split(split_size, dim=1)
                    for key, part in (
                        (gguf.MODEL_TENSOR.FFN_GATE_EXP, gate_b),
                        (gguf.MODEL_TENSOR.FFN_UP_EXP, up_b),
                    ):
                        target = self.format_tensor_name(key, bid)
                        self._validate_lora_pair(target, lora_a, part, (expected[0], split_size, expected[2]))
                        yield (target + ".lora_a", lora_a)
                        yield (target + ".lora_b", part)
                    return

                target = self._dsv4_target_name(name, bid)
                expected = self._dsv4_expected_shape(suffix, bid)
                self._validate_lora_pair(target, lora_a, lora_b, expected)
                yield (target + ".lora_a", lora_a)
                yield (target + ".lora_b", lora_b)

            def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
                if self.model_arch == gguf.MODEL_ARCH.DEEPSEEK4 and bid is not None:
                    suffix = self._dsv4_layer_suffix(name, bid)
                    if suffix is not None:
                        yield from self._modify_dsv4_tensors(data_torch, name, bid)
                        return
                dest = list(super().modify_tensors(data_torch, name, bid))
                # some archs may have the same tensor for lm_head and output (tie word embeddings)
                # in this case, adapters targeting lm_head will fail when using llama-export-lora
                # therefore, we ignore them for now
                # see: https://github.com/ggml-org/llama.cpp/issues/9065
                if name == "lm_head.weight" and len(dest) == 0:
                    raise ValueError("lm_head is present in adapter, but is ignored in base model")
                for dest_name, dest_data in dest:
                    # mergekit-extract-lora add these layernorm to the adapter
                    if "_norm" in dest_name:
                        assert dest_data.dim() == 1
                        yield (dest_name, dest_data)
                        continue

                    # otherwise, we must get the lora_A and lora_B tensors
                    assert isinstance(dest_data, LoraTorchTensor)
                    lora_a, lora_b = dest_data.get_lora_A_B()

                    # note: mergekit-extract-lora flip and transpose A and B
                    # here we only need to transpose token_embd.lora_a, see llm_build_inp_embd()
                    if "token_embd.weight" in dest_name:
                        lora_a = lora_a.T

                    yield (dest_name + ".lora_a", lora_a)
                    yield (dest_name + ".lora_b", lora_b)

        alpha: float = lparams["lora_alpha"]

        model_instance = LoraModel(
            dir_base_model,
            ftype,
            fname_out,
            is_big_endian=args.bigendian,
            use_temp_file=False,
            eager=args.no_lazy,
            dry_run=args.dry_run,
            dir_lora_model=dir_lora,
            lora_alpha=alpha,
            hparams=hparams,
            # The adapter exporter only needs config metadata.  Passing the HF
            # model ID here would enumerate every remote weight shard even
            # though no base tensor is converted.
            remote_hf_model_id=None,
        )

        logger.info("Exporting model...")
        model_instance.write()
        logger.info(f"Model successfully exported to {model_instance.fname_out}")
