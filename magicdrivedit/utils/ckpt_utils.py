import functools
import json
import operator
import os
import shutil
import logging
import random
from typing import Tuple
from glob import glob
import numpy as np

import torch
import torch.distributed as dist
import torch.nn as nn
from colossalai.booster import Booster
from colossalai.checkpoint_io import GeneralCheckpointIO
from colossalai.utils import get_current_device
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from magicdrivedit.acceleration.parallel_states import get_data_parallel_group
from magicdrivedit.acceleration.communications import gather_tensors, serialize_state, deserialize_state
from .misc import get_logger
from .train_utils import update_ema

hf_endpoint = os.environ.get("HF_ENDPOINT")
if hf_endpoint is None:
    hf_endpoint = "https://huggingface.co"

pretrained_models = {}


def reparameter(ckpt, name=None, model=None):
    model_name = name
    name = os.path.basename(name)
    if not dist.is_initialized() or dist.get_rank() == 0:
        get_logger().info("loading pretrained model: %s", model_name)

    # no need pos_embed
    if "pos_embed_temporal" in ckpt:
        del ckpt["pos_embed_temporal"]
    if "pos_embed" in ckpt:
        del ckpt["pos_embed"]
    # different text length
    if "y_embedder.y_embedding" in ckpt:
        if ckpt["y_embedder.y_embedding"].shape[0] < model.y_embedder.y_embedding.shape[0]:
            get_logger().info(
                "Extend y_embedding from %s to %s",
                ckpt["y_embedder.y_embedding"].shape[0],
                model.y_embedder.y_embedding.shape[0],
            )
            additional_length = model.y_embedder.y_embedding.shape[0] - ckpt["y_embedder.y_embedding"].shape[0]
            new_y_embedding = torch.zeros(additional_length, model.y_embedder.y_embedding.shape[1])
            new_y_embedding[:] = ckpt["y_embedder.y_embedding"][-1]
            ckpt["y_embedder.y_embedding"] = torch.cat([ckpt["y_embedder.y_embedding"], new_y_embedding], dim=0)
        elif ckpt["y_embedder.y_embedding"].shape[0] > model.y_embedder.y_embedding.shape[0]:
            get_logger().info(
                "Shrink y_embedding from %s to %s",
                ckpt["y_embedder.y_embedding"].shape[0],
                model.y_embedder.y_embedding.shape[0],
            )
            ckpt["y_embedder.y_embedding"] = ckpt["y_embedder.y_embedding"][: model.y_embedder.y_embedding.shape[0]]

    return ckpt


def find_model(model_name, model=None):
    """
    Finds a pre-trained DiT model, downloading it if necessary. Alternatively, loads a model from a local path.
    """
    if model_name in pretrained_models:  # Find/download our pre-trained DiT checkpoints
        pass
    else:  # Load a custom DiT checkpoint:
        assert os.path.isfile(model_name), f"Could not find DiT checkpoint at {model_name}"
        model_ckpt = torch.load(model_name, map_location=lambda storage, loc: storage)
        model_ckpt = reparameter(model_ckpt, model_name, model=model)
    return model_ckpt


def load_from_sharded_state_dict(model, ckpt_path, model_name="model", strict=False):
    ckpt_io = GeneralCheckpointIO()
    ckpt_io.load_model(model, os.path.join(ckpt_path, model_name), strict=strict)


def model_sharding(model: torch.nn.Module):
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    for _, param in model.named_parameters():
        padding_size = (world_size - param.numel() % world_size) % world_size
        if padding_size > 0:
            padding_param = torch.nn.functional.pad(param.data.view(-1), [0, padding_size])
        else:
            padding_param = param.data.view(-1)
        splited_params = padding_param.split(padding_param.numel() // world_size)
        splited_params = splited_params[global_rank]
        param.data = splited_params


def model_gathering(model: torch.nn.Module, model_shape_dict: dict):
    global_rank = dist.get_rank()
    global_size = dist.get_world_size()
    for name, param in model.named_parameters():
        all_params = [torch.empty_like(param.data) for _ in range(global_size)]
        dist.all_gather(all_params, param.data, group=dist.group.WORLD)
        if int(global_rank) == 0:
            all_params = torch.cat(all_params)
            param.data = remove_padding(all_params, model_shape_dict[name]).view(model_shape_dict[name])
    dist.barrier()


def remove_padding(tensor: torch.Tensor, original_shape: Tuple) -> torch.Tensor:
    return tensor[: functools.reduce(operator.mul, original_shape)]


def record_model_param_shape(model: torch.nn.Module) -> dict:
    param_shape = {}
    for name, param in model.named_parameters():
        param_shape[name] = param.shape
    return param_shape


def load_checkpoint(model, ckpt_path, save_as_pt=False, model_name="model", strict=False):
    # original flow
    if ckpt_path.endswith(".pt") or ckpt_path.endswith(".pth"):
        state_dict = find_model(ckpt_path, model=model)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
        get_logger().info("Missing keys: %s", missing_keys)
        get_logger().info("Unexpected keys: %s", unexpected_keys)
    elif ckpt_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(ckpt_path)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        get_logger().info(f"Missing keys: {missing_keys}")
        get_logger().info(f"Unexpected keys: {unexpected_keys}")
    elif os.path.isdir(ckpt_path):
        load_from_sharded_state_dict(model, ckpt_path, model_name, strict=strict)
        get_logger().info("Model checkpoint loaded from %s", ckpt_path)
        if save_as_pt:
            save_path = os.path.join(ckpt_path, model_name + "_ckpt.pt")
            torch.save(model.state_dict(), save_path)
            get_logger().info("Model checkpoint saved to %s", save_path)
    else:
        raise ValueError(f"Invalid checkpoint path: {ckpt_path}")


def load_json(file_path: str):
    with open(file_path, "r") as f:
        return json.load(f)


def save_json(data, file_path: str):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


# save and load for training


def save(
    booster: Booster,
    save_dir: str,
    model: nn.Module = None,
    ema: nn.Module = None,
    optimizer: Optimizer = None,
    lr_scheduler: _LRScheduler = None,
    sampler=None,
    epoch: int = None,
    step: int = None,
    global_step: int = None,
    batch_size: int = None,
):
    save_dir = os.path.join(save_dir, f"epoch{epoch}-global_step{global_step}")
    os.makedirs(os.path.join(save_dir, "model"), exist_ok=True)

    if model is not None:
        booster.save_model(model, os.path.join(save_dir, "model"), shard=True)
    if optimizer is not None:
        booster.save_optimizer(optimizer, os.path.join(save_dir, "optimizer"), shard=True, size_per_shard=4096)
    if lr_scheduler is not None:
        booster.save_lr_scheduler(lr_scheduler, os.path.join(save_dir, "lr_scheduler"))
    if dist.get_rank() == 0:
        running_states = {
            "epoch": epoch,
            "step": step,
            "global_step": global_step,
            "batch_size": batch_size,
            "dp_world_size": dist.get_world_size(get_data_parallel_group())
        }
        save_json(running_states, os.path.join(save_dir, "running_states.json"))

        if ema is not None:
            torch.save(ema.state_dict(), os.path.join(save_dir, "ema.pt"))

        if sampler is not None:
            # only for VariableVideoBatchSampler
            torch.save(sampler.state_dict(
                step if batch_size is None else step * batch_size
            ), os.path.join(save_dir, "sampler"))
    dist.barrier()
    RandomStateManager.save_random_state_to_file(os.path.join(save_dir, "random_state.pth"))
    dist.barrier()
    return save_dir


def load(
    booster: Booster,
    load_dir: str,
    model: nn.Module = None,
    ema: nn.Module = None,
    optimizer: Optimizer = None,
    lr_scheduler: _LRScheduler = None,
    sampler=None,
    local_master=False,
) -> Tuple[int, int]:
    if load_dir.endswith("latest"):
        load_dir = find_latest(load_dir)
    assert os.path.exists(load_dir), f"Checkpoint directory {load_dir} does not exist"
    assert os.path.exists(os.path.join(load_dir, "running_states.json")), "running_states.json does not exist"
    running_states = load_json(os.path.join(load_dir, "running_states.json"))
    if model is not None:
        booster.load_model(model, os.path.join(load_dir, "model"))
    if ema is not None:
        # ema is not boosted, so we don't use booster.load_model
        try:
            ema.load_state_dict(
                torch.load(os.path.join(load_dir, "ema.pt"), map_location=torch.device("cpu")),
                strict=False,
            )
        except Exception as e:
            print(e)
            logging.exception(e)
            logging.warning(f"Got {e} from ema loading, we will not load ema model!")
            update_ema(ema, model.module, decay=0, sharded=False)
    if optimizer is not None:
        try:
            booster.load_optimizer(optimizer, os.path.join(load_dir, "optimizer"))
        except ValueError as e:
            print(e)
            logging.exception(e)
            logging.warning(f"Got {e} from optim loading, we will not load optimizer!")
    if lr_scheduler is not None:
        booster.load_lr_scheduler(lr_scheduler, os.path.join(load_dir, "lr_scheduler"))
    if sampler is not None:
        sampler.load_state_dict(torch.load(os.path.join(load_dir, "sampler")))
    try:
        RandomStateManager.load_random_state_from_file(os.path.join(load_dir, "random_state.pth"))
    except Exception as e:
        print(e)
        logging.exception(e)
        logging.warning(f"We skip random state loading!")
    dist.barrier()

    return (
        running_states["epoch"],
        running_states["step"],
    )


def find_latest(load_dir: str):
    exp_dir = os.path.dirname(load_dir)
    all_ckpt = glob(os.path.join(exp_dir, "epoch*"))
    steps = [int(d.split("_")[-1][4:]) for d in all_ckpt]  # epochN-global_stepM
    idx, step = max(enumerate(steps), key=lambda x: x[1])  # argmax
    return all_ckpt[idx]


def prepare_ckpt(path, download=False):
    return path


class RandomStateManager:
    def __init__(self, verbose=False) -> None:
        self.verbose = verbose

    def __enter__(self):
        # Save the current random state
        if self.verbose:
            logging.info("record your random state")
        self.state = self.random_state()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Restore the saved random state
        if self.verbose:
            logging.info("restore your random state")
        self.load_random_state(self.state)

    @staticmethod
    def random_state():
        """
        Save the current random state into a dictionary.
        TODO: should we use dp_group here? Maybe not, to support context.
        
        Returns:
        dict: A dictionary containing the random states of torch, numpy, and random modules.
        """
        state = {
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state(get_current_device()) if torch.cuda.is_available() else None,
            'numpy': np.random.get_state(),
            'random': random.getstate()
        }
        return state

    @staticmethod
    def load_random_state(state):
        """
        Load the random state from a dictionary.
        
        Parameters:
        state (dict): A dictionary containing the random states of torch, numpy, and random modules.
        """
        torch.set_rng_state(state['torch'])
        if state['cuda'] is not None:
            torch.cuda.set_rng_state(state['cuda'], get_current_device())
        np.random.set_state(state['numpy'])
        random.setstate(state['random'])

    @staticmethod
    def save_random_state_to_file(filepath, group=None):
        """
        Save the current random state to a file.
        
        Parameters:
        filepath (str): The path to the file where the random state will be saved.
        """
        if group is None:
            group = get_data_parallel_group()

        world_rank = dist.get_rank()
        world_size = dist.get_world_size()
        dp_rank = dist.get_rank(group=group)
        state = RandomStateManager.random_state()  # Calls the static method

        # Serialize the state to a tensor
        state = {f"rank_{dp_rank}": state}
        serialized_state = serialize_state(state).cuda()

        # Gather all states to the master rank
        gathered_states = gather_tensors(serialized_state)

        if world_rank == 0:
            state_dict = {}
            # Convert gathered tensors back to states and save them
            for i in range(world_size):
                # some key may be duplicated, we just ignore them!
                rank_state = deserialize_state(gathered_states[i])
                rank_key = list(rank_state.keys())[0]
                if rank_key in state_dict:
                    logging.info(f"{rank_key} from rank {i} already in state_dict, we skip!")
                else:
                    state_dict.update(rank_state)
            torch.save(state_dict, filepath)

    @staticmethod
    def load_random_state_from_file(filepath, group=None):
        """
        Load the random state from a file.
        
        Parameters:
        filepath (str): The path to the file from which the random state will be loaded.
        group: Please use data_parallel group!
        """
        if os.path.exists(filepath):
            state_dict = torch.load(filepath)
        else:
            raise FileNotFoundError(filepath)
        if group is None:
            group = get_data_parallel_group()

        # Load the corresponding state for this rank
        world_rank = dist.get_rank()
        dp_rank = dist.get_rank(group=group)
        dp_world_size = dist.get_world_size(group=group)
        if dp_world_size != len(state_dict):
            logging.warning(
                f"You have dp_world_size={dp_world_size}, but {len(state_dict)} "
                f"ranks from your random states. Directly loading may lead to "
                "inconsistency between ranks."
            )
        if dp_world_size > len(state_dict):
            dp_rank = dp_rank % len(state_dict)
        state = state_dict[f'rank_{dp_rank}']
        print(f"rank {world_rank} load from rank {dp_rank}")
        RandomStateManager.load_random_state(state)  # Calls the static method
        
        
        
def drop_condition(_model_args, uncond_cam, uncond_rel_pos,
                   uncond_y, keys, drop_mask, NC=7, inplace=True):
    keys = deepcopy(keys)
    unchanged_keys = ["mv_order_map", "t_order_map", "height", "width", "num_frames", "fps"]
    handled_keys = []
    model_args = {}
    if "y" in keys and "y" in _model_args:
        handled_keys.append("y")
        if inplace:
            y = _model_args["y"]
        else:
            y = _model_args["y"].clone()
        y[drop_mask].set_(uncond_y[drop_mask])
        model_args['y'] = y
        keys.remove("y")
    if "bbox" in keys and "bbox" in _model_args and _model_args["bbox"] is not None:
        raise NotImplementedError
    if "bbox2d" in keys and "bbox2d" in _model_args and _model_args["bbox2d"] is not None:
        handled_keys.append("bbox2d")
        _bbox = _model_args['bbox2d']
        bbox = {}
        B = drop_mask.shape[0]
        for k in _bbox.keys():
            null_item = torch.zeros_like(_bbox[k])
            if k in ('boxes_2d', 'masks', 'heading', 'depth', 'instance_id', 'positive_embeddings', 'gt_category_2d'):
                item = rearrange(_bbox[k], '(B NC) ... -> B NC ...', NC=NC)
                if not inplace:
                    item = item.clone()
                null_item = rearrange(null_item, '(B NC) ... -> B NC ...', NC=NC)
                item[drop_mask].set_(null_item[drop_mask])
                item = rearrange(item, 'B NC ... -> (B NC) ...')
                bbox[k] = item
            else:
                item = _bbox[k]
                if not inplace:
                    item = item.clone()
                item[drop_mask].set_(null_item[drop_mask])
                bbox[k] = item
        model_args['bbox2d'] = bbox
        keys.remove("bbox2d")
    if "cams" in keys and "cams" in _model_args:
        handled_keys.append("cams")
        cams = _model_args['cams']  # BxNC, T, 1, 3, 7
        if not inplace:
            cams = cams.clone()
        null_cams = torch.zeros_like(cams)
        BNC, T, L = null_cams.shape[:3]
        null_cams = null_cams.reshape(-1, 3, 7)
        null_cams[:] = uncond_cam[None]
        cams = cams.reshape(-1, NC, T, L, 3, 7)
        null_cams = null_cams.reshape(-1, NC, T, L, 3, 7)
        cams[drop_mask].set_(null_cams[drop_mask])
        cams = cams.reshape(BNC, T, L, 3, 7)
        model_args['cams'] = cams
        keys.remove("cams")
    if "rel_pos" in keys and "rel_pos" in _model_args:
        handled_keys.append("rel_pos")
        rel_pos = _model_args['rel_pos']  # BxNC, T, 1, 4, 4
        if not inplace:
            rel_pos = rel_pos.clone()
        null_rel_pos = torch.zeros_like(rel_pos)
        BNC, T, L, _R, _C = null_rel_pos.shape
        null_rel_pos = null_rel_pos.reshape(-1, _R, _C)
        null_rel_pos[:, :3] = uncond_rel_pos[None]
        null_rel_pos = null_rel_pos.reshape(-1, NC, T, L, _R, _C)
        rel_pos = rel_pos.reshape(-1, NC, T, L, _R, _C)
        rel_pos[drop_mask].set_(null_rel_pos[drop_mask])
        rel_pos = rel_pos.reshape(BNC, T, L, _R, _C)
        model_args['rel_pos'] = rel_pos
        keys.remove("rel_pos")
    if "maps" in keys and "maps" in _model_args and _model_args["maps"] is not None:
        raise NotImplementedError
    if len(keys) > 0:
        raise RuntimeError(f"{keys} left unhandled with {_model_args.keys()}")
    for k in _model_args.keys():
        if k in handled_keys:
            continue
        else:
            if isinstance(_model_args[k], torch.Tensor):
                model_args[k] = _model_args[k]
                if not inplace:
                    model_args[k] = model_args[k].clone()
            else:
                if inplace:
                    model_args[k] = _model_args[k]
                else:
                    model_args[k] = deepcopy(_model_args[k])
    return model_args
