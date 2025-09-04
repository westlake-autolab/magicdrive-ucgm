from functools import partial
from copy import deepcopy

import torch
from tqdm import tqdm
import logging
import os

from magicdrivedit.registry import SCHEDULERS
from magicdrivedit.utils.inference_utils import replace_with_null_condition

from .rectified_flow import RFlowScheduler, timestep_transform


@SCHEDULERS.register_module("rflow")
class RFLOW:
    def __init__(
        self,
        num_sampling_steps=10,
        num_timesteps=1000,
        cfg_scale=4.0,
        use_discrete_timesteps=False,
        use_timestep_transform=False,
        **kwargs,
    ):
        self.num_sampling_steps = num_sampling_steps
        self.num_timesteps = num_timesteps
        self.cfg_scale = cfg_scale
        self.use_discrete_timesteps = use_discrete_timesteps
        self.use_timestep_transform = use_timestep_transform

        self.scheduler = RFlowScheduler(
            num_timesteps=num_timesteps,
            num_sampling_steps=num_sampling_steps,
            use_discrete_timesteps=use_discrete_timesteps,
            use_timestep_transform=use_timestep_transform,
            **kwargs,
        )
        
        # 设置调试日志
        self.debug_logger = self._setup_debug_logger()
        self.global_step_counter = 0  # 全局步骤计数器
        self.log_until_step = 100     # 记录前100步


    # ===== CURRENT VERSION METHOD (COMMENTED OUT) =====
    # def add_noise(self, *args, **kwargs):
    #     return self.scheduler.add_noise(*args, **kwargs)
    
    def _setup_debug_logger(self):
        """设置调试日志记录器"""
        logger = logging.getLogger('RFLOW_DEBUG')
        logger.setLevel(logging.INFO)
        
        # 避免重复添加handler
        if not logger.handlers:
            # 创建日志文件
            log_file = 'rflow_timestep_debug.log'
            handler = logging.FileHandler(log_file, mode='w')
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
            # 同时输出到控制台（可选）
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger

    def sample(
        self,
        model,
        text_encoder,
        z,
        prompts,
        device,
        neg_prompts=None,
        additional_args=None,
        mask=None,
        guidance_scale=None,
        progress=True,
    ):
        # if no specific guidance scale is provided, use the default scale when initializing the scheduler
        if guidance_scale is None:
            guidance_scale = self.cfg_scale

        n = len(prompts)
        # text encoding
        model_args = text_encoder.encode(prompts)
        if neg_prompts is not None:
            y_null = text_encoder.encode(neg_prompts)['y']
        else:
            y_null = text_encoder.null(n).to(device)
        model_args["y"] = torch.cat([model_args["y"], y_null], 0)
        if additional_args is not None:
            model_args.update(additional_args)

        # prepare timesteps
        # ===== CURRENT VERSION (COMMENTED OUT) =====
        # B = z.shape[0]
        # timesteps = self.scheduler.prepare_sampled_timesteps(B, device=device, additional_args=additional_args, num_sampling_steps=self.num_sampling_steps)
        # print(timesteps)
        
        # ===== OG VERSION (RESTORED) =====
        timesteps = [(1.0 - i / self.num_sampling_steps) * self.num_timesteps for i in range(self.num_sampling_steps)]
        
        # 🔍 DEBUG: 记录原始timestep生成
        self.debug_logger.info(f"RFLOW开始: 原始timesteps前5步: {timesteps[:5]}, 后5步: {timesteps[-5:]}, 总共{len(timesteps)}步")
        self.debug_logger.info(f"RFLOW参数: num_sampling_steps={self.num_sampling_steps}, num_timesteps={self.num_timesteps}")
        
        if self.use_discrete_timesteps:
            timesteps = [int(round(t)) for t in timesteps]
            self.debug_logger.info(f"RFLOW离散化后: 前5步: {timesteps[:5]}, 后5步: {timesteps[-5:]}")
            
        timesteps = [torch.tensor([t] * z.shape[0], device=device) for t in timesteps]
        
        if self.use_timestep_transform:
            self.debug_logger.info(f"RFLOW使用timestep_transform, cog_style={self.scheduler.cog_style_trans}")
            timesteps = [timestep_transform(
                t, additional_args, num_timesteps=self.num_timesteps,
                cog_style=self.scheduler.cog_style_trans,
            ) for t in timesteps]
            self.debug_logger.info(f"RFLOW变换后前5步: {[t[0].item() for t in timesteps[:5]]}")
        else:
            self.debug_logger.info(f"RFLOW不使用timestep_transform")

        if mask is not None:
            noise_added = torch.zeros_like(mask, dtype=torch.bool)
            noise_added = noise_added | (mask == 1)

        progress_wrap = partial(tqdm, leave=False) if progress else (lambda x: x)
        for i, t in progress_wrap(enumerate(timesteps)):
            # mask for adding noise
            if mask is not None:
                mask_t = mask * self.num_timesteps
                x0 = z.clone()
                x_noise = self.scheduler.add_noise(x0, torch.randn_like(x0), t)

                mask_t_upper = mask_t >= t.unsqueeze(1)
                model_args["x_mask"] = mask_t_upper.repeat(2, 1)
                mask_add_noise = mask_t_upper & ~noise_added

                z = torch.where(mask_add_noise[:, None, :, None, None], x_noise, x0)
                noise_added = mask_t_upper

            # classifier-free guidance
            z_in = torch.cat([z, z], 0)
            t = torch.cat([t, t], 0)

            pred = model(z_in, t, **model_args)
            if pred.shape[1] == z_in.shape[1] * 2:
                pred = pred.chunk(2, dim=1)[0]
            else:
                assert pred.shape[1] == z_in.shape[1]
            pred = -pred
            pred_cond, pred_uncond = pred.chunk(2, dim=0)
            v_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

            # update z
            # ===== CURRENT VERSION (COMMENTED OUT) =====
            # dt = timesteps[i] - timesteps[i + 1] if i < len(timesteps) - 1 else timesteps[i]
            # dt = dt / self.num_timesteps
            # z = z - v_pred * dt[:, None, None, None, None]
            
            dt = timesteps[i] - timesteps[i + 1] if i < len(timesteps) - 1 else timesteps[i]
            dt = dt / self.num_timesteps
            z = z + v_pred * dt[:, None, None, None, None]

            if mask is not None:
                z = torch.where(mask_t_upper[:, None, :, None, None], z, x0)

        return z

    # ===== CURRENT VERSION METHOD (COMMENTED OUT) =====
    # def step(self, model, z, t, guidance_scale=0.0, mask=None):
    #     if mask is not None:
    #         mask_t = mask * self.num_timesteps
    #         x0 = z.clone()
    #         x_noise = self.scheduler.add_noise(x0, torch.randn_like(x0), t)
    #         mask_t_upper = mask_t >= t.unsqueeze(1)
    #         model_args["x_mask"] = mask_t_upper.repeat(2, 1)
    #         mask_add_noise = mask_t_upper & ~noise_added
    #
    #         z = torch.where(mask_add_noise[:, None, :, None, None], x_noise, x0)
    #         noise_added = mask_t_upper
    #
    #     if guidance_scale is not None:
    #         z_in = torch.cat([z, z], dim=0)
    #         t = torch.cat([t, t], dim=0)
    #     
    #     pred = model(z_in, t, **model_args)
    #     if pred.shape[1] == z_in.shape[1] * 2:  # predict both x0 and noise
    #         pred = pred.chunk(2, dim=1)[0]
    #     else:
    #         assert pred.shape[1] == z_in.shape[1]
    #     
    #     if guidance_scale is not None:
    #         pred_cond, pred_uncond = pred.chunk(2, dim=0)
    #         v_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
    #     else:
    #         v_pred = pred
    #
    #     dt = timesteps[i] - timesteps[i + 1] if i < len(timesteps) - 1 else timesteps[i]
    #     dt = dt / self.num_timesteps
    #     z_prev = z_prev - v_pred * dt[:, None, None, None, None]
    #
    #     if mask is not None:
    #         z_prev = torch.where(mask_t_upper[:, None, :, None, None], z, x0)
    #     return z_prev

    def training_losses(self, model, x_start, model_kwargs=None, noise=None, mask=None, weights=None, t=None):
        return self.scheduler.training_losses(model, x_start, model_kwargs, noise, mask, weights, t)


@SCHEDULERS.register_module("rflow-slice")
class RFLOW_SLICE(RFLOW):

    # ===== CURRENT VERSION DECORATOR (COMMENTED OUT) =====
    # @torch.inference_mode
    def sample(
        self,
        model,
        text_encoder,
        z,
        prompts,
        device,
        neg_prompts=None,
        additional_args=None,
        mask=None,
        guidance_scale=None,
        progress=True,
    ):
        # if no specific guidance scale is provided, use the default scale when initializing the scheduler
        if guidance_scale is None:
            guidance_scale = self.cfg_scale

        n = len(prompts)
        # text encoding
        model_args = text_encoder.encode(prompts)
        if additional_args is not None:
            model_args.update(additional_args)
        if neg_prompts is not None:
            y_null = text_encoder.encode(neg_prompts)['y']
        else:
            y_null = text_encoder.null(n).to(device)

        # prepare timesteps
        timesteps = [(1.0 - i / self.num_sampling_steps) * self.num_timesteps for i in range(self.num_sampling_steps)]
        # ===== CURRENT VERSION DEBUG (COMMENTED OUT) =====
        # print(timesteps)
        if self.use_discrete_timesteps:
            timesteps = [int(round(t)) for t in timesteps]
        timesteps = [torch.tensor([t] * z.shape[0], device=device) for t in timesteps]
        if self.use_timestep_transform:
            timesteps = [timestep_transform(
                t, additional_args, num_timesteps=self.num_timesteps,
                cog_style=self.scheduler.cog_style_trans,
            ) for t in timesteps]
        # ===== CURRENT VERSION DEBUG (COMMENTED OUT) =====
        # print(timesteps)

        if mask is not None:
            noise_added = torch.zeros_like(mask, dtype=torch.bool)
            noise_added = noise_added | (mask == 1)

        progress_wrap = partial(tqdm, leave=False) if progress else (lambda x: x)
        for i, t in progress_wrap(enumerate(timesteps)):
            # mask for adding noise
            if mask is not None:
                mask_t = mask * self.num_timesteps
                x0 = z.clone()
                x_noise = self.scheduler.add_noise(x0, torch.randn_like(x0), t)

                mask_t_upper = mask_t >= t.unsqueeze(1)
                mask_add_noise = mask_t_upper & ~noise_added

                z = torch.where(mask_add_noise[:, None, :, None, None], x_noise, x0)
                noise_added = mask_t_upper

            # 1. all cond
            _model_args = {k:v for k, v in model_args.items()}
            if mask is not None:
                _model_args["x_mask"] = mask_t_upper

            pred = model(z, t, **_model_args)
            if pred.shape[1] == z.shape[1] * 2:
                pred = pred.chunk(2, dim=1)[0]
            else:
                assert pred.shape[1] == z.shape[1]
            all_pred = -pred

            # 2. all uncond
            _model_args = replace_with_null_condition(
                model_args, model.camera_embedder.uncond_cam.to(device),
                model.frame_embedder.uncond_cam.to(device), y_null,
                ["y", "bbox", "cams", "rel_pos", "maps"], append=False)
            if mask is not None:
                _model_args["x_mask"] = mask_t_upper

            pred = model(z, t, **_model_args)
            if pred.shape[1] == z.shape[1] * 2:
                pred = pred.chunk(2, dim=1)[0]
            else:
                assert pred.shape[1] == z.shape[1]
            null_pred = -pred

            # classifier-free guidance
            v_pred = null_pred + guidance_scale * (all_pred - null_pred)

            # update z
            # ===== CURRENT VERSION (COMMENTED OUT) =====
            # dt = timesteps[i] - timesteps[i + 1] if i < len(timesteps) - 1 else timesteps[i]
            # dt = dt / self.num_timesteps
            # z = z - v_pred * dt[:, None, None, None, None]
            
            # ===== OG VERSION (RESTORED) =====
            dt = timesteps[i] - timesteps[i + 1] if i < len(timesteps) - 1 else timesteps[i]
            dt = dt / self.num_timesteps
            z = z + v_pred * dt[:, None, None, None, None]

            if mask is not None:
                z = torch.where(mask_t_upper[:, None, :, None, None], z, x0)

        return z
