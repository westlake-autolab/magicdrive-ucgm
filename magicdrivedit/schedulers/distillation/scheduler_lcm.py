import copy
import torch
import torch.nn.functional as F
from tqdm import tqdm
from functools import partial

from magicdrivedit.registry import SCHEDULERS
from magicdrivedit.schedulers.rf.rectified_flow import RFlowScheduler, mean_flat
from magicdrivedit.utils.inference_utils import add_null_condition


# From LCMScheduler.get_scalings_for_boundary_condition_discrete
# def scalings_for_boundary_conditions(timestep, sigma_data=0.5, timestep_scaling=10.0):
#     c_skip = sigma_data**2 / ((timestep / 0.1) ** 2 + sigma_data**2)
#     c_out = (timestep / 0.1) / ((timestep / 0.1) ** 2 + sigma_data**2) ** 0.5
#     return c_skip, c_out

def rf_boundary_conditions(timestep, prev_timesteps, num_timesteps):
    dt = (timestep - prev_timesteps) / num_timesteps
    return torch.ones_like(timestep), dt

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


@SCHEDULERS.register_module("lcm")
class LCMScheduler(RFlowScheduler):

    def __init__(self, *args,
                 w_min=1.0, w_max=5.0,
                 cfg_scale=None, num_infer_sampling_steps=None,
                 with_cfg=True, **kwargs):
        super(LCMScheduler, self).__init__(*args, **kwargs)
        self.w_min = int(w_min)
        self.w_max = int(w_max)

        # inference parameters
        self.num_infer_sampling_steps = num_infer_sampling_steps if num_infer_sampling_steps is not None else self.num_sampling_steps
        self.cfg_scale = cfg_scale
        self.with_cfg = with_cfg

    def sample_t(self, x, additional_args=None):
        B = x.shape[0]
        device = x.device
        index = torch.randint(0, self.num_sampling_steps, (B,), device=device).long()
        prev_index = index + 1
        timesteps = self.prepare_sampled_timesteps(B, device, additional_args=additional_args, num_timesteps=self.num_timesteps, num_sampling_steps=self.num_sampling_steps, with_zero=True)
        timesteps = timesteps[:, 0]
        start_timesteps = timesteps[index]
        prev_timesteps = timesteps[prev_index]
        return start_timesteps, prev_timesteps


    def predict_x_prev(self, v_pred, x_t, start_timesteps, timesteps):
        dt = (start_timesteps - timesteps) / self.num_timesteps
        x_prev = x_t + v_pred * dt[:, None, None, None, None]
        return x_prev

    def ddm_training_losses(self,
                            model,
                            teacher_model,
                            target_model,  # ema model
                            x_start,
                            model_kwargs=None,
                            noise=None,
                            mask=None,
                            nulls={},
                            weights=None, t=None):
        return self.lcm_training_losses(model,
                                        teacher_model,
                                        target_model,
                                        x_start,
                                        model_kwargs=model_kwargs,
                                        noise=noise,
                                        mask=mask,
                                        nulls=nulls,
                                        **kwargs)

    def lcm_training_losses(self,
                            model,
                            teacher_model,
                            target_model,  # ema model
                            x_start,
                            model_kwargs=None,
                            noise=None,
                            mask=None,
                            nulls={},
                            weights=None, t=None):
        device = x_start.device 
        dtype = x_start.dtype

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)
        B = noise.shape[0]
        start_timesteps, timesteps = self.sample_t(x_start, additional_args=model_kwargs)

        x_t = self.add_noise(x_start, noise, start_timesteps.to(dtype))
        if mask is not None:
            timesteps0 = torch.zeros_like(timesteps)
            x_t0 = self.add_noise(x_start, noise, timesteps0)
            x_t = torch.where(mask[:, None, :, None, None], x_t, x_t0)

        x_t = x_t.to(dtype)

        # dt = (start_timesteps - timesteps)
        c_skip_start, c_out_start = rf_boundary_conditions(start_timesteps, timesteps, num_timesteps=self.num_timesteps)
        c_skip_start, c_out_start = [append_dims(x, noise.ndim) for x in [c_skip_start, c_out_start]]

        # est_prev_timesteps = timesteps - 1
        # est_prev_timesteps = torch.where(est_prev_timesteps < 0, torch.zeros_like(timesteps), est_prev_timesteps)
        c_skip = torch.ones_like(timesteps)
        c_out = torch.where(timesteps<1, torch.zeros_like(timesteps), torch.ones_like(timesteps) / self.num_timesteps)
        # c_skip, c_out = rf_boundary_conditions(timesteps, est_prev_timesteps, num_sampling_steps=self.num_sampling_steps)
        c_skip, c_out = [append_dims(x, x_start.ndim) for x in [c_skip, c_out]]

        # guidance_scale = ((self.w_max - self.w_min) * torch.rand((B,)) + self.w_min).to(device, dtype=dtype)
        guidance_scale = torch.randint(self.w_min, self.w_max, (B, ), device=device, dtype=dtype)
        model_kwargs["guidance_scale"] = guidance_scale

        pred = model(x_t, start_timesteps, **model_kwargs)
        assert pred.shape[1] == x_t.shape[1]  # no cfg
        v_pred = pred
        pred_x0 = self.predict_x_prev(v_pred, x_t, start_timesteps, timesteps=0)
        # if mask is not None:
        #     mask_t = mask * self.num_timesteps
        #     mask_t_upper = mask_t >= t.unsqueeze(1)
        #     pred_x0 = torch.where(mask_t_upper[:, None, :, None, None], pred_x0, x0)

        model_pred = c_skip_start * x_t + c_out_start * pred_x0

        # teacher model does not use guidance scale emb input
        model_kwargs.pop("guidance_scale", None)

        # == teacher prediction computation ==
        # teacher cfg
        y = model_kwargs.pop("y")
        device_str = str(device).split(':')[0]
        with torch.no_grad():
            with torch.autocast(device_str, dtype=dtype):
                teacher_x_t = torch.cat([x_t, x_t], 0)
                teacher_t = torch.cat([start_timesteps, start_timesteps], 0)
                cfg_model_kwargs = add_null_condition(
                    copy.deepcopy(model_kwargs),
                    teacher_model.camera_embedder.uncond_cam.to(device),
                    teacher_model.frame_embedder.uncond_cam.to(device),
                    prepend=False)
                cfg_model_kwargs["y"] = torch.cat([y, nulls['y']], dim=0)
                teacher_pred = teacher_model(teacher_x_t, teacher_t, **cfg_model_kwargs)
                teacher_pred_cond, teacher_pred_uncond = teacher_pred.chunk(2, dim=0)
                teacher_v_pred = teacher_pred_cond + append_dims(guidance_scale, teacher_pred_cond.ndim) * (teacher_pred_cond - teacher_pred_uncond)
                teacher_x_prev = self.predict_x_prev(teacher_v_pred, x_t, start_timesteps, timesteps)

        # x_t-1 -> x0
        with torch.no_grad():
            with torch.autocast(device_str, dtype=dtype):
                target_model = target_model.to(dtype)
                model_kwargs["y"] = y
                target_v_pred = target_model(teacher_x_prev.to(dtype), timesteps.to(dtype), **model_kwargs)
                target_x0_pred = self.predict_x_prev(target_v_pred, x_t, start_timesteps, timesteps=0)
                target = c_skip * teacher_x_prev + c_out * target_x0_pred

        loss = mean_flat(F.mse_loss(model_pred.float(), target.float(), reduce=False), mask=mask)
        losses_dict = {'loss': loss}
        return losses_dict

    
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
        if self.with_cfg:
            if neg_prompts is not None:
                y_null = text_encoder.encode(neg_prompts)['y']
            else:
                y_null = text_encoder.null(n).to(device)
            model_args["y"] = torch.cat([model_args["y"], y_null], 0)
        if additional_args is not None:
            model_args.update(additional_args)

        # prepare timesteps
        B = z.shape[0]
        timesteps = self.prepare_sampled_timesteps(B, device=device, additional_args=additional_args, num_sampling_steps=self.num_infer_sampling_steps)
        # timesteps = timesteps.unsqueeze(1)

        if mask is not None:
            noise_added = torch.zeros_like(mask, dtype=torch.bool)
            noise_added = noise_added | (mask == 1)

        progress_wrap = partial(tqdm, leave=False) if progress else (lambda x: x)
        for i, t in progress_wrap(enumerate(timesteps)):
            # mask for adding noise
            if mask is not None:
                mask_t = mask * self.num_timesteps
                x0 = z.clone()
                x_noise = self.add_noise(x0, torch.randn_like(x0), t)

                mask_t_upper = mask_t >= t.unsqueeze(1)
                if self.with_cfg:
                    model_args["x_mask"] = mask_t_upper.repeat(2, 1)
                else:
                    model_args["x_mask"] = mask_t_upper
                mask_add_noise = mask_t_upper & ~noise_added

                z = torch.where(mask_add_noise[:, None, :, None, None], x_noise, x0)
                noise_added = mask_t_upper

            # classifier-free guidance
            if self.with_cfg:
                z_in = torch.cat([z, z], 0)
                t = torch.cat([t, t], 0)
            else:
                z_in = z
            pred = model(z_in, t, **model_args)
            if pred.shape[1] == z_in.shape[1] * 2:
                assert False, 'bug'
                pred = pred.chunk(2, dim=1)[0]
            else:
                assert pred.shape[1] == z_in.shape[1]
            if self.with_cfg:
                pred_cond, pred_uncond = pred.chunk(2, dim=0)
                v_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
            else:
                v_pred = pred

            # update z
            dt = timesteps[i] - timesteps[i + 1] if i < len(timesteps) - 1 else timesteps[i]
            dt = dt / self.num_timesteps
            z = z + v_pred * dt[:, None, None, None, None]

            if mask is not None:
                z = torch.where(mask_t_upper[:, None, :, None, None], z, x0)

        return z
