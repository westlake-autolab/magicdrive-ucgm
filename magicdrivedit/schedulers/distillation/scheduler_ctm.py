import os.path as osp

import numpy as np
import torch
from einops import rearrange



from magicdrivedit.registry import SCHEDULERS
from magicdrivedit.utils.inference_utils import add_null_condition
from magicdrivedit.schedulers.rf.rectified_flow import mean_flat
from magicdrivedit.schedulers.distillation.scheduler_lcm import LCMScheduler, rf_boundary_conditions

# LPIPS._weights_url = osp.join(EXTERNAL_TORCHHUB_DIR, 'lpips_weights.pt')
from magicdrivedit.schedulers.losses import LPIPSLoss

@SCHEDULERS.register_module("ctm")
class CTMScheduler(LCMScheduler):

    def __init__(self, *args, num_solver_steps, sigma_min=0.02, ctm_loss_type='lpips', **kwargs):
        super(CTMScheduler, self).__init__(*args, **kwargs)
        assert self.num_sampling_steps >= num_solver_steps > 0
        self.num_solver_steps = num_solver_steps
        self.sigma_min = sigma_min

        # dist_loss = None
        if ctm_loss_type == 'lpips':
            # dist_loss = LPIPS(replace_pooling=True, reduction="none")
            # self.lpips_minibatch_size = 4
            self.lpip_loss = LPIPSLoss()
        self.ctm_loss_type = ctm_loss_type
        # self._dist_loss = dist_loss
        self.with_cfg = False

    # def lpip_loss(self, pred, target):
    #     B = pred.shape[0]
    #     dist_losses = []
    #     for bi in range(0, B, self.lpips_minibatch_size):
    #         dist_loss = self._dist_loss(pred[bi:bi + self.lpips_minibatch_size], target[bi:bi + self.lpips_minibatch_size])
    #         dist_losses.append(dist_loss)
    #     dist_loss = torch.cat(dist_losses, dim=0)
    #     return dist_loss


    def sample_t_and_s(self, x):
        B = x.shape[0]
        device = x.device
        t = (self.distribution.sample((B,))[:, 0] * (self.num_sampling_steps - 1)).to(dtype=torch.int64)
        t_np = t.numpy()
        t = t.to(device)
        # t = np.random.randint(0, self.num_sampling_steps - 1, (B,))
        s = np.random.randint(t_np, self.num_sampling_steps,)  # pytorch randint does not support tensor as argument
        s = torch.tensor(s, device=device, dtype=torch.int64)
        return t, s

    def ddm_training_losses(self,
                            model,
                            teacher_model,
                            target_model,  # ema model
                            x_start,
                            model_kwargs=None,
                            noise=None,
                            mask=None,
                            nulls={},
                            vae=None,
                            weights=None, t=None, step=0):

        if model_kwargs is None:
            model_kwargs = {}

        if step % 2 == 0:
            ctm_loss = self.ctm_loss(model, teacher_model, target_model, x_start,
                                    model_kwargs=model_kwargs,
                                    noise=noise,
                                    mask=mask,
                                    nulls=nulls,
                                    vae=vae,
                                    weights=weights,
                                    t=t)
            loss = ctm_loss
            losses_dict = {'loss': loss, 'ctm_loss': ctm_loss}
        if step % 2 == 1:
            dsm_loss = self.dsm_loss(model, x_start, model_kwargs=model_kwargs, noise=None, mask=mask)
            loss = dsm_loss
            losses_dict = {'loss': loss, 'dsm_loss': dsm_loss}

        # With hard-coded model weight param
        # dsm_weight = self.calculate_adaptive_dsm_weight(ctm_loss.mean(), dsm_loss.mean(), model.module.base_blocks_t[-1].mlp.fc1.weight)
        # loss = ctm_loss + dsm_weight * dsm_loss

        # losses_dict = {'loss': loss, 'dsm_loss': dsm_loss, 'ctm_loss': ctm_loss}
        return losses_dict

    def ctm_loss(self,
                 model,
                 teacher_model,
                 target_model,  # ema model
                 x_start,
                 model_kwargs=None,
                 noise=None,
                 mask=None,
                 nulls={},
                 vae=None,
                 weights=None, t=None):

        device = x_start.device
        dtype = x_start.dtype
        if noise is None:
            noise = torch.randn_like(x_start)
        B = noise.shape[0]

        # step smaller noise larger
        # timesteps smaller noise larger
        full_timesteps = self.prepare_sampled_timesteps(B, device,
                                                        additional_args=model_kwargs,
                                                        num_timesteps=self.num_timesteps,
                                                        num_sampling_steps=self.num_sampling_steps,
                                                        with_zero=True)
        full_timesteps = full_timesteps[:, 0].to(dtype)

        # t, s, e
        t_step, s_step = self.sample_t_and_s(x_start)
        t_timesteps = full_timesteps[t_step]
        s_timesteps = full_timesteps[s_step]
        e_timesteps = torch.ones_like(t_timesteps) * self.sigma_min  # very small step

        # u
        solver_step_range = min(self.num_solver_steps, max(torch.min(self.num_sampling_steps - t_step), 0)).cpu().item()
        num_solver_step = max(int(np.round(np.random.uniform() * solver_step_range)), 1)
        u_step = min(t_step + num_solver_step, s_step - 1)
        u_timesteps = full_timesteps[u_step]

        # generate x_t
        x_t = self.add_noise(x_start, noise, t_timesteps.to(dtype))
        if mask is not None:
            timesteps0 = torch.zeros_like(e_timesteps)
            x_t0 = self.add_noise(x_start, noise, timesteps0)
            x_t = torch.where(mask[:, None, :, None, None], x_t, x_t0)
        x_t = x_t.to(dtype)

        # guidance_scale = ((self.w_max - self.w_min) * torch.rand((B,)) + self.w_min).to(device, dtype=dtype)
        if self.with_cfg:
            guidance_scale = torch.randint(self.w_min, self.w_max, (B, ), device=device, dtype=dtype)
            model_kwargs["guidance_scale"] = guidance_scale

        # == compute CTM estimate ==
        # compute x_s from model(x_t, t, s)
        pred_s = model(x_t, t_timesteps, timestep_s=s_timesteps, **model_kwargs)
        assert pred_s.shape[1] == x_t.shape[1]  # no cfg
        v_pred_s = pred_s
        pred_x_s = self.predict_x_prev(v_pred_s, x_t, t_timesteps, s_timesteps)
        # model_x_s = c_skip_ts * x_t + c_out_ts * pred_s
        # compute x_s
        pred_e = teacher_model(pred_x_s, s_timesteps, timestep_s=None, **model_kwargs)
        assert pred_e.shape[1] == x_t.shape[1]  # no cfg
        v_pred_e = pred_e
        pred_x_e = self.predict_x_prev(v_pred_e, pred_x_s, s_timesteps, e_timesteps)
 
        # teacher model does not use guidance scale emb input
        model_kwargs.pop("guidance_scale", None)

        # == teacher prediction computation ==
        # teacher cfg
        # y = model_kwargs.pop("y")
        device_str = str(device).split(':')[0]

        with torch.no_grad():
            # solve t to target_u
            solver_x_t = x_t
            with torch.autocast(device_str, dtype=dtype):
                for solver_step in range(t_step, u_step):
                    solver_step = torch.ones_like(t_step) * solver_step
                    solver_t = full_timesteps[solver_step]
                    solver_s = full_timesteps[solver_step+1]
                    teacher_pred = teacher_model(solver_x_t, solver_t, timestep_s=solver_s, **model_kwargs)
                    teacher_v_pred = teacher_pred
                    solver_x_t = self.predict_x_prev(teacher_v_pred, solver_x_t, solver_t, solver_s)
            x_u = solver_x_t

            # target_u to target_s, student model with stop gradient
            target_pred_s = model(x_u, u_timesteps, timestep_s=s_timesteps, **model_kwargs)
            assert target_pred_s.shape[1] == x_t.shape[1]  # no cfg
            target_v_pred_s = target_pred_s
            target_x_s = self.predict_x_prev(target_v_pred_s, x_u, u_timesteps, s_timesteps)

            # target_s to target_e
            target_pred_e = teacher_model(target_x_s, s_timesteps, timestep_s=None, **model_kwargs)
            assert target_pred_e.shape[1] == x_t.shape[1]  # no cfg
            target_v_pred_e = target_pred_e
            target_pred_x_e = self.predict_x_prev(target_v_pred_e, target_x_s, s_timesteps, e_timesteps)

        # pred_x_e = pred_x_e.float()
        # target_pred_x_e = target_pred_x_e.float()
        ctm_loss = self.get_ctm_loss(pred_x_e, target_pred_x_e, vae=vae, dtype=dtype)
        return ctm_loss

    def get_ctm_loss(self, pred, target, loss_type="lpips", vae=None, dtype=None):
        if self.ctm_loss_type == 'l2':
            loss = mean_flat(F.mse_loss(pred, target, reduce=False), mask=mask)
            return loss
        elif self.ctm_loss_type == 'pseudo_huber':
            pseudo_huber_constant = 0.001
            loss = mean_flat(torch.sqrt((pred - target) ** 2 + pseudo_huber_constant ** 2) - pseudo_huber_constant)
            return loss
        elif self.ctm_loss_type == 'lpips':
            if dtype is not None:
                pred = pred.to(dtype)
                pred = rearrange(pred, 'B (C NC) T H W -> (B NC) C T H W', NC=7)
                target = target.to(dtype)
                target = rearrange(target, 'B (C NC) T H W -> (B NC) C T H W', NC=7)
            # with torch.no_grad():
            if True:
                pred_data = vae.decode(pred)
                pred_data = rearrange(pred_data, 'B C T H W -> (B T) C H W')
                target_data = vae.decode(target)
                target_data = rearrange(target_data, 'B C T H W -> (B T) C H W')
                pred_data = (pred_data + 1) / 2.0
                target_data = (target_data + 1) / 2.0
                lpips_loss = self.lpip_loss(pred_data, target_data)
            return lpips_loss

    def dsm_loss(self, model, x_start, model_kwargs=None, noise=None, mask=None):
        t = self.distribution.sample((x_start.shape[0],))[:, 0].to(x_start.device)  # 0-1, very small noise

        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape

        x_t = self.add_noise(x_start, noise, t)
        if mask is not None:
            t0 = torch.zeros_like(t)
            x_t0 = self.add_noise(x_start, noise, t0)
            x_t = torch.where(mask[:, None, :, None, None], x_t, x_t0)
        model_output = model(x_t, t, timestep_s=t, **model_kwargs)  # demand timestep_s<=1 should always give x0
        if model_output.shape[1] == 2 * x_t.shape[1]:
            model_output = model_output.chunk(2, dim=1)[0]
        velocity_pred = model_output
        if self.out_clip is not None:
            velocity_pred = velocity_pred.clamp(min=-self.out_clip, max=self.out_clip)

        if weights is None:
            loss = mean_flat((velocity_pred - (x_start - noise)).pow(2), mask=mask)
        else:
            weight = _extract_into_tensor(weights, t, x_start.shape)
            loss = mean_flat(weight * (velocity_pred - (x_start - noise)).pow(2), mask=mask)
        return loss

    def calculate_adaptive_dsm_weight(self, ctm_loss_mean, dsm_loss_mean, last_layer):
        ctm_loss_grad = torch.autograd.grad(ctm_loss_mean, last_layer, retain_graph=True)[0]
        dsm_loss_grad = torch.autograd.grad(dsm_loss_mean, last_layer, retain_graph=True)[0]
        dsm_weight = torch.norm(ctm_loss_grad) / (torch.norm(dsm_loss_grad) + 1e-4)
        dsm_weight = torch.clamp(weight, 0.0, 1e4).detach()
        return dsm_weight
