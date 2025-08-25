import copy
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import gui, tqdm
from einops import repeat, rearrange
import torch.distributed as dist

from magicdrivedit.registry import SCHEDULERS
from magicdrivedit.schedulers.scheduler import FewstepScheduler
from magicdrivedit.schedulers.rf.rectified_flow import timestep_transform
from magicdrivedit.schedulers.transports import TRANSPORTS
from magicdrivedit.utils.inference_utils import replace_with_null_condition
import copy

# from magicdrivedit.acceleration.communications import all_to_all

def bp():
   import sys, pudb
   import torch.distributed as dist
   ok_rank = (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
   if ok_rank and sys.stdin.isatty():  # 只有有终端的 rank0 才停
      pudb.set_trace()

@SCHEDULERS.register_module("ucgm", force=True)
class UCGMScheduler(FewstepScheduler):

   def __init__(self,
                *args,
                transport_type: str = "ReLinear",
                consistc_ratio: float = 0.0,
                enhanced_target_config=dict(
                  # lab_drop_ratio= 0.1,
                  enhanced_use_ema=True,
                  enhanced_ratio=0.0,
                  enhanced_style=None,
                  enhanced_range=[0.00, 0.75],
                ),
                loss_config=dict(
                  scaled_cbl_eps=0.0,
                  wt_cosine_loss=False,
                  weight_function=None,
                  dispersive_loss_weight=None,
                  dispersive_intra=2.0,
                  dispersive_inter=1.0,
                  mean_var_loss_weight=None,
                ),
                time_dist_ctrl=[1.0, 1.0, 1.0],
                infer_config=dict(
                  time_dist_ctrl=[1.0, 1.0, 1.0],
                  stochast_ratio=1.0,
                  extrapol_ratio=0.0,  # extrapolation for accelerating multistep sampling
                  sampling_order=1,
                  rfba_gap_steps=[0.001, 0.001]),
                cog_style_trans=False,
                **kwargs):
      super(UCGMScheduler, self).__init__(*args, **kwargs)

      self.cog_style_trans = cog_style_trans

      self.consistc_ratio = consistc_ratio
      # self.lab_drop_ratio = enhanced_target_config['lab_drop_ratio']
      self.enhanced_use_ema = enhanced_target_config['enhanced_use_ema']
      self.enhanced_ratio = enhanced_target_config['enhanced_ratio']
      self.enhanced_style = enhanced_target_config['enhanced_style']
      self.enhanced_range = enhanced_target_config['enhanced_range']
      self.scaled_cbl_eps = loss_config['scaled_cbl_eps']
      self.wt_cosine_loss = loss_config['wt_cosine_loss']
      self.weight_function = loss_config.get('weight_function', None)
      self.time_dist_ctrl = time_dist_ctrl

      dispersive_loss_weight = loss_config.get('dispersive_loss_weight', None)
      if dispersive_loss_weight is not None:
         dispersive_loss_weight = float(dispersive_loss_weight)
      self.dispersive_loss_weight = dispersive_loss_weight
      self.dispersive_intra = float(loss_config.get('dispersive_intra', 1.0))
      self.dispersive_inter = float(loss_config.get('dispersive_inter', 1.0))
      # mean_var_loss_weight = loss_config.get('mean_var_loss_weight', None)
      # if mean_var_loss_weight is not None:
      #    mean_var_loss_weight = float(mean_var_loss_weight)
      # self.mean_var_loss_weight = mean_var_loss_weight
      self.mean_var_loss_weight = None

      if self.enhanced_ratio > 1.0 and self.consistc_ratio == 0.0:
         self.enhanced_ratio = (self.enhanced_ratio - 1.0) / self.enhanced_ratio
         Warning("The enhance ratio larger than 1.0 is not supported")

      self.step = 0

      transport = TRANSPORTS[transport_type]()
      self.transport = transport
      self.alpha_in, self.gamma_in = transport.alpha_in, transport.gamma_in
      self.alpha_to, self.gamma_to = transport.alpha_to, transport.gamma_to

      if self.gamma_in(torch.tensor(0)).abs().item() < 0.005:
         self.integ_st = 0  # Start point if integral from 0 to 1
         self.alpha_in, self.gamma_in = self.gamma_in, self.alpha_in
         self.alpha_to, self.gamma_to = self.gamma_to, self.alpha_to
      elif self.alpha_in(torch.tensor(0)).abs().item() < 0.005:
         self.integ_st = 1  # Start point if integral from 1 to 0
      else:
         raise ValueError("Invalid Alpha and Gamma functions")

      # inference related
      """
      rfba_gap_steps (list[float], optional): Controls the boundary offsets
         [start_gap, end_gap] for timestep scheduling.
      Recommended configurations:
      - start_gap: Typically set to 0.0 or a small value like 0.001 (performance-dependent)
      - end_gap: Depends on model type:
         * Pure multi-step models (consistc_ratio=0.0): 0.005 or smaller
         * Pure few-step models (consistc_ratio=1.0): Between 0.2-0.8
         * Hybrid models (0.0 < consistc_ratio < 1.0): Match end_gap to consistc_ratio value
      Defaults to [0.001, 0.001].
      """
      self.rfba_gap_steps = infer_config.get('rfba_gap_steps', [0.001, 0.75])
      # self.rfba_gap_steps [0.001, 0.5]
      self.infer_time_dist_ctrl = infer_config.get('time_dist_ctrl', [1.0, 1.0, 1.0])
      self.infer_stochast_ratio = infer_config.get('stochast_ratio', 0.0)
      self.infer_extrapol_ratio = infer_config.get('extrapol_ratio', 1.0)
      self.sampling_order = infer_config.get('sampling_order', 0.0)

   def sample_beta(self, theta_1, theta_2, size):
      beta_dist = torch.distributions.Beta(theta_1, theta_2)
      beta_samples = beta_dist.sample(size)
      return beta_samples

   def kumaraswamy_transform(self, t, a, b, c):
      return (1 - (1 - t**a) ** b) ** c

   def add_noise(self, x_start, noise, t):
      # Reshape for proper broadcasting: [B] -> [B, 1, 1, 1, 1]
      alpha_t = torch.as_tensor(self.alpha_in(t), device=x_start.device, dtype=x_start.dtype).view(-1, 1, 1, 1, 1)
      gamma_t = torch.as_tensor(self.gamma_in(t), device=x_start.device, dtype=x_start.dtype).view(-1, 1, 1, 1, 1)
      x_t = alpha_t * noise + gamma_t * x_start
      return x_t

   def predict(self, x_hat, z_hat, t_next, noise=0.0, stochast_ratio=0.0):
      # Reshape for proper broadcasting: [B] -> [B, 1, 1, 1, 1]
      alpha_t = torch.as_tensor(self.alpha_in(t_next), device=x_hat.device, dtype=x_hat.dtype).view(-1, 1, 1, 1, 1)
      gamma_t = torch.as_tensor(self.gamma_in(t_next), device=x_hat.device, dtype=x_hat.dtype).view(-1, 1, 1, 1, 1)
      x_next = alpha_t * z_hat * ((1 - stochast_ratio)**0.5) + gamma_t * x_hat
      x_next += noise * (stochast_ratio ** 0.5)
      return x_next

   def predict_heun(self, x_cur, z_hat, z_pri, t, t_next, noise=0.0, stochast_ratio=0.0):
      # Reshape for proper broadcasting: [B] -> [B, 1, 1, 1, 1]
      a = torch.as_tensor(self.gamma_in(t_next) / self.gamma_in(t), device=x_cur.device, dtype=x_cur.dtype).view(-1, 1, 1, 1, 1)
      b = torch.as_tensor(0.5 * self.alpha_in(t_next) - self.gamma_in(t_next) * self.alpha_in(t) / self.gamma_in(t), device=x_cur.device, dtype=x_cur.dtype).view(-1, 1, 1, 1, 1)
      x_next = a * x_cur + b * (z_hat + z_pri)
      return x_next

   def forward(self, model, x_t=None, t=None, **model_kwargs):
      # Reshape for proper broadcasting: [B] -> [B, 1, 1, 1, 1]
      alpha_in_t = torch.as_tensor(self.alpha_in(t), device=x_t.device, dtype=x_t.dtype).view(-1, 1, 1, 1, 1)
      gamma_in_t = torch.as_tensor(self.gamma_in(t), device=x_t.device, dtype=x_t.dtype).view(-1, 1, 1, 1, 1)
      alpha_to_t = torch.as_tensor(self.alpha_to(t), device=x_t.device, dtype=x_t.dtype).view(-1, 1, 1, 1, 1)
      gamma_to_t = torch.as_tensor(self.gamma_to(t), device=x_t.device, dtype=x_t.dtype).view(-1, 1, 1, 1, 1)
      
      dent = alpha_in_t * gamma_to_t - gamma_in_t * alpha_to_t
      
      # 现在t应该已经是正确的1D tensor [batch_size]
      _t = t if self.integ_st == 1 else 1 - t
      unscaled_t = _t * self.num_timesteps
      unscaled_t = unscaled_t.to(dtype=x_t.dtype)

      _out = model(x_t, unscaled_t, **model_kwargs)
      if isinstance(_out, tuple):
          F_t0 = _out[0]
          outs = _out[1] if len(_out) > 1 else {}
      elif isinstance(_out, dict):
          # 常见键：'F_t0' 或 'pred'，否则取第一个值
          F_t0 = _out.get('F_t0', _out.get('pred', next(iter(_out.values()))))
          outs = {k: v for k, v in _out.items() if k not in ('F_t0', 'pred')}
      else:
          F_t0 = _out
          outs = {}
      F_t = (-1) ** (1 - self.integ_st) * F_t0
      z_hat = (x_t * gamma_to_t - F_t * gamma_in_t) / dent
      x_hat = (F_t * alpha_in_t - x_t * alpha_to_t) / dent
      return x_hat, z_hat, F_t, dent, outs

   def enhance_target(self, target, enhance_idxs, pred_w_c, pred_wo_c):
      # Ensure dtype consistency for index operations
      enhanced_value = (target[enhance_idxs] + self.enhanced_ratio * (pred_w_c[enhance_idxs] - pred_wo_c[enhance_idxs])).to(target.dtype)
      target[enhance_idxs] = enhanced_value
      
      non_enhanced_value = ((target[~enhance_idxs] + pred_w_c[~enhance_idxs]) * 0.50).to(target.dtype)
      target[~enhance_idxs] = non_enhanced_value
      return target

   def prepare_training_timestamps(self, x_start, model_kwargs):
      t = self.sample_beta(self.time_dist_ctrl[0], self.time_dist_ctrl[1], [x_start.size(0)]).to(x_start.device)
      t0 = t
      if self.use_timestep_transform:
         # t现在已经是1D tensor，直接调用timestep_transform
         t = timestep_transform(t, model_kwargs, num_timesteps=1, cog_style=self.cog_style_trans)
      return t.to(x_start)

   def ddm_training_losses(self,
                           model,
                           teacher_model,
                           ema_model,  # ema model
                           x_start,
                           model_kwargs=None,
                           noise=None,
                           mask=None,
                           nulls={},
                           vae=None,
                           weights=None,
                           t=None,
                           step=0):
      device = x_start.device
      dtype = x_start.dtype
      
      if hasattr(model, 'module'):
         model_module = model.module
      else:
         model_module = model

      if noise is None:
         noise = torch.randn_like(x_start)
      B = noise.shape[0]  # B, (C NC), T, H, W
      T, H, W = noise.shape[2:]

      # after patchifier
      H = H // 2
      W = W // 2

      # dropped = torch.rand([B], dtype=torch.float32, device=device) < self.lab_drop_ratio
      dropped = model_kwargs['drop_cond_mask']

      """
      model_kwargs = drop_condition(model_kwargs,
         model_module.camera_embedder.uncond_cam.to(device),
         model_module.frame_embedder.uncond_cam.to(device),
         nulls['y'],
         ["y", "cams", "rel_pos", "bbox"], drop_mask=dropped)
      if 'drop_cond_mask' in model_kwargs:
         model_kwargs['drop_cond_mask'] = torch.reshape(dropped, model_kwargs['drop_cond_mask'].shape)
      """

      cfg_model_kwargs = replace_with_null_condition(
         copy.deepcopy(model_kwargs),
         model_module.camera_embedder.uncond_cam.to(device),
         model_module.frame_embedder.uncond_cam.to(device),
         nulls['y'],
         ["y", "cams", "rel_pos", "bbox", "maps"], append=False)


      t = self.sample_beta(self.time_dist_ctrl[0], self.time_dist_ctrl[1], [x_start.size(0)])
      t = t.to(x_start.device)
      t0 = t
      if self.use_timestep_transform:
         # t现在已经是1D tensor，直接调用timestep_transform
         t = timestep_transform(t, model_kwargs, num_timesteps=1, cog_style=self.cog_style_trans)
      t = torch.clamp(t * self.time_dist_ctrl[2], min=0, max=1)
      t = t.to(x_start)

      # x_t = self.alpha_in(t) * noise + self.gamma_in(t) * x_start
      x_t = self.add_noise(x_start, noise, t)

      x_wc_t, z_wc_t, F_th_t, den_t, outs = self.forward(model, x_t, t, **model_kwargs)
      xs_target, zs_target, target = x_start, noise, noise * self.alpha_to(t) + x_start * self.gamma_to(t)

      rng_state = torch.cuda.get_rng_state()
      with torch.no_grad():
         if self.consistc_ratio != 0.0 or self.enhanced_ratio != 0.0:
            if self.enhanced_use_ema:
               enhanced_model = ema_model.to(x_t.dtype)
               #import torch.distributed as dist
               #rank = dist.get_rank()
               #print(f"Rank {rank}: enhanced_model.camera_embedder.uncond_cam.shape = {enhanced_model.camera_embedder.uncond_cam.shape}")
               #print(f"Rank {rank}: enhanced_model.t_embedder.mlp[0].weight.shape = {enhanced_model.t_embedder.mlp[0].weight.shape}")               
               #bp()
            else:
               enhanced_model = model.module
         else:
            enhanced_model = None

         if self.enhanced_ratio != 0.0:
            cfg_enhanced_outs = None
            enhanced_outs = None
            if self.enhanced_style == 'fc-vs-fe':  # To learning enhanced target score function
                # Get enhanced learning target that is compatible for most scenarios
               torch.cuda.set_rng_state(rng_state)
               refer_x, refer_z, _, _, cfg_enhanced_outs = self.forward(enhanced_model, x_t, t, **cfg_model_kwargs)
               torch.cuda.set_rng_state(rng_state)
               predc_x, predc_z, _, _, enhanced_outs = self.forward(enhanced_model, x_t, t, **model_kwargs)
            elif self.enhanced_style == "ft-vs-fe":  # Lightning version to performs "fc-vs-fe"
               torch.cuda.set_rng_state(rng_state)
               refer_x, refer_z, _, _, cfg_enhanced_outs = self.forward(enhanced_model, x_t, t, **cfg_model_kwargs)
               # Get enhanced learning target to support multi-step models training
               predc_x, predc_z = x_wc_t.data, z_wc_t.data
            elif self.enhanced_style == "fc-vs-xz":  # Lightning version to performs "fc-vs-fe"
               # Get enhanced learning target to facilitate few-step model training
               refer_x, refer_z = xs_target, zs_target
               predc_x, predc_z, _, _, enhanced_outs = self.forward(enhanced_model, x_t, t, **model_kwargs)
            else:
               raise ValueError(f"Unsupported target enhancement mode: {self.enhanced_style}")

            enhance_idxs = (t.flatten() < self.enhanced_range[1]) & (t.flatten() > self.enhanced_range[0])
            enhance_idxs = enhance_idxs & ~dropped.bool()
            xs_target = self.enhance_target(xs_target, enhance_idxs, predc_x, refer_x)
            zs_target = self.enhance_target(zs_target, enhance_idxs, predc_z, refer_z)
            target = zs_target * self.alpha_to(t) + xs_target * self.gamma_to(t)

         if self.consistc_ratio != 0.0:
            # Calculate the value of f^x_t and f^x_{\lambda t}
            def xfunc(r):
               torch.cuda.set_rng_state(rng_state)
               # xr = self.alpha_in(r) * noise + self.gamma_in(r) * x_start
               xr = self.add_noise(x_start, noise, r)
               if hasattr(model, 'module'):
                  model_for_consist = model.module
               else:
                  model_for_consist = model
               _, _, F_th_r, den_r, consist_outs = self.forward(model_for_consist, xr, r, **model_kwargs)
               if self.enhanced_ratio != 0.0:  # use enhanced zs_target & enhanced xs_target
                  xr = zs_target * self.alpha_in(r) + xs_target * self.gamma_in(r)
               pred_x = (F_th_r * self.alpha_in(r) - xr * self.alpha_to(r)) / den_r
               return pred_x, consist_outs

            # Calculate the derivative of f^x_t w.r.t. t
            if self.consistc_ratio == 1.0:
               epsilon = 0.005
               fc1_dt = 1 / (2 * epsilon)
               x_t_plus_e, x_t_plus_outs = xfunc(t + epsilon)
               x_t_minus_e, x_t_minus_outs = xfunc(t - epsilon)
               df_dv_dt = x_t_plus_e * fc1_dt - x_t_minus_e * fc1_dt
            else:
               epsilon = t - self.consistc_ratio * t
               fc1_dt = 1 / epsilon
               x_t = zs_target * self.alpha_in(t) + xs_target * self.gamma_in(t)
               predict_ex = F_th_t.data * self.alpha_in(t) - x_t * self.alpha_to(t)
               x_t_minus_e, x_t_minus_outs = xfunc(t - epsilon)
               df_dv_dt = predict_ex / den_t * fc1_dt - x_t_minus_e * fc1_dt
            # Calculate the learning target for F_{\theta}
            df_dv_dt = torch.clamp(df_dv_dt, min=-1, max=1)
            # weight_fc = 4 / torch.sin(t * np.pi / 2).clamp(min=0.01)
            weight_fc = 4 / torch.sin(t * 1.57)
            target = F_th_t.data - (self.alpha_in(t) / den_t * weight_fc) * df_dv_dt
      # Fix: NC should be number of cameras, not batch size
      # cams shape: [B*NC, T, 1, 3, 7], so NC = cams.shape[0] // B
      B = x_start.shape[0]
      NC = model_kwargs['cams'].shape[0] // B
      diffusion_loss = self.loss_func(F_th_t, target, NC=NC)

      if self.weight_function == "Cosine":
         # return {'loss': loss * torch.cos(t * np.pi / 2).clamp(min=0.001).flatten()}
         diffusion_loss = diffusion_loss * torch.cos(t * 1.57).flatten()

      loss_dict = {'loss': diffusion_loss, 'diffusion_loss': diffusion_loss}

      if self.mean_var_loss_weight is not None:
         feat_mean_loss = 0
         # for x_feat in outs['xs']:
         #    # x_feat = outs['xs'][-1]  # output after controlnet zeroada x: (NC, (T, S), C)
         #    _feat_mean_loss = (x_feat / 5) ** 2
         #    feat_mean_loss = feat_mean_loss + _feat_mean_loss.mean()
         for l2_xs in outs['l2_xs']:
            feat_mean_loss = feat_mean_loss + l2_xs
         # feat_var_loss = 0.0001 * (x_feat.var() / 3) ** 2
         loss_dict['feat_mean_loss'] = feat_mean_loss
         # loss_dict['feat_var_loss'] = feat_var_loss
         loss_dict['loss'] = loss_dict['loss'] + self.mean_var_loss_weight * (feat_mean_loss) #+ feat_var_loss)

      if self.dispersive_loss_weight is not None:
         x_feat = outs['xs'][-1]  # output after controlnet zeroada x: (NC, (T, S), C)

         pH, pW = 2, 2
         _H, _W = H // pH, W // pW
         x_feat = rearrange(x_feat, 'NC (T H pH W pW) C -> (NC T) (H W) (pH pW C)', T=T, H=_H, W=_W, pH=pH, pW=pW)

         diff_l2 = (x_feat.unsqueeze(0) - x_feat.unsqueeze(1)) ** 2
         intra_img_dispersity = (-diff_l2.mean([0, 2]) / self.dispersive_intra).exp().mean()
         inter_img_dispersity = (-diff_l2.mean([1, 2]) / self.dispersive_inter).exp().mean()
         dispersive_loss = 0.5 * (intra_img_dispersity + inter_img_dispersity)

         """ batchwise
         Bn = x_feat.shape[0]
         world_size = dist.get_world_size()
         all_x_feat = torch.empty([world_size * Bn, *x_feat.shape[1:]], dtype=x_feat.dtype, device=x_feat.device)
         dist.all_gather_into_tensor(all_x_feat, x_feat)
         dispersive_l2 = (all_x_feat.unsqueeze(0) - all_x_feat.unsqueeze(1)).mean(2)
         diff_temp = -dispersive_l2 / self.dispersive_T
         computed_dispersive_loss = diff_temp.exp().mean()
         dispersive_loss = torch.tensor(0.0, dtype=diffusion_loss.dtype, device=diffusion_loss.device)
         if dist.get_rank() == 0:
            scatter_list = [computed_dispersive_loss] * world_size
         else:
            scatter_list = None
         dist.scatter(dispersive_loss, scatter_list)
         """
         loss_dict['dispersive_loss'] = dispersive_loss
         loss_dict['loss'] = loss_dict['loss'] + self.dispersive_loss_weight * loss_dict['dispersive_loss']

      return loss_dict

   def loss_func(self, pd, pd_hat, NC):
      pd = rearrange(pd, 'B (C NC) T H W -> B C NC T H W', NC=NC)
      pd_hat = rearrange(pd_hat, 'B (C NC) T H W -> B C NC T H W', NC=NC)
      pseudo_huber_loss = torch.sqrt(((pd - pd_hat) ** 2).mean(dim=(1, 3, 4, 5)) + self.scaled_cbl_eps**2) - self.scaled_cbl_eps
      pseudo_huber_loss = pseudo_huber_loss.mean(dim=1)

      cosine_loss = (1 - F.cosine_similarity(pd, pd_hat, dim=1)).mean(dim=(1, 2, 3, 4)) if self.wt_cosine_loss else 0
      loss = pseudo_huber_loss + cosine_loss
      #print(f"DEBUG loss_func: pseudo_huber_loss.shape = {pseudo_huber_loss.shape}, cosine_loss.shape = {cosine_loss.shape if hasattr(cosine_loss, 'shape') else type(cosine_loss)}, loss.shape = {loss.shape}")
      if torch.isnan(loss.detach().cpu()).any():
         print('nan')
      # losses = {'loss': pseudo_huber_loss, 'cosine_loss': cosine_loss}
      loss = pseudo_huber_loss + cosine_loss
      return loss

   @torch.inference_mode
   def sample(self,
              model,
              text_encoder,
              z,
              prompts,
              device,
              neg_prompts=None,
              additional_args=None,
              mask=None,
              guidance_scale=None,
              nulls={},
              progress=True,):

      if guidance_scale is None:
         guidance_scale = self.cfg_scale

      assert additional_args
      n = len(prompts)

      model_kwargs = {}
      if 'y' not in additional_args:
         model_kwargs.update(text_encoder.encode(prompts))
      if additional_args is not None:
         model_kwargs.update(additional_args)

      if neg_prompts is not None:
         y_null = text_encoder.encode(neg_prompts)['y']
      else:
         y_null = text_encoder.null(n).to(device)

      B = z.shape[0]

      # prepare timesteps
      num_steps = self.num_infer_sampling_steps
      if self.sampling_order == 2:  # use half of the steps for second-order solver
         num_steps = (num_steps + 1) // 2

      # Time step discretization.
      # num_steps = num_steps + 1 if (self.rfba_gap_steps[1] - 0.0) == 0.0 else num_steps
      # print(f'{self.rfba_gap_steps[0]},{1.0 - self.rfba_gap_steps[1]},{num_steps}')

      # rfba_gap_steps: [0.001, 0.60]
      t_steps = torch.linspace(self.rfba_gap_steps[0], 1.0 - self.rfba_gap_steps[1], num_steps, dtype=torch.float64)
      t_steps = 1 - t_steps
      # t_steps = 1 - torch.arange(num_steps, dtype=torch.float64) / num_steps
      # t_steps = t_steps[:-1] if (self.rfba_gap_steps[1] - 0.0) == 0 else t_steps
      print(t_steps)
      
      # if self.use_timestep_transform:
      #    t_steps = torch.stack([timestep_transform(
      #       _t, model_kwargs, num_timesteps=1, #base_resolution=540 * 960, base_num_frames=33,
      #       cog_style=self.cog_style_trans) for _t in t_steps], dim=0)
      
      t_steps = self.kumaraswamy_transform(t_steps, *self.infer_time_dist_ctrl)
      print(t_steps)
      t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])
      t_steps = t_steps.to(z)
      # t_steps = torch.cat([(1 - t_steps), torch.zeros_like(t_steps[:1])])
      print(t_steps)

      # Prepare the buffer for the first order prediction
      x_hats, z_hats, buffer_freq = [], [], 1

      x_cur = z
      if mask is not None:
         noise_mask = torch.ones_like(mask, dtype=torch.bool)
      else:
         noise_mask = None

      # 添加模型解包逻辑
      if hasattr(model, 'module'):
          model_module = model.module
      else:
          model_module = model

      # cfg准备
      if self.with_cfg and guidance_scale:
         null_model_kwargs = replace_with_null_condition(
            model_kwargs.copy(),
            model_module.camera_embedder.uncond_cam.to(device),
            model_module.frame_embedder.uncond_cam.to(device),
            y_null,
            ["y", "cams", "rel_pos", "bbox", "maps"], append=False)
         if 'c' in nulls and nulls['c'] is not None:
            null_model_kwargs['c'] = nulls['c']
         if 'drop_cond_mask' in null_model_kwargs:
            null_model_kwargs['drop_cond_mask'] = torch.zeros_like(null_model_kwargs['drop_cond_mask'])
      else:
         null_model_kwargs = {}
      
      cache = {'samples': [z], 'x_hats': [], 'z_hats': []}
      dtype = z.dtype
      
      for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
         # 扩展时间步为batch维度
         t_cur = t_cur.expand(B)
         t_next = t_next.expand(B)
         is_last_step = i == num_steps - 1

         x_next = self.sample_step(model, x_cur, t_cur, model_kwargs, t_next,
                                   noise_mask=noise_mask,
                                   guidance_scale=guidance_scale,
                                   null_model_kwargs=null_model_kwargs,
                                   buffer_freq=buffer_freq if i>buffer_freq else 0,
                                   cache=cache, is_last_step=is_last_step)

         x_cur = x_next #.to(dtype)

      return x_cur

   def sample_step(self, model, x, t, model_kwargs, t_next,
                   noise_mask=None,
                   guidance_scale=None,
                   null_model_kwargs={},
                   buffer_freq=1, cache={}, is_last_step=False):
      outs = {}
      x_hat, z_hat, F_t, _, _ = self.forward(model, x, t, **model_kwargs)
      x_hat0 = x_hat
      z_hat0 = z_hat
      if self.with_cfg and guidance_scale:
         cfg_x_hat, cfg_z_hat, cfg_F_t, _, _ = self.forward(model, x, t, **null_model_kwargs)
         x_hat = cfg_x_hat + guidance_scale * (x_hat - cfg_x_hat)
         z_hat = cfg_z_hat + guidance_scale * (z_hat - cfg_z_hat)
      cache['samples'].append(x_hat)

      # Apply extrapolation for prediction, buffer_freq is 0 if current step does not fulfill
      if self.infer_extrapol_ratio > 0:
         cache['x_hats'].append(x_hat)
         cache['z_hats'].append(z_hat)

      if buffer_freq > 0 and self.infer_extrapol_ratio > 0:
         z_hat = z_hat + self.infer_extrapol_ratio * (z_hat - cache['z_hats'][-buffer_freq - 1])
         x_hat = x_hat + self.infer_extrapol_ratio * (x_hat - cache['x_hats'][-buffer_freq - 1])
         cache['x_hats'].pop(0)
         cache['z_hats'].pop(0)

      if self.infer_stochast_ratio == "SDE":
         stochast_ratio = (torch.sqrt((t_next - t).abs()) * torch.sqrt(2 * self.alpha_in(t))) / self.alpha_in(t_next)
         stochast_ratio = torch.clamp(stochast_ratio.square(), min=0, max=1)
         noise = torch.randn_like(x)
      else:
         stochast_ratio = self.infer_stochast_ratio
         noise = torch.randn_like(x) if stochast_ratio > 0 else 0.0

      if noise_mask is not None:
         noise_mask = repeat(noise_mask, 'B T -> B 1 T 1 1')
         noise = noise * noise_mask
      x_next = self.predict(x_hat, z_hat, t_next, noise, stochast_ratio=stochast_ratio)

      # # vn = cfg_F_t + guidance_scale * (F_t - cfg_F_t)
      # vn = F_t
      # xn = x - vn * (t - t_next)
      # x_next = xn
      if self.sampling_order == 1:
         return x_next

      if is_last_step:
         return x_next

      # Apply second order correction, Heun-like
      x_pri, z_pri, _, _, _ = self.forward(model, x_next, t_next, **model_kwargs)
      x_next = self.predict_heun(x, z_hat, z_pri, t, t_next)
      return x_next

# if __name__ == '__main__':
#    import pudb
#    from omegaconf import OmegaConf
#    from magicdrivedit.registry import build_module
#    cfg = OmegaConf.load("configs/midata/mi_camera/lcm/midata_cosmos_ucgm.yaml")
#    scheduler = build_module(OmegaConf.to_container(cfg.scheduler), SCHEDULERS)
#    t = scheduler.sample_beta(scheduler.time_dist_ctrl[0], scheduler.time_dist_ctrl[1], [1000, 1, 1, 1, 1])
#    scheduler.prepare_training_timestamps()
#    pudb.set_trace()
#    print(t)