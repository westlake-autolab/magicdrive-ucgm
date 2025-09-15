import torch
import math
from typing import Dict, Union, Optional, List
from .sampling import BaseDiffusionSampler
from .sampling_utils import to_d
from einops import rearrange, repeat
from tqdm import tqdm

from omegaconf import ListConfig, OmegaConf
from vwm.util import append_dims, default, instantiate_from_config

class VistaUCGMSampler(BaseDiffusionSampler):
    
    def __init__(self, 
                 # UCGM核心参数
                 infer_extrapol_ratio: float = 0.0, 
                 infer_stochast_ratio: float = 1.0,
                 infer_consistc_ratio: float = 0.0,
                 sampling_order: int = 1,
                 rfba_gap_steps: List[float] = [0.001, 0.60],
                 infer_time_dist_ctrl: List[float] = [1.0, 1.0, 1.0],
                 integ_st: int = 1,  # 默认积分方向: 1->0
                 rho: float = 7.0,
                 sigma_min: float = 0.002,
                 sigma_max: float = 80.0,
                 
                 # Vista基类参数
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.infer_extrapol_ratio = infer_extrapol_ratio
        self.infer_stochast_ratio = infer_stochast_ratio
        self.infer_consistc_ratio = infer_consistc_ratio
        self.sampling_order = sampling_order
        self.rfba_gap_steps = rfba_gap_steps
        self.infer_time_dist_ctrl = infer_time_dist_ctrl
        self.integ_st = integ_st
        self.rho = rho
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        
        # UCGM传输函数 (对应ucgm.py)
        self.alpha_in = lambda t: 1 - t
        self.gamma_in = lambda t: t
        self.alpha_to = lambda t: torch.tensor(-1.0, device=t.device if hasattr(t, 'device') else 'cpu')
        self.gamma_to = lambda t: torch.tensor(1.0, device=t.device if hasattr(t, 'device') else 'cpu')



        if self.gamma_in(torch.tensor(0)).abs().item() < 0.005:
            self.integ_st = 0  # Start point if integral from 0 to 1
            self.alpha_in, self.gamma_in = self.gamma_in, self.alpha_in
            self.alpha_to, self.gamma_to = self.gamma_to, self.alpha_to
        elif self.alpha_in(torch.tensor(0)).abs().item() < 0.005:
            self.integ_st = 1  # Start point if integral from 1 to 0
        else:
            raise ValueError("Invalid Alpha and Gamma functions")

    def kumaraswamy_transform(self, t, a, b, c):
        return (1 - (1 - t**a) ** b) ** c

    def add_noise(self, x_start, noise, t):
      return self.alpha_in(t) * noise + self.gamma_in(t) * x_start

    def predict(self, x_hat, z_hat, t_next, noise=0.0, stochast_ratio=0.0):
      x_next = self.alpha_in(t_next) * z_hat * ((1 - stochast_ratio)**0.5) + self.gamma_in(t_next) * x_hat
      x_next += noise * (stochast_ratio ** 0.5)
      return x_next

    def predict_heun(self, x_cur, z_hat, z_pri, t, t_next, noise=0.0, stochast_ratio=0.0):
      a = self.gamma_in(t_next) / self.gamma_in(t)
      b = 0.5 * self.alpha_in(t_next) - self.gamma_in(t_next) * self.alpha_in(t) / self.gamma_in(t)
      x_next = a * x_cur + b * (z_hat + z_pri)
      return x_next
    
    def t_to_sigma(self, t):
        ramp = 1 - t  # t=1->ramp=0, t=0->ramp=1
        min_inv_rho = self.sigma_min ** (1 / self.rho)
        max_inv_rho = self.sigma_max ** (1 / self.rho)
        sigma = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.rho
        return sigma


    def forward(self, model, x_t=None, t=None, cond = None, cond_mask = None, uc = None):
        dent = self.alpha_in(t) * self.gamma_to(t) - self.gamma_in(t) * self.alpha_to(t)
        sigma = self.t_to_sigma(t)
        
        # 调试输出 - 只在前几步输出
        if hasattr(self, '_debug_step'):
            self._debug_step += 1
        else:
            self._debug_step = 1
        
        if self._debug_step <= 100:
            print(f"  Step {self._debug_step}: t={t.item():.4f}, sigma={sigma.item():.4f}")
        
        # 确保sigma是[batch_size]形状的张量，而不是标量
        if sigma.dim() == 0:  # 如果是标量
            sigma = sigma.expand(x_t.shape[0])  # 扩展为[batch_size]

        F_t0 = self.denoise(x_t, model, sigma, cond, 
                        cond_mask, uc)
        
        # 🔍 调试denoiser输出
        print(f"📊 Denoiser输出调试 (Step t={t.item():.3f}):")
        print(f"  x_t shape: {x_t.shape}")
        print(f"  F_t0 shape: {F_t0.shape}")
        print(f"  F_t0 范围: [{F_t0.min().item():.4f}, {F_t0.max().item():.4f}]")
        print(f"  F_t0 均值: {F_t0.mean().item():.4f}, 标准差: {F_t0.std().item():.4f}")
        print(f"  sigma: {sigma.item() if sigma.dim() == 0 else sigma[0].item():.4f}")
        
        F_t = (-1) ** (1 - self.integ_st) * F_t0
        z_hat = (x_t * self.gamma_to(t) - F_t * self.gamma_in(t)) / dent
        x_hat = (F_t * self.alpha_in(t) - x_t * self.alpha_to(t)) / dent
        return x_hat, z_hat, F_t, dent

    def sample_step(self, model, x_cur, t_cur, t_next,
                   noise_mask=None,
                   guidance_scale=None,
                   buffer_freq=1, cache={}, is_last_step=False, gamma=0.0, cond = None, cond_mask = None, uc = None):
        

        '''
        # Vista stochasity
        sigma_hat = sigma * (gamma + 1.0)
        if gamma > 0:
            eps = torch.randn_like(x) * self.s_noise
            x = x + eps * append_dims(sigma_hat ** 2 - sigma ** 2, x.ndim) ** 0.5
        '''

        outs = {}
        x_hat, z_hat, F_t, _ = self.forward(model, x_cur, t_cur, cond, cond_mask, uc)
 
        
        x_hat0 = x_hat
        z_hat0 = z_hat
        
        
        cache['samples'].append(x_hat)

        if self.infer_extrapol_ratio > 0:
            cache['x_hats'].append(x_hat)
            cache['z_hats'].append(z_hat)

        if buffer_freq > 0 and self.infer_extrapol_ratio > 0 and len(cache['z_hats']) > buffer_freq:
            z_hat = z_hat + self.infer_extrapol_ratio * (z_hat - cache['z_hats'][-buffer_freq - 1])
            x_hat = x_hat + self.infer_extrapol_ratio * (x_hat - cache['x_hats'][-buffer_freq - 1])
            cache['x_hats'].pop(0)
            cache['z_hats'].pop(0)

        if self.infer_stochast_ratio == "SDE":
            stochast_ratio = (torch.sqrt((t_next - t_cur).abs()) * torch.sqrt(2 * self.alpha_in(t_cur))) / self.alpha_in(t_next)
            stochast_ratio = torch.clamp(stochast_ratio.square(), min=0, max=1)
            noise = torch.randn_like(x_cur)
        else:
            stochast_ratio = self.infer_stochast_ratio
            noise = torch.randn_like(x_cur) if stochast_ratio > 0 else 0.0

        if noise_mask is not None:
            noise_mask = repeat(noise_mask, 'B T -> B 1 T 1 1')
            noise = noise * noise_mask
        
        x_next = self.predict(x_hat, z_hat, t_next, noise, stochast_ratio=stochast_ratio)

        if self.sampling_order == 1:
            return x_next

        if is_last_step:
            return x_next

        # Apply second order correction, Heun-like
        x_pri, z_pri, _, _ = self.forward(model, x_next, t_next, cond, cond_mask, uc)
        x_next = self.predict_heun(x_cur, z_hat, z_pri, t_cur, t_next)
        return x_next
    

    def __call__(self, 
                 denoiser,
                 x,  # x is randn
                 cond,
                 uc=None,
                 cond_frame=None,
                 cond_mask=None,
                 num_steps=None,
                 rfba_gap_steps=[0.001, 0.005]):

        z = x
        uc = default(uc, cond)

        assert self.sampling_order in [1, 2]
        effective_steps = self.num_steps
        
        effective_steps = (effective_steps + 1) // 2 if self.sampling_order == 2 else effective_steps
        t_steps = torch.linspace(self.rfba_gap_steps[0], 1.0 - self.rfba_gap_steps[1], effective_steps, dtype=torch.float64)
        t_steps = 1 - t_steps

        t_steps = self.kumaraswamy_transform(t_steps, *self.infer_time_dist_ctrl)
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])
        t_steps = t_steps.to(z)
        
        #Vista - 先计算replace_cond_frames
        replace_cond_frames = cond_mask is not None and cond_mask.any()
        
        # 调试输出
        print(f"🔍 UCGM Debug:")
        print(f"  effective_steps: {effective_steps}")
        print(f"  t_steps range: [{t_steps.min().item():.4f}, {t_steps.max().item():.4f}]")
        print(f"  t_steps: {t_steps[:5].tolist()}...")  # 前5个值
        print(f"  sigma_min: {self.sigma_min}, sigma_max: {self.sigma_max}")
        print(f"  cond_mask非零位置: {cond_mask.nonzero().flatten().tolist() if cond_mask is not None else 'None'}")
        print(f"  replace_cond_frames: {replace_cond_frames}")

        x_hats, z_hats, buffer_freq = [], [], 1

        
        cache = {'samples': [x], 'x_hats': [], 'z_hats': []}
        dtype = z.dtype
        
        print(f"\n⏳ ===== UCGM采样循环开始 =====")
        print(f"  初始x范围: [{x.min().item():.4f}, {x.max().item():.4f}]")
        print(f"  初始z范围: [{z.min().item():.4f}, {z.max().item():.4f}]")
        
        x_cur = z
        
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            # Vista
            if replace_cond_frames:
                x_cur = x_cur * append_dims(1 - cond_mask, x_cur.ndim) + \
                        cond_frame * append_dims(cond_mask, cond_frame.ndim)


            is_last_step = i == effective_steps - 1
            
            

            x_next = self.sample_step(
                denoiser, x_cur, t_cur, t_next,
                noise_mask=None,
                buffer_freq=1,
                cache=cache,
                is_last_step=is_last_step,
                cond = cond,
                cond_mask = cond_mask,
                uc = uc
            )
            
            x_cur = x_next
        
        if replace_cond_frames:
            x_cur = x_cur * append_dims(1 - cond_mask, x_cur.ndim) + \
                    cond_frame * append_dims(cond_mask, cond_frame.ndim)
        
        return x_cur


# 工厂函数，便于集成
def create_vista_ucgm_sampler(num_steps=25, 
                             infer_extrapol_ratio=0.0,
                             infer_stochast_ratio=1.0, 
                             infer_consistc_ratio=0.0,
                             sampling_order=1,
                             rfba_gap_steps=[0.001, 0.60],
                             infer_time_dist_ctrl=[1.0, 1.0, 1.0],
                             discretization_config=None,
                             guider_config=None):
    """
    便捷的Vista UCGM采样器创建函数
    
    参数完全对应MagicDrive的UCGM配置
    """
    return VistaUCGMSampler(
        num_steps=num_steps,
        discretization_config=discretization_config,
        guider_config=guider_config,
        infer_extrapol_ratio=infer_extrapol_ratio,
        infer_stochast_ratio=infer_stochast_ratio,
        infer_consistc_ratio=infer_consistc_ratio,
        sampling_order=sampling_order,
        rfba_gap_steps=rfba_gap_steps,
        infer_time_dist_ctrl=infer_time_dist_ctrl
    )
