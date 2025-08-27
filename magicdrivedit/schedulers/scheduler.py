
import torch


def default_latent_frame_fn(num_img_frames, cog_style=False):
    if cog_style:
        num_frames = num_img_frames // 4 + num_img_frames % 2
    else:
        num_frames = num_img_frames // 17 * 5
    return num_frames


class BaseScheduler(object):

    def __init__(self,
                 num_timesteps=1000,
                 num_sampling_steps=10,
                 num_infer_sampling_steps=None,
                 use_timestep_transform=False,
                 transform_scale=1.0,
                 with_cfg=False,
                 cfg_scale=None):

        ## timestep related
        if num_infer_sampling_steps is None:
            num_infer_sampling_steps = num_sampling_steps

        self._num_timesteps = num_timesteps
        self._num_sampling_steps = num_sampling_steps
        self._num_infer_sampling_steps = num_infer_sampling_steps

        ### timesteps transform
        self.use_timestep_transform = use_timestep_transform
        self.transform_scale = transform_scale

        ## CFG-related
        self._with_cfg = with_cfg
        self.cfg_scale = cfg_scale

    @property
    def num_timesteps(self) -> int:
        return self._num_timesteps

    @property
    def num_sampling_steps(self) -> int:
        return self._num_sampling_steps

    @property
    def num_infer_sampling_steps(self):
        return self._num_infer_sampling_steps

    @property
    def with_cfg(self) -> bool:
        return self._with_cfg

    @classmethod
    def timestep_transform(cls, t, model_kwargs,
                           base_resolution=512 * 512,
                           base_num_frames=1,
                           scale=1.0,
                           num_timesteps=1,
                           cog_style=False,
                           latent_frame_fn=default_latent_frame_fn,
                           ):
        # Force fp16 input to fp32 to avoid nan output
        for key in ["height", "width", "num_frames"]:
            if model_kwargs[key].dtype == torch.float16:
                model_kwargs[key] = model_kwargs[key].float()

        t = t / num_timesteps
        resolution = model_kwargs["height"] * model_kwargs["width"]
        ratio_space = (resolution / base_resolution).sqrt()
        # NOTE: currently, we do not take fps into account
        # NOTE: temporal_reduction is hardcoded, this should be equal to the temporal reduction factor of the vae
        # TODO: hard-coded, may change later!
        if model_kwargs["num_frames"][0] == 1:
            num_frames = torch.ones_like(model_kwargs["num_frames"])
        else:
            num_frames = latent_frame_fn(model_kwargs["num_frames"], cog_style=cog_style)
        assert (num_frames >= 1).all(), "num_frames cannot be less than 1"
        ratio_time = (num_frames / base_num_frames).sqrt()

        ratio = ratio_space * ratio_time * scale
        assert (ratio > 0).all(), "ratio cannot be 0"
        new_t = ratio * t / (1 + (ratio - 1) * t)

        new_t = new_t * num_timesteps
        return new_t


class FewstepScheduler(BaseScheduler):

    def __init__(self, need_teacher=False, need_ema=False, **kwargs):
        super(FewstepScheduler, self).__init__(**kwargs)
        self._need_teacher = need_teacher
        self._need_ema = need_ema

    @property
    def need_teacher(self) -> bool:
        return self._need_teacher

    @property
    def need_ema(self) -> bool:
        return self._need_ema
