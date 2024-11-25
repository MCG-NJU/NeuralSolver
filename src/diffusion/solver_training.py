import torch.nn.functional
import random
import torch.nn as nn
from src.diffusion.base.sampling import BaseSampler

class BaseTrainer(nn.Module):
    def __init__(self,
                 min_cfg_aug=1.0,
                 max_cfg_aug=1.0,
                 buffer_size=1024,
        ):
        super().__init__()
        self.min_cfg_aug = min_cfg_aug
        self.max_cfg_aug = max_cfg_aug
        self.buffer_size = buffer_size
    def setup(self, target_sampler: BaseSampler, source_sampler: BaseSampler):
        self.target_sampler = target_sampler
        self.source_sampler = source_sampler

    def _impl_trainstep(self, net, images, labels):
        raise NotImplementedError

    def __call__(self, nets, images, labels):
        aug_cfg = torch.rand(images.shape[0], 1, 1, 1, device=images.device) * (self.max_cfg_aug-self.min_cfg_aug) + self.min_cfg_aug
        target_sampler_guidance = self.target_sampler.guidance
        source_sampler_guidance = self.source_sampler.guidance
        self.target_sampler.guidance = aug_cfg
        self.source_sampler.guidance = aug_cfg
        net = random.choice(nets)
        out = self._impl_trainstep(net, images, labels)
        self.target_sampler.guidance = target_sampler_guidance
        self.source_sampler.guidance = source_sampler_guidance
        return out


class TrajsTrainer(BaseTrainer):
    def _impl_trainstep(self, net, images, labels):
        target_traj = self.target_sampler(net, images, labels, return_x_trajectory=True)
        source_traj = self.source_sampler(net, images, labels, return_x_trajectory=True)

        source_t = self.source_sampler._timesteps()
        source_t = torch.round(source_t*(len(target_traj)-1)).long()
        # select the corresponding target_traj
        selected_target_traj = [target_traj[i] for i in source_t]
        selected_source_traj = source_traj[1:]
        loss = 0.0
        out= dict()
        for i, (t, s) in enumerate(zip(selected_target_traj, selected_source_traj)):
            iter_loss = torch.nn.functional.mse_loss(t, s, reduction="mean")
            loss = loss + iter_loss
            out[f"iter{i}"] = iter_loss
        out["loss"] = loss
        return out


class TrajsReWeightTrainer(BaseTrainer):
    def _impl_trainstep(self, net, images, labels):
        target_traj = self.target_sampler(net, images, labels, return_x_trajectory=True)
        source_traj = self.source_sampler(net, images, labels, return_x_trajectory=True)

        source_t = self.source_sampler._timesteps()
        source_t = torch.round(source_t*(len(target_traj)-1)).long()
        # select the corresponding target_traj
        selected_target_traj = [target_traj[i] for i in source_t]
        selected_source_traj = source_traj[1:]
        final_loss = torch.nn.functional.huber_loss(target_traj[-1], source_traj[-1], reduction="mean", delta=0.001)*1000
        loss = 0.0
        out= dict()
        for i, (t, s) in enumerate(zip(selected_target_traj, selected_source_traj)):
            iter_loss = torch.nn.functional.mse_loss(t, s, reduction="mean")
            loss = loss + iter_loss
            out[f"iter{i}"] = iter_loss
        out["loss"] = loss + final_loss
        return out