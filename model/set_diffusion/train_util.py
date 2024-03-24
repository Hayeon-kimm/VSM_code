import copy
import functools
import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
import numpy as np
from . import logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
import pickle

INITIAL_LOG_LOSS_SCALE = 20.0


def r(x):
    _max = np.max(x)
    _min = np.min(x)
    x = (x - _min) / (_max - _min)
    return x




class TrainLoop:
    def __init__(
            self,
            *,
            model,
            data,
            batch_size,
            microbatch,
            lr,
            ema_rate,
            log_interval,
            save_interval,
            resume_checkpoint,
            use_fp16=False,
            fp16_scale_growth=1e-3,
            schedule_sampler=None,
            weight_decay=0.0,
            lr_anneal_steps=0,
            val_loader=None,
            args=None,
    ):
        self.args = args
        self.val_loader = val_loader

        self.model = model
        self.data = data
        self.batch_size = batch_size
        self.microbatch = batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(self.model.diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size  # * dist.get_world_size()

        self.use_ddp = False

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True

        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    resume_checkpoint, map_location="cuda"
                )

        # dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = th.load_state_dict(
                    ema_checkpoint, map_location="cuda"
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = th.load_state_dict(
                opt_checkpoint, map_location="cuda"
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        while (
                (not self.lr_anneal_steps
                 or self.step + self.resume_step < self.lr_anneal_steps) and self.step < 200000 + 1
        ):
            cond = None
            try:
                batch = next(self.data)
            except StopIteration:
                self.data = iter(self.data)
                batch = next(self.data)
            batch, label = batch

            self.model.train()
            self.run_step(batch, label, cond, tag='train')
            if self.step % self.log_interval == 0:
                logger.dumpkvs(step=self.step)
            self.step += 1
            if (self.step - 1) % self.save_interval == 0:
                self.save()
                memory = self.model.memory
                memory_dict = {'memory': memory}
                f_save = open('./memory.pkl', 'wb')
                pickle.dump(memory_dict, f_save)
                f_save.close()

    def run_step(self, batch, label, cond, tag='test'):
        self.forward_backward(batch, label, cond, tag)

        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            if self.step > 2000 and self.step % 10 == 0:
                self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, label, cond, tag, thresh=0.3):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            # micro = batch[i : i + self.microbatch].to("cuda")

            batch = batch.to("cuda")
            dim = np.prod(batch.shape[2:])
            bs = batch.shape[0]
            ns = batch.shape[1]
            
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(bs, batch.device)
            # repeat t for ns element in each set
            t = th.repeat_interleave(t, ns, dim=0)
            weights = th.repeat_interleave(weights, ns, dim=0)

            compute_losses = functools.partial(
                self.model,
                batch,
                label,
                t,
                tag,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )
            # this loss is already in bpd and weights ar 1
            loss = (losses["loss"] * weights)
            # sum over the sample_size to obtain the per-set loss
            loss = loss.view(bs, ns).sum(-1)

        
            loss += losses["klc"]
            loss += losses["klm"]
            
            loss = loss.mean() / ns
            log_loss_dict(
                self.model.diffusion, t, {k: v * weights for k, v in losses.items() if k not in ["klc"] and k not in ["klm"]}
            )
            if "klc" in losses:
                log_loss_dict(
                    self.model.diffusion, t, {"klc": losses["klc"].mean()}, False
                )
            if "klm" in losses:
                log_loss_dict(
                    self.model.diffusion, t, {"klm": losses["klm"].mean()}, False
                )

            self.mp_trainer.backward(loss)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            # if dist.get_rank() == 0:
            logger.log(f"saving model {rate}...")
            if not rate:
                filename = f"model{(self.step + self.resume_step):06d}.pt"
            else:
                filename = f"ema_{rate}_{(self.step + self.resume_step):06d}.pt"
            with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        # if dist.get_rank() == 0:
        with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step + self.resume_step):06d}.pt"),
                "wb",
        ) as f:
            th.save(self.opt.state_dict(), f)

        # dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path

    return None


def log_loss_dict(diffusion, ts, losses, q=True):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        if q:
            for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
                quartile = int(4 * sub_t / diffusion.num_timesteps)
                logger.logkv_mean(f"{key}_q{quartile}", sub_loss)