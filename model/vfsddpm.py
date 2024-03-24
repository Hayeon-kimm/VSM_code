import torch
import torch as th
import torch.distributions as td
import torch.nn.functional as F
from torch import nn  # , einsum, rearrange
import numpy as np
from model.set_diffusion.nn import SiLU, timestep_embedding
from model.set_diffusion.script_util import (
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from model.vit import ViT
from model.set_diffusion.nn import mean_flat
from model.memory_vit import *
def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)

class SiLU(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)
    
def log(t, eps=1e-20):
    return torch.log(t + eps)

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature=1., dim=-1):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim=dim)

def top_k(logits, thres=0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

class VFSDDPM(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.ns = args.sample_size
        self.bs = args.batch_size
        self.patch_size = args.patch_size
        self.image_size = args.image_size
        self.encoder_mode = args.encoder_mode
        self.hdim = args.hdim
        self.mode_conditioning = args.mode_conditioning
        self.mode_context = args.mode_context
        self.memory = Memory(args.hdim)
        self.postpool = nn.Sequential(
            SiLU(),
            linear(
                self.hdim*2,
                self.hdim,
            ),
        )
        self.encoder = ViT(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_classes=args.hdim,  # not important for the moment
            dim=args.hdim,
            pool=args.pool,  # use avg patch_avg
            channels=args.in_channels,
            dropout=args.dropout,
            emb_dropout=args.dropout,
            depth=6,
            heads=12,
            mlp_dim=args.hdim,
            ns=self.ns,
        )
        self.posterior = nn.Sequential(
            nn.Linear(args.hdim, args.hdim),
            SiLU(),
            nn.Linear(
                args.hdim, 2 * args.hdim
            ),
        )
        self.posterior_m = nn.Sequential(
            nn.Linear(args.hdim, args.hdim),
            SiLU(),
            nn.Linear(
                args.hdim, 2 * args.hdim
            ),
        )
        self.posterior_c = nn.Sequential(
            nn.Linear(args.hdim*2, args.hdim*2),
            SiLU(),
            nn.Linear(
                args.hdim*2, 2 * args.hdim
            ),
        )
        self.prior = nn.Sequential(
            nn.Linear(args.hdim, args.hdim),
            SiLU(),
            nn.Linear(
                args.hdim, 2 * args.hdim
            ),
        )
        self.generative_model, self.diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )


    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def forward_c(self, x_set, label=None, t=None, tag='test'):
        bs, ns, ch, h, w = x_set.size()
        t_emb = None
        if t is not None:
            t_emb = self.generative_model.time_embed(timestep_embedding(t))
        out = self.encoder.forward_set(x_set, t_emb=t_emb, c_old=None)
        hc = out["hc"]
        if tag == 'train':
            self.memory.memory_update(hc, hc, label, 0.3)
        mc = self.memory(hc, label, tag)
        hc = torch.cat((mc, hc), dim=-1)
        hc = self.postpool(hc)
        qm, qv = self.posterior_m(hc).chunk(2, dim=2)
        zeros = torch.zeros(qm.size()).to(qm.device)
        ones = torch.ones(qm.size()).to(qm.device)
        mqd = self.normal(qm, qv)
        m = mqd.rsample()
        mpd = self.normal(zeros, ones)
        cqm, cqv = self.posterior_c(torch.cat((m, out["hc"]), dim=-1)).chunk(2, dim=2)
        cpm, cpv = self.prior(m).chunk(2, dim=2)
        cqd = self.normal(cqm, cqv)
        cpd = self.normal(cpm, cpv)
        c = cqd.rsample()
        return {"c": c, "cqd": cqd, "cpd": cpd, "qm": cqm, "mqd": mqd, "mpd": mpd}


    def normal(self, loc: torch.Tensor, log_var: torch.Tensor, temperature=None):
        log_std = log_var / 2
        # if temperature:
        #     log_std = log_std * temperature
        scale = torch.exp(log_std)
        distro = td.Normal(loc=loc, scale=scale)
        return distro

    def forward(self, batch, label, t, tag='test'):
        """
        forward input set X, compute c and condition ddpm on c.
        """
        bs, ns, ch, h, w = batch.size()

        c_list = []
        for i in range(batch.shape[1]):
            ix = torch.LongTensor([k for k in range(batch.shape[1]) if k != i])
            x_set_tmp = batch[:, ix]

            out = self.forward_c(x_set_tmp, label, t, tag)
            c_set_tmp = out["c"]
            c_list.append(c_set_tmp.unsqueeze(1))

        c_set = torch.cat(c_list, dim=1)
        x = batch.view(-1, ch, self.image_size, self.image_size)

        if self.mode_conditioning == "lag":
            # (b*ns, np, dim)
            c = c_set.view(-1, c_set.size(-2), c_set.size(-1))
        else:
            # (b*ns, dim)
            c = c_set.view(-1, c_set.size(-1))

        losses = self.diffusion.training_losses(self.generative_model, x, t, c)
        losses["klc"] = self.loss_c(out)
        losses["klm"] = self.loss_m(out)
        return losses

    def loss_c(self, out):
        """
        compute the KL between two normal distribution.
        Here the context c is a continuous vector.
        """
        klc = td.kl_divergence(out['cqd'], out['cpd'])
        klc = mean_flat(klc) / np.log(2.0)
        return klc
    def loss_m(self, out):
        """
        compute the KL between two normal distribution.
        Here the context c is a continuous vector.
        """
        klc = td.kl_divergence(out['mqd'], out['mpd'])
        klc = mean_flat(klc) / np.log(2.0)
        return klc

    def loss_c_discrete(self, out):
        """
        Compute the KL between two categorical distributions.
        Here c is a set of vectors each representing the logits for the codebook.
        """
        log_qy = F.log_softmax(out["logits"], dim=-1)
        log_uniform = torch.log(torch.tensor([1. / log_qy.shape[-1]], device=log_qy.device))
        klc = F.kl_div(log_uniform, log_qy, None, None, 'none', log_target=True)
        klc = mean_flat(klc) / np.log(2.0)
        return klc

    def sample_conditional(self, x_set, sample_size, k=1):
        out = self.forward_c(x_set, None)  # improve with per-layer conditioning using t
        c_set = out["qm"]
        if self.mode_conditioning == "lag":
            # (b*ns, np, dim)
            c = c_set.view(-1, c_set.size(-2), c_set.size(-1))
        else:
            # (b*ns, dim)
            c = c_set.view(-1, c_set.size(-1))
        klc = self.loss_c(out)
        return {"c": c, "qm": out["qm"], "klc": klc}
