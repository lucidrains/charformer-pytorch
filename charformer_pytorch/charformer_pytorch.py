import math
from math import gcd
import functools
import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

# helpers

def exists(val):
    return val is not None

def lcm(*numbers):
    return int(functools.reduce(lambda x, y: int((x * y) / gcd(x, y)), numbers, 1))

def masked_mean(tensor, mask, dim = -1):
    diff_len = len(tensor.shape) - len(mask.shape)
    mask = mask[(..., *((None,) * diff_len))]
    tensor.masked_fill_(~mask, 0.)

    total_el = mask.sum(dim = dim)
    mean = tensor.sum(dim = dim) / total_el.clamp(min = 1.)
    mean.masked_fill_(total_el == 0, 0.)
    return mean

def next_divisible_length(seqlen, multiple):
    return math.ceil(seqlen / multiple) * multiple

def pad_to_multiple(tensor, multiple, *, seq_dim, dim = -1, value = 0.):
    seqlen = tensor.shape[seq_dim]
    length = next_divisible_length(seqlen, multiple)
    if length == seqlen:
        return tensor
    remainder = length - seqlen
    pad_offset = (0,) * (-1 - dim) * 2
    return F.pad(tensor, (*pad_offset, 0, remainder), value = value)

# main class

class GBST(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        max_block_size = 4,
        downsample_factor = 4,
        score_consensus_attn = True
    ):
        super().__init__()
        self.pos_emb = nn.Embedding(max_block_size, dim)
        self.token_emb = nn.Embedding(num_tokens, dim)

        self.score_fn = nn.Sequential(
            nn.Linear(dim, 1),
            Rearrange('... () -> ...')
        )

        self.score_consensus_attn = score_consensus_attn

        assert downsample_factor <= max_block_size, 'final downsample factor should be less than the maximum block size'
        self.block_sizes = [*range(1, max_block_size + 1)]
        self.block_pad_multiple = lcm(*self.block_sizes)
        self.downsample_factor = downsample_factor

    def forward(self, x, mask = None):
        b, n, block_mult, ds_factor, device = *x.shape, self.block_pad_multiple, self.downsample_factor, x.device
        m = next_divisible_length(n, ds_factor)

        # get character token embeddings

        x = self.token_emb(x)

        # pad both sequence and mask to length visibile by all block sizes from 0 to max block size

        x = pad_to_multiple(x, block_mult, seq_dim = 1, dim = -2)

        if exists(mask):
            mask = pad_to_multiple(mask, block_mult, seq_dim = 1, dim = -1, value = False)

        # compute representations for all blocks by mean pooling

        block_reprs = []
        for block_size in self.block_sizes:
            pos_range = torch.arange(block_size, device = device)
            pos_emb = self.pos_emb(pos_range)

            blocks = rearrange(x, 'b (n m) d -> b n m d', m = block_size)
            pos_emb = repeat(pos_emb, 'm d -> b n m d', b = b, n = blocks.shape[1])

            blocks = blocks + pos_emb # add intra-block positional embedding

            if exists(mask):
                mask_blocks = rearrange(mask, 'b (n m) -> b n m', m = block_size)
                block_repr = masked_mean(blocks, mask_blocks, dim = -2)
            else:
                block_repr = blocks.mean(dim = -2)

            block_repr = repeat(block_repr, 'b n d -> b (n r) d', r = block_size)
            block_reprs.append(block_repr)

        block_reprs = torch.stack(block_reprs, dim = 2)

        # calculate scores and softmax across the block size dimension

        scores = self.score_fn(block_reprs)
        scores = scores.softmax(dim = 2)

        # do the cheap consensus attention, eq (5) in paper

        if self.score_consensus_attn:
            scores = einsum('b i d, b j d -> b i j', scores, scores).softmax(dim = -1) @ scores

        # multiply the block representations by the position-wise scores

        scores = rearrange(scores, 'b n m -> b n m ()')
        x = (block_reprs * scores).sum(dim = 2)

        # truncate to length divisible by downsample factor

        x = x[:, :m]

        if exists(mask):
            mask = mask[:, :m]

        # final mean pooling downsample

        x = rearrange(x, 'b (n m) d -> b n m d', m = ds_factor)

        if exists(mask):
            mask = rearrange(mask, 'b (n m) -> b n m', m = ds_factor)
            x = masked_mean(x, mask, dim = 2)
            mask = torch.any(mask, dim = -1)
        else:
            x = x.mean(dim = -2)

        return x, mask
