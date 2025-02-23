from functools import partial
from abc import abstractmethod
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from lvdm.models.utils_diffusion import timestep_embedding
from lvdm.common import checkpoint
from lvdm.basics import (
    zero_module,
    conv_nd,
    linear,
    avg_pool_nd,
    normalization
)
from lvdm.modules.attention import SpatialTransformer, TemporalTransformer
try:
    import xformers
    import xformers.ops
    from xformers.ops import fmha
    from xformers.ops.fmha.attn_bias import BlockDiagonalMask
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False
import time

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """
    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context=None, batch_size=None, mask=None, attn_bias=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb, batch_size=batch_size)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context, mask=mask, attn_bias=attn_bias)
            elif isinstance(layer, TemporalTransformer):
                x = rearrange(x, '(b f) c h w -> b c f h w', b=batch_size)
                x = layer(x, context, mask=mask, attn_bias=attn_bias)
                x = rearrange(x, 'b c f h w -> (b f) c h w')
            else:
                x = layer(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode='nearest')
        else:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.use_conv:
            x = self.conv(x)
        return x


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    :param use_temporal_conv: if True, use the temporal convolution.
    :param use_image_dataset: if True, the temporal parameters will not be optimized.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        use_conv=False,
        up=False,
        down=False,
        use_temporal_conv=False,
        tempspatial_aware=False
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.use_temporal_conv = use_temporal_conv

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

        if self.use_temporal_conv:
            self.temopral_conv = TemporalConvBlock(
                self.out_channels,
                self.out_channels,
                dropout=0.1,
                spatial_aware=tempspatial_aware
            )

    def forward(self, x, emb, batch_size=None):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        input_tuple = (x, emb)
        if batch_size:
            forward_batchsize = partial(self._forward, batch_size=batch_size)
            return checkpoint(forward_batchsize, input_tuple, self.parameters(), self.use_checkpoint)
        return checkpoint(self._forward, input_tuple, self.parameters(), self.use_checkpoint)

    def _forward(self, x, emb, batch_size=None):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        h = self.skip_connection(x) + h

        if self.use_temporal_conv and batch_size:
            h = rearrange(h, '(b t) c h w -> b c t h w', b=batch_size)
            h = self.temopral_conv(h)
            h = rearrange(h, 'b c t h w -> (b t) c h w')
        return h


class TemporalConvBlock(nn.Module):
    """
    Adapted from modelscope: https://github.com/modelscope/modelscope/blob/master/modelscope/models/multi_modal/video_synthesis/unet_sd.py
    """
    def __init__(self, in_channels, out_channels=None, dropout=0.0, spatial_aware=False):
        super(TemporalConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        th_kernel_shape = (3, 1, 1) if not spatial_aware else (3, 3, 1)
        th_padding_shape = (1, 0, 0) if not spatial_aware else (1, 1, 0)
        tw_kernel_shape = (3, 1, 1) if not spatial_aware else (3, 1, 3)
        tw_padding_shape = (1, 0, 0) if not spatial_aware else (1, 0, 1)

        # conv layers
        self.conv1 = nn.Sequential(
            nn.GroupNorm(32, in_channels), nn.SiLU(),
            nn.Conv3d(in_channels, out_channels, th_kernel_shape, padding=th_padding_shape))
        self.conv2 = nn.Sequential(
            nn.GroupNorm(32, out_channels), nn.SiLU(), nn.Dropout(dropout),
            nn.Conv3d(out_channels, in_channels, tw_kernel_shape, padding=tw_padding_shape))
        self.conv3 = nn.Sequential(
            nn.GroupNorm(32, out_channels), nn.SiLU(), nn.Dropout(dropout),
            nn.Conv3d(out_channels, in_channels, th_kernel_shape, padding=th_padding_shape))
        self.conv4 = nn.Sequential(
            nn.GroupNorm(32, out_channels), nn.SiLU(), nn.Dropout(dropout),
            nn.Conv3d(out_channels, in_channels, tw_kernel_shape, padding=tw_padding_shape))

        # zero out the last layer params,so the conv block is identity
        nn.init.zeros_(self.conv4[-1].weight)
        nn.init.zeros_(self.conv4[-1].bias)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        return identity + x

class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: in_channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(self,
                 in_channels,
                 model_channels,
                 out_channels,
                 num_res_blocks,
                 attention_resolutions,
                 dropout=0.0,
                 channel_mult=(1, 2, 4, 8),
                 conv_resample=True,
                 dims=2,
                 context_dim=None,
                 use_scale_shift_norm=False,
                 resblock_updown=False,
                 num_heads=-1,
                 num_head_channels=-1,
                 transformer_depth=1,
                 use_linear=False,
                 use_checkpoint=False,
                 temporal_conv=False,
                 tempspatial_aware=False,
                 temporal_attention=True,
                 use_relative_position=True,
                 use_causal_attention=False,
                 temporal_length=None,
                 use_fp16=False,
                 addition_attention=False,
                 temporal_selfatt_only=True,
                 image_cross_attention=False,
                 image_cross_attention_scale_learnable=False,
                 default_fs=4,
                 fs_condition=False,
                ):
        super(UNetModel, self).__init__()
        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'
        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.temporal_attention = temporal_attention
        time_embed_dim = model_channels * 4
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        temporal_self_att_only = True
        self.addition_attention = addition_attention
        self.temporal_length = temporal_length
        self.image_cross_attention = image_cross_attention
        self.image_cross_attention_scale_learnable = image_cross_attention_scale_learnable
        self.default_fs = default_fs
        self.fs_condition = fs_condition

        self.mask_percentages = []
        self.sampling_step = 0
        self.efficient = True
        # self.efficient = False
        ## Time embedding blocks
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        if fs_condition:
            self.fps_embedding = nn.Sequential(
                linear(model_channels, time_embed_dim),
                nn.SiLU(),
                linear(time_embed_dim, time_embed_dim),
            )
            nn.init.zeros_(self.fps_embedding[-1].weight)
            nn.init.zeros_(self.fps_embedding[-1].bias)
        ## Input Block
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(conv_nd(dims, in_channels, model_channels, 3, padding=1))
            ]
        )
        if self.addition_attention:
            self.init_attn=TimestepEmbedSequential(
                TemporalTransformer(
                    model_channels,
                    n_heads=8,
                    d_head=num_head_channels,
                    depth=transformer_depth,
                    context_dim=context_dim,
                    use_checkpoint=use_checkpoint, only_self_att=temporal_selfatt_only, 
                    causal_attention=False, relative_position=use_relative_position, 
                    temporal_length=temporal_length))

        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(ch, time_embed_dim, dropout,
                        out_channels=mult * model_channels, dims=dims, use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm, tempspatial_aware=tempspatial_aware,
                        use_temporal_conv=temporal_conv
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    layers.append(
                        SpatialTransformer(ch, num_heads, dim_head, 
                            depth=transformer_depth, context_dim=context_dim, use_linear=use_linear,
                            use_checkpoint=use_checkpoint, disable_self_attn=False, 
                            video_length=temporal_length, image_cross_attention=self.image_cross_attention,
                            image_cross_attention_scale_learnable=self.image_cross_attention_scale_learnable,                      
                        )
                    )
                    if self.temporal_attention:
                        layers.append(
                            TemporalTransformer(ch, num_heads, dim_head,
                                depth=transformer_depth, context_dim=context_dim, use_linear=use_linear,
                                use_checkpoint=use_checkpoint, only_self_att=temporal_self_att_only, 
                                causal_attention=use_causal_attention, relative_position=use_relative_position, 
                                temporal_length=temporal_length
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(ch, time_embed_dim, dropout, 
                            out_channels=out_ch, dims=dims, use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True
                        )
                        if resblock_updown
                        else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        layers = [
            ResBlock(ch, time_embed_dim, dropout,
                dims=dims, use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm, tempspatial_aware=tempspatial_aware,
                use_temporal_conv=temporal_conv
            ),
            SpatialTransformer(ch, num_heads, dim_head, 
                depth=transformer_depth, context_dim=context_dim, use_linear=use_linear,
                use_checkpoint=use_checkpoint, disable_self_attn=False, video_length=temporal_length, 
                image_cross_attention=self.image_cross_attention,image_cross_attention_scale_learnable=self.image_cross_attention_scale_learnable                
            )
        ]
        if self.temporal_attention:
            layers.append(
                TemporalTransformer(ch, num_heads, dim_head,
                    depth=transformer_depth, context_dim=context_dim, use_linear=use_linear,
                    use_checkpoint=use_checkpoint, only_self_att=temporal_self_att_only, 
                    causal_attention=use_causal_attention, relative_position=use_relative_position, 
                    temporal_length=temporal_length
                )
            )
        layers.append(
            ResBlock(ch, time_embed_dim, dropout,
                dims=dims, use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm, tempspatial_aware=tempspatial_aware, 
                use_temporal_conv=temporal_conv
                )
        )

        ## Middle Block
        self.middle_block = TimestepEmbedSequential(*layers)

        ## Output Block
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(ch + ich, time_embed_dim, dropout,
                        out_channels=mult * model_channels, dims=dims, use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm, tempspatial_aware=tempspatial_aware,
                        use_temporal_conv=temporal_conv
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    layers.append(
                        SpatialTransformer(ch, num_heads, dim_head, 
                            depth=transformer_depth, context_dim=context_dim, use_linear=use_linear,
                            use_checkpoint=use_checkpoint, disable_self_attn=False, video_length=temporal_length,
                            image_cross_attention=self.image_cross_attention,image_cross_attention_scale_learnable=self.image_cross_attention_scale_learnable    
                        )
                    )
                    if self.temporal_attention:
                        layers.append(
                            TemporalTransformer(ch, num_heads, dim_head,
                                depth=transformer_depth, context_dim=context_dim, use_linear=use_linear,
                                use_checkpoint=use_checkpoint, only_self_att=temporal_self_att_only, 
                                causal_attention=use_causal_attention, relative_position=use_relative_position, 
                                temporal_length=temporal_length
                            )
                        )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(ch, time_embed_dim, dropout,
                            out_channels=out_ch, dims=dims, use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

    def forward(self, x, timesteps, context=None, features_adapter=None, fs=None, **kwargs):
        
        self.sampling_step += 1
        print('sampling_step:', self.sampling_step)
        # y = self.forward_org(x, timesteps, context, features_adapter, fps, **kwargs)
        # return y
        
        # mask = self.compute_similarity_mask(x, threshold=0.95)
        # mask = None
        # mask = self.batched_find_idxs_to_keep(x, threshold=0.5, tubelet_size=1, patch_size=1)
        print('x shape:', x.shape)
        if timesteps <= 400:
            mask = self.batched_find_idxs_to_keep(x, threshold=0.1, tubelet_size=1, patch_size=1)
        else:
            mask = self.batched_find_idxs_to_keep(x, threshold=0.025, tubelet_size=1, patch_size=1)
        print('------------------')
        print('timesteps',  timesteps)
        total_tokens = mask.numel()
        filtered_tokens = (mask == 0).sum().item()
        filtered_percentage = 100.0 * filtered_tokens / total_tokens
        print(f"Mask Filtering: {filtered_percentage:.2f}% tokens filtered")
        self.mask_percentages.append(filtered_percentage)       
        
        b,_,t,_,_ = x.shape
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False).type(x.dtype)
        emb = self.time_embed(t_emb)
        
        ## repeat t times for context [(b t) 77 768] & time embedding
        ## check if we use per-frame image conditioning
        _, l_context, _ = context.shape
        if l_context == 77 + t*16: ## !!! HARD CODE here
            context_text, context_img = context[:,:77,:], context[:,77:,:]
            context_text = context_text.repeat_interleave(repeats=t, dim=0)
            context_img = rearrange(context_img, 'b (t l) c -> (b t) l c', t=t)
            context = torch.cat([context_text, context_img], dim=1)
        else:
            context = context.repeat_interleave(repeats=t, dim=0)
        emb = emb.repeat_interleave(repeats=t, dim=0)
        
        # get mask_dict and attn_bias_dict
        mask_dict = {}
        attn_bias_dict = {}
        
        cur_h, cur_w = x.shape[-2:]
        org_h, org_w = cur_h, cur_w
        for i in range(4):
            mask_dict[(cur_h, cur_w)] = {}
            mask_dict[(cur_h, cur_w)]['mask'] = mask
            mask_dict[(cur_h, cur_w)]['spatial'] = {}
            mask_dict[(cur_h, cur_w)]['temporal'] = {}
            mask_dict = self.compute_mask_dict_spatial(mask_dict, cur_h, cur_w)
            mask_dict = self.compute_mask_dict_temporal(mask_dict, cur_h, cur_w)
            # spatial mode
            seq_len_q_spatial = cur_h * cur_w # h * w
            seq_len_q_temporal = x.shape[2] # t
            if context is not None:
                seq_len_kv_spatial = context.shape[1]
                seq_len_kv_temporal = context.shape[1]
            else:
                seq_len_kv_spatial = seq_len_q_spatial
                seq_len_kv_temporal = seq_len_q_temporal
            attn_bias_dict[(cur_h, cur_w)] = {}
            attn_bias_dict[(cur_h, cur_w)]['spatial'] = {}
            attn_bias_dict[(cur_h, cur_w)]['temporal'] = {}
            attn_bias_dict[(cur_h, cur_w)]['spatial']['self'], _, _ = create_block_diagonal_attention_mask(mask, seq_len_q_spatial, mode='spatial')
            attn_bias_dict[(cur_h, cur_w)]['temporal']['self'], _, _ = create_block_diagonal_attention_mask(mask, seq_len_q_temporal, mode='temporal')
            attn_bias_dict[(cur_h, cur_w)]['spatial']['cross'], _, _ = create_block_diagonal_attention_mask(mask, seq_len_kv_spatial, mode='spatial')
            attn_bias_dict[(cur_h, cur_w)]['temporal']['cross'], _, _ = create_block_diagonal_attention_mask(mask, seq_len_kv_temporal, mode ='temporal')
            # attn_bias_dict[(cur_h, cur_w)]['spatial']['self'], _, _ = create_fake_block_diagonal_attention_mask(mask, seq_len_q)
            # attn_bias_dict[(cur_h, cur_w)]['temporal']['self'], _, _ = create_fake_block_diagonal_attention_mask(mask, seq_len_q)
            # attn_bias_dict[(cur_h, cur_w)]['spatial']['cross'], _, _ = create_fake_block_diagonal_attention_mask(mask, seq_len_kv)
            # attn_bias_dict[(cur_h, cur_w)]['temporal']['cross'], _, _ = create_fake_block_diagonal_attention_mask(mask, seq_len_kv)
            cur_h, cur_w = cur_h//2, cur_w//2
            mask = self.resize_mask(mask, cur_h, cur_w)
        ## always in shape (b t) c h w, except for temporal layer
        x = rearrange(x, 'b c t h w -> (b t) c h w')

        ## combine emb
        if self.fs_condition:
            if fs is None:
                fs = torch.tensor(
                    [self.default_fs] * b, dtype=torch.long, device=x.device)
            fs_emb = timestep_embedding(fs, self.model_channels, repeat_only=False).type(x.dtype)

            fs_embed = self.fps_embedding(fs_emb)
            fs_embed = fs_embed.repeat_interleave(repeats=t, dim=0)
            emb = emb + fs_embed

        h = x.type(self.dtype)
        adapter_idx = 0
        hs = []
        for id, module in enumerate(self.input_blocks):
            new_h, new_w = h.shape[-2:]
            if self.efficient:
                h = module(h, emb, context=context, batch_size=b, mask=mask_dict[(new_h, new_w)], attn_bias=attn_bias_dict[(new_h, new_w)])
            else:
                h = module(h, emb, context=context, batch_size=b)
            if id ==0 and self.addition_attention:
                if self.efficient:
                    h = self.init_attn(h, emb, context=context, batch_size=b, mask=mask_dict[(new_h, new_w)], attn_bias=attn_bias_dict[(new_h, new_w)])
                else:
                    h = self.init_attn(h, emb, context=context, batch_size=b)
            ## plug-in adapter features
            if ((id+1)%3 == 0) and features_adapter is not None:
                h = h + features_adapter[adapter_idx]
                adapter_idx += 1
            hs.append(h)
        if features_adapter is not None:
            assert len(features_adapter)==adapter_idx, 'Wrong features_adapter'

        if self.efficient:
            new_h, new_w = h.shape[-2:]
            h = self.middle_block(h, emb, context=context, batch_size=b, mask=mask_dict[(new_h, new_w)], attn_bias=attn_bias_dict[(new_h, new_w)])
        else:
            h = self.middle_block(h, emb, context=context, batch_size=b)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            new_h, new_w = h.shape[-2:]
            if self.efficient:
                h = module(h, emb, context=context, batch_size=b, mask=mask_dict[(new_h, new_w)], attn_bias=attn_bias_dict[(new_h, new_w)])
            else:
                h = module(h, emb, context=context, batch_size=b)
        h = h.type(x.dtype)
        y = self.out(h)
        
        # reshape back to (b c t h w)
        y = rearrange(y, '(b t) c h w -> b c t h w', b=b)
        return y
    
    def forward_org(self, x, timesteps, context=None, features_adapter=None, fs=None, **kwargs):
        b,_,t,_,_ = x.shape
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False).type(x.dtype)
        emb = self.time_embed(t_emb)
        
        ## repeat t times for context [(b t) 77 768] & time embedding
        ## check if we use per-frame image conditioning
        _, l_context, _ = context.shape
        if l_context == 77 + t*16: ## !!! HARD CODE here
            context_text, context_img = context[:,:77,:], context[:,77:,:]
            context_text = context_text.repeat_interleave(repeats=t, dim=0)
            context_img = rearrange(context_img, 'b (t l) c -> (b t) l c', t=t)
            context = torch.cat([context_text, context_img], dim=1)
        else:
            context = context.repeat_interleave(repeats=t, dim=0)
        emb = emb.repeat_interleave(repeats=t, dim=0)
        
        ## always in shape (b t) c h w, except for temporal layer
        x = rearrange(x, 'b c t h w -> (b t) c h w')

        ## combine emb
        if self.fs_condition:
            if fs is None:
                fs = torch.tensor(
                    [self.default_fs] * b, dtype=torch.long, device=x.device)
            fs_emb = timestep_embedding(fs, self.model_channels, repeat_only=False).type(x.dtype)

            fs_embed = self.fps_embedding(fs_emb)
            fs_embed = fs_embed.repeat_interleave(repeats=t, dim=0)
            emb = emb + fs_embed

        h = x.type(self.dtype)
        adapter_idx = 0
        hs = []
        for id, module in enumerate(self.input_blocks):
            h = module(h, emb, context=context, batch_size=b)
            if id ==0 and self.addition_attention:
                h = self.init_attn(h, emb, context=context, batch_size=b)
            ## plug-in adapter features
            if ((id+1)%3 == 0) and features_adapter is not None:
                h = h + features_adapter[adapter_idx]
                adapter_idx += 1
            hs.append(h)
        if features_adapter is not None:
            assert len(features_adapter)==adapter_idx, 'Wrong features_adapter'

        h = self.middle_block(h, emb, context=context, batch_size=b)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context=context, batch_size=b)
        h = h.type(x.dtype)
        y = self.out(h)
        
        # reshape back to (b c t h w)
        y = rearrange(y, '(b t) c h w -> b c t h w', b=b)
        return y
    
    def batched_find_idxs_to_keep(self, 
                                x: torch.Tensor, 
                              threshold: int=2, 
                              tubelet_size: int=2,
                              patch_size: int=16) -> torch.Tensor:
        """
        Find the static tokens in a video tensor, and return a mask
        that selects tokens that are not repeated.

        Args:
        - x (torch.Tensor): A tensor of shape [B, C, T, H, W].
        - threshold (int): The mean intensity threshold for considering
                a token as static.
        - tubelet_size (int): The temporal length of a token.
        Returns:
        - mask (torch.Tensor): A bool tensor of shape [B, T, H, W] 
            that selects tokens that are not repeated.

        """
        # Ensure input has the format [B, C, T, H, W]
        assert len(x.shape) == 5, "Input must be a 5D tensor"
        #ipdb.set_trace()
        # Convert to float32 if not already
        x = x.type(torch.float32)
        
        # Calculate differences between frames with a step of tubelet_size, ensuring batch dimension is preserved
        # Compare "front" of first token to "back" of second token
        diffs = x[:, :, (2*tubelet_size-1)::tubelet_size] - x[:, :, :-tubelet_size:tubelet_size]
        # Ensure nwe track negative movement.
        diffs = torch.abs(diffs)
        
        # Apply average pooling over spatial dimensions while keeping the batch dimension intact
        avg_pool_blocks = F.avg_pool3d(diffs, (1, patch_size, patch_size))
        # Compute the mean along the channel dimension, preserving the batch dimension
        avg_pool_blocks = torch.mean(avg_pool_blocks, dim=1, keepdim=True)
        # Create a dummy first frame for each item in the batch
        first_frame = torch.ones_like(avg_pool_blocks[:, :, 0:1]) * 255
        # first_frame = torch.zeros_like(avg_pool_blocks[:, :, 0:1])
        # Concatenate the dummy first frame with the rest of the frames, preserving the batch dimension
        avg_pool_blocks = torch.cat([first_frame, avg_pool_blocks], dim=2)
        # Determine indices to keep based on the threshold, ensuring the operation is applied across the batch
        # Update mask: 0 for high similarity, 1 for low similarity
        keep_idxs = avg_pool_blocks.squeeze(1) > threshold  
        keep_idxs = keep_idxs.unsqueeze(1)
        keep_idxs = keep_idxs.float()
        # Flatten out everything but the batch dimension
        # keep_idxs = keep_idxs.flatten(1)
        #ipdb.set_trace()
        return keep_idxs

    def compute_similarity_mask(self, latent, threshold=0.95):
        """
        Compute frame-wise similarity for latent and generate mask.

        Args:
        - latent (torch.Tensor): Latent tensor of shape [n, c, t, h, w].
        - threshold (float): Similarity threshold to determine whether to skip computation.

        Returns:
        - mask (torch.Tensor): Mask tensor of shape [n, 1, t, h, w],
        where mask = 0 means skip computation, mask = 1 means recompute.
        """
        n, c, t, h, w = latent.shape
        mask = torch.ones((n, 1, t, h, w), device=latent.device)  # Initialize mask with all 1s

        for frame_idx in range(1, t):  # Start from the second frame
            curr_frame = latent[:, :, frame_idx, :, :]  # Current frame [n, c, h, w]
            prev_frame = latent[:, :, frame_idx - 1, :, :]  # Previous frame [n, c, h, w]

            # Compute token-wise cosine similarity
            dot_product = (curr_frame * prev_frame).sum(dim=1, keepdim=True)  # [n, 1, h, w]
            norm_curr = curr_frame.norm(dim=1, keepdim=True)
            norm_prev = prev_frame.norm(dim=1, keepdim=True)
            similarity = dot_product / (norm_curr * norm_prev + 1e-8)  # Avoid division by zero

            # Update mask: 0 for high similarity, 1 for low similarity
            mask[:, :, frame_idx, :, :] = (similarity <= threshold).float()
        # mask = torch.round(mask).to(torch.int) # 0.0 -> 0, 1.0 -> 1
        return mask
    
    def resize_mask(self, mask, target_h, target_w):
        """
        Resize the mask to match the new spatial dimensions of x.

        Args:
        - mask (torch.Tensor): Input mask of shape [b, 1, t, h, w].
        - target_h (int): Target height.
        - target_w (int): Target width.

        Returns:
        - resized_mask (torch.Tensor): Resized mask of shape [b, 1, t, target_h, target_w].
        """
        if mask is None:
            return mask
        batch, _, t, h, w = mask.shape

        if h == target_h and w == target_w:
            return mask  # No resizing needed

        # Reshape to [b * t, 1, h, w]
        mask = mask.view(batch * t, 1, h, w)

        # Resize to [b * t, 1, target_h, target_w]
        resized_mask = F.interpolate(mask, size=(target_h, target_w), mode="bilinear", align_corners=False)

        # Ensure the mask is binary (0 or 1)
        resized_mask = (resized_mask > 0.5).float()

        # Reshape back to [b, 1, t, target_h, target_w]
        resized_mask = resized_mask.view(batch, 1, t, target_h, target_w)

        return resized_mask
    
    def compute_mask_dict_spatial(self, mask_dict, cur_h, cur_w):
        mask = mask_dict[(cur_h, cur_w)]['mask']
        indices = []
        _mask = torch.round(mask).to(torch.int) # 0.0 -> 0, 1.0 -> 1
        indices1 = torch.nonzero(_mask.reshape(1, -1).squeeze(0))
        _mask = rearrange(_mask, 'b 1 t h w -> (b t) (h w)')
        # for i in range(_mask.size(0)):
        #     index_per_batch = torch.where(_mask[i].bool())[0]
        #     indices.append(index_per_batch)
        mask_dict[(cur_h, cur_w)]['spatial']['indices'] = indices
        mask_dict[(cur_h, cur_w)]['spatial']['indices1'] = indices1
        mask_bool = _mask.bool()
        mask_bool = mask_bool.T
        device = mask.device
        batch_size, seq_len = mask_bool.shape
        # print('------------------')
        time_stamp = time.time()
        arange_indices = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1).to(device)
        # print('time for arange_indices:', time.time()-time_stamp)
        time_stamp = time.time()
        nonzero_indices = torch.nonzero(mask_bool, as_tuple=True)
        valid_indices = torch.zeros_like(arange_indices)
        valid_indices[nonzero_indices[0], torch.cumsum(mask_bool.int(), dim=1)[mask_bool] - 1] = arange_indices[mask_bool]
        cumsum_mask = torch.cumsum(mask_bool.int(), dim=1)
        # print('time for cumsum_mask:', time.time()-time_stamp)
        time_stamp = time.time()
        nearest_indices = torch.clip(cumsum_mask - 1, min=0)
        # print('time for nearest_indices:', time.time()-time_stamp)
        time_stamp = time.time()
        actual_indices = valid_indices.gather(1, nearest_indices)
        mask_dict[(cur_h, cur_w)]['spatial']['actual_indices'] = actual_indices
        return mask_dict
        
    def compute_mask_dict_temporal(self, mask_dict, cur_h, cur_w):
        mask = mask_dict[(cur_h, cur_w)]['mask']
        indices = []
        _mask = torch.round(mask).to(torch.int)
        _mask = rearrange(_mask, 'b 1 t h w -> (b h w) (t)')
        indices1 = torch.nonzero(_mask.reshape(1, -1).squeeze(0))
        mask_dict[(cur_h, cur_w)]['temporal']['indices'] = indices
        mask_dict[(cur_h, cur_w)]['temporal']['indices1'] = indices1
        mask_bool = _mask.bool()
        device = mask.device
        batch_size, seq_len = mask_bool.shape
        arange_indices = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1).to(device)
        nonzero_indices = torch.nonzero(mask_bool, as_tuple=True)
        valid_indices = torch.zeros_like(arange_indices)
        valid_indices[nonzero_indices[0], torch.cumsum(mask_bool.int(), dim=1)[mask_bool] - 1] = arange_indices[mask_bool]
        cumsum_mask = torch.cumsum(mask_bool.int(), dim=1)
        nearest_indices = torch.clip(cumsum_mask - 1, min=0)
        actual_indices = valid_indices.gather(1, nearest_indices)
        mask_dict[(cur_h, cur_w)]['temporal']['actual_indices'] = actual_indices
        return mask_dict
    
def create_fake_block_diagonal_attention_mask(mask, kv_seqlen):
    """
    将 mask 和 kv_seqlen 转换为 BlockDiagonalMask，用于高效的注意力计算。
    
    Args:
        mask (torch.Tensor): 输入的掩码，标记哪些 token 应该被忽略。
        kv_seqlen (torch.Tensor): 键/值的序列长度。
        heads (int): 注意力头的数量。

    Returns:
        BlockDiagonalMask: 转换后的注意力掩码，用于高效的计算。
    """
    # 计算 q_seqlen: 通过 mask 来提取有效的查询 token 数量
    # mask = rearrange(mask, 'b 1 t h w -> (b t) (h w)') 
    mask = torch.round(mask).to(torch.int) # 0.0 -> 0, 1.0 -> 1
    mask = rearrange(mask, 'b 1 t h w -> (b t) (h w)') 
    # q_seqlen = mask.sum(dim=-1) 
    num = mask.shape[0]
    lengh = mask.shape[1]
    q_seqlen = [int(lengh / 4)] * num
    
    kv_seqlen = [kv_seqlen] * len(q_seqlen)  # 重复 kv_seqlen 次

    # 生成 BlockDiagonalPaddedKeysMask
    attn_bias = BlockDiagonalMask.from_seqlens(
        q_seqlen,  
        kv_seqlen=kv_seqlen,  # 键/值的序列长度
    )
    
    return attn_bias, q_seqlen, kv_seqlen

def create_block_diagonal_attention_mask(mask, kv_seqlen, mode='spatial'):
    """
    将 mask 和 kv_seqlen 转换为 BlockDiagonalMask，用于高效的注意力计算。
    
    Args:
        mask (torch.Tensor): 输入的掩码，标记哪些 token 应该被忽略。
        kv_seqlen (torch.Tensor): 键/值的序列长度。
        heads (int): 注意力头的数量。

    Returns:
        BlockDiagonalPaddedKeysMask: 转换后的注意力掩码，用于高效的计算。
    """
    # 计算 q_seqlen: 通过 mask 来提取有效的查询 token 数量
    mask = torch.round(mask).to(torch.int) # 0.0 -> 0, 1.0 -> 1
    if mode == 'spatial':
        mask = rearrange(mask, 'b 1 t h w -> (b t) (h w)')
    else:
        mask = rearrange(mask, 'b 1 t h w -> (b h w) (t)')
    
    q_seqlen = mask.sum(dim=-1)  # 计算每个批次中有效的查询 token 数量
    q_seqlen = q_seqlen.tolist()
    
    kv_seqlen = [kv_seqlen] * len(q_seqlen)  # 重复 kv_seqlen 次

    # 生成 BlockDiagonalPaddedKeysMask
    attn_bias = BlockDiagonalMask.from_seqlens(
        q_seqlen,  
        kv_seqlen=kv_seqlen,  # 键/值的序列长度
    )
    
    return attn_bias, q_seqlen, kv_seqlen