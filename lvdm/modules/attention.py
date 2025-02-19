import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from functools import partial
try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False
from lvdm.common import (
    checkpoint,
    exists,
    default,
)
from lvdm.basics import zero_module


class RelativePosition(nn.Module):
    """ https://github.com/evelinehong/Transformer_Relative_Position_PyTorch/blob/master/relative_position.py """

    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        device = self.embeddings_table.device
        range_vec_q = torch.arange(length_q, device=device)
        range_vec_k = torch.arange(length_k, device=device)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = final_mat.long()
        embeddings = self.embeddings_table[final_mat]
        return embeddings


class CrossAttention(nn.Module):

    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0., 
                 relative_position=False, temporal_length=None, video_length=None, image_cross_attention=False, image_cross_attention_scale=1.0, image_cross_attention_scale_learnable=False, text_context_len=77):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        
        self.relative_position = relative_position
        if self.relative_position:
            assert(temporal_length is not None)
            self.relative_position_k = RelativePosition(num_units=dim_head, max_relative_position=temporal_length)
            self.relative_position_v = RelativePosition(num_units=dim_head, max_relative_position=temporal_length)
        else:
            ## only used for spatial attention, while NOT for temporal attention
            if XFORMERS_IS_AVAILBLE and temporal_length is None:
                self.forward = self.efficient_forward

        self.video_length = video_length
        self.image_cross_attention = image_cross_attention
        self.image_cross_attention_scale = image_cross_attention_scale
        self.text_context_len = text_context_len
        self.image_cross_attention_scale_learnable = image_cross_attention_scale_learnable
        if self.image_cross_attention:
            self.to_k_ip = nn.Linear(context_dim, inner_dim, bias=False)
            self.to_v_ip = nn.Linear(context_dim, inner_dim, bias=False)
            if image_cross_attention_scale_learnable:
                self.register_parameter('alpha', nn.Parameter(torch.tensor(0.)) )


    def forward(self, x, context=None, mask=None, attn_bias=None, mode=None, use_bias=False):
        spatial_self_attn = (context is None)
        k_ip, v_ip, out_ip = None, None, None
        
        assert mode == 'temporal'

        if mask is not None:
            # indices = mask['indices']
            indices1 = mask[mode]['indices1']
            indices2 = indices1.squeeze()
            actual_indices = mask[mode]['actual_indices']
            mask = mask['mask']
        if mask is not None:
            # time_stamp = time.time()
            mask = torch.round(mask).to(torch.int) # 0.0 -> 0, 1.0 -> 1
            mask = rearrange(mask, 'b 1 t h w -> (b t) (h w)') 
            if context is not None:
                seq_len_kv = context.shape[1]
                attn_bias = attn_bias[mode]['cross']
            else:
                attn_bias = attn_bias[mode]['self']
        
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)
        if attn_bias is not None:
            b, _, _ = x.shape # (bhw, t, c) 
            # indices1 = torch.nonzero(mask.reshape(1, -1).squeeze(0))
            cat_x = x.reshape(-1, x.shape[-1])
            cat_x= torch.index_select(cat_x, 0, indices1.squeeze())
            # time_stamp = time.time()
            # print('org prepare cat_x time: ', time.time() - time_stamp)
            cat_q = self.to_q(cat_x)
            head_dim = x.shape[-1] // self.heads
            cat_q = cat_q.view(1, -1, self.heads, head_dim) 
            cat_k = k.view(1, -1, self.heads, head_dim)
            cat_v = v.view(1, -1, self.heads, head_dim)
            
            out_with_bias = xformers.ops.memory_efficient_attention(cat_q, cat_k, cat_v, 
                                                                    attn_bias=attn_bias, op=None)
            out_with_bias = out_with_bias.reshape(1, out_with_bias.shape[1], self.heads * head_dim)
            out_with_bias = self.to_out(out_with_bias)
            if use_bias:
                return out_with_bias
                # return out_with_bias + cat_x.unsqueeze(0)
            # print('attn time with mask: ', time.time() - time_stamp)
            
            # time_stamp = time.time()
            out = torch.zeros_like(x)
            q_dtype = cat_q.dtype
            out = out.to(q_dtype)
            out = out.reshape(b, out.shape[1], self.heads * self.dim_head) 
            
            out = out.reshape(-1, out.shape[-1])
            out.index_put_((indices2,), out_with_bias.squeeze())
            out = out.reshape(b, x.shape[1], self.heads * self.dim_head)
            actual_indices = actual_indices.unsqueeze(-1).expand(-1, -1, out.shape[-1])
            out = out.gather(1, actual_indices)
            return out
        
        print('org branch')
        if self.image_cross_attention and not spatial_self_attn:
            print("image_cross_attention")
            context, context_image = context[:,:self.text_context_len,:], context[:,self.text_context_len:,:]
            k = self.to_k(context)
            v = self.to_v(context)
            k_ip = self.to_k_ip(context_image)
            v_ip = self.to_v_ip(context_image)
        else:
            if not spatial_self_attn:
                context = context[:,:self.text_context_len,:]
                print("not spatial_self_attn")
            k = self.to_k(context)
            v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
        if self.relative_position:
            len_q, len_k, len_v = q.shape[1], k.shape[1], v.shape[1]
            k2 = self.relative_position_k(len_q, len_k)
            sim2 = einsum('b t d, t s d -> b t s', q, k2) * self.scale # TODO check 
            sim += sim2
        del k

        # if exists(mask):
        #     ## feasible for causal attention mask only
        #     max_neg_value = -torch.finfo(sim.dtype).max
        #     mask = repeat(mask, 'b i j -> (b h) i j', h=h)
        #     sim.masked_fill_(~(mask>0.5), max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = torch.einsum('b i j, b j d -> b i d', sim, v)
        if self.relative_position:
            v2 = self.relative_position_v(len_q, len_v)
            out2 = einsum('b t s, t s d -> b t d', sim, v2) # TODO check
            out += out2
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)


        ## for image cross-attention
        if k_ip is not None:
            k_ip, v_ip = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (k_ip, v_ip))
            sim_ip =  torch.einsum('b i d, b j d -> b i j', q, k_ip) * self.scale
            del k_ip
            sim_ip = sim_ip.softmax(dim=-1)
            out_ip = torch.einsum('b i j, b j d -> b i d', sim_ip, v_ip)
            out_ip = rearrange(out_ip, '(b h) n d -> b n (h d)', h=h)


        if out_ip is not None:
            if self.image_cross_attention_scale_learnable:
                out = out + self.image_cross_attention_scale * out_ip * (torch.tanh(self.alpha)+1)
            else:
                out = out + self.image_cross_attention_scale * out_ip
        
        return self.to_out(out)
    
    def efficient_forward(self, x, context=None, mask=None, attn_bias=None, mode=None, use_bias=False):
        spatial_self_attn = (context is None)
        k_ip, v_ip, out_ip = None, None, None
        assert mode == 'spatial'

        if mask is not None:
            # indices = mask['indices']
            indices1 = mask[mode]['indices1']
            indices2 = indices1.squeeze()
            actual_indices = mask[mode]['actual_indices']
            mask = mask['mask']
        if mask is not None:
            # time_stamp = time.time()
            mask = torch.round(mask).to(torch.int) # 0.0 -> 0, 1.0 -> 1
            mask = rearrange(mask, 'b 1 t h w -> (b t) (h w)') 
            if context is not None:
                seq_len_kv = context.shape[1]
                attn_bias = attn_bias[mode]['cross']
            else:
                attn_bias = attn_bias[mode]['self']
                
        # q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)
        if attn_bias is not None:
            b, _, _ = x.shape # (bhw, t, c) 
            # indices1 = torch.nonzero(mask.reshape(1, -1).squeeze(0))
            cat_x = x.reshape(-1, x.shape[-1])
            cat_x= torch.index_select(cat_x, 0, indices1.squeeze())
            # time_stamp = time.time()
            # print('org prepare cat_x time: ', time.time() - time_stamp)
            cat_q = self.to_q(cat_x)
            head_dim = x.shape[-1] // self.heads
            cat_q = cat_q.view(1, -1, self.heads, head_dim) 
            cat_k = k.view(1, -1, self.heads, head_dim)
            cat_v = v.view(1, -1, self.heads, head_dim)
            
            out_with_bias = xformers.ops.memory_efficient_attention(cat_q, cat_k, cat_v, 
                                                                    attn_bias=attn_bias, op=None)
            out_with_bias = out_with_bias.reshape(1, out_with_bias.shape[1], self.heads * head_dim)
            out_with_bias = self.to_out(out_with_bias)
            if use_bias:
                return out_with_bias
                # return out_with_bias + cat_x.unsqueeze(0)
            # print('attn time with mask: ', time.time() - time_stamp)
            
            # time_stamp = time.time()
            out = torch.zeros_like(x)
            q_dtype = cat_q.dtype
            out = out.to(q_dtype)
            out = out.reshape(b, out.shape[1], self.heads * self.dim_head) 
            
            out = out.reshape(-1, out.shape[-1])
            out.index_put_((indices2,), out_with_bias.squeeze())
            out = out.reshape(b, x.shape[1], self.heads * self.dim_head)
            actual_indices = actual_indices.unsqueeze(-1).expand(-1, -1, out.shape[-1])
            out = out.permute(1, 0, 2)
            out = out.gather(1, actual_indices).permute(1, 0, 2)
            return out

        print('org branch')


        if self.image_cross_attention and not spatial_self_attn:
            context, context_image = context[:,:self.text_context_len,:], context[:,self.text_context_len:,:]
            k = self.to_k(context)
            v = self.to_v(context)
            k_ip = self.to_k_ip(context_image)
            v_ip = self.to_v_ip(context_image)
        else:
            if not spatial_self_attn:
                context = context[:,:self.text_context_len,:]
            k = self.to_k(context)
            v = self.to_v(context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )
        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=None)
        
        ## for image cross-attention
        if k_ip is not None:
            k_ip, v_ip = map(
                lambda t: t.unsqueeze(3)
                .reshape(b, t.shape[1], self.heads, self.dim_head)
                .permute(0, 2, 1, 3)
                .reshape(b * self.heads, t.shape[1], self.dim_head)
                .contiguous(),
                (k_ip, v_ip),
            )
            out_ip = xformers.ops.memory_efficient_attention(q, k_ip, v_ip, attn_bias=None, op=None)
            out_ip = (
                out_ip.unsqueeze(0)
                .reshape(b, self.heads, out.shape[1], self.dim_head)
                .permute(0, 2, 1, 3)
                .reshape(b, out.shape[1], self.heads * self.dim_head)
            )

        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        if out_ip is not None:
            if self.image_cross_attention_scale_learnable:
                out = out + self.image_cross_attention_scale * out_ip * (torch.tanh(self.alpha)+1)
            else:
                out = out + self.image_cross_attention_scale * out_ip
           
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):

    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                disable_self_attn=False, attention_cls=None, video_length=None, image_cross_attention=False, image_cross_attention_scale=1.0, image_cross_attention_scale_learnable=False, text_context_len=77):
        super().__init__()
        attn_cls = CrossAttention if attention_cls is None else attention_cls
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
            context_dim=context_dim if self.disable_self_attn else None)
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head, dropout=dropout, video_length=video_length, image_cross_attention=image_cross_attention, image_cross_attention_scale=image_cross_attention_scale, image_cross_attention_scale_learnable=image_cross_attention_scale_learnable,text_context_len=text_context_len)
        self.image_cross_attention = image_cross_attention

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint


    def forward(self, x, context=None, mask=None, attn_bias=None, mode='spatial', **kwargs):
        ## implementation tricks: because checkpointing doesn't support non-tensor (e.g. None or scalar) arguments
        input_tuple = (x,)      ## should not be (x), otherwise *input_tuple will decouple x into multiple arguments
        if context is not None:
            input_tuple = (x, context)
        input_tuple = (x, context, mask, attn_bias, mode)
        # if mask is not None:
        #     forward_mask = partial(self._forward, mask=mask)
        #     return checkpoint(forward_mask, (x,), self.parameters(), self.checkpoint)
        return checkpoint(self._forward, input_tuple, self.parameters(), self.checkpoint)


    def _forward(self, x, context=None, mask=None, attn_bias=None, mode='spatial'):
        x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None, mask=mask,
                       attn_bias=attn_bias, mode=mode) + x
        if mask is not None:
        # if timesteps.cpu()[0] <= -1: # org mode
            indices1 = mask[mode]['indices1']
            indices2 = indices1.squeeze()
            actual_indices = mask[mode]['actual_indices']
            cat_x1 = self.attn2(self.norm2(x), context=context, mask=mask, attn_bias=attn_bias, mode=mode, use_bias=True)
            x2 = self.token_reuse(x, cat_x1, indices2, actual_indices, mode) + x
            cat_x2 = x2.reshape(-1, x2.shape[-1])
            cat_x2= torch.index_select(cat_x2, 0, indices1.squeeze())
            # use_bias=True 的情况下，返回的x1是 out_with_bias + cat_x，形状为(1, token_len, dim)
            cat_x3 = self.ff(self.norm3(cat_x2))
            x3 = self.token_reuse(x, cat_x3, indices2, actual_indices, mode) + x2
            
            x = x3
        else:
            x = self.attn2(self.norm2(x), context=context, mask=mask, attn_bias=attn_bias, mode=mode) + x
            x = self.ff(self.norm3(x)) + x
        # x = self.attn2(self.norm2(x), context=context, mask=mask, attn_bias=attn_bias, mode=mode) + x
        # x = self.ff(self.norm3(x)) + x
        return x
    
    def token_reuse(self, x_in, x_out, indices, actual_indices, mode):
        out = torch.zeros_like(x_in)
        out = out.reshape(-1, out.shape[-1])
        out.index_put_((indices,), x_out.squeeze())
        out = out.reshape(x_in.shape[0], x_in.shape[1], -1) 
        actual_indices = actual_indices.unsqueeze(-1).expand(-1, -1, out.shape[-1])
        if mode == 'spatial':
            out = out.permute(1, 0, 2)
            out = out.gather(1, actual_indices).permute(1, 0, 2)
        else:
            out = out.gather(1, actual_indices)
        return out


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data in spatial axis.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """

    def __init__(self, in_channels, n_heads, d_head, depth=1, dropout=0., context_dim=None,
                 use_checkpoint=True, disable_self_attn=False, use_linear=False, video_length=None,
                 image_cross_attention=False, image_cross_attention_scale_learnable=False):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        attention_cls = None
        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(
                inner_dim,
                n_heads,
                d_head,
                dropout=dropout,
                context_dim=context_dim,
                disable_self_attn=disable_self_attn,
                checkpoint=use_checkpoint,
                attention_cls=attention_cls,
                video_length=video_length,
                image_cross_attention=image_cross_attention,
                image_cross_attention_scale_learnable=image_cross_attention_scale_learnable,
                ) for d in range(depth)
        ])
        if not use_linear:
            self.proj_out = zero_module(nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
        self.use_linear = use_linear


    def forward(self, x, context=None, mask=None, attn_bias=None, **kwargs):
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context, mask=mask, attn_bias=attn_bias, mode='spatial', **kwargs)
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in
    
    
class TemporalTransformer(nn.Module):
    """
    Transformer block for image-like data in temporal axis.
    First, reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head, depth=1, dropout=0., context_dim=None,
                 use_checkpoint=True, use_linear=False, only_self_att=True, causal_attention=False, causal_block_size=1,
                 relative_position=False, temporal_length=None):
        super().__init__()
        self.only_self_att = only_self_att
        self.relative_position = relative_position
        self.causal_attention = causal_attention
        self.causal_block_size = causal_block_size

        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Conv1d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        if not use_linear:
            self.proj_in = nn.Conv1d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        if relative_position:
            assert(temporal_length is not None)
            attention_cls = partial(CrossAttention, relative_position=True, temporal_length=temporal_length)
        else:
            attention_cls = partial(CrossAttention, temporal_length=temporal_length)
        if self.causal_attention:
            assert(temporal_length is not None)
            self.mask = torch.tril(torch.ones([1, temporal_length, temporal_length]))

        if self.only_self_att:
            context_dim = None
        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(
                inner_dim,
                n_heads,
                d_head,
                dropout=dropout,
                context_dim=context_dim,
                attention_cls=attention_cls,
                checkpoint=use_checkpoint) for d in range(depth)
        ])
        if not use_linear:
            self.proj_out = zero_module(nn.Conv1d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
        self.use_linear = use_linear

    def forward(self, x, context=None, mask=None, attn_bias=None):
        b, c, t, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = rearrange(x, 'b c t h w -> (b h w) c t').contiguous()
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'bhw c t -> bhw t c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)

        temp_mask = None
        if self.causal_attention:
            # slice the from mask map
            temp_mask = self.mask[:,:t,:t].to(x.device)

        # if temp_mask is not None:
        #     mask = temp_mask.to(x.device)
        #     mask = repeat(mask, 'l i j -> (l bhw) i j', bhw=b*h*w)
        # else:
        #     mask = None

        if self.only_self_att:
            ## note: if no context is given, cross-attention defaults to self-attention
            for i, block in enumerate(self.transformer_blocks):
                x = block(x, mask=mask, attn_bias=attn_bias, mode='temporal')
            x = rearrange(x, '(b hw) t c -> b hw t c', b=b).contiguous()
        else:
            x = rearrange(x, '(b hw) t c -> b hw t c', b=b).contiguous()
            context = rearrange(context, '(b t) l con -> b t l con', t=t).contiguous()
            for i, block in enumerate(self.transformer_blocks):
                # calculate each batch one by one (since number in shape could not greater then 65,535 for some package)
                for j in range(b):
                    context_j = repeat(
                        context[j],
                        't l con -> (t r) l con', r=(h * w) // t, t=t).contiguous()
                    ## note: causal mask will not applied in cross-attention case
                    x[j] = block(x[j], context=context_j)
        
        if self.use_linear:
            x = self.proj_out(x)
            x = rearrange(x, 'b (h w) t c -> b c t h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = rearrange(x, 'b hw t c -> (b hw) c t').contiguous()
            x = self.proj_out(x)
            x = rearrange(x, '(b h w) c t -> b c t h w', b=b, h=h, w=w).contiguous()

        return x + x_in
    

class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_
