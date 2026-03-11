import math
from typing import Optional, Union

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as mnp
import mindspore.mint as mint
import mindspore.mint.nn.functional as F

import mindspore.nn as nn

class BlockBase(nn.Cell): 
    """Block abstract module"""

    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__() 
        self.input_size = input_size
        self.output_size = output_size

def get_activation(name: str = "relu") -> nn.Cell:
    """Select an activation function by name (MindSpore版)

    Args:
        name: str
            activation function name,
            one of ["relu", "gelu", "swish", "sigmoid"],
            default "relu".
    """
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "swish":
        return Swish()
    if name == "sigmoid":
        return nn.Sigmoid()
    return mint.nn.Identity()


def adaptive_enc_mask(x_len: int,
                      chunk_start_idx: list[int],
                      left_window: int = 0,
                      right_window: int = 0) -> ms.Tensor:
    """
    The function is very important for Transformer Transducer Streaming mode
    Args:
        x_len: sequence length
        chunk_start_idx: first idx of each chunk, such as [0,18,36,48]. 
        It also supports adaptive chunk size [0,10,15,45]
        left_window: how many left chunks can be seen
        right_window: how many right chunks can be seen. It is used for 
        chunk overlap model.
    Returns:
        mask (ms.Tensor): a mask tensor for streaming model
    """
    chunk_start_idx = ms.Tensor(chunk_start_idx, dtype=ms.int64)
    start_pad = F.pad(chunk_start_idx,(1, 0))  # append 0 to the beginning, so it becomes [0, 0, 18, 36, 48]
    end_pad = F.pad(chunk_start_idx, (0, 1), value=x_len)  # append x_len to the end, so it becomes [0,18,36,48, x_len]
    seq_range = mint.unsqueeze(mint.arange(0, x_len), -1) 
    mask1 = seq_range < end_pad
    mask2 = seq_range >= start_pad
    mask = ops.logical_and(mask1, mask2)
    idx = mint.nonzero(mask, as_tuple=False)[:, 1]  # [x_len]
    # seq_range_expand
    seq_range_expand = mint.arange(0, x_len)
    seq_range_expand = mint.unsqueeze(seq_range_expand, 0)  # shape: [1, x_len]
    seq_range_expand = ops.broadcast_to(seq_range_expand, (x_len, x_len))
    # idx_left/boundary_left
    idx_left = idx - left_window
    idx_left[idx_left < 0] = 0
    boundary_left = start_pad[idx_left]
    mask_left = seq_range_expand >= mint.unsqueeze(boundary_left,-1)
    idx_right = idx + right_window
    idx_right[idx_right > len(chunk_start_idx)] = len(chunk_start_idx)
    boundary_right = end_pad[idx_right]
    mask_right = seq_range_expand < mint.unsqueeze(boundary_right,-1)
    return ops.logical_and(mask_left, mask_right)


class Swish(nn.Cell): 
    """Implement Swish activation module. (MindSpore版)"""

    def __init__(self) -> None:
        super().__init__()
        self.act_fn = nn.Sigmoid() 

    def construct(self, x): 
        """Apply Swish function

        Args:
            x: ms.Tensor
                Input.
        """
        return x * self.act_fn(x)


class GLU(nn.Cell): 
    """Implement Gated Linear Unit (GLU) module (MindSpore版)"""

    def __init__(self, dim: int = -1, act_name: str = "sigmoid") -> None:
        super().__init__()
        self.dim = dim
        self.act_name = act_name.lower()

        if self.act_name == "relu":
            self.act_fn = nn.ReLU()
        elif self.act_name == "gelu":
            self.act_fn = nn.GELU()
        elif self.act_name == "swish":
            self.act_fn = Swish()
        elif self.act_name == "sigmoid":
            self.act_fn = nn.Sigmoid()
        else:
            self.act_fn = mint.nn.Identity()

    def construct(self, x):  
        """GLU forward
        Apply activation function on the second half of input matrices.

        Args:
            x: ms.Tensor
                Input.
        """
        half_x, gate = mint.chunk(x,2, dim=self.dim)
        return half_x * self.act_fn(gate)


class GLUPointWiseConv(nn.Cell):  
    """GLUPointWiseConv module (MindSpore版)"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_size: int,
        glu_type: str = "sigmoid",
        bias_in_glu: bool = True,
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.glu_type = glu_type
        self.output_dim = output_dim
        self.bias_in_glu = bias_in_glu

        pad = (kernel_size - 1) if causal else (kernel_size - 1) // 2
        self.ext_pw_conv_1d = mint.nn.Conv1d(
                input_dim,
                output_dim * 2,
                kernel_size,
                1,
                padding=pad,
            )

        if glu_type == "sigmoid":
            self.glu_act = nn.Sigmoid()
        elif glu_type == "relu":
            self.glu_act = nn.ReLU()
        elif glu_type == "gelu":
            self.glu_act = nn.GELU()
        elif glu_type == "swish":
            self.glu_act = Swish()
        else:
            raise ValueError(f"Unsupported activation type {self.glu_type}")

        if bias_in_glu:
            self.b1 = ms.Parameter(mint.zeros((1, output_dim, 1)), name="b1")
            self.b2 = ms.Parameter(mint.zeros((1, output_dim, 1)), name="b2")

    def construct(self, x: ms.Tensor)->ms.Tensor:  
        # to be consistent with GLULinear, we assume the input always has the
        # channel (#dim) in the last dimension of the tensor, so need to
        # switch the dimension first for 1D-Conv case
        x = mint.permute(x, (0, 2, 1))
        x = self.ext_pw_conv_1d(x)
        # x shape: [B, 2*output_dim, T]
        out1 = x[:, :self.output_dim, :]
        out2 = x[:, self.output_dim:self.output_dim * 2, :]
        if self.glu_type == "bilinear":
            if self.bias_in_glu:
                out = (out1 + self.b1) * (out2 + self.b2)
            else:
                out = out1 * out2
        else:
            if self.bias_in_glu:
                out = (out1 + self.b1) * self.glu_act(out2 + self.b2)
            else:
                out = out1 * self.glu_act(out2)
        x = mint.permute(out,(0, 2, 1))
        return x


class DepthWiseSeperableConv1d(nn.Cell): 
    """DepthWiseSeperableConv1d module used in Convnet module (MindSpore版)
    for the conformer, for more details see:
    https://arxiv.org/pdf/2005.08100v1.pdf

    Args:
        input_dim: int
            input channel size.
        depthwise_seperable_out_channel: int
            if set different to 0, the number of 
             depthwise_seperable_out_channel will be used as a channel_out
             of the second conv1d layer.
             otherwise, it equals to 0, the second conv1d layer is skipped.
        kernel_size: int
            kernel_size
        depthwise_multiplier: int
            number of input_dim channels duplication. this value
            will be used to compute the hidden channels of the Conv1D.
        padding: int, optional
            padding for the conv1d,
             default: 0.
    """

    def __init__(
        self,
        input_dim: int,
        depthwise_seperable_out_channel: int,
        kernel_size: int,
        depthwise_multiplier: int,
        padding: int = 0,
    ) -> None:
        super().__init__()
        self.dw_conv = mint.nn.Conv1d(
            input_dim,
            input_dim * depthwise_multiplier,
            kernel_size,
            1,
            padding=padding,
            groups=input_dim,
        )

        if depthwise_seperable_out_channel != 0:
            self.pw_conv = mint.nn.Conv1d(
                input_dim * depthwise_multiplier,
                depthwise_seperable_out_channel,
                1,
                1,
                0,
            )
        else:
            self.pw_conv = mint.nn.Identity()
        self.depthwise_seperable_out_channel = depthwise_seperable_out_channel

    def construct(self, x):  
        """
        Args:
            x: input tensor
        """
        x = self.dw_conv(x)
        if self.depthwise_seperable_out_channel != 0:
            x = self.pw_conv(x)
        return x

class ConvModule(nn.Cell): 
    """ConvModule Module for the conformer block. (MindSpore版)"""

    def __init__(
        self,
        input_dim: int,
        ext_pw_out_channel: int,
        depthwise_seperable_out_channel: int,
        ext_pw_kernel_size: int,
        kernel_size: int,
        depthwise_multiplier: int,
        dropout_rate: float,
        causal: bool = False,
        batch_norm: bool = False,
        chunk_se: int = 0,
        chunk_size: int = 18,
        activation: str = "relu",
        glu_type: str = "sigmoid",
        bias_in_glu: bool = True,
        linear_glu_in_convm: bool = False,
        export: bool = False,
    ) -> None:
        super().__init__()
        self.layer_norm = mint.nn.LayerNorm(input_dim)
        self.input_dim = input_dim
        self.ext_pw_out_channel = ext_pw_out_channel
        self.ext_pw_kernel_size = ext_pw_kernel_size
        self.depthwise_seperable_out_channel = depthwise_seperable_out_channel
        self.glu_type = glu_type
        self.bias_in_glu = bias_in_glu
        self.linear_glu_in_convm = linear_glu_in_convm
        self.causal = causal

        self._add_ext_pw_layer()

        self.batch_norm = batch_norm
        self.kernel_size = kernel_size

        if batch_norm:
            self.bn_layer = mint.nn.BatchNorm1d(input_dim)

        self.act = get_activation(activation) 
        self.dropout = mint.nn.Dropout(p=dropout_rate) 
        self.export = export

        if causal:
            padding = 0 if export else kernel_size - 1
        else:
            padding = (kernel_size - 1) // 2

        self.dw_sep_conv_1d = DepthWiseSeperableConv1d(
            input_dim,
            depthwise_seperable_out_channel,
            kernel_size,
            depthwise_multiplier,
            padding=padding,
        )

        if depthwise_seperable_out_channel != 0:
            if input_dim != depthwise_seperable_out_channel:
                self.ln2 = mint.nn.Linear(depthwise_seperable_out_channel, input_dim)  # 【修改5】nn.Linear->nn.Dense
        else:
            if depthwise_multiplier != 1:
                self.ln2 = mint.nn.Linear(input_dim * depthwise_multiplier, input_dim)  # 【修改6】nn.Linear->nn.Dense

    def _add_ext_pw_layer(self) -> None:
        self.ln1 = self.glu = self.bn_layer = self.ext_pw_conv_1d = mint.nn.Identity()
        self.squeeze_excitation = mint.nn.Identity()
        self.apply_ln1 = self.fix_len1 = False

        if self.ext_pw_out_channel != 0:
            if self.causal:
                pad = self.ext_pw_kernel_size - 1
            else:
                pad = (self.ext_pw_kernel_size - 1) // 2

            self.ext_pw_conv_1d = mint.nn.Conv1d(
                    self.input_dim,
                    self.ext_pw_out_channel,
                    self.ext_pw_kernel_size,
                    1,
                    padding=pad,
                )
            self.fix_len1 = self.causal and self.ext_pw_kernel_size > 1

            if self.linear_glu_in_convm:
                self.glu = GLULinear(
                    self.input_dim,
                    self.ext_pw_out_channel,
                    self.glu_type,
                    self.bias_in_glu,
                )
            else:
                self.glu = GLUPointWiseConv(
                    self.input_dim,
                    self.ext_pw_out_channel,
                    self.ext_pw_kernel_size,
                    self.glu_type,
                    self.bias_in_glu,
                    self.causal,
                )

            if self.input_dim != self.ext_pw_out_channel:
                self.apply_ln1 = True
                self.ln1 = mint.nn.Linear(self.ext_pw_out_channel, self.input_dim)
            else:
                self.apply_ln1 = False
        else:
            self.pw_conv_simplify_w = ms.Parameter(mint.ones(3), name="pw_conv_simplify_w")
            self.pw_conv_simplify_b = ms.Parameter(mint.zeros(3), name="pw_conv_simplify_b")

    def construct(self, x):  
        x = self.layer_norm(x)

        if self.ext_pw_out_channel != 0:
            x = self.glu(x)
            if self.causal and self.ext_pw_kernel_size > 1:
                x = x[:, :-(self.ext_pw_kernel_size - 1), :]
            if self.apply_ln1:
                x = self.ln1(x)
        else:
            x_0 = x * self.pw_conv_simplify_w[0] + self.pw_conv_simplify_b[0]
            x_1 = x * self.pw_conv_simplify_w[1] + self.pw_conv_simplify_b[1]
            x = x_0 + x_1

        x = mint.permute(x,(0, 2, 1))

        x = self.dw_sep_conv_1d(x)
        if self.causal and self.kernel_size > 1:
            x = x[:, :, :-(self.kernel_size - 1)]
        if hasattr(self, "ln2"):
            x = mint.permute(x, (0, 2, 1))
            x = self.ln2(x)
            x = mint.permute(x, (0, 2, 1))
        if self.batch_norm:
            x = self.bn_layer(x)
        x = self.act(x)

        if self.ext_pw_out_channel != 0:
            x = self.ext_pw_conv_1d(x)
            if self.fix_len1:
                x = x[:, :, :-(self.ext_pw_kernel_size - 1)]

            if self.apply_ln1:
                x = mint.permute(x, (0, 2, 1))
                x = self.ln1(x)
                x = mint.permute(x, (0, 2, 1))

            x = mint.permute(x, (0, 2, 1))
        else:
            x = mint.unsqueeze(x, 1)
            x = mint.permute(x, (0, 1, 3, 2))
            x = x * self.pw_conv_simplify_w[2] + self.pw_conv_simplify_b[2]
            x = mint.squeeze(x, 1)

        x = self.dropout(x)
        return x

class GLULinear(nn.Cell): 
    """Linear + GLU module (MindSpore版)

    Args:
        input_dim: int
            input size
        output_dim: int
            output size.
        glu_type:
            activation function name used in glu module.
            default "sigmoid" (swish function).
        bias_in_glu: bool, optional
            If True, the additive bias is added. Default False.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        glu_type: str = "sigmoid",
        bias_in_glu: bool = True,
    ) -> None:
        super().__init__()
        self.linear = mint.nn.Linear(input_dim, output_dim * 2, bias_in_glu)
        self.glu_act = GLU(-1, glu_type) 

    def construct(self, x): 
        x = self.linear(x)
        return self.glu_act(x)

class FeedForward(nn.Cell):  
    """FeedForward Module. (MindSpore版)
    For more details see Conformer paper:
        https://arxiv.org/pdf/2005.08100.pdf

    Args:
        d_model: int
            input size.
        d_inner: int
            output size.
        dropout_rate: float,
            dropout rate.
        activation: str,
            activation function name,
            one of ["relu", "swish", "sigmoid"],
            sigmoid activation is only used with "glu_in_fnn=True",
            default "sigmoid".
        bias_in_glu: bool, optional
    """

    def __init__(
        self,
        d_model: int,
        d_inner: int,
        dropout_rate: float,
        activation: str = "sigmoid",
        bias_in_glu: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner

        self.layer_norm = mint.nn.LayerNorm(d_model)
        module = GLULinear(d_model, d_inner, activation, bias_in_glu)  
        self.net = nn.SequentialCell(  
            module,
            mint.nn.Dropout(p=dropout_rate), 
            mint.nn.Linear(d_inner, d_model),   
            mint.nn.Dropout(p=dropout_rate),
        )

    def construct(self, x): 
        out = self.net(self.layer_norm(x))
        return out

def _pre_hook(
    state_dict: dict,
    prefix: str,
    local_metadata: dict,
    strict: bool,
    missing_keys: list,
    unexpected_keys: list,
    error_msgs: list,
) -> None:
    """Perform pre-hook in load_state_dict for backward compatibility.

    Note:
        We saved self.pe until v.0.5.2 but we have omitted it later.
        Therefore, we remove the item "pe" from `state_dict` for backward 
        compatibility.

    """
    k = prefix + "pe"
    if k in state_dict:
        state_dict.pop(k)


class T5RelativeAttentionLogitBias(nn.Cell): 

    def __init__(self,
                 num_heads: int,
                 num_buckets: int = -1,
                 max_distance: int = 1000,
                 symmetric: bool = False) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.symmetric = symmetric
        self._skip_bucketing = self.num_buckets < 0
        if self._skip_bucketing:
            self.num_buckets = max_distance
        else:
            raise NotImplementedError(
                "T5 attention bias with bucketed positions is not yet tested")
        if not self.symmetric:
            self.num_buckets *= 2
        self.bias_values = mint.nn.Embedding(self.num_buckets, self.num_heads)

    def construct(self, x: ms.Tensor) -> ms.Tensor: 
        maxpos = x.shape[1]
        context_position = mint.arange(maxpos, dtype=ms.int64).expand_dims(1)  # [L, 1]
        memory_position = mint.arange(maxpos, dtype=ms.int64).expand_dims(0)  # [1, L]
        relative_position = memory_position - context_position  # [L, L]

        relative_position = relative_position.masked_fill(
            relative_position < -self.max_distance, -self.max_distance)
        relative_position = relative_position.masked_fill(
            relative_position > self.max_distance - 1, self.max_distance - 1)

        # mapping from relative position to index in the bias parameter
        if self._skip_bucketing:
            bias_idx = relative_position
        else:
            bias_idx = self._bucket_relative_position(relative_position)
        if self.symmetric:
            bias_idx = mint.abs(bias_idx)
        else:
            bias_idx = bias_idx + (self.num_buckets // 2)
        t5_rel_att_bias = self.bias_values(bias_idx)  # [L, L, H]
        t5_rel_att_bias = mint.permute(t5_rel_att_bias,(2, 0, 1))
        t5_rel_att_bias = mint.unsqueeze(t5_rel_att_bias, 0)  # [1, H, L, L]
        return t5_rel_att_bias

    def _bucket_relative_position(self, relative_position):
        relative_buckets = 0
        if not hasattr(self, "causal"):
            self.causal = False  
        if not self.causal:
            self.num_buckets = self.num_buckets // 2  
            relative_buckets += (relative_position > 0).astype(ms.int64) * self.num_buckets
            relative_position = mint.abs(relative_position)
        else:
            relative_position = -mint.min(relative_position, mint.zeros_like(relative_position))
        

        max_exact = self.num_buckets // 2
        is_small = relative_position < max_exact

        relative_position_float = relative_position.astype(ms.float32)
        relative_position_if_large = max_exact + (
            mint.log(relative_position_float / max_exact) /
            math.log(self.max_distance / max_exact) *
            (self.num_buckets - max_exact)
        )
        relative_position_if_large = relative_position_if_large.astype(ms.int64)
        relative_position_if_large = mint.min(
            relative_position_if_large,
            mint.full_like(relative_position_if_large, self.num_buckets - 1)
        )

        relative_buckets += mint.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets


class AbsolutePositionalEncoding(nn.Cell): 
    """Absolute Positional encoding module. (MindSpore版)
    实现绝对正弦位置编码
    """

    def __init__(self,
                 d_model: int,
                 dropout_rate: float,
                 max_len: int = 5000) -> None:
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = mint.nn.Dropout(p=dropout_rate)  
        self.pe = None
        self.extend_pe(ms.Tensor(ops.zeros((1, max_len)), ms.float32))  # 【修改3】初始化用ms.Tensor
        

    def extend_pe(self, x: ms.Tensor) -> None:  
        """Reset the positional encodings.

        Args:
            x: input tensor
        """
        if self.pe is not None and self.pe.shape[1] >= x.shape[1]:
            
            if self.pe.dtype != x.dtype:
                self.pe = self.pe.astype(x.dtype)
            return
        pe = mint.zeros((x.shape[1], self.d_model))
        position = mint.unsqueeze(mint.arange(0, x.shape[1], dtype=ms.float32),1)
        div_term = mint.exp(
            mint.arange(0, self.d_model, 2, dtype=ms.float32) *
            -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = mint.sin(position * div_term)
        pe[:, 1::2] = mint.cos(position * div_term)
        pe = mint.unsqueeze(pe, 0)  
        self.pe = pe.astype(x.dtype) 

    def construct(self, x:ms.Tensor)->ms.Tensor: 
        """Add positional encoding.

        Args:
            x: Input tensor. shape is (batch, time, ...)

        Returns:
            Encoded tensor. Its shape is (batch, time, ...)
        """
        self.extend_pe(x)
        x = x * self.xscale + self.pe[:, :x.shape[1]]
        x = self.dropout(x)
        return x


class MeanVarianceNormLayer(nn.Cell):  
    """Mean/variance normalization layer. (MindSpore版)

    Will subtract mean and multiply input by inverted standard deviation.
    Typically used as a very first layer in a model.

    Args:
        input_size: int
            layer input size.
    """

    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.global_mean = ms.Parameter(mint.zeros(input_size, dtype=ms.float32), name="global_mean")
        self.global_invstd = ms.Parameter(mint.ones(input_size, dtype=ms.float32), name="global_invstd")

    def construct(self, input_: ms.Tensor) -> ms.Tensor:  
        """MeanVarianceNormLayer Forward

        Args:
            input_: input tensor.
        """
        return (input_ - self.global_mean) * self.global_invstd


class CausalConv1D(mint.nn.Conv1d):  
    """
    MindSpore版：Causal 1D卷积
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Union[str, int] = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        dtype=None,
    ) -> None:
        self.cache_drop_size = None
        
        if padding is None:
            self._left_padding = kernel_size - 1
            self._right_padding = stride - 1
        else:
            if stride != 1 and padding != kernel_size - 1:
                raise ValueError(
                    "No striding allowed for non-symmetric convolutions!")
            if isinstance(padding, int):
                self._left_padding = padding
                self._right_padding = padding
            elif (isinstance(padding, (list, tuple)) and len(padding) == 2
                  and padding[0] + padding[1] == kernel_size - 1):
                self._left_padding = padding[0]
                self._right_padding = padding[1]
            else:
                raise ValueError(f"Invalid padding param: {padding}!")

        self._max_cache_len = self._left_padding

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            dtype=dtype,
        )

    def update_cache(
            self,
            x: ms.Tensor,
            cache: Optional[ms.Tensor] = None  
        )-> tuple[ms.Tensor, Optional[ms.Tensor]]:
        if cache is None:
            new_x = F.pad(x,  pad=(self._left_padding, self._right_padding))  
            next_cache = cache
        else:
            new_x = F.pad(x, pad=(0, self._right_padding))
            new_x = mint.cat([cache, new_x], dim=-1)
            if self.cache_drop_size and self.cache_drop_size > 0:
                next_cache = new_x[:, :, :-self.cache_drop_size]
            else:
                next_cache = new_x
            next_cache = next_cache[:, :, -cache.shape[-1]:]
        return new_x, next_cache

    def construct(
        self,
        x: ms.Tensor,
        cache: Optional[ms.Tensor] = None 
    )-> Union[ms.Tensor, tuple[ms.Tensor, Optional[ms.Tensor]]]:
        x, cache = self.update_cache(x, cache=cache)
        x = super().construct(x)
        if cache is None:
            return x
        else:
            return x, cache


class CausalConv2D(mint.nn.Conv2d):  
    """
    MindSpore版：Causal 2D卷积
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Union[str, int] = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        dtype=None,
    ) -> None:
        if padding is not None:
            raise ValueError(
                "Argument padding should be set to None for CausalConv2D.")
        self._left_padding = kernel_size - 1
        self._right_padding = stride - 1
        
        padding = 0
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            dtype,
        )

    def construct(self, x: ms.Tensor) -> ms.Tensor:  
        x = F.pad(
            x,
            pad=(self._left_padding, self._right_padding, 0, 0),
        )
        x = super().construct(x)
        return x


class NemoConvSubsampling(nn.Cell):  

    def __init__(
        self,
        feat_in: int,
        feat_out: int,
        subsampling_factor: int = 4,
        subsampling: str = "dw_striding",
        conv_channels: int = 256,
        subsampling_conv_chunking_factor: int = 1,
        activation: nn.Cell = nn.ReLU(), 
        is_causal: bool = False,
    ) -> None:
        super().__init__()
        self._subsampling = subsampling
        self._conv_channels = conv_channels
        self._feat_in = feat_in
        self._feat_out = feat_out

        if subsampling_factor % 2 != 0:
            raise ValueError("Sampling factor should be a multiply of 2!")
        self._sampling_num = int(math.log(subsampling_factor, 2))
        self.subsampling_factor = subsampling_factor
        self.is_causal = is_causal
        self.subsampling_causal_cond = subsampling in (
            "dw_striding",
            "striding",
            "striding_conv1d",
        )

        if (subsampling_conv_chunking_factor != -1
                and subsampling_conv_chunking_factor != 1
                and subsampling_conv_chunking_factor % 2 != 0):
            raise ValueError(
                "subsampling_conv_chunking_factor should be -1, 1, or a power of 2"
            )
        self.subsampling_conv_chunking_factor = subsampling_conv_chunking_factor

        in_channels = 1
        layers = []

        
        if subsampling == "dw_striding":
            self._stride = 2
            self._kernel_size = 3
            self._ceil_mode = False

            if self.is_causal:
                self._left_padding = self._kernel_size - 1
                self._right_padding = self._stride - 1
                self._max_cache_len = subsampling_factor + 1
            else:
                self._left_padding = (self._kernel_size - 1) // 2
                self._right_padding = (self._kernel_size - 1) // 2
                self._max_cache_len = 0

            # Layer 1
            if self.is_causal:
                layers.append(
                    CausalConv2D(
                        in_channels=in_channels,
                        out_channels=conv_channels,
                        kernel_size=self._kernel_size,
                        stride=self._stride,
                        padding=None,
                    ))
            else:
                layers.append(
                    mint.nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=conv_channels,
                        kernel_size=self._kernel_size,
                        stride=self._stride,
                        padding=self._left_padding,
                    ))
            in_channels = conv_channels
            layers.append(activation)

            for i in range(self._sampling_num - 1):
                if self.is_causal:
                    layers.append(
                        CausalConv2D(
                            in_channels=in_channels,
                            out_channels=in_channels,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=None,
                            group=in_channels,
                        ))
                else:
                    layers.append(
                        mint.nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=in_channels,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=self._left_padding,
                            groups=in_channels,
                        ))

                layers.append(
                    mint.nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=conv_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        groups=1,
                    ))
                layers.append(activation)
                in_channels = conv_channels

        elif subsampling == "striding":
            self._stride = 2
            self._kernel_size = 3
            self._ceil_mode = False

            if self.is_causal:
                self._left_padding = self._kernel_size - 1
                self._right_padding = self._stride - 1
                self._max_cache_len = subsampling_factor + 1
            else:
                self._left_padding = (self._kernel_size - 1) // 2
                self._right_padding = (self._kernel_size - 1) // 2
                self._max_cache_len = 0

            for i in range(self._sampling_num):
                if self.is_causal:
                    layers.append(
                        CausalConv2D(
                            in_channels=in_channels,
                            out_channels=conv_channels,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=None,
                        ))
                else:
                    layers.append(
                        mint.nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=conv_channels,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=self._left_padding,
                        ))
                layers.append(activation)
                in_channels = conv_channels

        elif subsampling == "striding_conv1d":
            in_channels = feat_in

            self._stride = 2
            self._kernel_size = 5
            self._ceil_mode = False

            if self.is_causal:
                self._left_padding = self._kernel_size - 1
                self._right_padding = self._stride - 1
                self._max_cache_len = subsampling_factor + 1
            else:
                self._left_padding = (self._kernel_size - 1) // 2
                self._right_padding = (self._kernel_size - 1) // 2
                self._max_cache_len = 0

            for i in range(self._sampling_num):
                if self.is_causal:
                    layers.append(
                        CausalConv1D(
                            in_channels=in_channels,
                            out_channels=(feat_out if self._sampling_num == i +
                                          1 else conv_channels),
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=None,
                        ))
                else:
                    layers.append(
                        mint.nn.Conv1d(
                            in_channels=in_channels,
                            out_channels=(feat_out if self._sampling_num == i +
                                          1 else conv_channels),
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=self._left_padding,
                        ))
                layers.append(activation)
                in_channels = conv_channels
        
        elif subsampling == "dw_striding_conv1d":
            in_channels = feat_in

            self._stride = 2
            self._kernel_size = 5
            self._ceil_mode = False

            self._left_padding = (self._kernel_size - 1) // 2
            self._right_padding = (self._kernel_size - 1) // 2

            # Layer 1
            layers.extend([
                mint.nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=self._kernel_size,
                    stride=self._stride,
                    padding=self._left_padding,
                    groups=in_channels,
                ),
                mint.nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=(feat_out if self._sampling_num == 1 else
                                  conv_channels),
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=1,
                ),
            ])
            in_channels = conv_channels
            layers.append(activation)

            for i in range(self._sampling_num - 1):
                layers.extend([
                    mint.nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        kernel_size=self._kernel_size,
                        stride=self._stride,
                        padding=self._left_padding,
                        groups=in_channels,
                    ),
                    mint.nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=(feat_out if self._sampling_num == i +
                                      2 else conv_channels),
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        groups=1,
                    ),
                ])
                layers.append(activation)
                in_channels = conv_channels
        else:
            raise ValueError(f"Not valid sub-sampling: {subsampling}!")

        if subsampling in ["dw_striding", "striding"]:
            in_length = ms.Tensor(feat_in, dtype=ms.float32)
            out_length = calc_length(
                lengths=in_length,
                all_paddings=self._left_padding + self._right_padding,
                kernel_size=self._kernel_size,
                stride=self._stride,
                ceil_mode=self._ceil_mode,
                repeat_num=self._sampling_num,
            )
            self.out = mint.nn.Linear(conv_channels * int(out_length), feat_out)
            self.conv2d_subsampling = True
        elif subsampling in ["striding_conv1d", "dw_striding_conv1d"]:
            self.out = None
            self.conv2d_subsampling = False
        else:
            raise ValueError(f"Not valid sub-sampling: {subsampling}!")

        self.conv = nn.SequentialCell(layers) 

    def get_sampling_frames(self) -> list:
        return [1, self.subsampling_factor]

    def get_streaming_cache_size(self) -> list:
        return [0, self.subsampling_factor + 1]

    def construct(self, x: ms.Tensor, mask: Optional[ms.Tensor])->tuple[ms.Tensor, Optional[ms.Tensor]]:  
        x = mint.unsqueeze(x,1) if self.conv2d_subsampling else mint.transpose(x,1, 2)
        
        # split inputs if chunking_factor is set
        if (self.subsampling_conv_chunking_factor != -1
                and self.conv2d_subsampling):
            if self.subsampling_conv_chunking_factor == 1:
                x_ceil = (2**31 / self._conv_channels * self._stride *
                          self._stride)
                
                need_to_split = x.numel() > x_ceil
                
            else:
                
                need_to_split = True

            if need_to_split:
                x, success = self.conv_split_by_batch(x)
                if not success:  
                    if self._subsampling == "dw_striding":
                        x = self.conv_split_by_channel(x)
                    else:
                        x = self.conv(x)  
            else:
                x = self.conv(x)
        else:
            x = self.conv(x)

        # Flatten Channel and Frequency Axes
        if self.conv2d_subsampling:
            b, c, t, f = x.shape
            x = mint.reshape(mint.transpose(x, 1, 2),(b, t, -1))
            x = self.out(x)
        else:
            x = mint.transpose(x, 1, 2)

        if mask is None:
            return x, None

        max_audio_length = x.shape[1]
        feature_lens = mask.sum(1)
        padding_length = mint.ceil(feature_lens / self.subsampling_factor)
        if self.is_causal and self.subsampling_causal_cond:
            feature_lens_remainder = feature_lens % self.subsampling_factor
            padding_length[feature_lens_remainder != 1] += 1
        pad_mask = mint.arange(0, max_audio_length).expand_dims(0).broadcast_to((padding_length.shape[0], max_audio_length)) < padding_length.expand_dims(1)
        return x, mint.unsqueeze(pad_mask, 1)

    def reset_parameters_ms(self):
        
        if self._subsampling == "dw_striding":
            
            scale = 1.0 / self._kernel_size
            dw_max = (self._kernel_size**2)**-0.5
            pw_max = self._conv_channels**-0.5
            
            
            uniform_init_scale = init.Uniform(scale=scale)
            uniform_init_dw = init.Uniform(scale=dw_max)
            uniform_init_pw = init.Uniform(scale=pw_max)

            
            self.conv[0].weight.set_data(uniform_init_scale)
            self.conv[0].bias.set_data(uniform_init_scale)

            
            for idx in range(2, len(self.conv), 3):
                
                self.conv[idx].weight.set_data(uniform_init_dw)
                self.conv[idx].bias.set_data(uniform_init_dw)

                
                self.conv[idx + 1].weight.set_data(uniform_init_pw)
                self.conv[idx + 1].bias.set_data(uniform_init_pw)

            
            fc_scale = (self._feat_out * self._feat_in / self._sampling_num)**-0.5
            uniform_init_fc = init.Uniform(scale=fc_scale)
            
            
            self.out.weight.set_data(uniform_init_fc)
            self.out.bias.set_data(uniform_init_fc)

    def conv_split_by_batch(self, x: ms.Tensor) -> tuple[ms.Tensor, bool]:  
        b = x.shape[0]
        if b == 1:
            return x, False

        if self.subsampling_conv_chunking_factor > 1:
            cf = self.subsampling_conv_chunking_factor
        else:
            x_ceil = 2**31 / self._conv_channels * self._stride * self._stride
            p = math.ceil(math.log(ops.size(x) / x_ceil, 2))
            cf = 2**p

        new_batch_size = b // cf
        if new_batch_size == 0:
            return x, False

        chunks = mint.split(x, new_batch_size, dim=0)
        out = mint.cat([self.conv(chunk) for chunk in chunks])
        return out, True

    def conv_split_by_channel(self, x):  
        x = self.conv[0](x)
        x = self.conv[1](x)
        for i in range(self._sampling_num - 1):
            _, c, t, _ = x.shape

            if self.subsampling_conv_chunking_factor > 1:
                cf = self.subsampling_conv_chunking_factor
            else:
                p = math.ceil(math.log(ops.size(x) / 2**31, 2))
                cf = 2**p

            new_c = int(c // cf)
            if new_c == 0:
                new_c = 1

            new_t = int(t // cf)
            if new_t == 0:
                new_t = 1

            x = self.channel_chunked_conv(self.conv[i * 3 + 2], new_c, x)
            
            chunks = mint.split(x, new_t, dim=2)
            x = mint.concat([self.conv[i * 3 + 3](chunk) for chunk in chunks], dim=2)
            x = self.conv[i * 3 + 4](x)
        return x

    def channel_chunked_conv_ms(self, conv: nn.Cell, chunk_size: int, x: ms.Tensor) -> ms.Tensor:
        """Performs channel chunked convolution in MindSpore"""

        ind = 0
        out_chunks = []
        
        chunks = mint.split(x, split_size_or_sections=chunk_size, dim=1) 

        for chunk in chunks:
            step = chunk.shape[1] 

            if self.is_causal:
                chunk = F.pad(
                    chunk,
                    pad=(
                        self._kernel_size - 1,
                        self._stride - 1,
                        self._kernel_size - 1,
                        self._stride - 1,
                    ),
                )
                ch_out = F.conv2d(
                    chunk,
                    conv.weight[ind:ind + step, :, :, :],
                    bias=conv.bias[ind:ind + step],
                    stride=self._stride,
                    padding=0,
                    groups=step,
                )
            else:
                ch_out = F.conv2d(
                    chunk,
                    conv.weight[ind:ind + step, :, :, :],
                    bias=conv.bias[ind:ind + step],
                    stride=self._stride,
                    padding=self._left_padding,
                    groups=step,
                )
            
            out_chunks.append(ch_out)
            ind += step

        return mint.cat(out_chunks, dim=1)


    def change_subsampling_conv_chunking_factor(
            self, subsampling_conv_chunking_factor: int) -> None:
        if (subsampling_conv_chunking_factor != -1
                and subsampling_conv_chunking_factor != 1
                and subsampling_conv_chunking_factor % 2 != 0):
            raise ValueError(
                "subsampling_conv_chunking_factor should be -1, 1, or a "\
                    "power of 2"
            )
        self.subsampling_conv_chunking_factor = subsampling_conv_chunking_factor



def calc_length(lengths: ms.Tensor,
                all_paddings: int,
                kernel_size: int,
                stride: int,
                ceil_mode: bool,
                repeat_num: int = 1)-> ms.Tensor:
    """Calculates the output length of a Tensor passed through a convolution or max pooling layer (MindSpore版)"""

    add_pad = all_paddings - kernel_size
    one = ms.Tensor(1.0, ms.float32)
    lengths = lengths.astype(ms.float32)
    for _ in range(repeat_num):
        lengths = mint.div(lengths + add_pad,stride)+ one
        lengths = mint.ceil(lengths) if ceil_mode else mint.floor(lengths)
    return lengths.astype(ms.int32)


class AttModule(nn.Cell):  
    """Attention abstraction module (MindSpore版)"""

    def __init__(self) -> None:
        super().__init__()
        self.export_mode = False

    def set_export(self, mode: bool = True) -> None:
        """set the export mode"""
        self.export_mode = mode

    def construct(  
        self,
        x: ms.Tensor,
        memory: Optional[ms.Tensor] = None,
        pos_emb: Optional[ms.Tensor] = None,
        att_mask: Optional[ms.Tensor] = None,
    ) -> tuple[ms.Tensor, ms.Tensor, Optional[ms.Tensor], Optional[ms.Tensor]]:
        """AttModule construct

        Args:
            x: input tensor.
            memory: memory tensor.
            pos_emb: positional encoder embedding.
            att_mask: attention mask tensor.
        """
        return x, memory, pos_emb, att_mask


class AttBlock(BlockBase, AttModule):
    """Attention Block module to support both Attention and Block module."""

    def memory_dims(self, max_len: bool = False) -> tuple[int, int]:
        """memory dimensions"""
        return (1, self.input_size)


def masked_softmax(
    scores: ms.Tensor,
    mask: Optional[ms.Tensor],
)-> ms.Tensor:
    
    if mask is not None:
        mask = mint.eq(mint.unsqueeze(mask, 1),0)
        scores = ops.masked_fill(scores, mask, -float('inf'))  
        attn = ops.masked_fill(mint.softmax(scores,dim=-1), mask, 0.0)
    else:
        attn = mint.softmax(scores,dim=-1)
    return attn



class MultiHeadedAttention(nn.Cell): 
    """Multi-Head Attention layer with optional relative position embedding and GLU. (MindSpore版)"""


    def __init__(
        self,
        n_head: int,
        n_feat: int,
        dropout_rate: float,
        attention_inner_dim: int = -1,
        glu_type: str = "swish",
        bias_in_glu: bool = True,
        use_pt_scaled_dot_product_attention: bool = False, 
        n_value: int = -1,
        group_size: int = 1,
    ) -> None:
        super().__init__()
        if n_value == -1:
            n_value = n_feat
        if attention_inner_dim == -1:
            attention_inner_dim = n_feat
        assert attention_inner_dim % n_head == 0

        self.d_k = attention_inner_dim // n_head
        self.inv_sqrt_d_k = 1.0 / math.sqrt(self.d_k)
        self.h = n_head
        assert n_head % group_size == 0, "group_size must divide n_head"
        self.g = group_size
        self.h_k = n_head // group_size

        
        self.linear_q = mint.nn.Linear(n_feat, attention_inner_dim)
        self.linear_k = mint.nn.Linear(n_feat, attention_inner_dim // group_size)
        self.linear_v = mint.nn.Linear(n_value, attention_inner_dim // group_size)
        self.linear_out = mint.nn.Linear(attention_inner_dim // group_size, n_value)

        self.attn = None  
        self.dropout = mint.nn.Dropout(p = dropout_rate)  
        self.dropout_rate = dropout_rate
        self.use_pt_scaled_dot_product_attention = (use_pt_scaled_dot_product_attention)

        if use_pt_scaled_dot_product_attention and group_size > 1:
            raise ValueError("Cannot use PT Scaled Attention with GQA")
        


    def construct(
        self,
        query: ms.Tensor,
        key: ms.Tensor,
        value: ms.Tensor,
        pos_k: Optional[ms.Tensor],
        pos_v: Optional[ms.Tensor],
        mask: Optional[ms.Tensor],
        relative_attention_bias: Optional[ms.Tensor] = None,
    )->ms.Tensor:
        n_batch = query.shape[0]

        
        q = self.linear_q(query).view((n_batch, -1, self.h, self.d_k))
        k = self.linear_k(key).view((n_batch, -1, self.h_k, self.d_k))
        v = self.linear_v(value).view((n_batch, -1, self.h_k, self.d_k))
        q = mint.transpose(q, 1,2) * self.inv_sqrt_d_k  
        k = mint.transpose(k, 1,2)  
        v = mint.transpose(v, 1,2)  

        
        if self.h != self.h_k:
            q = q.reshape((n_batch, self.g, self.h_k, -1, self.d_k))
            
            A = mint.einsum('b g h t d, b h s d -> b h t s', q, k)
        else:
            
            A = mint.matmul(q, mint.transpose(k, -2, -1))
        if pos_k is not None:
            if self.h != self.h_k:
                
                B = mint.einsum('b g h t d, t s d -> b h t s', q, pos_k)
            else:
                reshape_q = mint.transpose(q.view((n_batch * self.h, -1, self.d_k)),0,1)
                B = mint.matmul(reshape_q, mint.transpose(pos_k, -2, -1))  
                B = mint.transpose(B, 0, 1)
                B = B.view((n_batch, self.h, pos_k.shape[0], pos_k.shape[1]))
            scores = A + B
        else:
            scores = A

        if relative_attention_bias is not None:
            scores = scores + relative_attention_bias

        attn = masked_softmax(scores, mask)  

        self.attn = attn

        p_attn = self.dropout(attn)

        
        x = mint.matmul(p_attn.astype(v.dtype), v)
        if pos_v is not None:
            reshape_attn = mint.transpose(p_attn.view((n_batch * self.h, pos_v.shape[0],pos_v.shape[1])),0,1)
            attn_v = mint.transpose(mint.matmul(reshape_attn, pos_v),0,1).view((n_batch, self.h, pos_v.shape[0],self.d_k))
            x = x + attn_v

        
        x = mint.transpose(x, 1,2).view((n_batch, -1, self.h_k * self.d_k))
        x = self.linear_out(x)
        return x



class MultiSequential(nn.SequentialCell):  
    """Multi-input multi-output MindSpore Sequential"""

    def construct(self, *args)-> tuple:
        """Forward method implementation."""
        for m in self:
            args = m(*args)
        return args


def get_offset(input_layer: str, time_reduction: int) -> int:
    """Get an offset. We will use the offset for determining #frames of a 
    subsampled feature.

    Args:
        input_layer: Type of an input layer
        time_reduction: time reduction factor for downsampling a feature
    Returns:
        int: offset
    """
    if input_layer in ("conv2d", "nemo_conv") and time_reduction == 4:
        return 3
    if input_layer in ("conv2d", ) and time_reduction == 6:
        return 1
    if input_layer in ("conv2d", "nemo_conv") and time_reduction == 8:
        return 7
    return 0


def unfold_tensor(xs_pad: ms.Tensor, max_seq_len: int) -> ms.Tensor:
    """
    For a given tensor with shape of (N, T, D), if sequence length T is 
    longer than max_seq_len, this function unfold it to a 
    (NT', max_seq_len, D) where T' is T // max_seq_len.
    Args:
        xs_pad: input tensor with shape (N, T, D)
        max_seq_len: maximum sequence length
    """
    _, _, D = xs_pad.shape
    xs_pad = mint.transpose(xs_pad,-1, -2)  # convert to N, D, T
    # N x D x 1 x T => N x (D x max_seq_len) x T'
    xs_pad = F.unfold(
        xs_pad[..., None, :],
        kernel_size=(1, max_seq_len),
        stride=(1, max_seq_len),
    )
    new_bsz, _, slen = xs_pad.shape
    # N x D x max_seq_len x T'
    xs_pad = xs_pad.view((new_bsz, -1, max_seq_len, slen))
    # N x T' x max_seq_len x D
    xs_pad = mint.permute(xs_pad,(0, 3, 2, 1))
    # NT' x max_seq_len x D
    xs_pad = xs_pad.view((-1, max_seq_len, D))
    return xs_pad