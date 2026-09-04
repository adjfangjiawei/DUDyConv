import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ============================================================
# 1. 模型配置
# ============================================================

@dataclass
class ModelConfig:
    vocab_size: int

    block_size: int = 1024

    embed_dim: int = 256

    # C: source 膨胀卷积的固定通道数。
    # 逻辑 source 为 [B,D,C,L]，实际按 chunk 生成 [B,D,C,T]。
    source_channels: int = 16

    # N: 动态卷积通道数。
    num_kernels: int = 16

    # K: 每个动态卷积核在时间 L 上的卷积核尺寸。
    dynamic_kernel_size: int = 3

    # source 膨胀卷积核尺寸。
    source_conv_kernel_size: int = 3

    # kernel 生成器的普通因果卷积核尺寸。
    # 注意：这里不是 dilation conv。
    kernel_gen_kernel_size: int = 3

    mlp_ratio: float = 4.0

    dropout: float = 0.1

    use_bias: bool = True

    use_sinusoidal_pos: bool = True

    # --------------------------------------------------------
    # checkpoint 控制逻辑：
    #
    # use_checkpoint=True:
    #   强制所有 checkpoint 点开启。
    #
    # use_checkpoint=False 且 adaptive_checkpoint=True:
    #   默认行为。所有 checkpoint 点按照估算激活大小自适应开启。
    #
    # use_checkpoint=False 且 adaptive_checkpoint=False:
    #   完全关闭 checkpoint。
    #
    # 注意：
    #   这比旧逻辑更灵活。
    #   旧逻辑是 use_checkpoint=True 就所有 checkpoint 点都开。
    #   新逻辑默认自动判断，只在激活足够大时才 checkpoint。
    # --------------------------------------------------------
    use_checkpoint: bool = False

    # 是否启用自适应 checkpoint。
    # 默认 True。
    #
    # 当 use_checkpoint=False 时生效。
    # 当 use_checkpoint=True 时，强制 checkpoint，忽略 adaptive_checkpoint。
    adaptive_checkpoint: bool = True

    # 自适应 checkpoint 的激活阈值，单位 MB。
    #
    # 只有某个 checkpoint 点估算的可节省激活 >= 该阈值时，
    # 才会实际启用 checkpoint。
    checkpoint_activation_threshold_mb: float = 512.0

    # 时间 chunk 大小。
    # 峰值中间态主要和 [B,D,C,T]、[B,D,T,N,K]、[B,D,T,K] 有关。
    chunk_size: int = 256

    # loss logits 分块大小。
    loss_chunk_size: int = 2048

    # 动态 kernel 是否在 K 维归一化。
    # 当前 kernel_generator 已经在 K 维做 softmax。
    # 因此动态卷积内不会再使用 normalize_kernel，
    # 否则会破坏 softmax 后的概率分布。
    normalize_kernel: bool = True

    # 动态 kernel 初始缩放。
    initial_scale: float = 0.1

    # Source 卷积实现方式。
    source_conv_impl: str = "einsum"

    # Source S2 是否启用：
    # 直接在 source conv 模块内完成：
    #   source conv -> GELU -> dropout -> source mix -> dropout
    # 输出 [B,D,T]。
    source_fused_mixed: bool = True

    # 动态卷积实现方式。
    dynamic_conv_impl: str = "materialized_kernel_stream_x"

    # 保留字段。
    dynamic_conv_tmp_mb_limit: float = 64.0

    # --------------------------------------------------------
    # 非线性残差增强项
    # --------------------------------------------------------
    use_nonlinear_residual: bool = True

    use_source_nonlinear_residual: bool = False

    nonlinear_residual_initial_scale: float = 0.05

    nonlinear_residual_bias: bool = True

    # --------------------------------------------------------
    # 是否使用预分配输出张量来收集 chunk 结果。
    #
    # 默认 True：
    #   避免 chunks list + torch.cat 造成完整序列额外复制峰值。
    #
    # 如果某些 PyTorch/checkpoint 组合触发 inplace/autograd
    # 版本计数问题，可以关闭。
    # --------------------------------------------------------
    use_preallocated_chunk_output: bool = True


# ============================================================
# 2. 参数统计
# ============================================================

def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = 0
    trainable = 0

    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n

    return total, trainable

def _dtype_nbytes(dtype: torch.dtype) -> int:
    """
    返回 dtype 的每元素字节数。
    """

    if dtype in (
        torch.float16,
        torch.bfloat16
    ):
        return 2

    if dtype == torch.float32:
        return 4

    if dtype == torch.float64:
        return 8

    if dtype in (
        torch.int8,
        torch.uint8,
        torch.bool
    ):
        return 1

    if dtype in (
        torch.int16,
        torch.uint16
    ):
        return 2

    if dtype in (
        torch.int32,
        torch.uint32
    ):
        return 4

    if dtype in (
        torch.int64,
        torch.uint64
    ):
        return 8

    return 4


def _bytes_to_mb(num_bytes: float) -> float:
    return float(num_bytes) / float(1024 ** 2)


def _adaptive_checkpoint_decision(
    training: bool,
    force_checkpoint: bool,
    adaptive_checkpoint: bool,
    estimated_activation_mb: float,
    threshold_mb: float
) -> bool:
    """
    统一 checkpoint 决策函数。

    规则：
        1. eval 模式永远不 checkpoint。
        2. force_checkpoint=True 时，训练模式强制 checkpoint。
        3. force_checkpoint=False 且 adaptive_checkpoint=True 时，
           根据 estimated_activation_mb >= threshold_mb 判断。
        4. force_checkpoint=False 且 adaptive_checkpoint=False 时，
           不 checkpoint。
    """

    if not training:
        return False

    if force_checkpoint:
        return True

    if not adaptive_checkpoint:
        return False

    return float(estimated_activation_mb) >= float(threshold_mb)


# ============================================================
# 3. 膨胀层数
# ============================================================

def compute_num_dilated_layers(block_size: int) -> int:
    """
    返回最小 n，使得 2^n > block_size。
    """

    if block_size <= 0:
        raise ValueError("block_size 必须为正数。")

    n = 0
    value = 1

    while value <= block_size:
        value *= 2
        n += 1

    return n


# ============================================================
# 4. 无参数正弦位置编码
# ============================================================

class SinusoidalPositionEncoding(nn.Module):
    """
    无参数正弦位置编码。

    返回：
        pe: [1,L,D]
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(
        self,
        length: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> torch.Tensor:
        position = torch.arange(
            length,
            device=device,
            dtype=dtype
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                self.embed_dim,
                2,
                device=device,
                dtype=dtype
            )
            *
            (-math.log(10000.0) / self.embed_dim)
        )

        pe = torch.zeros(
            length,
            self.embed_dim,
            device=device,
            dtype=dtype
        )

        pe[:, 0::2] = torch.sin(position * div_term)

        if self.embed_dim % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])

        return pe.unsqueeze(0)

class TokenNonlinearResidual(nn.Module):
    """
    逐 token 的非线性残差增强模块。

    输入：
        x: [B,D,T]

    输出：
        out: [B,D,T]

    计算：
        z1 = W1 x
        z2 = W2 z1

        out = scale * (silu(z1) + silu(z2))

    对应残差结构：
        y = x + out + f(x)

    特点：
        1. 只在 D 维做逐 token 通道混合，不跨时间。
        2. 不破坏因果性。
        3. 复杂度随序列长度线性增长。
        4. scale 可学习，建议初始化为 0，使初始模型等价于原模型。
    """

    def __init__(
        self,
        embed_dim: int,
        initial_scale: float = 0.0,
        bias: bool = True
    ):
        super().__init__()

        if embed_dim <= 0:
            raise ValueError("embed_dim 必须为正数。")

        self.embed_dim = embed_dim

        self.w1 = nn.Linear(
            embed_dim,
            embed_dim,
            bias=bias
        )

        self.w2 = nn.Linear(
            embed_dim,
            embed_dim,
            bias=bias
        )

        self.scale = nn.Parameter(
            torch.tensor(float(initial_scale))
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(
            self.w1.weight,
            mean=0.0,
            std=0.02
        )

        if self.w1.bias is not None:
            nn.init.zeros_(
                self.w1.bias
            )

        nn.init.normal_(
            self.w2.weight,
            mean=0.0,
            std=0.02
        )

        if self.w2.bias is not None:
            nn.init.zeros_(
                self.w2.bias
            )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(
                f"x 应为 [B,D,T]，但得到 {x.shape}"
            )

        B, D, T = x.shape

        if D != self.embed_dim:
            raise ValueError(
                f"embed_dim 不匹配，期望 {self.embed_dim}，实际 {D}"
            )

        x_btd = x.transpose(
            1,
            2
        )

        z1 = self.w1(
            x_btd
        )

        z2 = self.w2(
            z1
        )

        out = F.silu(
            z1
        ) + F.silu(
            z2
        )

        out = out.transpose(
            1,
            2
        ).contiguous()

        out = out * self.scale

        return out

# ============================================================
# 5. [B,D,L] 上对 D 做 LayerNorm
# ============================================================

class ChannelLayerNorm(nn.Module):
    """
    对 [B,D,L] 的 D 维做 LayerNorm。
    不跨时间。
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"x 应为 [B,D,L]，但得到 {x.shape}")

        B, D, L = x.shape

        if D != self.embed_dim:
            raise ValueError(
                f"embed_dim 不匹配，期望 {self.embed_dim}，实际 {D}"
            )

        y = x.transpose(1, 2)
        y = self.norm(y)
        y = y.transpose(1, 2).contiguous()

        return y

def build_dilated_source_windows(
    x_full: torch.Tensor,
    start: int,
    end: int,
    kernel_size: int,
    dilation: int
) -> torch.Tensor:
    """
    为 D 维不共享 source 膨胀卷积构造当前 chunk 的 dilated windows。

    输入：
        x_full: [B,D,L]
        start: 当前 chunk 起点
        end: 当前 chunk 终点
        kernel_size: S
        dilation: 膨胀率

    输出：
        windows: [B,D,T,S]

    语义：
        windows[:, :, local_t, j] =
            x_full[:, :, global_t - dilation * (S - 1 - j)]

        j=0   对应最远过去
        j=S-1 对应当前位置

    越界过去位置填 0。
    """

    if x_full.dim() != 3:
        raise ValueError(
            f"x_full 应为 [B,D,L]，但得到 {x_full.shape}"
        )

    if kernel_size <= 0:
        raise ValueError("kernel_size 必须为正数。")

    if dilation <= 0:
        raise ValueError("dilation 必须为正数。")

    B, D, L = x_full.shape

    start_i = int(start)
    end_i = int(end)
    S = int(kernel_size)
    dilation_i = int(dilation)

    if not (0 <= start_i < end_i <= L):
        raise ValueError(
            f"非法 chunk 边界 start={start_i}, end={end_i}, L={L}"
        )

    T = end_i - start_i
    device = x_full.device

    t = torch.arange(
        start_i,
        end_i,
        device=device,
        dtype=torch.long
    )

    windows = x_full.new_zeros(
        B,
        D,
        T,
        S
    )

    for j in range(S):
        offset = dilation_i * (S - 1 - j)

        idx = t - offset

        valid = idx >= 0

        if valid.any():
            idx_clamped = idx.clamp(
                min=0,
                max=L - 1
            )

            x_j = x_full.index_select(
                dim=-1,
                index=idx_clamped
            )

            mask = valid.to(
                dtype=x_full.dtype
            ).view(
                1,
                1,
                T
            )

            windows[
                :,
                :,
                :,
                j
            ] = x_j * mask

    return windows
class ChunkedDUnsharedDilatedSourceConv1d(nn.Module):
    """
    D 维不共享的严格因果膨胀 source 卷积。

    逻辑输入：
        x_full: [B,D,L]

    标准 source 输出：
        source_chunk: [B,D,C,T]

    参数：
        weight: [D,C,S]
        bias:   [D,C]

    卷积语义：
        j=0   对应最远过去
        j=S-1 对应当前位置

        source[:, d, c, t] =
            sum_j x[:, d, t - dilation * (S - 1 - j)] * weight[d, c, j]

    当前实现包含两条路径：

    1. forward_chunk / _source_conv_chunk_grouped_conv1d:
        普通 source conv 路径，返回完整 [B,D,C,T]。

    2. forward_mixed_chunk:
        source_fused_mixed=True 时使用的融合路径。

        不再物化完整 [B,D,C,T]。
        使用:
            windows: [B,D,T,S]

        然后按 source channel C 分块:
            preact_c = einsum("bdts,dcs->bdct", windows, weight_c)
            source_c = activation(preact_c)
            source_c = dropout(source_c)
            mixed_c = einsum("bdct,dc->bdt", source_c, source_mix_weight_c)
            mixed += mixed_c

        当前版本：
            source C-block checkpoint 改为自适应。

    checkpoint 逻辑：
        force_checkpoint=True:
            训练时强制 checkpoint。

        force_checkpoint=False 且 adaptive_checkpoint=True:
            按估算激活大小自适应 checkpoint。

        force_checkpoint=False 且 adaptive_checkpoint=False:
            不 checkpoint。
    """

    def __init__(
        self,
        embed_dim: int,
        source_channels: int,
        kernel_size: int,
        dilation: int,
        bias: bool = True,
        impl: str = "einsum",
        force_checkpoint: bool = False,
        adaptive_checkpoint: bool = True,
        checkpoint_activation_threshold_mb: float = 512.0
    ):
        super().__init__()

        if embed_dim <= 0:
            raise ValueError("embed_dim 必须为正数。")

        if source_channels <= 0:
            raise ValueError("source_channels 必须为正数。")

        if kernel_size <= 0:
            raise ValueError("kernel_size 必须为正数。")

        if dilation <= 0:
            raise ValueError("dilation 必须为正数。")

        if impl not in (
            "einsum",
            "auto",
            "grouped_conv1d"
        ):
            raise ValueError(
                f"source conv impl 必须为 'einsum'、'auto' 或 'grouped_conv1d'，但得到 {impl}"
            )

        if checkpoint_activation_threshold_mb <= 0:
            raise ValueError("checkpoint_activation_threshold_mb 必须为正数。")

        self.embed_dim = embed_dim
        self.source_channels = source_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.impl = impl

        self.force_checkpoint = bool(
            force_checkpoint
        )

        self.adaptive_checkpoint = bool(
            adaptive_checkpoint
        )

        self.checkpoint_activation_threshold_mb = float(
            checkpoint_activation_threshold_mb
        )

        self.weight = nn.Parameter(
            torch.empty(
                embed_dim,
                source_channels,
                kernel_size
            )
        )

        if bias:
            self.bias = nn.Parameter(
                torch.empty(
                    embed_dim,
                    source_channels
                )
            )
        else:
            self.bias = None

        # forward_mixed_chunk 中 C 分块大小。
        #
        # 注意：
        #   这个值越大，单个 C-block 的 [B,D,Cc,T] 激活越大。
        #   当前默认保持 16，比较安全。
        #
        # source C-block checkpoint 是否启用，会根据 Cc 自适应判断。
        self.fused_mixed_c_chunk = 16

        # 普通 forward_chunk 的 D 分块 fallback 上限。
        # 仅在单次 grouped conv 可能触发 32-bit indexing 时使用。
        # 不是模型参数，不影响 state_dict。
        self.grouped_conv_d_chunk_fallback = 128

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(
            self.weight,
            mean=0.0,
            std=0.02
        )

        if self.bias is not None:
            nn.init.zeros_(
                self.bias
            )

    def _build_grouped_conv_segment(
        self,
        x_full: torch.Tensor,
        start: int,
        end: int
    ) -> Tuple[torch.Tensor, int, int, int, int, int, int, int]:
        """
        为 grouped conv 路径构造局部 segment。

        输入:
            x_full: [B,D,L]
            start/end: chunk 区间 [start,end)

        返回:
            segment: [B,D,pad_left+T]
            B, D, L, T, S, dilation_i, pad_left
        """

        if x_full.dim() != 3:
            raise ValueError(
                f"x_full 应为 [B,D,L]，但得到 {x_full.shape}"
            )

        B, D, L = x_full.shape

        if D != self.embed_dim:
            raise ValueError(
                f"embed_dim 不匹配，期望 {self.embed_dim}，实际 {D}"
            )

        start_i = int(start)
        end_i = int(end)

        if not (0 <= start_i < end_i <= L):
            raise ValueError(
                f"非法 chunk 边界 start={start_i}, end={end_i}, L={L}"
            )

        T = end_i - start_i
        S = int(self.kernel_size)
        dilation_i = int(self.dilation)
        pad_left = dilation_i * (S - 1)

        seg_start = max(
            0,
            start_i - pad_left
        )

        seg_end = end_i

        segment = x_full[
            :,
            :,
            seg_start:seg_end
        ]

        actual_left = start_i - seg_start
        missing_left = pad_left - actual_left

        if missing_left > 0:
            zeros = x_full.new_zeros(
                B,
                D,
                missing_left
            )

            segment = torch.cat(
                [
                    zeros,
                    segment
                ],
                dim=-1
            )

        expected_segment_len = pad_left + T

        if segment.size(-1) != expected_segment_len:
            raise RuntimeError(
                f"source conv 局部 segment 长度错误，"
                f"期望 {expected_segment_len}，实际 {segment.size(-1)}，"
                f"start={start_i}, end={end_i}, pad_left={pad_left}, "
                f"seg_start={seg_start}, seg_end={seg_end}"
            )

        return (
            segment,
            B,
            D,
            L,
            T,
            S,
            dilation_i,
            pad_left
        )

    def _source_conv_chunk_grouped_conv1d(
        self,
        x_full: torch.Tensor,
        start: int,
        end: int
    ) -> torch.Tensor:
        """
        标准 grouped Conv1d 实现的 D-unshared dilated source conv。

        输入:
            x_full: [B,D,L]

        输出:
            source: [B,D,C,T]

        当前额外支持:
            当单次 grouped conv 的 input/output 可能超过 CUDA 32-bit indexing
            限制时，自动按 D 维分块执行 grouped conv。
        """

        (
            segment,
            B,
            D,
            L,
            T,
            S,
            dilation_i,
            pad_left
        ) = self._build_grouped_conv_segment(
            x_full=x_full,
            start=start,
            end=end
        )

        C = self.source_channels

        input_numel = segment.numel()
        output_numel = B * D * C * T

        index_limit = 2 ** 31 - 1
        safe_limit = int(index_limit * 0.90)

        need_split_d = (
            input_numel > safe_limit
            or output_numel > safe_limit
        )

        if not need_split_d:
            weight_conv = self.weight.reshape(
                D * C,
                1,
                S
            )

            if self.bias is not None:
                bias_conv = self.bias.reshape(
                    D * C
                )
            else:
                bias_conv = None

            y = F.conv1d(
                segment,
                weight_conv,
                bias_conv,
                stride=1,
                padding=0,
                dilation=dilation_i,
                groups=D
            )

            if y.size(-1) != T:
                y = y[
                    :,
                    :,
                    -T:
                ]

            if y.size(-1) != T:
                raise RuntimeError(
                    f"source grouped conv 输出长度错误，期望 {T}，实际 {y.size(-1)}"
                )

            source = y.reshape(
                B,
                D,
                C,
                T
            )

            return source

        segment_len = segment.size(-1)

        max_dc_by_input = safe_limit // max(
            1,
            B * segment_len
        )

        max_dc_by_output = safe_limit // max(
            1,
            B * C * T
        )

        max_dc = min(
            D,
            max_dc_by_input,
            max_dc_by_output,
            self.grouped_conv_d_chunk_fallback
        )

        max_dc = max(
            1,
            int(max_dc)
        )

        if max_dc >= 128:
            d_chunk = 128
        elif max_dc >= 64:
            d_chunk = 64
        elif max_dc >= 32:
            d_chunk = 32
        elif max_dc >= 16:
            d_chunk = 16
        elif max_dc >= 8:
            d_chunk = 8
        elif max_dc >= 4:
            d_chunk = 4
        elif max_dc >= 2:
            d_chunk = 2
        else:
            d_chunk = 1

        source_chunks: List[torch.Tensor] = []

        for d_start in range(0, D, d_chunk):
            d_end = min(
                d_start + d_chunk,
                D
            )

            Dc = d_end - d_start

            segment_i = segment[
                :,
                d_start:d_end,
                :
            ]

            weight_i = self.weight[
                d_start:d_end,
                :,
                :
            ].contiguous().reshape(
                Dc * C,
                1,
                S
            )

            if self.bias is not None:
                bias_i = self.bias[
                    d_start:d_end,
                    :
                ].contiguous().reshape(
                    Dc * C
                )
            else:
                bias_i = None

            y_i = F.conv1d(
                segment_i,
                weight_i,
                bias_i,
                stride=1,
                padding=0,
                dilation=dilation_i,
                groups=Dc
            )

            if y_i.size(-1) != T:
                y_i = y_i[
                    :,
                    :,
                    -T:
                ]

            if y_i.size(-1) != T:
                raise RuntimeError(
                    f"source grouped conv D 分块输出长度错误，"
                    f"期望 {T}，实际 {y_i.size(-1)}，"
                    f"d_start={d_start}, d_end={d_end}"
                )

            source_i = y_i.reshape(
                B,
                Dc,
                C,
                T
            )

            source_chunks.append(
                source_i
            )

            del segment_i
            del weight_i
            del bias_i
            del y_i
            del source_i

        source = torch.cat(
            source_chunks,
            dim=1
        )

        del source_chunks

        expected_shape = (
            B,
            D,
            C,
            T
        )

        if source.shape != expected_shape:
            raise RuntimeError(
                f"source 分块 grouped conv 输出 shape 错误，"
                f"期望 {expected_shape}，实际 {source.shape}"
            )

        return source

    def _source_conv_chunk(
        self,
        x_full: torch.Tensor,
        start: int,
        end: int
    ) -> torch.Tensor:
        """
        source conv 统一入口。

        当前:
            "einsum"
            "auto"
            "grouped_conv1d"

        在普通 forward_chunk 路径均走 grouped F.conv1d。
        若张量过大，会在 _source_conv_chunk_grouped_conv1d 内部按 D 分块 fallback。
        """

        if self.impl in (
            "einsum",
            "auto",
            "grouped_conv1d"
        ):
            return self._source_conv_chunk_grouped_conv1d(
                x_full=x_full,
                start=start,
                end=end
            )

        raise ValueError(
            f"未知 source conv impl: {self.impl}"
        )

    def forward_chunk(
        self,
        x_full: torch.Tensor,
        start: int,
        end: int
    ) -> torch.Tensor:
        """
        标准 source conv chunk。

        输入：
            x_full: [B,D,L]

        输出：
            source_chunk: [B,D,C,T]

        注意:
            如果 C 和 T 很大，本函数仍然会返回完整 [B,D,C,T]。
            大配置训练建议使用 source_fused_mixed=True，
            走 forward_mixed_chunk，避免物化完整 source。
        """

        source = self._source_conv_chunk(
            x_full=x_full,
            start=start,
            end=end
        )

        return source

    def _choose_fused_mixed_c_chunk(
        self,
        B: int,
        D: int,
        T: int,
        dtype: torch.dtype
    ) -> int:
        """
        为 forward_mixed_chunk 选择 source channel 分块大小。

        目标:
            1. 避免单个 [B,D,Cc,T] 过大。
            2. 尽量不把 C 分块切得太小，减少性能损失。

        默认上限来自:
            self.fused_mixed_c_chunk
        """

        base_c_chunk = max(
            1,
            int(self.fused_mixed_c_chunk)
        )

        index_limit = 2 ** 31 - 1
        safe_limit = int(index_limit * 0.50)

        max_c_by_index = safe_limit // max(
            1,
            B * D * T
        )

        max_c = min(
            self.source_channels,
            base_c_chunk,
            max_c_by_index
        )

        max_c = max(
            1,
            int(max_c)
        )

        if max_c >= 64:
            c_chunk = 64
        elif max_c >= 32:
            c_chunk = 32
        elif max_c >= 16:
            c_chunk = 16
        elif max_c >= 8:
            c_chunk = 8
        elif max_c >= 4:
            c_chunk = 4
        elif max_c >= 2:
            c_chunk = 2
        else:
            c_chunk = 1

        return c_chunk

    def _estimate_source_c_block_checkpoint_mb(
        self,
        B: int,
        D: int,
        T: int,
        Cc: int,
        dtype: torch.dtype
    ) -> float:
        """
        估算 source fused mixed 单个 C-block checkpoint 可节省激活。

        source_c_block 内部主要产生：
            preact_c: [B,D,Cc,T]
            GELU 输出: [B,D,Cc,T]
            dropout 输出/mask: 近似 [B,D,Cc,T]
            einsum 相关反传保存/中间态

        windows: [B,D,T,S] 是 block 外部输入，不计入 checkpoint
        可以节省的内部激活。

        使用保守系数 4.0：
            estimated = 4 * B * D * Cc * T * bytes
        """

        bytes_per_elem = _dtype_nbytes(
            dtype
        )

        estimated_bytes = (
            4.0
            * float(B)
            * float(D)
            * float(Cc)
            * float(T)
            * float(bytes_per_elem)
        )

        return _bytes_to_mb(
            estimated_bytes
        )

    def _should_checkpoint_source_c_block(
        self,
        B: int,
        D: int,
        T: int,
        Cc: int,
        dtype: torch.dtype
    ) -> bool:
        """
        source C-block checkpoint 自适应决策。
        """

        estimated_mb = self._estimate_source_c_block_checkpoint_mb(
            B=B,
            D=D,
            T=T,
            Cc=Cc,
            dtype=dtype
        )

        return _adaptive_checkpoint_decision(
            training=self.training,
            force_checkpoint=self.force_checkpoint,
            adaptive_checkpoint=self.adaptive_checkpoint,
            estimated_activation_mb=estimated_mb,
            threshold_mb=self.checkpoint_activation_threshold_mb
        )

    def forward_mixed_chunk(
        self,
        x_full: torch.Tensor,
        start: int,
        end: int,
        source_mix_weight: torch.Tensor,
        source_mix_bias: torch.Tensor,
        activation: nn.Module,
        dropout_module: nn.Module,
        post_mix_dropout_module: nn.Module
    ) -> torch.Tensor:
        """
        S2 融合路径。

        直接完成：
            source conv -> GELU -> dropout -> source mix -> dropout

        输入：
            x_full: [B,D,L]
            source_mix_weight: [D,C]
            source_mix_bias:   [D]

        输出：
            mixed_chunk: [B,D,T]

        当前版本:
            1. 不再先完整生成 source: [B,D,C,T]。
            2. 构造 windows: [B,D,T,S]。
            3. 按 C 分块计算 source conv + activation + dropout + source mix。
            4. source C-block checkpoint 改为自适应：
                - force_checkpoint=True 时强制 checkpoint；
                - adaptive_checkpoint=True 时按估算激活大小判断；
                - adaptive_checkpoint=False 时关闭。
        """

        if x_full.dim() != 3:
            raise ValueError(
                f"x_full 应为 [B,D,L]，但得到 {x_full.shape}"
            )

        B, D, L = x_full.shape

        if D != self.embed_dim:
            raise ValueError(
                f"embed_dim 不匹配，期望 {self.embed_dim}，实际 {D}"
            )

        if source_mix_weight.shape != (
            self.embed_dim,
            self.source_channels
        ):
            raise ValueError(
                f"source_mix_weight shape 错误，期望 "
                f"{(self.embed_dim, self.source_channels)}，"
                f"实际 {source_mix_weight.shape}"
            )

        if source_mix_bias.shape != (
            self.embed_dim,
        ):
            raise ValueError(
                f"source_mix_bias shape 错误，期望 {(self.embed_dim,)}，"
                f"实际 {source_mix_bias.shape}"
            )

        start_i = int(start)
        end_i = int(end)

        if not (0 <= start_i < end_i <= L):
            raise ValueError(
                f"非法 chunk 边界 start={start_i}, end={end_i}, L={L}"
            )

        T = end_i - start_i
        S = int(self.kernel_size)

        windows = build_dilated_source_windows(
            x_full=x_full,
            start=start_i,
            end=end_i,
            kernel_size=S,
            dilation=int(self.dilation)
        )
        # [B,D,T,S]

        mixed = x_full.new_zeros(
            B,
            D,
            T
        )

        C = self.source_channels

        c_chunk = self._choose_fused_mixed_c_chunk(
            B=B,
            D=D,
            T=T,
            dtype=x_full.dtype
        )

        for c_start in range(0, C, c_chunk):
            c_end = min(
                c_start + c_chunk,
                C
            )

            Cc = c_end - c_start

            weight_c = self.weight[
                :,
                c_start:c_end,
                :
            ]

            if self.bias is not None:
                bias_c = self.bias[
                    :,
                    c_start:c_end
                ]
            else:
                bias_c = None

            mix_c = source_mix_weight[
                :,
                c_start:c_end
            ]

            def source_c_block(
                windows_ref: torch.Tensor,
                weight_ref: torch.Tensor,
                mix_ref: torch.Tensor,
                bias_ref: Optional[torch.Tensor],
                D_ref: int = D,
                Cc_ref: int = Cc
            ) -> torch.Tensor:
                preact_c = torch.einsum(
                    "bdts,dcs->bdct",
                    windows_ref,
                    weight_ref
                )

                if bias_ref is not None:
                    preact_c = preact_c + bias_ref.view(
                        1,
                        D_ref,
                        Cc_ref,
                        1
                    )

                source_c = activation(
                    preact_c
                )

                source_c = dropout_module(
                    source_c
                )

                mixed_c = torch.einsum(
                    "bdct,dc->bdt",
                    source_c,
                    mix_ref
                )

                return mixed_c

            should_checkpoint = self._should_checkpoint_source_c_block(
                B=B,
                D=D,
                T=T,
                Cc=Cc,
                dtype=x_full.dtype
            )

            if should_checkpoint:
                mixed_c = checkpoint(
                    source_c_block,
                    windows,
                    weight_c,
                    mix_c,
                    bias_c,
                    use_reentrant=False,
                    preserve_rng_state=True
                )
            else:
                mixed_c = source_c_block(
                    windows,
                    weight_c,
                    mix_c,
                    bias_c
                )

            mixed = mixed + mixed_c

            del weight_c
            del bias_c
            del mix_c
            del mixed_c

        del windows

        mixed = mixed + source_mix_bias.view(
            1,
            D,
            1
        )

        mixed = post_mix_dropout_module(
            mixed
        )

        return mixed


# 为了兼容旧名称，保留别名。
# 如果外部代码仍然引用 ChunkedDSharedDilatedSourceConv1d，
# 实际会得到新的 D 维不共享实现。
ChunkedDSharedDilatedSourceConv1d = ChunkedDUnsharedDilatedSourceConv1d
# ============================================================
# 8. source 通道混合 C -> 1
# ============================================================
class SourceChannelMix(nn.Module):
    """
    将 source_chunk: [B,D,C,T] 混合为 [B,D,T]。

    和动态卷积里的 kernel_mix 类似：
    - 参数为 [D,C]
    - 不做时间卷积
    - 不产生 cache
    - 不共享 D

    注意：
        在启用 source_fused_mixed=True 时，
        DilatedUnsharedLayer 不再直接调用本模块的 forward，
        而是把 self.mix 和 self.bias 传给 source_conv.forward_mixed_chunk。
        这样可以缩短 [B,D,C,T] 的生命周期。
    """

    def __init__(
        self,
        embed_dim: int,
        source_channels: int
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.source_channels = source_channels

        self.mix = nn.Parameter(
            torch.ones(
                embed_dim,
                source_channels
            ) / float(source_channels)
        )

        self.bias = nn.Parameter(
            torch.zeros(embed_dim)
        )

    def forward(self, source_chunk: torch.Tensor) -> torch.Tensor:
        if source_chunk.dim() != 4:
            raise ValueError(
                f"source_chunk 应为 [B,D,C,T]，但得到 {source_chunk.shape}"
            )

        B, D, C, T = source_chunk.shape

        if D != self.embed_dim:
            raise ValueError(
                f"embed_dim 不匹配，期望 {self.embed_dim}，实际 {D}"
            )

        if C != self.source_channels:
            raise ValueError(
                f"source_channels 不匹配，期望 {self.source_channels}，实际 {C}"
            )

        out = torch.einsum(
            "bdct,dc->bdt",
            source_chunk,
            self.mix
        )

        out = out + self.bias.view(
            1,
            D,
            1
        )

        return out

class StreamingKernelGenConv1d(nn.Module):
    """
    kernel 生成分支，D-mix + depthwise causal Conv1d 版本。

    输入:
        mixed_source_chunk/full: [B,D,T] 或 [B,D,L]

    输出:
        kernel_chunk: [B,T,N,K]

    语义:
        1. 先对 D 维做逐 token mix:
               [B,D,T] -> [B,N*K,T]

           每个 (N,K) 位置有自己的一组 D-mix 参数:
               d_mix.weight: [N*K,D,1]
               d_mix.bias:   [N*K]

        2. 在 [N*K] 通道维做 LayerNorm:
               [B,N*K,S] -> transpose view -> [B,S,N*K] -> LN -> [B,N*K,S]

        3. 对 N*K 个通道分别做 depthwise 严格因果 Conv1d:
               Conv1d(
                   in_channels=N*K,
                   out_channels=N*K,
                   kernel_size=kernel_gen_kernel_size,
                   groups=N*K
               )

        4. 残差连接:
               logits_chunk = identity_chunk + depthwise_conv(LN(local_segment))

        5. reshape:
               [B,N*K,T] -> [B,N,K,T] -> [B,T,N,K]

        6. 在 K 维 softmax:
               kernel = softmax(kernel, dim=-1)

    当前性能优化:
        _apply_norm_bct 中去掉 transpose 后的第一次 contiguous。

        原来:
            y = x.transpose(1, 2).contiguous()

        现在:
            y = x.transpose(1, 2)

        对 D=1024,N*K=8192,T=32768 的配置，
        这可以减少一次约 512MB 的大拷贝。
    """

    def __init__(
        self,
        embed_dim: int,
        num_kernels: int,
        dynamic_kernel_size: int,
        kernel_gen_kernel_size: int,
        bias: bool = True
    ):
        super().__init__()

        if embed_dim <= 0:
            raise ValueError("embed_dim 必须为正数。")

        if num_kernels <= 0:
            raise ValueError("num_kernels 必须为正数。")

        if dynamic_kernel_size <= 0:
            raise ValueError("dynamic_kernel_size 必须为正数。")

        if kernel_gen_kernel_size <= 0:
            raise ValueError("kernel_gen_kernel_size 必须为正数。")

        self.embed_dim = embed_dim
        self.num_kernels = num_kernels
        self.dynamic_kernel_size = dynamic_kernel_size
        self.kernel_gen_kernel_size = kernel_gen_kernel_size
        self.padding_left = kernel_gen_kernel_size - 1

        self.kernel_channels = num_kernels * dynamic_kernel_size

        self.d_mix = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=self.kernel_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True
        )

        self.norm = nn.LayerNorm(
            self.kernel_channels
        )

        self.depthwise_conv = nn.Conv1d(
            in_channels=self.kernel_channels,
            out_channels=self.kernel_channels,
            kernel_size=kernel_gen_kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=self.kernel_channels,
            bias=bias
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(
            self.d_mix.weight,
            mean=0.0,
            std=1.0 / math.sqrt(float(self.embed_dim))
        )

        if self.d_mix.bias is not None:
            nn.init.zeros_(
                self.d_mix.bias
            )

        nn.init.normal_(
            self.depthwise_conv.weight,
            mean=0.0,
            std=0.02
        )

        if self.depthwise_conv.bias is not None:
            nn.init.zeros_(
                self.depthwise_conv.bias
            )

        nn.init.ones_(
            self.norm.weight
        )

        nn.init.zeros_(
            self.norm.bias
        )

    def make_empty_cache(
        self,
        B: int,
        D: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """
        返回输入 cache:
            [B,D,1,padding_left]

        注意:
            cache 仍然缓存 D-mix 之前的输入，因为 D-mix 在最后因果卷积之前。
        """

        if D != self.embed_dim:
            raise ValueError(
                f"embed_dim 不匹配，期望 {self.embed_dim}，实际 {D}"
            )

        return torch.zeros(
            B,
            D,
            1,
            self.padding_left,
            device=device,
            dtype=dtype
        )

    def _apply_norm_bct(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        对 [B,M,T] 的 M 维做 LayerNorm。

        输入:
            x: [B,M,T]

        输出:
            y: [B,M,T]

        性能优化:
            去掉 transpose 后的第一次 contiguous。

            旧写法:
                y = x.transpose(1, 2).contiguous()
                y = self.norm(y)
                y = y.transpose(1, 2).contiguous()

            新写法:
                y = x.transpose(1, 2)
                y = self.norm(y)
                y = y.transpose(1, 2).contiguous()

            LayerNorm 可以接受非 contiguous 的 [B,T,M] view。
            最后仍然 contiguous 回 [B,M,T]，供后续 Conv1d 使用。

        对当前大配置:
            x: [1,8192,32768]
            单次 contiguous copy 约 512MB。
        """

        if x.dim() != 3:
            raise ValueError(
                f"x 应为 [B,M,T]，但得到 {x.shape}"
            )

        B, M, T = x.shape

        if M != self.kernel_channels:
            raise ValueError(
                f"kernel_channels 不匹配，期望 {self.kernel_channels}，实际 {M}"
            )

        y = x.transpose(
            1,
            2
        )
        # [B,T,M]，非 contiguous view

        y = self.norm(
            y
        )
        # [B,T,M]

        y = y.transpose(
            1,
            2
        ).contiguous()
        # [B,M,T]

        return y

    def _logits_to_kernel(
        self,
        logits: torch.Tensor,
        expected_T: int
    ) -> torch.Tensor:
        """
        输入:
            logits: [B,N*K,T]

        输出:
            kernel: [B,T,N,K]
        """

        if logits.dim() != 3:
            raise ValueError(
                f"logits 应为 [B,N*K,T]，但得到 {logits.shape}"
            )

        B, M, T = logits.shape

        if M != self.kernel_channels:
            raise ValueError(
                f"logits 通道数错误，期望 {self.kernel_channels}，实际 {M}"
            )

        expected_T_i = int(expected_T)

        if T != expected_T_i:
            raise RuntimeError(
                f"logits 长度错误，期望 {expected_T_i}，实际 {T}"
            )

        y = logits.reshape(
            B,
            self.num_kernels,
            self.dynamic_kernel_size,
            T
        )
        # [B,N,K,T]

        kernel = y.permute(
            0,
            3,
            1,
            2
        ).contiguous()
        # [B,T,N,K]

        kernel = F.softmax(
            kernel,
            dim=-1
        )

        return kernel

    def _forward_from_padded_input(
        self,
        x_padded: torch.Tensor,
        expected_T: int
    ) -> torch.Tensor:
        """
        输入:
            x_padded: [B,D,pad+T]

        输出:
            kernel: [B,T,N,K]

        计算:
            1. D-mix:
                   u_padded = d_mix(x_padded)
                   [B,D,pad+T] -> [B,N*K,pad+T]

            2. 局部 LayerNorm:
                   u_norm = LN(u_padded)

            3. depthwise causal conv:
                   conv_out = depthwise_conv(u_norm)
                   [B,N*K,pad+T] -> [B,N*K,T]

            4. residual:
                   identity = u_padded[:, :, -T:]
                   logits = identity + conv_out

            5. reshape + softmax:
                   [B,N*K,T] -> [B,T,N,K]
        """

        if x_padded.dim() != 3:
            raise ValueError(
                f"x_padded 应为 [B,D,S]，但得到 {x_padded.shape}"
            )

        B, D, S = x_padded.shape

        if D != self.embed_dim:
            raise ValueError(
                f"embed_dim 不匹配，期望 {self.embed_dim}，实际 {D}"
            )

        T = int(expected_T)

        if T <= 0:
            raise ValueError("expected_T 必须为正数。")

        expected_len = self.padding_left + T

        if S != expected_len:
            raise RuntimeError(
                f"x_padded 长度错误，期望 {expected_len}，实际 {S}"
            )

        u_padded = self.d_mix(
            x_padded
        )
        # [B,N*K,pad+T]

        u_norm = self._apply_norm_bct(
            u_padded
        )
        # [B,N*K,pad+T]

        conv_out = self.depthwise_conv(
            u_norm
        )
        # [B,N*K,T]

        if conv_out.size(-1) != T:
            conv_out = conv_out[
                :,
                :,
                -T:
            ]

        if conv_out.size(-1) != T:
            raise RuntimeError(
                f"kernel generator depthwise conv 输出长度错误，"
                f"期望 {T}，实际 {conv_out.size(-1)}"
            )

        identity = u_padded[
            :,
            :,
            -T:
        ]

        logits = identity + conv_out
        # [B,N*K,T]

        kernel = self._logits_to_kernel(
            logits=logits,
            expected_T=T
        )

        return kernel

    def forward_chunk(
        self,
        mixed_source_chunk: torch.Tensor,
        cache: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        streaming cache 版本。

        输入:
            mixed_source_chunk: [B,D,T]
            cache:
                None 或 [B,D,1,padding_left]

        输出:
            kernel:    [B,T,N,K]
            new_cache: [B,D,1,padding_left]
        """

        if mixed_source_chunk.dim() != 3:
            raise ValueError(
                f"mixed_source_chunk 应为 [B,D,T]，但得到 {mixed_source_chunk.shape}"
            )

        B, D, T = mixed_source_chunk.shape

        if D != self.embed_dim:
            raise ValueError(
                f"embed_dim 不匹配，期望 {self.embed_dim}，实际 {D}"
            )

        if cache is None:
            cache = self.make_empty_cache(
                B=B,
                D=D,
                device=mixed_source_chunk.device,
                dtype=mixed_source_chunk.dtype
            )

        expected_cache_shape = (
            B,
            D,
            1,
            self.padding_left
        )

        if cache.shape != expected_cache_shape:
            raise ValueError(
                f"kernel cache shape 错误，期望 {expected_cache_shape}，实际 {cache.shape}"
            )

        cache_flat = cache.squeeze(
            2
        )
        # [B,D,pad]

        x_padded = torch.cat(
            [
                cache_flat,
                mixed_source_chunk
            ],
            dim=-1
        )
        # [B,D,pad+T]

        kernel = self._forward_from_padded_input(
            x_padded=x_padded,
            expected_T=T
        )

        if self.padding_left > 0:
            new_cache_flat = x_padded[
                :,
                :,
                -self.padding_left:
            ]
        else:
            new_cache_flat = x_padded[
                :,
                :,
                :0
            ]

        new_cache = new_cache_flat.unsqueeze(
            2
        ).contiguous()
        # [B,D,1,pad]

        return kernel, new_cache

    def forward_chunk_from_full(
        self,
        mixed_source_full: torch.Tensor,
        start: int,
        end: int
    ) -> torch.Tensor:
        """
        无 cache 的局部因果版本。

        输入:
            mixed_source_full: [B,D,L]
            start: 当前 chunk 起点
            end: 当前 chunk 终点

        输出:
            kernel: [B,T,N,K]

        语义:
            与完整序列计算等价。

        对当前 chunk [start,end)，depthwise causal conv 只需要额外看到左侧
        padding_left 个 token:

            seg_start = max(0, start - padding_left)
            seg_end   = end

        如果 start 左侧不足 padding_left，则左侧补 0。
        """

        if mixed_source_full.dim() != 3:
            raise ValueError(
                f"mixed_source_full 应为 [B,D,L]，但得到 {mixed_source_full.shape}"
            )

        B, D, L = mixed_source_full.shape

        if D != self.embed_dim:
            raise ValueError(
                f"embed_dim 不匹配，期望 {self.embed_dim}，实际 {D}"
            )

        start_i = int(start)
        end_i = int(end)

        if not (0 <= start_i < end_i <= L):
            raise ValueError(
                f"非法 chunk 边界 start={start_i}, end={end_i}, L={L}"
            )

        T = end_i - start_i
        pad = self.padding_left

        seg_start = max(
            0,
            start_i - pad
        )

        seg_end = end_i

        segment = mixed_source_full[
            :,
            :,
            seg_start:seg_end
        ]

        actual_left = start_i - seg_start
        missing_left = pad - actual_left

        if missing_left > 0:
            zeros = mixed_source_full.new_zeros(
                B,
                D,
                missing_left
            )

            segment = torch.cat(
                [
                    zeros,
                    segment
                ],
                dim=-1
            )

        expected_segment_len = pad + T

        if segment.size(-1) != expected_segment_len:
            raise RuntimeError(
                f"kernel generator 局部 segment 长度错误，"
                f"期望 {expected_segment_len}，实际 {segment.size(-1)}，"
                f"start={start_i}, end={end_i}, pad={pad}, "
                f"seg_start={seg_start}, seg_end={seg_end}"
            )

        kernel = self._forward_from_padded_input(
            x_padded=segment,
            expected_T=T
        )

        return kernel


def build_local_norm_segment_for_causal_windows(
    x_full: torch.Tensor,
    norm: nn.LayerNorm,
    start: int,
    end: int,
    kernel_size: int
) -> Tuple[torch.Tensor, int]:
    """
    为动态卷积构造当前 chunk 所需的局部 LayerNorm 片段。

    输入：
        x_full: [B,D,L]
        norm: nn.LayerNorm(embed_dim)
        start/end: 当前 chunk 区间 [start,end)
        kernel_size: K

    输出：
        segment_norm: [B,D,S]
        seg_start: int

    其中：
        seg_start = max(0, start - K + 1)
        seg_end = end
        S = seg_end - seg_start

    这个 segment_norm 等价于完整 x_full 在该局部区间上的 LN_D 结果。
    """

    if x_full.dim() != 3:
        raise ValueError(
            f"x_full 应为 [B,D,L]，但得到 {x_full.shape}"
        )

    if kernel_size <= 0:
        raise ValueError("kernel_size 必须为正数。")

    B, D, L = x_full.shape

    start_i = int(start)
    end_i = int(end)
    K = int(kernel_size)

    if not (0 <= start_i < end_i <= L):
        raise ValueError(
            f"非法 chunk 边界 start={start_i}, end={end_i}, L={L}"
        )

    seg_start = max(
        0,
        start_i - K + 1
    )

    seg_end = end_i

    segment = x_full[
        :,
        :,
        seg_start:seg_end
    ]

    segment_bsd = segment.transpose(
        1,
        2
    ).contiguous()

    segment_bsd = norm(segment_bsd)

    segment_norm = segment_bsd.transpose(
        1,
        2
    ).contiguous()

    return segment_norm, seg_start

class ChunkedUnsharedDynamicConvLayer(nn.Module):
    """
    单层动态卷积。

    输入：
        h_full:       [B,D,L]
        kernel_chunk: [B,T,N,K]

    输出：
        h_chunk_out: [B,D,T]

    当前版本:
        1. kernel_generator 输出的动态卷积核不带 D 维:
               [B,T,N,K]

        2. kernel_mix 保持 [D,N]。

        3. stream_k 路径:
               不物化完整 [B,D,T,K]。
               每个 K offset 单独计算:
                   kernel_r: [B,T,N]
                   mixed_r = kernel_r @ kernel_mix.T -> [B,T,D] -> [B,D,T]

        4. materialized_kernel_stream_x 路径:
               新增性能路径。
               一次性物化:
                   mixed_btkd: [B,T,K,D]

               但不物化:
                   windows: [B,D,T,K]

               这样把 K 次 matmul 合并成一次 batched matmul，
               用约 [B,T,K,D] 的额外显存换性能。

        5. materialized_windows 路径:
               调试/对比路径。
               会同时物化:
                   windows:      [B,D,T,K]
                   mixed_kernel: [B,D,T,K]
               大配置下不建议默认使用。
    """

    def __init__(
        self,
        embed_dim: int,
        num_kernels: int,
        dynamic_kernel_size: int,
        mlp_ratio: float,
        dropout: float,
        normalize_kernel: bool = True,
        initial_scale: float = 0.1,
        dynamic_conv_impl: str = "stream_k",
        dynamic_conv_tmp_mb_limit: float = 64.0,
        use_nonlinear_residual: bool = False,
        nonlinear_residual_initial_scale: float = 0.0,
        nonlinear_residual_bias: bool = True
    ):
        super().__init__()

        if embed_dim <= 0:
            raise ValueError("embed_dim 必须为正数。")

        if num_kernels <= 0:
            raise ValueError("num_kernels 必须为正数。")

        if dynamic_kernel_size <= 0:
            raise ValueError("dynamic_kernel_size 必须为正数。")

        if mlp_ratio <= 0:
            raise ValueError("mlp_ratio 必须为正数。")

        if dynamic_conv_impl not in (
            "stream_k",
            "materialized_kernel_stream_x",
            "materialized_windows",
            "auto"
        ):
            raise ValueError(
                f"dynamic_conv_impl 必须为 'stream_k'、"
                f"'materialized_kernel_stream_x'、"
                f"'materialized_windows' 或 'auto'，但得到 {dynamic_conv_impl}"
            )

        if dynamic_conv_tmp_mb_limit <= 0:
            raise ValueError("dynamic_conv_tmp_mb_limit 必须为正数。")

        self.embed_dim = embed_dim
        self.num_kernels = num_kernels
        self.dynamic_kernel_size = dynamic_kernel_size

        self.normalize_kernel = normalize_kernel

        self.dynamic_conv_impl = dynamic_conv_impl
        self.dynamic_conv_tmp_mb_limit = float(
            dynamic_conv_tmp_mb_limit
        )

        self.use_nonlinear_residual = bool(
            use_nonlinear_residual
        )

        self.norm1 = nn.LayerNorm(
            embed_dim
        )

        self.kernel_scale = nn.Parameter(
            torch.tensor(
                float(initial_scale)
            )
        )

        self.kernel_mix = nn.Parameter(
            torch.ones(
                embed_dim,
                num_kernels
            ) / float(num_kernels)
        )

        self.out_proj = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=1,
            bias=True
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.norm2 = nn.LayerNorm(
            embed_dim
        )

        hidden_dim = int(
            embed_dim * mlp_ratio
        )

        self.mlp = nn.Sequential(
            nn.Linear(
                embed_dim,
                hidden_dim
            ),
            nn.GELU(),
            nn.Linear(
                hidden_dim,
                embed_dim
            ),
            nn.Dropout(
                dropout
            )
        )

        if self.use_nonlinear_residual:
            self.nonlinear_residual1 = TokenNonlinearResidual(
                embed_dim=embed_dim,
                initial_scale=nonlinear_residual_initial_scale,
                bias=nonlinear_residual_bias
            )

            self.nonlinear_residual2 = TokenNonlinearResidual(
                embed_dim=embed_dim,
                initial_scale=nonlinear_residual_initial_scale,
                bias=nonlinear_residual_bias
            )
        else:
            self.nonlinear_residual1 = None
            self.nonlinear_residual2 = None

    def _normalize_kernel(
        self,
        kernel: torch.Tensor
    ) -> torch.Tensor:
        """
        旧版动态卷积核归一化函数。

        当前默认不会调用它，因为 kernel_generator 已经在 K 维做 softmax。
        保留该函数用于兼容旧实验或未来切换非 softmax kernel。
        """

        mean = kernel.mean(
            dim=-1,
            keepdim=True
        )

        var = (
            kernel - mean
        ).pow(
            2
        ).mean(
            dim=-1,
            keepdim=True
        )

        kernel = (
            kernel - mean
        ) / torch.sqrt(
            var + 1e-5
        )

        return kernel

    def _validate_kernel_shape(
        self,
        h_full: torch.Tensor,
        kernel_chunk: torch.Tensor,
        start: int,
        end: int
    ) -> Tuple[int, int, int, int, int, int]:
        """
        校验 kernel shape。

        h_full:
            [B,D,L]

        kernel_chunk:
            [B,T,N,K]
        """

        if h_full.dim() != 3:
            raise ValueError(
                f"h_full 应为 [B,D,L]，但得到 {h_full.shape}"
            )

        if kernel_chunk.dim() != 4:
            raise ValueError(
                f"kernel_chunk 应为 [B,T,N,K]，但得到 {kernel_chunk.shape}"
            )

        B, D, L = h_full.shape

        if D != self.embed_dim:
            raise ValueError(
                f"embed_dim 不匹配，期望 {self.embed_dim}，实际 {D}"
            )

        start_i = int(start)
        end_i = int(end)

        if not (0 <= start_i < end_i <= L):
            raise ValueError(
                f"非法 chunk 边界 start={start_i}, end={end_i}, L={L}"
            )

        T = end_i - start_i

        Bk, Tk, N, K = kernel_chunk.shape

        if Bk != B or Tk != T:
            raise ValueError(
                f"kernel_chunk shape 与 h_full/chunk 不匹配，"
                f"h_full={h_full.shape}, start={start_i}, end={end_i}, "
                f"kernel_chunk={kernel_chunk.shape}"
            )

        if N != self.num_kernels:
            raise ValueError(
                f"num_kernels 不匹配，期望 {self.num_kernels}，实际 {N}"
            )

        if K != self.dynamic_kernel_size:
            raise ValueError(
                f"dynamic_kernel_size 不匹配，期望 {self.dynamic_kernel_size}，实际 {K}"
            )

        return B, D, L, T, N, K

    def _mix_kernel_n_dimension(
        self,
        kernel_chunk: torch.Tensor
    ) -> torch.Tensor:
        """
        融合完整 N 维。

        输入：
            kernel_chunk: [B,T,N,K]

        参数：
            kernel_mix: [D,N]

        输出：
            mixed_kernel: [B,D,T,K]

        注意：
            该函数会生成完整 [B,D,T,K]。
            大配置下不建议默认调用。
        """

        if kernel_chunk.dim() != 4:
            raise ValueError(
                f"kernel_chunk 应为 [B,T,N,K]，但得到 {kernel_chunk.shape}"
            )

        B, T, N, K = kernel_chunk.shape

        if N != self.num_kernels:
            raise ValueError(
                f"num_kernels 不匹配，期望 {self.num_kernels}，实际 {N}"
            )

        if K != self.dynamic_kernel_size:
            raise ValueError(
                f"dynamic_kernel_size 不匹配，期望 {self.dynamic_kernel_size}，实际 {K}"
            )

        mixed_btkd = torch.matmul(
            kernel_chunk.permute(
                0,
                1,
                3,
                2
            ).contiguous(),
            self.kernel_mix.transpose(
                0,
                1
            )
        )
        # [B,T,K,D]

        mixed_kernel = mixed_btkd.permute(
            0,
            3,
            1,
            2
        ).contiguous()
        # [B,D,T,K]

        return mixed_kernel

    def _mix_kernel_single_offset(
        self,
        kernel_r: torch.Tensor
    ) -> torch.Tensor:
        """
        融合单个 K offset 的 N 维。

        输入:
            kernel_r: [B,T,N]

        参数:
            kernel_mix: [D,N]

        输出:
            mixed_r: [B,D,T]

        计算:
            mixed_r[b,d,t] =
                sum_n kernel_r[b,t,n] * kernel_mix[d,n]

        实现:
            使用标准 matmul:

                [B,T,N] @ [N,D] -> [B,T,D]
                transpose -> [B,D,T]

        该路径不会生成:
            [B,D,T,K]
        """

        if kernel_r.dim() != 3:
            raise ValueError(
                f"kernel_r 应为 [B,T,N]，但得到 {kernel_r.shape}"
            )

        B, T, N = kernel_r.shape

        if N != self.num_kernels:
            raise ValueError(
                f"kernel_r 的 N 维错误，期望 {self.num_kernels}，实际 {N}"
            )

        mixed_btd = torch.matmul(
            kernel_r,
            self.kernel_mix.transpose(
                0,
                1
            )
        )
        # [B,T,D]

        mixed_r = mixed_btd.transpose(
            1,
            2
        ).contiguous()
        # [B,D,T]

        return mixed_r

    def _apply_dynamic_conv_stream_k(
        self,
        h_full: torch.Tensor,
        kernel_chunk: torch.Tensor,
        start: int,
        end: int
    ) -> torch.Tensor:
        """
        默认显存安全 stream_k 路径。

        不显式构造:
            windows:      [B,D,T,K]
            mixed_kernel: [B,D,T,K]

        每个 offset r 内部只构造:
            x_r:     [B,D,T]
            mixed_r: [B,D,T]

        mixed_r 使用标准 matmul 计算。
        """

        B, D, L, T, N, K = self._validate_kernel_shape(
            h_full=h_full,
            kernel_chunk=kernel_chunk,
            start=start,
            end=end
        )

        start_i = int(start)
        end_i = int(end)

        segment_norm, seg_start = build_local_norm_segment_for_causal_windows(
            x_full=h_full,
            norm=self.norm1,
            start=start_i,
            end=end_i,
            kernel_size=K
        )
        # [B,D,S]

        S = segment_norm.size(-1)
        device = h_full.device

        out = h_full.new_zeros(
            B,
            D,
            T
        )

        t = torch.arange(
            start_i,
            end_i,
            device=device,
            dtype=torch.long
        )

        for r in range(K):
            offset = K - 1 - r
            global_idx = t - offset
            valid = global_idx >= 0

            if valid.any():
                local_idx = global_idx - seg_start

                local_idx_clamped = local_idx.clamp(
                    min=0,
                    max=S - 1
                )

                x_r = segment_norm.index_select(
                    dim=-1,
                    index=local_idx_clamped
                )
                # [B,D,T]

                mask = valid.to(
                    dtype=h_full.dtype
                ).view(
                    1,
                    1,
                    T
                )

                x_r = x_r * mask

                kernel_r = kernel_chunk[
                    :,
                    :,
                    :,
                    r
                ]
                # [B,T,N]

                mixed_r = self._mix_kernel_single_offset(
                    kernel_r=kernel_r
                )
                # [B,D,T]

                out = out + x_r * mixed_r

                del x_r
                del kernel_r
                del mixed_r

        del segment_norm
        del t

        out = out * self.kernel_scale

        return out

    def _apply_dynamic_conv_materialized_kernel_stream_x(
        self,
        h_full: torch.Tensor,
        kernel_chunk: torch.Tensor,
        start: int,
        end: int
    ) -> torch.Tensor:
        """
        折中性能路径。

        物化:
            mixed_btkd: [B,T,K,D]

        不物化:
            windows: [B,D,T,K]

        输入:
            h_full:       [B,D,L]
            kernel_chunk: [B,T,N,K]

        输出:
            out: [B,D,T]

        核心思想:
            原 stream_k 路径中，每个 K offset 都会做一次:

                kernel_r: [B,T,N]
                kernel_mix.T: [N,D]
                mixed_r = kernel_r @ kernel_mix.T
                mixed_r: [B,T,D] -> [B,D,T]

            K=32 时就是 32 次 matmul。

            本路径一次性做:

                kernel_btnk = kernel_chunk.permute(0,1,3,2)
                [B,T,K,N]

                mixed_btkd = kernel_btnk @ kernel_mix.T
                [B,T,K,D]

            然后仍然逐 K offset gather x_r 并累加:

                out += x_r * mixed_btkd[:, :, r, :].transpose(1,2)

        当前大配置显存:
            B=1,T=32768,K=32,D=1024

            mixed_btkd:
                [1,32768,32,1024]
                bf16 约 2GB

        比完整 materialized_windows 更安全，因为不额外物化:
            windows: [B,D,T,K]，同样约 2GB。
        """

        B, D, L, T, N, K = self._validate_kernel_shape(
            h_full=h_full,
            kernel_chunk=kernel_chunk,
            start=start,
            end=end
        )

        start_i = int(start)
        end_i = int(end)

        segment_norm, seg_start = build_local_norm_segment_for_causal_windows(
            x_full=h_full,
            norm=self.norm1,
            start=start_i,
            end=end_i,
            kernel_size=K
        )
        # [B,D,S]

        S = segment_norm.size(-1)
        device = h_full.device

        out = h_full.new_zeros(
            B,
            D,
            T
        )

        kernel_btnk = kernel_chunk.permute(
            0,
            1,
            3,
            2
        ).contiguous()
        # [B,T,K,N]

        mixed_btkd = torch.matmul(
            kernel_btnk,
            self.kernel_mix.transpose(
                0,
                1
            )
        )
        # [B,T,K,D]

        del kernel_btnk

        t = torch.arange(
            start_i,
            end_i,
            device=device,
            dtype=torch.long
        )

        for r in range(K):
            offset = K - 1 - r
            global_idx = t - offset
            valid = global_idx >= 0

            if valid.any():
                local_idx = global_idx - seg_start

                local_idx_clamped = local_idx.clamp(
                    min=0,
                    max=S - 1
                )

                x_r = segment_norm.index_select(
                    dim=-1,
                    index=local_idx_clamped
                )
                # [B,D,T]

                mask = valid.to(
                    dtype=h_full.dtype
                ).view(
                    1,
                    1,
                    T
                )

                x_r = x_r * mask

                mixed_r = mixed_btkd[
                    :,
                    :,
                    r,
                    :
                ].transpose(
                    1,
                    2
                )
                # [B,D,T] view，通常不强制 contiguous

                out = out + x_r * mixed_r

                del x_r
                del mixed_r

        del mixed_btkd
        del segment_norm
        del t

        out = out * self.kernel_scale

        return out

    def _apply_dynamic_conv_materialized_windows(
        self,
        h_full: torch.Tensor,
        kernel_chunk: torch.Tensor,
        start: int,
        end: int
    ) -> torch.Tensor:
        """
        调试/对比路径。

        显式构造:
            windows:      [B,D,T,K]
            mixed_kernel: [B,D,T,K]

        该路径不是默认推荐路径。
        大配置下会额外占用较多显存。
        """

        B, D, L, T, N, K = self._validate_kernel_shape(
            h_full=h_full,
            kernel_chunk=kernel_chunk,
            start=start,
            end=end
        )

        start_i = int(start)
        end_i = int(end)

        segment_norm, seg_start = build_local_norm_segment_for_causal_windows(
            x_full=h_full,
            norm=self.norm1,
            start=start_i,
            end=end_i,
            kernel_size=K
        )

        S = segment_norm.size(-1)
        device = h_full.device

        windows = h_full.new_zeros(
            B,
            D,
            T,
            K
        )

        t = torch.arange(
            start_i,
            end_i,
            device=device,
            dtype=torch.long
        )

        for r in range(K):
            offset = K - 1 - r
            global_idx = t - offset
            valid = global_idx >= 0

            if valid.any():
                local_idx = global_idx - seg_start

                local_idx_clamped = local_idx.clamp(
                    min=0,
                    max=S - 1
                )

                x_r = segment_norm.index_select(
                    dim=-1,
                    index=local_idx_clamped
                )
                # [B,D,T]

                mask = valid.to(
                    dtype=h_full.dtype
                ).view(
                    1,
                    1,
                    T
                )

                windows[
                    :,
                    :,
                    :,
                    r
                ] = x_r * mask

                del x_r

        mixed_kernel = self._mix_kernel_n_dimension(
            kernel_chunk=kernel_chunk
        )
        # [B,D,T,K]

        out = torch.sum(
            windows * mixed_kernel,
            dim=-1
        )

        out = out * self.kernel_scale

        del segment_norm
        del windows
        del mixed_kernel
        del t

        return out

    def _apply_dynamic_conv(
        self,
        h_full: torch.Tensor,
        kernel_chunk: torch.Tensor,
        start: int,
        end: int
    ) -> torch.Tensor:
        impl = self.dynamic_conv_impl

        if impl == "stream_k":
            return self._apply_dynamic_conv_stream_k(
                h_full=h_full,
                kernel_chunk=kernel_chunk,
                start=start,
                end=end
            )

        if impl == "materialized_kernel_stream_x":
            return self._apply_dynamic_conv_materialized_kernel_stream_x(
                h_full=h_full,
                kernel_chunk=kernel_chunk,
                start=start,
                end=end
            )

        if impl == "materialized_windows":
            return self._apply_dynamic_conv_materialized_windows(
                h_full=h_full,
                kernel_chunk=kernel_chunk,
                start=start,
                end=end
            )

        if impl == "auto":
            return self._apply_dynamic_conv_materialized_kernel_stream_x(
                h_full=h_full,
                kernel_chunk=kernel_chunk,
                start=start,
                end=end
            )

        raise ValueError(
            f"未知 dynamic_conv_impl: {impl}"
        )

    def forward_chunk(
        self,
        h_full: torch.Tensor,
        kernel_chunk: torch.Tensor,
        start: int,
        end: int
    ) -> torch.Tensor:
        if h_full.dim() != 3:
            raise ValueError(
                f"h_full 应为 [B,D,L]，但得到 {h_full.shape}"
            )

        B, D, L = h_full.shape

        if D != self.embed_dim:
            raise ValueError(
                f"embed_dim 不匹配，期望 {self.embed_dim}，实际 {D}"
            )

        start_i = int(start)
        end_i = int(end)

        if not (0 <= start_i < end_i <= L):
            raise ValueError(
                f"非法 chunk 边界 start={start_i}, end={end_i}, L={L}"
            )

        h_chunk = h_full[
            :,
            :,
            start_i:end_i
        ]

        identity = h_chunk

        out = self._apply_dynamic_conv(
            h_full=h_full,
            kernel_chunk=kernel_chunk,
            start=start_i,
            end=end_i
        )

        out = self.out_proj(
            out
        )

        out = self.dropout(
            out
        )

        if self.use_nonlinear_residual:
            h = (
                identity
                + self.nonlinear_residual1(
                    identity
                )
                + out
            )
        else:
            h = identity + out

        identity2 = h

        y = h.transpose(
            1,
            2
        )

        y = self.norm2(
            y
        )

        y = self.mlp(
            y
        )

        y = y.transpose(
            1,
            2
        ).contiguous()

        if self.use_nonlinear_residual:
            h = (
                identity2
                + self.nonlinear_residual2(
                    identity2
                )
                + y
            )
        else:
            h = identity2 + y

        return h

class DilatedUnsharedLayer(nn.Module):
    """
    单层结构。

    输入:
        source_input_full: [B,D,L]
        dynamic_h_full:    [B,D,L]

    输出:
        next_source_full:  [B,D,L]
        next_dynamic_full: [B,D,L]

    当前版本重点:
        1. source chunk checkpoint 包住 build_dilated_source_windows，
           避免 windows [B,D,T,S] 作为 checkpoint 输入长期保存。

        2. dynamic chunk checkpoint 按整层累计激活判断。

        3. forward 新增 disable_inner_checkpoint。
           当外层 backbone 已经启用 layer-level checkpoint 时，
           内部 source/dynamic chunk checkpoint 自动禁用，避免嵌套 checkpoint。

    checkpoint 语义:
        use_checkpoint=True:
            训练时强制 checkpoint。

        use_checkpoint=False 且 adaptive_checkpoint=True:
            按估算激活自适应 checkpoint。

        use_checkpoint=False 且 adaptive_checkpoint=False:
            不 checkpoint。
    """

    def __init__(
        self,
        embed_dim: int,
        source_channels: int,
        num_kernels: int,
        dynamic_kernel_size: int,
        source_conv_kernel_size: int,
        kernel_gen_kernel_size: int,
        dilation: int,
        mlp_ratio: float,
        dropout: float,
        normalize_kernel: bool,
        initial_scale: float,
        use_bias: bool,
        use_checkpoint: bool,
        source_conv_impl: str = "einsum",
        source_fused_mixed: bool = True,
        dynamic_conv_impl: str = "materialized_kernel_stream_x",
        dynamic_conv_tmp_mb_limit: float = 64.0,
        use_nonlinear_residual: bool = False,
        use_source_nonlinear_residual: bool = False,
        nonlinear_residual_initial_scale: float = 0.0,
        nonlinear_residual_bias: bool = True,
        use_preallocated_chunk_output: bool = True,
        adaptive_checkpoint: bool = True,
        checkpoint_activation_threshold_mb: float = 512.0
    ):
        super().__init__()

        if embed_dim <= 0:
            raise ValueError("embed_dim 必须为正数。")

        if source_channels <= 0:
            raise ValueError("source_channels 必须为正数。")

        if num_kernels <= 0:
            raise ValueError("num_kernels 必须为正数。")

        if dynamic_kernel_size <= 0:
            raise ValueError("dynamic_kernel_size 必须为正数。")

        if source_conv_kernel_size <= 0:
            raise ValueError("source_conv_kernel_size 必须为正数。")

        if kernel_gen_kernel_size <= 0:
            raise ValueError("kernel_gen_kernel_size 必须为正数。")

        if dilation <= 0:
            raise ValueError("dilation 必须为正数。")

        if checkpoint_activation_threshold_mb <= 0:
            raise ValueError("checkpoint_activation_threshold_mb 必须为正数。")

        self.embed_dim = int(embed_dim)
        self.source_channels = int(source_channels)
        self.num_kernels = int(num_kernels)
        self.dynamic_kernel_size = int(dynamic_kernel_size)
        self.source_conv_kernel_size = int(source_conv_kernel_size)
        self.kernel_gen_kernel_size = int(kernel_gen_kernel_size)
        self.mlp_ratio = float(mlp_ratio)
        self.dynamic_conv_impl = str(dynamic_conv_impl)
        self.dilation = int(dilation)

        self.force_checkpoint = bool(use_checkpoint)
        self.adaptive_checkpoint = bool(adaptive_checkpoint)
        self.checkpoint_activation_threshold_mb = float(
            checkpoint_activation_threshold_mb
        )

        self.source_fused_mixed = bool(source_fused_mixed)

        self.use_source_nonlinear_residual = bool(
            use_source_nonlinear_residual
        )

        self.use_preallocated_chunk_output = bool(
            use_preallocated_chunk_output
        )

        self.source_norm = ChannelLayerNorm(
            embed_dim
        )

        self.source_conv = ChunkedDUnsharedDilatedSourceConv1d(
            embed_dim=embed_dim,
            source_channels=source_channels,
            kernel_size=source_conv_kernel_size,
            dilation=dilation,
            bias=use_bias,
            impl=source_conv_impl,
            force_checkpoint=self.force_checkpoint,
            adaptive_checkpoint=self.adaptive_checkpoint,
            checkpoint_activation_threshold_mb=self.checkpoint_activation_threshold_mb
        )

        self.source_act = nn.GELU()

        self.source_dropout = nn.Dropout(
            dropout
        )

        self.source_mix = SourceChannelMix(
            embed_dim=embed_dim,
            source_channels=source_channels
        )

        if self.use_source_nonlinear_residual:
            self.source_nonlinear_residual = TokenNonlinearResidual(
                embed_dim=embed_dim,
                initial_scale=nonlinear_residual_initial_scale,
                bias=nonlinear_residual_bias
            )
        else:
            self.source_nonlinear_residual = None

        self.kernel_generator = StreamingKernelGenConv1d(
            embed_dim=embed_dim,
            num_kernels=num_kernels,
            dynamic_kernel_size=dynamic_kernel_size,
            kernel_gen_kernel_size=kernel_gen_kernel_size,
            bias=use_bias
        )

        self.dynamic_layer = ChunkedUnsharedDynamicConvLayer(
            embed_dim=embed_dim,
            num_kernels=num_kernels,
            dynamic_kernel_size=dynamic_kernel_size,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            normalize_kernel=normalize_kernel,
            initial_scale=initial_scale,
            dynamic_conv_impl=dynamic_conv_impl,
            dynamic_conv_tmp_mb_limit=dynamic_conv_tmp_mb_limit,
            use_nonlinear_residual=use_nonlinear_residual,
            nonlinear_residual_initial_scale=nonlinear_residual_initial_scale,
            nonlinear_residual_bias=nonlinear_residual_bias
        )

    def _num_chunks(
        self,
        L: int,
        chunk_size: int
    ) -> int:
        step = max(
            1,
            int(chunk_size)
        )

        return int(
            math.ceil(
                float(L) / float(step)
            )
        )

    def _estimate_source_chunk_checkpoint_mb(
        self,
        B: int,
        D: int,
        T: int,
        dtype: torch.dtype
    ) -> float:
        bytes_per_elem = _dtype_nbytes(
            dtype
        )

        S = int(
            self.source_conv_kernel_size
        )

        C = int(
            self.source_channels
        )

        windows_elems = (
            1.0
            * float(B)
            * float(D)
            * float(T)
            * float(S)
        )

        source_internal_elems = (
            4.0
            * float(B)
            * float(D)
            * float(C)
            * float(T)
        )

        residual_elems = (
            1.0
            * float(B)
            * float(D)
            * float(T)
        )

        total_elems = (
            windows_elems
            + source_internal_elems
            + residual_elems
        )

        estimated_bytes = (
            total_elems
            * float(bytes_per_elem)
        )

        return _bytes_to_mb(
            estimated_bytes
        )

    def _should_checkpoint_source_chunk(
        self,
        B: int,
        D: int,
        L: int,
        T: int,
        chunk_size: int,
        dtype: torch.dtype,
        disable_inner_checkpoint: bool = False
    ) -> bool:
        if disable_inner_checkpoint:
            return False

        source_chunk_mb = self._estimate_source_chunk_checkpoint_mb(
            B=B,
            D=D,
            T=T,
            dtype=dtype
        )

        num_chunks = self._num_chunks(
            L=L,
            chunk_size=chunk_size
        )

        estimated_layer_mb = (
            source_chunk_mb
            * float(num_chunks)
        )

        return _adaptive_checkpoint_decision(
            training=self.training,
            force_checkpoint=self.force_checkpoint,
            adaptive_checkpoint=self.adaptive_checkpoint,
            estimated_activation_mb=estimated_layer_mb,
            threshold_mb=self.checkpoint_activation_threshold_mb
        )

    def _estimate_dynamic_chunk_checkpoint_mb(
        self,
        B: int,
        D: int,
        T: int,
        dtype: torch.dtype
    ) -> float:
        bytes_per_elem = _dtype_nbytes(
            dtype
        )

        N = int(
            self.num_kernels
        )

        K = int(
            self.dynamic_kernel_size
        )

        pad = int(
            self.kernel_gen_kernel_size - 1
        )

        hidden_dim = int(
            float(D) * float(self.mlp_ratio)
        )

        kernel_gen_elems = (
            4.0
            * float(B)
            * float(N)
            * float(K)
            * float(T + pad)
        )

        base_dyn_elems = (
            4.0
            * float(B)
            * float(D)
            * float(T)
        )

        impl = self.dynamic_conv_impl

        if impl in (
            "materialized_kernel_stream_x",
            "auto"
        ):
            extra_dyn_elems = (
                float(B)
                * float(T)
                * float(K)
                * float(D)
            )
        elif impl == "materialized_windows":
            extra_dyn_elems = (
                2.0
                * float(B)
                * float(D)
                * float(T)
                * float(K)
            )
        else:
            extra_dyn_elems = 0.0

        mlp_elems = (
            3.0
            * float(B)
            * float(T)
            * float(hidden_dim)
        )

        total_elems = (
            kernel_gen_elems
            + base_dyn_elems
            + extra_dyn_elems
            + mlp_elems
        )

        estimated_bytes = (
            total_elems
            * float(bytes_per_elem)
        )

        return _bytes_to_mb(
            estimated_bytes
        )

    def _should_checkpoint_dynamic_chunk(
        self,
        B: int,
        D: int,
        L: int,
        T: int,
        chunk_size: int,
        dtype: torch.dtype,
        disable_inner_checkpoint: bool = False
    ) -> bool:
        if disable_inner_checkpoint:
            return False

        dynamic_chunk_mb = self._estimate_dynamic_chunk_checkpoint_mb(
            B=B,
            D=D,
            T=T,
            dtype=dtype
        )

        num_chunks = self._num_chunks(
            L=L,
            chunk_size=chunk_size
        )

        estimated_layer_mb = (
            dynamic_chunk_mb
            * float(num_chunks)
        )

        return _adaptive_checkpoint_decision(
            training=self.training,
            force_checkpoint=self.force_checkpoint,
            adaptive_checkpoint=self.adaptive_checkpoint,
            estimated_activation_mb=estimated_layer_mb,
            threshold_mb=self.checkpoint_activation_threshold_mb
        )

    def _kernel_dynamic_chunk_body(
        self,
        dynamic_h_full: torch.Tensor,
        next_source_full: torch.Tensor,
        start: int,
        end: int
    ) -> torch.Tensor:
        kernel_chunk = self.kernel_generator.forward_chunk_from_full(
            mixed_source_full=next_source_full,
            start=start,
            end=end
        )

        dynamic_chunk = self.dynamic_layer.forward_chunk(
            h_full=dynamic_h_full,
            kernel_chunk=kernel_chunk,
            start=start,
            end=end
        )

        return dynamic_chunk

    def _compute_source_mixed_chunk(
        self,
        source_input_norm_full: torch.Tensor,
        start: int,
        end: int
    ) -> torch.Tensor:
        if self.source_fused_mixed:
            source_mixed_chunk = self.source_conv.forward_mixed_chunk(
                x_full=source_input_norm_full,
                start=start,
                end=end,
                source_mix_weight=self.source_mix.mix,
                source_mix_bias=self.source_mix.bias,
                activation=self.source_act,
                dropout_module=self.source_dropout,
                post_mix_dropout_module=self.source_dropout
            )

            return source_mixed_chunk

        source_chunk = self.source_conv.forward_chunk(
            x_full=source_input_norm_full,
            start=start,
            end=end
        )

        source_chunk = self.source_act(
            source_chunk
        )

        source_chunk = self.source_dropout(
            source_chunk
        )

        source_mixed_chunk = self.source_mix(
            source_chunk
        )

        source_mixed_chunk = self.source_dropout(
            source_mixed_chunk
        )

        return source_mixed_chunk

    def _source_chunk_body(
        self,
        source_input_full: torch.Tensor,
        source_input_norm_full: torch.Tensor,
        start: int,
        end: int
    ) -> torch.Tensor:
        source_mixed_chunk = self._compute_source_mixed_chunk(
            source_input_norm_full=source_input_norm_full,
            start=start,
            end=end
        )

        source_identity_chunk = source_input_full[
            :,
            :,
            start:end
        ]

        if self.use_source_nonlinear_residual:
            next_source_chunk = (
                source_identity_chunk
                + self.source_nonlinear_residual(
                    source_identity_chunk
                )
                + source_mixed_chunk
            )
        else:
            next_source_chunk = (
                source_identity_chunk
                + source_mixed_chunk
            )

        return next_source_chunk

    def _compute_next_source_full_preallocated(
        self,
        source_input_full: torch.Tensor,
        source_input_norm_full: torch.Tensor,
        chunk_size: int,
        disable_inner_checkpoint: bool = False
    ) -> torch.Tensor:
        if source_input_full.dim() != 3:
            raise ValueError(
                f"source_input_full 应为 [B,D,L]，但得到 {source_input_full.shape}"
            )

        if source_input_norm_full.dim() != 3:
            raise ValueError(
                f"source_input_norm_full 应为 [B,D,L]，但得到 {source_input_norm_full.shape}"
            )

        if source_input_full.shape != source_input_norm_full.shape:
            raise ValueError(
                f"source_input_full 和 source_input_norm_full shape 必须相同，"
                f"source_input_full={source_input_full.shape}, "
                f"source_input_norm_full={source_input_norm_full.shape}"
            )

        B, D, L = source_input_full.shape

        if D != self.embed_dim:
            raise ValueError(
                f"embed_dim 不匹配，期望 {self.embed_dim}，实际 {D}"
            )

        step = max(
            1,
            int(chunk_size)
        )

        next_source_full = source_input_full.new_empty(
            B,
            D,
            L
        )

        for start in range(0, L, step):
            end = min(
                start + step,
                L
            )

            T = end - start

            should_checkpoint = self._should_checkpoint_source_chunk(
                B=B,
                D=D,
                L=L,
                T=T,
                chunk_size=step,
                dtype=source_input_full.dtype,
                disable_inner_checkpoint=disable_inner_checkpoint
            )

            if should_checkpoint:
                def source_chunk_forward(
                    source_input_full_ref: torch.Tensor,
                    source_input_norm_full_ref: torch.Tensor,
                    start_ref=start,
                    end_ref=end
                ) -> torch.Tensor:
                    return self._source_chunk_body(
                        source_input_full=source_input_full_ref,
                        source_input_norm_full=source_input_norm_full_ref,
                        start=start_ref,
                        end=end_ref
                    )

                next_source_chunk = checkpoint(
                    source_chunk_forward,
                    source_input_full,
                    source_input_norm_full,
                    use_reentrant=True,
                    preserve_rng_state=True
                )
            else:
                next_source_chunk = self._source_chunk_body(
                    source_input_full=source_input_full,
                    source_input_norm_full=source_input_norm_full,
                    start=start,
                    end=end
                )

            next_source_full[
                :,
                :,
                start:end
            ] = next_source_chunk

            del next_source_chunk

        expected_shape = (
            B,
            D,
            L
        )

        if next_source_full.shape != expected_shape:
            raise RuntimeError(
                f"next_source_full shape 错误，期望 {expected_shape}，实际 {next_source_full.shape}"
            )

        return next_source_full

    def _compute_next_source_full_cat(
        self,
        source_input_full: torch.Tensor,
        source_input_norm_full: torch.Tensor,
        chunk_size: int,
        disable_inner_checkpoint: bool = False
    ) -> torch.Tensor:
        if source_input_full.dim() != 3:
            raise ValueError(
                f"source_input_full 应为 [B,D,L]，但得到 {source_input_full.shape}"
            )

        if source_input_norm_full.dim() != 3:
            raise ValueError(
                f"source_input_norm_full 应为 [B,D,L]，但得到 {source_input_norm_full.shape}"
            )

        if source_input_full.shape != source_input_norm_full.shape:
            raise ValueError(
                f"source_input_full 和 source_input_norm_full shape 必须相同，"
                f"source_input_full={source_input_full.shape}, "
                f"source_input_norm_full={source_input_norm_full.shape}"
            )

        B, D, L = source_input_full.shape

        if D != self.embed_dim:
            raise ValueError(
                f"embed_dim 不匹配，期望 {self.embed_dim}，实际 {D}"
            )

        step = max(
            1,
            int(chunk_size)
        )

        next_source_chunks: List[torch.Tensor] = []

        for start in range(0, L, step):
            end = min(
                start + step,
                L
            )

            T = end - start

            should_checkpoint = self._should_checkpoint_source_chunk(
                B=B,
                D=D,
                L=L,
                T=T,
                chunk_size=step,
                dtype=source_input_full.dtype,
                disable_inner_checkpoint=disable_inner_checkpoint
            )

            if should_checkpoint:
                def source_chunk_forward(
                    source_input_full_ref: torch.Tensor,
                    source_input_norm_full_ref: torch.Tensor,
                    start_ref=start,
                    end_ref=end
                ) -> torch.Tensor:
                    return self._source_chunk_body(
                        source_input_full=source_input_full_ref,
                        source_input_norm_full=source_input_norm_full_ref,
                        start=start_ref,
                        end=end_ref
                    )

                next_source_chunk = checkpoint(
                    source_chunk_forward,
                    source_input_full,
                    source_input_norm_full,
                    use_reentrant=True,
                    preserve_rng_state=True
                )
            else:
                next_source_chunk = self._source_chunk_body(
                    source_input_full=source_input_full,
                    source_input_norm_full=source_input_norm_full,
                    start=start,
                    end=end
                )

            next_source_chunks.append(
                next_source_chunk
            )

            del next_source_chunk

        next_source_full = torch.cat(
            next_source_chunks,
            dim=-1
        )

        next_source_chunks.clear()
        del next_source_chunks

        expected_shape = (
            B,
            D,
            L
        )

        if next_source_full.shape != expected_shape:
            raise RuntimeError(
                f"next_source_full shape 错误，期望 {expected_shape}，实际 {next_source_full.shape}"
            )

        return next_source_full

    def _compute_next_source_full(
        self,
        source_input_full: torch.Tensor,
        source_input_norm_full: torch.Tensor,
        chunk_size: int,
        disable_inner_checkpoint: bool = False
    ) -> torch.Tensor:
        if self.use_preallocated_chunk_output:
            return self._compute_next_source_full_preallocated(
                source_input_full=source_input_full,
                source_input_norm_full=source_input_norm_full,
                chunk_size=chunk_size,
                disable_inner_checkpoint=disable_inner_checkpoint
            )

        return self._compute_next_source_full_cat(
            source_input_full=source_input_full,
            source_input_norm_full=source_input_norm_full,
            chunk_size=chunk_size,
            disable_inner_checkpoint=disable_inner_checkpoint
        )

    def _compute_next_dynamic_full_preallocated(
        self,
        dynamic_h_full: torch.Tensor,
        next_source_full: torch.Tensor,
        chunk_size: int,
        disable_inner_checkpoint: bool = False
    ) -> torch.Tensor:
        if dynamic_h_full.dim() != 3:
            raise ValueError(
                f"dynamic_h_full 应为 [B,D,L]，但得到 {dynamic_h_full.shape}"
            )

        if next_source_full.dim() != 3:
            raise ValueError(
                f"next_source_full 应为 [B,D,L]，但得到 {next_source_full.shape}"
            )

        if dynamic_h_full.shape != next_source_full.shape:
            raise ValueError(
                f"dynamic_h_full 和 next_source_full shape 必须相同，"
                f"dynamic_h_full={dynamic_h_full.shape}, "
                f"next_source_full={next_source_full.shape}"
            )

        B, D, L = dynamic_h_full.shape

        if D != self.embed_dim:
            raise ValueError(
                f"embed_dim 不匹配，期望 {self.embed_dim}，实际 {D}"
            )

        step = max(
            1,
            int(chunk_size)
        )

        next_dynamic_full = dynamic_h_full.new_empty(
            B,
            D,
            L
        )

        for start in range(0, L, step):
            end = min(
                start + step,
                L
            )

            T = end - start

            should_checkpoint = self._should_checkpoint_dynamic_chunk(
                B=B,
                D=D,
                L=L,
                T=T,
                chunk_size=step,
                dtype=dynamic_h_full.dtype,
                disable_inner_checkpoint=disable_inner_checkpoint
            )

            if should_checkpoint:
                def dynamic_chunk_forward(
                    dynamic_h_full_ref: torch.Tensor,
                    next_source_full_ref: torch.Tensor,
                    start_ref=start,
                    end_ref=end
                ) -> torch.Tensor:
                    return self._kernel_dynamic_chunk_body(
                        dynamic_h_full=dynamic_h_full_ref,
                        next_source_full=next_source_full_ref,
                        start=start_ref,
                        end=end_ref
                    )

                dynamic_chunk = checkpoint(
                    dynamic_chunk_forward,
                    dynamic_h_full,
                    next_source_full,
                    use_reentrant=True,
                    preserve_rng_state=True
                )
            else:
                dynamic_chunk = self._kernel_dynamic_chunk_body(
                    dynamic_h_full=dynamic_h_full,
                    next_source_full=next_source_full,
                    start=start,
                    end=end
                )

            next_dynamic_full[
                :,
                :,
                start:end
            ] = dynamic_chunk

            del dynamic_chunk

        expected_shape = (
            B,
            D,
            L
        )

        if next_dynamic_full.shape != expected_shape:
            raise RuntimeError(
                f"next_dynamic_full shape 错误，期望 {expected_shape}，实际 {next_dynamic_full.shape}"
            )

        return next_dynamic_full

    def _compute_next_dynamic_full_cat(
        self,
        dynamic_h_full: torch.Tensor,
        next_source_full: torch.Tensor,
        chunk_size: int,
        disable_inner_checkpoint: bool = False
    ) -> torch.Tensor:
        if dynamic_h_full.dim() != 3:
            raise ValueError(
                f"dynamic_h_full 应为 [B,D,L]，但得到 {dynamic_h_full.shape}"
            )

        if next_source_full.dim() != 3:
            raise ValueError(
                f"next_source_full 应为 [B,D,L]，但得到 {next_source_full.shape}"
            )

        if dynamic_h_full.shape != next_source_full.shape:
            raise ValueError(
                f"dynamic_h_full 和 next_source_full shape 必须相同，"
                f"dynamic_h_full={dynamic_h_full.shape}, "
                f"next_source_full={next_source_full.shape}"
            )

        B, D, L = dynamic_h_full.shape

        if D != self.embed_dim:
            raise ValueError(
                f"embed_dim 不匹配，期望 {self.embed_dim}，实际 {D}"
            )

        step = max(
            1,
            int(chunk_size)
        )

        next_dynamic_chunks: List[torch.Tensor] = []

        for start in range(0, L, step):
            end = min(
                start + step,
                L
            )

            T = end - start

            should_checkpoint = self._should_checkpoint_dynamic_chunk(
                B=B,
                D=D,
                L=L,
                T=T,
                chunk_size=step,
                dtype=dynamic_h_full.dtype,
                disable_inner_checkpoint=disable_inner_checkpoint
            )

            if should_checkpoint:
                def dynamic_chunk_forward(
                    dynamic_h_full_ref: torch.Tensor,
                    next_source_full_ref: torch.Tensor,
                    start_ref=start,
                    end_ref=end
                ) -> torch.Tensor:
                    return self._kernel_dynamic_chunk_body(
                        dynamic_h_full=dynamic_h_full_ref,
                        next_source_full=next_source_full_ref,
                        start=start_ref,
                        end=end_ref
                    )

                dynamic_chunk = checkpoint(
                    dynamic_chunk_forward,
                    dynamic_h_full,
                    next_source_full,
                    use_reentrant=True,
                    preserve_rng_state=True
                )
            else:
                dynamic_chunk = self._kernel_dynamic_chunk_body(
                    dynamic_h_full=dynamic_h_full,
                    next_source_full=next_source_full,
                    start=start,
                    end=end
                )

            next_dynamic_chunks.append(
                dynamic_chunk
            )

            del dynamic_chunk

        next_dynamic_full = torch.cat(
            next_dynamic_chunks,
            dim=-1
        )

        next_dynamic_chunks.clear()
        del next_dynamic_chunks

        expected_shape = (
            B,
            D,
            L
        )

        if next_dynamic_full.shape != expected_shape:
            raise RuntimeError(
                f"next_dynamic_full shape 错误，期望 {expected_shape}，实际 {next_dynamic_full.shape}"
            )

        return next_dynamic_full

    def _compute_next_dynamic_full(
        self,
        dynamic_h_full: torch.Tensor,
        next_source_full: torch.Tensor,
        chunk_size: int,
        disable_inner_checkpoint: bool = False
    ) -> torch.Tensor:
        if self.use_preallocated_chunk_output:
            return self._compute_next_dynamic_full_preallocated(
                dynamic_h_full=dynamic_h_full,
                next_source_full=next_source_full,
                chunk_size=chunk_size,
                disable_inner_checkpoint=disable_inner_checkpoint
            )

        return self._compute_next_dynamic_full_cat(
            dynamic_h_full=dynamic_h_full,
            next_source_full=next_source_full,
            chunk_size=chunk_size,
            disable_inner_checkpoint=disable_inner_checkpoint
        )

    def forward(
        self,
        source_input_full: torch.Tensor,
        dynamic_h_full: torch.Tensor,
        chunk_size: int,
        disable_inner_checkpoint: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if source_input_full.dim() != 3:
            raise ValueError(
                f"source_input_full 应为 [B,D,L]，但得到 {source_input_full.shape}"
            )

        if dynamic_h_full.dim() != 3:
            raise ValueError(
                f"dynamic_h_full 应为 [B,D,L]，但得到 {dynamic_h_full.shape}"
            )

        if source_input_full.shape != dynamic_h_full.shape:
            raise ValueError(
                f"source_input_full 和 dynamic_h_full shape 必须相同，"
                f"source_input_full={source_input_full.shape}, "
                f"dynamic_h_full={dynamic_h_full.shape}"
            )

        B, D, L = source_input_full.shape

        if D != self.embed_dim:
            raise ValueError(
                f"embed_dim 不匹配，期望 {self.embed_dim}，实际 {D}"
            )

        step = max(
            1,
            int(chunk_size)
        )

        source_input_norm_full = self.source_norm(
            source_input_full
        )

        next_source_full = self._compute_next_source_full(
            source_input_full=source_input_full,
            source_input_norm_full=source_input_norm_full,
            chunk_size=step,
            disable_inner_checkpoint=disable_inner_checkpoint
        )

        del source_input_norm_full

        next_dynamic_full = self._compute_next_dynamic_full(
            dynamic_h_full=dynamic_h_full,
            next_source_full=next_source_full,
            chunk_size=step,
            disable_inner_checkpoint=disable_inner_checkpoint
        )

        expected_shape = (
            B,
            D,
            L
        )

        if next_source_full.shape != expected_shape:
            raise RuntimeError(
                f"next_source_full shape 错误，期望 {expected_shape}，实际 {next_source_full.shape}"
            )

        if next_dynamic_full.shape != expected_shape:
            raise RuntimeError(
                f"next_dynamic_full shape 错误，期望 {expected_shape}，实际 {next_dynamic_full.shape}"
            )

        return next_source_full, next_dynamic_full


# ============================================================
# 13. 多层 backbone
# ============================================================
class DilatedUnsharedDynamicConvBackbone(nn.Module):
    """
    多层膨胀 unshared 动态卷积 backbone。

    层数:
        最小 n，使得 2^n > block_size。

    dilation:
        第 0 层 2
        第 1 层 4
        第 2 层 8
        ...

    当前关键修复:
        1. layer-level checkpoint 使用 use_reentrant=True。
           这是长上下文下真正省显存的模式。
           第一次 forward 不记录整层内部 autograd graph。

        2. 如果 layer-level checkpoint 已触发，
           调用 layer 时传入 disable_inner_checkpoint=True，
           避免内部 source/dynamic chunk checkpoint 嵌套。

        3. 如果 layer-level checkpoint 没触发，
           layer 内部仍然可以按自适应规则开启 source/dynamic chunk checkpoint。

    checkpoint 语义:
        config.use_checkpoint=True:
            强制 layer-level checkpoint。

        config.use_checkpoint=False 且 config.adaptive_checkpoint=True:
            按估算激活自适应开启 layer-level checkpoint。

        config.use_checkpoint=False 且 config.adaptive_checkpoint=False:
            关闭 layer-level checkpoint。
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        self.num_layers = compute_num_dilated_layers(
            config.block_size
        )

        self.force_checkpoint = bool(
            config.use_checkpoint
        )

        self.adaptive_checkpoint = bool(
            config.adaptive_checkpoint
        )

        self.checkpoint_activation_threshold_mb = float(
            config.checkpoint_activation_threshold_mb
        )

        if self.checkpoint_activation_threshold_mb <= 0:
            raise ValueError(
                "checkpoint_activation_threshold_mb 必须为正数。"
            )

        self.layers = nn.ModuleList()

        for i in range(self.num_layers):
            dilation = 2 ** (i + 1)

            layer = DilatedUnsharedLayer(
                embed_dim=config.embed_dim,
                source_channels=config.source_channels,
                num_kernels=config.num_kernels,
                dynamic_kernel_size=config.dynamic_kernel_size,
                source_conv_kernel_size=config.source_conv_kernel_size,
                kernel_gen_kernel_size=config.kernel_gen_kernel_size,
                dilation=dilation,
                mlp_ratio=config.mlp_ratio,
                dropout=config.dropout,
                normalize_kernel=config.normalize_kernel,
                initial_scale=config.initial_scale,
                use_bias=config.use_bias,
                use_checkpoint=config.use_checkpoint,
                source_conv_impl=config.source_conv_impl,
                source_fused_mixed=config.source_fused_mixed,
                dynamic_conv_impl=config.dynamic_conv_impl,
                dynamic_conv_tmp_mb_limit=config.dynamic_conv_tmp_mb_limit,
                use_nonlinear_residual=config.use_nonlinear_residual,
                use_source_nonlinear_residual=config.use_source_nonlinear_residual,
                nonlinear_residual_initial_scale=config.nonlinear_residual_initial_scale,
                nonlinear_residual_bias=config.nonlinear_residual_bias,
                use_preallocated_chunk_output=config.use_preallocated_chunk_output,
                adaptive_checkpoint=config.adaptive_checkpoint,
                checkpoint_activation_threshold_mb=config.checkpoint_activation_threshold_mb
            )

            self.layers.append(
                layer
            )

    def _estimate_layer_checkpoint_mb(
        self,
        B: int,
        D: int,
        L: int,
        dtype: torch.dtype
    ) -> float:
        """
        估算 layer-level checkpoint 可节省激活。

        一个完整状态:
            [B,D,L]

        对你的配置:
            B=1,D=256,L=262144,bf16

            [B,D,L]:
                1 * 256 * 262144 * 2 = 128 MiB

            4 倍:
                512 MiB

        阈值默认 512MB，所以该配置会触发 layer checkpoint。
        """

        bytes_per_elem = _dtype_nbytes(
            dtype
        )

        estimated_bytes = (
            4.0
            * float(B)
            * float(D)
            * float(L)
            * float(bytes_per_elem)
        )

        return _bytes_to_mb(
            estimated_bytes
        )

    def _should_checkpoint_layer(
        self,
        B: int,
        D: int,
        L: int,
        dtype: torch.dtype
    ) -> bool:
        estimated_mb = self._estimate_layer_checkpoint_mb(
            B=B,
            D=D,
            L=L,
            dtype=dtype
        )

        return _adaptive_checkpoint_decision(
            training=self.training,
            force_checkpoint=self.force_checkpoint,
            adaptive_checkpoint=self.adaptive_checkpoint,
            estimated_activation_mb=estimated_mb,
            threshold_mb=self.checkpoint_activation_threshold_mb
        )

    def forward(
        self,
        h: torch.Tensor
    ) -> torch.Tensor:
        if h.dim() != 3:
            raise ValueError(
                f"h 应为 [B,D,L]，但得到 {h.shape}"
            )

        B, D, L = h.shape

        if D != self.config.embed_dim:
            raise ValueError(
                f"embed_dim 不匹配，期望 {self.config.embed_dim}，实际 {D}"
            )

        source_input = h
        dynamic_h = h

        for layer_index, layer in enumerate(self.layers):
            should_checkpoint = self._should_checkpoint_layer(
                B=B,
                D=D,
                L=L,
                dtype=h.dtype
            )

            if should_checkpoint:
                def layer_forward(
                    source_input_full: torch.Tensor,
                    dynamic_h_full: torch.Tensor,
                    layer_ref=layer
                ) -> Tuple[torch.Tensor, torch.Tensor]:
                    return layer_ref(
                        source_input_full=source_input_full,
                        dynamic_h_full=dynamic_h_full,
                        chunk_size=self.config.chunk_size,
                        disable_inner_checkpoint=True
                    )

                source_input, dynamic_h = checkpoint(
                    layer_forward,
                    source_input,
                    dynamic_h,
                    use_reentrant=True,
                    preserve_rng_state=True
                )
            else:
                source_input, dynamic_h = layer(
                    source_input_full=source_input,
                    dynamic_h_full=dynamic_h,
                    chunk_size=self.config.chunk_size,
                    disable_inner_checkpoint=False
                )

        return dynamic_h


# ============================================================
# 14. 完整语言模型
# ============================================================

class DilatedUnsharedDynamicConvLM(nn.Module):
    """
    完整语言模型。

    输入：
        input_ids: [B,L]

    输出：
        logits: Optional [B,L,V]
        loss:   Optional scalar

    训练时 targets 不为 None：
        分块计算 loss，不返回完整 logits。

    当前关键修复：
        loss chunk 也必须 checkpoint。

    原因：
        即使 backbone 已经 checkpoint，
        loss loop 中每个 loss_chunk 都会把 lm_head logits 和
        cross_entropy 内部 log_softmax 反传激活挂到 total_loss 上。

        对长序列:
            L = 262144
            loss_chunk_size = 8192
            num_loss_chunks = 32
            vocab_size = 8097

        每个 CE 内部 fp32 buffer:
            8192 * 8097 * 4 bytes ≈ 253 MiB

        32 个 chunk:
            253 MiB * 32 ≈ 7.9 GiB

        再加 bf16 logits:
            8192 * 8097 * 2 bytes ≈ 126 MiB/chunk
            126 MiB * 32 ≈ 4.0 GiB

        所以 loss graph 本身能吃 8GB~12GB。
        这就是当前 OOM 的关键点。

    checkpoint 语义：
        config.use_checkpoint=True:
            强制 loss chunk checkpoint。

        config.use_checkpoint=False 且 config.adaptive_checkpoint=True:
            根据整段 loss 累计估算自动 checkpoint。

        config.use_checkpoint=False 且 config.adaptive_checkpoint=False:
            不 checkpoint。
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.embed_dim
        )

        self.pos_encoding = SinusoidalPositionEncoding(
            embed_dim=config.embed_dim
        )

        self.dropout = nn.Dropout(
            config.dropout
        )

        self.backbone = DilatedUnsharedDynamicConvBackbone(
            config=config
        )

        self.final_norm = nn.LayerNorm(
            config.embed_dim
        )

        self.lm_head = nn.Linear(
            config.embed_dim,
            config.vocab_size,
            bias=False
        )

        # 权重绑定
        self.lm_head.weight = self.token_embedding.weight

        self.apply(
            self._init_weights
        )

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

            if module.bias is not None:
                nn.init.zeros_(
                    module.bias
                )

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

        elif isinstance(module, nn.Conv1d):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

            if module.bias is not None:
                nn.init.zeros_(
                    module.bias
                )

        elif isinstance(module, ChunkedDUnsharedDilatedSourceConv1d):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

            if module.bias is not None:
                nn.init.zeros_(
                    module.bias
                )

        elif isinstance(module, StreamingKernelGenConv1d):
            module.reset_parameters()

    def _estimate_loss_chunk_checkpoint_mb(
        self,
        B: int,
        L: int,
        loss_chunk_size: int,
        dtype: torch.dtype
    ) -> float:
        """
        估算整个 loss loop 如果不 checkpoint 会长期保存的激活。

        每个 loss chunk 主要有：

        1. logits_chunk:
            [B,T,V]
            dtype 通常为 bf16/fp16/fp32。

        2. cross_entropy 内部 log_softmax / softmax buffer:
            PyTorch cross_entropy 通常会产生 fp32 级别中间。
            当前日志已经证明：
                [8192,8097] * 4 bytes ≈ 254 MiB

        因此保守估算：
            logits:
                B * T * V * dtype_bytes

            CE fp32 buffer:
                B * T * V * 4

            反传额外:
                再按 1 份 dtype logits 估算。

        单 chunk:
            B * T * V * (4 + 2 * dtype_bytes)

        整个 loss loop:
            单 chunk * num_loss_chunks
        """

        V = int(
            self.config.vocab_size
        )

        step = max(
            1,
            int(loss_chunk_size)
        )

        num_chunks = int(
            math.ceil(
                float(L) / float(step)
            )
        )

        T_eff = min(
            step,
            int(L)
        )

        dtype_bytes = _dtype_nbytes(
            dtype
        )

        bytes_per_token_vocab = (
            4.0
            + 2.0 * float(dtype_bytes)
        )

        estimated_bytes_per_chunk = (
            float(B)
            * float(T_eff)
            * float(V)
            * bytes_per_token_vocab
        )

        estimated_total_bytes = (
            estimated_bytes_per_chunk
            * float(num_chunks)
        )

        return _bytes_to_mb(
            estimated_total_bytes
        )

    def _should_checkpoint_loss_chunk(
        self,
        B: int,
        L: int,
        loss_chunk_size: int,
        dtype: torch.dtype
    ) -> bool:
        """
        loss chunk checkpoint 自适应决策。

        关键：
            必须按整个 loss loop 的累计显存判断，
            不能只看单个 loss chunk。

        对你的配置：
            B=1
            L=262144
            loss_chunk_size=8192
            V=8097
            dtype=bf16

            单 chunk:
                CE fp32 buffer ≈ 253 MiB
                bf16 logits 等 ≈ 253 MiB 左右
                合计估算 ≈ 506 MiB

            32 chunks:
                约 16 GiB 级别估算

            必须 checkpoint。
        """

        estimated_mb = self._estimate_loss_chunk_checkpoint_mb(
            B=B,
            L=L,
            loss_chunk_size=loss_chunk_size,
            dtype=dtype
        )

        return _adaptive_checkpoint_decision(
            training=self.training,
            force_checkpoint=bool(self.config.use_checkpoint),
            adaptive_checkpoint=bool(self.config.adaptive_checkpoint),
            estimated_activation_mb=estimated_mb,
            threshold_mb=float(self.config.checkpoint_activation_threshold_mb)
        )

    def _loss_chunk_body(
        self,
        h_chunk: torch.Tensor,
        targets_chunk: torch.Tensor
    ) -> torch.Tensor:
        """
        单个 loss chunk 的完整计算体。

        输入:
            h_chunk:       [B,T,D]
            targets_chunk: [B,T]

        输出:
            loss_chunk: scalar, reduction=sum

        被 checkpoint 后：
            logits_chunk 和 cross_entropy 内部 fp32 log_softmax
            不会在 forward 后长期保存，而是在 backward 时重算。
        """

        if h_chunk.dim() != 3:
            raise ValueError(
                f"h_chunk 应为 [B,T,D]，但得到 {h_chunk.shape}"
            )

        if targets_chunk.dim() != 2:
            raise ValueError(
                f"targets_chunk 应为 [B,T]，但得到 {targets_chunk.shape}"
            )

        B, T, D = h_chunk.shape

        if D != self.config.embed_dim:
            raise ValueError(
                f"embed_dim 不匹配，期望 {self.config.embed_dim}，实际 {D}"
            )

        if targets_chunk.shape != (
            B,
            T
        ):
            raise ValueError(
                f"targets_chunk shape 错误，期望 {(B, T)}，实际 {targets_chunk.shape}"
            )

        logits_chunk = self.lm_head(
            h_chunk
        )

        loss_chunk = F.cross_entropy(
            logits_chunk.reshape(
                -1,
                logits_chunk.size(-1)
            ),
            targets_chunk.reshape(
                -1
            ),
            reduction="sum"
        )

        return loss_chunk

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ):
        if input_ids.dim() != 2:
            raise ValueError(
                f"input_ids 应为 [B,L]，但得到 {input_ids.shape}"
            )

        B, L = input_ids.shape

        if L > self.config.block_size:
            raise ValueError(
                f"输入长度 L={L} 超过 block_size={self.config.block_size}"
            )

        if targets is not None and targets.shape != input_ids.shape:
            raise ValueError(
                f"targets shape 应与 input_ids 相同，"
                f"input_ids={input_ids.shape}, targets={targets.shape}"
            )

        device = input_ids.device

        tok = self.token_embedding(
            input_ids
        )

        if self.config.use_sinusoidal_pos:
            pos = self.pos_encoding(
                length=L,
                device=device,
                dtype=tok.dtype
            )

            x = tok + pos
        else:
            x = tok

        x = self.dropout(
            x
        )

        # [B,L,D] -> [B,D,L]
        h = x.transpose(
            1,
            2
        ).contiguous()

        h = self.backbone(
            h
        )

        # [B,D,L] -> [B,L,D]
        h = h.transpose(
            1,
            2
        ).contiguous()

        h = self.final_norm(
            h
        )

        if targets is not None:
            chunk_size = max(
                1,
                int(self.config.loss_chunk_size)
            )

            should_checkpoint_loss = self._should_checkpoint_loss_chunk(
                B=B,
                L=L,
                loss_chunk_size=chunk_size,
                dtype=h.dtype
            )

            total_loss = h.new_zeros(())
            total_tokens = 0

            for start in range(0, L, chunk_size):
                end = min(
                    start + chunk_size,
                    L
                )

                h_chunk = h[
                    :,
                    start:end,
                    :
                ]

                targets_chunk = targets[
                    :,
                    start:end
                ]

                if should_checkpoint_loss:
                    def loss_chunk_forward(
                        h_chunk_ref: torch.Tensor,
                        targets_chunk_ref: torch.Tensor
                    ) -> torch.Tensor:
                        return self._loss_chunk_body(
                            h_chunk=h_chunk_ref,
                            targets_chunk=targets_chunk_ref
                        )

                    loss_chunk = checkpoint(
                        loss_chunk_forward,
                        h_chunk,
                        targets_chunk,
                        use_reentrant=True,
                        preserve_rng_state=False
                    )
                else:
                    loss_chunk = self._loss_chunk_body(
                        h_chunk=h_chunk,
                        targets_chunk=targets_chunk
                    )

                total_loss = total_loss + loss_chunk

                total_tokens += targets_chunk.numel()

                del h_chunk
                del targets_chunk
                del loss_chunk

            loss = total_loss / float(
                total_tokens
            )

            return None, loss

        logits = self.lm_head(
            h
        )

        return logits, None

# ============================================================
# 15. 兼容别名
# ============================================================

BlockMaskedCausalDynamicConvLM = DilatedUnsharedDynamicConvLM

DynamicConvLM = DilatedUnsharedDynamicConvLM


# ============================================================
# 16. 自检
# ============================================================

def _causal_smoke_test():
    torch.manual_seed(1234)

    config = ModelConfig(
        vocab_size=128,
        block_size=64,
        embed_dim=32,
        source_channels=8,
        num_kernels=4,
        dynamic_kernel_size=3,
        source_conv_kernel_size=3,
        kernel_gen_kernel_size=3,
        mlp_ratio=2.0,
        dropout=0.0,
        use_checkpoint=False,
        chunk_size=8,
        loss_chunk_size=16,
        normalize_kernel=True,
        initial_scale=0.1
    )

    model = DilatedUnsharedDynamicConvLM(config)

    model.eval()

    x1 = torch.randint(
        0,
        config.vocab_size,
        (2, 32)
    )

    x2 = x1.clone()

    x2[:, 20:] = torch.randint(
        0,
        config.vocab_size,
        x2[:, 20:].shape
    )

    with torch.no_grad():
        y1, _ = model(x1)
        y2, _ = model(x2)

    max_diff = (
        y1[:, :20, :] - y2[:, :20, :]
    ).abs().max().item()

    print("causal max diff before changed suffix:", max_diff)

    assert max_diff < 1e-5, "因果性测试失败。"

    model.train()

    targets = torch.randint(
        0,
        config.vocab_size,
        (2, 32)
    )

    _, loss = model(
        input_ids=x1,
        targets=targets
    )

    print("loss:", float(loss.detach().cpu()))

    loss.backward()

    total, trainable = count_parameters(model)

    print("total parameters:", total)
    print("trainable parameters:", trainable)


def _chunk_equivalence_test():
    torch.manual_seed(5678)

    config_a = ModelConfig(
        vocab_size=128,
        block_size=64,
        embed_dim=32,
        source_channels=8,
        num_kernels=4,
        dynamic_kernel_size=3,
        source_conv_kernel_size=3,
        kernel_gen_kernel_size=3,
        mlp_ratio=2.0,
        dropout=0.0,
        use_checkpoint=False,
        chunk_size=4,
        loss_chunk_size=16,
        normalize_kernel=True,
        initial_scale=0.1
    )

    model_a = DilatedUnsharedDynamicConvLM(config_a)

    state = model_a.state_dict()

    config_b = ModelConfig(
        vocab_size=128,
        block_size=64,
        embed_dim=32,
        source_channels=8,
        num_kernels=4,
        dynamic_kernel_size=3,
        source_conv_kernel_size=3,
        kernel_gen_kernel_size=3,
        mlp_ratio=2.0,
        dropout=0.0,
        use_checkpoint=False,
        chunk_size=16,
        loss_chunk_size=16,
        normalize_kernel=True,
        initial_scale=0.1
    )

    model_b = DilatedUnsharedDynamicConvLM(config_b)

    model_b.load_state_dict(state)

    model_a.eval()
    model_b.eval()

    x = torch.randint(
        0,
        config_a.vocab_size,
        (2, 32)
    )

    with torch.no_grad():
        y_a, _ = model_a(x)
        y_b, _ = model_b(x)

    max_diff = (
        y_a - y_b
    ).abs().max().item()

    print("chunk equivalence max diff:", max_diff)

    assert max_diff < 1e-5, "chunk size 等价测试失败。"


def _checkpoint_training_smoke_test():
    torch.manual_seed(9012)

    config = ModelConfig(
        vocab_size=128,
        block_size=64,
        embed_dim=32,
        source_channels=8,
        num_kernels=4,
        dynamic_kernel_size=3,
        source_conv_kernel_size=3,
        kernel_gen_kernel_size=3,
        mlp_ratio=2.0,
        dropout=0.0,
        use_checkpoint=True,
        chunk_size=8,
        loss_chunk_size=16,
        normalize_kernel=True,
        initial_scale=0.1
    )

    model = DilatedUnsharedDynamicConvLM(config)

    model.train()

    x = torch.randint(
        0,
        config.vocab_size,
        (2, 32)
    )

    targets = torch.randint(
        0,
        config.vocab_size,
        (2, 32)
    )

    _, loss = model(
        input_ids=x,
        targets=targets
    )

    print("checkpoint training loss:", float(loss.detach().cpu()))

    loss.backward()

    total_grad_norm = 0.0

    for p in model.parameters():
        if p.grad is not None:
            total_grad_norm += float(
                p.grad.detach().float().norm().cpu()
            )

    print("checkpoint training grad norm sum:", total_grad_norm)

    assert math.isfinite(total_grad_norm), "checkpoint 训练梯度出现非有限值。"

