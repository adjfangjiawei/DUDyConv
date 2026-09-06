import os
import json
import time
import math
import random
import argparse
import threading
import queue

from dataclasses import asdict
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from backbone_dil_unshared import (
    ModelConfig,
    BlockMaskedCausalDynamicConvLM,
    count_parameters,
    compute_num_dilated_layers,
)


torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


# ============================================================
# 1. 固定随机种子
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ============================================================
# 2. 文本文件检查
# ============================================================

def check_text_file(data_path: str):
    """
    只检查文件是否存在、是否为空，不把文件整体读入内存。
    """

    if not os.path.isfile(data_path):
        raise FileNotFoundError(
            f"\n没有找到训练文本文件: {data_path}\n"
            f"请确认当前目录下存在 a.txt，或者通过 --data_path 指定正确路径。\n"
        )

    if os.path.getsize(data_path) == 0:
        raise ValueError(f"{data_path} 是空文件，无法训练。")


# ============================================================
# 3. 流式构建字符词表
# ============================================================

def build_char_vocab_streaming(
    data_path: str,
    chunk_size: int = 1024 * 1024 * 16
) -> Tuple[Dict[str, int], Dict[int, str], int]:
    """
    流式构建字符级词表，适合超大文本。

    不会执行：
        text = f.read()

    返回：
        stoi
        itos
        total_chars
    """

    check_text_file(data_path)

    charset = set()
    total_chars = 0

    print("=" * 80)
    print("开始流式构建字符级词表")
    print(f"Data path: {data_path}")
    print(f"Chunk size: {chunk_size:,} chars")
    print("=" * 80)

    with open(data_path, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            charset.update(chunk)
            total_chars += len(chunk)

            if total_chars % (chunk_size * 16) < chunk_size:
                print(
                    f"已扫描字符数: {total_chars:,}, "
                    f"当前词表大小: {len(charset):,}"
                )

    if total_chars == 0:
        raise ValueError(f"{data_path} 是空文件，无法训练。")

    chars = sorted(list(charset))

    stoi = {
        ch: i
        for i, ch in enumerate(chars)
    }

    itos = {
        i: ch
        for ch, i in stoi.items()
    }

    print("=" * 80)
    print("词表构建完成")
    print(f"总字符数: {total_chars:,}")
    print(f"词表大小: {len(stoi):,}")
    print("=" * 80)

    return stoi, itos, total_chars


# ============================================================
# 4. 未知字符 fallback
# ============================================================

def get_unknown_fallback_id(
    stoi: Dict[str, int]
) -> int:
    """
    获取词表外字符的兜底 token id。

    策略：
    1. 优先使用普通空格 ' '；
    2. 如果没有普通空格，尝试常见空白字符；
    3. 如果都没有，则使用词表中 id 最小的 token；
    4. 如果词表为空，才报错。
    """

    preferred_chars = [
        " ",
        "\n",
        "\t",
        "\r"
    ]

    for ch in preferred_chars:
        if ch in stoi:
            return int(stoi[ch])

    if len(stoi) > 0:
        return int(min(stoi.values()))

    raise ValueError("词表为空，无法为未知字符选择兜底 token。")


# ============================================================
# 5. memmap 编码
# ============================================================

def encode_text_to_memmap(
    data_path: str,
    stoi: Dict[str, int],
    save_dir: str,
    total_chars: Optional[int] = None,
    chunk_size: int = 1024 * 1024 * 4,
    force_rebuild: bool = False
) -> Tuple[str, int, str]:
    """
    将文本编码为 numpy memmap 文件。

    当前策略：
    - 如果已有 memmap，并且没有 force_rebuild：
        直接复用；
        不检查 data_path；
        不检查当前文本；
        不检查当前词表是否兼容；
        不重新扫描；
        不重新编码。
    - 如果没有可复用 memmap，才检查文本文件并进行流式统计/编码。
    - 编码时遇到词表外字符：
        不报 KeyError；
        映射到 fallback token。
    """

    os.makedirs(save_dir, exist_ok=True)

    vocab_size = len(stoi)

    if vocab_size <= 65536:
        dtype = np.uint16
        dtype_name = "uint16"
    else:
        dtype = np.uint32
        dtype_name = "uint32"

    memmap_path = os.path.join(
        save_dir,
        f"tokens_{dtype_name}.memmap"
    )

    meta_path = os.path.join(
        save_dir,
        "tokens_memmap_meta.json"
    )

    if (
        not force_rebuild
        and os.path.isfile(memmap_path)
    ):
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)

                meta_dtype = str(meta.get("dtype", dtype_name))
                meta_total_tokens = int(meta.get("total_tokens", -1))

                if meta_dtype == dtype_name and meta_total_tokens > 0:
                    print("=" * 80)
                    print("检测到已有 tokens memmap，直接复用，不重新编码，不检查文本，不检查词表")
                    print(f"Memmap path: {memmap_path}")
                    print(f"Meta path: {meta_path}")
                    print(f"Total tokens: {meta_total_tokens:,}")
                    print(f"DType: {dtype_name}")
                    print("=" * 80)

                    return memmap_path, meta_total_tokens, dtype_name

            except Exception as e:
                print("=" * 80)
                print("读取已有 memmap meta 失败，将根据 memmap 文件大小推断 token 数")
                print("仍然不会检查文本，也不会重新编码")
                print(f"Error: {e}")
                print("=" * 80)

        file_size = os.path.getsize(memmap_path)
        item_size = np.dtype(dtype).itemsize

        if file_size > 0 and file_size % item_size == 0:
            inferred_total_tokens = file_size // item_size

            print("=" * 80)
            print("检测到已有 tokens memmap，meta 不可用或不存在，已根据文件大小直接复用")
            print("不检查文本，不检查词表，不重新编码")
            print(f"Memmap path: {memmap_path}")
            print(f"Memmap file size: {file_size:,} bytes")
            print(f"Total tokens inferred: {inferred_total_tokens:,}")
            print(f"DType: {dtype_name}")
            print("=" * 80)

            return memmap_path, inferred_total_tokens, dtype_name

        raise RuntimeError(
            f"检测到已有 memmap 文件，但文件大小无法按 dtype={dtype_name} 推断 token 数: {memmap_path}"
        )

    check_text_file(data_path)

    fallback_id = get_unknown_fallback_id(stoi)

    fallback_char = None

    for ch, idx in stoi.items():
        if int(idx) == int(fallback_id):
            fallback_char = ch
            break

    if fallback_char is None:
        fallback_char = "<unknown-existing-token>"

    print("=" * 80)
    print("未知字符处理策略")
    print("编码时遇到词表外字符不会报错")
    print(f"Fallback token id: {fallback_id}")
    print(f"Fallback token char: {repr(fallback_char)}")
    print("=" * 80)

    if total_chars is None:
        print("=" * 80)
        print("未检测到可复用 memmap，开始流式统计字符数")
        print("=" * 80)

        total_chars = 0

        with open(data_path, "r", encoding="utf-8") as f:
            while True:
                chunk = f.read(chunk_size)

                if not chunk:
                    break

                total_chars += len(chunk)

                if total_chars % (chunk_size * 16) < chunk_size:
                    print(f"已统计字符数: {total_chars:,}")

    print("=" * 80)
    print("开始流式编码文本到 memmap")
    print(f"Data path: {data_path}")
    print(f"Memmap path: {memmap_path}")
    print(f"Total chars/tokens: {total_chars:,}")
    print(f"DType: {dtype_name}")
    print("=" * 80)

    arr = np.memmap(
        memmap_path,
        dtype=dtype,
        mode="w+",
        shape=(total_chars,)
    )

    offset = 0

    with open(data_path, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            encoded = np.fromiter(
                (
                    stoi.get(ch, fallback_id)
                    for ch in chunk
                ),
                dtype=dtype,
                count=len(chunk)
            )

            arr[
                offset : offset + len(encoded)
            ] = encoded

            offset += len(encoded)

            if offset % (chunk_size * 16) < chunk_size:
                print(f"已编码 tokens: {offset:,} / {total_chars:,}")

    arr.flush()

    if offset != total_chars:
        raise RuntimeError(
            f"编码 token 数量不一致，预期 {total_chars:,}，实际 {offset:,}"
        )

    meta = {
        "data_path": os.path.abspath(data_path),
        "memmap_path": os.path.abspath(memmap_path),
        "total_tokens": int(total_chars),
        "dtype": dtype_name,
        "vocab_size": int(vocab_size),
        "unknown_fallback_id": int(fallback_id),
        "unknown_fallback_char": fallback_char
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            meta,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("=" * 80)
    print("memmap 编码完成")
    print(f"Memmap path: {memmap_path}")
    print(f"Total tokens: {total_chars:,}")
    print("=" * 80)

    return memmap_path, total_chars, dtype_name


# ============================================================
# 6. 解码 / 词表保存加载
# ============================================================

def decode_ids(ids: List[int], itos: Dict[int, str]) -> str:
    fallback_char = " "

    if " " not in set(itos.values()):
        fallback_char = ""

    return "".join(
        [
            itos.get(
                int(i),
                fallback_char
            )
            for i in ids
        ]
    )


def load_vocab(save_dir: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    path = os.path.join(
        save_dir,
        "vocab.json"
    )

    if not os.path.isfile(path):
        raise FileNotFoundError(f"没有找到词表文件: {path}")

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    stoi = obj["stoi"]

    itos = {
        int(k): v
        for k, v in obj["itos"].items()
    }

    print("=" * 80)
    print("检测到已有词表，已直接加载，绝不重复构建词表")
    print(f"Vocab path: {path}")
    print(f"Vocab size: {len(stoi):,}")
    print("=" * 80)

    return stoi, itos


def save_vocab(
    save_dir: str,
    stoi: Dict[str, int],
    itos: Dict[int, str]
):
    path = os.path.join(
        save_dir,
        "vocab.json"
    )

    obj = {
        "stoi": stoi,
        "itos": {
            str(k): v
            for k, v in itos.items()
        }
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            ensure_ascii=False,
            indent=2
        )


# ============================================================
# 7. memmap 随机截取训练数据
# ============================================================

class RandomTextChunkMemmapDataset:
    """
    针对超大文本的随机截取数据集。

    数据存储在磁盘 memmap 中，不会整体加载进内存。

    每个 step 随机截取 block_size + 1 个 token：

        chunk = tokens[i : i + block_size + 1]

        x = chunk[:-1]
        y = chunk[1:]

    性能优化版：

    新增 get_batch_cpu():

        - 后台线程可以调用 get_batch_cpu() 从 memmap 读取数据；
        - 返回 CPU tensor；
        - 如果目标 device 是 cuda，则使用 pinned memory；
        - 后续可由 CudaPrefetchBatchLoader 使用独立 CUDA stream
          异步拷贝到 GPU。

    保留 get_batch():

        - 兼容旧代码；
        - 同步读取并拷贝到 self.device；
        - eval 阶段可以继续使用。
    """

    def __init__(
        self,
        memmap_path: str,
        total_tokens: int,
        dtype_name: str,
        block_size: int,
        device: torch.device,
        train_fraction: float = 0.9
    ):
        if total_tokens < block_size + 2:
            raise ValueError(
                f"文本 token 数量太少: {total_tokens}，"
                f"至少需要 block_size + 2 = {block_size + 2}"
            )

        if dtype_name == "uint16":
            dtype = np.uint16
        elif dtype_name == "uint32":
            dtype = np.uint32
        else:
            raise ValueError(f"未知 memmap dtype_name: {dtype_name}")

        if not os.path.isfile(memmap_path):
            raise FileNotFoundError(f"没有找到 memmap 文件: {memmap_path}")

        self.tokens = np.memmap(
            memmap_path,
            dtype=dtype,
            mode="r",
            shape=(total_tokens,)
        )

        split_index = int(total_tokens * train_fraction)

        if split_index < block_size + 2:
            split_index = block_size + 2

        if total_tokens - split_index < block_size + 2:
            split_index = max(
                block_size + 2,
                total_tokens - block_size - 2
            )

        self.train_start = 0
        self.train_end = split_index

        self.val_start = split_index
        self.val_end = total_tokens

        if self.val_end - self.val_start < block_size + 2:
            self.val_start = self.train_start
            self.val_end = self.train_end

        self.total_tokens = int(total_tokens)
        self.block_size = int(block_size)
        self.device = device
        self.memmap_path = memmap_path
        self.dtype_name = dtype_name

        # CUDA 训练时启用 pinned memory，方便 non_blocking=True 异步 H2D。
        self.pin_memory = (
            self.device.type == "cuda"
        )

        print("=" * 80)
        print("Memmap Dataset 初始化完成")
        print(f"Memmap path: {self.memmap_path}")
        print(f"Total tokens: {self.total_tokens:,}")
        print(f"DType: {self.dtype_name}")
        print(f"Train range: [{self.train_start:,}, {self.train_end:,})")
        print(f"Val range:   [{self.val_start:,}, {self.val_end:,})")
        print(f"Block size: {self.block_size:,}")
        print(f"Pin memory: {self.pin_memory}")
        print("=" * 80)

    def _get_split_range(
        self,
        split: str
    ) -> Tuple[int, int]:
        if split == "train":
            start_base = self.train_start
            end_base = self.train_end
        elif split in ["val", "test"]:
            start_base = self.val_start
            end_base = self.val_end
        else:
            raise ValueError(f"未知 split: {split}")

        return start_base, end_base

    def get_batch_cpu(
        self,
        split: str,
        batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        只在 CPU 上准备 batch。

        返回：
            x_cpu: [B, block_size], dtype=torch.long, CPU tensor
            y_cpu: [B, block_size], dtype=torch.long, CPU tensor

        如果 self.pin_memory=True，则返回 pinned memory tensor，
        后续可以：

            x_cpu.to(cuda, non_blocking=True)

        配合独立 CUDA stream 实现异步 H2D 拷贝。
        """

        start_base, end_base = self._get_split_range(
            split=split
        )

        max_start = end_base - start_base - self.block_size - 1

        if max_start <= 0:
            raise ValueError("数据太短，无法采样。")

        starts = np.random.randint(
            low=0,
            high=max_start + 1,
            size=(batch_size,)
        )

        x_np = np.empty(
            shape=(batch_size, self.block_size),
            dtype=np.int64
        )

        y_np = np.empty(
            shape=(batch_size, self.block_size),
            dtype=np.int64
        )

        for i, rel_start in enumerate(starts):
            s = start_base + int(rel_start)

            chunk = self.tokens[
                s : s + self.block_size + 1
            ]

            x_np[
                i,
                :
            ] = chunk[:-1]

            y_np[
                i,
                :
            ] = chunk[1:]

        x_cpu = torch.from_numpy(
            x_np
        )

        y_cpu = torch.from_numpy(
            y_np
        )

        if self.pin_memory:
            x_cpu = x_cpu.pin_memory()
            y_cpu = y_cpu.pin_memory()

        return x_cpu, y_cpu

    def get_batch(
        self,
        split: str,
        batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        兼容旧接口。

        同步：
            1. CPU 从 memmap 读取；
            2. CPU tensor 转到 self.device。

        训练阶段建议使用 CudaPrefetchBatchLoader。
        评估阶段可以继续使用这个简单接口。
        """

        x_cpu, y_cpu = self.get_batch_cpu(
            split=split,
            batch_size=batch_size
        )

        x = x_cpu.to(
            device=self.device,
            non_blocking=True
        )

        y = y_cpu.to(
            device=self.device,
            non_blocking=True
        )

        return x, y

class CudaPrefetchBatchLoader:
    """
    CUDA 预取 batch loader。

    目的：
        在 GPU 训练当前 batch 的同时，
        后台线程提前从 memmap 读取后续 batch，
        并尽量用独立 CUDA stream 异步搬运到 GPU。

    适用：
        - RandomTextChunkMemmapDataset
        - 单进程单卡训练
        - memmap 随机截取
        - batch_size 较小但 block_size 很长的长上下文训练

    工作流程：

        CPU producer thread:
            dataset.get_batch_cpu(split, batch_size)
            放入 CPU queue

        主训练线程:
            从 CPU queue 取一个 CPU batch
            在 prefetch_stream 上执行:
                x_gpu = x_cpu.to(cuda, non_blocking=True)
                y_gpu = y_cpu.to(cuda, non_blocking=True)

            训练当前 batch 时，后台线程已经在准备下一个 CPU batch。

    注意：
        - 如果 device 不是 cuda，则退化为同步 get_batch。
        - 如果出现后台线程异常，会在主线程 next() 时抛出。
        - stop() 应在训练结束或异常时调用。
    """

    def __init__(
        self,
        dataset: RandomTextChunkMemmapDataset,
        split: str,
        batch_size: int,
        device: torch.device,
        prefetch_batches: int = 2
    ):
        self.dataset = dataset
        self.split = split
        self.batch_size = int(batch_size)
        self.device = device
        self.prefetch_batches = max(
            1,
            int(prefetch_batches)
        )

        self.use_cuda = (
            self.device.type == "cuda"
        )

        self.cpu_queue = queue.Queue(
            maxsize=self.prefetch_batches
        )

        self.stop_event = threading.Event()

        self.worker_exception = None

        self.worker = None

        self.prefetch_stream = None

        self.next_x = None
        self.next_y = None

        self.started = False

        if self.use_cuda:
            self.prefetch_stream = torch.cuda.Stream(
                device=self.device
            )

    def _worker_loop(self):
        """
        CPU 后台线程。

        不接触 CUDA。
        只负责：
            - 从 memmap 随机读取；
            - 构造 CPU pinned tensor；
            - 放入队列。
        """

        try:
            while not self.stop_event.is_set():
                batch = self.dataset.get_batch_cpu(
                    split=self.split,
                    batch_size=self.batch_size
                )

                while not self.stop_event.is_set():
                    try:
                        self.cpu_queue.put(
                            batch,
                            timeout=0.1
                        )
                        break
                    except queue.Full:
                        continue

        except BaseException as e:
            self.worker_exception = e

            try:
                self.cpu_queue.put_nowait(
                    e
                )
            except Exception:
                pass

    def start(self):
        """
        启动后台线程，并预取第一个 GPU batch。
        """

        if self.started:
            return

        self.started = True

        if not self.use_cuda:
            return

        self.worker = threading.Thread(
            target=self._worker_loop,
            daemon=True
        )

        self.worker.start()

        # 先同步发起第一个 GPU batch 的预取。
        self._preload_next_gpu_batch()

    def stop(self):
        """
        停止后台线程。
        """

        self.stop_event.set()

        if self.worker is not None:
            self.worker.join(
                timeout=2.0
            )

        self.worker = None

        self.next_x = None
        self.next_y = None

    def _raise_if_worker_failed(self):
        if self.worker_exception is not None:
            raise RuntimeError(
                "CudaPrefetchBatchLoader 后台数据线程异常"
            ) from self.worker_exception

    def _get_cpu_batch_from_queue(
        self
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从 CPU queue 里取一个 batch。

        如果后台线程出错，则抛出异常。
        如果队列暂时为空，则继续等待，而不是直接崩溃。
        """

        while True:
            self._raise_if_worker_failed()

            try:
                item = self.cpu_queue.get(
                    timeout=1.0
                )
            except queue.Empty:
                if self.worker is not None and not self.worker.is_alive():
                    self._raise_if_worker_failed()
                    raise RuntimeError(
                        "CudaPrefetchBatchLoader 后台数据线程已退出，但没有返回 batch。"
                    )

                continue

            if isinstance(item, BaseException):
                self.worker_exception = item
                self._raise_if_worker_failed()

            x_cpu, y_cpu = item

            return x_cpu, y_cpu


    def _preload_next_gpu_batch(self):
        """
        从 CPU queue 取 batch，并在独立 CUDA stream 中异步拷贝到 GPU。
        """

        if not self.use_cuda:
            return

        x_cpu, y_cpu = self._get_cpu_batch_from_queue()

        with torch.cuda.stream(
            self.prefetch_stream
        ):
            self.next_x = x_cpu.to(
                device=self.device,
                non_blocking=True
            )

            self.next_y = y_cpu.to(
                device=self.device,
                non_blocking=True
            )

    def next(
        self
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回一个已经在 GPU 上的 batch。

        CUDA 模式：
            - 等待 prefetch stream 完成当前 next_x/next_y；
            - record_stream，防止 allocator 过早复用；
            - 立即启动下一个 batch 的异步预取；
            - 返回当前 batch。

        CPU 模式：
            - 退化为 dataset.get_batch。
        """

        if not self.use_cuda:
            return self.dataset.get_batch(
                split=self.split,
                batch_size=self.batch_size
            )

        if not self.started:
            self.start()

        torch.cuda.current_stream(
            device=self.device
        ).wait_stream(
            self.prefetch_stream
        )

        x = self.next_x
        y = self.next_y

        if x is None or y is None:
            raise RuntimeError(
                "CudaPrefetchBatchLoader 内部错误：next_x/next_y 为空。"
            )

        x.record_stream(
            torch.cuda.current_stream(
                device=self.device
            )
        )

        y.record_stream(
            torch.cuda.current_stream(
                device=self.device
            )
        )

        # 启动下一个 batch 的预取。
        self._preload_next_gpu_batch()

        return x, y


# ============================================================
# 8. 评估
# ============================================================

@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    dataset: RandomTextChunkMemmapDataset,
    batch_size: int,
    eval_iters: int,
    precision: str,
    device: torch.device
) -> Dict[str, float]:
    model.eval()

    result = {}

    use_fp16 = (
        precision == "fp16"
        and device.type == "cuda"
    )

    use_bf16 = (
        precision == "bf16"
        and device.type == "cuda"
    )

    autocast_enabled = (
        device.type == "cuda"
        and (
            use_fp16
            or use_bf16
        )
    )

    if use_fp16:
        autocast_dtype = torch.float16
    elif use_bf16:
        autocast_dtype = torch.bfloat16
    else:
        autocast_dtype = torch.float32

    for split in ["train", "val"]:
        losses = []

        for _ in range(eval_iters):
            x, y = dataset.get_batch(
                split=split,
                batch_size=batch_size
            )

            with torch.amp.autocast(
                "cuda",
                enabled=autocast_enabled,
                dtype=autocast_dtype
            ):
                _, loss = model(
                    input_ids=x,
                    targets=y
                )

            losses.append(float(loss.item()))

            del x
            del y
            del loss

        result[split] = sum(losses) / len(losses)

    model.train()

    return result


# ============================================================
# 9. 保存与加载
# ============================================================

def save_config(
    save_dir: str,
    config: ModelConfig,
    args_dict: Dict
):
    path = os.path.join(
        save_dir,
        "config.json"
    )

    obj = {
        "model_config": asdict(config),
        "args": args_dict
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            ensure_ascii=False,
            indent=2
        )


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer],
    epoch: int,
    global_step: int,
    best_val_loss: float,
    stoi: Dict[str, int],
    itos: Dict[int, str],
    config: ModelConfig,
    args_dict: Dict
):
    ckpt = {
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "stoi": stoi,
        "itos": {
            str(k): v
            for k, v in itos.items()
        },
        "config": asdict(config),
        "args": args_dict
    }

    torch.save(
        ckpt,
        path
    )


def _upgrade_config_dict(
    config_dict: Dict,
    args_dict: Optional[Dict] = None
) -> Dict:
    """
    兼容旧 checkpoint/config，并补齐新版 ModelConfig 字段。

    兼容点：
        1. 旧 checkpoint 可能没有 adaptive_checkpoint。
        2. 旧 checkpoint 可能没有 checkpoint_activation_threshold_mb。
        3. 旧 checkpoint 可能没有 use_preallocated_chunk_output。
        4. 旧 checkpoint 可能没有 cuda_warmup_repeat。
        5. 旧 checkpoint 可能使用 stream_k/materialized_windows。
        6. 新版默认 dynamic_conv_impl 建议为 cuda_fused。
    """

    if args_dict is None:
        args_dict = {}

    config_dict = dict(config_dict)

    vocab_size = int(
        config_dict.get(
            "vocab_size"
        )
    )

    block_size = int(
        config_dict.get(
            "block_size",
            args_dict.get(
                "block_size",
                1024
            )
        )
    )

    embed_dim = int(
        config_dict.get(
            "embed_dim",
            args_dict.get(
                "embed_dim",
                256
            )
        )
    )

    if "source_channels" in config_dict:
        source_channels = int(
            config_dict[
                "source_channels"
            ]
        )
    elif "source_num_blocks" in config_dict:
        source_channels = int(
            config_dict[
                "source_num_blocks"
            ]
        )
    else:
        source_channels = int(
            args_dict.get(
                "source_channels",
                16
            )
        )

    num_kernels = int(
        config_dict.get(
            "num_kernels",
            args_dict.get(
                "num_kernels",
                16
            )
        )
    )

    dynamic_kernel_size = int(
        config_dict.get(
            "dynamic_kernel_size",
            args_dict.get(
                "dynamic_kernel_size",
                3
            )
        )
    )

    source_conv_kernel_size = int(
        config_dict.get(
            "source_conv_kernel_size",
            args_dict.get(
                "source_conv_kernel_size",
                3
            )
        )
    )

    kernel_gen_kernel_size = int(
        config_dict.get(
            "kernel_gen_kernel_size",
            args_dict.get(
                "kernel_gen_kernel_size",
                3
            )
        )
    )

    mlp_ratio = float(
        config_dict.get(
            "mlp_ratio",
            args_dict.get(
                "mlp_ratio",
                4.0
            )
        )
    )

    dropout = float(
        config_dict.get(
            "dropout",
            args_dict.get(
                "dropout",
                0.1
            )
        )
    )

    use_bias = bool(
        config_dict.get(
            "use_bias",
            True
        )
    )

    use_sinusoidal_pos = bool(
        config_dict.get(
            "use_sinusoidal_pos",
            True
        )
    )

    use_checkpoint = bool(
        config_dict.get(
            "use_checkpoint",
            not bool(
                args_dict.get(
                    "no_checkpoint",
                    False
                )
            )
        )
    )

    adaptive_checkpoint = bool(
        config_dict.get(
            "adaptive_checkpoint",
            args_dict.get(
                "adaptive_checkpoint",
                True
            )
        )
    )

    checkpoint_activation_threshold_mb = float(
        config_dict.get(
            "checkpoint_activation_threshold_mb",
            args_dict.get(
                "checkpoint_activation_threshold_mb",
                512.0
            )
        )
    )

    chunk_size = int(
        config_dict.get(
            "chunk_size",
            args_dict.get(
                "chunk_size",
                256
            )
        )
    )

    loss_chunk_size = int(
        config_dict.get(
            "loss_chunk_size",
            args_dict.get(
                "loss_chunk_size",
                2048
            )
        )
    )

    if "normalize_kernel" in config_dict:
        normalize_kernel = bool(
            config_dict[
                "normalize_kernel"
            ]
        )
    elif "kernel_norm" in args_dict:
        normalize_kernel = bool(
            args_dict.get(
                "kernel_norm",
                True
            )
        )
    elif "no_kernel_norm" in args_dict:
        normalize_kernel = not bool(
            args_dict.get(
                "no_kernel_norm",
                False
            )
        )
    else:
        normalize_kernel = True

    initial_scale = float(
        config_dict.get(
            "initial_scale",
            args_dict.get(
                "initial_scale",
                0.1
            )
        )
    )

    source_conv_impl = str(
        config_dict.get(
            "source_conv_impl",
            args_dict.get(
                "source_conv_impl",
                "einsum"
            )
        )
    )

    source_fused_mixed = bool(
        config_dict.get(
            "source_fused_mixed",
            not bool(
                args_dict.get(
                    "no_source_fused_mixed",
                    False
                )
            )
        )
    )

    dynamic_conv_impl = str(
        config_dict.get(
            "dynamic_conv_impl",
            args_dict.get(
                "dynamic_conv_impl",
                "cuda_fused"
            )
        )
    )

    dynamic_conv_tmp_mb_limit = float(
        config_dict.get(
            "dynamic_conv_tmp_mb_limit",
            args_dict.get(
                "dynamic_conv_tmp_mb_limit",
                64.0
            )
        )
    )

    cuda_warmup_repeat = int(
        config_dict.get(
            "cuda_warmup_repeat",
            args_dict.get(
                "cuda_warmup_repeat",
                3
            )
        )
    )

    use_nonlinear_residual = bool(
        config_dict.get(
            "use_nonlinear_residual",
            not bool(
                args_dict.get(
                    "no_nonlinear_residual",
                    False
                )
            )
        )
    )

    use_source_nonlinear_residual = bool(
        config_dict.get(
            "use_source_nonlinear_residual",
            bool(
                args_dict.get(
                    "use_source_nonlinear_residual",
                    False
                )
            )
        )
    )

    nonlinear_residual_initial_scale = float(
        config_dict.get(
            "nonlinear_residual_initial_scale",
            args_dict.get(
                "nonlinear_residual_initial_scale",
                0.05
            )
        )
    )

    nonlinear_residual_bias = bool(
        config_dict.get(
            "nonlinear_residual_bias",
            not bool(
                args_dict.get(
                    "no_nonlinear_residual_bias",
                    False
                )
            )
        )
    )

    use_preallocated_chunk_output = bool(
        config_dict.get(
            "use_preallocated_chunk_output",
            args_dict.get(
                "use_preallocated_chunk_output",
                True
            )
        )
    )

    upgraded = {
        "vocab_size": vocab_size,
        "block_size": block_size,
        "embed_dim": embed_dim,
        "source_channels": source_channels,
        "num_kernels": num_kernels,
        "dynamic_kernel_size": dynamic_kernel_size,
        "source_conv_kernel_size": source_conv_kernel_size,
        "kernel_gen_kernel_size": kernel_gen_kernel_size,
        "mlp_ratio": mlp_ratio,
        "dropout": dropout,
        "use_bias": use_bias,
        "use_sinusoidal_pos": use_sinusoidal_pos,
        "use_checkpoint": use_checkpoint,
        "adaptive_checkpoint": adaptive_checkpoint,
        "checkpoint_activation_threshold_mb": checkpoint_activation_threshold_mb,
        "chunk_size": chunk_size,
        "loss_chunk_size": loss_chunk_size,
        "normalize_kernel": normalize_kernel,
        "initial_scale": initial_scale,
        "source_conv_impl": source_conv_impl,
        "source_fused_mixed": source_fused_mixed,
        "dynamic_conv_impl": dynamic_conv_impl,
        "dynamic_conv_tmp_mb_limit": dynamic_conv_tmp_mb_limit,
        "cuda_warmup_repeat": cuda_warmup_repeat,
        "use_nonlinear_residual": use_nonlinear_residual,
        "use_source_nonlinear_residual": use_source_nonlinear_residual,
        "nonlinear_residual_initial_scale": nonlinear_residual_initial_scale,
        "nonlinear_residual_bias": nonlinear_residual_bias,
        "use_preallocated_chunk_output": use_preallocated_chunk_output
    }

    return upgraded


def load_checkpoint(
    checkpoint_path: str,
    device: torch.device
):
    ckpt = torch.load(
        checkpoint_path,
        map_location=device
    )

    args_dict = ckpt.get(
        "args",
        {}
    )

    config_dict = _upgrade_config_dict(
        ckpt["config"],
        args_dict=args_dict
    )

    config = ModelConfig(
        **config_dict
    )

    stoi = ckpt["stoi"]

    itos = {
        int(k): v
        for k, v in ckpt["itos"].items()
    }

    model = BlockMaskedCausalDynamicConvLM(
        config=config
    ).to(device)

    state_dict = ckpt["model_state_dict"]

    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        print("=" * 80)
        print("检测到 checkpoint 权重包含 _orig_mod. 前缀，正在自动去除")
        print("=" * 80)

        state_dict = {
            k.replace("_orig_mod.", "", 1): v
            for k, v in state_dict.items()
        }

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model.eval()

    return model, stoi, itos, config, ckpt


def load_training_checkpoint(
    checkpoint_path: str,
    device: torch.device
):
    """
    加载训练 checkpoint，用于从中断处继续训练。
    """

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"没有找到 resume checkpoint: {checkpoint_path}"
        )

    ckpt = torch.load(
        checkpoint_path,
        map_location=device
    )

    args_dict = ckpt.get(
        "args",
        {}
    )

    config_dict = _upgrade_config_dict(
        ckpt["config"],
        args_dict=args_dict
    )

    config = ModelConfig(
        **config_dict
    )

    stoi = ckpt["stoi"]

    itos = {
        int(k): v
        for k, v in ckpt["itos"].items()
    }

    return (
        ckpt,
        config,
        stoi,
        itos,
        args_dict
    )


# ============================================================
# 10. 推理补齐
# ============================================================

def right_pad_to_block_size_for_inference(
    input_ids: torch.Tensor,
    block_size: int,
    pad_id: int
) -> Tuple[torch.Tensor, int]:
    """
    推理时将真实上下文右侧补齐到 block_size。

    因模型严格因果，真实位置不会看到右侧补位 token。

    返回：
        padded_input_ids: [B, block_size]
        real_len: 真实上下文长度

    后续必须取：
        logits[:, real_len - 1, :]
    """

    if input_ids.dim() != 2:
        raise ValueError(
            f"input_ids 应为 [B,L]，但得到 {input_ids.shape}"
        )

    B, L = input_ids.shape

    if L > block_size:
        input_ids = input_ids[
            :,
            -block_size:
        ]
        L = block_size

    real_len = L

    if real_len <= 0:
        raise ValueError("real_len 必须大于 0。")

    pad_len = block_size - real_len

    if pad_len > 0:
        pad = torch.full(
            size=(B, pad_len),
            fill_value=pad_id,
            dtype=input_ids.dtype,
            device=input_ids.device
        )

        padded_input_ids = torch.cat(
            [
                input_ids,
                pad
            ],
            dim=1
        )
    else:
        padded_input_ids = input_ids

    if padded_input_ids.size(1) != block_size:
        raise RuntimeError(
            f"推理补齐失败，期望长度 {block_size}，实际 {padded_input_ids.size(1)}"
        )

    return padded_input_ids, real_len


# ============================================================
# 11. 生成
# ============================================================

@torch.no_grad()
def generate(
    model: BlockMaskedCausalDynamicConvLM,
    start_text: str,
    stoi: Dict[str, int],
    itos: Dict[int, str],
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: torch.device,
    precision: str = "fp32"
) -> str:
    """
    生成文本。

    当前版本不再将推理输入右侧 padding 到 block_size。

    推理策略：
    - start_text 按字符编码；
    - 如果当前上下文长度超过 block_size，只取最后 block_size 个 token；
    - 如果当前上下文长度小于 block_size，保持原始长度，不补齐；
    - 直接用当前真实长度 context 前向；
    - 下一个 token 从 logits[:, -1, :] 取得。

    注意：
    - 这里和旧版本不同，旧版本会调用 right_pad_to_block_size_for_inference；
    - 当前版本不会调用 right_pad_to_block_size_for_inference；
    - 因此不会插入任何右侧 padding token；
    - 也不会使用 pad_id。
    """

    model.eval()

    if len(start_text) == 0:
        raise ValueError("start_text 不能为空。")

    fallback_id = get_unknown_fallback_id(stoi)

    ids = [
        stoi.get(
            ch,
            fallback_id
        )
        for ch in start_text
    ]

    input_ids = torch.tensor(
        ids,
        dtype=torch.long,
        device=device
    ).unsqueeze(0)

    block_size = model.config.block_size

    use_fp16 = (
        precision == "fp16"
        and device.type == "cuda"
    )

    use_bf16 = (
        precision == "bf16"
        and device.type == "cuda"
    )

    autocast_enabled = (
        device.type == "cuda"
        and (
            use_fp16
            or use_bf16
        )
    )

    if use_fp16:
        autocast_dtype = torch.float16
    elif use_bf16:
        autocast_dtype = torch.bfloat16
    else:
        autocast_dtype = torch.float32

    for _ in range(max_new_tokens):
        context = input_ids[
            :,
            -block_size:
        ]

        with torch.amp.autocast(
            "cuda",
            enabled=autocast_enabled,
            dtype=autocast_dtype
        ):
            logits, _ = model(
                input_ids=context,
                targets=None
            )

        next_logits = logits[
            :,
            -1,
            :
        ]

        if temperature <= 0:
            next_id = torch.argmax(
                next_logits,
                dim=-1,
                keepdim=True
            )
        else:
            next_logits = next_logits / temperature

            if top_k is not None and top_k > 0:
                top_k_eff = min(
                    int(top_k),
                    next_logits.size(-1)
                )

                values, _ = torch.topk(
                    next_logits,
                    k=top_k_eff,
                    dim=-1
                )

                threshold = values[
                    :,
                    -1
                ].unsqueeze(-1)

                next_logits = torch.where(
                    next_logits < threshold,
                    torch.full_like(
                        next_logits,
                        float("-inf")
                    ),
                    next_logits
                )

            probs = F.softmax(
                next_logits,
                dim=-1
            )

            next_id = torch.multinomial(
                probs,
                num_samples=1
            )

        input_ids = torch.cat(
            [
                input_ids,
                next_id
            ],
            dim=1
        )

        del context
        del logits
        del next_logits
        del next_id

    return decode_ids(
        input_ids[0].tolist(),
        itos
    )

# ============================================================
# 12. Debug: 模型结构信息
# ============================================================

def debug_print_model_structure(
    model: BlockMaskedCausalDynamicConvLM
):
    config = model.config

    num_layers = compute_num_dilated_layers(
        config.block_size
    )

    kernel_channels = config.num_kernels * config.dynamic_kernel_size

    print("=" * 80)
    print("Dilated Unshared Dynamic Conv 模型结构检查")
    print(f"block_size: {config.block_size}")
    print(f"embed_dim D: {config.embed_dim}")
    print(f"source_channels C: {config.source_channels}")
    print(f"num_kernels N: {config.num_kernels}")
    print(f"dynamic_kernel_size K: {config.dynamic_kernel_size}")
    print(f"N*K kernel_channels: {kernel_channels}")
    print(f"source_conv_kernel_size: {config.source_conv_kernel_size}")
    print(f"kernel_gen_kernel_size: {config.kernel_gen_kernel_size}")
    print(f"chunk_size: {config.chunk_size}")
    print(f"loss_chunk_size: {config.loss_chunk_size}")
    print(f"normalize_kernel: {config.normalize_kernel}")
    print(f"initial_scale: {config.initial_scale}")
    print(f"use_checkpoint: {config.use_checkpoint}")
    print(f"adaptive_checkpoint: {getattr(config, 'adaptive_checkpoint', True)}")
    print(f"checkpoint_activation_threshold_mb: {getattr(config, 'checkpoint_activation_threshold_mb', 512.0)}")
    print(f"source_conv_impl: {config.source_conv_impl}")
    print(f"source_fused_mixed: {config.source_fused_mixed}")
    print(f"dynamic_conv_impl: {config.dynamic_conv_impl}")
    print(f"dynamic_conv_tmp_mb_limit: {config.dynamic_conv_tmp_mb_limit}")
    print(f"cuda_warmup_repeat: {getattr(config, 'cuda_warmup_repeat', 3)}")
    print(f"use_nonlinear_residual: {config.use_nonlinear_residual}")
    print(f"use_source_nonlinear_residual: {config.use_source_nonlinear_residual}")
    print(f"nonlinear_residual_initial_scale: {config.nonlinear_residual_initial_scale}")
    print(f"nonlinear_residual_bias: {config.nonlinear_residual_bias}")
    print(f"use_preallocated_chunk_output: {getattr(config, 'use_preallocated_chunk_output', True)}")
    print(f"computed dilated layers: {num_layers}")
    print("dilations:", [2 ** (i + 1) for i in range(num_layers)])
    print("-" * 80)
    print("Kernel generator 结构:")
    print("  input:        [B,D,T]")
    print("  D-mix:        [B,D,T] -> [B,N*K,T]")
    print("  LayerNorm:    over N*K")
    print("  Conv:         depthwise causal Conv1d, groups=N*K")
    print("  Residual:     logits = identity + conv(LN(identity_local_segment))")
    print("  reshape:      [B,N*K,T] -> [B,N,K,T]")
    print("  softmax dim:  K")
    print("  flip K:       使 kk=0 表示当前位置")
    print("  output:       [B,K,N,T]")
    print("-" * 80)
    print("Dynamic conv 结构:")
    print("  h_full:       [B,D,L]")
    print("  kernel_chunk: [B,K,N,T]")
    print("  kernel_mix:   [D,N]")
    print("  output:       [B,D,T]")
    print("=" * 80)


# ============================================================
# 13. Debug: shape 检查
# ============================================================

@torch.no_grad()
def debug_check_shapes(
    model: BlockMaskedCausalDynamicConvLM,
    vocab_size: int,
    device: torch.device,
    precision: str = "fp32"
):
    model.eval()

    B = 2
    L = min(
        64,
        model.config.block_size
    )

    x = torch.randint(
        low=0,
        high=vocab_size,
        size=(B, L),
        device=device
    )

    print("=" * 80)
    print("Shape 检查")
    print(f"input_ids: {x.shape}")

    tok = model.token_embedding(x)

    print(f"token embedding: {tok.shape}")

    if model.config.use_sinusoidal_pos:
        pos = model.pos_encoding(
            length=L,
            device=device,
            dtype=tok.dtype
        )
        print(f"pos encoding: {pos.shape}")

    use_fp16 = (
        precision == "fp16"
        and device.type == "cuda"
    )

    use_bf16 = (
        precision == "bf16"
        and device.type == "cuda"
    )

    autocast_enabled = (
        device.type == "cuda"
        and (
            use_fp16
            or use_bf16
        )
    )

    if use_fp16:
        autocast_dtype = torch.float16
    elif use_bf16:
        autocast_dtype = torch.bfloat16
    else:
        autocast_dtype = torch.float32

    with torch.amp.autocast(
        "cuda",
        enabled=autocast_enabled,
        dtype=autocast_dtype
    ):
        logits, loss = model(
            input_ids=x,
            targets=x
        )

    if logits is None:
        print("logits: None，因为 targets 不为 None 时模型分块计算 loss，不返回完整 logits。")
    else:
        print(f"logits: {logits.shape}")

    print(f"loss: {loss.item():.6f}")
    print("Shape 检查通过")
    print("=" * 80)

    model.train()


# ============================================================
# 14. Debug: 模型严格因果性检查
# ============================================================

@torch.no_grad()
def debug_check_model_causality(
    model: BlockMaskedCausalDynamicConvLM,
    vocab_size: int,
    device: torch.device,
    precision: str = "fp32"
):
    """
    修改后半段 token，检查前半段 logits 是否不变。
    """

    model.eval()

    L = min(
        128,
        model.config.block_size
    )

    if L < 16:
        print("L 太短，跳过因果性测试。")
        return

    mid = L // 2

    x1 = torch.randint(
        low=0,
        high=vocab_size,
        size=(1, L),
        device=device
    )

    x2 = x1.clone()

    x2[
        :,
        mid:
    ] = torch.randint(
        low=0,
        high=vocab_size,
        size=(1, L - mid),
        device=device
    )

    use_fp16 = (
        precision == "fp16"
        and device.type == "cuda"
    )

    use_bf16 = (
        precision == "bf16"
        and device.type == "cuda"
    )

    autocast_enabled = (
        device.type == "cuda"
        and (
            use_fp16
            or use_bf16
        )
    )

    if use_fp16:
        autocast_dtype = torch.float16
    elif use_bf16:
        autocast_dtype = torch.bfloat16
    else:
        autocast_dtype = torch.float32

    with torch.amp.autocast(
        "cuda",
        enabled=autocast_enabled,
        dtype=autocast_dtype
    ):
        logits1, _ = model(
            input_ids=x1,
            targets=None
        )

        logits2, _ = model(
            input_ids=x2,
            targets=None
        )

    diff_before = (
        logits1[
            :,
            :mid,
            :
        ]
        -
        logits2[
            :,
            :mid,
            :
        ]
    ).abs().max().item()

    diff_after = (
        logits1[
            :,
            mid:,
            :
        ]
        -
        logits2[
            :,
            mid:,
            :
        ]
    ).abs().max().item()

    tolerance = 1e-4 if precision in ["fp16", "bf16"] else 1e-5

    print("=" * 80)
    print("模型严格因果性测试")
    print(f"L={L}, 修改位置从 {mid} 开始")
    print(f"normalize_kernel: {model.config.normalize_kernel}")
    print(f"diff_before_mid = {diff_before:.12f}")
    print(f"diff_after_mid  = {diff_after:.12f}")
    print(f"tolerance       = {tolerance:.12f}")

    if diff_before <= tolerance:
        print("因果性测试通过：未来 token 未影响未来之前位置的 logits。")
    else:
        print("警告：因果性测试未通过，diff_before_mid 超过阈值，可能存在因果泄露。")

    print("=" * 80)

    model.train()

@torch.no_grad()
def warmup_model_cuda_dynamic_conv_if_needed(
    model: nn.Module,
    config: ModelConfig,
    args,
    device: torch.device,
    precision: str,
    batch_size: int
) -> Optional[List[dict]]:
    """
    创建模型或加载模型权重后，对模型内部 CUDA fused dynamic conv 做 warmup。

    调用条件：
        1. device 是 cuda；
        2. args.no_cuda_warmup 为 False；
        3. config.dynamic_conv_impl 是 "cuda_fused" 或 "auto"。

    注意：
        - 只 warmup 动态卷积 CUDA 算子 plan；
        - 不执行完整模型 forward；
        - 不读数据；
        - 不影响训练随机状态中的模型参数；
        - warmup 使用空 tensor，只为 cuda_ops 选择/缓存 plan。
    """

    if device.type != "cuda":
        print("=" * 80)
        print("CUDA warmup 跳过：当前 device 不是 cuda")
        print("=" * 80)
        return None

    if bool(
        getattr(
            args,
            "no_cuda_warmup",
            False
        )
    ):
        print("=" * 80)
        print("CUDA warmup 跳过：命令行指定 --no_cuda_warmup")
        print("=" * 80)
        return None

    if str(config.dynamic_conv_impl) not in (
        "cuda_fused",
        "auto"
    ):
        print("=" * 80)
        print(
            "CUDA warmup 跳过：dynamic_conv_impl 不是 cuda_fused/auto，"
            f"当前为 {config.dynamic_conv_impl}"
        )
        print("=" * 80)
        return None

    # 暂时硬编码fp32，应当根据网络实际进入这一层的类型进行warmup，绝对不对根据外面设的精度来
    warmup_dtype = torch.float32

    if warmup_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "请求 bf16 warmup，但当前 GPU 不支持 bf16。"
        )

    repeat = int(
        getattr(
            args,
            "cuda_warmup_repeat",
            getattr(
                config,
                "cuda_warmup_repeat",
                3
            )
        )
    )

    warmup_all_chunks = bool(
        getattr(
            args,
            "cuda_warmup_all_chunks",
            False
        )
    )

    include_backward = not bool(
        getattr(
            args,
            "cuda_warmup_forward_only",
            False
        )
    )

    print("=" * 80)
    print("开始 CUDA fused dynamic conv warmup")
    print(f"dynamic_conv_impl: {config.dynamic_conv_impl}")
    print(f"batch_size: {batch_size}")
    print(f"seq_len: {config.block_size}")
    print(f"chunk_size: {config.chunk_size}")
    print(f"embed_dim D: {config.embed_dim}")
    print(f"num_kernels N: {config.num_kernels}")
    print(f"dynamic_kernel_size K: {config.dynamic_kernel_size}")
    print(f"dtype: {warmup_dtype}")
    print(f"repeat: {repeat}")
    print(f"include_backward: {include_backward}")
    print(f"warmup_all_chunks: {warmup_all_chunks}")
    print("=" * 80)

    t0 = time.time()

    warmup_info = model.warmup_cuda_dynamic_conv(
        batch_size=batch_size,
        seq_len=config.block_size,
        chunk_size=config.chunk_size,
        device=device,
        dtype=warmup_dtype,
        repeat=repeat,
        include_backward=include_backward,
        clear_existing_cache=True,
        warmup_all_chunks=warmup_all_chunks
    )

    elapsed = time.time() - t0

    print("=" * 80)
    print("CUDA fused dynamic conv warmup 完成")
    print(f"耗时: {elapsed:.2f}s")
    print(f"warmup entries: {len(warmup_info)}")
    print("-" * 80)

    max_print = int(
        getattr(
            args,
            "cuda_warmup_print_limit",
            32
        )
    )

    for i, item in enumerate(warmup_info[:max_print]):
        layer_index = item.get(
            "layer_index",
            None
        )

        prefix = (
            f"layer={layer_index} "
            if layer_index is not None
            else ""
        )

        msg = (
            f"{prefix}"
            f"start={item.get('start')} "
            f"end={item.get('end')} "
            f"T={item.get('T')} "
            f"fwd={item.get('forward_plan_id')}:{item.get('forward_plan_name')}"
        )

        if "backward_plan_id" in item:
            msg += (
                f" "
                f"bwd={item.get('backward_plan_id')}:{item.get('backward_plan_name')}"
            )

        print(msg)

    if len(warmup_info) > max_print:
        print(
            f"... 还有 {len(warmup_info) - max_print} 条 warmup 信息未打印"
        )

    print("=" * 80)

    return warmup_info

# ============================================================
# 15. 参数量随 block_size 检查
# ============================================================

def verify_parameter_count_independent_of_block_size(
    config: ModelConfig,
    device: torch.device
):
    c1_dict = asdict(config)
    c2_dict = asdict(config)

    c1_dict["block_size"] = 512
    c2_dict["block_size"] = 2048

    c1 = ModelConfig(**c1_dict)
    c2 = ModelConfig(**c2_dict)

    m1 = BlockMaskedCausalDynamicConvLM(c1).to(device)
    m2 = BlockMaskedCausalDynamicConvLM(c2).to(device)

    n1, _ = count_parameters(m1)
    n2, _ = count_parameters(m2)

    print("=" * 80)
    print("参数量是否随 block_size 改变检查")
    print(f"block_size=512  total params: {n1:,}")
    print(f"block_size=2048 total params: {n2:,}")
    print(f"是否相等: {n1 == n2}")
    print("注意：新版模型层数由 block_size 决定，因此参数量会随 block_size 改变。")
    print("这是当前 dilated-unshared 结构的预期行为。")
    print("=" * 80)


# ============================================================
# 16. 构建 config
# ============================================================

def build_model_config_from_args(
    args,
    vocab_size: int
) -> ModelConfig:
    return ModelConfig(
        vocab_size=vocab_size,
        block_size=args.block_size,
        embed_dim=args.embed_dim,
        source_channels=args.source_channels,
        num_kernels=args.num_kernels,
        dynamic_kernel_size=args.dynamic_kernel_size,
        source_conv_kernel_size=args.source_conv_kernel_size,
        kernel_gen_kernel_size=args.kernel_gen_kernel_size,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        use_bias=True,
        use_sinusoidal_pos=True,
        use_checkpoint=not args.no_checkpoint,
        adaptive_checkpoint=args.adaptive_checkpoint,
        checkpoint_activation_threshold_mb=args.checkpoint_activation_threshold_mb,
        chunk_size=args.chunk_size,
        loss_chunk_size=args.loss_chunk_size,
        normalize_kernel=not args.no_kernel_norm,
        initial_scale=args.initial_scale,
        source_conv_impl=args.source_conv_impl,
        source_fused_mixed=not args.no_source_fused_mixed,
        dynamic_conv_impl=args.dynamic_conv_impl,
        dynamic_conv_tmp_mb_limit=args.dynamic_conv_tmp_mb_limit,
        cuda_warmup_repeat=args.cuda_warmup_repeat,
        use_nonlinear_residual=not args.no_nonlinear_residual,
        use_source_nonlinear_residual=args.use_source_nonlinear_residual,
        nonlinear_residual_initial_scale=args.nonlinear_residual_initial_scale,
        nonlinear_residual_bias=not args.no_nonlinear_residual_bias,
        use_preallocated_chunk_output=not args.no_preallocated_chunk_output
    )


# ============================================================
# 17. 训练
# ============================================================
def train(args):
    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(args.device)

    os.makedirs(
        args.save_dir,
        exist_ok=True
    )

    print("=" * 80)
    print("Dilated Unshared Dynamic Convolution Character LM")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Data path: {args.data_path}")
    print(f"Save dir: {args.save_dir}")
    print(f"Resume from: {args.resume_from}")
    print(f"Epochs target: {args.epochs}")
    print(f"Steps per epoch: {args.steps_per_epoch}")
    print(f"Batch size: {args.batch_size}")
    print(f"Requested block size / ctxlen: {args.block_size}")
    print(f"Embed dim D: {args.embed_dim}")
    print(f"Source channels C: {args.source_channels}")
    print(f"Num kernels N: {args.num_kernels}")
    print(f"Kernel size K: {args.dynamic_kernel_size}")
    print(f"N*K kernel channels: {args.num_kernels * args.dynamic_kernel_size}")
    print(f"Kernel generator kernel size: {args.kernel_gen_kernel_size}")
    print(f"Source conv kernel size: {args.source_conv_kernel_size}")
    print(f"Source conv impl: {args.source_conv_impl}")
    print(f"Source fused mixed: {not args.no_source_fused_mixed}")
    print(f"Dynamic conv impl: {args.dynamic_conv_impl}")
    print(f"Dynamic conv tmp MB limit: {args.dynamic_conv_tmp_mb_limit}")
    print(f"CUDA warmup disabled: {args.no_cuda_warmup}")
    print(f"CUDA warmup repeat: {args.cuda_warmup_repeat}")
    print(f"CUDA warmup all chunks: {args.cuda_warmup_all_chunks}")
    print(f"CUDA warmup forward only: {args.cuda_warmup_forward_only}")
    print(f"Use nonlinear residual: {not args.no_nonlinear_residual}")
    print(f"Use source nonlinear residual: {args.use_source_nonlinear_residual}")
    print(f"Nonlinear residual initial scale: {args.nonlinear_residual_initial_scale}")
    print(f"Nonlinear residual bias: {not args.no_nonlinear_residual_bias}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Loss chunk size: {args.loss_chunk_size}")
    print(f"Kernel normalization: {not args.no_kernel_norm}")
    print(f"Activation checkpointing: {not args.no_checkpoint}")
    print(f"Adaptive checkpointing: {args.adaptive_checkpoint}")
    print(f"Checkpoint threshold MB: {args.checkpoint_activation_threshold_mb}")
    print(f"Use preallocated chunk output: {not args.no_preallocated_chunk_output}")
    print(f"Text chunk size: {args.text_chunk_size:,}")
    print(f"Rebuild memmap: {args.rebuild_memmap}")
    print(f"Precision: {args.precision}")
    print("=" * 80)

    resume_ckpt = None

    vocab_path = os.path.join(
        args.save_dir,
        "vocab.json"
    )

    if args.resume_from is not None:
        (
            resume_ckpt,
            config,
            stoi,
            itos,
            resume_args_dict
        ) = load_training_checkpoint(
            checkpoint_path=args.resume_from,
            device=device
        )

        print("=" * 80)
        print("恢复训练 checkpoint 已加载")
        print(f"Checkpoint: {args.resume_from}")
        print(f"Checkpoint epoch: {resume_ckpt.get('epoch')}")
        print(f"Checkpoint global_step: {resume_ckpt.get('global_step')}")
        print(f"Checkpoint best_val_loss: {resume_ckpt.get('best_val_loss')}")
        print("=" * 80)

        config.use_checkpoint = not args.no_checkpoint
        config.adaptive_checkpoint = args.adaptive_checkpoint
        config.checkpoint_activation_threshold_mb = args.checkpoint_activation_threshold_mb
        config.normalize_kernel = not args.no_kernel_norm
        config.initial_scale = args.initial_scale
        config.chunk_size = args.chunk_size
        config.loss_chunk_size = args.loss_chunk_size
        config.source_conv_impl = args.source_conv_impl
        config.source_fused_mixed = not args.no_source_fused_mixed
        config.dynamic_conv_impl = args.dynamic_conv_impl
        config.dynamic_conv_tmp_mb_limit = args.dynamic_conv_tmp_mb_limit
        config.cuda_warmup_repeat = args.cuda_warmup_repeat
        config.use_nonlinear_residual = not args.no_nonlinear_residual
        config.use_source_nonlinear_residual = args.use_source_nonlinear_residual
        config.nonlinear_residual_initial_scale = args.nonlinear_residual_initial_scale
        config.nonlinear_residual_bias = not args.no_nonlinear_residual_bias
        config.use_preallocated_chunk_output = not args.no_preallocated_chunk_output

        total_chars = None

        print("=" * 80)
        print("使用 checkpoint 中的模型结构配置")
        print("应用当前命令行的 checkpoint/kernel_norm/initial_scale/chunk/cuda 参数")
        print("跳过当前文本对 checkpoint 词表的兼容性检查")
        print("不使用文件字节数作为字符数")
        print("=" * 80)

    else:
        if os.path.isfile(vocab_path):
            stoi, itos = load_vocab(args.save_dir)

            total_chars = None
            vocab_size = len(stoi)

            config = build_model_config_from_args(
                args=args,
                vocab_size=vocab_size
            )

            print("=" * 80)
            print("已使用已有 vocab.json")
            print("不重新构建词表")
            print("不流式扫描检查当前文本")
            print("不使用文件字节数作为字符数")
            print("=" * 80)

        else:
            check_text_file(args.data_path)

            stoi, itos, total_chars = build_char_vocab_streaming(
                data_path=args.data_path,
                chunk_size=args.text_chunk_size
            )

            vocab_size = len(stoi)

            config = build_model_config_from_args(
                args=args,
                vocab_size=vocab_size
            )

            save_vocab(
                args.save_dir,
                stoi,
                itos
            )

            print("=" * 80)
            print("未检测到已有 vocab.json，已完成首次词表构建并保存")
            print("=" * 80)

    vocab_size = len(stoi)

    if total_chars is None:
        print("文本字符数: 未统计，等待 memmap 阶段决定是否需要统计")
    else:
        print(f"文本字符数: {total_chars:,}")

    print(f"词表大小: {vocab_size:,}")

    memmap_path, total_tokens, dtype_name = encode_text_to_memmap(
        data_path=args.data_path,
        stoi=stoi,
        save_dir=args.save_dir,
        total_chars=total_chars,
        chunk_size=args.text_chunk_size,
        force_rebuild=args.rebuild_memmap
    )

    dataset = RandomTextChunkMemmapDataset(
        memmap_path=memmap_path,
        total_tokens=total_tokens,
        dtype_name=dtype_name,
        block_size=config.block_size,
        device=device,
        train_fraction=args.train_fraction
    )

    model = BlockMaskedCausalDynamicConvLM(
        config=config
    ).to(device)

    if resume_ckpt is not None:
        state_dict = resume_ckpt["model_state_dict"]

        if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
            print("=" * 80)
            print("检测到 checkpoint 权重包含 _orig_mod. 前缀，正在自动去除")
            print("=" * 80)

            state_dict = {
                k.replace("_orig_mod.", "", 1): v
                for k, v in state_dict.items()
            }

        model.load_state_dict(
            state_dict,
            strict=True
        )

        print("=" * 80)
        print("模型权重已从 checkpoint 恢复")
        print("=" * 80)

    warmup_model_cuda_dynamic_conv_if_needed(
        model=model,
        config=config,
        args=args,
        device=device,
        precision=args.precision,
        batch_size=args.batch_size
    )

    total_params, trainable_params = count_parameters(model)

    print("=" * 80)
    print("参数统计")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Activation checkpointing: {config.use_checkpoint}")
    print("=" * 80)

    print("模型配置:")
    print(
        json.dumps(
            asdict(config),
            ensure_ascii=False,
            indent=2
        )
    )
    print("=" * 80)

    debug_print_model_structure(model)

    save_config(
        args.save_dir,
        config,
        vars(args)
    )

    if args.debug_shapes:
        debug_check_shapes(
            model=model,
            vocab_size=vocab_size,
            device=device,
            precision=args.precision
        )

    if args.debug_causality:
        debug_check_model_causality(
            model=model,
            vocab_size=vocab_size,
            device=device,
            precision=args.precision
        )

    if args.debug_param_length:
        verify_parameter_count_independent_of_block_size(
            config=config,
            device=device
        )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        fused=(device.type == "cuda"),
    )

    if resume_ckpt is not None:
        optimizer_state_dict = resume_ckpt.get(
            "optimizer_state_dict",
            None
        )

        if optimizer_state_dict is not None:
            optimizer.load_state_dict(
                optimizer_state_dict
            )

            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device=device)

            print("=" * 80)
            print("Optimizer 状态已从 checkpoint 恢复")
            print("=" * 80)
        else:
            print("=" * 80)
            print("警告：checkpoint 中没有 optimizer_state_dict，将使用新的 optimizer 状态继续训练")
            print("=" * 80)

    use_fp16 = (
        args.precision == "fp16"
        and device.type == "cuda"
    )

    use_bf16 = (
        args.precision == "bf16"
        and device.type == "cuda"
    )

    if use_bf16:
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "当前 GPU 不支持 bf16。请改用 --precision fp16。"
            )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_fp16
    )

    if resume_ckpt is not None:
        start_epoch = int(
            resume_ckpt.get(
                "epoch",
                0
            )
        ) + 1

        global_step = int(
            resume_ckpt.get(
                "global_step",
                0
            )
        )

        best_val_loss = float(
            resume_ckpt.get(
                "best_val_loss",
                float("inf")
            )
        )
    else:
        start_epoch = 1
        global_step = 0
        best_val_loss = float("inf")

    if start_epoch > args.epochs:
        print("=" * 80)
        print(
            f"注意：checkpoint 已训练到 epoch={start_epoch - 1}，"
            f"但当前 --epochs={args.epochs}。"
        )
        print("如果想继续训练，请把 --epochs 设置得更大。")
        print("=" * 80)
        return

    best_path = os.path.join(
        args.save_dir,
        "best_model.pth"
    )

    last_path = os.path.join(
        args.save_dir,
        "last_model.pth"
    )

    model.train()

    prefetch_batches = int(
        getattr(
            args,
            "prefetch_batches",
            2
        )
    )

    train_loader = CudaPrefetchBatchLoader(
        dataset=dataset,
        split="train",
        batch_size=args.batch_size,
        device=device,
        prefetch_batches=prefetch_batches
    )

    if device.type == "cuda":
        train_loader.start()

    train_start_time = time.time()
    last_log_time = train_start_time
    last_log_step = int(global_step)

    print("=" * 80)
    print("开始训练")
    print(f"Start epoch: {start_epoch}")
    print(f"Target epochs: {args.epochs}")
    print(f"Initial global_step: {global_step}")
    print(f"Initial best_val_loss: {best_val_loss}")
    print(f"Data prefetch: {'enabled' if device.type == 'cuda' else 'disabled on CPU'}")
    print(f"Prefetch batches: {prefetch_batches if device.type == 'cuda' else 0}")
    print("=" * 80)

    autocast_enabled = (
        device.type == "cuda"
        and (
            use_fp16
            or use_bf16
        )
    )

    if use_fp16:
        autocast_dtype = torch.float16
    elif use_bf16:
        autocast_dtype = torch.bfloat16
    else:
        autocast_dtype = torch.float32

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            epoch_start_time = time.time()
            running_loss = 0.0

            print("")
            print("-" * 80)
            print(f"Epoch {epoch}/{args.epochs}")
            print("-" * 80)

            for step in range(1, args.steps_per_epoch + 1):
                global_step += 1

                x, y = train_loader.next()

                optimizer.zero_grad(
                    set_to_none=True
                )

                with torch.amp.autocast(
                    "cuda",
                    enabled=autocast_enabled,
                    dtype=autocast_dtype
                ):
                    _, loss = model(
                        input_ids=x,
                        targets=y
                    )

                if not torch.isfinite(loss):
                    print("=" * 80)
                    print("检测到非有限 loss，跳过当前 step")
                    print(f"loss = {loss.item()}")
                    print("=" * 80)

                    optimizer.zero_grad(
                        set_to_none=True
                    )

                    del x
                    del y
                    del loss

                    continue

                scaler.scale(loss).backward()

                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)

                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        args.grad_clip
                    )

                    if not torch.isfinite(grad_norm):
                        print("=" * 80)
                        print("检测到非有限梯度，跳过 optimizer.step")
                        print(f"grad_norm = {grad_norm}")
                        print("=" * 80)

                        optimizer.zero_grad(
                            set_to_none=True
                        )

                        scaler.update()

                        del x
                        del y
                        del loss

                        continue

                scaler.step(optimizer)
                scaler.update()

                optimizer.zero_grad(
                    set_to_none=True
                )

                loss_value = float(loss.item())
                running_loss += loss_value

                if global_step % args.log_interval == 0:
                    if device.type == "cuda":
                        torch.cuda.synchronize(
                            device=device
                        )

                    now_time = time.time()

                    log_elapsed = now_time - last_log_time

                    steps_since_last_log = max(
                        1,
                        int(global_step) - int(last_log_step)
                    )

                    sec_per_step = log_elapsed / float(
                        steps_since_last_log
                    )

                    total_elapsed = now_time - train_start_time

                    print(
                        f"Epoch [{epoch}/{args.epochs}] "
                        f"Step [{step}/{args.steps_per_epoch}] "
                        f"Global [{global_step}] "
                        f"Loss: {loss_value:.6f} "
                        f"EpochAvg: {running_loss / step:.6f} "
                        f"LogElapsed: {log_elapsed:.2f}s "
                        f"Sec/Step: {sec_per_step:.4f}s "
                        f"TotalElapsed: {total_elapsed:.2f}s"
                    )

                    last_log_time = now_time
                    last_log_step = int(global_step)

                if global_step % args.eval_interval == 0:
                    losses = estimate_loss(
                        model=model,
                        dataset=dataset,
                        batch_size=args.batch_size,
                        eval_iters=args.eval_iters,
                        precision=args.precision,
                        device=device
                    )

                    train_loss = losses["train"]
                    val_loss = losses["val"]

                    print("=" * 80)
                    print(
                        f"Eval global_step={global_step}: "
                        f"train_loss={train_loss:.6f}, "
                        f"val_loss={val_loss:.6f}"
                    )
                    print("=" * 80)

                    save_checkpoint(
                        path=last_path,
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        global_step=global_step,
                        best_val_loss=best_val_loss,
                        stoi=stoi,
                        itos=itos,
                        config=config,
                        args_dict=vars(args)
                    )

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss

                        save_checkpoint(
                            path=best_path,
                            model=model,
                            optimizer=optimizer,
                            epoch=epoch,
                            global_step=global_step,
                            best_val_loss=best_val_loss,
                            stoi=stoi,
                            itos=itos,
                            config=config,
                            args_dict=vars(args)
                        )

                        print(
                            f"New best saved: {best_path}, best_val_loss={best_val_loss:.6f}"
                        )

                    if args.sample_during_train:
                        try:
                            sample = generate(
                                model=model,
                                start_text=args.start_text,
                                stoi=stoi,
                                itos=itos,
                                max_new_tokens=args.sample_max_new_tokens,
                                temperature=args.temperature,
                                top_k=args.top_k,
                                device=device,
                                precision=args.precision
                            )

                            print("=" * 80)
                            print("训练中采样:")
                            print(sample)
                            print("=" * 80)

                        except Exception as e:
                            print(f"采样失败: {e}")

                    model.train()

                    if device.type == "cuda":
                        torch.cuda.synchronize(
                            device=device
                        )

                    last_log_time = time.time()
                    last_log_step = int(global_step)

                del x
                del y
                del loss

            epoch_time = time.time() - epoch_start_time

            print(
                f"Epoch {epoch} finished. "
                f"avg_loss={running_loss / args.steps_per_epoch:.6f}, "
                f"time={epoch_time:.2f}s"
            )

            save_checkpoint(
                path=last_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                best_val_loss=best_val_loss,
                stoi=stoi,
                itos=itos,
                config=config,
                args_dict=vars(args)
            )

            if device.type == "cuda":
                torch.cuda.synchronize(
                    device=device
                )

            last_log_time = time.time()
            last_log_step = int(global_step)

    finally:
        train_loader.stop()

    total_time = time.time() - train_start_time

    print("=" * 80)
    print("训练结束")
    print(f"Total time: {total_time:.2f}s")
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"Best model path: {best_path}")
    print(f"Last model path: {last_path}")
    print("=" * 80)

    if args.generate_after_train:
        sample = generate(
            model=model,
            start_text=args.start_text,
            stoi=stoi,
            itos=itos,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device,
            precision=args.precision
        )

        print(sample)

# ============================================================
# 18. 仅生成模式
# ============================================================

def generate_only(args):
    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(args.device)

    if args.checkpoint_path is None:
        raise ValueError(
            "generate_only 模式必须指定 --checkpoint_path"
        )

    model, stoi, itos, config, ckpt = load_checkpoint(
        checkpoint_path=args.checkpoint_path,
        device=device
    )

    config.dynamic_conv_impl = args.dynamic_conv_impl
    config.chunk_size = args.chunk_size
    config.cuda_warmup_repeat = args.cuda_warmup_repeat

    print("=" * 80)
    print("加载 checkpoint 完成")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Epoch: {ckpt.get('epoch')}")
    print(f"Global step: {ckpt.get('global_step')}")
    print(f"Best val loss: {ckpt.get('best_val_loss')}")
    print(f"Dynamic conv impl: {config.dynamic_conv_impl}")
    print(f"Chunk size: {config.chunk_size}")
    print("=" * 80)

    warmup_model_cuda_dynamic_conv_if_needed(
        model=model,
        config=config,
        args=args,
        device=device,
        precision=args.precision,
        batch_size=1
    )

    sample = generate(
        model=model,
        start_text=args.start_text,
        stoi=stoi,
        itos=itos,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
        precision=args.precision
    )

    print(sample)

# ============================================================
# 19. argparse
# ============================================================
def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Dilated unshared causal dynamic convolution character language model"
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="a.txt"
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="./checkpoints_char_dil_unshared_dynamic_conv_lm"
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None
    )

    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="从指定 checkpoint 继续训练，例如 ./checkpoints_char_dil_unshared_dynamic_conv_lm/last_model.pth"
    )

    parser.add_argument(
        "--no_checkpoint",
        action="store_true",
        help="关闭 activation checkpointing。默认开启。"
    )

    parser.add_argument(
        "--adaptive_checkpoint",
        action="store_true",
        default=True,
        help="启用自适应 checkpoint。默认开启。"
    )

    parser.add_argument(
        "--no_adaptive_checkpoint",
        dest="adaptive_checkpoint",
        action="store_false",
        help="关闭自适应 checkpoint。"
    )

    parser.add_argument(
        "--checkpoint_activation_threshold_mb",
        type=float,
        default=512.0,
        help="自适应 checkpoint 激活阈值，单位 MB。"
    )

    parser.add_argument(
        "--generate_only",
        action="store_true"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1000
    )

    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=1000
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1
    )

    parser.add_argument(
        "--block_size",
        type=int,
        default=17920
    )

    parser.add_argument(
        "--train_fraction",
        type=float,
        default=0.9
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.1
    )

    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto"
    )

    parser.add_argument(
        "--amp",
        action="store_true"
    )

    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=[
            "fp32",
            "fp16",
            "bf16"
        ],
        help="训练精度：fp32、fp16 或 bf16"
    )

    parser.add_argument(
        "--embed_dim",
        type=int,
        default=256
    )

    parser.add_argument(
        "--source_channels",
        type=int,
        default=16,
        help="source 膨胀卷积输出通道数 C。"
    )

    parser.add_argument(
        "--num_dynamic_layers",
        type=int,
        default=6,
        help="兼容旧脚本参数。新版层数由 block_size 自动计算，此参数不参与模型结构。"
    )

    parser.add_argument(
        "--num_kernels",
        type=int,
        default=16
    )

    parser.add_argument(
        "--dynamic_kernel_size",
        type=int,
        default=3
    )

    parser.add_argument(
        "--source_conv_kernel_size",
        type=int,
        default=3
    )

    parser.add_argument(
        "--kernel_gen_kernel_size",
        type=int,
        default=3
    )

    parser.add_argument(
        "--source_num_blocks",
        type=int,
        default=4,
        help="兼容旧脚本参数。新版请使用 --source_channels。"
    )

    parser.add_argument(
        "--mlp_ratio",
        type=float,
        default=4.0
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1
    )

    parser.add_argument(
        "--initial_scale",
        type=float,
        default=0.1
    )

    parser.add_argument(
        "--no_kernel_norm",
        action="store_true"
    )

    parser.add_argument(
        "--chunk_size",
        type=int,
        default=256,
        help="backbone 时间 chunk 大小。"
    )

    parser.add_argument(
        "--loss_chunk_size",
        type=int,
        default=2048,
        help="loss logits 分块大小。"
    )

    parser.add_argument(
        "--source_conv_impl",
        type=str,
        default="einsum",
        choices=[
            "einsum",
            "auto",
            "grouped_conv1d"
        ],
        help="source 膨胀卷积实现方式。"
    )

    parser.add_argument(
        "--no_source_fused_mixed",
        action="store_true",
        help="关闭 source conv -> GELU -> dropout -> source mix 的融合路径。默认开启。"
    )

    parser.add_argument(
        "--dynamic_conv_impl",
        type=str,
        default="cuda_fused",
        choices=[
            "cuda_fused",
            "stream_k",
            "materialized_kernel_stream_x",
            "materialized_windows",
            "auto"
        ],
        help="动态卷积实现方式。默认 cuda_fused。"
    )

    parser.add_argument(
        "--dynamic_conv_tmp_mb_limit",
        type=float,
        default=64.0,
        help="保留字段，用于未来 fallback 或调试。"
    )

    parser.add_argument(
        "--cuda_warmup_repeat",
        type=int,
        default=3,
        help="cuda_ops dynamic conv warmup repeat。"
    )

    parser.add_argument(
        "--no_cuda_warmup",
        action="store_true",
        help="关闭创建/加载模型后的 CUDA dynamic conv warmup。"
    )

    parser.add_argument(
        "--cuda_warmup_all_chunks",
        action="store_true",
        help="warmup 所有实际 chunk。默认只 warmup 代表性 chunk。"
    )

    parser.add_argument(
        "--cuda_warmup_forward_only",
        action="store_true",
        help="只 warmup forward，不 warmup backward。"
    )

    parser.add_argument(
        "--cuda_warmup_print_limit",
        type=int,
        default=32,
        help="最多打印多少条 warmup plan 信息。"
    )

    parser.add_argument(
        "--no_nonlinear_residual",
        action="store_true",
        help="关闭 token-wise 非线性残差增强。默认开启。"
    )

    parser.add_argument(
        "--use_source_nonlinear_residual",
        action="store_true",
        help="在 source 残差中启用 token-wise 非线性残差增强。默认关闭。"
    )

    parser.add_argument(
        "--nonlinear_residual_initial_scale",
        type=float,
        default=0.05,
        help="非线性残差增强项的初始缩放。"
    )

    parser.add_argument(
        "--no_nonlinear_residual_bias",
        action="store_true",
        help="关闭非线性残差线性层 bias。默认使用 bias。"
    )

    parser.add_argument(
        "--no_preallocated_chunk_output",
        action="store_true",
        help="关闭预分配 chunk 输出，改用 list + torch.cat。"
    )

    parser.add_argument(
        "--log_interval",
        type=int,
        default=50
    )

    parser.add_argument(
        "--eval_interval",
        type=int,
        default=500
    )

    parser.add_argument(
        "--eval_iters",
        type=int,
        default=100
    )

    parser.add_argument(
        "--generate_after_train",
        action="store_true"
    )

    parser.add_argument(
        "--sample_during_train",
        action="store_true"
    )

    parser.add_argument(
        "--sample_max_new_tokens",
        type=int,
        default=200
    )

    parser.add_argument(
        "--start_text",
        type=str,
        default="从前"
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=500
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=50
    )

    parser.add_argument(
        "--debug_shapes",
        action="store_true"
    )

    parser.add_argument(
        "--debug_causality",
        action="store_true"
    )

    parser.add_argument(
        "--debug_param_length",
        action="store_true"
    )

    parser.add_argument(
        "--rebuild_memmap",
        action="store_true",
        help="强制重新把文本编码为 tokens memmap。默认如果已有匹配 memmap，则复用。"
    )

    parser.add_argument(
        "--text_chunk_size",
        type=int,
        default=1024 * 1024 * 16,
        help="流式读取文本时的 chunk 大小，单位为字符。"
    )

    parser.add_argument(
        "--prefetch_batches",
        type=int,
        default=2,
        help="CUDA 数据预取队列大小。"
    )

    return parser


# ============================================================
# 20. main
# ============================================================

def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.num_kernels <= 0:
        raise ValueError("num_kernels 必须为正数。")

    if args.dynamic_kernel_size <= 0:
        raise ValueError("dynamic_kernel_size 必须为正数。")

    if args.embed_dim <= 0:
        raise ValueError("embed_dim 必须为正数。")

    if args.source_channels <= 0:
        raise ValueError("source_channels 必须为正数。")

    if args.source_conv_kernel_size <= 0:
        raise ValueError("source_conv_kernel_size 必须为正数。")

    if args.kernel_gen_kernel_size <= 0:
        raise ValueError("kernel_gen_kernel_size 必须为正数。")

    if args.chunk_size <= 0:
        raise ValueError("chunk_size 必须为正数。")

    if args.loss_chunk_size <= 0:
        raise ValueError("loss_chunk_size 必须为正数。")

    if args.block_size <= 0:
        raise ValueError("block_size 必须为正数。")

    if args.train_fraction <= 0.0 or args.train_fraction >= 1.0:
        raise ValueError("train_fraction 必须在 0 和 1 之间。")

    if args.source_conv_impl not in (
        "einsum",
        "auto",
        "grouped_conv1d"
    ):
        raise ValueError(
            f"source_conv_impl 必须为 'einsum'、'auto' 或 'grouped_conv1d'，"
            f"但得到 {args.source_conv_impl}"
        )

    if args.dynamic_conv_impl not in (
        "cuda_fused",
        "stream_k",
        "materialized_kernel_stream_x",
        "materialized_windows",
        "auto"
    ):
        raise ValueError(
            f"dynamic_conv_impl 必须为 'cuda_fused'、'stream_k'、"
            f"'materialized_kernel_stream_x'、'materialized_windows' 或 'auto'，"
            f"但得到 {args.dynamic_conv_impl}"
        )

    if args.dynamic_conv_tmp_mb_limit <= 0:
        raise ValueError("dynamic_conv_tmp_mb_limit 必须为正数。")

    if args.cuda_warmup_repeat <= 0:
        raise ValueError("cuda_warmup_repeat 必须为正数。")

    if args.cuda_warmup_print_limit <= 0:
        raise ValueError("cuda_warmup_print_limit 必须为正数。")

    if args.checkpoint_activation_threshold_mb <= 0:
        raise ValueError("checkpoint_activation_threshold_mb 必须为正数。")

    if args.nonlinear_residual_initial_scale < 0:
        raise ValueError("nonlinear_residual_initial_scale 不能为负数。")

    if args.prefetch_batches <= 0:
        raise ValueError("prefetch_batches 必须为正数。")

    if args.amp and args.precision == "fp32":
        args.precision = "fp16"

    if args.generate_only:
        generate_only(args)
    else:
        train(args)


if __name__ == "__main__":
    main()

# python -u convtransformer_dil_unshared.py \
#   --data_path /cloud/cloud-ssd1/a.txt \
#   --save_dir /cloud/cloud-ssd1/ckpt_dil_unshared \
#   --block_size 65536 \
#   --embed_dim 512 \
#   --source_channels 1024 \
#   --num_kernels 256 \
#   --dynamic_kernel_size 32 \
#   --chunk_size 32768 \
#   --loss_chunk_size 32768 \
#   --precision bf16 \
#   --log_interval 1 \
#   --eval_iters 20 \
#   --source_conv_kernel_size 16


# 

#  python convtransformer_dil_unshared.py   --data_path /mnt/d/chinesenovel/a.txt   --save_dir ./ckpt_dil_unshared   --block_size 32768   --embed_dim 256   --source_channels 6   --num_kernels 6   --dynamic_kernel_size 3   --chunk_size 16384   --loss_chunk_size 16384   --precision fp16  --log_interval 1 --source_conv_kernel_size 16 --no_checkpoint --resume_from ckpt_dil_unshared/last_model.pth  极小配置适合上古显卡，一般分为4个配置 small big large huge small单4090 big 10 5090 large 100 5090 huge 1000 5090

# #!/usr/bin/env bash
# set -u

# DATA_PATH="/cloud/cloud-ssd1/a.txt"
# SAVE_DIR="/cloud/cloud-ssd1/ckpt_dil_unshared"

# BLOCK_SIZE=262144
# EMBED_DIM=256
# SOURCE_CHANNELS=3
# NUM_KERNELS=6
# Source_Conv_Kernel_Size=6
# DYNAMIC_KERNEL_SIZE=3
# CHUNK_SIZE=65536
# LOSS_CHUNK_SIZE=65536
# PRECISION="bf16"
# LOG_INTERVAL=1

# LOG_FILE="$SAVE_DIR/train.log"

# mkdir -p "$SAVE_DIR"

# while true
# do
#   echo "==================================================" | tee -a "$LOG_FILE"
#   echo "Restart at $(date)" | tee -a "$LOG_FILE"
#   echo "==================================================" | tee -a "$LOG_FILE"

#   if [ -f "$SAVE_DIR/last_model.pth" ]; then
#     echo "Found checkpoint: $SAVE_DIR/last_model.pth" | tee -a "$LOG_FILE"
#     RESUME_ARG="--resume_from $SAVE_DIR/last_model.pth"
#   else
#     echo "No checkpoint found, start fresh." | tee -a "$LOG_FILE"
#     RESUME_ARG=""
#   fi

#   echo "Running command:" | tee -a "$LOG_FILE"
#   echo "python -u convtransformer_dil_unshared.py --data_path $DATA_PATH --save_dir $SAVE_DIR $RESUME_ARG --block_size $BLOCK_SIZE --embed_dim $EMBED_DIM --source_channels $SOURCE_CHANNELS --num_kernels $NUM_KERNELS --dynamic_kernel_size $DYNAMIC_KERNEL_SIZE --chunk_size $CHUNK_SIZE --loss_chunk_size $LOSS_CHUNK_SIZE --precision $PRECISION --log_interval $LOG_INTERVAL" | tee -a "$LOG_FILE"

#   python -u convtransformer_dil_unshared.py \
#     --data_path "$DATA_PATH" \
#     --save_dir "$SAVE_DIR" \
#     $RESUME_ARG \
#     --block_size "$BLOCK_SIZE" \
#     --embed_dim "$EMBED_DIM" \
#     --source_channels "$SOURCE_CHANNELS" \
#     --num_kernels "$NUM_KERNELS" \
#     --dynamic_kernel_size "$DYNAMIC_KERNEL_SIZE" \
#     --chunk_size "$CHUNK_SIZE" \
#     --loss_chunk_size "$LOSS_CHUNK_SIZE" \
#     --precision "$PRECISION" \
#     --log_interval "$LOG_INTERVAL" \
#     --eval_iters 20 \
#     --source_conv_kernel_size "$Source_Conv_Kernel_Size" \
#     --no_checkpoint \
#     >> "$LOG_FILE" 2>&1

#   EXIT_CODE=$?
#   echo "Process exited with code $EXIT_CODE at $(date)" | tee -a "$LOG_FILE"

#                                                                                                                                        19,0-1        Top
# 4090 26万上下文训练配置