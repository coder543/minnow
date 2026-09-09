"""Independent .mnw decoding for the one-layer-at-a-time Python oracle.

Packed tensors are decoded one projection at a time. NVFP4 projections use
FP32 arithmetic on decoded E2M1/E4M3 operands, then apply the outer scales.
This module is never used by the Rust runtime.
"""
import ctypes
import json
import math
import os
from pathlib import Path
import struct
import types

import msgpack
import numpy as np
import torch

from weight_io import ALIGN, CHUNK, RESERVE, WeightReader, available_memory

FP4 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
E4M3 = np.array([
    (i & 7) / 512 if i < 8 else (1 + (i & 7) / 8) * 2.0 ** ((i >> 3) - 7)
    for i in range(127)
], dtype=np.float32)


def nearest(values, table):
    """Round to the nearest code, breaking ties toward an even code."""
    high = np.searchsorted(table, values).clip(0, len(table) - 1)
    low = (high - 1).clip(0)
    a, b = abs(values - table[low]), abs(table[high] - values)
    return np.where((a < b) | ((a == b) & (low % 2 == 0)), low, high)


def quantize_activation(x):
    values = x.float().cpu().numpy()
    maximum = abs(values).max(axis=-1, keepdims=True)
    global_scale = np.where(maximum == 0, np.float32(1),
                            np.maximum(maximum / np.float32(2688), np.finfo(np.float32).tiny))
    blocks = values.reshape(*values.shape[:-1], -1, 16)
    maximum = abs(blocks).max(axis=-1)
    scales = nearest((maximum / np.float32(6)) / global_scale, E4M3).clip(1)
    scales = np.where(maximum == 0, 56, scales)
    divisor = E4M3[scales][..., None] * global_scale[..., None]
    codes = nearest(abs(blocks / divisor), FP4)
    decoded = np.copysign(FP4[codes], blocks) * E4M3[scales][..., None]
    return (torch.from_numpy(decoded.reshape(values.shape)).to(x.device),
            torch.from_numpy(global_scale).to(x.device))


def nvfp4_linear(self, x):
    operands, outer = quantize_activation(x)
    # FP32 GEMM avoids rounding the accumulator before the outer scales.
    y = torch.nn.functional.linear(operands, self.weight.float())
    return (y * (outer * self.nvfp4_global)).to(x.dtype)


def integer_linear(self, x):
    shape, group = x.shape, self.integer_group
    values = x.float().cpu().numpy().reshape(-1, shape[-1] // group, group)
    maximum = abs(values).max(axis=-1, keepdims=True)
    scales = np.where(maximum == 0, np.float32(1),
                      np.maximum(maximum / np.float32(127), np.finfo(np.float32).tiny))
    codes = torch.from_numpy(np.rint(values / scales).clip(-127, 127)).to(x.device)
    scales = torch.from_numpy(scales[..., 0]).to(x.device)
    output = torch.zeros((codes.shape[0], self.weight.shape[0]), device=x.device)
    # Float32 exactly represents each integer group dot (at most 128*127*127).
    for k in range(codes.shape[1]):
        dot = torch.nn.functional.linear(codes[:, k], self.weight[:, k*group:(k+1)*group].float())
        output += dot * (scales[:, k, None] * self.integer_scales[:, k])
    return output.reshape(*shape[:-1], self.weight.shape[0]).to(x.dtype)


class ContainerReader(WeightReader):
    def __init__(self, path, int8_activations=False):
        self.int8_activations = int8_activations
        self.path = Path(path)
        with self.path.open('rb', buffering=0) as f:
            header = f.read(64)
            if len(header) != 64 or header[:8] != b'MINNOW01':
                raise ValueError('invalid .mnw header')
            offset, length, size = struct.unpack_from('<QQQ', header, 8)
            if (size != self.path.stat().st_size or offset < ALIGN or offset % ALIGN
                    or not 0 < length <= 128 * 1024**2 or offset + length != size):
                raise ValueError('invalid .mnw manifest bounds')
            f.seek(offset)
            self.manifest = msgpack.unpackb(f.read(length), raw=False)
            os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        if self.manifest['version'] != 1 or self.manifest['architecture'] != 'llada2_moe':
            raise ValueError('unsupported .mnw architecture/version')
        self.config = json.loads(self.manifest['assets']['config.json'])
        self.info = self.manifest['tensors']
        self.tensors = {}
        regions = []
        for name, info in self.info.items():
            shape, encoding = info['shape'], info['encoding']
            if not shape or len(shape) > 8 or any(n <= 0 for n in shape):
                raise ValueError(f'invalid tensor shape: {name}')
            count = math.prod(shape)
            item_size = {'bf16': 2, 'f16': 2, 'f32': 4, 'i8_sym': 1, 'i8_mma': 1,
                         'i4_sym': .5, 'i4_mma': .5, 'nvfp4': .5}[encoding]
            if info['data']['bytes'] != count * item_size:
                raise ValueError(f'invalid tensor size: {name}')
            for region in [info['data']] + ([info['scales']] if 'scales' in info else []):
                start, size = region['offset'], region['bytes']
                if start < ALIGN or start % ALIGN or size <= 0 or start + size > offset:
                    raise ValueError(f'invalid tensor region: {name}')
                regions.append((start, start + size))
        regions.sort()
        if any(a[1] > b[0] for a, b in zip(regions, regions[1:])):
            raise ValueError('overlapping .mnw regions')
        self.fds = [os.open(self.path, os.O_RDONLY | os.O_DIRECT)]
        for name, info in self.info.items():
            if info['encoding'] in ['bf16', 'f16', 'f32']:
                self.tensors[name] = (self.fds[0], info['data']['offset'], info['data']['bytes'],
                                      info['encoding'].upper(), info['shape'])
        self.storage = bytearray(CHUNK + 3 * ALIGN)
        address = ctypes.addressof(ctypes.c_char.from_buffer(self.storage))
        start = (-address) % ALIGN
        self.buffer = memoryview(self.storage)[start:start + CHUNK + 2 * ALIGN]

    def region(self, region):
        size, offset = region['bytes'], region['offset']
        # Only individual routed projections enter this path; large floating
        # tensors use the inherited streaming read into their final allocation.
        if size > 64 * 1024**2:
            raise ValueError('quantized reference projection exceeds 64 MiB')
        result = bytearray(size)
        for copied in range(0, size, CHUNK):
            count = min(CHUNK, size - copied)
            pos = offset + copied
            aligned = pos // ALIGN * ALIGN
            skip = pos - aligned
            requested = ((skip + count + ALIGN - 1) // ALIGN) * ALIGN
            got = os.preadv(self.fds[0], [self.buffer[:requested]], aligned)
            if got < skip + count:
                raise EOFError('short direct read')
            result[copied:copied + count] = self.buffer[skip:skip + count]
        return result

    def read(self, name, dtype, device):
        if name in self.tensors:
            return super().read(name, dtype, device)
        if available_memory() < RESERVE:
            raise RuntimeError('reference memory reserve reached')
        info = self.info[name]
        out, k = info['shape']
        encoding, group = info['encoding'], info['group_size']
        if (out % 8 or not group or k % group or
                (encoding == 'nvfp4' and (group != 16 or k % 64))):
            raise ValueError('invalid quantized projection shape')
        expected_scales = out * k // group * (1 if encoding == 'nvfp4' else 2)
        if info['scales']['bytes'] != expected_scales:
            raise ValueError('invalid quantized scale size')
        codes = np.frombuffer(self.region(info['data']), dtype=np.uint8)
        scales = np.frombuffer(self.region(info['scales']),
                               dtype=np.uint8 if encoding == 'nvfp4' else '<f2')
        if encoding == 'nvfp4':
            if (scales >= 127).any() or not 0 < info['global_scale'] < float('inf'):
                raise ValueError('invalid NVFP4 scales')
            codes = np.stack((codes & 15, codes >> 4), axis=-1)
            codes = codes.reshape(out // 8, k // 64, 2, 8, 4, 4, 2)
            codes = codes.transpose(0, 3, 1, 2, 4, 5, 6).reshape(out, k)
            scales = scales.reshape(out // 8, k // 64, 8, 4).transpose(0, 2, 1, 3)
            scales = E4M3[scales.reshape(out, k // 16)]
            values = FP4[codes & 7] * np.where(codes & 8, np.float32(-1), np.float32(1))
        else:
            if encoding.startswith('i4_'):
                codes = np.stack((codes & 15, codes >> 4), axis=-1).astype(np.int8)
                codes = (codes ^ 8) - 8
            else:
                codes = codes.view(np.int8)
            if encoding.endswith('_mma'):
                codes = codes.reshape(out // 8, k // 16, 8, 4, 2, 2)
                codes = codes.transpose(0, 2, 1, 4, 3, 5).reshape(out, k)
                scales = scales.reshape(out // 8, k // group, 8).transpose(0, 2, 1)
            values = codes.reshape(out, k).astype(np.float32)
            scales = scales.reshape(out, k // group).astype(np.float32)
            if not np.isfinite(scales).all() or (scales <= 0).any():
                raise ValueError('invalid integer scales')
            if self.int8_activations:
                # One pending projection only. Attach its scales directly when
                # the caller installs this parameter in the current layer.
                self.pending_scales = (name, torch.from_numpy(scales).to(device))
                return torch.from_numpy(values).to(device=device, dtype=dtype)
        values = (values.reshape(out, -1, group) * scales[..., None]).reshape(out, k)
        # NVFP4 retains its exact block operands; global scales are applied by
        # the projection forward, just as specified by block-scaled arithmetic.
        return torch.from_numpy(values).to(device=device, dtype=dtype)

    def configure_projection(self, name, owner):
        info = self.info[name]
        if info['encoding'] == 'nvfp4':
            if not isinstance(owner, torch.nn.Linear):
                raise ValueError('NVFP4 requires a linear projection')
            owner.nvfp4_global = info['global_scale']
            owner.forward = types.MethodType(nvfp4_linear, owner)
        elif self.int8_activations and info['encoding'].startswith(('i4_', 'i8_')):
            if not isinstance(owner, torch.nn.Linear) or self.pending_scales[0] != name:
                raise ValueError('integer reference projection assignment mismatch')
            owner.integer_group = info['group_size']
            owner.integer_scales = self.pending_scales[1]
            del self.pending_scales
            owner.forward = types.MethodType(integer_linear, owner)
