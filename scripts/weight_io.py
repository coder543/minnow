"""Direct-I/O checkpoint reader with bounded staging and shared model ownership.

No safetensors.safe_open (which mmaps), no state-dict copy of the model, and no
CPU model followed by .to('cuda'). Supports the original read-only shards.
"""
import ctypes
import fcntl
import json
import os
from pathlib import Path
import struct

ALIGN = 4096
CHUNK = 8 * 1024 * 1024
RESERVE = 16 * 1024**3


def available_memory():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('MemAvailable is missing')


class ModelLease:
    def __init__(self):
        path = f'/tmp/minnow-model-{os.geteuid()}.lock'
        self.file = open(path, 'a+b')
        try:
            fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.file.close()
            raise RuntimeError(f'another minnow/reference model is resident; lock {path}') from None

    def close(self):
        self.file.close()


class WeightReader:
    def __init__(self, path):
        self.path = Path(path)
        self.fds = []
        self.tensors = {}
        index = self.path / 'model.safetensors.index.json'
        names = sorted(set(json.loads(index.read_text())['weight_map'].values())) if index.exists() else ['model.safetensors']
        for name in names:
            assert Path(name).name == name, 'invalid shard filename'
            file = self.path / name
            with file.open('rb', buffering=0) as f:
                length = struct.unpack('<Q', f.read(8))[0]
                if length > 16 * 1024**2:
                    raise ValueError('invalid safetensors header length')
                header = json.loads(f.read(length))
                os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            fd = os.open(file, os.O_RDONLY | os.O_DIRECT)
            self.fds.append(fd)
            for tensor, info in header.items():
                if tensor == '__metadata__':
                    continue
                begin, end = info['data_offsets']
                self.tensors[tensor] = (fd, 8 + length + begin, end - begin, info['dtype'], info['shape'])
        self.storage = bytearray(CHUNK + 3 * ALIGN)
        address = ctypes.addressof(ctypes.c_char.from_buffer(self.storage))
        start = (-address) % ALIGN
        self.buffer = memoryview(self.storage)[start:start + CHUNK + 2 * ALIGN]

    def close(self):
        for fd in self.fds:
            os.close(fd)
        self.fds.clear()
        self.buffer.release()

    def read(self, name, dtype, device):
        import math
        import torch
        fd, offset, size, stored_dtype, shape = self.tensors[name]
        stored_dtype = {'BF16': torch.bfloat16, 'F32': torch.float32, 'F16': torch.float16}[stored_dtype]
        element_size = torch.empty((), dtype=stored_dtype).element_size()
        assert math.prod(shape) * element_size == size
        if available_memory() < RESERVE:
            raise RuntimeError('memory headroom fell below 16 GiB; stopping reference load')
        out = torch.empty(shape, dtype=dtype, device=device)
        flat = out.view(-1)
        copied = 0
        while copied < size:
            count = min(CHUNK, size - copied)
            pos = offset + copied
            aligned = pos // ALIGN * ALIGN
            skip = pos - aligned
            requested = ((skip + count + ALIGN - 1) // ALIGN) * ALIGN
            got = os.preadv(fd, [self.buffer[:requested]], aligned)
            if got < skip + count:
                raise EOFError(f'short direct read for {name}')
            source = torch.frombuffer(self.buffer[skip:skip + count], dtype=stored_dtype)
            flat[copied // element_size:(copied + count) // element_size].copy_(source)
            if str(device).startswith('cuda'):
                torch.cuda.synchronize()
            del source
            copied += count
        return out
