#!/usr/bin/env python3
"""Copy a checkpoint to local storage without mmap or a file-cache weight copy.

Linux O_DIRECT on both ends; one aligned 8 MiB buffer. Hash each source while
copying, then verify the destination with direct reads before publishing it.
The source is read-only and an existing destination is never overwritten.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import time

ALIGN = 4096
CHUNK = 8 * 1024**2


def aligned_buffer():
    storage = bytearray(CHUNK + ALIGN)
    start = -ctypes.addressof(ctypes.c_char.from_buffer(storage)) % ALIGN
    return memoryview(storage)[start:start + CHUNK]


def copy_file(source, destination, buffer):
    size = source.stat().st_size
    digest = hashlib.sha256()
    src = os.open(source, os.O_RDONLY | os.O_DIRECT)
    dst = None
    try:
        dst = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_DIRECT, 0o644)
        offset = 0
        while offset < size:
            count = min(CHUNK, size - offset)
            aligned = (count + ALIGN - 1) // ALIGN * ALIGN
            got = os.preadv(src, [buffer[:aligned]], offset)
            if got < count:
                raise IOError(f'short read: {source}')
            digest.update(buffer[:count])
            buffer[count:aligned] = b'\0' * (aligned - count)
            if os.pwritev(dst, [buffer[:aligned]], offset) != aligned:
                raise IOError(f'short write: {destination}')
            offset += count
        os.ftruncate(dst, size)
        os.fsync(dst)
    finally:
        os.close(src)
        if dst is not None:
            os.close(dst)
    return size, digest.hexdigest()


def verify_file(path, size, expected, buffer):
    digest = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    try:
        offset = 0
        while offset < size:
            count = min(CHUNK, size - offset)
            aligned = (count + ALIGN - 1) // ALIGN * ALIGN
            if os.preadv(fd, [buffer[:aligned]], offset) < count:
                raise IOError(f'short verification read: {path}')
            digest.update(buffer[:count])
            offset += count
    finally:
        os.close(fd)
    if digest.hexdigest() != expected or path.stat().st_size != size:
        raise IOError(f'checksum mismatch: {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    source, destination = args.source.resolve(), args.destination.absolute()
    if destination.exists():
        parser.error(f'destination already exists: {destination}')
    if not source.is_dir():
        parser.error(f'not a checkpoint directory: {source}')
    files = sorted(p for p in source.iterdir() if p.is_file())
    if not any(p.name.endswith('.safetensors') for p in files):
        parser.error('source has no safetensors shards')
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(destination.name + f'.copying-{os.getpid()}')
    staging.mkdir()
    buffer = aligned_buffer()
    manifest = {'source': str(source), 'files': {}}
    start = time.monotonic()
    for file in files:
        before = time.monotonic()
        target = staging / file.name
        size, digest = copy_file(file, target, buffer)
        verify_file(target, size, digest, buffer)
        manifest['files'][file.name] = {'bytes': size, 'sha256': digest}
        print(f'{file.name}: {size / 1024**2:.1f} MiB copied and verified in {time.monotonic() - before:.2f}s', flush=True)
    manifest['elapsed_seconds'] = time.monotonic() - start
    (staging / '.minnow-copy.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if destination.exists():
        raise FileExistsError(destination)
    staging.rename(destination)
    print(f'Checkpoint ready: {destination} ({manifest["elapsed_seconds"]:.2f}s)', flush=True)


if __name__ == '__main__':
    main()
