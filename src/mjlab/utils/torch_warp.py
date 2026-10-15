"""Ordering between Torch and Warp work on Apple GPUs (Metal)."""

from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import Iterator

import torch
import warp as wp


@contextmanager
def _metal_torch_fence(device: wp.Device) -> Iterator[None]:
  torch.mps.synchronize()
  try:
    yield
  finally:
    wp.synchronize_device(device)


def torch_fence(device: wp.Device) -> AbstractContextManager[None]:
  """Order Torch and Warp work around Warp launches on ``device``.

  On Metal, Torch (MPS) and Warp encode on separate MTLCommandQueues with no
  ordering between them, so Torch's queue is drained before the block and
  Warp's after it. This relies on Warp's native Metal dispatch path, which
  wraps Torch-visible arrays zero-copy. On CUDA and CPU this is a no-op.
  """
  if getattr(device, "is_metal", False):
    return _metal_torch_fence(device)
  return nullcontext()
