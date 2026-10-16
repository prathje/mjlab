"""Ordering between Torch and Warp work on Apple GPUs (Metal)."""

from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import Callable, Iterator

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

  On Metal, Warp releases with ``wp.metal_torch_batch`` order Warp launches on
  arrays shared with Torch against Torch's MPS work on the GPU, through Torch's
  command queue, without blocking the CPU. Inside the returned context those
  launches stay batched and are ordered once on exit, so Torch must not access
  simulation arrays inside it. Older Warp releases encode on a separate queue
  with no ordering against Torch; there, Torch's queue is drained before the
  block and Warp's after it. Either way this relies on Warp's native Metal
  dispatch path, which wraps Torch-visible arrays zero-copy. On CUDA and CPU
  this is a no-op.
  """
  if not getattr(device, "is_metal", False):
    return nullcontext()
  # Added to Warp's Metal fork after upstream 1.14: look it up dynamically.
  batch: Callable[[wp.Device], AbstractContextManager[None]] | None = getattr(
    wp, "metal_torch_batch", None
  )
  if batch is not None:
    return batch(device)
  return _metal_torch_fence(device)
