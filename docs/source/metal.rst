.. _metal:

Apple Silicon GPU (Metal)
=========================

On macOS, mjlab can run simulation and training on the Apple GPU through
Metal ports of Warp and MuJoCo Warp:

- Warp: `prathje/warp-metal <https://github.com/prathje/warp-metal>`_,
  branch ``metal-integ`` (Warp 1.18).
- MuJoCo Warp: `prathje/mujoco_warp_metal
  <https://github.com/prathje/mujoco_warp_metal>`_, branch ``metal-3.15``
  (MuJoCo Warp 3.15, with MuJoCo 3.15).

These are experimental forks. Linux keeps the upstream packages: ``warp-lang``
from the NVIDIA index and ``mujoco-warp`` / ``mujoco`` 3.11 from PyPI. The
macOS-only dependencies and sources in ``pyproject.toml`` all carry a
``sys_platform == 'darwin'`` marker, so the Linux resolution in ``uv.lock`` is
the same as upstream mjlab.

macOS also pins ``torch`` to 2.11, because Warp's MPS interop breaks on torch
2.14, and adds ``mlx``, ``mlx-metal``, ``pyobjc-framework-metal`` and
``pyobjc-framework-cocoa``, which Warp's Metal backend needs at runtime.

Setup
-----

Warp's ``setup.py`` packages a prebuilt native library and cannot build it
from a git checkout, so ``uv sync`` cannot install the Warp fork from its git
source. ``pyproject.toml`` declares the fork's metadata statically
(``[[tool.uv.dependency-metadata]]``), which keeps ``uv lock`` working on every
platform. To install, build Warp locally and install it editable:

.. code-block:: bash

   # 1. Build the Warp fork (needs Xcode command line tools).
   git clone -b metal-integ https://github.com/prathje/warp-metal ../warp-metal
   cd ../warp-metal
   uv venv --python 3.12 .venv
   uv pip install numpy
   .venv/bin/python build_lib.py --no-cuda -j 10
   cd -

   # 2. Install mjlab without the Warp git source, then add the local build.
   uv sync --no-install-package warp-lang
   uv pip install --no-deps -e ../warp-metal

To work on MuJoCo Warp as well, replace the git install with a local checkout:

.. code-block:: bash

   uv pip install --no-deps -e ../mujoco_warp_metal

``uv run`` re-syncs the environment and would put the git versions back. Use
``uv run --no-sync`` (or ``.venv/bin/python``) after an editable override.

Running
-------

The Metal backend is opt-in. Set ``WARP_ENABLE_METAL=1`` and pass
``device="mps"`` (Warp treats ``mps`` as an alias for ``metal:0``):

.. code-block:: bash

   WARP_ENABLE_METAL=1 uv run --no-sync train Mjlab-Cartpole-Balance \
     --gpu-ids None --env.scene.num-envs 1024

With ``WARP_ENABLE_METAL=1`` and no CUDA devices, ``train`` selects ``mps``.
``--gpu-ids None`` skips CUDA GPU selection.

What mjlab does differently on Metal
------------------------------------

All of these are gated on the Warp device being a Metal device. CUDA and CPU
code paths are unchanged.

- Graph capture: step, forward, reset and sense are captured into Metal
  indirect command buffers when ``wp.config.metal_native_dispatch`` is on
  (the default). Before the first capture, mjlab runs one eager ``forward()``
  so MuJoCo Warp can build the index arrays it creates on first use.
- Torch/Warp ordering: Torch (MPS) and Warp encode on separate Metal command
  queues with no ordering between them. ``Simulation`` drains Torch's queue
  before each Warp launch and Warp's queue after it, so Torch never reads
  stale simulation state.
- ``TorchArray`` only sets up CUDA streams on CUDA devices.
