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

These are experimental forks, and you have to opt in to them. mjlab's own
``pyproject.toml`` and ``uv.lock`` do not mention them, so a default install
on macOS or Linux, and the published wheel, get the same packages as upstream
mjlab.

The opt-in environment is a separate uv project in ``metal/``. It installs
mjlab editable from the repository root and overrides mjlab's pins with what
the forks need:

- ``warp-lang`` and ``mujoco-warp`` from the fork branches above;
- ``mujoco`` 3.15, which ``mujoco_warp`` metal-3.15 needs;
- ``torch`` 2.11, because Warp's Torch/MPS interop breaks on torch 2.14;
- ``mlx``, ``mlx-metal``, ``pyobjc-framework-metal`` and
  ``pyobjc-framework-cocoa``, which Warp's Metal backend needs at runtime.

Setup
-----

Warp's ``setup.py`` packages a prebuilt native library and cannot build it
from a git checkout, so uv cannot install the Warp fork from its git source.
``metal/pyproject.toml`` declares the fork's metadata statically, which keeps
``uv lock`` working. To install, build Warp locally, then install it editable
into the Metal environment:

.. code-block:: bash

   # 1. Build the Warp fork (needs the Xcode command line tools).
   git clone -b metal-integ https://github.com/prathje/warp-metal ../warp-metal
   cd ../warp-metal
   uv venv --python 3.12 .venv
   uv pip install numpy
   .venv/bin/python build_lib.py --no-cuda -j 10
   cd -

   # 2. Create the Metal environment (metal/.venv) without the Warp git
   #    source, then add the local build.
   uv sync --project metal --no-install-package warp-lang
   uv pip install --python metal/.venv --no-deps -e ../warp-metal

To work on MuJoCo Warp as well, replace the git install with a local checkout:

.. code-block:: bash

   uv pip install --python metal/.venv --no-deps -e ../mujoco_warp_metal

Run commands with ``metal/.venv/bin/python``, or with
``uv run --project metal --no-sync``. A plain ``uv run`` re-syncs and would
put the git versions back. Don't use ``uv run`` without ``--project metal``,
because that syncs the default, non-Metal environment in ``.venv``.

Running
-------

The Metal backend is opt-in at runtime too. Set ``WARP_ENABLE_METAL=1`` and
use device ``mps`` (Warp treats ``mps`` as an alias for ``metal:0``):

.. code-block:: bash

   WARP_ENABLE_METAL=1 metal/.venv/bin/train Mjlab-Cartpole-Balance \
     --gpu-ids None --env.scene.num-envs 1024

   WARP_ENABLE_METAL=1 metal/.venv/bin/play Mjlab-Cartpole-Balance \
     --agent random --device mps

With ``WARP_ENABLE_METAL=1`` and no CUDA devices, ``train`` selects ``mps``
by itself. ``--gpu-ids None`` skips CUDA GPU selection. ``play`` defaults to
``cuda:0`` or ``cpu``, so pass ``--device mps``.

What mjlab does differently on Metal
------------------------------------

All of these are gated on the Warp device being a Metal device. CUDA and CPU
code paths are unchanged.

- Graph capture: step, forward, reset and sense are captured into Metal
  indirect command buffers when ``wp.config.metal_native_dispatch`` is on
  (the default). Before every capture, mjlab runs one eager ``forward()``
  so MuJoCo Warp can build the index arrays it creates on first use.
- Torch/Warp ordering: Torch (MPS) and Warp encode on separate Metal command
  queues with no ordering between them. ``Simulation`` drains Torch's queue
  before each Warp launch and Warp's queue after it, so Torch never reads
  stale simulation state. The fence relies on Warp's native Metal dispatch
  path (``WARP_METAL_NATIVE_DISPATCH``, on by default), which shares buffers
  with Torch zero-copy.
- ``TorchArray`` only sets up CUDA streams on CUDA devices.

Code that launches Warp kernels on simulation arrays outside ``Simulation``
should wrap them in ``sim.torch_fence()`` (or
``mjlab.utils.torch_warp.torch_fence(device)``), which is a no-op on CUDA and
CPU.

Known limitations
-----------------

- MuJoCo Warp's convex collision kernel (``ccd_kernel``, used for e.g. box and
  mesh pairs) does not yet compile on Metal; models that need it fail with a
  ``MetalCodegenError``. Cartpole and G1 velocity-flat do not need it.
- Camera sensors fail on Metal (``wp.Texture2D`` is not supported yet).
- ``BuiltinDcMotorActuator`` targets the MuJoCo 3.11 ``<dcmotor>`` API and
  raises ``NotImplementedError`` on MuJoCo 3.15.
- The Torch/Warp fences add a full queue drain per launch.
