# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""ONNX → TFLite conversion using the ``onnx2tf`` library.

``onnx2tf`` (PINTO0309) converts an ONNX graph to TFLite.  Version 2.0+
uses a fast ``flatbuffer_direct`` backend; earlier 1.x releases go through
the TensorFlow ``TFLiteConverter``.

The converter uses the ``onnx2tf`` Python API directly (rather than
shelling out to the CLI) so that we can:

* Apply a compatibility shim for older ``onnx2tf`` releases that call
  :func:`numpy.load` on pickled data without ``allow_pickle=True``.
* Redirect ``onnx2tf``'s built-in ``download_test_image_data()`` to use
  locally-prepared calibration data instead of downloading from GitHub
  (which can fail in many environments).

``onnx2tf`` uses ``download_test_image_data()`` in two contexts:

1. **Output validation** — always runs to compare ONNX-vs-TF outputs.
2. **INT8 calibration** — when ``output_integer_quantized_tflite=True``,
   uses the same function as a representative dataset source.

Both calls are redirected to local data via the
``_patch_validation_download()`` context manager.  This avoids the
network dependency and lets the caller supply proper calibration images
for INT8 quantization.

INT8 quantization
-----------------
When ``quantization="int8"`` the caller **should** supply representative
calibration images via *calibration_data*.  Accepted formats:

* A **directory path** containing JPEG/PNG images — the converter
  automatically loads, resizes, and converts them to the correct format.
  This is the easiest approach: just point to your dataset folder.
* A ``.npy`` file path — shape ``(N, H, W, 3)``, dtype ``float32``,
  values in ``[0, 1]``.
* A :class:`numpy.ndarray` with the same constraints.

Pixel values must be in ``[0, 1]`` (divided by 255 but **not**
ImageNet-normalized — the converter normalises to ImageNet statistics
internally before passing the data to ``onnx2tf``).

If no calibration data is provided, random noise is used instead and a
warning is emitted.  This is sufficient for ``fp32`` / ``fp16`` conversion
but will produce **poor accuracy** for ``int8``.

Note:
    The resulting ``.tflite`` model expects the same input normalization as
    the ONNX model: ImageNet mean/std (``mean=[0.485, 0.456, 0.406]``,
    ``std=[0.229, 0.224, 0.225]``).  The caller is responsible for applying
    this normalization at inference time.
"""

from __future__ import annotations

import contextlib
import inspect
import os
import sys
from itertools import cycle
from pathlib import Path
from typing import Any, Generator, cast

import numpy as np
from numpy.typing import NDArray

from rfdetr.utilities.logger import get_logger

logger = get_logger()

# Supported quantization modes.
_VALID_QUANTIZATIONS: set[str | None] = {None, "fp32", "fp16", "int8"}

# Number of random calibration samples generated when none are provided.
_DEFAULT_CALIB_SAMPLES: int = 20

# Supported image file extensions for directory-based calibration.
_IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})

# Default number of images to sample from a directory for calibration.
_DEFAULT_DIR_CALIB_SAMPLES: int = 100

# RGB ImageNet statistics — cycled for models with != 3 channels.
_IMAGENET_MEAN_RGB: tuple[float, float, float] = (0.485, 0.456, 0.406)
_IMAGENET_STD_RGB: tuple[float, float, float] = (0.229, 0.224, 0.225)

# Keep the legacy 3-channel arrays for any internal callers that reference them.
_IMAGENET_MEAN: NDArray[np.float32] = np.array(_IMAGENET_MEAN_RGB, dtype=np.float32)
_IMAGENET_STD: NDArray[np.float32] = np.array(_IMAGENET_STD_RGB, dtype=np.float32)


# Detect onnx2tf's GridSample replacement kwarg at import time so it can be
# forwarded to convert().  Kwarg name has drifted across onnx2tf versions; we
# match any parameter containing both "grid" and "pseudo" (case-insensitive).
# Workaround for onnx2tf#274 — onnx2tf's GridSample lowering produces values
# that diverge from ONNX while onnx2tf's own elementwise-close validator
# silently passes.  RF-DETR's deformable cross-attention uses F.grid_sample
# once per decoder layer; without this replacement, top-1 detection scores
# collapse from ~0.6 to ~0.02.
def _detect_gridsample_kwarg() -> str | None:
    """Return the onnx2tf GridSample replacement kwarg name, or None if absent."""
    try:
        from onnx2tf import convert as _c

        return next(
            (name for name in inspect.signature(_c).parameters if "grid" in name.lower() and "pseudo" in name.lower()),
            None,
        )
    except ImportError:
        return None


_GRIDSAMPLE_KWARG: str | None = _detect_gridsample_kwarg()


def _imagenet_stats_for_channels(
    num_channels: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Return per-channel (mean, std) arrays cycled from the RGB ImageNet stats.

    Mirrors ``RFDETR.__init__`` (``detr.py:226-230``) so calibration data is
    normalised with exactly the statistics the trained model expects.  For a
    3-channel model the result is the standard RGB ImageNet mean/std.  For a
    1-channel grayscale model only the red-channel value is used.  For models
    with more than 3 channels the RGB triplet is repeated cyclically.

    Args:
        num_channels: Number of input channels ``C`` in the NHWC calibration
            array.  Must be a positive integer.

    Returns:
        A tuple ``(mean, std)`` where each element is a float32 NumPy array
        of length *num_channels*.

    Examples:
        >>> mean, std = _imagenet_stats_for_channels(1)
        >>> float(mean[0])
        0.485...
        >>> mean, std = _imagenet_stats_for_channels(3)
        >>> mean.tolist()
        [0.485..., 0.456..., 0.406...]
    """
    mean = np.array(
        [m for _, m in zip(range(num_channels), cycle(_IMAGENET_MEAN_RGB))],
        dtype=np.float32,
    )
    std = np.array(
        [s for _, s in zip(range(num_channels), cycle(_IMAGENET_STD_RGB))],
        dtype=np.float32,
    )
    return mean, std


def _imagenet_normalize(calib: NDArray[np.float32]) -> NDArray[np.float32]:
    """Normalise NHWC calibration data with channel-matched ImageNet stats.

    The number of channels is inferred from ``calib.shape[-1]`` and the
    ImageNet RGB statistics are cycled to match it.  This correctly handles
    1-channel (grayscale) models, standard 3-channel RGB models, and any
    other channel count.

    Args:
        calib: Float32 array of shape ``(N, H, W, C)`` with values in
            ``[0, 1]``.

    Returns:
        Float32 array of the same shape with ImageNet-normalised values.

    Examples:
        >>> import numpy as np
        >>> arr = np.zeros((2, 8, 8, 1), dtype=np.float32)
        >>> out = _imagenet_normalize(arr)
        >>> out.shape
        (2, 8, 8, 1)
    """
    num_channels = calib.shape[-1]
    mean, std = _imagenet_stats_for_channels(num_channels)
    return (calib - mean) / std


def _fold_constant_expands(model: Any) -> Any:
    """Constant-fold ``Expand`` ops whose inputs are all compile-time constants.

    ``onnx2tf`` 1.26+ fails with ``ValueError: Output tensors of a Functional
    model must be the output of a TensorFlow Layer`` when an ``Expand`` op's
    input is a raw constant tensor (e.g. a position-embedding index grid
    ``[[0][1]...[31]]``) rather than a ``Layer`` output.

    This function targets only ``Expand`` nodes where every input can be
    resolved to a constant value at graph-build time.  Dynamic ``Expand``
    nodes (whose shape input depends on model inputs) are left untouched,
    avoiding the ``Tile`` / ``Concat`` rank-mismatch regressions introduced
    by running full onnxsim simplification.

    Args:
        model: An ONNX ``ModelProto`` to patch in place.

    Returns:
        The same ``ModelProto`` with all-constant ``Expand`` nodes replaced by
        initializer tensors.
    """
    import numpy as np
    import onnx
    from onnx import numpy_helper

    # Build a map from tensor name → constant numpy value.
    const_values: dict[str, Any] = {}
    for init in model.graph.initializer:
        const_values[init.name] = numpy_helper.to_array(init)
    for node in model.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    const_values[node.output[0]] = numpy_helper.to_array(attr.t)

    nodes_to_remove: list[Any] = []
    initializers_to_add: list[Any] = []
    folded = 0

    for node in model.graph.node:
        if node.op_type != "Expand":
            continue
        if not all(inp in const_values for inp in node.input):
            continue  # at least one dynamic input — leave untouched

        data = const_values[node.input[0]]
        shape = const_values[node.input[1]].astype(int).tolist()
        try:
            folded_value = np.broadcast_to(data, shape).copy()
        except ValueError:
            continue  # broadcast not possible — leave untouched

        init_tensor = numpy_helper.from_array(folded_value.astype(data.dtype), name=node.output[0])
        initializers_to_add.append(init_tensor)
        const_values[node.output[0]] = folded_value
        nodes_to_remove.append(node)
        folded += 1

    for node in nodes_to_remove:
        model.graph.node.remove(node)
    for init in initializers_to_add:
        model.graph.initializer.append(init)

    if folded:
        logger.debug(f"Constant-folded {folded} all-constant Expand node(s) for onnx2tf compatibility")

    try:
        onnx.checker.check_model(model)
    except Exception as exc:
        logger.warning(f"ONNX check failed after Expand constant-folding: {exc}")

    return model


def _replace_gelu_erf_with_tanh_approx(model: Any) -> Any:
    """Replace Erf-based GELU nodes with a tanh-GELU approximation.

    ``onnx2tf`` 1.x converts ONNX ``Erf`` nodes to ``FlexErf`` (a Select TF
    op).  ``FlexErf`` requires the Flex delegate, which is absent from the
    standard ``tflite_runtime`` wheel.  Additionally, some ``onnx2tf``
    versions auto-detect the GELU pattern and apply a substitution that
    produces incorrect activations.  Both issues are eliminated by replacing
    the 12 Erf-based GELU subgraphs in the DINOv2 backbone before ``onnx2tf``
    ever sees them.

    The Erf-GELU pattern is:

    .. code-block:: text

        x ──► Div(x, √2) ──► Erf ──► Add(erf_out, 1) ──► Mul(x, add_out)
                                                           ──► Mul(mul_out, 0.5) ──► output

    It is replaced with the tanh-GELU approximation
    ``x * 0.5 * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))``:

    .. code-block:: text

        x ──► Mul(x, x) ──► Mul(x², x) ──► Mul(x³, 0.044715)
          ──► Add(x, scaled_x3) ──► Mul(sum, √(2/π)) ──► Tanh
          ──► Add(tanh_out, 1) ──► Mul(x, add_out) ──► Mul(mul_out, 0.5) ──► output

    ``Tanh`` is a built-in TFLite op available in all runtimes without any
    Flex delegate.  The maximum absolute error vs. exact GELU is ~0.0002,
    well within model tolerance (the original Erf-GELU used by PyTorch has
    no approximation error at float32 precision, but the tanh approximation
    difference is smaller than float16 quantization noise).

    Shared scalar initializers (``tanh_gelu_coeff``, ``tanh_gelu_scale``,
    ``tanh_gelu_one``, ``tanh_gelu_half``) are added once and reused by all
    replaced subgraphs.

    Args:
        model: An ONNX ``ModelProto`` to patch in place.

    Returns:
        The same ``ModelProto`` with all Erf-based GELU subgraphs replaced.
    """
    import math

    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    # Shared scalar initializers — added once, reused by all 12 GELU replacements.
    tanh_gelu_coeff = np.float32(0.044715)
    tanh_gelu_scale = np.float32(math.sqrt(2.0 / math.pi))  # ≈ 0.7978845608
    tanh_gelu_one = np.float32(1.0)
    tanh_gelu_half = np.float32(0.5)
    shared_inits = {
        "__tanh_gelu_coeff__": tanh_gelu_coeff,
        "__tanh_gelu_scale__": tanh_gelu_scale,
        "__tanh_gelu_one__": tanh_gelu_one,
        "__tanh_gelu_half__": tanh_gelu_half,
    }
    existing_init_names = {init.name for init in model.graph.initializer}
    for name, val in shared_inits.items():
        if name not in existing_init_names:
            model.graph.initializer.append(numpy_helper.from_array(np.array(val, dtype=np.float32), name=name))

    # ---- Scan for the 5-node Erf-GELU pattern --------------------------------
    # Pattern (in order):
    #   Div(x, sqrt2_const)        → erf_input
    #   Erf(erf_input)             → erf_out
    #   Add(erf_out, 1.0_const)    → add_out
    #   Mul(x, add_out)            → mul1_out
    #   Mul(mul1_out, 0.5_const)   → gelu_out
    #
    # We identify patterns by walking from every Erf node backward/forward.

    # Map: output_name → node
    output_to_node: dict[str, Any] = {}
    for node in model.graph.node:
        for out in node.output:
            output_to_node[out] = node

    # Map: input_name → list of nodes that consume it
    input_to_nodes: dict[str, list[Any]] = {}
    for node in model.graph.node:
        for inp in node.input:
            input_to_nodes.setdefault(inp, []).append(node)

    init_values: dict[str, np.ndarray] = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
    # torch.onnx exports small scalars as Constant nodes, not initializers.
    # Include their outputs so _is_scalar_close matches them correctly.
    for _cn in model.graph.node:
        if _cn.op_type == "Constant":
            for _attr in _cn.attribute:
                if _attr.name == "value":
                    init_values[_cn.output[0]] = numpy_helper.to_array(_attr.t)

    def _is_scalar_close(name: str, value: float) -> bool:
        """Return True if tensor *name* is a scalar initializer or Constant node output ≈ *value*."""
        if name not in init_values:
            return False
        arr = init_values[name]
        return arr.size == 1 and abs(float(arr.flat[0]) - value) < 1e-4

    nodes_to_remove: list[Any] = []
    new_nodes: list[Any] = []
    replacements = 0

    for node in list(model.graph.node):
        if node.op_type != "Erf":
            continue

        erf_input = node.input[0]
        erf_out = node.output[0]

        # Erf input must come from a Div node
        div_node = output_to_node.get(erf_input)
        if div_node is None or div_node.op_type != "Div":
            continue

        # Div: inputs are (x, sqrt2); sqrt2 ≈ 1.4142
        div_inputs = list(div_node.input)
        if len(div_inputs) != 2 or not _is_scalar_close(div_inputs[1], math.sqrt(2.0)):
            continue
        gelu_input = div_inputs[0]  # the original x fed to GELU

        # Add(erf_out, 1.0) must consume erf_out
        add_consumers = [n for n in input_to_nodes.get(erf_out, []) if n.op_type == "Add"]
        if len(add_consumers) != 1:
            continue
        add_node = add_consumers[0]
        add_inputs = list(add_node.input)
        # one input is erf_out, other must be scalar 1.0
        one_input = next((inp for inp in add_inputs if inp != erf_out), None)
        if one_input is None or not _is_scalar_close(one_input, 1.0):
            continue
        add_out = add_node.output[0]

        # Mul(x, add_out): one input is gelu_input, other is add_out
        mul1_consumers = [n for n in input_to_nodes.get(add_out, []) if n.op_type == "Mul"]
        if len(mul1_consumers) != 1:
            continue
        mul1_node = mul1_consumers[0]
        mul1_inputs = list(mul1_node.input)
        if gelu_input not in mul1_inputs:
            continue
        mul1_out = mul1_node.output[0]

        # Mul(mul1_out, 0.5): one input is mul1_out
        mul2_consumers = [n for n in input_to_nodes.get(mul1_out, []) if n.op_type == "Mul"]
        if len(mul2_consumers) != 1:
            continue
        mul2_node = mul2_consumers[0]
        mul2_inputs = list(mul2_node.input)
        half_input = next((inp for inp in mul2_inputs if inp != mul1_out), None)
        if half_input is None or not _is_scalar_close(half_input, 0.5):
            continue
        gelu_out = mul2_node.output[0]  # final output of the GELU subgraph

        # ---- Pattern matched — build tanh-GELU replacement ------------------
        uid = str(replacements)
        x2_name = f"__tanh_gelu_x2_{uid}__"
        x3_name = f"__tanh_gelu_x3_{uid}__"
        scaled_x3_name = f"__tanh_gelu_scaled_x3_{uid}__"
        inner_name = f"__tanh_gelu_inner_{uid}__"
        scaled_name = f"__tanh_gelu_scaled_{uid}__"
        tanh_name = f"__tanh_gelu_tanh_{uid}__"
        tanh_p1_name = f"__tanh_gelu_tanh_p1_{uid}__"
        x_tanh_p1_name = f"__tanh_gelu_x_tanh_p1_{uid}__"
        # final output tensor reuses gelu_out so downstream graph is unchanged

        new_nodes.extend(
            [
                helper.make_node("Mul", [gelu_input, gelu_input], [x2_name]),
                helper.make_node("Mul", [x2_name, gelu_input], [x3_name]),
                helper.make_node("Mul", [x3_name, "__tanh_gelu_coeff__"], [scaled_x3_name]),
                helper.make_node("Add", [gelu_input, scaled_x3_name], [inner_name]),
                helper.make_node("Mul", [inner_name, "__tanh_gelu_scale__"], [scaled_name]),
                helper.make_node("Tanh", [scaled_name], [tanh_name]),
                helper.make_node("Add", [tanh_name, "__tanh_gelu_one__"], [tanh_p1_name]),
                helper.make_node("Mul", [gelu_input, tanh_p1_name], [x_tanh_p1_name]),
                helper.make_node("Mul", [x_tanh_p1_name, "__tanh_gelu_half__"], [gelu_out]),
            ]
        )

        nodes_to_remove.extend([div_node, node, add_node, mul1_node, mul2_node])
        replacements += 1

    for node in nodes_to_remove:
        try:
            model.graph.node.remove(node)
        except ValueError:
            pass  # already removed (shared node across two patterns — unlikely but safe)

    model.graph.node.extend(new_nodes)

    # Topologically sort all nodes (Kahn's algorithm) so that onnx2tf —
    # which assumes topological order — can process the patched graph.
    # Initializer names and graph input names are "already defined" at t=0.
    defined: set[str] = set()
    defined.update(init.name for init in model.graph.initializer)
    defined.update(inp.name for inp in model.graph.input)
    # Constant nodes (no inputs) are also implicitly defined.
    remaining = list(model.graph.node)
    sorted_nodes: list[Any] = []
    max_passes = len(remaining) + 1
    for _ in range(max_passes):
        if not remaining:
            break
        ready = [n for n in remaining if all(i == "" or i in defined for i in n.input)]
        if not ready:
            # Cycle or truly unresolvable; keep remaining order as-is.
            sorted_nodes.extend(remaining)
            remaining = []
            break
        for n in ready:
            sorted_nodes.append(n)
            defined.update(n.output)
            remaining.remove(n)
    del model.graph.node[:]
    model.graph.node.extend(sorted_nodes)

    logger.debug(
        "tanh-GELU replacement: %d Erf-GELU subgraph(s) replaced; %d Erf node(s) remaining",
        replacements,
        sum(1 for n in model.graph.node if n.op_type == "Erf"),
    )

    if replacements > 0:
        try:
            onnx.checker.check_model(model)
        except Exception as exc:
            logger.warning("ONNX check failed after tanh-GELU replacement: %s", exc)

    return model


def _patch_onnx_for_tflite(onnx_path: Path, output_dir: Path) -> Path:
    """Apply targeted ONNX graph patches to work around ``onnx2tf`` 1.x bugs.

    Known bugs in ``onnx2tf`` 1.x affecting RF-DETR's DINOv2 backbone:

    * **Constant Expand failure** — onnx2tf cannot lower an ``Expand`` node
      whose input is a raw constant tensor (not a ``Layer`` output) into a
      ``tf_keras.Functional`` model.  Fix: constant-fold all ``Expand`` nodes
      with fully-constant inputs, replacing them with initializers.

    * **Erf/FlexErf** — onnx2tf converts ``Erf`` nodes (used in DINOv2's
      GELU activations) to ``FlexErf`` Select TF ops.  ``FlexErf`` requires
      the Flex delegate, absent from ``tflite_runtime``; additionally some
      onnx2tf versions auto-detect the GELU pattern and apply an incorrect
      substitution.  Both cause corrupted logits (all negative, scores ≈ 0.01).
      Fix: replace all 12 Erf-GELU subgraphs with tanh-GELU approximation
      before conversion.

    This is a surgical graph-level patch.  Unlike running ``onnxsim``
    (full constant propagation), it does not alter the backbone embedding
    ``Tile`` or ``Concat`` ops, avoiding the rank-mismatch failures that
    full simplification introduces.

    The patched model is saved to
    ``output_dir/_simplified/{onnx_path.name}`` so the original is preserved
    and the stem remains unchanged for downstream ``.tflite`` file-name
    matching.

    Args:
        onnx_path: Path to the original ``.onnx`` file.
        output_dir: Directory where the patched model is cached.

    Returns:
        Path to the patched ``.onnx`` file, or *onnx_path* unchanged when
        ``onnx`` is not importable.
    """
    try:
        import onnx
    except ImportError:
        logger.debug("onnx not installed — skipping ONNX patching for onnx2tf compatibility")
        return onnx_path

    model = onnx.load(str(onnx_path))
    model = _fold_constant_expands(model)
    model = _replace_gelu_erf_with_tanh_approx(model)
    remaining_erf = sum(1 for n in model.graph.node if n.op_type == "Erf")
    if remaining_erf != 0:
        raise RuntimeError(
            f"GELU patch incomplete: {remaining_erf} Erf node(s) remain. "
            "The ONNX graph may use Mul(x, 1/sqrt(2)) instead of Div(x, sqrt(2)) — "
            "inspect and extend the matcher in _replace_gelu_erf_with_tanh_approx."
        )

    patched_dir = output_dir / "_simplified"
    patched_dir.mkdir(parents=True, exist_ok=True)
    patched_path = patched_dir / onnx_path.name
    onnx.save(model, str(patched_path))
    logger.info(f"ONNX patched for onnx2tf: {onnx_path.name} → {patched_path}")
    return patched_path


def _check_onnx2tf_available() -> None:
    """Verify that the ``onnx2tf`` package is importable.

    Raises:
        ImportError: If ``onnx2tf`` cannot be imported.
    """
    try:
        import onnx2tf  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "onnx2tf is not installed. TFLite export requires both ONNX and "
            "TFLite export dependencies. Install them with: "
            "pip install rfdetr[onnx,tflite]"
        ) from exc


@contextlib.contextmanager
def _numpy_allow_pickle() -> Generator[None, None, None]:
    """Temporarily patch :func:`numpy.load` to set ``allow_pickle=True``.

    ``onnx2tf`` 1.x calls ``np.load()`` on its bundled calibration data
    without passing ``allow_pickle=True``.  NumPy ≥ 1.16.3 defaults that
    flag to ``False`` and raises :class:`ValueError` for pickled files.

    This context manager monkey-patches ``np.load`` for the duration of the
    ``onnx2tf`` conversion and restores the original afterwards.
    """
    _original_load = np.load

    def _patched_load(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("allow_pickle", True)
        return _original_load(*args, **kwargs)

    np.load = _patched_load  # type: ignore[assignment,unused-ignore]
    try:
        yield
    finally:
        np.load = _original_load  # type: ignore[assignment,unused-ignore]


@contextlib.contextmanager
def _patch_validation_download(npy_path: str) -> Generator[None, None, None]:
    """Redirect ``download_test_image_data()`` to use local calibration data.

    ``onnx2tf`` calls ``download_test_image_data()`` during conversion to
    fetch test images from GitHub.  The function is called in two places:

    1. **Validation** — compares ONNX-vs-TF outputs (all conversions).
    2. **INT8 calibration** — builds a representative dataset when
       ``output_integer_quantized_tflite=True``.

    This download can fail in many environments (firewalls, CI, air-gapped
    systems, or when the upstream file is unavailable).  This context
    manager monkey-patches the function in all known module locations to
    return the data from the calibration ``.npy`` file we already prepared.

    We intentionally do **not** use ``custom_input_op_name_np_data_path``
    because that code path triggers a ``tf.tile`` rank mismatch in onnx2tf
    1.x when processing models with DINOv2-style embeddings and N > 1
    calibration samples.  Patching the download function achieves the same
    goal without that issue.

    Args:
        npy_path: Path to the ``.npy`` file containing calibration data in
            NHWC format.
    """

    def _replacement() -> NDArray[Any]:
        # Calibration data prepared by _prepare_calibration_data() is always
        # a plain float32 ndarray — never pickled.  allow_pickle=False is
        # intentional here; allow_pickle=True is handled by _numpy_allow_pickle()
        # for onnx2tf's own internal np.load calls.
        return cast(NDArray[Any], np.load(npy_path, allow_pickle=False))

    originals: dict[str, Any] = {}
    modules = [
        "onnx2tf.utils.common_functions",
        "onnx2tf.onnx2tf",
    ]
    for mod_name in modules:
        mod = sys.modules.get(mod_name)
        if mod and hasattr(mod, "download_test_image_data"):
            originals[mod_name] = getattr(mod, "download_test_image_data")
            setattr(mod, "download_test_image_data", _replacement)

    try:
        yield
    finally:
        for mod_name, original in originals.items():
            mod = sys.modules.get(mod_name)
            if mod:
                setattr(mod, "download_test_image_data", original)


@contextlib.contextmanager
def _skip_int16_activation_quantization() -> Generator[None, None, None]:
    """Make TFLite's INT8-with-int16-activations calibration fail fast.

    ``onnx2tf``, when ``output_integer_quantized_tflite=True``, emits four INT8
    variants: ``_integer_quant``, ``_full_integer_quant``,
    ``_integer_quant_with_int16_act``, ``_full_integer_quant_with_int16_act``.

    The two int16-activation variants are incompatible with the pseudo-GridSample
    replacement (see the GridSample replacement kwarg in the ``convert()`` call):
    pseudo-GridSample emits ``Cast(float→int32)`` ops to index the bilinear
    gather, and the int16-activation calibrator requires a continuous min/max
    range on every intermediate tensor. The ``Cast`` outputs are integer-valued,
    the calibrator records empty min/max, and quantization aborts with::

        RuntimeError: Max and min for dynamic tensors should be recorded
        during calibration: ... Empty min/max for tensor ... /Cast

    ``onnx2tf`` catches this and logs a warning, but the *calibration loop
    itself* still runs the full representative dataset before the error fires.
    On a Colab CPU runtime this takes long enough to look like a hang. This
    patch short-circuits the calibration call when ``q_activations_type`` is
    ``int16``, preserving ``onnx2tf``'s existing warning-and-continue behavior
    but eliminating the wasted compute.

    The two non-int16 variants (``_integer_quant``, ``_full_integer_quant``)
    are unaffected and still generate normally.
    """
    try:
        import tensorflow as tf
        from tensorflow.lite.python import lite as _tf_lite
    except ImportError:
        # No TF available — nothing to patch, nothing to skip
        yield
        return

    # Locate the method onnx2tf calls. In current TF (>=2.13) it's on
    # TFLiteConverterBase; older versions had it on TFLiteConverter directly.
    # Patch whichever exists.
    target_cls = getattr(_tf_lite, "TFLiteConverterBase", None) or _tf_lite.TFLiteConverter
    if not hasattr(target_cls, "_quantize"):
        logger.debug("No _quantize method on TFLiteConverter; int16-activation skip patch is a no-op")
        yield
        return

    original = target_cls._quantize
    int16 = tf.int16

    def _patched_quantize(self, model, q_in_type, q_out_type, q_activations_type, *args, **kwargs):  # type: ignore[no-untyped-def]
        if q_activations_type == int16:
            # Mirror the message onnx2tf would log after the slow path fails,
            # so the user sees the same outcome without the wait.
            raise RuntimeError(
                "INT8 quantization with int16 activations skipped: "
                "incompatible with pseudo-GridSample Cast ops "
                "(RF-DETR converter optimization)."
            )
        return original(self, model, q_in_type, q_out_type, q_activations_type, *args, **kwargs)

    target_cls._quantize = _patched_quantize
    try:
        yield
    finally:
        target_cls._quantize = original


def _load_calibration_images(
    image_dir: Path,
    height: int,
    width: int,
    channels: int = 3,
    max_images: int = _DEFAULT_DIR_CALIB_SAMPLES,
) -> NDArray[np.float32]:
    """Load images from a directory and prepare them for calibration.

    Images are loaded, resized to ``(height, width)`` with BILINEAR resampling
    (matching ``_run_inference``), converted to ``float32`` in ``[0, 1]``, and
    stacked into an NHWC array.

    Args:
        image_dir: Directory containing image files (JPEG, PNG, etc.).
        height: Target image height matching the model input.
        width: Target image width matching the model input.
        channels: Number of channels the model expects.  Use ``1`` for
            grayscale models (images are opened as ``"L"``), ``3`` for RGB.
        max_images: Maximum number of images to load.  Files are sorted
            alphabetically and the first *max_images* are used.

    Returns:
        A ``float32`` NumPy array of shape ``(N, height, width, channels)``
        with pixel values in ``[0, 1]``.

    Raises:
        FileNotFoundError: If *image_dir* does not exist or contains no
            supported image files.
    """
    from PIL import Image

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Calibration image directory not found: {image_dir}")

    image_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS)

    if not image_paths:
        raise FileNotFoundError(
            f"No supported image files found in {image_dir}. Supported extensions: {sorted(_IMAGE_EXTENSIONS)}"
        )

    image_paths = image_paths[:max_images]
    logger.info(f"Loading {len(image_paths)} calibration images from {image_dir} (resizing to {height}x{width})")

    pil_mode = "L" if channels == 1 else "RGB"
    arrays: list[NDArray[np.float32]] = []
    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert(pil_mode).resize((width, height), Image.Resampling.BILINEAR)
            arr = np.asarray(img, dtype=np.float32) / np.float32(255.0)
            if arr.ndim == 2:  # L-mode → (H, W); add channel axis
                arr = arr[:, :, np.newaxis]
            if arr.shape[-1] != channels:
                logger.debug("Skipping %s: %d channels, model expects %d", img_path, arr.shape[-1], channels)
                continue
            arrays.append(arr)
        except Exception:
            logger.debug(f"Skipping unreadable image: {img_path}")
            continue

    if not arrays:
        raise FileNotFoundError(f"No readable images found in {image_dir}")

    logger.info(f"Loaded {len(arrays)} calibration images")
    return np.stack(arrays).astype(np.float32, copy=False)


def _get_onnx_input_info(onnx_path: Path) -> tuple[str, list[int]]:
    """Read the first input tensor's name and shape from an ONNX model.

    Args:
        onnx_path: Path to the ``.onnx`` file.

    Returns:
        A ``(name, dims)`` tuple where *dims* is the NCHW shape list,
        e.g. ``("input", [1, 3, 560, 560])``.
    """
    try:
        import onnx
    except ImportError as exc:
        raise ImportError(
            "onnx is not installed. TFLite export requires both ONNX and "
            "TFLite export dependencies. Install them with: "
            "pip install rfdetr[onnx,tflite]"
        ) from exc

    model = onnx.load(str(onnx_path))
    inp = model.graph.input[0]
    name = inp.name
    dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    return name, dims


def _prepare_calibration_data(
    onnx_path: Path,
    calibration_data: str | os.PathLike[str] | np.ndarray | None,
    output_dir: Path,
    quantization: str | None,
    max_images: int = _DEFAULT_DIR_CALIB_SAMPLES,
) -> Path:
    """Prepare calibration data as a ``.npy`` file for ``onnx2tf``.

    The returned path points to a ``.npy`` file containing an NHWC float32
    array with pixel values in ``[0, 1]``.  This file is loaded by the
    ``_patch_validation_download()`` context manager, which replaces
    ``onnx2tf``'s built-in ``download_test_image_data()`` call.  ``onnx2tf``
    uses this data for both ONNX-vs-TF output validation and (when INT8 is
    requested) as a representative calibration dataset.

    Args:
        onnx_path: Path to the source ``.onnx`` file (used to read the
            input tensor NCHW shape for random data generation and for
            determining the target resolution when loading images from
            a directory).
        calibration_data: One of:

            * ``None`` — generate random calibration data.  Sufficient for
              fp32/fp16 but emits a warning for int8.
            * A **directory path** containing JPEG/PNG images — images are
              loaded, resized to the model input resolution, and converted
              to the correct format automatically.
            * A path to a ``.npy`` file containing an array of shape
              ``(N, H, W, 3)``, dtype float32, values in ``[0, 1]``.
            * A :class:`numpy.ndarray` with the same constraints.
        output_dir: Directory where a temporary ``.npy`` file may be
            written when *calibration_data* is ``None``, a directory, or
            an ndarray.
        quantization: The requested quantization mode (used only to decide
            whether to emit a warning).
        max_images: Maximum number of images to load when
            *calibration_data* is a directory path.  Ignored for other
            calibration data formats.

    Returns:
        Path to the ``.npy`` calibration data file containing an NHWC
        ``float32`` array with ImageNet-normalised pixel values
        (approximately ``[-2.1, 2.6]``).

    Raises:
        FileNotFoundError: If *calibration_data* is a path that does not
            exist, or a directory with no supported images.
    """
    if calibration_data is None:
        if quantization == "int8":
            logger.warning(
                "No calibration_data provided for INT8 quantization. Using "
                "random data — this will produce poor quantization accuracy. "
                "For best results, pass calibration_data with representative "
                "images from your dataset."
            )
        _, input_dims = _get_onnx_input_info(onnx_path)
        # input_dims is NCHW, e.g. [1, 3, 384, 384].
        _, c, h, w = input_dims
        # NHWC, float32, [0, 1] range — normalised to ImageNet stats below.
        calib = np.random.rand(_DEFAULT_CALIB_SAMPLES, h, w, c).astype(np.float32)
        calib = _imagenet_normalize(calib)
        npy_path = output_dir / "_rfdetr_calib_data.npy"
        np.save(str(npy_path), calib)
        logger.debug(f"Generated random calibration data: shape={calib.shape}, saved to {npy_path}")
    elif isinstance(calibration_data, np.ndarray):
        _, input_dims = _get_onnx_input_info(onnx_path)
        _, model_c, _, _ = input_dims
        if calibration_data.ndim != 4 or calibration_data.shape[-1] != model_c:
            raise ValueError(
                f"calibration_data has shape {calibration_data.shape}; model expects "
                f"NHWC with last dim == {model_c}. For a 1-channel model, "
                f"pass an array of shape (N, H, W, 1)."
            )
        npy_path = output_dir / "_rfdetr_calib_data.npy"
        norm_calib = _imagenet_normalize(calibration_data.astype(np.float32, copy=False))
        np.save(str(npy_path), norm_calib)
        logger.info(f"Using provided calibration array: shape={calibration_data.shape}")
    else:
        data_path = Path(calibration_data)
        if data_path.is_dir():
            # Directory of images — load, resize, and convert.
            _, input_dims = _get_onnx_input_info(onnx_path)
            _, _c, h, w = input_dims
            calib = _load_calibration_images(data_path, height=h, width=w, channels=_c, max_images=max_images)
            calib = _imagenet_normalize(calib)
            npy_path = output_dir / "_rfdetr_calib_data.npy"
            np.save(str(npy_path), calib)
            logger.info(f"Prepared calibration data from image directory: shape={calib.shape}, saved to {npy_path}")
        elif data_path.is_file():
            _, input_dims = _get_onnx_input_info(onnx_path)
            _, model_c, _, _ = input_dims
            _loaded = np.load(str(data_path), allow_pickle=False).astype(np.float32, copy=False)
            if _loaded.ndim != 4 or _loaded.shape[-1] != model_c:
                raise ValueError(
                    f"calibration_data file has shape {_loaded.shape}; model expects "
                    f"NHWC with last dim == {model_c}. For a 1-channel model, "
                    f"pass an array of shape (N, H, W, 1)."
                )
            npy_path = output_dir / "_rfdetr_calib_data.npy"
            np.save(str(npy_path), _imagenet_normalize(_loaded))
            logger.info(f"Using calibration data from: {data_path}")
        else:
            raise FileNotFoundError(f"Calibration data path not found: {data_path}")

    return npy_path


def export_tflite(
    onnx_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    quantization: str | None = None,
    calibration_data: str | os.PathLike[str] | np.ndarray | None = None,
    verbosity: str = "error",
    max_images: int = _DEFAULT_DIR_CALIB_SAMPLES,
    *,
    verbose: bool = False,
) -> Path:
    """Convert an ONNX model to TFLite via ``onnx2tf``.

    Uses the ``onnx2tf`` Python API with a NumPy compatibility shim so
    that both 1.x and 2.x releases of ``onnx2tf`` work correctly.

    Args:
        onnx_path: Path to the source ``.onnx`` file.
        output_dir: Directory where TFLite artifacts will be written.
            ``onnx2tf`` creates files named ``{stem}_float32.tflite`` and
            ``{stem}_float16.tflite`` (plus ``{stem}_integer_quant.tflite``
            when ``quantization="int8"``).
        quantization: Quantization mode.

            * ``None`` / ``"fp32"`` — default FP32 + FP16 output.
            * ``"fp16"`` — same as above (onnx2tf always emits both).
            * ``"int8"`` — additionally produce an INT8-quantized model.
        calibration_data: Representative data used by ``onnx2tf`` for
            output validation (fp32/fp16) and INT8 calibration.  Accepts:

            * ``None`` — auto-generate random data (warns for int8).
            * A **directory path** containing JPEG/PNG images — images
              are loaded, resized, and converted automatically.
            * A path to a ``.npy`` file — shape ``(N, H, W, 3)``,
              dtype float32, pixel values in ``[0, 1]``.
            * A :class:`numpy.ndarray` with the same format.

            For INT8 quantization, provide real images from your dataset
            for best accuracy.
        verbosity: Log verbosity passed to ``onnx2tf``.  One of
            ``"debug"``, ``"info"``, ``"warn"``, ``"error"`` (default).
        max_images: Maximum number of images to load when
            *calibration_data* is a directory path.  Defaults to 100.
            Ignored for other calibration data formats.
        verbose: When ``True``, stream ``onnx2tf`` per-node progress —
            useful for monitoring long conversions (5–15 min on
            transformer-based models).  Defaults to ``False`` (silent).

    Returns:
        The path to the primary ``*_float32.tflite`` file.

    Raises:
        FileNotFoundError: If *onnx_path* does not exist or
            *calibration_data* points to a missing file.
        ImportError: If ``onnx2tf`` is not installed.
        ValueError: If *quantization* is not a recognized mode.
        RuntimeError: If the conversion fails.

    Note:
        This function is **not thread-safe**.  It globally monkey-patches
        :func:`numpy.load` (via :func:`_numpy_allow_pickle`) and
        ``onnx2tf.download_test_image_data`` (via
        :func:`_patch_validation_download`) for the duration of the
        conversion.  Concurrent calls from multiple threads will interfere
        with each other.  Run conversion in a subprocess if isolation is
        required.
    """
    onnx_path = Path(onnx_path)
    output_dir = Path(output_dir)

    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    if quantization not in _VALID_QUANTIZATIONS:
        raise ValueError(
            f"Unsupported quantization mode {quantization!r}. "
            f"Choose from: {sorted(q for q in _VALID_QUANTIZATIONS if q is not None)}"
        )

    _check_onnx2tf_available()

    # Force-import onnx2tf submodules so that _patch_validation_download()
    # can patch them.  onnx2tf's __init__.py may not import all submodules
    # eagerly in all versions, so we ensure they are in sys.modules before
    # entering the patching context manager.
    import onnx2tf.onnx2tf as _onnx2tf_mod
    import onnx2tf.utils.common_functions as _onnx2tf_common

    del _onnx2tf_mod, _onnx2tf_common  # imported for side-effect only

    output_dir.mkdir(parents=True, exist_ok=True)

    # Patch the ONNX graph to work around onnx2tf 1.x bugs before conversion.
    # See _patch_onnx_for_tflite() for details of the two targeted fixes.
    onnx_path_for_conversion = _patch_onnx_for_tflite(onnx_path, output_dir)

    calib_npy_path = _prepare_calibration_data(
        onnx_path, calibration_data, output_dir, quantization, max_images=max_images
    )

    # onnx2tf names output files based on the ONNX model stem.
    model_stem = onnx_path.stem

    logger.info(f"Converting ONNX → TFLite (quantization={quantization!r}, verbosity={verbosity!r}): {onnx_path}")

    from onnx2tf import convert

    convert_kwargs: dict[str, Any] = {
        "input_onnx_file_path": str(onnx_path_for_conversion),
        "output_folder_path": str(output_dir),
        "output_signaturedefs": True,
        "non_verbose": not verbose,
        "verbosity": verbosity,
    }

    if quantization == "int8":
        convert_kwargs["output_integer_quantized_tflite"] = True

    if _GRIDSAMPLE_KWARG is not None:
        convert_kwargs[_GRIDSAMPLE_KWARG] = True
        logger.debug(f"Enabling onnx2tf GridSample replacement: {_GRIDSAMPLE_KWARG}=True")
    else:
        logger.warning(
            "Installed onnx2tf has no GridSample replacement kwarg. "
            "If the exported TFLite model produces low-confidence detections, "
            "this is likely onnx2tf#274 and you should pin onnx2tf to a version "
            "that supports the replacement (e.g. onnx2tf<2.4)."
        )

    def _run_convert(kwargs: dict[str, Any]) -> None:
        # _patch_validation_download redirects onnx2tf's
        # download_test_image_data() to return our calibration data.
        # onnx2tf uses this data for both ONNX/TF output validation and
        # (when int8 is requested) as a representative calibration dataset.
        #
        # We intentionally do NOT pass custom_input_op_name_np_data_path
        # because that code path in onnx2tf 1.x triggers a tf.tile rank
        # mismatch when processing the DINOv2 backbone with N > 1 samples.
        # The patched download function achieves the same goal without that
        # issue.
        #
        # output_signaturedefs=True is required because segmentation
        # models produce ONNX node names (e.g.
        # "/segmentation_head/blocks.2/dwconv/Conv/kernel") that contain
        # leading "/" characters which violate the saved_model naming
        # pattern. Enabling signature defs bypasses this restriction.
        with (
            _numpy_allow_pickle(),
            _patch_validation_download(str(calib_npy_path)),
            _skip_int16_activation_quantization(),
        ):
            convert(**kwargs)

    try:
        _run_convert(convert_kwargs)
    except Exception as first_exc:
        # onnx2tf auto-generates {stem}_auto.json in the output directory when
        # it encounters ops it cannot convert directly (e.g. TopK with a 1-D
        # k tensor).  Retrying with param_replacement_file resolves most such
        # failures without any manual intervention.
        auto_json_path = output_dir / f"{model_stem}_auto.json"
        if auto_json_path.is_file():
            logger.info(
                f"onnx2tf generated replacement JSON: {auto_json_path.name}. "
                "Retrying conversion with param_replacement_file..."
            )
            try:
                convert_kwargs["param_replacement_file"] = str(auto_json_path)
                _run_convert(convert_kwargs)
            except Exception as retry_exc:
                logger.error(f"onnx2tf conversion failed (retry with {auto_json_path.name}): {retry_exc}")
                raise RuntimeError(
                    f"onnx2tf conversion failed (retry with {auto_json_path.name}): {retry_exc}"
                ) from retry_exc
        else:
            logger.error(f"onnx2tf conversion failed: {first_exc}")
            raise RuntimeError(f"onnx2tf conversion failed: {first_exc}") from first_exc

    # onnx2tf always emits _float32.tflite; INT8 additionally emits _integer_quant.tflite.
    # Return the most specific file for the requested quantization.
    expected_name = f"{model_stem}_integer_quant.tflite" if quantization == "int8" else f"{model_stem}_float32.tflite"
    primary = output_dir / expected_name

    if not primary.is_file():
        # Fallback: look for any .tflite file produced from this specific ONNX stem.
        # Scoped to {stem}_*.tflite to avoid returning a stale artifact from a
        # previous export in a reused output directory (review C2).
        tflite_files = sorted(output_dir.glob(f"{model_stem}_*.tflite"))
        if tflite_files:
            primary = tflite_files[0]
            logger.info(f"Expected {expected_name} not found; using {primary.name} instead.")
        else:
            raise RuntimeError(
                f"onnx2tf completed but no .tflite file matching '{model_stem}_*.tflite' was found in {output_dir}"
            )

    logger.info(f"TFLite model exported to: {primary}")
    return primary
