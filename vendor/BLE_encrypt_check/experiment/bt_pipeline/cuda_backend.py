"""Small CUDA/CuPy adapter used by the wideband DSP pipeline.

This module keeps CuPy imports lazy so the CPU-only parser can still be imported
on machines without CUDA installed.
"""

from contextlib import contextmanager

import numpy as np


class CudaUnavailableError(RuntimeError):
    """Raised when a caller explicitly requests CUDA but it is not usable."""


def get_cupy():
    """Return the CuPy module, or None when CuPy cannot be imported."""
    try:
        import cupy as cp
    except Exception:
        return None
    return cp


def cuda_error_message():
    """Return a short explanation when CUDA is not currently available."""
    cp = get_cupy()
    if cp is None:
        return "CuPy is not installed or cannot be imported."
    try:
        device_count = cp.cuda.runtime.getDeviceCount()
    except Exception as exc:
        return f"CUDA runtime probe failed: {exc}"
    if device_count < 1:
        return "No CUDA-capable device was detected."
    return ""


def is_cuda_available():
    """Return True when CuPy can see at least one CUDA device."""
    return cuda_error_message() == ""


def require_cuda():
    """Raise a helpful error if CUDA was requested but is not usable."""
    message = cuda_error_message()
    if message:
        raise CudaUnavailableError(message)
    return get_cupy()


def get_array_module(use_cuda=False):
    """Return numpy or cupy according to the requested execution backend."""
    if not use_cuda:
        return np
    return require_cuda()


def is_device_array(array):
    """Return True for CuPy arrays without requiring CuPy at import time."""
    cp = get_cupy()
    return cp is not None and isinstance(array, cp.ndarray)


def to_device(array, dtype=None, device_id=None):
    """Copy or view an array on the selected CUDA device."""
    cp = require_cuda()
    with use_device(device_id):
        return cp.asarray(array, dtype=dtype)


def to_host(array, dtype=None):
    """Return a NumPy array, copying from GPU when needed."""
    if is_device_array(array):
        cp = get_cupy()
        host = cp.asnumpy(array)
    else:
        host = np.asarray(array)
    if dtype is not None:
        return host.astype(dtype, copy=False)
    return host


@contextmanager
def use_device(device_id=None):
    """Temporarily select a CUDA device when a device id is provided."""
    if device_id is None:
        yield
        return

    cp = require_cuda()
    with cp.cuda.Device(device_id):
        yield


def get_default_stream():
    """Return CuPy's current stream for callers that need explicit sync points."""
    cp = require_cuda()
    return cp.cuda.get_current_stream()


def synchronize():
    """Synchronize the current CUDA stream when CUDA is available."""
    cp = get_cupy()
    if cp is None:
        return
    try:
        cp.cuda.get_current_stream().synchronize()
    except Exception:
        return


def describe_cuda_status():
    """Return a compact status dictionary for logs and environment reports."""
    cp = get_cupy()
    status = {
        "cupy_importable": cp is not None,
        "cuda_available": False,
        "device_count": 0,
        "devices": [],
        "error": "",
    }
    if cp is None:
        status["error"] = "CuPy is not installed or cannot be imported."
        return status

    status["cupy_version"] = getattr(cp, "__version__", "")
    try:
        device_count = cp.cuda.runtime.getDeviceCount()
    except Exception as exc:
        status["error"] = str(exc)
        return status

    status["device_count"] = int(device_count)
    status["cuda_available"] = device_count > 0
    for device_id in range(device_count):
        props = cp.cuda.runtime.getDeviceProperties(device_id)
        name = props.get("name", b"")
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
        status["devices"].append(
            {
                "id": device_id,
                "name": name,
                "total_memory": int(props.get("totalGlobalMem", 0)),
                "compute_capability": (
                    int(props.get("major", 0)),
                    int(props.get("minor", 0)),
                ),
            }
        )
    return status
