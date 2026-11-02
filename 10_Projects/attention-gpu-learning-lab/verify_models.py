#!/usr/bin/env python3
"""CPU-only numerical checks of the lesson's model; requires NumPy, not CUDA.

These checks do not execute or validate the PyTorch, CUDA, or Triton kernels.
"""
import json
import numpy as np


def online(q, k, v, block_q, block_k):
    out = np.empty(q.shape[:-1] + (v.shape[-1],))
    for i in range(0, q.shape[-2], block_q):
        qi = q[..., i:i+block_q, :]
        m = np.full(qi.shape[:-1] + (1,), -np.inf)
        ell = np.zeros_like(m)
        u = np.zeros(qi.shape[:-1] + (v.shape[-1],))
        for j in range(0, k.shape[-2], block_k):
            z = qi @ k[..., j:j+block_k, :].swapaxes(-2, -1) / np.sqrt(q.shape[-1])
            new_m = np.maximum(m, z.max(-1, keepdims=True))
            alpha = np.exp(m-new_m)
            e = np.exp(z-new_m)
            ell = alpha*ell + e.sum(-1, keepdims=True)
            u = alpha*u + e @ v[..., j:j+block_k, :]
            m = new_m
        out[..., i:i+block_q, :] = u/ell
    return out


def main():
    b,h,n,d,s = 1,8,4096,64,2
    x, a = b*h*n*d*s, b*h*n*n*s
    f = 4*b*h*n*n*d
    report = {
        "shape": [b,h,n,d], "storage_bytes_per_element": s,
        "X_bytes": x, "A_bytes": a, "two_GEMMs_FLOPs": f,
        "eager_ideal_logical_bytes": 6*a+4*x,
        "scale_fused_ideal_logical_bytes": 4*a+4*x,
        "Q_resident_BM128_no_cross_tile_cache_bytes": 2*x+2*(n//128)*x,
        "compulsory_cold_input_bytes": 4*x,
        "eager_peak_live_tensor_bytes": 2*a+3*x,
        "eager_AI_GEMM_FLOPs_per_byte": f/(6*a+4*x),
        "virtual_GPU_dense_GEMM_TFLOPs":100,
        "virtual_GPU_DRAM_TBps":1,
        "eager_model_lower_bound_ms":max(f/1e14,(6*a+4*x)/1e12)*1e3,
        "Q_resident_model_lower_bound_ms":max(f/1e14,(2+2*(n//128))*x/1e12)*1e3,
    }
    rng = np.random.default_rng(0)
    q,k,v = (rng.normal(size=shape) for shape in ((2,13,8),(2,17,8),(2,17,5)))
    z = q@k.swapaxes(-2,-1)/np.sqrt(q.shape[-1])
    p = np.exp(z-z.max(-1,keepdims=True))
    p /= p.sum(-1,keepdims=True)
    ref = p@v
    errors = []
    for bq in (1,5,13):
        for bk in (1,4,7,17):
            out = online(q,k,v,bq,bk)
            np.testing.assert_allclose(out,ref,rtol=1e-12,atol=1e-12)
            errors.append(float(np.max(np.abs(out-ref))))
    report["CPU_numpy_online_max_abs_error_across_12_tile_settings"] = max(errors)
    report["GPU_tests_executed"] = False
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
