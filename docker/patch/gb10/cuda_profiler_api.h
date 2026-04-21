#ifndef __CUDA_PROFILER_API_H__
#define __CUDA_PROFILER_API_H__

/* CUDA 13 dropped the public cuda_profiler_api.h header, but the runtime still
 * exports cudaProfilerStart / cudaProfilerStop from libcudart.so. This shim
 * declares them against the public types so legacy callers (e.g., TE 2.10,
 * which only includes the header without actually calling the APIs in the
 * affected TUs) continue to compile.
 *
 * Installed by slime GB10 port. See docker/patch/gb10/ and NOTES_GB10.md.
 */

#include <cuda_runtime_api.h>

#if defined(__cplusplus)
extern "C" {
#endif

extern __host__ cudaError_t CUDARTAPI cudaProfilerStart(void);
extern __host__ cudaError_t CUDARTAPI cudaProfilerStop(void);

#if defined(__cplusplus)
}
#endif

#endif /* __CUDA_PROFILER_API_H__ */
