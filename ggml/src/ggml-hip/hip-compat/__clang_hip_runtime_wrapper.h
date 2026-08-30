#ifndef GGML_HIP_CLANG_RUNTIME_WRAPPER_H
#define GGML_HIP_CLANG_RUNTIME_WRAPPER_H

// Keep device math declarations ahead of MSVC <cmath>.
#if defined(__HIP__)
#include <__clang_cuda_math_forward_declares.h>
#endif

#include_next <__clang_hip_runtime_wrapper.h>

#endif
