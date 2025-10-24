import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 16}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 32}, num_warps=8),
    ],
    key=['M', 'N', 'K']
)
@triton.jit
def optimized_matmul_kernel(
    A, B, C, 
    M, N, K, 
    stride_am, stride_ak, 
    stride_bk, stride_bn, 
    stride_cm, stride_cn, 
    BLOCK_SIZE: tl.constexpr
):
    """Optimized matrix multiplication kernel using Triton."""
    
    pid = tl.program_id(0)  
    num_cols = N // BLOCK_SIZE  
    row = pid // num_cols  
    col = pid % num_cols  

    a_ptr = A + row * stride_am + tl.arange(0, BLOCK_SIZE)[:, None] * stride_ak
    b_ptr = B + tl.arange(0, BLOCK_SIZE)[None, :] * stride_bn + col * stride_bk

    
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

  
    for k in range(0, K, BLOCK_SIZE):
        a = tl.load(a_ptr, mask=a_ptr < A + M * stride_am, other=0.0)
        b = tl.load(b_ptr, mask=b_ptr < B + K * stride_bk, other=0.0)
        acc += tl.dot(a, b)  
        
        
        a_ptr += BLOCK_SIZE * stride_ak
        b_ptr += BLOCK_SIZE * stride_bk

    
    c_ptr = C + row * stride_cm + col * stride_cn
    tl.store(c_ptr, acc)

M, N, K = 1024, 1024, 1024
BLOCK_SIZE = 16  

A = torch.randn((M, K), device="cuda", dtype=torch.float32)
B = torch.randn((K, N), device="cuda", dtype=torch.float32)
C = torch.empty((M, N), device="cuda", dtype=torch.float32)

grid = (M // BLOCK_SIZE) * (N // BLOCK_SIZE)

optimized_matmul_kernel[grid](
    A, B, C, 
    M, N, K, 
    A.stride(0), A.stride(1), 
    B.stride(0), B.stride(1), 
    C.stride(0), C.stride(1), 
    BLOCK_SIZE
)

C_ref = torch.matmul(A, B)
print(torch.allclose(C, C_ref, atol=1e-3))