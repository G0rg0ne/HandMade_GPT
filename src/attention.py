import torch

torch.manual_seed(1337)

def trick(x):
    B, T, C = x.shape
    xbow = torch.zeros((B, T, C), dtype=x.dtype)
    for b in range(B):
        for t in range(T):
            xprev = x[b, :t+1]  # (t+1, C)
            xbow[b, t] = torch.mean(xprev, dim=0)
    return xbow

def fast_trick(x):
    B, T, C = x.shape
    dtype = x.dtype
    wei = torch.tril(torch.ones(T, T, dtype=dtype))  # (T, T)
    wei = wei / wei.sum(1, keepdim=True)  # row-normalized
    xbow2 = torch.matmul(wei, x)  # (1, T, T) @ (B, T, C) -> (B, T, C)
    return xbow2

if __name__ == "__main__":
    B, T, C = 4, 8, 2
    x = torch.randn(B, T, C)
    xbow_app1 = trick(x)
    xbow_app2 = fast_trick(x)
    print(torch.allclose(xbow_app1, xbow_app2, atol=1e-6))  # Should print True
