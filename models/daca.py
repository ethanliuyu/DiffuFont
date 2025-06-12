import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
class DACA(nn.Module):
    """
    Deformable Attention Content Aggregation Module (DACA)
    """

    def __init__(self, in_channels, num_heads=4, num_points=4, t_embed_dim=128, t_mlp=None, base_pts=None):
        super().__init__()
        self.C = in_channels
        self.M = num_heads
        self.R = num_points

        # Predict spatial offsets Δp^m ∈ ℝ^{2×H×W} shared by each head.
        # For a 2×2 kernel (R=4 sampling points) each head learns **one** shared offset
        # instead of individual offsets per point.
        self.offset_conv = nn.Conv2d(2*self.C, 2*self.M, kernel_size=3, padding=1)
        # Predict attention weight A^m ∈ [0,1] for each head (shared among all sampling points).
        self.attn_conv   = nn.Conv2d(2*self.C, self.M, kernel_size=3, padding=1)
        # Learnable sampling-point weights W_p^m(r); initialized uniformly.
        self.Wp = nn.Parameter(torch.ones(self.M, self.R) / self.R)
        # Project M·C channels back to C to fuse heads.
        self.head_fuse = nn.Conv2d(self.M*self.C, self.C, kernel_size=1)

        # Normalized coordinates of the R base sampling points
        if base_pts is not None:
            # user-defined sampling points
            grid = torch.tensor(base_pts, dtype=torch.float32)
            assert grid.shape == (self.R, 2), f"base_pts 形状应为 ({self.R}, 2)"
        else:
            # default: 2×2 kernel
            if self.R == 4:
                xs = torch.tensor([-0.5, 0.5])
                ys = torch.tensor([-0.5, 0.5])
                grid = torch.stack(torch.meshgrid(ys, xs), dim=-1).view(-1, 2)  # (4,2)
            else:
                # uniformly distribute points inside square [-0.5,0.5]
                import itertools
                side = int(self.R ** 0.5)
                assert side * side == self.R, 'num_points 必须为平方数，或请自定义 base_pts'
                xs = torch.linspace(-0.5, 0.5, side)
                ys = torch.linspace(-0.5, 0.5, side)
                grid = torch.tensor(list(itertools.product(ys, xs)), dtype=torch.float32)
        self.register_buffer('base_pts', grid)

        # optional timestep MLP
        if t_mlp is not None:
            self.t_mlp = t_mlp
        elif t_embed_dim is not None:
            self.t_mlp = nn.Sequential(
                nn.Linear(1, t_embed_dim), nn.ReLU(inplace=True),
                nn.Linear(t_embed_dim, 1), nn.Sigmoid()
            )
        else:
            self.t_mlp = None

    def forward(self, F_x, F_c, t):
        B, C, H, W = F_x.shape

        # 1) predict offsets and attention
        # As suggested in the paper, interleave Query and Content features along channels:
        # [x0, c0, x1, c1, ...]
        # stack → permute → reshape to (B, 2C, H, W)
        tmp = torch.stack((F_x, F_c), dim=2)  # (B, C, 2, H, W)
        Fs = tmp.permute(0, 2, 1, 3, 4).reshape(B, 2 * C, H, W)
        offsets = self.offset_conv(Fs).view(B, self.M, 2, H, W)  # (B,M,2,H,W)
        attn = torch.sigmoid(self.attn_conv(Fs).view(B, self.M, 1, H, W))  # (B,M,1,H,W)

        # 2) sample R points with grid_sample
        ys = torch.linspace(-1, 1, H, device=F_x.device)
        xs = torch.linspace(-1, 1, W, device=F_x.device)
        base_grid = torch.stack(torch.meshgrid(ys, xs), dim=-1)[None, None]  # (1,1,H,W,2)

        samples = []
        # For every spatial location p on F_c use the 2×2 kernel's 4 relative sampling points p_r + Δp.
        for r in range(self.R):
            dp = self.base_pts[r:r + 1].view(1, 1, 1, 1, 2)  # (1,1,1,1,2)
            off = offsets.permute(0, 1, 3, 4, 2)  # (B,M,H,W,2)
            grid = base_grid + dp + off  # (B,M,H,W,2)
            grid = grid.view(B * self.M, H, W, 2)

            Vrep = F_c.unsqueeze(1).repeat(1, self.M, 1, 1, 1).view(B * self.M, C, H, W)
            samp = F.grid_sample(Vrep, grid, align_corners=True)  # (B*M,C,H,W)
            samples.append(samp.view(B, self.M, C, H, W))

        # stack → (R, B, M, C, H, W)
        stacked = torch.stack(samples, dim=0)

        # --- Important: reshape Wp from (M,R) to (R,1,M,1,1,1) ---
        # 1) transpose to (R,M)
        # 2) unsqueeze broadcast dims: batch and C/H/W
        Wp = self.Wp.t().unsqueeze(1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # Wp.shape == (R, 1, M, 1, 1, 1)

        # A: attn is (B,M,1,H,W); unsqueeze front dim so it matches stacked dims.
        A = attn.unsqueeze(0)  # (1, B, M, 1, H, W)

        # Weighted sum along R → (B, M, C, H, W)
        head_vals = (stacked * Wp * A).sum(dim=0)

        # 3) fuse heads back to C channels
        head_vals = head_vals.view(B, self.M * C, H, W)
        F_e = self.head_fuse(head_vals)  # (B,C,H,W)

        # 4) predict interpolation factor α from timestep t
        if self.t_mlp is not None:
            t_in = t.view(-1, 1).float()
            alpha = self.t_mlp(t_in).view(B, 1, 1, 1)  # (B,1,1,1)
            F_out = alpha * F_e + (1 - alpha) * F_x
        else:
            F_out = F_e

        loss_offset = offsets.abs().mean()
        # store for external logging
        self.last_offset_loss = loss_offset
        return F_out, loss_offset,offsets

# Quick sanity test
if __name__ == "__main__":
    B, C, H, W = 2, 64, 32, 32
    model = DACA(in_channels=C, num_heads=3, num_points=4)
    Fx = torch.randn(B, C, H, W)
    Fc = torch.randn(B, C, H, W)
    t  = torch.tensor([10.0, 20.0])
    out, loss,offsets = model(Fx, Fc, t)
    print("Out:", out.shape, "Offset loss:", loss.item())
    # offsets: (B, M, 2, H, W)


    offsets_np = offsets.detach().cpu().numpy()
    B, M, _, H, W = offsets_np.shape

    for m in range(M):  # 遍历每个head
        dx = offsets_np[0, m, 0]  # (H, W)
        dy = offsets_np[0, m, 1]  # (H, W)
        X, Y = np.meshgrid(np.arange(W), np.arange(H))
        plt.figure(figsize=(6, 6))
        plt.quiver(X, Y, dx, dy, angles='xy', scale_units='xy', scale=1)
        plt.title(f'Offset Field (Head {m})')
        plt.gca().invert_yaxis()
        plt.show()
