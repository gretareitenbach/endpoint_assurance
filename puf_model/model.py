import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResidualDilatedBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dilation: int = 1, dropout: float = 0.0):
        super().__init__()
        pad = dilation
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=pad, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=pad, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SEBlock(out_ch)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.skip = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.dropout(out)
        return F.relu(out + identity, inplace=True)


class AffinePairSTN(nn.Module):
    """
    Lightweight pair-wise STN that predicts an affine transform for the verification scan.
    This is a practical stand-in for TPS during preliminary model development.
    """

    def __init__(self, in_channels: int = 2):
        super().__init__()
        self.loc = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 6),
        )

        # Initialize close to identity transform.
        nn.init.zeros_(self.fc[-1].weight)
        self.fc[-1].bias.data.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    def forward(self, baseline: torch.Tensor, verification: torch.Tensor) -> torch.Tensor:
        pair = torch.cat([baseline, verification], dim=1)
        theta = self.fc(self.loc(pair)).view(-1, 2, 3)
        grid = F.affine_grid(theta, verification.size(), align_corners=False)
        return F.grid_sample(verification, grid, mode="bilinear", padding_mode="border", align_corners=False)


class SharedEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 32, dropout: float = 0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.layer1 = ResidualDilatedBlock(base_channels, base_channels, dilation=1, dropout=dropout)
        self.layer2 = nn.Sequential(
            nn.MaxPool2d(2),
            ResidualDilatedBlock(base_channels, base_channels * 2, dilation=2, dropout=dropout),
        )
        self.layer3 = nn.Sequential(
            nn.MaxPool2d(2),
            ResidualDilatedBlock(base_channels * 2, base_channels * 4, dilation=3, dropout=dropout),
        )
        self.layer4 = nn.Sequential(
            nn.MaxPool2d(2),
            ResidualDilatedBlock(base_channels * 4, base_channels * 4, dilation=4, dropout=dropout),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feat_map = self.layer4(x)
        pooled = self.pool(feat_map).flatten(1)
        return feat_map, pooled


class DifferentialFusion(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, dep_feat: torch.Tensor, rec_feat: torch.Tensor) -> torch.Tensor:
        diff = rec_feat - dep_feat
        fused = torch.cat([diff, rec_feat, dep_feat], dim=1)
        return self.fuse(fused)


class SiameseAttentionalPUF(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.stn = AffinePairSTN(in_channels * 2)
        self.encoder = SharedEncoder(in_channels=in_channels, base_channels=base_channels, dropout=dropout)
        self.fusion = DifferentialFusion(base_channels * 4)

        head_in = base_channels * 4
        self.head_dropout = nn.Dropout(dropout)
        self.tamper_head = nn.Sequential(
            nn.Linear(head_in, head_in // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_in // 2, 1),
        )
        self.wear_head = nn.Sequential(
            nn.Linear(head_in, head_in // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_in // 2, 1),
        )

    def forward(self, baseline: torch.Tensor, verification: torch.Tensor):
        aligned_verification = self.stn(baseline, verification)

        dep_feat, _ = self.encoder(baseline)
        rec_feat, _ = self.encoder(aligned_verification)

        z_diff = self.fusion(dep_feat, rec_feat)
        z_pool = F.adaptive_avg_pool2d(z_diff, 1).flatten(1)
        z_pool = self.head_dropout(z_pool)

        tamper_logits = self.tamper_head(z_pool).squeeze(1)
        wear_logits = self.wear_head(z_pool).squeeze(1)

        return {
            "tamper_logits": tamper_logits,
            "wear_logits": wear_logits,
            "aligned_verification": aligned_verification,
            "z_diff": z_diff,
        }
