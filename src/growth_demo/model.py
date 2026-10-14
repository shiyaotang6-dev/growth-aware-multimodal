"""A compact independent implementation of stage-aware multimodal learning."""

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


@dataclass
class ModelConfig:
    feature_dim: int = 64
    stage_dim: int = 16
    num_stages: int = 3
    num_classes: int = 5
    attention_heads: int = 4
    pretrained_rgb: bool = False

    def to_dict(self):
        return asdict(self)


class HyperspectralEncoder(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 8, (3, 3, 3), padding=1),
            nn.GroupNorm(2, 8),
            nn.GELU(),
            nn.Conv3d(8, 16, 3, stride=2, padding=1),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv3d(16, 32, 3, stride=2, padding=1),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(32, feature_dim),
        )

    def forward(self, cube):
        return self.features(cube)


class ModalityGraphAttention(nn.Module):
    """Two modality nodes exchange messages, conditioned on growth stage."""

    def __init__(self, feature_dim, stage_dim, heads):
        super().__init__()
        if feature_dim % heads:
            raise ValueError("feature_dim must be divisible by attention_heads")
        self.stage_projection = nn.Linear(stage_dim, feature_dim)
        self.attention = nn.MultiheadAttention(feature_dim, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(feature_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2), nn.GELU(), nn.Linear(feature_dim * 2, feature_dim)
        )
        self.norm2 = nn.LayerNorm(feature_dim)

    def forward(self, nodes, stage):
        conditioned = nodes + self.stage_projection(stage).unsqueeze(1)
        messages, adjacency = self.attention(
            conditioned, conditioned, nodes, need_weights=True, average_attn_weights=True
        )
        nodes = self.norm1(nodes + messages)
        return self.norm2(nodes + self.feedforward(nodes)), adjacency


class GrowthAwareMultiTaskNet(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        cfg = self.config
        weights = ResNet18_Weights.DEFAULT if cfg.pretrained_rgb else None
        self.rgb_encoder = resnet18(weights=weights)
        self.rgb_encoder.fc = nn.Linear(self.rgb_encoder.fc.in_features, cfg.feature_dim)
        self.hsi_encoder = HyperspectralEncoder(cfg.feature_dim)
        self.rgb_norm = nn.LayerNorm(cfg.feature_dim)
        self.hsi_norm = nn.LayerNorm(cfg.feature_dim)
        self.stage_embedding = nn.Embedding(cfg.num_stages, cfg.stage_dim)
        self.gate = nn.Sequential(
            nn.Linear(cfg.feature_dim * 2 + cfg.stage_dim, cfg.feature_dim),
            nn.GELU(),
            nn.Linear(cfg.feature_dim, 2),
        )
        self.graph = ModalityGraphAttention(cfg.feature_dim, cfg.stage_dim, cfg.attention_heads)
        self.shared = nn.Sequential(
            nn.Linear(cfg.feature_dim + cfg.stage_dim, cfg.feature_dim), nn.GELU(), nn.Dropout(0.1)
        )
        self.classifier = nn.Linear(cfg.feature_dim, cfg.num_classes)
        self.regressor = nn.Linear(cfg.feature_dim, 1)

    def forward(self, rgb, hsi, stage):
        rgb_features = self.rgb_norm(self.rgb_encoder(rgb))
        hsi_features = self.hsi_norm(self.hsi_encoder(hsi))
        stage_features = self.stage_embedding(stage)
        gates = self.gate(torch.cat((rgb_features, hsi_features, stage_features), dim=-1)).softmax(-1)
        nodes = torch.stack((rgb_features, hsi_features), dim=1)
        nodes, adjacency = self.graph(nodes * gates.unsqueeze(-1), stage_features)
        fused = (nodes * gates.unsqueeze(-1)).sum(dim=1)
        shared = self.shared(torch.cat((fused, stage_features), dim=-1))
        return {
            "logits": self.classifier(shared),
            "spad_normalized": self.regressor(shared).squeeze(-1),
            "gates": gates,
            "attention": adjacency,
        }
