import torch
import torch.nn as nn
from .core import AsymmetricPsiTNeuron


class NeuroFractalNet(nn.Module):
    """Reseau de neurones complexe pour la detection de maladies neurodegeneratives.

    Architecture :
        1. Couche d'embedding complexe : projette les 64 canaux EEG en representation complexe
        2. Couche ΨTA 1 : extraction des features chaotiques (64 -> 128)
        3. Couche ΨTA 2 : abstraction des patterns pathologiques (128 -> 64)
        4. Couches reelles : classification finale (128 -> 32 -> 2)

    L'astuce : les couches ΨTA traitent le signal en complexe (preservant amplitude + phase),
    puis on deplie en reel (partie reelle + imaginaire) pour la classification finale.
    """

    def __init__(
        self,
        n_channels=64,
        n_times=256,
        complex_hidden1=128,
        complex_hidden2=64,
        real_hidden=32,
        n_classes=2,
        dropout=0.3,
    ):
        super().__init__()

        self.n_channels = n_channels

        self.input_proj = nn.Linear(n_times, n_times, dtype=torch.complex64)

        self.psi_layer1 = AsymmetricPsiTNeuron(n_channels, complex_hidden1)
        self.psi_layer2 = AsymmetricPsiTNeuron(complex_hidden1, complex_hidden2)

        self.norm1 = nn.LayerNorm([complex_hidden1])
        self.norm2 = nn.LayerNorm([complex_hidden2])

        real_input_dim = complex_hidden2 * 2

        self.classifier = nn.Sequential(
            nn.Linear(real_input_dim, real_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(real_hidden, real_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(real_hidden // 2, n_classes),
        )

    def forward(self, x):
        if x.dtype != torch.complex64:
            x = x.to(torch.complex64)

        if x.dim() == 3:
            x = x.squeeze(1)

        x = self.input_proj(x)

        h = self.psi_layer1(x)
        h_real = torch.cat([torch.real(h), torch.imag(h)], dim=1)
        h_real = self.norm1(h_real)

        h = self.psi_layer2(h)
        h_real = torch.cat([torch.real(h), torch.imag(h)], dim=1)
        h_real = self.norm2(h_real)

        out = self.classifier(h_real)
        return out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BaselineCNN(nn.Module):
    """Modele de reference : CNN reel classique (sans complexe).

    Pour comparer avec ΨTA et prouver la superiorite de l'approche complexe.
    """

    def __init__(self, n_channels=64, n_times=256, n_classes=2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Linear(n_times, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * n_channels, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)

        if x.is_complex():
            x = torch.cat([torch.real(x), torch.imag(x)], dim=1)

        batch_size = x.shape[0]
        x = x.float()

        features = []
        for ch in range(x.shape[1]):
            feat = self.features(x[:, ch, :])
            features.append(feat)

        x = torch.cat(features, dim=1)
        out = self.classifier(x)
        return out
