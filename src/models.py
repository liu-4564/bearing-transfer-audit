"""1D convolutional models: a shallow WDCNN-style net and a 1D ResNet. Input (B, 1, L)."""
import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """WDCNN style: one wide first-stage kernel, then a stack of narrow ones.

    Adaptive pooling keeps any input length valid.
    """

    def __init__(self, n_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 64, stride=8, padding=32), nn.BatchNorm1d(16), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(256, 64), nn.ReLU(),
                                  nn.Dropout(0.3), nn.Linear(64, n_classes))

    def forward(self, x):
        return self.head(self.net(x))


class ResBlock1D(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(cin, cout, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(cout), nn.ReLU(),
            nn.Conv1d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm1d(cout))
        self.short = (nn.Sequential(nn.Conv1d(cin, cout, 1, stride=stride, bias=False),
                                    nn.BatchNorm1d(cout))
                      if stride != 1 or cin != cout else nn.Identity())
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.conv(x) + self.short(x))


class ResNet1D(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(1, 32, 7, stride=2, padding=3, bias=False),
                                  nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2))
        self.layers = nn.Sequential(
            ResBlock1D(32, 32), ResBlock1D(32, 64, stride=2),
            ResBlock1D(64, 64), ResBlock1D(64, 128, stride=2),
            ResBlock1D(128, 128), nn.AdaptiveAvgPool1d(1))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, n_classes))

    def forward(self, x):
        return self.head(self.layers(self.stem(x)))
