import torch
import torch.nn as nn
import torch.nn.functional as F

# === Attention Layer ===
class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len, feat_dim)
        weights = self.attn(x)  # (batch, seq_len, 1)
        weights = F.softmax(weights, dim=1)
        context = torch.sum(weights * x, dim=1)  # (batch, feat_dim)
        return context, weights


# === ResNet 1D Block ===
class BasicBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels,
                                      kernel_size=1, stride=stride)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# === ResNet 1D + Attention ===
class ResNet1DAttention(nn.Module):
    def __init__(self, num_classes=8):
        super(ResNet1DAttention, self).__init__()
        self.layer1 = BasicBlock1D(12, 64, stride=2)
        self.layer2 = BasicBlock1D(64, 128, stride=2)
        self.layer3 = BasicBlock1D(128, 128, stride=2)

        self.attention = Attention(128)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = x.permute(0, 2, 1)  # (batch, seq_len, feat_dim=128)
        x, _ = self.attention(x)

        out = self.fc(x)
        return out
