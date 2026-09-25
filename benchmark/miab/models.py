"""W3：模型定义（benchmark 唯一定义来源）。

- mlp_512_256_128_64：RMIA 论文 Appendix A 的 Purchase-100 规格（§5.1）
- mnist_cnn：方案 §5.2 规格
- resnet18_cifar：CIFAR 风格 ResNet18（3×3 stem、无 maxpool），
  收编自 Attack/utils_secmia/mia_evals/resnet.py（与其两份历史拷贝内容一致）
- resnet18_cifar_gn：GroupNorm 变体（DP-SGD 在 CIFAR 上使用，§10.4）
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """4 隐层 MLP，层宽 [512, 256, 128, 64]，ReLU。"""

    def __init__(self, input_dim: int, num_classes: int, hidden: List[int] = None):
        super().__init__()
        hidden = hidden or [512, 256, 128, 64]
        dims = [input_dim] + hidden + [num_classes]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MnistCNN(nn.Module):
    """方案 §5.2：Conv(1→32) → Conv(32→64) → MaxPool → FC128 → FC10。"""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, 1, stride=stride, bias=False),
                norm_layer(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet(nn.Module):
    """CIFAR 风格 ResNet：3×3 stem、无 maxpool、自适应池化。"""

    def __init__(self, block, num_blocks, num_channels=3, num_classes=10, norm="batch"):
        super().__init__()
        norm_layer = nn.BatchNorm2d if norm == "batch" else _GroupNormFactory()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(num_channels, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], 1, norm_layer)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], 2, norm_layer)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], 2, norm_layer)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], 2, norm_layer)
        self.classifier = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride, norm_layer):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s, norm_layer))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = torch.flatten(out, 1)
        return self.classifier(out)


class _GroupNormFactory:
    """DP-SGD 用：BatchNorm → GroupNorm(32)（宽 64..512 均可整除）。"""

    def __init__(self, num_groups: int = 32):
        self.num_groups = num_groups

    def __call__(self, num_channels: int) -> nn.GroupNorm:
        return nn.GroupNorm(self.num_groups, num_channels)


def resnet18_cifar(num_classes: int = 10, num_channels: int = 3, norm: str = "batch") -> ResNet:
    return ResNet(BasicBlock, [2, 2, 2, 2], num_channels=num_channels, num_classes=num_classes, norm=norm)


def build_model(arch: str, input_dim: int, num_classes: int) -> nn.Module:
    if arch == "mlp_512_256_128_64":
        return MLP(input_dim, num_classes)
    if arch == "mnist_cnn":
        return MnistCNN(num_classes)
    if arch == "resnet18_cifar":
        return resnet18_cifar(num_classes=num_classes, norm="batch")
    if arch == "resnet18_cifar_gn":
        return resnet18_cifar(num_classes=num_classes, norm="group")
    raise ValueError(f"unknown arch: {arch}")
