#!/usr/bin/env python3
"""Export OSNet x0.25 to ONNX for DeepStream SGIE.

Standalone script — no torchreid installation required.
Architecture copied directly from KaiyangZhou/deep-person-reid (MIT license).

Dependencies (venv with torch + gdown is sufficient):
    pip install torch gdown onnxruntime

Usage:
    source ~/deep-person-reid/.venv/bin/activate
    python3 scripts/export_osnet_onnx.py

    # Or with explicit paths:
    python3 scripts/export_osnet_onnx.py \
        --weights /data/sentinel/models/osnet/osnet_x0_25_imagenet.pth \
        --output  /data/sentinel/models/osnet/osnet_x0_25.onnx

Output: [1, 512] — 512-dim appearance embedding per person crop.
Input:  [1, 3, 256, 128] — RGB, H=256, W=128, normalized to [0,1].
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# OSNet x0.25 architecture
# Source: github.com/KaiyangZhou/deep-person-reid (MIT license)
# ---------------------------------------------------------------------------

class ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, ks, stride=1, padding=0, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, ks, stride=stride,
                              padding=padding, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1x1(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1, stride=stride,
                              padding=0, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1x1Linear(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1, stride=stride,
                              padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return self.bn(self.conv(x))


class LightConv3x3(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1,
                               bias=False, groups=out_ch)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv2(self.conv1(x))))


class ChannelGate(nn.Module):
    def __init__(self, in_ch, num_gates=None, reduction=16):
        super().__init__()
        if num_gates is None:
            num_gates = in_ch
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_ch, in_ch // reduction, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_ch // reduction, num_gates, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        g = self.sigmoid(self.fc2(self.relu(self.fc1(self.pool(x)))))
        return x * g


class OSBlock(nn.Module):
    def __init__(self, in_ch, out_ch, bottleneck_reduction=4, **kwargs):
        super().__init__()
        mid = out_ch // bottleneck_reduction
        self.conv1 = Conv1x1(in_ch, mid)
        self.conv2a = LightConv3x3(mid, mid)
        self.conv2b = nn.Sequential(LightConv3x3(mid, mid),
                                    LightConv3x3(mid, mid))
        self.conv2c = nn.Sequential(LightConv3x3(mid, mid),
                                    LightConv3x3(mid, mid),
                                    LightConv3x3(mid, mid))
        self.conv2d = nn.Sequential(LightConv3x3(mid, mid),
                                    LightConv3x3(mid, mid),
                                    LightConv3x3(mid, mid),
                                    LightConv3x3(mid, mid))
        self.gate = ChannelGate(mid)
        self.conv3 = Conv1x1Linear(mid, out_ch)
        self.downsample = (Conv1x1Linear(in_ch, out_ch)
                           if in_ch != out_ch else None)

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2 = (self.gate(self.conv2a(x1)) + self.gate(self.conv2b(x1)) +
               self.gate(self.conv2c(x1)) + self.gate(self.conv2d(x1)))
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        return F.relu(x3 + identity)


class OSNet(nn.Module):
    def __init__(self, num_classes, blocks, layers, channels, feature_dim=512):
        super().__init__()
        self.feature_dim = feature_dim
        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(blocks[0], layers[0],
                                      channels[0], channels[1], reduce=True)
        self.conv3 = self._make_layer(blocks[1], layers[1],
                                      channels[1], channels[2], reduce=True)
        self.conv4 = self._make_layer(blocks[2], layers[2],
                                      channels[2], channels[3], reduce=False)
        self.conv5 = Conv1x1(channels[3], channels[3])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels[3], feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(feature_dim, num_classes)
        self._init_params()

    def _make_layer(self, block, num, in_ch, out_ch, reduce):
        layers = [block(in_ch, out_ch)]
        for _ in range(1, num):
            layers.append(block(out_ch, out_ch))
        if reduce:
            layers += [Conv1x1(out_ch, out_ch), nn.AvgPool2d(2, stride=2)]
        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        v = self.pool(x).view(x.size(0), -1)
        v = self.fc(v)
        return v  # 512-dim embedding (inference mode)


def _osnet_x0_25(num_classes=1000):
    return OSNet(
        num_classes,
        blocks=[OSBlock, OSBlock, OSBlock],
        layers=[2, 2, 2],
        channels=[16, 64, 96, 128],
    )


# ---------------------------------------------------------------------------
# Weight download + loading
# ---------------------------------------------------------------------------
GDRIVE_ID = "1rb8UN5ZzPKRc_xvtHlyDh-cSz88YX9hs"
HF_URL = ("https://huggingface.co/kaiyangzhou/osnet/resolve/main/"
          "osnet_x0_25_imagenet.pth")


def _download_weights(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading OSNet x0.25 ImageNet weights -> {dest}")
    try:
        import gdown
        gdown.download(id=GDRIVE_ID, output=str(dest), quiet=False)
        if dest.exists() and dest.stat().st_size > 1_000_000:
            print("Downloaded via gdown.")
            return
    except Exception as e:
        print(f"gdown failed ({e}), trying HuggingFace...")
    import urllib.request
    urllib.request.urlretrieve(HF_URL, str(dest))
    print("Downloaded from HuggingFace.")


def _load_weights(model: nn.Module, weights_path: Path) -> None:
    state = torch.load(weights_path, map_location="cpu")
    model_dict = model.state_dict()
    new_state = OrderedDict()
    matched, skipped = [], []
    for k, v in state.items():
        k = k.replace("module.", "")
        if k in model_dict and model_dict[k].shape == v.shape:
            new_state[k] = v
            matched.append(k)
        else:
            skipped.append(k)
    model_dict.update(new_state)
    model.load_state_dict(model_dict)
    print(f"Weights: {len(matched)} matched, {len(skipped)} skipped.")
    if skipped:
        print(f"  Skipped: {skipped[:5]}")


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def export(weights_path: Path, output_path: Path) -> None:
    if not weights_path.exists():
        _download_weights(weights_path)

    model = _osnet_x0_25(num_classes=1000)
    _load_weights(model, weights_path)
    model.eval()

    dummy = torch.zeros(1, 3, 256, 128)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        opset_version=17,
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
    )
    print(f"Exported: {output_path}")

    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(output_path),
                                    providers=["CPUExecutionProvider"])
        out = sess.run(None, {"input": dummy.numpy()})[0]
        print(f"Sanity check — output shape: {out.shape}")
        assert out.shape == (1, 512), f"Unexpected shape: {out.shape}"
        print("OK — 512-dim embeddings confirmed.")
    except ImportError:
        print("onnxruntime not installed, skipping sanity check.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export OSNet x0.25 to ONNX (no torchreid required)")
    ap.add_argument(
        "--weights",
        default="/data/sentinel/models/osnet/osnet_x0_25_imagenet.pth",
        help="Path to .pth weights file. Downloaded automatically if absent.",
    )
    ap.add_argument(
        "--output",
        default="/data/sentinel/models/osnet/osnet_x0_25.onnx",
        help="Output ONNX file path.",
    )
    args = ap.parse_args()
    export(Path(args.weights), Path(args.output))


if __name__ == "__main__":
    main()
