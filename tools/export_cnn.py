"""Export the lightweight CNN-BiLSTM-Attention model (best speaker-independent fold) to ONNX for free-tier hosting.

Usage (anywhere with torch + torchaudio; Kaggle or this PC):
    python export_cnn.py best_p2_CNN-BiLSTM-Attn_full.pt  out_dir
Writes out_dir/model.onnx and out_dir/labels.json (drop both into backend/model/).
The architecture below is copied verbatim from the training notebook so the state dict loads exactly.
"""
import sys, os, json, math
import torch, torch.nn as nn, torch.nn.functional as F, torchaudio

SR, N_SAMPLES, TOP_DB = 16_000, 48_000, 30
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
N_CLASSES = len(EMOTIONS)
LR_A = 2e-3   # only used by param_groups(), irrelevant for export

def utt_norm(x):
    """Per-utterance zero-mean / unit-variance normalisation (also removes mic-gain differences)."""
    x = x - x.mean(dim=1, keepdim=True)
    return x / torch.sqrt((x ** 2).mean(dim=1, keepdim=True) + 1e-7)

class MelFrontEnd(nn.Module):
    """Log-mel via fixed conv1d-STFT (cos/sin kernels * Hann window). Pure conv/matmul ops => exports cleanly to ONNX."""
    def __init__(self, sr=SR, n_fft=1024, hop=256, n_mels=64, f_min=20., f_max=8000.):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        win = torch.hann_window(n_fft)
        k = torch.arange(n_fft // 2 + 1).unsqueeze(1).float(); n = torch.arange(n_fft).unsqueeze(0).float()
        ang = 2 * math.pi * k * n / n_fft
        self.register_buffer("cos_k", (torch.cos(ang) * win).unsqueeze(1))      # (F,1,n_fft)
        self.register_buffer("sin_k", (-torch.sin(ang) * win).unsqueeze(1))
        fb = torchaudio.functional.melscale_fbanks(n_fft // 2 + 1, f_min, f_max, n_mels, sr, norm="slaney", mel_scale="slaney")
        self.register_buffer("mel_fb", fb)                                      # (F, n_mels)
    def forward(self, x):                                                       # x: (B,T)
        x = F.pad(x.unsqueeze(1), (self.n_fft // 2, self.n_fft // 2), mode="reflect")
        re_, im_ = F.conv1d(x, self.cos_k, stride=self.hop), F.conv1d(x, self.sin_k, stride=self.hop)
        power = re_ ** 2 + im_ ** 2                                             # (B,F,frames)
        mel = torch.matmul(power.transpose(1, 2), self.mel_fb).transpose(1, 2)  # (B,n_mels,frames)
        lm = torch.log(mel + 1e-6)
        lm = (lm - lm.mean(dim=(1, 2), keepdim=True)) / torch.sqrt(lm.var(dim=(1, 2), keepdim=True, unbiased=False) + 1e-5)
        return lm.unsqueeze(1)                                                  # (B,1,n_mels,frames)

class CNNBiLSTMAttention(nn.Module):
    def __init__(self, n_classes=N_CLASSES, n_mels=64, hidden=128, dropout=0.3):
        super().__init__()
        self.frontend = MelFrontEnd(n_mels=n_mels)
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=10, iid_masks=True)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=20, iid_masks=True)
        def block(cin, cout, pool):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                                 nn.MaxPool2d(pool), nn.Dropout2d(0.1))
        self.cnn = nn.Sequential(block(1, 32, (2, 2)), block(32, 64, (2, 2)), block(64, 128, (2, 1)), block(128, 128, (2, 1)))
        self.lstm = nn.LSTM(128 * (n_mels // 16), hidden, batch_first=True, bidirectional=True)
        self.att = nn.Sequential(nn.Linear(2 * hidden, 128), nn.Tanh(), nn.Linear(128, 1))   # additive attention
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, n_classes))
    def forward(self, wav):
        x = self.frontend(utt_norm(wav))                         # (B,1,M,T)
        if self.training:                                        # SpecAugment (train only)
            x = self.time_mask(self.freq_mask(x))
        x = self.cnn(x)                                          # (B,C,M/16,T/4)
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)           # (B,T',C*F)
        h, _ = self.lstm(x)                                      # (B,T',2H)
        w = torch.softmax(self.att(h), dim=1)                    # attention over time
        return self.head((w * h).sum(dim=1))
    def param_groups(self):
        return [{"params": [p for p in self.parameters() if p.requires_grad], "lr": LR_A}]



if __name__ == "__main__":
    ckpt, out = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "."
    os.makedirs(out, exist_ok=True)
    m = CNNBiLSTMAttention(); m.load_state_dict(torch.load(ckpt, map_location="cpu")); m.eval()
    dummy = torch.randn(1, N_SAMPLES)
    kw = dict(input_names=["waveform"], output_names=["logits"], dynamic_axes={"waveform": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17)
    try:
        torch.onnx.export(m, (dummy,), os.path.join(out, "model.onnx"), dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(m, (dummy,), os.path.join(out, "model.onnx"), **kw)
    json.dump({"labels": EMOTIONS, "sample_rate": SR, "window_samples": N_SAMPLES, "top_db": TOP_DB,
               "architecture": "CNN-BiLSTM-Attn"}, open(os.path.join(out, "labels.json"), "w"), indent=2)
    print("exported ->", out, f"({os.path.getsize(os.path.join(out, 'model.onnx'))/1e6:.1f} MB)")
