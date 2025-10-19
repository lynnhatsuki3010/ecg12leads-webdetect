import numpy as np
import scipy.io
from scipy.signal import butter, filtfilt, iirnotch
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import io, base64, json, sys
from model import ResNet1DAttention  # phải đảm bảo file model.py có class này
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8') # in chữ tiếng việt

# Load model
def load_model(model_path="resnet_attention_resampling_posweight1.pth"):
    model = ResNet1DAttention(num_classes=8)  # sửa num_classes đúng theo khi train
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()


# xử lý tín hiệu ECG
def load_ecg_mat(mat_file):
    """Đọc file .mat chứa tín hiệu ECG"""
    mat = scipy.io.loadmat(mat_file)
    return mat["val"].astype(float)  # (12, N)


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return b, a


def apply_bandpass(sig, fs, lowcut=0.5, highcut=40.0, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return filtfilt(b, a, sig)


def apply_notch(sig, fs, notch_freq=50.0, quality=30.0):
    b, a = iirnotch(notch_freq / (fs / 2), quality)
    return filtfilt(b, a, sig)


def denoise_all_leads(signals, fs=500):
    """Lọc nhiễu toàn bộ 12 đạo trình"""
    denoised = np.zeros_like(signals, dtype=float)
    for i in range(signals.shape[0]):
        x = apply_notch(signals[i, :], fs)
        x = apply_bandpass(x, fs)
        denoised[i, :] = x
    return denoised

def plot_ecg_12leads(signals, fs=500):
    """Vẽ 12 đạo trình ECG riêng lẻ và trả về danh sách ảnh base64"""
    leads = [
        "I", "II", "III", "aVR", "aVL", "aVF",
        "V1", "V2", "V3", "V4", "V5", "V6"
    ]
    images = []
    t = np.arange(signals.shape[1]) / fs

    for i in range(12):
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.plot(t, signals[i, :], linewidth=0.8)
        ax.set_title(leads[i])
        ax.set_ylabel("mV")
        ax.set_xlabel("Time (s)")
        ax.grid(True, linestyle="--", linewidth=0.5)

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png", dpi=120)
        plt.close(fig)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")
        images.append(img_b64)

    return images




def preprocess_ecg(mat_file, fs=500):
    """Tiền xử lý ECG đầu vào"""
    signals = load_ecg_mat(mat_file)
    raw_mV = signals / 1000.0
    denoised = denoise_all_leads(raw_mV, fs)
    z = (denoised - denoised.mean(axis=1, keepdims=True)) / (
        denoised.std(axis=1, keepdims=True) + 1e-8
    )
    z = np.clip(z, -10, 10)
    return z  # (12, N)


# dự đoán
def detect_ecg(mat_file):
    """Trả về xác suất từng nhãn + hình ảnh ECG sau lọc nhiễu"""
    x = preprocess_ecg(mat_file)
    images_b64 = plot_ecg_12leads(x)  # vẽ và encode ảnh

    x_input = np.expand_dims(x, axis=0)
    x_tensor = torch.tensor(x_input, dtype=torch.float32)

    with torch.no_grad():
        output = model(x_tensor)
        probs = torch.sigmoid(output).cpu().numpy().flatten()

    # === dùng ngưỡng tối ưu ===
    best_thresholds = np.array([0.5, 0.75, 0.75, 0.65, 0.8, 0.6, 0.9, 0.9])

    # === ÁP DỤNG NGƯỠNG CHO XÁC SUẤT ===
    # Nếu xác suất < threshold, giảm nhẹ confidence; nếu > threshold, tăng nhẹ
    adjusted_probs = np.clip((probs - best_thresholds + 0.5), 0, 1)

    labels = [
        "426177001", "426783006", "164890007", "427084000",
        "427393009", "164889003", "429622005", "39732003"
    ]
    labels = labels[:len(probs)]

    snomed_to_name = {
        "426177001": "Nhịp chậm xoang",
        "426783006": "Nhịp xoang bình thường",
        "164890007": "Cuồng nhĩ",
        "427084000": "Chênh lên đoạn ST",
        "427393009": "Loạn nhịp xoang",
        "164889003": "Rung nhĩ",
        "429622005": "Chênh xuống đoạn ST",
        "39732003": "Trục điện tim lệch trái"
    }

    results = {
        snomed_to_name.get(labels[i], labels[i]): float(adjusted_probs[i])
        for i in range(len(labels))
    }

    return {
        "results": results,
        "images": images_b64
    }






import sys, json

if __name__ == "__main__":
    try:
        mat_file = sys.argv[1]
        output = detect_ecg(mat_file)
        print(json.dumps(output, ensure_ascii=False)) #TV
    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False))

