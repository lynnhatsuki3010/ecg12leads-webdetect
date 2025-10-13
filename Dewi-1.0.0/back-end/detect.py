import numpy as np
import scipy.io
from scipy.signal import butter, filtfilt, iirnotch
import torch
import torch.nn.functional as F
from model import ResNet1DAttention  # phải đảm bảo file model.py có class này

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
    """Trả về xác suất từng nhãn"""
    x = preprocess_ecg(mat_file)  # (12, N)
    x = np.expand_dims(x, axis=0)  # (1, 12, N)
    x_tensor = torch.tensor(x, dtype=torch.float32)

    with torch.no_grad():
        output = model(x_tensor)  # (1, num_classes)
        probs = torch.sigmoid(output).cpu().numpy().flatten()

    # Danh sách nhãn 
    labels = [
        "426177001",
        "426783006",
        "164890007",
        "427084000",
        "427393009",
        "164889003",
        "429622005",
        "39732003",
    ]

    # Nếu num_classes < len(labels) → cắt lại
    labels = labels[: len(probs)]

    results = {labels[i]: float(probs[i]) for i in range(len(labels))}
    return results


import sys, json

if __name__ == "__main__":
    try:
        mat_file = sys.argv[1]
        results = detect_ecg(mat_file)
        print(json.dumps(results, ensure_ascii=False))  # xuất JSON hợp lệ
    except Exception as e:
        print(json.dumps({"error": f"Lỗi phân tích model: {str(e)}"}, ensure_ascii=False))
