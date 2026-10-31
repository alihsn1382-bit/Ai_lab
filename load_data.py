import os
import numpy as np
import librosa

# تنظیمات اولیه
DATASET_PATH = "dataset"  # مسیر پوشه دیتاست
SAMPLE_RATE = 16000       # نرخ نمونه‌برداری (برای این دیتاست معمولاً 16kHz است)
DURATION = 1              # طول هر کلیپ صوتی (1 ثانیه)
N_MFCC = 13               # تعداد ضرایب MFCC که استخراج می‌کنیم

def extract_features(file_path):
    """
    این تابع یک فایل صوتی را می‌خواند و ویژگی MFCC آن را استخراج می‌کند.
    """
    try:
        # خواندن فایل صوتی
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        
        # استخراج ویژگی MFCC
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        
        # میانگین‌گیری از ویژگی‌ها در طول زمان (برای اینکه ابعاد ثابت شود)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled
    except Exception as e:
        print(f"خطا در خواندن فایل {file_path}: {e}")
        return None

def load_dataset():
    """
    این تابع کل پوشه دیتاست را می‌خواند و ویژگی‌ها و برچسب‌ها را برمی‌گرداند.
    """
    X = [] # ویژگی‌ها
    y = [] # برچسب‌ها (اسم کلاس‌ها)

    # پیمایش پوشه‌ها
    for label in os.listdir(DATASET_PATH):
        label_path = os.path.join(DATASET_PATH, label)
        
        # اگر آیتم یک پوشه بود (نه فایل)
        if os.path.isdir(label_path):
            print(f"در حال پردازش دستور: {label}")
            for file_name in os.listdir(label_path):
                if file_name.endswith('.wav'):
                    file_path = os.path.join(label_path, file_name)
                    features = extract_features(file_path)
                    if features is not None:
                        X.append(features)
                        y.append(label)

    # تبدیل به آرایه‌های NumPy
    X = np.array(X)
    y = np.array(y)
    
    print(f"تعداد کل نمونه‌ها: {len(X)}")
    print(f"ابعاد ویژگی‌ها: {X.shape}")
    
    return X, y

if name == "__main__":
    # اجرای تابع
    X, y = load_dataset()
    
    # ذخیره داده‌ها برای استفاده در گام‌های بعدی
    np.save("X_features.npy", X)
    np.save("y_labels.npy", y)
    print("داده‌ها با موفقیت در فایل‌های X_features.npy و y_labels.npy ذخیره شدند.")