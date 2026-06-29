import numpy as np
import cv2
import tensorflow as tf
import keras
import os

# ==========================================
# 🛠️ TUYỆT CHIÊU MONKEY PATCHING FIX LỖI KERAS 3
# Chặn đứng lỗi 'quantization_config' mà không cần giải nén model
# ==========================================
_original_dense_init = keras.layers.Dense.__init__
_original_conv2d_init = keras.layers.Conv2D.__init__

def _patched_dense_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None) # Xóa tham số rác nếu có
    _original_dense_init(self, *args, **kwargs)

def _patched_conv2d_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None) # Xóa tham số rác nếu có
    _original_conv2d_init(self, *args, **kwargs)

# Áp dụng bản vá vào hệ thống Keras
keras.layers.Dense.__init__ = _patched_dense_init
keras.layers.Conv2D.__init__ = _patched_conv2d_init
# ==========================================

# 1. Khai báo các custom objects
@keras.saving.register_keras_serializable(package="custom")
def _reduce_max_tc_fn(x):
    return tf.reduce_max(x, axis=-1, keepdims=True)

@keras.saving.register_keras_serializable(package="custom")
def _reduce_max_et_fn(x):
    return tf.reduce_max(x, axis=-1, keepdims=True)

# --- CHẠY TEST ---
print("🚀 Đang khởi động script test logic Classification (Đã vá lỗi Keras 3)...")

MODEL_PATH = r"D:\AIModel\fold_1_best.keras" 
IMAGE_PATH = r"D:\medicalSystem\ServerBackend\src\mri_uploads\brisc2025_train_01152_me_ax_t1.jpg"

try:
    print(f"🧠 Đang nạp model từ: {MODEL_PATH}")
    custom_objs = {
        '_reduce_max_tc_fn': _reduce_max_tc_fn,
        '_reduce_max_et_fn': _reduce_max_et_fn
    }
    
    # Nạp model an toàn
    model = keras.models.load_model(
        MODEL_PATH, 
        custom_objects=custom_objs, 
        compile=False,      
        safe_mode=False     
    )
    print("✅ Load model thành công! Lỗi 'quantization_config' đã bị tiêu diệt.")
    
    # 3. Đọc ảnh và tiền xử lý cơ bản cho Xg
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise ValueError(f"Không đọc được ảnh từ {IMAGE_PATH}, check lại đường dẫn!")
        
    img_resized = cv2.resize(img, (224, 224))
    img_norm = (img_resized - img_resized.mean()) / (img_resized.std() + 1e-8)
    Xg = np.expand_dims(img_norm, axis=0).astype('float32')

    # 4. MOCKING DATA (Làm giả 4 inputs)
    Xl = np.copy(Xg) 
    Xtc = np.zeros((1, 224, 224, 3), dtype='float32')
    Xet = np.zeros((1, 224, 224, 3), dtype='float32')
    Sf = np.zeros((1, 11), dtype='float32')

    # 5. TEST PREDICT
    print("⚙️ Bắt đầu đưa 5 luồng input vào model...")
    preds = model.predict([Xg, Xl, Xtc, Xet, Sf], verbose=0)
    
    CLASS_NAMES = {0: "Glioma", 1: "Meningioma", 2: "Pituitary"}
    best_idx = int(np.argmax(preds[0]))
    confidence = preds[0][best_idx] * 100
    
    print(f"\n🎯 TEST THÀNH CÔNG TỐT ĐẸP!")
    print(f"📊 Kết quả dự đoán (dùng mask giả): {CLASS_NAMES[best_idx]} ({confidence:.2f}%)")

except Exception as e:
    print(f"\n❌ TOANG! Cần kiểm tra lại:")
    import traceback
    traceback.print_exc()