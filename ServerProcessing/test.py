import os
import cv2
import json
import zipfile
import tempfile
import numpy as np
import tensorflow as tf
import keras
from keras.layers import Layer

# ==========================================
# 🛠️ 1. MONKEY PATCHING (VÁ LỖI KERAS 3)
# ==========================================
_original_dense_init = keras.layers.Dense.__init__
_original_conv2d_init = keras.layers.Conv2D.__init__

def _patched_dense_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None)
    _original_dense_init(self, *args, **kwargs)

def _patched_conv2d_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None)
    _original_conv2d_init(self, *args, **kwargs)

keras.layers.Dense.__init__ = _patched_dense_init
keras.layers.Conv2D.__init__ = _patched_conv2d_init

# ==========================================
# 🧠 2. CUSTOM OBJECTS (CHO CẢ SEG & CLASS)
# ==========================================
@keras.saving.register_keras_serializable() 
class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([keras.layers.Dense(ff_dim, activation="gelu"), keras.layers.Dense(embed_dim)])
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)

    def build(self, input_shape):
        tensor_shape = tuple(1 if d is None else d for d in input_shape)
        dummy = tf.zeros(tensor_shape)
        self.call(dummy)
        super().build(input_shape)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "num_heads": self.num_heads, "ff_dim": self.ff_dim, "rate": self.rate})
        return config

@keras.saving.register_keras_serializable(package="custom")
def _reduce_max_tc_fn(x): return tf.reduce_max(x, axis=-1, keepdims=True)

@keras.saving.register_keras_serializable(package="custom")
def _reduce_max_et_fn(x): return tf.reduce_max(x, axis=-1, keepdims=True)

# ==========================================
# 📦 3. LOGIC LOAD MODELS
# ==========================================
def load_seg_model(model_path):
    print("📦 Đang trích xuất Topology cho UNETR (Segmentation)...")
    with zipfile.ZipFile(model_path, 'r') as archive:
        config_dict = json.loads(archive.read('config.json').decode('utf-8'))
        if 'compile_config' in config_dict: del config_dict['compile_config']
        model = keras.saving.deserialize_keras_object(config_dict, custom_objects={'TransformerBlock': TransformerBlock})
        temp_dir = tempfile.gettempdir()
        archive.extract('model.weights.h5', path=temp_dir)
        weights_path = os.path.join(temp_dir, 'model.weights.h5')
        model.load_weights(weights_path, skip_mismatch=True)
        try: os.remove(weights_path)
        except: pass
        return model

def load_cls_model(model_path):
    print("🧠 Đang nạp ConvNeXt (Classification)...")
    return keras.models.load_model(
        model_path, 
        custom_objects={'_reduce_max_tc_fn': _reduce_max_tc_fn, '_reduce_max_et_fn': _reduce_max_et_fn}, 
        compile=False, safe_mode=False
    )

# ==========================================
# 🧮 4. FEATURE EXTRACTION HÌNH HỌC
# ==========================================
def preprocess_clahe(img):
    img = img.astype('uint8') if img.dtype != 'uint8' else img
    img = cv2.GaussianBlur(img, (11,11), 0)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l)
    return cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2BGR)

def normalize(img):
    mask = img > 0
    if mask.sum() > 0: return (img - img[mask].mean()) / (img[mask].std() + 1e-8)
    return img

def _wt_bbox(mask, padding, shape):
    c = np.where(mask > 0.5)
    if len(c[0]) == 0: return None
    return (max(0,c[0].min()-padding), min(shape[0],c[0].max()+padding),
            max(0,c[1].min()-padding), min(shape[1],c[1].max()+padding))

def _compactness(m):
    u8 = (m>0.5).astype('uint8')*255
    cnts,_ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts: return 0.
    c = max(cnts, key=cv2.contourArea)
    P = cv2.arcLength(c, True)
    return float(4*np.pi*cv2.contourArea(c)/(P*P+1e-6))

def _solidity(m):
    u8 = (m>0.5).astype('uint8')*255
    cnts,_ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts: return 0.
    c = max(cnts, key=cv2.contourArea)
    return float(cv2.contourArea(c)/(cv2.contourArea(cv2.convexHull(c))+1e-6))

def _centroid(m, shape):
    c = np.where(m > 0.5)
    if len(c[0]) == 0: return 0.
    H,W = shape[:2]
    return float(np.sqrt(((c[0].mean()-H/2)/H)**2 + ((c[1].mean()-W/2)/W)**2))

def compute_seg_features(wt, tc, et, shape, image):
    eps, E = 1e-6, 50
    tu = tc if tc.sum() > E else wt
    eu = et if et.sum() > E else wt
    aw = float(wt.sum()) + eps
    
    f = [float(tu.sum())/aw, float(eu.sum())/(float(tu.sum())+eps),
         float(eu.sum())/aw, _centroid(wt, shape), _compactness(tu), _solidity(eu)]
         
    g = cv2.cvtColor(image.astype('float32'), cv2.COLOR_BGR2GRAY) if image.ndim==3 else image.astype('float32')
    tm, em = (tu>0.5), (eu>0.5)
    mt = float(g[tm].mean()) if tm.any() else 0.
    me = float(g[em].mean()) if em.any() else 0.
    st = float(g[tm].std())  if tm.any() else 0.
    se = float(g[em].std())  if em.any() else 0.
    
    return np.array(f + [mt, me, st, se, float(me/(abs(mt)+eps))], 'float32')


# ==========================================
# 🚀 5. MAIN PIPELINE E2E TEST
# ==========================================
if __name__ == "__main__":
    print("="*50)
    print(" BẮT ĐẦU TEST TOÀN TRÌNH (PREPROCESS -> SEG -> CLS)")
    print("="*50)

    # Đường dẫn file
    SEG_MODEL_PATH = r"D:\AIModel\unet_wt_tc_et_best.keras"
    CLS_MODEL_PATH = r"D:\AIModel\fold_1_best.keras"
    IMAGE_PATH = r"D:\medicalSystem\ServerBackend\src\mri_uploads\brisc2025_train_01152_me_ax_t1.jpg"

    # Load 2 Models
    model_seg = load_seg_model(SEG_MODEL_PATH)
    model_cls = load_cls_model(CLS_MODEL_PATH)
    
    print("\n[1/3] CHẠY GIẢ LẬP PREPROCESS (Phòng 1)...")
    img_raw = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    img_256 = cv2.resize(img_raw, (256, 256))
    
    # Chập 3 kênh để giả lập FLAIR, T1ce, T2. Giữ nguyên uint8 để tí nữa còn chạy CLAHE.
    img_3c_uint8 = np.stack([img_256, img_256, img_256], axis=-1)
    
    # Normalization cho Segmentation (Nó cần float32)
    img_3c_norm = normalize(img_3c_uint8.astype(np.float32))
    input_seg = np.expand_dims(img_3c_norm, axis=0) # Shape: (1, 256, 256, 3)

    print("\n[2/3] CHẠY SEGMENTATION (Phòng 2)...")
    preds_seg = model_seg.predict(input_seg, verbose=0)
    
    # Cắt lấy mask của lát cắt đầu tiên (và duy nhất)
    wt_mask = (preds_seg[0, :, :, 0] > 0.5).astype(np.float32)
    tc_mask = (preds_seg[0, :, :, 1] > 0.5).astype(np.float32)
    et_mask = (preds_seg[0, :, :, 2] > 0.5).astype(np.float32)
    
    print(f"  -> Đã tìm thấy u! Kích thước WT: {np.sum(wt_mask)} pixels.")

    print("\n[3/3] CHẠY CLASSIFICATION (Phòng 3)...")
    # Fix triệt để lỗi ép kiểu: Truyền ảnh uint8 gốc vào CLAHE
    img_g_clahe = preprocess_clahe(img_3c_uint8)
    img_g_clahe_224 = cv2.resize(img_g_clahe, (224, 224))
    Xg = np.expand_dims(normalize(img_g_clahe_224.astype('float32')), axis=0)

    # Xử lý Local Crop (Xl) và Mask từ ảnh Float32 đã normalize
    bbox = _wt_bbox(wt_mask, 25, img_3c_norm.shape)
    if bbox:
        y0, y1, x0, x1 = bbox
        xl = cv2.resize(img_3c_norm[y0:y1, x0:x1], (224, 224))
        tc_c = cv2.resize(tc_mask[y0:y1, x0:x1], (224, 224), interpolation=cv2.INTER_NEAREST)
        et_c = cv2.resize(et_mask[y0:y1, x0:x1], (224, 224), interpolation=cv2.INTER_NEAREST)
    else:
        xl = cv2.resize(img_3c_norm, (224, 224))
        tc_c = np.zeros((224, 224), 'float32')
        et_c = np.zeros((224, 224), 'float32')
        
    Xl = np.expand_dims(normalize(xl.astype('float32')), axis=0)
    Xtc = np.expand_dims(np.stack([(tc_c>0.5).astype('float32')*255.]*3, -1), axis=0)
    Xet = np.expand_dims(np.stack([(et_c>0.5).astype('float32')*255.]*3, -1), axis=0)
    
    # Feature Extraction (Sf)
    Sf = np.expand_dims(compute_seg_features(wt_mask, tc_mask, et_mask, img_3c_norm.shape, image=img_3c_norm), axis=0)

    print("  -> Bơm 5 tensor dữ liệu vào mô hình ConvNeXt...")
    preds_cls = model_cls.predict([Xg, Xl, Xtc, Xet, Sf], verbose=0)
    
    CLASS_NAMES = {0: "Glioma", 1: "Meningioma", 2: "Pituitary"}
    best_idx = int(np.argmax(preds_cls[0]))
    confidence = preds_cls[0][best_idx] * 100
    
    print("\n" + "="*50)
    print(f" 🎯 CHỐT KẾT QUẢ CUỐI CÙNG: {CLASS_NAMES[best_idx]} (Độ tin cậy: {confidence:.2f}%)")
    print("="*50)
    print(f"Chi tiết phân bổ xác suất: Glioma ({preds_cls[0][0]:.3f}) | Meningioma ({preds_cls[0][1]:.3f}) | Pituitary ({preds_cls[0][2]:.3f})")