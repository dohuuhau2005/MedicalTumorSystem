import os
import cv2
import json
import zipfile
import tempfile
import numpy as np
import tensorflow as tf
import keras
from keras.layers import Layer
import matplotlib.pyplot as plt
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

def load_ensemble_models(base_model_dir):
    models = []
    print(f"🧠 Đang triệu tập hội đồng 4 chuyên gia AI Classification từ: {base_model_dir}...")
    custom_objs = {'_reduce_max_tc_fn': _reduce_max_tc_fn, '_reduce_max_et_fn': _reduce_max_et_fn}
    
    for i in range(1, 5): 
        model_path = os.path.join(base_model_dir, f"fold_{i}_best.keras")
        if os.path.exists(model_path):
            print(f"  -> Đang nạp Fold {i}...")
            m = keras.models.load_model(model_path, custom_objects=custom_objs, compile=False, safe_mode=False)
            models.append(m)
        else:
            print(f"  ⚠️ CẢNH BÁO: Không tìm thấy Fold {i} tại {model_path}")
    return models

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
# 🚀 5. MAIN PIPELINE E2E TEST (HỘI CHẨN 4 FOLD)
# ==========================================
if __name__ == "__main__":
    print("="*60)
    print(" BẮT ĐẦU TEST TOÀN TRÌNH: PREPROCESS -> SEG -> 4 FOLDS CLS")
    print("="*60)

    # Cấu hình đường dẫn
    SEG_MODEL_PATH = r"D:\AIModel\unet_wt_tc_et_best.keras"
    CLS_MODEL_DIR = r"D:\AIModel"
    IMAGE_PATH = r"D:\medicalSystem\ServerBackend\src\mri_uploads\brisc2025_train_03544_pi_ax_t1.jpg"

    # Load Models
    model_seg = load_seg_model(SEG_MODEL_PATH)
    models_cls = load_ensemble_models(CLS_MODEL_DIR)
    
    print("\n[1/3] CHẠY GIẢ LẬP PREPROCESS (Phòng 1)...")
    img_raw = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

    print("Raw shape:", img_raw.shape)
    print("dtype:", img_raw.dtype)
    print("min:", img_raw.min())
    print("max:", img_raw.max())

    plt.figure(figsize=(5,5))
    plt.imshow(img_raw, cmap="gray")
    plt.title("Original")
    plt.axis("off")
    plt.show()

    if img_raw is None:
        raise ValueError("Không tìm thấy ảnh, check lại tên file!")
        
    img_256 = cv2.resize(img_raw, (256, 256))
    print(img_256.shape)
    print(img_256.min(), img_256.max())

    plt.figure(figsize=(5,5))
    plt.imshow(img_256,cmap="gray")
    plt.title("Resize 256")
    plt.axis("off")
    plt.show()

    # Chập 3 kênh để giả lập FLAIR, T1ce, T2. Giữ nguyên uint8 cho CLAHE sau này.
    img_3c_uint8 = np.stack([img_256, img_256, img_256], axis=-1)
    
    # Normalization cho Segmentation
    img_3c_norm = normalize(img_3c_uint8.astype(np.float32))

    print(img_3c_norm.shape)
    print(img_3c_norm.min())
    print(img_3c_norm.max())
    print(img_3c_norm.mean())
    input_seg = np.expand_dims(img_3c_norm, axis=0) 

    show = img_3c_norm[:,:,0]

    show = (show-show.min())/(show.max()-show.min()+1e-8)

    plt.figure(figsize=(5,5))
    plt.imshow(show,cmap='gray')
    plt.title("Normalized")
    plt.axis("off")
    plt.show()


    print("\n[2/3] CHẠY SEGMENTATION (Phòng 2)...")
    print(input_seg.shape)
    print(input_seg.dtype)

    for c in range(3):
        print(
            "channel",
            c,
            input_seg[0,:,:,c].min(),
            input_seg[0,:,:,c].max(),
            input_seg[0,:,:,c].mean()
        )

    fig,ax=plt.subplots(1,3,figsize=(12,4))

    for i in range(3):

        img=input_seg[0,:,:,i]

        img=(img-img.min())/(img.max()-img.min()+1e-8)

        ax[i].imshow(img,cmap='gray')
        ax[i].set_title(f'Channel {i}')
        ax[i].axis('off')

    plt.show()
    preds_seg = model_seg.predict(input_seg, verbose=0)
    
    # Rút mask
    wt_mask = (preds_seg[0, :, :, 0] > 0.5).astype(np.float32)
    tc_mask = (preds_seg[0, :, :, 1] > 0.5).astype(np.float32)
    et_mask = (preds_seg[0, :, :, 2] > 0.5).astype(np.float32)
    
    print(f"  -> Hoàn tất! Kích thước vùng phù nề (WT): {np.sum(wt_mask)} pixels.")

    fig,ax=plt.subplots(1,3,figsize=(15,5))

    titles=["WT","TC","ET"]

    for i in range(3):

        ax[i].imshow(preds_seg[0,:,:,i],cmap='jet')

        ax[i].set_title(titles[i])

        ax[i].axis("off")

    plt.show()

    print("WT:",wt_mask.sum())
    print("TC:",tc_mask.sum())
    print("ET:",et_mask.sum())




    overlay = img_3c_uint8.copy()

    alpha=0.5

    green=np.zeros_like(overlay)
    green[:,:,1]=255

    red=np.zeros_like(overlay)
    red[:,:,2]=255

    yellow=np.zeros_like(overlay)
    yellow[:,:,0]=255
    yellow[:,:,1]=255

    overlay=np.where(
        wt_mask[...,None]>0.5,
        cv2.addWeighted(overlay,1-alpha,green,alpha,0),
        overlay
    )

    overlay=np.where(
        tc_mask[...,None]>0.5,
        cv2.addWeighted(overlay,1-alpha,red,alpha,0),
        overlay
    )

    overlay=np.where(
        et_mask[...,None]>0.5,
        cv2.addWeighted(overlay,1-alpha,yellow,alpha,0),
        overlay
    )

    plt.figure(figsize=(8,8))
    plt.imshow(overlay)
    plt.title("Segmentation Overlay")
    plt.axis("off")
    plt.show()

    print("\n[3/3] CHẠY CLASSIFICATION (Phòng 3 - Hội chẩn 4 AI)...")
    img_g_clahe = preprocess_clahe(img_3c_uint8)
    img_g_clahe_224 = cv2.resize(img_g_clahe, (224, 224))
    Xg = np.expand_dims(normalize(img_g_clahe_224.astype('float32')), axis=0)

    # Local Crop (Xl) và Mask
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

    print("  -> Bơm 5 tensor dữ liệu vào HỘI ĐỒNG 4 AI ConvNeXt...")
    all_fold_preds = []
    
    for idx, m in enumerate(models_cls):
        preds = m.predict([Xg, Xl, Xtc, Xet, Sf], verbose=0)
        all_fold_preds.append(preds[0])
        
    # Tính trung bình xác suất của 4 Folds
    final_ensemble_prob = np.mean(all_fold_preds, axis=0)
    
    CLASS_NAMES = {0: "Glioma", 1: "Meningioma", 2: "Pituitary"}
    best_idx = int(np.argmax(final_ensemble_prob))
    confidence = final_ensemble_prob[best_idx] * 100
    
    print("\n" + "="*60)
    print(f" 🎯 CHỐT KẾT QUẢ HỘI CHẨN 4 FOLD: {CLASS_NAMES[best_idx]} (Độ tin cậy: {confidence:.2f}%)")
    print("="*60)
    print(f"Chi tiết phân bổ xác suất (Mean 4 Folds):")
    print(f"  - Glioma     : {final_ensemble_prob[0]*100:.2f}%")
    print(f"  - Meningioma : {final_ensemble_prob[1]*100:.2f}%")
    print(f"  - Pituitary  : {final_ensemble_prob[2]*100:.2f}%")