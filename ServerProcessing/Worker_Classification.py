import pika
import json
import os
import time
import threading
from flask import Flask
from pymongo import MongoClient
import numpy as np
import cv2

# Thư viện giải mã từ Node.js
import hashlib
import base64
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad
from Crypto.Signature import pkcs1_15
from Crypto.Hash import SHA256
from Crypto.PublicKey import RSA
from dotenv import load_dotenv

# Thư viện AI
import tensorflow as tf
import keras

load_dotenv()

# ==========================================
# 0. SETUP FLASK HEALTH-CHECK (PORT 9003)
# ==========================================
app = Flask(__name__)
@app.route('/')
def health_check():
    return "🚀 Python Worker AI_CLASSIFICATION đang chạy mượt mà!", 200

def run_flask_port_9003():
    port = int(os.getenv('WORKER_PORT_3', 9003))
    print(f"🌐 Worker Classification Listener đang mở tại Port {port}...")
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    app.run(host='0.0.0.0', port=port, use_reloader=False)

threading.Thread(target=run_flask_port_9003, daemon=True).start()

# ==========================================
# 0.5. KHAI BÁO CUSTOM FUNCTION CHO KERAS 3 HIỂU (CHỐT HẠ WT, TC, ET)
# ==========================================
@keras.saving.register_keras_serializable(package="custom")
def _reduce_max_tc_fn(x):
    import tensorflow as tf
    return tf.reduce_max(x, axis=-1, keepdims=True)

@keras.saving.register_keras_serializable(package="custom")
def _reduce_max_et_fn(x):
    import tensorflow as tf
    return tf.reduce_max(x, axis=-1, keepdims=True)

@keras.saving.register_keras_serializable(package="custom")
def _reduce_max_wt_fn(x):
    import tensorflow as tf
    return tf.reduce_max(x, axis=-1, keepdims=True)

# ==========================================
# 1. KẾT NỐI MONGODB & NẠP HỘI ĐỒNG 4 AI
# ==========================================
mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
mongo_client = MongoClient(mongo_uri)
db = mongo_client["Patients"]
medical_collection = db["medicalSystem"]
history_collection = db["patientHistory"]

# Danh sách nhãn bệnh
CLASS_NAMES = {0: "Glioma", 1: "Meningioma", 2: "Pituitary"}

# --- TUYỆT CHIÊU BÓC TÁCH VÀ DỌN RÁC JSON CHO KERAS ---
def load_model_by_topology_cleaner(model_path, custom_objects=None):
    import zipfile
    import tempfile
    import json

    def scrub_dict(d):
        """Hàm đệ quy lùng sục và tiêu diệt triệt để 'quantization_config'"""
        if isinstance(d, dict):
            if 'quantization_config' in d:
                del d['quantization_config']
            for k, v in d.items():
                scrub_dict(v)
        elif isinstance(d, list):
            for item in d:
                scrub_dict(item)

    print(f"📦 Đang bóc tách và dọn rác model: {os.path.basename(model_path)}...")
    with zipfile.ZipFile(model_path, 'r') as archive:
        config_json = archive.read('config.json').decode('utf-8')
        config_dict = json.loads(config_json)

        if 'compile_config' in config_dict:
            del config_dict['compile_config']

        scrub_dict(config_dict)

        # 🚨 ĐÃ TRUYỀN CUSTOM_OBJECTS VÀO ĐÂY
        model = keras.saving.deserialize_keras_object(config_dict, custom_objects=custom_objects)

        temp_dir = tempfile.gettempdir()
        archive.extract('model.weights.h5', path=temp_dir)
        weights_path = os.path.join(temp_dir, 'model.weights.h5')

        model.load_weights(weights_path, skip_mismatch=True)

        try:
            os.remove(weights_path)
        except:
            pass

        return model

def load_ensemble_models():
    """ Nạp cùng lúc 4 Fold AI vào RAM bằng Topology """
    models = []
    
    # 🚨 ĐÃ GIỮ NGUYÊN ĐƯỜNG DẪN CỦA ÔNG
    base_model_dir = r"D:\AIModel" 
    
    print(f"🧠 Đang triệu tập hội đồng 4 chuyên gia AI từ: {base_model_dir}...")
    for i in range(1, 5): 
        # 🚨 ĐÃ GIỮ NGUYÊN TÊN FILE CỦA ÔNG
        model_path = os.path.join(base_model_dir, f"fold_{i}_best.keras")
        
        if os.path.exists(model_path):
            print(f"  -> Đang nạp Fold {i}...")
            
            # TRUYỀN CẢ 3 HÀM CUSTOM VÀO ĐỂ KÊ THÊM GỐI CHO KÊ YÊN TÂM NGỦ
            m = load_model_by_topology_cleaner(
                model_path, 
                custom_objects={
                    '_reduce_max_tc_fn': _reduce_max_tc_fn,
                    '_reduce_max_et_fn': _reduce_max_et_fn,
                    '_reduce_max_wt_fn': _reduce_max_wt_fn
                }
            )
            
            models.append(m)
        else:
            print(f"  ⚠️ CẢNH BÁO: Không tìm thấy Fold {i} tại {model_path}")
            
    print(f"✅ Đã nạp thành công {len(models)}/4 mô hình! Sẵn sàng hội chẩn.")
    return models

# Tải 4 models sẵn vào bộ nhớ ngay khi bật server
ensemble_models = load_ensemble_models()

# ... (PHẦN DƯỚI GIỮ NGUYÊN KHÔNG ĐỔI) ...

# ==========================================
# 2. HÀM GIẢI MÃ & VERIFY CHỮ KÝ (NHẬN TỪ PHÒNG 2)
# ==========================================
def verify_rsa_signature(cipher_file, cipher_patient, signature_b64):
    pub_key_path = os.path.join(os.path.dirname(__file__), 'public_chuẩn.pem')
    with open(pub_key_path, 'rb') as f:
        public_key = RSA.import_key(f.read())
        
    h = SHA256.new()
    h.update(cipher_file.encode('utf-8'))
    h.update(cipher_patient.encode('utf-8'))
    
    try:
        pkcs1_15.new(public_key).verify(h, base64.b64decode(signature_b64))
        return True
    except:
        return False

def decrypt_aes(iv_hex, cipher_hex):
    aes_secret = os.getenv('AESKey')
    key = hashlib.sha256(aes_secret.encode('utf-8')).digest()
    iv = bytes.fromhex(iv_hex)
    cipher_bytes = bytes.fromhex(cipher_hex)
    
    cipher = AES.new(key, AES.MODE_CBC, iv)
    decrypted_padded = cipher.decrypt(cipher_bytes)
    return unpad(decrypted_padded, AES.block_size).decode('utf-8')

# ==========================================
# CÁC HÀM TRÍCH XUẤT ĐẶC TRƯNG (LẤY TỪ V7 NOTEBOOK)
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
    if mask.sum() > 0:
        m = img[mask].mean()
        s = img[mask].std()
        return (img - m) / (s + 1e-8)
    return img

def normalize_global(img):
    return (img - img.mean()) / (img.std() + 1e-8)

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
    A = cv2.contourArea(c)
    P = cv2.arcLength(c, True)
    return float(4*np.pi*A/(P*P+1e-6))

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
    eps = 1e-6
    E = 50 # EMPTY_MASK_THRESH
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
# 3. HÀM MAIN: HỨNG MESSAGE & HỘI CHẨN CHỐT BỆNH
# ==========================================
def process_rabbitmq_message(ch, method, properties, body):
    try:
        data = json.loads(body.decode('utf-8'))
        
        # --- BƯỚC 1: XÁC THỰC VÀ GIẢI MÃ ---
        if not verify_rsa_signature(data['cipherFile'], data['cipherPatient'], data['signature']):
            print("🚨 LỖI RSA: Từ chối xử lý dữ liệu giả mạo!")
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return
            
        npz_path = decrypt_aes(data['ivFile'], data['cipherFile']) # Đọc file .npz thay vì .npy
        patient_id = decrypt_aes(data['ivPatient'], data['cipherPatient'])
        print(f"\n📥 [Phòng Classification] Nhận Job BN: {patient_id}")
        
# --- BƯỚC 2: CHUẨN BỊ 5 INPUTS CHO MÔ HÌNH ---
        payload = np.load(npz_path)
        orig_slices = payload['original'] 
        wt_preds = payload['wt']
        tc_preds = payload['tc']
        et_preds = payload['et']
        
        Xg_list, Xl_list, Xtc_list, Xet_list, Sf_list = [], [], [], [], []
        
        for i in range(len(orig_slices)):
            img_orig = orig_slices[i].astype('float32') # (256, 256, 3)
            wt_orig, tc_orig, et_orig = wt_preds[i], tc_preds[i], et_preds[i]
            
            # 🚨 QUAN TRỌNG: Resize tất cả về 224x224 và áp dụng CLAHE làm ảnh gốc (giống hệt Colab)
            img_224 = cv2.resize(img_orig, (224, 224))
            img_clahe = preprocess_clahe(img_224).astype('float32') # Đây là 'img_f' của Colab
            
            wt = cv2.resize(wt_orig.astype('float32'), (224, 224), interpolation=cv2.INTER_NEAREST)
            tc = cv2.resize(tc_orig.astype('float32'), (224, 224), interpolation=cv2.INTER_NEAREST)
            et = cv2.resize(et_orig.astype('float32'), (224, 224), interpolation=cv2.INTER_NEAREST)
            
            # Chuẩn bị ảnh Global (X_g) - Dùng thẳng ảnh CLAHE (0-255), KHÔNG NORMALIZE
            Xg_list.append(img_clahe)
            
            # Chuẩn bị ảnh Local (X_l) và các Mask (X_tc, X_et)
            bbox = _wt_bbox(wt, 25, img_clahe.shape)
            if bbox is None:
                xl = cv2.resize(img_clahe, (224, 224))
                tc_c = np.zeros((224, 224), 'float32')
                et_c = np.zeros((224, 224), 'float32')
            else:
                y0, y1, x0, x1 = bbox
                # Cắt từ ảnh img_clahe
                xl = cv2.resize(img_clahe[y0:y1, x0:x1], (224, 224))
                tc_c = cv2.resize(tc[y0:y1, x0:x1], (224, 224), interpolation=cv2.INTER_NEAREST)
                et_c = cv2.resize(et[y0:y1, x0:x1], (224, 224), interpolation=cv2.INTER_NEAREST)
                
            Xl_list.append(xl) # KHÔNG NORMALIZE
            Xtc_list.append(np.stack([(tc_c>0.5).astype('float32')*255.]*3, -1))
            Xet_list.append(np.stack([(et_c>0.5).astype('float32')*255.]*3, -1))
            
            # Trích xuất 11 features hình học từ mặt nạ 224 và ảnh CLAHE
            Sf_list.append(compute_seg_features(wt, tc, et, img_clahe.shape, image=img_clahe))

        # Chuyển đổi sang numpy array chuẩn
        Xg = np.array(Xg_list, dtype='float32')
        Xl = np.array(Xl_list, dtype='float32')
        Xtc = np.array(Xtc_list, dtype='float32')
        Xet = np.array(Xet_list, dtype='float32')
        Sf = np.array(Sf_list, dtype='float32')

        print(f"⚙️ Chạy inference trên {len(orig_slices)} lát cắt với 5 luồng input chuẩn.")

# --- BƯỚC 3: HỘI CHẨN TẬP TRUNG (CHỈ KHẢO SÁT LÁT CẮT CÓ U) ---
        print("📊 Đang lọc các lát cắt có chứa khối u để hội chẩn...")
        
        # 1. Lọc index của các lát cắt có diện tích Whole Tumor (WT) > 50 pixels
        valid_indices = []
        for i in range(len(orig_slices)):
            if np.sum(wt_preds[i]) > 50:
                valid_indices.append(i)
                
        # Fallback an toàn: Nếu Segmentation fail (không tìm thấy u), lấy tạm 3 lát cắt ở giữa não
        if len(valid_indices) == 0:
            print("⚠️ CẢNH BÁO: Không tìm thấy khối u rõ ràng, sẽ dùng các lát cắt trung tâm!")
            mid = len(orig_slices) // 2
            valid_indices = [max(0, mid-1), mid, min(len(orig_slices)-1, mid+1)]
            
        print(f"🔬 Đã chắt lọc được {len(valid_indices)}/{len(orig_slices)} lát cắt chứa u để AI chẩn đoán.")
        
        # 2. Rút trích dữ liệu của riêng các lát cắt hợp lệ
        Xg_valid = Xg[valid_indices]
        Xl_valid = Xl[valid_indices]
        Xtc_valid = Xtc[valid_indices]
        Xet_valid = Xet[valid_indices]
        Sf_valid = Sf[valid_indices]

        # 3. Tiến hành Ensemble
        all_fold_preds = []
        for idx, m in enumerate(ensemble_models):
            # AI chỉ predict trên các lát cắt valid
            preds = m.predict([Xg_valid, Xl_valid, Xtc_valid, Xet_valid, Sf_valid], batch_size=16, verbose=0)
            
            # Lấy trung bình cộng HOẶC lấy Max Confidence (ở đây dùng mean trên các lát cắt có u)
            mean_pred_per_fold = np.mean(preds, axis=0)
            all_fold_preds.append(mean_pred_per_fold)
            
        # Tính trung bình của cả 4 Folds
        final_ensemble_prob = np.mean(all_fold_preds, axis=0)
        
        # Chốt hạ
        best_class_idx = int(np.argmax(final_ensemble_prob))
        best_class_name = CLASS_NAMES[best_class_idx]
        confidence = float(final_ensemble_prob[best_class_idx] * 100)
        
        print(f"🎯 KẾT QUẢ HỘI CHẨN CUỐI CÙNG: {best_class_name} (Độ tin cậy: {confidence:.2f}%)")

        # --- BƯỚC 4: CẬP NHẬT MONGODB STATUS = 3 ---
        time_now = int(time.time() * 1000)  
        medical_collection.update_one(
            {"idpatient": patient_id},
            {"$set": {
                "status": 3, 
                "time": time_now,
                "diagnosis": best_class_name,
                "confidence": round(confidence, 2)
            }}
        )
        file_name_only = os.path.basename(npz_path) 
        
        history_collection.insert_one({
            "idpatient": patient_id,
            "timestamp": time_now, # Lưu time_now để sau này query sắp xếp (sort) cho dễ
            "result_file_name": file_name_only, # Vd: 021_flair_1783408229996_2D_ready_Segmented_Ready.npz
            "diagnosis": best_class_name,
            "confidence": round(confidence, 2)
        })
        
        print("💾 Đã lưu kết quả vĩnh viễn vào Lịch sử (patientHistory)")
        print("💾 Đã update MongoDB (Status 3: Phân loại xong)")
        ch.basic_ack(delivery_tag=method.delivery_tag)
        print("✅✅ ĐÃ HOÀN THÀNH TOÀN BỘ QUY TRÌNH!\n")

    except Exception as e:
        print(f"❌ Lỗi Pipeline Classification: {str(e)}")
        
        # 1. Báo cáo lỗi vào database để Frontend (React) biết đường dừng Loading
        try:
            # Lưu ý: Cần khai báo 'patient_id' ở đầu hàm (ngay sau khi giải mã) 
            # để dùng được ở đây trong trường hợp lỗi xảy ra sau khi đã giải mã.
            if 'patient_id' in locals():
                time_now = int(time.time() * 1000)
                medical_collection.update_one(
                    {"idpatient": patient_id},
                    {"$set": {
                        "status": -1,  # -1 Tượng trưng cho lỗi AI
                        "time": time_now,
                        "diagnosis": "Lỗi xử lý hệ thống AI"
                    }}
                )
                print(f"💾 Đã cập nhật MongoDB (Status -1) cho bệnh nhân {patient_id}")
        except Exception as db_err:
            print(f"⚠️ Lỗi khi cập nhật DB báo lỗi: {db_err}")

        # 2. Hủy bỏ bức thư "độc" này để không làm nghẽn hệ thống
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

# ==========================================
# 4. KHỞI ĐỘNG WORKER
# ==========================================
def start_worker():
    rabbit_url = os.getenv('serverRabitMQ', 'amqp://localhost:5672')
    parameters = pika.URLParameters(rabbit_url)
    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()

    # Nhận dữ liệu từ Queue số 3
    queue_name = 'AI_CLASSIFICATION_QUEUE'
    channel.queue_declare(queue=queue_name, durable=True)
    
    # Chỉ nhận 1 job mỗi lần để tránh quá tải RAM
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=queue_name, on_message_callback=process_rabbitmq_message)

    print(f'🎧 Worker AI_CLASSIFICATION đang đợi việc từ {queue_name}...')
    channel.start_consuming()

if __name__ == '__main__':
    start_worker()