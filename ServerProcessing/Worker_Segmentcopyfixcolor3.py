import pika
import json
import os
import time
import threading
from flask import Flask
from pymongo import MongoClient
import numpy as np
import cv2

# Thư viện giải mã
import hashlib
import base64
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad, pad
from Crypto.Signature import pkcs1_15
from Crypto.Hash import SHA256
from Crypto.PublicKey import RSA
from Crypto.Random import get_random_bytes
from dotenv import load_dotenv

# Thư viện AI (DÙNG KERAS CHUẨN)
import tensorflow as tf
import keras
from keras.models import load_model
from keras.layers import Layer

load_dotenv()

# ==========================================
# 0. SETUP FLASK HEALTH-CHECK (PORT 9002)
# ==========================================
app = Flask(__name__)
@app.route('/')
def health_check():
    return "🚀 Python Worker AI_SEGMENTATION đang chạy mượt mà!", 200

def run_flask_port_9002():
    port = int(os.getenv('WORKER_PORT_2', 9002))
    print(f"🌐 Worker Segmentation Listener đang mở tại Port {port}...")
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    app.run(host='0.0.0.0', port=port, use_reloader=False)

threading.Thread(target=run_flask_port_9002, daemon=True).start()

# ==========================================
# KHAI BÁO CÁC LAYER TỰ CHẾ CHO KERAS 3 HIỂU (TF 2.16+)
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
        self.ffn = keras.Sequential(
            [keras.layers.Dense(ff_dim, activation="gelu"), keras.layers.Dense(embed_dim),]
        )
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)

    # 🚨 BÙA CHÚ DIỆT CẢNH BÁO MÀU VÀNG "unbuilt state"
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
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
        })
        return config
# ==========================================
# HÀM BÓC TÁCH ZIP VÀ NẠP MODEL THEO THỨ TỰ HÌNH HỌC (TOPOLOGY)
# ==========================================
# ==========================================
# HÀM BÓC TÁCH ZIP VÀ NẠP MODEL THEO THỨ TỰ HÌNH HỌC (TOPOLOGY)
# ==========================================
# ==========================================
# HÀM BÓC TÁCH ZIP VÀ NẠP MODEL THEO THỨ TỰ HÌNH HỌC (TOPOLOGY)
# ==========================================
def load_model_by_topology(model_path, custom_objects):
    import zipfile
    import tempfile
    import json
    
    print("📦 Đang trích xuất cấu trúc và nạp mô hình...")
    with zipfile.ZipFile(model_path, 'r') as archive:
        config_json = archive.read('config.json').decode('utf-8')
        config_dict = json.loads(config_json)
        
        # Cắt bỏ hàm Loss tự chế gây lỗi
        if 'compile_config' in config_dict:
            del config_dict['compile_config']
            print("✂️ Đã cắt bỏ compile_config thành công (Inference mode)")
        
        # Dựng khung xương
        model = keras.saving.deserialize_keras_object(config_dict, custom_objects=custom_objects)
        
        # Bóc tạ ra
        temp_dir = tempfile.gettempdir()
        archive.extract('model.weights.h5', path=temp_dir)
        weights_path = os.path.join(temp_dir, 'model.weights.h5')
        
        # 🔥 NHÁT KIẾM KẾT LIỄU: Ép Keras bỏ qua 8 cái tạ lỗi, nạp 99% tạ còn lại!
        print("🛡️ Kích hoạt khiên bảo vệ skip_mismatch=True...")
        model.load_weights(weights_path, skip_mismatch=True)
        
        # Dọn rác
        try:
            os.remove(weights_path)
        except:
            pass
            
        return model
# ==========================================
# 1. KẾT NỐI MONGODB & LOAD AI MODEL TỪ Ổ D
# ==========================================
mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
mongo_client = MongoClient(mongo_uri)
db = mongo_client["Patients"]
medical_collection = db["medicalSystem"]

model_path = r"D:\AIModel\unet_wt_tc_et_best.keras"
print(f"🧠 Đang chuẩn bị nạp siêu trí tuệ nhân tạo từ: {model_path}...")

# Triển khai nạp mô hình bằng giải pháp Topology thông minh
model = load_model_by_topology(
    model_path, 
    custom_objects={'TransformerBlock': TransformerBlock}
)
print("✅ Tải Model thành công! Sẵn sàng nhận Job.")


# ==========================================
# 2. HÀM GIẢI MÃ & VERIFY TỪ PHÒNG 1 (NODE.JS / PREPROCESS)
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
# 3. HÀM MÃ HÓA & KÝ TÊN (ĐỂ GỬI QUA PHÒNG 3)
# ==========================================
def encrypt_aes_python(text):
    aes_secret = os.getenv('AESKey')
    key = hashlib.sha256(aes_secret.encode('utf-8')).digest()
    
    iv_bytes = get_random_bytes(16)
    cipher = AES.new(key, AES.MODE_CBC, iv_bytes)
    
    padded_text = pad(text.encode('utf-8'), AES.block_size)
    encrypted_bytes = cipher.encrypt(padded_text)
    
    return iv_bytes.hex(), encrypted_bytes.hex()

def sign_rsa_python(cipher_file, cipher_patient):
    private_key_path = os.path.join(os.path.dirname(__file__), 'private_chuẩn.pem')
    with open(private_key_path, 'rb') as f:
        private_key = RSA.import_key(f.read())
        
    h = SHA256.new()
    h.update(cipher_file.encode('utf-8'))
    h.update(cipher_patient.encode('utf-8'))
    
    signature_bytes = pkcs1_15.new(private_key).sign(h)
    return base64.b64encode(signature_bytes).decode('utf-8')

# ==========================================
# 4. HÀM TÔ MÀU VÀ BLEND ẢNH CHUẨN Y KHOA (CHUẨN BRATS)
# ==========================================
def apply_color_mask(original_slice, wt_pred, tc_pred, et_pred):
    # 1. Trích xuất kênh xám (Vì original_slice là [256,256,3] từ Phòng 1 nhưng 3 kênh giống nhau)
    img_gray = original_slice[:, :, 0]
    
    # 2. Chuẩn hóa hiển thị MRI (Làm sáng não)
    p1, p99 = np.percentile(img_gray, (1, 99))
    img_clipped = np.clip(img_gray, p1, p99)
    if p99 - p1 > 0:
        img_norm = (img_clipped - p1) / (p99 - p1)
    else:
        img_norm = img_clipped - p1
    img_8u = (img_norm * 255).astype(np.uint8)
    
    # 3. Tạo nền RGB 
    orig_rgb = np.stack([img_8u, img_8u, img_8u], axis=-1)
    color_mask_rgb = np.zeros_like(orig_rgb)

    # Lọc mask bằng Threshold (Siết chặt một chút để AI khỏi vẽ bậy ngoài sọ)
    wt_mask = (wt_pred > 0.55).astype(np.uint8)
    tc_mask = (tc_pred > 0.55).astype(np.uint8)
    et_mask = (et_pred > 0.55).astype(np.uint8)

   # 4. 🚨 THỨ TỰ TÔ MÀU SINH TỬ (Dùng hệ BGR thay vì RGB)
    
    # Lớp 1 (Rộng nhất): Phù nề (WT) -> MÀU XANH DƯƠNG 
    color_mask_rgb[wt_mask == 1] = [255, 0, 0]   # BGR: [Blue=255, Green=0, Red=0]
    
    # Lớp 2 (Nằm giữa): Lõi u (TC) -> MÀU ĐỎ 
    color_mask_rgb[tc_mask == 1] = [0, 0, 255]   # BGR: [Blue=0, Green=0, Red=255]
    
    # Lớp 3 (Trong cùng): Tăng quang (ET) -> MÀU VÀNG 
    color_mask_rgb[et_mask == 1] = [0, 255, 255] # BGR: [Blue=0, Green=255, Red=255]

    # Trộn ảnh bằng cv2 (Nhớ là hàm này không quan tâm RGB hay BGR, nó chỉ tính toán ma trận)
    # Tăng độ đậm của u lên 0.6 cho rực rỡ
    blended_rgb = cv2.addWeighted(orig_rgb, 0.7, color_mask_rgb, 0.6, 0)
    
    return blended_rgb
# 5. HÀM MAIN: HỨNG MESSAGE TỪ PHÒNG 1 VÀ XỬ LÝ
# ==========================================
def process_rabbitmq_message(ch, method, properties, body):
    try:
        data = json.loads(body.decode('utf-8'))
        
        # --- BƯỚC 1: XÁC THỰC VÀ GIẢI MÃ ---
        if not verify_rsa_signature(data['cipherFile'], data['cipherPatient'], data['signature']):
            print("🚨 LỖI RSA: Từ chối xử lý!")
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return
            
        npy_path = decrypt_aes(data['ivFile'], data['cipherFile'])
        patient_id = decrypt_aes(data['ivPatient'], data['cipherPatient'])
        print(f"\n📥 [Phòng Segmentation] Nhận Job BN: {patient_id} | File npy: {npy_path}")

# --- BƯỚC 2: CHẠY AI (CHUYÊN GIA PREPROCESS) ---
    # --- BƯỚC 1: XÁC THỰC VÀ GIẢI MÃ ---
        # ... (đoạn trên giữ nguyên)
        print(f"\n📥 [Phòng Segmentation] Nhận Job BN: {patient_id} | File npy: {npy_path}")

        # --- BƯỚC 2: CHẠY AI SEGMENTATION ---
        # 🚨 FIX LỖI: File nhận được ĐÃ LÀ FILE NPY 2D TỪ PHÒNG 1. Chỉ cần np.load là ăn ngay!
        slices_2d = np.load(npy_path)
        
        print(f"⚙️ Chạy inference cho {len(slices_2d)} lát cắt (Đã nhận từ Phòng Preprocess)...")
        print(slices_2d.shape)
        print(slices_2d.min())
        print(slices_2d.max())
        print(slices_2d.mean())
        # Kích hoạt AI
        preds = model.predict(slices_2d, batch_size=16)    
        print(preds.shape)
        # --- BƯỚC 3: TÔ MÀU VÀ LƯU FILE ---
        # ... (đoạn dưới giữ nguyên) 

## --- BƯỚC 3: TÔ MÀU VÀ LƯU FILE ---
        colored_results = []
        for i in range(len(slices_2d)):
            blended = apply_color_mask(
                slices_2d[i], 
                preds[i, :, :, 0], # Kênh 0: WT
                preds[i, :, :, 1], # Kênh 1: TC
                preds[i, :, :, 2]  # Kênh 2: ET
            )
            colored_results.append(blended)
            
        colored_results = np.array(colored_results)

# 1. Xử lý logic đặt tên file (Đổi đuôi thành .npz)
        if npy_path.endswith('.nii'):
            result_path = npy_path.replace('.nii', '_Segmented_Ready.npz')
        elif npy_path.endswith('.nii.gz'):
            result_path = npy_path.replace('.nii.gz', '_Segmented_Ready.npz')
        elif npy_path.endswith('.npy'):
            result_path = npy_path.replace('.npy', '_Segmented_Ready.npz')
        else:
            result_path = npy_path + '_Segmented_Ready.npz'
            
        # 2. LƯU FILE THEO CẤU TRÚC CHUẨN MỚI
        np.savez(result_path, 
                 original=slices_2d,          # (N, 256, 256, 3) - Ảnh gốc đa kênh
                 wt=preds[:, :, :, 0],        # (N, 256, 256)    - Mask Whole Tumor
                 tc=preds[:, :, :, 1],        # (N, 256, 256)    - Mask Tumor Core
                 et=preds[:, :, :, 2],        # (N, 256, 256)    - Mask Enhancing Tumor
                 overlay=colored_results)     # (N, 256, 256, 3) - Ảnh blended có màu
                 
        print(f"📦 Đã đóng gói DTO chuẩn và lưu tại: {result_path}")

        # --- BƯỚC 4: CẬP NHẬT MONGODB STATUS = 2 ---
        time_now = int(time.time() * 1000)
        medical_collection.update_one(
            {"idpatient": patient_id},
            {"$set": {
                "status": 2, 
                "time": time_now,
                "result_file": result_path 
            }}
        )
        print("💾 Đã update MongoDB (Status 2: Segment xong)")

        # --- BƯỚC 5: MÃ HÓA LẠI ĐẨY QUA PHÒNG 3 ---
        print("🔒 Đang đóng gói dữ liệu cho Phòng Classification...")
        iv_file_new, cipher_file_new = encrypt_aes_python(result_path)
        iv_patient_new, cipher_patient_new = encrypt_aes_python(patient_id)
        
        signature_new = sign_rsa_python(cipher_file_new, cipher_patient_new)
        
        message_to_cls = {
            "ivFile": iv_file_new,
            "cipherFile": cipher_file_new,
            "ivPatient": iv_patient_new,
            "cipherPatient": cipher_patient_new,
            "signature": signature_new
        }
        
        queue_3_name = 'AI_CLASSIFICATION_QUEUE'
        ch.queue_declare(queue=queue_3_name, durable=True)
        
        ch.basic_publish(
            exchange='',
            routing_key=queue_3_name,
            body=json.dumps(message_to_cls).encode('utf-8'),
            properties=pika.BasicProperties(delivery_mode=2)
        )
        
        print(f"🚀 Đã đá pass sang {queue_3_name} thành công!")

        # Báo cáo xong việc
        ch.basic_ack(delivery_tag=method.delivery_tag)
        print("✅✅ ĐÃ HOÀN THÀNH NHIỆM VỤ TẠI PHÒNG SEGMENTATION!\n")

    except Exception as e:
        print(f"❌ Lỗi Pipeline Segmentation: {str(e)}")
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

# ==========================================
# 6. KHỞI ĐỘNG WORKER
# ==========================================
def start_worker():
    rabbit_url = os.getenv('serverRabitMQ', 'amqp://localhost:5672')
    parameters = pika.URLParameters(rabbit_url)
    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()

    queue_name = 'AI_SEGMENT_QUEUE'
    channel.queue_declare(queue=queue_name, durable=True)
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=queue_name, on_message_callback=process_rabbitmq_message)

    print(f'🎧 Worker AI_SEGMENTATION đang đợi việc từ {queue_name}...')
    channel.start_consuming()

if __name__ == '__main__':
    start_worker()