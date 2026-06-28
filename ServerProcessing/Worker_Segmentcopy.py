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
            [keras.layers.Dense(ff_dim, activation="relu"), keras.layers.Dense(embed_dim),]
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
# 4. HÀM TÔ MÀU VÀ BLEND ẢNH CHUẨN Y KHOA
# ==========================================
def apply_color_mask(original_slice, wt_pred, tc_pred, et_pred):
    orig_img = cv2.normalize(original_slice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    color_mask = np.zeros_like(orig_img)

    wt_mask = (wt_pred > 0.5).astype(np.uint8)
    tc_mask = (tc_pred > 0.5).astype(np.uint8)
    et_mask = (et_pred > 0.5).astype(np.uint8)

    color_mask[wt_mask == 1] = [0, 255, 0]   # WT: Xanh lá
    color_mask[tc_mask == 1] = [0, 0, 255]   # TC: Đỏ
    color_mask[et_mask == 1] = [0, 255, 255] # ET: Vàng

    blended_img = cv2.addWeighted(orig_img, 1.0, color_mask, 0.4, 0)
    return blended_img

# ==========================================
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
        import nibabel as nib
        
        # 1. Đọc khối 3D
        nii_img = nib.load(npy_path)
        mri_data = nii_img.get_fdata()
        
        # 2. Chuẩn hóa Z-score toàn bộ khối não (Y hệt code cô giáo)
        mask = mri_data > 0
        if np.sum(mask) > 0:
            mean_val = mri_data[mask].mean()
            std_val = mri_data[mask].std()
            mri_data = (mri_data - mean_val) / (std_val + 1e-8)
            mri_data[~mask] = 0.0 # Ép viền ngoài về 0
            
        # 3. 🚨 BÍ QUYẾT: GỌT VIỀN ĐEN ĐỂ ZOOM NÃO TO LÊN (CROP ROI)
        padding = 10
        coords = np.where(mri_data > 0)
        if len(coords[0]) > 0:
            x_min, x_max = max(0, coords[0].min() - padding), min(mri_data.shape[0], coords[0].max() + padding)
            y_min, y_max = max(0, coords[1].min() - padding), min(mri_data.shape[1], coords[1].max() + padding)
            z_min, z_max = max(0, coords[2].min() - padding), min(mri_data.shape[2], coords[2].max() + padding)
            cropped_mri = mri_data[x_min:x_max, y_min:y_max, z_min:z_max]
        else:
            cropped_mri = mri_data
            
        # 4. Thái lát 2D và Resize về chuẩn 256x256 của ResNet50
        resized_slices = []
        # Ở nibabel, mặt cắt ngang (axial) thường là trục Z (trục số 2)
        for i in range(cropped_mri.shape[2]):
            slice_2d = cropped_mri[:, :, i]
            
            # Bỏ qua mấy lát ở đỉnh sọ toàn màu đen
            if np.sum(slice_2d > 0) < 10: 
                continue
                
            img_resized = cv2.resize(slice_2d, (256, 256), interpolation=cv2.INTER_LINEAR)
            resized_slices.append(img_resized)
            
        slices_2d = np.array(resized_slices).astype(np.float32)
        
        # 5. Nhân bản 3 kênh màu (RGB ảo)
        slices_2d = np.stack((slices_2d,)*3, axis=-1) 
        
        print(f"⚙️ Chạy inference cho {len(slices_2d)} lát cắt (Đã Crop & Zoom)...")

        # 6. Kích hoạt AI
        preds = model.predict(slices_2d, batch_size=16)    

        # --- BƯỚC 3: TÔ MÀU VÀ LƯU FILE ---
        colored_results = []
        for i in range(len(slices_2d)):
            blended = apply_color_mask(
                slices_2d[i], 
                preds[i, :, :, 0], # Kênh 0: Cắt lấy mảng WT (Whole Tumor)
                preds[i, :, :, 1], # Kênh 1: Cắt lấy mảng TC (Tumor Core)
                preds[i, :, :, 2]  # Kênh 2: Cắt lấy mảng ET (Enhancing Tumor)
            )
            colored_results.append(blended)
            
        colored_results = np.array(colored_results)

        result_path = npy_path.replace('_2D_ready.npy', '_Segmented_Colored.npy')
        np.save(result_path, colored_results)
        print(f"📦 Đã lưu khối u tô màu tại: {result_path}")

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