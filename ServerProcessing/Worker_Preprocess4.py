import pika
import json
import os
import time
import threading
from flask import Flask
from pymongo import MongoClient
import nibabel as nib
import numpy as np
import cv2
from Crypto.Util.Padding import pad
from Crypto.Random import get_random_bytes

# Thư viện giải mã
import hashlib
import base64
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad
from Crypto.Signature import pkcs1_15
from Crypto.Hash import SHA256
from Crypto.PublicKey import RSA
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 0. SETUP FLASK HEALTH-CHECK (PORT 9001)
# ==========================================
app = Flask(__name__)
@app.route('/')
def health_check():
    return "🚀 Python Worker AI_Preprocess đang chạy mượt mà!", 200

def run_flask_port_9001():
    port = int(os.getenv('WORKER_PORT', 9001))
    print(f"🌐 Worker Listener đang mở tại Port {port}...")
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    app.run(host='0.0.0.0', port=port, use_reloader=False)

threading.Thread(target=run_flask_port_9001, daemon=True).start()

# ==========================================
# 1. KẾT NỐI MONGODB
# ==========================================
mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
mongo_client = MongoClient(mongo_uri)
db = mongo_client["Patients"]
medical_collection = db["medicalSystem"]

# ==========================================
# 2. HÀM GIẢI MÃ & VERIFY CẬP NHẬT 3 KÊNH
# ==========================================
def verify_rsa_signature(c_flair, c_t1ce, c_t2, c_patient, signature_b64):
    pub_key_path = os.path.join(os.path.dirname(__file__), 'public_chuẩn.pem')
    with open(pub_key_path, 'rb') as f:
        public_key = RSA.import_key(f.read())
        
    h = SHA256.new()
    h.update(c_flair.encode('utf-8'))
    h.update(c_t1ce.encode('utf-8'))
    h.update(c_t2.encode('utf-8'))
    h.update(c_patient.encode('utf-8'))
    
    try:
        pkcs1_15.new(public_key).verify(h, base64.b64decode(signature_b64))
        return True
    except (ValueError, TypeError):
        return False

def decrypt_aes(iv_hex, cipher_hex):
    aes_secret = os.getenv('AESKey')
    key = hashlib.sha256(aes_secret.encode('utf-8')).digest()
    iv = bytes.fromhex(iv_hex)
    cipher_bytes = bytes.fromhex(cipher_hex)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    decrypted_padded = cipher.decrypt(cipher_bytes)
    return unpad(decrypted_padded, AES.block_size).decode('utf-8')

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
# 3. HÀM XỬ LÝ ẢNH Y KHOA (CHẬP 3 KÊNH)
# ==========================================
def normalize(img):
    mask = img > 0
    if np.sum(mask) > 0:
        mean = img[mask].mean()
        std = img[mask].std()
        img_norm = (img - mean) / (std + 1e-8)
        img_norm[~mask] = 0.0 
        return img_norm
    return img

def process_3_channels(flair, t1ce, t2, padding=10, size=256):
    """ Hàm này cắt viền và xếp chồng 3 kênh màu lại với nhau """
    # Tạo mặt nạ tổng hợp để lấy viền rộng nhất của não
    mask = (flair > 0) | (t1ce > 0) | (t2 > 0)
    coords = np.where(mask)
    if len(coords[0]) == 0: return []
    
    # Tính tọa độ crop
    x_min, x_max = max(0, coords[0].min() - padding), min(mask.shape[0], coords[0].max() + padding)
    y_min, y_max = max(0, coords[1].min() - padding), min(mask.shape[1], coords[1].max() + padding)
    z_min, z_max = max(0, coords[2].min() - padding), min(mask.shape[2], coords[2].max() + padding)
    
    # Cắt cả 3 ảnh
    c_flair = flair[x_min:x_max, y_min:y_max, z_min:z_max]
    c_t1ce = t1ce[x_min:x_max, y_min:y_max, z_min:z_max]
    c_t2 = t2[x_min:x_max, y_min:y_max, z_min:z_max]
    
    images = []
    # Quét qua từng lát cắt (trục Z)
    for i in range(c_flair.shape[2]):
        sl_flair = np.rot90(c_flair[:, :, i], k=1)
        sl_t1ce = np.rot90(c_t1ce[:, :, i], k=1)
        sl_t2 = np.rot90(c_t2[:, :, i], k=1)
        
        # Lọc nhiễu sọ đen
        if np.sum(sl_flair > 0) < 10: continue
        
        # Resize
        sl_flair = cv2.resize(sl_flair, (size, size))
        sl_t1ce = cv2.resize(sl_t1ce, (size, size))
        sl_t2 = cv2.resize(sl_t2, (size, size))
        
        # 🚨 BÍ QUYẾT TẠO ẢNH ĐA KÊNH: Chập 3 ma trận lại làm 1 
        # (Thứ tự chuẩn của Colab: FLAIR, T1ce, T2)
        img_3c = np.stack([sl_flair, sl_t1ce, sl_t2], axis=-1)
        images.append(img_3c.astype(np.float32))
        
    return np.array(images)

# ==========================================
# 4. HÀM MAIN: HỨNG MESSAGE TỪ RABBITMQ
# ==========================================
def process_rabbitmq_message(ch, method, properties, body):
    try:
        data = json.loads(body.decode('utf-8'))
        
        # --- BƯỚC 1: XÁC THỰC VÀ GIẢI MÃ ---
        # Kiểm tra mộc với cả 3 file
        if not verify_rsa_signature(data['cipherFlair'], data['cipherT1ce'], data['cipherT2'], data['cipherPatient'], data['signature']):
            print("🚨 CẢNH BÁO: CHỮ KÝ GIẢ MẠO! TỪ CHỐI XỬ LÝ.")
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return
            
        # Giải mã lấy 3 đường dẫn
        path_flair = decrypt_aes(data['ivFlair'], data['cipherFlair'])
        path_t1ce = decrypt_aes(data['ivT1ce'], data['cipherT1ce'])
        path_t2 = decrypt_aes(data['ivT2'], data['cipherT2'])
        patient_id = decrypt_aes(data['ivPatient'], data['cipherPatient'])
        
        print(f"✅ Đã tiếp nhận MRI (3 kênh) của BN: {patient_id}")

        # --- BƯỚC 2: ĐỌC VÀ CHUẨN HÓA CẢ 3 FILE ---
        print("⚙️ Đang đọc và đồng bộ hóa 3 kênh ảnh...")
        flair_data = normalize(nib.load(path_flair).get_fdata())
        t1ce_data = normalize(nib.load(path_t1ce).get_fdata())
        t2_data = normalize(nib.load(path_t2).get_fdata())
        
        # Cắt viền và xếp chồng thành ảnh đa kênh
        slices_2d = process_3_channels(flair_data, t1ce_data, t2_data)
        
        # Lấy tên file flair để tạo tên file chung xuất ra
        if path_flair.endswith('.nii'):
            out_path = path_flair.replace('.nii', '_2D_ready.npy')
        elif path_flair.endswith('.nii.gz'):
            out_path = path_flair.replace('.nii.gz', '_2D_ready.npy')
        else:
            out_path = path_flair + "_2D_ready.npy"

        np.save(out_path, slices_2d)
        print(f"📦 Đã lưu ảnh 2D đa kênh tại: {out_path}")

        # --- BƯỚC 3: UPDATE MONGODB ---
        time_now = int(time.time() * 1000)
        medical_collection.update_one(
            {"idpatient": patient_id},
            {"$set": {"status": 1, "time": time_now}}
        )
        print("💾 Đã update MongoDB (Status 1: Tiền xử lý xong)")

        # --- BƯỚC 4: GỬI QUA PHÒNG SEGMENTATION ---
        # Lúc này chỉ cần gửi 1 file Numpy duy nhất (.npy)
        iv_file_new, cipher_file_new = encrypt_aes_python(out_path)
        iv_patient_new, cipher_patient_new = encrypt_aes_python(patient_id)
        signature_new = sign_rsa_python(cipher_file_new, cipher_patient_new)
        
        message_to_seg = {
            "ivFile": iv_file_new,
            "cipherFile": cipher_file_new,
            "ivPatient": iv_patient_new,
            "cipherPatient": cipher_patient_new,
            "signature": signature_new
        }
        
        queue_2_name = 'AI_SEGMENT_QUEUE'
        ch.queue_declare(queue=queue_2_name, durable=True)
        ch.basic_publish(
            exchange='', routing_key=queue_2_name,
            body=json.dumps(message_to_seg).encode('utf-8'),
            properties=pika.BasicProperties(delivery_mode=2)
        )
        
        print(f"🚀 Đã bắn cục data qua {queue_2_name} thành công!")
        ch.basic_ack(delivery_tag=method.delivery_tag)
        print("============================================\n")

    except Exception as e:
        print(f"❌ Lỗi Pipeline: {str(e)}")
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

def start_worker():
    rabbit_url = os.getenv('serverRabitMQ', 'amqp://localhost:5672')
    parameters = pika.URLParameters(rabbit_url)
    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()

    channel.queue_declare(queue='Patient_QUEUE', durable=True)
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue='Patient_QUEUE', on_message_callback=process_rabbitmq_message)

    print('🎧 Worker AI_Preprocess đang nghe RabbitMQ...')
    channel.start_consuming()

if __name__ == '__main__':
    start_worker()