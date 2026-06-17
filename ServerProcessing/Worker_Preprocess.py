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
    # Tắt log của Flask cho đỡ rác terminal
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    app.run(host='0.0.0.0', port=port, use_reloader=False)

# Khởi chạy Port 9001 ở một luồng (thread) chạy ngầm
threading.Thread(target=run_flask_port_9001, daemon=True).start()

# ==========================================
# 1. KẾT NỐI MONGODB
# ==========================================
mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
mongo_client = MongoClient(mongo_uri)
db = mongo_client["Patients"]
medical_collection = db["medicalSystem"]

# ==========================================
# 2. HÀM GIẢI MÃ & VERIFY (KHỚP 100% NODE.JS)
# ==========================================
def verify_rsa_signature(cipher_file, cipher_patient, signature_b64):
    """ Dùng Public Key để kiểm chứng mộc RSA từ Node.js """
    pub_key_path = os.path.join(os.path.dirname(__file__), 'public_chuẩn.pem')
    with open(pub_key_path, 'rb') as f:
        public_key = RSA.import_key(f.read())
        
    # Tạo chuỗi băm y hệt Node.js: sign.update(cipherFile) -> sign.update(cipherPatient)
    h = SHA256.new()
    h.update(cipher_file.encode('utf-8'))
    h.update(cipher_patient.encode('utf-8'))
    
    try:
        # Kiểm tra mộc
        pkcs1_15.new(public_key).verify(h, base64.b64decode(signature_b64))
        return True
    except (ValueError, TypeError):
        return False

def decrypt_aes(iv_hex, cipher_hex):
    """ Giải mã AES 256 CBC y hệt Node.js """
    # 1. Băm cái AESKey trong .env ra giống hàm crypto.createHash('sha256') của Node
    aes_secret = os.getenv('AESKey')
    if not aes_secret:
        raise ValueError("Thiếu AESKey trong file .env")
        
    key = hashlib.sha256(aes_secret.encode('utf-8')).digest()
    iv = bytes.fromhex(iv_hex)
    cipher_bytes = bytes.fromhex(cipher_hex)
    
    # 2. Giải mã
    cipher = AES.new(key, AES.MODE_CBC, iv)
    decrypted_padded = cipher.decrypt(cipher_bytes)
    
    # 3. Gỡ đệm PKCS7 (NodeJS mặc định dùng cái này)
    decrypted_text = unpad(decrypted_padded, AES.block_size).decode('utf-8')
    return decrypted_text

# ==========================================
# 2.5. HÀM MÃ HÓA & KÝ TÊN ĐỂ GỬI ĐI
# ==========================================
def encrypt_aes_python(text):
    """ Bắt chước hàm EncryptAES.js của Node """
    aes_secret = os.getenv('AESKey')
    key = hashlib.sha256(aes_secret.encode('utf-8')).digest()
    
    iv_bytes = get_random_bytes(16)
    cipher = AES.new(key, AES.MODE_CBC, iv_bytes)
    
    # Ép đệm PKCS7 rồi mã hóa
    padded_text = pad(text.encode('utf-8'), AES.block_size)
    encrypted_bytes = cipher.encrypt(padded_text)
    
    # Trả về mã Hex giống hệt toString('hex') của Node.js
    return iv_bytes.hex(), encrypted_bytes.hex()

def sign_rsa_python(cipher_file, cipher_patient):
    """ Bắt chước đóng mộc của SeenMessage.js """
    private_key_path = os.path.join(os.path.dirname(__file__), 'private_chuẩn.pem')
    with open(private_key_path, 'rb') as f:
        private_key = RSA.import_key(f.read())
        
    h = SHA256.new()
    h.update(cipher_file.encode('utf-8'))
    h.update(cipher_patient.encode('utf-8'))
    
    signature_bytes = pkcs1_15.new(private_key).sign(h)
    return base64.b64encode(signature_bytes).decode('utf-8')

# ==========================================
# 3. HÀM XỬ LÝ ẢNH Y KHOA (Y CHANG ĐỒ ÁN)
# ==========================================
def normalize(img):
    mask = img > 0
    if np.sum(mask) > 0:
        mean = img[mask].mean()
        std = img[mask].std()
        return (img - mean) / (std + 1e-8)
    return img

def crop_roi_realtime(image, padding=10):
    coords = np.where(image > 0)
    if len(coords[0]) == 0: return image
    x_min, x_max = max(0, coords[0].min() - padding), min(image.shape[0], coords[0].max() + padding)
    y_min, y_max = max(0, coords[1].min() - padding), min(image.shape[1], coords[1].max() + padding)
    z_min, z_max = max(0, coords[2].min() - padding), min(image.shape[2], coords[2].max() + padding)
    return image[x_min:x_max, y_min:y_max, z_min:z_max]

def volume_to_slices_realtime(volume, size=256):
    images = []
    for i in range(volume.shape[2]):
        img = volume[:, :, i]
        if np.sum(img > 0) < 10: continue
        img_resized = cv2.resize(img, (size, size))
        img_3c = np.stack([img_resized, img_resized, img_resized], axis=-1)
        images.append(img_3c.astype(np.float32))
    return np.array(images)

# ==========================================
# 4. HÀM MAIN: HỨNG MESSAGE TỪ RABBITMQ
# ==========================================
def process_rabbitmq_message(ch, method, properties, body):
    try:
        data = json.loads(body.decode('utf-8'))
        
        # --- BƯỚC 1: XÁC THỰC VÀ GIẢI MÃ TỪ NODE.JS ---
        if not verify_rsa_signature(data['cipherFile'], data['cipherPatient'], data['signature']):
            print("🚨 CẢNH BÁO: CHỮ KÝ GIẢ MẠO! TỪ CHỐI XỬ LÝ.")
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return
            
        file_path = decrypt_aes(data['ivFile'], data['cipherFile'])
        patient_id = decrypt_aes(data['ivPatient'], data['cipherPatient'])
        print(f"✅ Đã tiếp nhận MRI của BN: {patient_id}")

        # --- BƯỚC 2: CẮT ẢNH 3D SANG 2D ---
        print("⚙️ Đang Crop 3D -> 2D...")
        mri_data = nib.load(file_path).get_fdata()
        normalized_mri = normalize(mri_data)
        cropped_mri = crop_roi_realtime(normalized_mri)
        slices_2d = volume_to_slices_realtime(cropped_mri, size=256)
        
        # File đã tiền xử lý xong (là input cho phòng số 2)
        out_path = file_path.replace('.nii.gz', '_2D_ready.npy')
        np.save(out_path, slices_2d)
        print(f"📦 Đã lưu ảnh 2D tại: {out_path}")

        # --- BƯỚC 3: UPDATE MONGODB (STATUS = 1) ---
        time_now = int(time.time() * 1000)
        medical_collection.update_one(
            {"idpatient": patient_id},
            {"$set": {"status": 1, "time": time_now}}
        )
        print("💾 Đã update MongoDB (Status 1: Tiền xử lý xong)")

        # --- BƯỚC 4: MÃ HÓA LẠI ĐỂ GỬI SANG PHÒNG 2 (AI SEGMENT) ---
        print("🔒 Đang mã hóa và đóng mộc cho Phòng 2...")
        # 4.1 Mã hóa đường dẫn npy mới và idpatient
        iv_file_new, cipher_file_new = encrypt_aes_python(out_path)
        iv_patient_new, cipher_patient_new = encrypt_aes_python(patient_id)
        
        # 4.2 Đóng mộc RSA
        signature_new = sign_rsa_python(cipher_file_new, cipher_patient_new)
        
        # 4.3 Đóng gói y chang format của Node.js
        message_to_seg = {
            "ivFile": iv_file_new,
            "cipherFile": cipher_file_new,
            "ivPatient": iv_patient_new,
            "cipherPatient": cipher_patient_new,
            "signature": signature_new
        }
        
        # --- BƯỚC 5: NÉM VÀO QUEUE 2 ---
        # Đảm bảo Queue 2 (Phòng Seg) tồn tại
        queue_2_name = 'AI_SEGMENT_QUEUE'
        ch.queue_declare(queue=queue_2_name, durable=True)
        
        ch.basic_publish(
            exchange='',
            routing_key=queue_2_name,
            body=json.dumps(message_to_seg).encode('utf-8'),
            properties=pika.BasicProperties(delivery_mode=2) # Persistent (Chống mất data khi sập)
        )
        
        print(f"🚀 Đã bắn cục data qua {queue_2_name} thành công!")

        # Báo cáo Node.js đã xong việc ở Queue 1
        ch.basic_ack(delivery_tag=method.delivery_tag)
        print("============================================\n")

    except Exception as e:
        print(f"❌ Lỗi Pipeline: {str(e)}")
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
# ==========================================
# 5. KHỞI ĐỘNG WORKER
# ==========================================
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