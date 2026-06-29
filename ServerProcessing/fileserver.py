import os
import json
import base64
import cv2
import numpy as np
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
CORS(app)

# Thư mục chứa các file .npz kết quả
UPLOAD_DIR = os.getenv('UPLOAD_DIR', r'D:\medicalSystem\ServerBackend\src\mri_uploads')

# --- HÀM HỖ TRỢ CHUYỂN NPZ THÀNH ẢNH BASE64 ---
def npz_to_base64(npz_path, key='overlay'):
    if not os.path.exists(npz_path): 
        return None
        
    data = np.load(npz_path)
    if key not in data: 
        return None
    
    img_array = data[key] # Ma trận thường có dạng (Slices, Height, Width)
    
    # 🌟 SỬA LỖI 1: Tự động tìm lát cắt (slice) có nhiều vùng u nhất, không lấy bừa ở giữa nữa
    # Tính tổng giá trị pixel của từng lát cắt
    slice_sums = np.sum(img_array, axis=(1, 2))
    max_slice_val = np.max(slice_sums)
    
    if max_slice_val > 0:
        best_idx = np.argmax(slice_sums) # Lát cắt chứa u nhiều nhất
    else:
        best_idx = img_array.shape[0] // 2 # Nếu không có u thì mới lấy ở giữa
        
    img = img_array[best_idx]
    
    # 🌟 SỬA LỖI 2: Chuẩn hóa ma trận về thang độ sáng 0 - 255 để hiển thị được
    max_pixel = np.max(img)
    
    if max_pixel > 0:
        if img.dtype == np.float32 or img.dtype == np.float64 or max_pixel <= 1.0:
            # Nếu mask ở dạng float (0.0 -> 1.0) hoặc binary mask (0 và 1)
            img = (img * 255).astype(np.uint8)
        else:
            # Nếu mask ở dạng nhãn lớp (0, 1, 2, 3...), khuếch đại lên max 255 để nhìn rõ
            img = (img * (255 / max_pixel)).astype(np.uint8)
    else:
        img = img.astype(np.uint8) # Toàn số 0 thì chịu chết, vẫn là đen thôi
        
    # Mã hóa thành PNG Base64
    _, buffer = cv2.imencode('.png', img)
    return "data:image/png;base64," + base64.b64encode(buffer).decode('utf-8')
# --- API 1: Tải file .npz gốc ---
@app.route('/results/file', methods=['GET'])
def download_file():
    filename = request.args.get('filename')
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(UPLOAD_DIR, safe_filename)
    
    if not os.path.exists(file_path):
        return jsonify({"error": "File không tồn tại"}), 404
    return send_file(file_path, as_attachment=True)

# --- API 2: Trả về ảnh Base64 cho React xem nhanh ---
@app.route('/results/image', methods=['GET'])
def get_image():
    filename = request.args.get('filename')
    # key: 'overlay', 'wt', 'tc', hoặc 'et'
    key = request.args.get('key', 'overlay') 
    
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(UPLOAD_DIR, safe_filename)
    
    img_b64 = npz_to_base64(file_path, key)
    if not img_b64:
        return jsonify({"error": "Không thể trích xuất ảnh"}), 404
        
    return jsonify({"image": img_b64})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9004)