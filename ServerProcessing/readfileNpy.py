import numpy as np

# Giả sử sau khi Worker 2 giải mã AES xong, ông lấy được đường dẫn này:
file_path = "D:\medicalSystem\ServerBackend\src\mri_uploads/123_BrainTumor_1781499991107_2D_ready.npy"

# 1. ĐỌC FILE NPY LÊN RAM (Chỉ 1 nốt nhạc)
slices_2d = np.load(file_path)

# 2. Kiểm tra xem nó có đúng hình dáng không
print("Kích thước cục data:", slices_2d.shape) 
# Output sẽ ra kiểu: (155, 256, 256, 3) -> Nghĩa là 155 tấm ảnh, kích thước 256x256, 3 kênh màu

# 3. TỐNG THẲNG VÀO MỒM CON AI (Vì data đã được chuẩn hóa Z-score sẵn ở Phòng 1 rồi)
# predictions = model.predict(slices_2d)