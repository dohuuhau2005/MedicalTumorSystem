import numpy as np
import matplotlib.pyplot as plt

# 1. Trỏ đến file .npy do AI VỪA TẠO RA
ai_output_path = r"D:\medicalSystem\ServerBackend\src\mri_uploads\56578_BrainTumor_1782018044624.nii.npy"

# 2. Đọc file
ai_images = np.load(ai_output_path)

# 3. Lấy lát cắt ở giữa để xem
slice_idx = len(ai_images) // 2
img_2d = ai_images[slice_idx]

# 4. Hiển thị (Bản thân file .npy này đã được AI tô màu sẵn rồi)
plt.figure(figsize=(6, 6))
# Đổi BGR (của OpenCV) sang RGB (của Matplotlib) để màu không bị ngược
img_rgb = img_2d[:, :, ::-1] 

plt.imshow(img_rgb) 
plt.title(f"Kết quả AI tô màu - Lát số {slice_idx}", fontsize=14, fontweight='bold')
plt.axis('off')
plt.show()