import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

file_path = "D:\\medicalSystem\\ServerBackend\\src\\mri_uploads\\565_BrainTumor_1781941410944.nii"

print("📦 Đang bóc hộp NIfTI...")
nii_obj = nib.load(file_path)
volume_3d = nii_obj.get_fdata()

print(f"✅ Kích thước khối 3D gốc: {volume_3d.shape}")
# BraTS thường có shape (240, 240, 155). Mặt cắt ngang (Axial) chuẩn nhất nằm ở trục thứ 3 (size 155).

# 1. LẤY MẶT CẮT NGANG (AXIAL) CHÍNH GIỮA ĐẦU
slice_idx = volume_3d.shape[2] // 2 
axial_slice = volume_3d[:, :, slice_idx]

# 2. XOAY ẢNH LẠI CHO ĐÚNG CHIỀU (Mũi hướng lên trên)
# Mặc định ma trận Numpy sẽ bị xoay ngang, ta dùng rot90 để dựng nó dậy
axial_slice = np.rot90(axial_slice)

# 3. TĂNG ĐỘ TƯƠNG PHẢN (BÍ QUYẾT Ở ĐÂY!)
# Bỏ qua 1% số pixel sáng chói nhất (nhiễu), lấy mốc sáng chuẩn để não "sáng bừng" lên
vmax_chuẩn = np.percentile(axial_slice, 99)

# 4. SHOW ẢNH LÊN
plt.figure(figsize=(6, 6))
# Ép vmax để tăng sáng, dùng cmap='gray' để hiện trắng đen
plt.imshow(axial_slice, cmap='gray', vmax=vmax_chuẩn) 
plt.title(f"🧠 Mặt cắt ngang (Axial) - Lát số {slice_idx}", fontsize=14, fontweight='bold')
plt.axis('off') # Tắt trục tọa độ đi cho pro
plt.show()