import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# 1. Nạp file đáp án gốc
seg_path = r"D:\\medicalSystem\\ServerBackend\\src\\mri_uploads\\56578_BrainTumor_1782018044624.nii"
seg_data = nib.load(seg_path).get_fdata()

# Lấy lát cắt ngang ở giữa
slice_idx = seg_data.shape[2] // 2
mask_2d = seg_data[:, :, slice_idx]
mask_2d = np.rot90(mask_2d) # Xoay cho đúng chiều

# 2. TẠO BẢNG MÀU RGB TRỐNG
color_mask = np.zeros((mask_2d.shape[0], mask_2d.shape[1], 3), dtype=np.uint8)

# 3. ÁNH XẠ CÁC CON SỐ SANG MÀU SẮC (Chuẩn Matplotlib: R-G-B)
# Vùng 2 (Edema - Thuộc phần rìa của WT) -> Xanh lá
color_mask[mask_2d == 2] = [0, 255, 0]

# Vùng 1 (Necrotic - Lõi u TC) -> Đỏ
color_mask[mask_2d == 1] = [255, 0, 0]

# Vùng 4 (Enhancing - U tăng quang ET) -> Vàng
color_mask[mask_2d == 4] = [255, 255, 0]

# 4. SHOW ẢNH ĐÁP ÁN ĐÃ TÔ MÀU
plt.figure(figsize=(6, 6))
plt.imshow(color_mask)
plt.title(f"Đáp án chuẩn gốc (Đã tô màu) - Lát số {slice_idx}", fontsize=14, fontweight='bold')
plt.axis('off')
plt.show()