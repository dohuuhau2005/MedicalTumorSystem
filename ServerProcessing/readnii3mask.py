import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

file_path = r"D:\medicalSystem\ServerBackend\src\mri_uploads\BraTS20_Training_001_seg.nii"

print("📦 Đang đọc file...")
nii_obj = nib.load(file_path)
seg = nii_obj.get_fdata()

print(f"✅ Shape: {seg.shape}")

# Lát cắt muốn xem
slice_idx = 60

# Lấy slice
slice_seg = seg[:, :, slice_idx]

# Tạo các mask BraTS
WT = (slice_seg > 0)                        # Whole Tumor = 1 + 2 + 4
TC = np.logical_or(slice_seg == 1,
                   slice_seg == 4)          # Tumor Core = 1 + 4
ET = (slice_seg == 4)                       # Enhancing Tumor = 4

# Xoay cho dễ nhìn
WT = np.rot90(WT)
TC = np.rot90(TC)
ET = np.rot90(ET)

# Hiển thị
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(WT, cmap='gray')
axes[0].set_title("WT (Whole Tumor)")
axes[0].axis('off')

axes[1].imshow(TC, cmap='gray')
axes[1].set_title("TC (Tumor Core)")
axes[1].axis('off')

axes[2].imshow(ET, cmap='gray')
axes[2].set_title("ET (Enhancing Tumor)")
axes[2].axis('off')

plt.suptitle(f"BraTS Slice {slice_idx}")
plt.tight_layout()
plt.show()