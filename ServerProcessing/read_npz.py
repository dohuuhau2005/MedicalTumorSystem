import numpy as np
import matplotlib.pyplot as plt
import os

def read_and_visualize_npz(file_path, slice_idx=None):
    """
    Hàm đọc và hiển thị file NPZ chuẩn y khoa từ hệ thống AI Segmentation
    """
    if not os.path.exists(file_path):
        print(f"❌ Không tìm thấy file: {file_path}")
        return

    print(f"📂 Đang mở file: {file_path}...")
    
    # 1. Đọc file .npz
    with np.load(file_path) as data:
        # Lấy ra các mảng dữ liệu đã được lưu
        original = data['original']
        wt = data['wt']
        tc = data['tc']
        et = data['et']
        overlay = data['overlay']

    total_slices = original.shape[0]
    print(f"✅ Đã tải thành công! Tổng số lát cắt (slices): {total_slices}")
    
    # 2. Chọn lát cắt để hiển thị
    # Nếu không truyền slice_idx, mặc định sẽ lấy lát cắt ở giữa (thường chứa u rõ nhất)
    if slice_idx is None:
        slice_idx = total_slices // 2
        
    if slice_idx >= total_slices or slice_idx < 0:
        print(f"⚠️ Lát cắt {slice_idx} không hợp lệ. Vui lòng chọn từ 0 đến {total_slices - 1}.")
        return

    print(f"👁️ Đang hiển thị lát cắt số: {slice_idx}")

    # Lấy dữ liệu của lát cắt được chọn
    img_orig = original[slice_idx]
    mask_wt = wt[slice_idx]
    mask_tc = tc[slice_idx]
    mask_et = et[slice_idx]
    img_overlay = overlay[slice_idx]

    # Vì overlay lưu bằng OpenCV (hệ BGR) trong pipeline của bạn, 
    # cần chuyển đổi sang RGB để matplotlib hiển thị đúng màu.
    img_overlay_rgb = img_overlay[..., ::-1]

    # 3. Vẽ hình (Plotting)
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    fig.suptitle(f"Patient MRI Segmentation - Slice {slice_idx}", fontsize=16)

    # Hiển thị ảnh gốc (Chỉ lấy kênh 0 để hiển thị xám cho chuẩn)
    axes[0].imshow(img_orig[:, :, 0], cmap='gray')
    axes[0].set_title("Original MRI")
    axes[0].axis('off')

    # Hiển thị WT (Whole Tumor)
    axes[1].imshow(mask_wt, cmap='Blues')
    axes[1].set_title("Whole Tumor (WT)")
    axes[1].axis('off')

    # Hiển thị TC (Tumor Core)
    axes[2].imshow(mask_tc, cmap='Reds')
    axes[2].set_title("Tumor Core (TC)")
    axes[2].axis('off')

    # Hiển thị ET (Enhancing Tumor)
    axes[3].imshow(mask_et, cmap='Wistia')
    axes[3].set_title("Enhancing Tumor (ET)")
    axes[3].axis('off')

    # Hiển thị Ảnh Overlay (Đã Blend màu)
    axes[4].imshow(img_overlay_rgb)
    axes[4].set_title("Color Overlay")
    axes[4].axis('off')

    plt.tight_layout()
    plt.show()

# ==========================================
# CHẠY THỬ NGHIỆM
# ==========================================
if __name__ == "__main__":
    # Thay đường dẫn này bằng đường dẫn tới file .npz thực tế của bạn
    sample_npz_path = r"D:\medicalSystem\ServerBackend\src\mri_uploads\024_flair_1783431439023_2D_ready_Segmented_Ready.npz" 
    
    # Xem lát cắt ở giữa
    read_and_visualize_npz(sample_npz_path)
    
    # Hoặc xem một lát cắt cụ thể (ví dụ lát cắt số 70)
    # read_and_visualize_npz(sample_npz_path, slice_idx=70)