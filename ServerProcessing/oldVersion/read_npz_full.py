import numpy as np
import matplotlib.pyplot as plt
import os

def export_all_slices_npz(file_path, output_dir="exported_slices"):
    """
    Hàm đọc file NPZ và XUẤT TOÀN BỘ các lát cắt thành file ảnh lưu vào thư mục.
    """
    if not os.path.exists(file_path):
        print(f"❌ Không tìm thấy file: {file_path}")
        return

    print(f"📂 Đang mở file: {file_path}...")
    
    # 1. Đọc file .npz
    with np.load(file_path) as data:
        original = data['original']
        wt = data['wt']
        tc = data['tc']
        et = data['et']
        overlay = data['overlay']

    total_slices = original.shape[0]
    print(f"✅ Đã tải thành công! Tổng số lát cắt (slices): {total_slices}")
    
    # 2. Tạo thư mục để lưu ảnh (nếu chưa có)
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Ảnh sẽ được lưu tại thư mục: {os.path.abspath(output_dir)}")

    # 3. Vòng lặp xuất toàn bộ các lát cắt
    for slice_idx in range(total_slices):
        
        img_orig = original[slice_idx]
        mask_wt = wt[slice_idx]
        mask_tc = tc[slice_idx]
        mask_et = et[slice_idx]
        img_overlay = overlay[slice_idx]

        # Chuyển đổi BGR sang RGB
        img_overlay_rgb = img_overlay[..., ::-1]

        # Khởi tạo khung vẽ
        fig, axes = plt.subplots(1, 5, figsize=(20, 5))
        fig.suptitle(f"Patient MRI Segmentation - Slice {slice_idx}", fontsize=16)

        # Vẽ từng thành phần
        axes[0].imshow(img_orig[:, :, 0], cmap='gray')
        axes[0].set_title("Original MRI")
        axes[0].axis('off')

        axes[1].imshow(mask_wt, cmap='Blues')
        axes[1].set_title("Whole Tumor (WT)")
        axes[1].axis('off')

        axes[2].imshow(mask_tc, cmap='Reds')
        axes[2].set_title("Tumor Core (TC)")
        axes[2].axis('off')

        axes[3].imshow(mask_et, cmap='Wistia')
        axes[3].set_title("Enhancing Tumor (ET)")
        axes[3].axis('off')

        axes[4].imshow(img_overlay_rgb)
        axes[4].set_title("Color Overlay")
        axes[4].axis('off')

        plt.tight_layout()
        
        # ==========================================
        # ĐIỂM KHÁC BIỆT: LƯU ẢNH THAY VÌ HIỂN THỊ
        # ==========================================
        # Định dạng tên file có số 0 ở đầu (VD: slice_001.png, slice_050.png) để sort cho chuẩn
        file_name = f"slice_{slice_idx:03d}.png"
        save_path = os.path.join(output_dir, file_name)
        
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig) # ⚠️ RẤT QUAN TRỌNG: Đóng figure để giải phóng RAM, chống tràn bộ nhớ
        
        # In tiến độ cho mỗi 10 lát cắt để theo dõi
        if slice_idx % 10 == 0 or slice_idx == total_slices - 1:
            print(f"⏳ Đang xử lý: {slice_idx}/{total_slices - 1}...")

    print("🎉 Hoàn tất! Bạn có thể vào thư mục để xem toàn bộ các lát cắt.")

# ==========================================
# CHẠY THỬ NGHIỆM
# ==========================================
if __name__ == "__main__":
    # Đường dẫn file NPZ
    sample_npz_path = r"D:\medicalSystem\ServerBackend\src\mri_uploads\050_flair_1787045014184_2D_ready_Segmented_Ready.npz" 
    
    # Đặt tên thư mục bạn muốn lưu toàn bộ hình vào (nó sẽ tự tạo thư mục này cùng cấp với code)
    folder_luu_anh = "Danh_Sach_Lat_Cat_MRI"
    
    export_all_slices_npz(sample_npz_path, output_dir=folder_luu_anh)