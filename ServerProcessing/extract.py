import numpy as np
import matplotlib.pyplot as plt
import cv2

def extract_and_show_masks(npy_path):
    print(f"📦 Đang đọc file kết quả: {npy_path}")
    data = np.load(npy_path)

    # 1. TÌM LÁT CẮT CÓ KHỐI U RÕ NHẤT
    # Ảnh nền xám có R=G=B. Chỗ nào R, G, B lệch nhau tức là chỗ đó được tô màu.
    # Ta tính tổng sự chênh lệch giữa kênh Blue và Red để tìm ra lát cắt có nhiều màu nhất.
    color_diff = np.abs(data[..., 0].astype(int) - data[..., 2].astype(int))
    color_intensity = np.sum(color_diff, axis=(1, 2))
    best_idx = np.argmax(color_intensity)

    if color_intensity[best_idx] == 0:
        print("⚠️ Không tìm thấy khối u nào được tô màu trong khối 3D này!")
        return

    print(f"🎯 Đang bóc tách lát cắt số: {best_idx}")
    img_bgr = data[best_idx]

    # Tách 3 kênh (B, G, R)
    B = img_bgr[:, :, 0].astype(int)
    G = img_bgr[:, :, 1].astype(int)
    R = img_bgr[:, :, 2].astype(int)

    # ==========================================
    # 2. DÙNG TOÁN HỌC ĐỂ BÓC TÁCH TỪNG MASK
    # Ngưỡng lệch màu = 50 (để tránh nhiễu do sai số làm tròn số nguyên)
    # ==========================================
    
    # WT (Màu Xanh Dương): Kênh Blue trội hơn hẳn Red và Green
    wt_mask = ((B > R + 50) & (B > G + 50)).astype(np.uint8)

    # TC (Màu Đỏ): Kênh Red trội hơn hẳn Blue và Green
    tc_mask = ((R > B + 50) & (R > G + 50)).astype(np.uint8)

    # ET (Màu Vàng): Kênh Red và Green đều cao, và trội hơn hẳn Blue
    et_mask = ((R > B + 50) & (G > B + 50)).astype(np.uint8)

    # ==========================================
    # 3. HIỂN THỊ KẾT QUẢ ĐỂ KIỂM CHỨNG
    # ==========================================
    # Đổi BGR sang RGB để Matplotlib hiển thị đúng màu ảnh gốc
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(20, 5))

    # Ảnh kết quả tổng hợp
    plt.subplot(1, 4, 1)
    plt.imshow(img_rgb)
    plt.title(f"Ảnh gốc (Lát {best_idx})", fontsize=16, fontweight='bold')
    plt.axis('off')

    # Mask WT
    plt.subplot(1, 4, 2)
    plt.imshow(wt_mask, cmap='gray')
    plt.title("WT Mask (Trích xuất từ Xanh Dương)", fontsize=14)
    plt.axis('off')

    # Mask TC
    plt.subplot(1, 4, 3)
    plt.imshow(tc_mask, cmap='gray')
    plt.title("TC Mask (Trích xuất từ Đỏ)", fontsize=14)
    plt.axis('off')

    # Mask ET
    plt.subplot(1, 4, 4)
    plt.imshow(et_mask, cmap='gray')
    plt.title("ET Mask (Trích xuất từ Vàng)", fontsize=14)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# ĐƯỜNG DẪN TỚI FILE KẾT QUẢ CỦA ÔNG
# 🚨 Nhớ sửa lại tên file cho đúng nhé
file_path = r"D:\medicalSystem\ServerBackend\src\mri_uploads\008_flair_1782675659893_Segmented_Colored.npy"

extract_and_show_masks(file_path)