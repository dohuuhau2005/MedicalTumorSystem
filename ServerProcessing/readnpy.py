import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import cv2

# ==========================================
# 1. ĐƯỜNG DẪN FILE NPY CẦN XEM
# (Ông copy đường dẫn từ file Segmented_Colored.npy nhé)
# ==========================================
file_path = r"D:\medicalSystem\ServerBackend\src\mri_uploads\123_BrainTumor_1781598428621.nii"

print(f"📦 Đang đọc file: {file_path}...")
data = np.load(file_path)
print(f"✅ Kích thước khối dữ liệu: {data.shape}") 
# Kích thước chuẩn sẽ là: (Số lát cắt, 256, 256, 3)

# ==========================================
# 2. XỬ LÝ MÀU SẮC
# Ở Worker_Segment ta dùng OpenCV (hệ màu BGR),
# Matplotlib lại dùng RGB, nên phải chuyển đổi lại để hiện ĐÚNG MÀU:
# - Xanh lá: WT (Toàn bộ u)
# - Đỏ: TC (Lõi u)
# - Vàng: ET (U tăng quang)
# ==========================================
print("🎨 Đang chuyển đổi hệ màu BGR -> RGB...")
data_rgb = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in data])

# ==========================================
# 3. TẠO GIAO DIỆN XEM ẢNH CÓ THANH CUỘN (SLIDER)
# ==========================================
fig, ax = plt.subplots(figsize=(7, 7))
plt.subplots_adjust(bottom=0.2) # Chừa khoảng trống bên dưới cho thanh cuộn

# Hiển thị lát cắt ở giữa (thường là chỗ thấy khối u rõ nhất)
mid_slice = len(data_rgb) // 2
im = ax.imshow(data_rgb[mid_slice])
ax.set_title(f"🧠 Lát cắt số {mid_slice} / {len(data_rgb)-1}", fontsize=14, fontweight='bold')
ax.axis('off') # Tắt mấy cái trục tọa độ đi cho đẹp

# Thiết lập thanh cuộn
ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
slider = Slider(
    ax=ax_slider, 
    label='Cuộn (Slice)', 
    valmin=0, 
    valmax=len(data_rgb)-1, 
    valinit=mid_slice, 
    valstep=1
)

# Hàm cập nhật hình ảnh khi kéo thanh cuộn
def update(val):
    idx = int(slider.val)
    im.set_data(data_rgb[idx])
    ax.set_title(f"🧠 Lát cắt số {idx} / {len(data_rgb)-1}", fontsize=14, fontweight='bold')
    fig.canvas.draw_idle()

slider.on_changed(update)

print("🚀 Đang mở giao diện xem ảnh...")
plt.show()