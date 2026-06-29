const express = require('express');
const path = require('path');
const fs = require('fs');
const router = express.Router();

// Cấu hình đường dẫn thư mục gốc nơi chứa các file npz
// Bác sửa lại đường dẫn này cho khớp với thư mục thực tế của project nhé
const UPLOAD_DIR = path.join(__dirname, '../src/mri_uploads');

// Endpoint: GET results/fileName?filename=011_flair_1782697151037_2D_ready_Segmented_Colored.npz
router.get('/fileName', (req, res) => {
    try {
        // 1. Nhận tên file từ Frontend gửi lên
        const rawFilename = req.query.filename;

        if (!rawFilename) {
            return res.status(400).json({ error: "Thiếu tham số filename" });
        }

        // 2. 🛡️ CHỐT CHẶN BẢO MẬT (Chống Path Traversal):
        // path.basename sẽ bóc tách và loại bỏ mọi đường dẫn thừa (như C:\ hoặc ../../)
        // Đảm bảo chỉ còn lại đúng cái tên file thuần túy (VD: 011_flair_...npz)
        const safeFilename = path.basename(rawFilename);

        // 3. Ghép tên file an toàn vào thư mục chứa kết quả chuẩn
        const absolutePath = path.join(UPLOAD_DIR, safeFilename);

        // 4. Kiểm tra file có tồn tại thật trên ổ cứng không
        if (!fs.existsSync(absolutePath)) {
            return res.status(404).json({ error: "Không tìm thấy file kết quả tại server!" });
        }

        // 5. Stream thẳng file về cho Frontend
        res.sendFile(absolutePath, (err) => {
            if (err) {
                console.error("Lỗi khi stream file về React:", err);
                if (!res.headersSent) {
                    res.status(500).json({ error: "Lỗi trong quá trình tải file" });
                }
            }
        });

    } catch (error) {
        console.error("Lỗi API tải file:", error);
        res.status(500).json({ error: "Lỗi server nội bộ" });
    }
});

module.exports = router;