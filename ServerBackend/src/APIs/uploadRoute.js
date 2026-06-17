const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { send } = require("../config/SeenMessage")
const router = express.Router();
const client = require('../config/MongoDB');
const { time } = require('console');

// 1. Tạo thư mục chứa MRI nếu chưa có (tránh lỗi crash khi thư mục không tồn tại)
const uploadDir = path.join(__dirname, '../mri_uploads');
if (!fs.existsSync(uploadDir)) {
    fs.mkdirSync(uploadDir, { recursive: true });
}
const timeNow = Date.now();
// 2. Cấu hình "Cái rổ" Multer (Nơi lưu và Cách đặt tên)
const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, uploadDir); // Ném file vào thư mục mri_uploads
    },
    filename: function (req, file, cb) {
        // Lấy idpatient từ form-data (Frontend phải gửi lên cùng lúc với file)
        const patientId = req.body.idpatient || 'UnknownBN';
        const type = req.body.type;
        // Lấy đuôi file gốc (thường là .gz hoặc .nii)
        // Nếu file là image.nii.gz, path.extname chỉ lấy được .gz. 
        // Trong y khoa nên lấy tên gốc để giữ nguyên đuôi.
        const originalName = file.originalname;
        const ext = originalName.endsWith('.nii.gz') ? '.nii.gz' : path.extname(originalName);

        // Đóng mộc tên chuẩn: BN10293_BrainTumor_1718079000.nii.gz
        const uniqueName = `${patientId}_${type}_${timeNow}${ext}`;
        cb(null, uniqueName);
    }
});

// Khởi tạo middleware multer
const upload = multer({
    storage: storage,
    limits: { fileSize: 100 * 1024 * 1024 } // Giới hạn 100MB để server không bị nghẽn
});

// =================================================================
// 3. API NHẬN FILE
// Chữ 'mri_file' ở dưới phải khớp y chang với tên field name ở Frontend gửi lên
// =================================================================
router.post('/uploadMRI', upload.single('mri_file'), async (req, res) => {
    try {
        // Lúc này, nhờ upload.single(), req.file và req.body đã hiện nguyên hình
        if (!req.file) {
            return res.status(400).json({ success: false, message: "Không tìm thấy file đính kèm!" });
        }

        const { idpatient, type } = req.body;
        const savedPath = req.file.path; // Đường dẫn tuyệt đối của file vừa lưu

        console.log(`✅ Đã lưu MRI của bệnh nhân ${idpatient} tại: ${savedPath}`);

        await client.connect();
        const db = client.db("Patients");
        const collection = db.collection("medicalSystem");

        const patientDataMongle = {
            idpatient: idpatient,
            status: 0,
            time: timeNow
        }

        //gọi rabbit MQ
        await send(idpatient, savedPath);
        await collection.insertOne(patientDataMongle);
        if (!idpatient || !savedPath) {
            return res.status(400).json({ success: false, message: "Thiếu thông tin bệnh nhân hoặc đường dẫn file!" });
        }
        return res.status(200).json({
            success: true,
            message: "Upload MRI thành công!",
            filename: req.file.filename,
            path: savedPath
        });

    } catch (error) {
        console.error("❌ Lỗi upload:", error);
        return res.status(500).json({ success: false, message: "Lỗi server khi lưu file" });
    }
});

module.exports = router;