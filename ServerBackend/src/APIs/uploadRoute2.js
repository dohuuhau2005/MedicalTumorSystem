const express = require("express");
const multer = require("multer");
const path = require("path");
const fs = require("fs");

const { send } = require("../config/SeenMessage2");
const client = require("../config/MongoDB");

const router = express.Router();

// =========================
// Tạo thư mục upload
// =========================
const uploadDir = path.join(__dirname, "../mri_uploads");

if (!fs.existsSync(uploadDir)) {
    fs.mkdirSync(uploadDir, { recursive: true });
}

// =========================
// Middleware tạo timestamp chung
// =========================
const setUploadTime = (req, res, next) => {
    req.uploadTime = Date.now();
    next();
};

// =========================
// Multer Storage
// =========================
const storage = multer.diskStorage({
    destination(req, file, cb) {
        cb(null, uploadDir);
    },

    filename(req, file, cb) {
        const patientId = req.body.idpatient || "UnknownBN";
        const type = file.fieldname; // t1ce | t2 | flair
        const timestamp = req.uploadTime;

        const originalName = file.originalname;

        const ext = originalName.endsWith(".nii.gz")
            ? ".nii.gz"
            : path.extname(originalName);

        cb(null, `${patientId}_${type}_${timestamp}${ext}`);
    }
});

const upload = multer({
    storage,
    limits: {
        fileSize: 300 * 1024 * 1024
    }
});

const uploadFields = upload.fields([
    { name: "t1ce", maxCount: 1 },
    { name: "t2", maxCount: 1 },
    { name: "flair", maxCount: 1 }
]);

// =========================
// Upload MRI
// =========================
router.post(
    "/uploadMRI",
    setUploadTime,
    uploadFields,
    async (req, res) => {
        try {
            if (
                !req.files?.t1ce ||
                !req.files?.t2 ||
                !req.files?.flair
            ) {
                return res.status(400).json({
                    success: false,
                    message: "Thiếu file! Cần đủ T1CE, T2 và FLAIR."
                });
            }

            const { idpatient } = req.body;

            if (!idpatient) {
                return res.status(400).json({
                    success: false,
                    message: "Thiếu mã bệnh nhân."
                });
            }

            const paths = {
                t1ce: req.files.t1ce[0].path,
                t2: req.files.t2[0].path,
                flair: req.files.flair[0].path
            };

            const filenames = {
                t1ce: req.files.t1ce[0].filename,
                t2: req.files.t2[0].filename,
                flair: req.files.flair[0].filename
            };

            console.log(`✅ MRI của ${idpatient}`);
            console.log(paths);

            await client.connect();

            const db = client.db("Patients");
            const collection = db.collection("medicalSystem");

            const patientDataMongo = {
                idpatient,
                status: 0,
                time: req.uploadTime,
                files: filenames,
                paths: paths
            };

            // Gửi RabbitMQ
            await send(idpatient, paths);

            // Lưu Mongo
            await collection.insertOne(patientDataMongo);

            return res.status(200).json({
                success: true,
                message: "Upload MRI thành công!",
                filenames,
                paths
            });

        } catch (err) {

            // rollback nếu upload xong nhưng xử lý lỗi
            if (req.files) {
                Object.values(req.files).forEach(fileArray => {
                    fileArray.forEach(file => {
                        if (fs.existsSync(file.path)) {
                            fs.unlinkSync(file.path);
                        }
                    });
                });
            }

            console.error(err);

            return res.status(500).json({
                success: false,
                message: "Lỗi server khi upload MRI."
            });
        }
    }
);

module.exports = router;