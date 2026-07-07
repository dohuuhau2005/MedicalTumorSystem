const express = require("express");
const multer = require("multer");
const path = require("path");
const fs = require("fs");

const { send } = require("../config/SeenMessage2");
const client = require("../config/MongoDB");

const router = express.Router();
// API Lấy danh sách lịch sử khám của 1 bệnh nhân
router.get('/patients/:id/history', async (req, res) => {
    try {
        const patientId = req.params.id;
        const db = client.db("Patients");
        // Truy vấn bảng patientHistory, sắp xếp timestamp giảm dần (-1)
        const historyList = await db.collection('patientHistory')
            .find({ idpatient: patientId })
            .sort({ timestamp: -1 })
            .toArray();

        res.status(200).json({
            success: true,
            total: historyList.length,
            history: historyList
        });
    } catch (error) {
        console.error("Lỗi lấy lịch sử:", error);
        res.status(500).json({ success: false, message: "Lỗi server" });
    }
});
module.exports = router;