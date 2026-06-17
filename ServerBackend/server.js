// msBenhNhan_BrainTumor_timestamp
require("dotenv").config();
const express = require("express");
const cors = require("cors");
const app = express();
const corsOptions = {
    origin: function (origin, callback) {
        // Cho phép các request không có origin (như Postman) hoặc từ localhost:3000
        const allowedOrigins = ['http://localhost:3000', 'http://192.168.1.10:3000', 'http://192.168.195.89:3000', 'http://127.0.0.1:3000'];
        if (!origin || allowedOrigins.indexOf(origin) !== -1) {
            callback(null, true);
        } else {
            callback(new Error('Chặn bởi CORS: Origin không được phép'));
        }
    },
    credentials: true, // Bắt buộc phải có để nhận Cookie/Authorization header
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With', 'Accept'],
    optionsSuccessStatus: 204 // Một số trình duyệt cũ (IE11) yêu cầu 204 thay vì 200
};

app.use(cors(corsOptions));
app.get("/", (req, res) => {
    res.send("Server Backend Nhan Request Dang Chay");
})

app.use(express.json())

app.use("/file", require("./src/APIs/uploadRoute"))










const PORT = process.env.port_serverBackend || 9999;
app.listen(PORT, '0.0.0.0', () => {
    console.log("========================================");
    console.log(`🚀 Server is running on port ${PORT}`);
    console.log(`🔗 Access at: http://localhost:${PORT}`);
    console.log("========================================");
});