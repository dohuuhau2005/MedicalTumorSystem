const amqp = require('amqplib');
const crypto = require('crypto');
const EncryptAES = require('./EncryptAES');
const fs = require('fs');
const path = require('path');

const send = async (idpatient, paths) => {
    try {
        const keyPath = path.join(__dirname, '../../private_chuẩn.pem');
        const PRIVATE_KEY = fs.readFileSync(keyPath, 'utf8');

        // 1. Mã hóa cả 3 file và patient
        const fFlair = EncryptAES(paths.flair);
        const fT1ce = EncryptAES(paths.t1ce);
        const fT2 = EncryptAES(paths.t2);
        const pData = EncryptAES(idpatient);

        // 2. Đóng mộc RSA (Phải ký đầy đủ 4 món theo đúng thứ tự)
        const sign = crypto.createSign('SHA256');
        sign.update(fFlair.encryptedText);
        sign.update(fT1ce.encryptedText);
        sign.update(fT2.encryptedText);
        sign.update(pData.encryptedText);

        const signature = sign.sign(PRIVATE_KEY, 'base64');

        // 3. Gói hàng (Chứa đầy đủ thông tin của 3 kênh)
        const messageToSend = {
            ivFlair: fFlair.iv,
            cipherFlair: fFlair.encryptedText,
            ivT1ce: fT1ce.iv,
            cipherT1ce: fT1ce.encryptedText,
            ivT2: fT2.iv,
            cipherT2: fT2.encryptedText,
            ivPatient: pData.iv,
            cipherPatient: pData.encryptedText,
            signature: signature
        };

        const rabbitUrl = process.env.serverRabitMQ || 'amqp://localhost:5672';
        const connection = await amqp.connect(rabbitUrl);
        const channel = await connection.createChannel();
        const queueName = 'Patient_QUEUE';

        await channel.assertQueue(queueName, { durable: true });
        const bufferData = Buffer.from(JSON.stringify(messageToSend));
        channel.sendToQueue(queueName, bufferData, { persistent: true });

        setTimeout(() => {
            connection.close();
        }, 500);
        console.log("🚀 Đã bắn cục data 3 kênh qua RabbitMQ thành công!");
    } catch (error) {
        console.error("❌ Lỗi gửi RabbitMQ:", error);
    }
};

module.exports = { send };