const amqp = require('amqplib');
const crypto = require('crypto');
const EncryptAES = require('./EncryptAES');
const fs = require('fs');
const path = require('path');
const send = async (idpatient, idfile) => {
    try {
        const keyPath = path.join(__dirname, '../../private_chuẩn.pem');
        const PRIVATE_KEY = fs.readFileSync(keyPath, 'utf8');

        const file = EncryptAES(idfile);
        const iv = file.iv;
        const encryptedText = file.encryptedText;

        const PatientData = EncryptAES(idpatient);
        const ivPatient = PatientData.iv;
        const encryptedPatient = PatientData.encryptedText;




        //  đóng mộc RSA
        const sign = crypto.createSign('SHA256');
        sign.update(encryptedText);
        sign.update(encryptedPatient);
        const signature = sign.sign(PRIVATE_KEY, 'base64');
        console.log(`message chuẩn bị gửi đi nè ${iv},${encryptedText},${ivPatient},${encryptedPatient}`)
        // Gói hàng
        const messageToSend = {
            ivFile: iv,
            cipherFile: encryptedText,
            ivPatient: ivPatient,
            cipherPatient: encryptedPatient,
            signature: signature
        };

        console.log(`đã gửi ${messageToSend}`)

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
        console.log("🚀 Đã bắn cục data mã hóa qua RabbitMQ thành công!");
    } catch (error) {
        console.error("❌ Lỗi gửi RabbitMQ:", error);
    }
};
module.exports = { send };