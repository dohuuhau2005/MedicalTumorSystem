const crypto = require('crypto');
const EncryptAES = (Text) => {
    const AES_SECRET_KEY = crypto.createHash('sha256').update(process.env.AESKey).digest();
    const iv = crypto.randomBytes(16);
    const cipher = crypto.createCipheriv('aes-256-cbc', AES_SECRET_KEY, iv);

    //let để cộng chuỗi
    let encryptedText = cipher.update(Text, 'utf8', 'hex');
    encryptedText += cipher.final('hex');

    // BẮT BUỘC TRẢ VỀ CẢ IV VÀ CHUỖI MÃ HÓA (Dùng Object)
    return {
        iv: iv.toString('hex'), // Đổi ra hex luôn cho dễ xài
        encryptedText: encryptedText
    };
};
module.exports = EncryptAES;
