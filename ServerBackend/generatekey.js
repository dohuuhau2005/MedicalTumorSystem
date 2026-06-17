const crypto = require('crypto');
const fs = require('fs');

// Rèn cặp chìa khóa RSA mới toanh, chuẩn form PKCS#8
const { publicKey, privateKey } = crypto.generateKeyPairSync('rsa', {
    modulusLength: 2048,
    publicKeyEncoding: { type: 'spki', format: 'pem' },
    privateKeyEncoding: { type: 'pkcs8', format: 'pem' }
});

// Lưu thẳng ra ổ cứng
fs.writeFileSync('private_chuẩn.pem', privateKey);
fs.writeFileSync('public_chuẩn.pem', publicKey);

console.log("✅ Đã rèn xong cặp Key chuẩn 100% của Node.js!");