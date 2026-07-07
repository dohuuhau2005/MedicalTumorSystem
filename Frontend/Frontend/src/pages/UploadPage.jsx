import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './UploadPage.css';

const UploadPage = () => {
    // State dữ liệu form
    const [formData, setFormData] = useState({
        idpatient: '',
        t1ce: null,
        t2: null,
        flair: null
    });

    // State quản lý tiến trình & kết quả
    const [isLoading, setIsLoading] = useState(false);
    const [currentStatus, setCurrentStatus] = useState(0);
    const [result, setResult] = useState(null);
    const [error, setError] = useState('');

    const pollingInterval = useRef(null);

    const handleInputChange = (e) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
    };

    const handleFileChange = (e) => {
        setFormData({ ...formData, [e.target.name]: e.target.files[0] });
    };

    // Hàm gọi API để kiểm tra trạng thái bệnh nhân
    const checkPatientStatus = async (patientId) => {
        try {
            const response = await axios.get(`http://localhost:9999/patients/${patientId}`);

            if (response.data && response.data.patient) {
                const patientData = response.data.patient;
                const status = patientData.status;

                setCurrentStatus(status);
                if (status === -1) {
                    clearInterval(pollingInterval.current);
                    setIsLoading(false);
                    setError("Hệ thống AI gặp sự cố khi xử lý file của bệnh nhân này. Vui lòng kiểm tra lại file MRI.");
                }
                // Nếu status = 3 (Hoàn tất phân loại và phân vùng)
                if (status === 3) {
                    // 1. Dừng Polling
                    clearInterval(pollingInterval.current);
                    setIsLoading(false);

                    let fileName = "";
                    let imgWT = "";
                    let imgTC = "";
                    let imgET = "";

                    // 2. Cắt tên file và gọi sang server Flask lấy Base64
                    if (patientData.result_file) {
                        // Lấy đúng tên file
                        fileName = patientData.result_file.split(/[/\\]/).pop();

                        try {
                            // Dùng Promise.all để gọi 3 API cùng lúc, tiết kiệm thời gian chờ
                            const [wtRes, tcRes, etRes] = await Promise.all([
                                axios.get(`http://127.0.0.1:9004/results/image?filename=${fileName}&key=wt`),
                                axios.get(`http://127.0.0.1:9004/results/image?filename=${fileName}&key=tc`),
                                axios.get(`http://127.0.0.1:9004/results/image?filename=${fileName}&key=et`)
                            ]);

                            // Gán Base64 vào biến
                            imgWT = wtRes.data.image || "";
                            imgTC = tcRes.data.image || "";
                            imgET = etRes.data.image || "";

                            console.log("🔥 Đã lấy thành công 3 ảnh Base64 từ Flask!");

                        } catch (downloadErr) {
                            console.error("Lỗi khi tải ảnh base64 từ Flask server:", downloadErr);
                        }
                    }

                    // 3. Cập nhật state result để hiển thị lên UI
                    setResult({
                        rawConfidence: patientData.confidence, // <-- BỔ SUNG ĐỂ XỬ LÝ LOGIC
                        accuracy: patientData.confidence ? `${patientData.confidence}%` : "Chưa xác định",
                        diagnosis: patientData.diagnosis || "Đang cập nhật...",
                        fileName: fileName,
                        wt: imgWT,
                        tc: imgTC,
                        et: imgET
                    });

                    // 4. Dọn dẹp localStorage
                    localStorage.removeItem('active_patient_id');
                }
            }
        } catch (err) {
            console.error('Lỗi khi check status:', err);
        }
    };

    // Nhấn nút dự đoán (Gửi file & kích hoạt polling)
    const handleSubmit = async (e) => {
        e.preventDefault();

        setIsLoading(true);
        setError('');
        setResult(null);
        setCurrentStatus(0);

        const submitData = new FormData();
        submitData.append('idpatient', formData.idpatient);
        submitData.append('t1ce', formData.t1ce);
        submitData.append('t2', formData.t2);
        submitData.append('flair', formData.flair);

        try {
            await axios.post('http://localhost:9999/file/uploadMRI', submitData, {
                headers: { 'Content-Type': 'multipart/form-data' },
            });

            const targetPatientId = formData.idpatient;
            localStorage.setItem('active_patient_id', targetPatientId);

            if (pollingInterval.current) clearInterval(pollingInterval.current);

            pollingInterval.current = setInterval(() => {
                checkPatientStatus(targetPatientId);
            }, 1500);

        } catch (err) {
            console.error('Lỗi khi upload file:', err);
            setError('Có lỗi khi upload dữ liệu MRI lên server.');
            setIsLoading(false);
        }
    };

    useEffect(() => {
        return () => {
            if (pollingInterval.current) clearInterval(pollingInterval.current);
        };
    }, []);

    return (
        <div className="upload-page">
            <div className="overlay"></div>

            <nav>
                <div className="logo"><h2 style={{ color: 'white', margin: 0 }}>MRI AI System</h2></div>
                <div className="nav-links">
                    <a href="/">Home</a>
                    <a href="/upload" className="active">Upload</a>

                    <a href="/about">About</a>
                    <a href="/history">History</a>
                </div>
            </nav>

            <main className="main-container">
                {/* ---------------- CARD UPLOAD ---------------- */}
                <div className="glass-card upload-section">
                    <h2>Tải lên dữ liệu MRI</h2>
                    <p>Vui lòng nhập ID và tải lên các tệp định dạng .nii</p>

                    <form onSubmit={handleSubmit}>
                        <div className="form-group">
                            <label htmlFor="idpatient">ID Bệnh nhân</label>
                            <input type="text" id="idpatient" name="idpatient"
                                required placeholder="Ví dụ: 011"
                                value={formData.idpatient} onChange={handleInputChange}
                                disabled={isLoading}
                            />
                        </div>

                        <div className="form-group">
                            <label htmlFor="t1ce">T1CE Image</label>
                            <input type="file" id="t1ce" name="t1ce" accept=".nii, .nii.gz"
                                required onChange={handleFileChange} disabled={isLoading}
                            />
                        </div>

                        <div className="form-group">
                            <label htmlFor="t2">T2 Image</label>
                            <input type="file" id="t2" name="t2" accept=".nii, .nii.gz"
                                required onChange={handleFileChange} disabled={isLoading}
                            />
                        </div>

                        <div className="form-group">
                            <label htmlFor="flair">FLAIR Image</label>
                            <input type="file" id="flair" name="flair" accept=".nii, .nii.gz"
                                required onChange={handleFileChange} disabled={isLoading}
                            />
                        </div>

                        {error && <div className="error-message">{error}</div>}

                        <button type="submit" className="btn-danger" disabled={isLoading}>
                            {isLoading ? 'Đang phân tích...' : 'Dự đoán'}
                        </button>
                    </form>

                    {/* TRACKER TIẾN TRÌNH AI */}
                    {isLoading && (
                        <div className="status-tracker">
                            <p className="tracker-title">Tiến trình xử lý mô hình AI:</p>
                            <div className="steps-container">
                                <div className={`step-item ${currentStatus >= 1 ? 'active' : ''}`}>
                                    <div className="step-circle">1</div>
                                    <span>Tiền xử lý</span>
                                </div>
                                <div className={`step-line ${currentStatus >= 2 ? 'active' : ''}`}></div>
                                <div className={`step-item ${currentStatus >= 2 ? 'active' : ''}`}>
                                    <div className="step-circle">2</div>
                                    <span>Segmentation</span>
                                </div>
                                <div className={`step-line ${currentStatus >= 3 ? 'active' : ''}`}></div>
                                <div className={`step-item ${currentStatus >= 3 ? 'active' : ''}`}>
                                    <div className="step-circle">3</div>
                                    <span>Classification</span>
                                </div>
                            </div>
                        </div>
                    )}
                </div>

                {/* ---------------- CARD RESULT ---------------- */}
                {result && (
                    <div className="glass-card result-section">
                        <h2>Kết quả phân tích hoàn tất</h2>

                        <div className="classification-box">
                            <h3>Chẩn đoán: <span className="text-accent">{result.diagnosis}</span></h3>
                            <h4 style={{ marginTop: '10px', color: 'var(--text2)' }}>
                                Độ chính xác (Confidence): <span style={{ color: '#00ffcc' }}>{result.accuracy}</span>
                            </h4>

                            {/* --- BỔ SUNG CẢNH BÁO NẾU DƯỚI 60% --- */}
                            {result.rawConfidence && result.rawConfidence < 60 && (
                                <div style={{
                                    marginTop: '15px',
                                    padding: '12px',
                                    border: '1px solid #ff4d4d',
                                    borderRadius: '8px',
                                    backgroundColor: 'rgba(255, 77, 77, 0.15)',
                                    color: '#ff4d4d',
                                    fontWeight: 'bold',
                                    display: 'inline-block'
                                }}>
                                    ⚠️ CẢNH BÁO: Độ tin cậy thấp. Yêu cầu hội chẩn chuyên gia y tế!
                                </div>
                            )}
                        </div>

                        <div className="segmentation-grid">
                            <div className="seg-item">
                                <h4>WT (Whole Tumor)</h4>
                                <div className="img-placeholder">
                                    {result.wt ? <img src={result.wt} alt="WT Result" /> : <span style={{ fontSize: '12px', color: 'var(--text2)' }}>Lỗi tải ảnh WT</span>}
                                </div>
                            </div>
                            <div className="seg-item">
                                <h4>TC (Tumor Core)</h4>
                                <div className="img-placeholder">
                                    {result.tc ? <img src={result.tc} alt="TC Result" /> : <span style={{ fontSize: '12px', color: 'var(--text2)' }}>Lỗi tải ảnh TC</span>}
                                </div>
                            </div>
                            <div className="seg-item">
                                <h4>ET (Enhancing Tumor)</h4>
                                <div className="img-placeholder">
                                    {result.et ? <img src={result.et} alt="ET Result" /> : <span style={{ fontSize: '12px', color: 'var(--text2)' }}>Lỗi tải ảnh ET</span>}
                                </div>
                            </div>
                        </div>
                    </div>
                )}
            </main>
        </div>
    );
};

export default UploadPage;