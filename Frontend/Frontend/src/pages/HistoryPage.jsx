import React, { useState } from 'react';
import axios from 'axios';
import './UploadPage.css';

const HistoryPage = () => {
  const [patientId, setPatientId] = useState('');
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [selected, setSelected] = useState(null);
  const [masks, setMasks] = useState({ wt: '', tc: '', et: '' });

  const handleSearch = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setHistory([]);
    setSelected(null);
    setMasks({ wt: '', tc: '', et: '' });
    try {
      const res = await axios.get(`http://localhost:9999/results/patients/${patientId}/history`);
      if (res.data && res.data.success) {
        setHistory(res.data.history || []);
      } else {
        setError('Không tìm thấy lịch sử cho bệnh nhân này.');
      }
    } catch (err) {
      setError('Lỗi khi lấy lịch sử.');
    }
    setLoading(false);
  };

  const handleSelect = async (item) => {
    setSelected(item);
    setMasks({ wt: '', tc: '', et: '' });
    try {
      const fileName = item.result_file_name;
      const [wt, tc, et] = await Promise.all([
        axios.get(`http://127.0.0.1:9004/results/image?filename=${fileName}&key=wt`),
        axios.get(`http://127.0.0.1:9004/results/image?filename=${fileName}&key=tc`),
        axios.get(`http://127.0.0.1:9004/results/image?filename=${fileName}&key=et`)
      ]);
      setMasks({
        wt: wt.data.image || '',
        tc: tc.data.image || '',
        et: et.data.image || ''
      });
    } catch (err) {
      setError('Lỗi khi tải ảnh mask.');
    }
  };

  return (
    <div className="upload-page">
      <div className="overlay"></div>
      <nav>
        <div className="logo"><h2 style={{ color: 'white', margin: 0 }}>MRI AI System</h2></div>
        <div className="nav-links">
          <a href="/">Home</a>
          <a href="/upload">Upload</a>
          <a href="/about">About</a>
          <a href="/history" className="active">History</a>
        </div>
      </nav>
      <main className="main-container">
        <div className="glass-card upload-section">
          <h2>Xem lịch sử bệnh nhân</h2>
          <form onSubmit={handleSearch}>
            <div className="form-group">
              <label htmlFor="idpatient">ID Bệnh nhân</label>
              <input
                type="text"
                id="idpatient"
                name="idpatient"
                required
                placeholder="Nhập ID bệnh nhân"
                value={patientId}
                onChange={e => setPatientId(e.target.value)}
                disabled={loading}
              />
            </div>
            <button type="submit" className="btn-danger" disabled={loading}>
              {loading ? 'Đang tìm...' : 'Tìm lịch sử'}
            </button>
          </form>
          {error && <div className="error-message">{error}</div>}
          {history.length > 0 && (
            <div style={{ marginTop: 30 }}>
              <h3>Lịch sử lưu ({history.length})</h3>
              <ul style={{ padding: 0, listStyle: 'none' }}>
                {history.map(item => (
                  <li key={item._id} style={{ marginBottom: 15 }}>
                    <button
                      style={{
                        background: selected && selected._id === item._id ? '#7E2DB8' : '#402B88',
                        color: 'white',
                        border: 'none',
                        borderRadius: 8,
                        padding: '10px 18px',
                        cursor: 'pointer',
                        width: '100%',
                        textAlign: 'left',
                        fontWeight: 'bold'
                      }}
                      onClick={() => handleSelect(item)}
                    >
                      {item.result_file_name} | {item.diagnosis} | {item.confidence}% | {new Date(item.timestamp).toLocaleString()}
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
        {selected && (
          <div className="glass-card result-section">
            <h2>Kết quả lịch sử</h2>
            <div className="classification-box">
              <h3>Chẩn đoán: <span className="text-accent">{selected.diagnosis}</span></h3>
              <h4 style={{ marginTop: '10px', color: 'var(--text2)' }}>
                Độ chính xác (Confidence): <span style={{ color: '#00ffcc' }}>{selected.confidence}%</span>
              </h4>
            </div>
            <div className="segmentation-grid">
              <div className="seg-item">
                <h4>WT (Whole Tumor)</h4>
                <div className="img-placeholder">
                  {masks.wt ? <img src={masks.wt} alt="WT Result" /> : <span style={{ fontSize: '12px', color: 'var(--text2)' }}>Chưa tải ảnh WT</span>}
                </div>
              </div>
              <div className="seg-item">
                <h4>TC (Tumor Core)</h4>
                <div className="img-placeholder">
                  {masks.tc ? <img src={masks.tc} alt="TC Result" /> : <span style={{ fontSize: '12px', color: 'var(--text2)' }}>Chưa tải ảnh TC</span>}
                </div>
              </div>
              <div className="seg-item">
                <h4>ET (Enhancing Tumor)</h4>
                <div className="img-placeholder">
                  {masks.et ? <img src={masks.et} alt="ET Result" /> : <span style={{ fontSize: '12px', color: 'var(--text2)' }}>Chưa tải ảnh ET</span>}
                </div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
};

export default HistoryPage;
