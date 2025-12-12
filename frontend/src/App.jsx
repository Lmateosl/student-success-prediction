import { useState, useEffect } from 'react';
import StudentForm from './components/StudentForm';
import PredictionResult from './components/PredictionResult';
import AdminPanel from './components/AdminPanel';
import { predictStudent, healthCheck } from './api/client';
import './App.css';

function App() {
  const [predictionResult, setPredictionResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [apiStatus, setApiStatus] = useState('checking');

  useEffect(() => {
    // Check API health on mount
    async function checkApi() {
      try {
        await healthCheck();
        setApiStatus('connected');
      } catch (err) {
        setApiStatus('error');
        console.error('API health check failed:', err);
      }
    }
    
    checkApi();
  }, []);

  const handlePrediction = async (studentData) => {
    setLoading(true);
    setError(null);
    
    try {
      const result = await predictStudent(studentData);
      setPredictionResult(result);
    } catch (err) {
      setError(err.message);
      console.error('Prediction failed:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-content">
          <h1>🎓 Student Success Prediction</h1>
          <p className="subtitle">
            First-year persistence prediction using a neural network model
          </p>
          <div className={`api-status ${apiStatus}`}>
            {apiStatus === 'checking' && '⏳ Connecting to API...'}
            {apiStatus === 'connected' && '✅ API Connected'}
            {apiStatus === 'error' && '❌ API Connection Failed - Make sure the backend is running on port 8000'}
          </div>
        </div>
      </header>

      <main className="app-main">
        <div className="main-grid">
          <div className="form-section">
            <StudentForm onSubmit={handlePrediction} loading={loading} />
          </div>

          <div className="result-section">
            {error && (
              <div className="error-banner">
                <strong>Error:</strong> {error}
              </div>
            )}
            <PredictionResult result={predictionResult} />
          </div>
        </div>

        <div className="admin-section">
          <AdminPanel />
        </div>
      </main>

      <footer className="app-footer">
        <p>
          COMP258 Group Project
        </p>
      </footer>
    </div>
  );
}

export default App;
