import { useState, useEffect } from 'react';
import { getRecentPredictions } from '../api/client';
import HelpIcon from './HelpIcon';
import './AdminPanel.css';

function AdminPanel() {
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isExpanded, setIsExpanded] = useState(false);

  const fetchPredictions = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const data = await getRecentPredictions(50);
      setPredictions(data.predictions);
    } catch (err) {
      setError('Failed to load predictions. ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    // Auto-fetch on mount if expanded
    if (isExpanded) {
      fetchPredictions();
    }
  }, [isExpanded]);

  const formatTimestamp = (timestamp) => {
    const date = new Date(timestamp);
    return date.toLocaleString();
  };

  const getPredictionLabel = (prediction) => {
    return prediction === 1 ? 'Likely to Persist' : 'At Risk';
  };

  const getPredictionClass = (prediction) => {
    return prediction === 1 ? 'success' : 'at-risk';
  };

  return (
    <div className="admin-panel">
      <div className="admin-header">
        <h2>
          📊 Admin Panel - Recent Predictions
          <HelpIcon text="This panel shows a log of all predictions made through the system. Each row contains the input data that was submitted, the prediction result (at risk or likely to persist), and the model's confidence level. This demonstrates the structured data access API capability." />
        </h2>
        <div className="admin-controls">
          <button
            className="refresh-button"
            onClick={fetchPredictions}
            disabled={loading}
          >
            {loading ? 'Loading...' : '🔄 Refresh'}
          </button>
          <button
            className="toggle-button"
            onClick={() => setIsExpanded(!isExpanded)}
          >
            {isExpanded ? '▼ Collapse' : '▶ Expand'}
          </button>
        </div>
      </div>

      {isExpanded && (
        <div className="admin-content">
          {error && (
            <div className="error-banner">
              {error}
            </div>
          )}

          {!error && predictions.length === 0 && !loading && (
            <div className="empty-message">
              No predictions recorded yet. Make your first prediction using the form above!
            </div>
          )}

          {predictions.length > 0 && (
            <div className="table-container">
              <table className="predictions-table">
                <thead>
                  <tr>
                    <th>Timestamp</th>
                    <th>First Term GPA</th>
                    <th>Second Term GPA</th>
                    <th>HS Average</th>
                    <th>Math Score</th>
                    <th>Residency</th>
                    <th>Funding</th>
                    <th>Prediction</th>
                    <th>Confidence</th>
                  </tr>
                </thead>
                <tbody>
                  {predictions.map((pred, index) => (
                    <tr key={index}>
                      <td className="timestamp">{formatTimestamp(pred.timestamp)}</td>
                      <td>{pred.first_term_gpa?.toFixed(2) || 'N/A'}</td>
                      <td>{pred.second_term_gpa?.toFixed(2) || 'N/A'}</td>
                      <td>{pred.high_school_average?.toFixed(1) || 'N/A'}</td>
                      <td>{pred.math_score?.toFixed(1) || 'N/A'}</td>
                      <td>{pred.residency === 1 ? 'Domestic' : pred.residency === 2 ? 'International' : 'N/A'}</td>
                      <td>{pred.funding || 'N/A'}</td>
                      <td>
                        <span className={`prediction-badge ${getPredictionClass(pred.prediction)}`}>
                          {getPredictionLabel(pred.prediction)}
                        </span>
                      </td>
                      <td className="confidence">{(pred.probability * 100).toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {predictions.length > 0 && (
            <div className="table-footer">
              <p>Showing {predictions.length} most recent prediction(s)</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default AdminPanel;

