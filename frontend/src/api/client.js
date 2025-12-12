// API client for Student Success Prediction backend

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

/**
 * Fetch feature columns metadata from the backend
 * @returns {Promise<Object>} Feature columns information
 */
export async function getFeatureColumns() {
  const response = await fetch(`${API_BASE_URL}/feature_columns`);
  if (!response.ok) {
    throw new Error('Failed to fetch feature columns');
  }
  return response.json();
}

/**
 * Make a prediction for a student
 * @param {Object} studentData - Student features
 * @returns {Promise<Object>} Prediction result with probabilities and predictions
 */
export async function predictStudent(studentData) {
  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      instances: [studentData],
    }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Prediction failed' }));
    throw new Error(error.detail || 'Failed to make prediction');
  }

  return response.json();
}

/**
 * Get recent predictions from the log
 * @param {number} limit - Maximum number of predictions to fetch
 * @returns {Promise<Object>} List of recent predictions
 */
export async function getRecentPredictions(limit = 50) {
  const response = await fetch(`${API_BASE_URL}/predictions?limit=${limit}`);
  if (!response.ok) {
    throw new Error('Failed to fetch recent predictions');
  }
  return response.json();
}

/**
 * Health check endpoint
 * @returns {Promise<Object>} API health status
 */
export async function healthCheck() {
  const response = await fetch(`${API_BASE_URL}/`);
  if (!response.ok) {
    throw new Error('API health check failed');
  }
  return response.json();
}

