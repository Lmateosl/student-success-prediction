import HelpIcon from './HelpIcon';
import './PredictionResult.css';

function PredictionResult({ result }) {
  if (!result) {
    return (
      <div className="prediction-result empty">
        <div className="empty-state">
          <h3>No Prediction Yet</h3>
          <p>Fill in the form on the left and click "Predict Success" to see the result.</p>
        </div>
      </div>
    );
  }

  const prediction = result.predictions[0];
  // probabilities[0] is ALWAYS the probability of class 1 (persisting)
  const persistenceProbability = result.probabilities[0];
  const atRiskProbability = 1 - persistenceProbability;
  
  // Class 1 = likely to persist, Class 0 = at risk
  const isPersisting = prediction === 1;

  return (
    <div className={`prediction-result ${isPersisting ? 'success' : 'at-risk'}`}>
      <div className="result-header">
        <h2>
          Prediction Result
          <HelpIcon text="This prediction indicates whether a student is likely to complete their first year of study (persist) or may drop out or fail (at risk). The model uses a neural network trained on historical student data to make this assessment." />
        </h2>
        <div className={`result-icon ${isPersisting ? 'success' : 'warning'}`}>
          {isPersisting ? '✅' : '⚠️'}
        </div>
      </div>

      <div className="result-summary">
        <h3>
          {isPersisting
            ? 'This student is likely to persist in first year.'
            : 'This student is at risk of not persisting in first year.'}
          <HelpIcon text={`The prediction is based on a threshold: if persistence probability is 50% or higher, the student is predicted to persist. If it's below 50%, they're predicted to be at risk. Current persistence probability: ${(persistenceProbability * 100).toFixed(1)}%`} position="bottom" />
        </h3>
      </div>

      <div className="probabilities">
        <div className="probability-item">
          <div className="probability-label">
            <span className="label-text">
              Persistence Probability
              <HelpIcon text="The probability (0-100%) that this student will successfully complete their first year of study. Higher percentages indicate greater confidence that the student will persist. 'Persistence' means the student continues their studies and does not drop out or fail out." />
            </span>
            <span className="probability-value success">
              {(persistenceProbability * 100).toFixed(1)}%
            </span>
          </div>
          <div className="probability-bar">
            <div
              className="probability-fill success"
              style={{ width: `${persistenceProbability * 100}%` }}
            ></div>
          </div>
        </div>

        <div className="probability-item">
          <div className="probability-label">
            <span className="label-text">
              At-Risk Probability
              <HelpIcon text="The probability (0-100%) that this student is at risk of not completing their first year. This is the complement of persistence probability - they always add up to 100%. Higher at-risk probability means the student may need additional support or intervention." />
            </span>
            <span className="probability-value warning">
              {(atRiskProbability * 100).toFixed(1)}%
            </span>
          </div>
          <div className="probability-bar">
            <div
              className="probability-fill warning"
              style={{ width: `${atRiskProbability * 100}%` }}
            ></div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default PredictionResult;

