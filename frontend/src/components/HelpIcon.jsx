import { useState } from 'react';
import './HelpIcon.css';

function HelpIcon({ text, position = 'top' }) {
  const [isVisible, setIsVisible] = useState(false);

  return (
    <span className="help-icon-container">
      <span
        className="help-icon"
        onMouseEnter={() => setIsVisible(true)}
        onMouseLeave={() => setIsVisible(false)}
        onFocus={() => setIsVisible(true)}
        onBlur={() => setIsVisible(false)}
        tabIndex={0}
        role="button"
        aria-label="Help"
      >
        ℹ️
      </span>
      {isVisible && (
        <div className={`help-tooltip help-tooltip-${position}`}>
          {text}
        </div>
      )}
    </span>
  );
}

export default HelpIcon;

