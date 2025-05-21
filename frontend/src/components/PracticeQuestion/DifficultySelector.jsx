import React from 'react';
import { useQuestion } from '../../context/QuestionContext';
import '../../styles/questions.css';

/**
 * DifficultySelector component
 * Allows users to select the difficulty level for practice questions
 */
const DifficultySelector = () => {
  const { skillLevel, setSkillLevel, difficultyLevels } = useQuestion();

  const handleDifficultyChange = (level) => {
    setSkillLevel(level);
  };

  // Helper function to get class name based on difficulty level
  const getDifficultyClass = (level) => {
    return `difficulty-${level.toLowerCase()}`;
  };

  return (
    <div className="difficulty-selector-container">
      <label htmlFor="difficulty-selector" className="form-label">
        Difficulty Level:
      </label>
      <div id="difficulty-selector" className="difficulty-selector">
        {difficultyLevels.map((level) => (
          <div
            key={level}
            className={`difficulty-option ${
              level === skillLevel ? 'selected' : ''
            } ${getDifficultyClass(level)}`}
            onClick={() => handleDifficultyChange(level)}
            role="button"
            tabIndex={0}
            aria-pressed={level === skillLevel}
          >
            <span className="difficulty-indicator"></span>
            <span className="difficulty-label">{level}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default DifficultySelector;