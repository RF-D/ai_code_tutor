import React, { useState, useEffect, useRef } from 'react';
import { useQuestion } from '../../context/QuestionContext';
import { useLanguage } from '../../context/LanguageContext';
import '../../styles/questions.css';

/**
 * TopicSelector component
 * Provides an input for selecting topics with suggestions based on the current language
 */
const TopicSelector = () => {
  const { topic, setTopic, topicSuggestions, fetchTopicSuggestions } = useQuestion();
  const { currentLanguageId, currentLanguageName } = useLanguage();
  const [inputValue, setInputValue] = useState(topic);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [isInputFocused, setIsInputFocused] = useState(false);
  const suggestionsRef = useRef(null);
  const inputRef = useRef(null);

  // Fetch suggestions when language changes or input changes
  useEffect(() => {
    if (isInputFocused) {
      fetchTopicSuggestions(inputValue);
    }
  }, [inputValue, currentLanguageId, fetchTopicSuggestions, isInputFocused]);

  // Handle input change
  const handleInputChange = (e) => {
    const value = e.target.value;
    setInputValue(value);
    setShowSuggestions(true);
  };

  // Handle selection of a suggestion
  const handleSelectSuggestion = (suggestion) => {
    setInputValue(suggestion);
    setTopic(suggestion);
    setShowSuggestions(false);
    inputRef.current?.blur();
  };

  // Handle input blur
  const handleInputBlur = (e) => {
    // Prevent closing suggestions if clicking on a suggestion
    if (suggestionsRef.current && suggestionsRef.current.contains(e.relatedTarget)) {
      return;
    }
    
    // Delay hiding suggestions to allow clicking on them
    setTimeout(() => {
      setShowSuggestions(false);
      setIsInputFocused(false);
      
      // Update the global topic state with current input value
      if (inputValue.trim() !== topic) {
        setTopic(inputValue.trim());
      }
    }, 200);
  };

  // Handle input focus
  const handleInputFocus = () => {
    setIsInputFocused(true);
    setShowSuggestions(true);
    fetchTopicSuggestions(inputValue);
  };

  // Handle form submission
  const handleSubmit = (e) => {
    e.preventDefault();
    setTopic(inputValue.trim());
  };

  return (
    <div className="topic-selector">
      <label htmlFor="topic-input" className="form-label">
        Topic ({currentLanguageName}):
      </label>
      <form onSubmit={handleSubmit}>
        <input
          ref={inputRef}
          id="topic-input"
          type="text"
          className="topic-input"
          value={inputValue}
          onChange={handleInputChange}
          onFocus={handleInputFocus}
          onBlur={handleInputBlur}
          placeholder={`Enter a ${currentLanguageName} topic...`}
          aria-label={`Enter a ${currentLanguageName} topic`}
        />
      </form>

      {showSuggestions && topicSuggestions.length > 0 && (
        <div 
          ref={suggestionsRef} 
          className="topic-suggestions"
          role="listbox"
          aria-label="Topic suggestions"
        >
          {topicSuggestions.map((suggestion) => (
            <div
              key={suggestion}
              className="topic-suggestion-item"
              onClick={() => handleSelectSuggestion(suggestion)}
              role="option"
              tabIndex={0}
              aria-selected={suggestion === inputValue}
            >
              {suggestion}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default TopicSelector;