import React from 'react';
import QuestionGenerator from '../components/PracticeQuestion/QuestionGenerator';
import TopicSelector from '../components/PracticeQuestion/TopicSelector';

const QuestionsPage = () => {
  return (
    <div className="questions-page">
      <h1>Practice Questions</h1>
      <div className="questions-container">
        <TopicSelector />
        <QuestionGenerator />
      </div>
    </div>
  );
};

export default QuestionsPage;