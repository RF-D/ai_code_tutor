import React, { createContext, useContext, useState, useCallback, useEffect } from 'react';
import { useAppContext } from './AppContext';
import { useLanguage } from './LanguageContext';

// Create context
const QuestionContext = createContext();

// Define difficulty levels
const difficultyLevels = ['Beginner', 'Intermediate', 'Advanced', 'Expert'];

// Provider component
export function QuestionProvider({ children }) {
  const { state, dispatch } = useAppContext();
  const { currentLanguageId } = useLanguage();

  // State for practice questions
  const [currentQuestion, setCurrentQuestion] = useState(null);
  const [questionsHistory, setQuestionsHistory] = useState([]);
  const [topic, setTopic] = useState('');
  const [skillLevel, setSkillLevel] = useState(difficultyLevels[0]); // Default: Beginner
  const [topicSuggestions, setTopicSuggestions] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // Track progress on current question
  const [questionProgress, setQuestionProgress] = useState({
    attempts: 0,
    completed: false,
    codeSubmissions: [],
    hintsUsed: 0,
    startTime: null,
    endTime: null
  });

  // Reset progress when a new question is loaded
  useEffect(() => {
    if (currentQuestion) {
      setQuestionProgress({
        attempts: 0,
        completed: false,
        codeSubmissions: [],
        hintsUsed: 0,
        startTime: new Date(),
        endTime: null
      });
    }
  }, [currentQuestion]);

  // Function to fetch a new question
  const fetchQuestion = useCallback(async (newTopic, newSkillLevel) => {
    try {
      setIsLoading(true);
      setError(null);
      
      // In a real implementation, this would call an API
      // For now, we'll simulate fetching a question
      
      // Update state with provided values if any
      const topicToUse = newTopic || topic;
      const levelToUse = newSkillLevel || skillLevel;
      
      // Store the current question in history if it exists
      if (currentQuestion) {
        setQuestionsHistory(prev => [...prev, {
          ...currentQuestion,
          progress: { ...questionProgress }
        }]);
      }
      
      // Simulate API call delay
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Create a mock question based on topic and skill level
      const mockQuestion = {
        id: Date.now().toString(),
        language: currentLanguageId,
        topic: topicToUse,
        skillLevel: levelToUse,
        title: `${levelToUse} ${topicToUse} Exercise in ${currentLanguageId}`,
        description: `This is a practice question about ${topicToUse} at the ${levelToUse} level.`,
        instructions: `Write a ${currentLanguageId} function that demonstrates ${topicToUse}.`,
        hints: [
          'Consider the basic syntax first',
          'Think about edge cases',
          'Optimize your solution if possible'
        ],
        sampleSolution: '// This would be the sample solution code',
        testCases: [
          { input: 'test input 1', expectedOutput: 'expected output 1' },
          { input: 'test input 2', expectedOutput: 'expected output 2' }
        ],
        createdAt: new Date().toISOString()
      };
      
      setCurrentQuestion(mockQuestion);
      
      // Update topic and skill level state if they were provided
      if (newTopic) setTopic(newTopic);
      if (newSkillLevel) setSkillLevel(newSkillLevel);
      
      return mockQuestion;
    } catch (err) {
      setError(err.message || 'Failed to fetch question');
      console.error('Error fetching question:', err);
      return null;
    } finally {
      setIsLoading(false);
    }
  }, [topic, skillLevel, currentLanguageId, questionProgress]);

  // Function to fetch topic suggestions based on language
  const fetchTopicSuggestions = useCallback(async (searchQuery = '') => {
    try {
      // In a real implementation, this would call an API
      // For now, we'll provide mock suggestions based on language
      
      const pythonTopics = [
        'Lists and Tuples', 'Dictionaries', 'Functions', 'Classes and OOP',
        'File Handling', 'Exception Handling', 'Decorators', 'Generators',
        'List Comprehensions', 'Lambda Functions', 'Modules and Packages'
      ];
      
      const javascriptTopics = [
        'Arrays and Objects', 'Functions', 'Closures', 'Promises',
        'Async/Await', 'DOM Manipulation', 'Event Handling', 'Callbacks',
        'Arrow Functions', 'ES6 Features', 'Modules'
      ];
      
      // Select topics based on current language
      let availableTopics = 
        currentLanguageId === 'python' ? pythonTopics : 
        currentLanguageId === 'javascript' ? javascriptTopics : 
        [];
      
      // Filter topics based on search query if provided
      if (searchQuery) {
        availableTopics = availableTopics.filter(topic => 
          topic.toLowerCase().includes(searchQuery.toLowerCase())
        );
      }
      
      setTopicSuggestions(availableTopics);
      return availableTopics;
    } catch (err) {
      console.error('Error fetching topic suggestions:', err);
      return [];
    }
  }, [currentLanguageId]);

  // Function to submit an answer for the current question
  const submitAnswer = useCallback((code) => {
    if (!currentQuestion) return null;
    
    // Update progress
    setQuestionProgress(prev => ({
      ...prev,
      attempts: prev.attempts + 1,
      codeSubmissions: [...prev.codeSubmissions, {
        code,
        timestamp: new Date().toISOString()
      }]
    }));
    
    // In a real implementation, this would call an API to evaluate the code
    // For now, we'll return a mock result
    return {
      correct: Math.random() > 0.5, // Randomly determine if correct
      feedback: 'This is feedback on your code submission.',
      testResults: currentQuestion.testCases.map(testCase => ({
        input: testCase.input,
        expectedOutput: testCase.expectedOutput,
        actualOutput: 'mock output',
        passed: Math.random() > 0.3 // Randomly determine if test passed
      }))
    };
  }, [currentQuestion]);

  // Function to mark the current question as completed
  const completeQuestion = useCallback(() => {
    setQuestionProgress(prev => ({
      ...prev,
      completed: true,
      endTime: new Date()
    }));
  }, []);

  // Function to request a hint
  const requestHint = useCallback(() => {
    if (!currentQuestion || !currentQuestion.hints) return null;
    
    // Update progress to track hint usage
    setQuestionProgress(prev => ({
      ...prev,
      hintsUsed: Math.min(prev.hintsUsed + 1, currentQuestion.hints.length)
    }));
    
    // Return the appropriate hint based on how many have been used
    const hintIndex = Math.min(questionProgress.hintsUsed, currentQuestion.hints.length - 1);
    return currentQuestion.hints[hintIndex];
  }, [currentQuestion, questionProgress.hintsUsed]);

  // Context value
  const value = {
    // Current question and history
    currentQuestion,
    questionsHistory,
    
    // Question parameters
    topic,
    setTopic,
    skillLevel,
    setSkillLevel,
    difficultyLevels,
    topicSuggestions,
    
    // Question progress
    questionProgress,
    
    // Status
    isLoading,
    error,
    
    // Functions
    fetchQuestion,
    fetchTopicSuggestions,
    submitAnswer,
    completeQuestion,
    requestHint,
    
    // Helper functions
    getCurrentHints: () => {
      if (!currentQuestion || !currentQuestion.hints) return [];
      return currentQuestion.hints.slice(0, questionProgress.hintsUsed);
    },
    getAvailableHintsCount: () => {
      if (!currentQuestion || !currentQuestion.hints) return 0;
      return currentQuestion.hints.length - questionProgress.hintsUsed;
    },
    getRemainingHints: () => {
      if (!currentQuestion || !currentQuestion.hints) return [];
      return currentQuestion.hints.slice(questionProgress.hintsUsed);
    }
  };

  return <QuestionContext.Provider value={value}>{children}</QuestionContext.Provider>;
}

// Custom hook for using the question context
export function useQuestion() {
  const context = useContext(QuestionContext);
  if (!context) {
    throw new Error('useQuestion must be used within a QuestionProvider');
  }
  return context;
}

export default QuestionContext;