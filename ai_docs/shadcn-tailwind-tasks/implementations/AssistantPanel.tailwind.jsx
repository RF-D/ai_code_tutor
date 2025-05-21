import React, { useState } from 'react';
import { FaPaperPlane } from 'react-icons/fa';

/**
 * AssistantPanel provides an interface for AI assistance with programming tasks
 * This version uses Tailwind CSS for styling.
 */
function AssistantPanel() {
  const [messages, setMessages] = useState([
    { 
      role: 'assistant', 
      content: 'Hello! I\'m your coding assistant. If you need help with your code or have questions about the problem, feel free to ask me.'
    }
  ]);
  
  const [userInput, setUserInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  
  const handleSendMessage = async () => {
    if (!userInput.trim() || isLoading) return;
    
    // Add user message to chat
    const userMessage = { role: 'user', content: userInput };
    setMessages(prev => [...prev, userMessage]);
    setUserInput('');
    setIsLoading(true);
    
    // Simulate API delay
    setTimeout(() => {
      // Simulate assistant response
      const assistantMessage = { 
        role: 'assistant', 
        content: `I'll help you with that. ${userInput.includes('?') ? 'Here\'s what you need to know:' : 'Here\'s what you can do:'} \n\nThis would be where the AI provides specific assistance based on your question. For now, this is just simulated.`
      };
      setMessages(prev => [...prev, assistantMessage]);
      setIsLoading(false);
    }, 1000);
  };
  
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };
  
  return (
    <div className="flex flex-col h-full rounded-md bg-white shadow-sm overflow-hidden dark:bg-slate-800">
      <div className="h-10 bg-slate-100 border-b border-slate-200 flex items-center px-4 font-semibold dark:bg-slate-800 dark:border-slate-700">
        AI Assistant
      </div>
      <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-4">
        {messages.map((message, index) => (
          <div 
            key={index}
            className={`max-w-[85%] p-2.5 px-4 rounded-2xl ${
              message.role === 'user' 
                ? 'bg-blue-600 text-white self-end rounded-br-none'
                : 'bg-slate-100 text-slate-800 self-start rounded-bl-none dark:bg-slate-700 dark:text-slate-100'
            }`}
          >
            {message.content}
          </div>
        ))}
        {isLoading && (
          <div className="self-start bg-slate-100 text-slate-800 p-4 rounded-2xl rounded-bl-none flex items-center dark:bg-slate-700 dark:text-slate-100">
            <div className="w-2 h-2 rounded-full bg-slate-400 mr-1 animate-pulse"></div>
            <div className="w-2 h-2 rounded-full bg-slate-400 mx-1 animate-pulse delay-75"></div>
            <div className="w-2 h-2 rounded-full bg-slate-400 ml-1 animate-pulse delay-150"></div>
          </div>
        )}
      </div>
      <div className="p-2 border-t border-slate-200 flex items-center dark:border-slate-700">
        <textarea
          value={userInput}
          onChange={(e) => setUserInput(e.target.value)}
          onKeyDown={handleKeyPress}
          placeholder="Ask for help or clarification..."
          className="flex-1 border border-slate-300 rounded-full px-4 py-2 text-sm focus:outline-none focus:border-blue-500 resize-none h-9 overflow-hidden dark:bg-slate-700 dark:border-slate-600 dark:text-slate-100"
        />
        <button 
          onClick={handleSendMessage}
          disabled={!userInput.trim() || isLoading}
          className={`ml-2 w-9 h-9 rounded-full flex items-center justify-center ${
            !userInput.trim() || isLoading
              ? 'bg-slate-300 cursor-not-allowed dark:bg-slate-600'
              : 'bg-blue-600 text-white hover:bg-blue-700 dark:bg-blue-500 dark:hover:bg-blue-600'
          }`}
        >
          <FaPaperPlane className="text-sm" />
        </button>
      </div>
    </div>
  );
}

export default AssistantPanel;