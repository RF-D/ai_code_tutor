import React, { useState, useEffect, useRef } from 'react';
import { FaPaperPlane, FaRobot } from 'react-icons/fa';
import ReactMarkdown from 'react-markdown';
import rehypeRaw from 'rehype-raw';
import remarkGfm from 'remark-gfm';

/**
 * AssistantPanel component provides a chat interface for asking questions
 * and receiving help from the AI assistant.
 */
function AssistantPanel() {
  const [messages, setMessages] = useState(() => {
    const savedMessages = localStorage.getItem('assistant_messages');
    return savedMessages ? JSON.parse(savedMessages) : [
      {
        id: '1',
        type: 'assistant',
        content: "👋 Hi there! I'm your coding assistant. How can I help you today? You can ask me about the current exercise, request hints, or get explanations on specific coding concepts."
      }
    ];
  });
  
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const chatContainerRef = useRef(null);
  const inputRef = useRef(null);

  // Save messages to localStorage when they change
  useEffect(() => {
    localStorage.setItem('assistant_messages', JSON.stringify(messages));
  }, [messages]);

  // Scroll to bottom of chat container when messages change
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages]);

  // Handle sending a new message
  const handleSendMessage = async (e) => {
    e?.preventDefault();
    
    if (!inputValue.trim() || isLoading) return;
    
    const userMessage = {
      id: Date.now().toString(),
      type: 'user',
      content: inputValue
    };
    
    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);
    
    try {
      // This would be replaced with an actual API call
      const response = await simulateAssistantResponse(inputValue);
      
      setMessages(prev => [...prev, {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: response.message
      }]);
    } catch (error) {
      setMessages(prev => [...prev, {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: "I'm sorry, I encountered an error while processing your request. Please try again."
      }]);
    } finally {
      setIsLoading(false);
      // Focus input after response
      setTimeout(() => {
        inputRef.current?.focus();
      }, 100);
    }
  };
  
  // Simulate API call (placeholder for actual implementation)
  const simulateAssistantResponse = (userMessage) => {
    return new Promise((resolve) => {
      setTimeout(() => {
        let response;
        
        if (userMessage.toLowerCase().includes('hint')) {
          response = {
            message: "Here's a hint: Think about breaking down the problem into smaller steps. Consider using a loop to iterate through the elements, and remember to handle edge cases."
          };
        } else if (userMessage.toLowerCase().includes('error')) {
          response = {
            message: "Let's debug that error. Check your syntax, especially around brackets and semicolons. Also, make sure all variables are properly declared before use."
          };
        } else {
          response = {
            message: "I understand you're asking about `" + userMessage + "`. Here are some thoughts:\n\n" +
            "```javascript\n// Example code related to your question\nfunction example() {\n  console.log('This demonstrates the concept');\n}\n```\n\n" +
            "Would you like me to elaborate further on any specific part?"
          };
        }
        
        resolve(response);
      }, 1500);
    });
  };

  return (
    <div className="panel assistant-panel">
      <div className="panel-header">
        <div><FaRobot /> AI Coding Assistant</div>
      </div>
      <div 
        className="chat-container"
        ref={chatContainerRef}
      >
        {messages.map(message => (
          <div 
            key={message.id} 
            className={`chat-message chat-message-${message.type}`}
          >
            <ReactMarkdown 
              rehypePlugins={[rehypeRaw]} 
              remarkPlugins={[remarkGfm]}
              components={{
                code({node, inline, className, children, ...props}) {
                  return inline ? (
                    <code className={className} {...props}>
                      {children}
                    </code>
                  ) : (
                    <div className="code-block-wrapper">
                      <pre className={className} {...props}>
                        <code>{children}</code>
                      </pre>
                    </div>
                  );
                }
              }}
            >
              {message.content}
            </ReactMarkdown>
          </div>
        ))}
        
        {isLoading && (
          <div className="chat-message chat-message-assistant loading-message">
            <div className="typing-indicator">
              <span></span>
              <span></span>
              <span></span>
            </div>
          </div>
        )}
      </div>
      
      <form 
        className="chat-input-container"
        onSubmit={handleSendMessage}
      >
        <input
          type="text"
          className="chat-input"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          placeholder="Ask for help or hints..."
          disabled={isLoading}
          ref={inputRef}
        />
        <button 
          type="submit" 
          className="send-button"
          disabled={isLoading || !inputValue.trim()}
        >
          <FaPaperPlane />
        </button>
      </form>
    </div>
  );
}

export default AssistantPanel;
