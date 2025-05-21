import React, { createContext, useContext, useState, useCallback } from 'react';
import { useAppContext } from './AppContext';

// Create the language context
const LanguageContext = createContext();

// Available languages (initially Python and JavaScript)
const availableLanguages = [
  {
    id: 'python',
    name: 'Python',
    extension: '.py',
    version: '3.x',
    settings: {
      tabSize: 4,
      insertSpaces: true,
      defaultIndent: '    '
    }
  },
  {
    id: 'javascript',
    name: 'JavaScript',
    extension: '.js',
    version: 'ES6+',
    settings: {
      tabSize: 2,
      insertSpaces: true,
      defaultIndent: '  '
    }
  }
];

// Provider component
export function LanguageProvider({ children }) {
  const { state, setUIState } = useAppContext();
  
  // Initial language is Python
  const [currentLanguage, setCurrentLanguage] = useState(availableLanguages[0]);
  const [languagePreferences, setLanguagePreferences] = useState(() => {
    // Create an object with preferences for each language
    const preferences = {};
    availableLanguages.forEach(lang => {
      preferences[lang.id] = { ...lang.settings };
    });
    return preferences;
  });

  // Function to change the current language
  const changeLanguage = useCallback((languageId) => {
    const language = availableLanguages.find(lang => lang.id === languageId);
    if (language) {
      setCurrentLanguage(language);
      
      // Update UI state if needed (e.g., adjust editor settings)
      setUIState({
        editorSettings: {
          tabSize: language.settings.tabSize,
          insertSpaces: language.settings.insertSpaces
        }
      });
    } else {
      console.error(`Language ${languageId} not found`);
    }
  }, [setUIState]);

  // Function to update language preferences
  const updateLanguagePreference = useCallback((languageId, settingKey, value) => {
    if (!languagePreferences[languageId]) {
      console.error(`Language ${languageId} not found in preferences`);
      return;
    }

    setLanguagePreferences(prev => ({
      ...prev,
      [languageId]: {
        ...prev[languageId],
        [settingKey]: value
      }
    }));

    // If this is the current language, also update the current settings
    if (languageId === currentLanguage.id) {
      setUIState({
        editorSettings: {
          [settingKey]: value
        }
      });
    }
  }, [languagePreferences, currentLanguage, setUIState]);

  // Get the current language settings
  const getCurrentLanguageSettings = useCallback(() => {
    return languagePreferences[currentLanguage.id] || currentLanguage.settings;
  }, [currentLanguage, languagePreferences]);

  // Context value
  const value = {
    // Current language
    currentLanguage,
    
    // All available languages
    availableLanguages,
    
    // Language preferences for all languages
    languagePreferences,
    
    // Functions
    changeLanguage,
    updateLanguagePreference,
    getCurrentLanguageSettings,
    
    // Helper properties
    currentLanguageId: currentLanguage.id,
    currentLanguageName: currentLanguage.name,
    
    // File extension for the current language (useful for Monaco Editor)
    currentExtension: currentLanguage.extension,
    
    // Tab size for the current language
    tabSize: getCurrentLanguageSettings().tabSize,
    
    // Whether to use spaces for indentation
    insertSpaces: getCurrentLanguageSettings().insertSpaces,
    
    // Default indentation for the current language
    defaultIndent: getCurrentLanguageSettings().defaultIndent
  };

  return <LanguageContext.Provider value={value}>{children}</LanguageContext.Provider>;
}

// Custom hook for using the language context
export function useLanguage() {
  const context = useContext(LanguageContext);
  if (!context) {
    throw new Error('useLanguage must be used within a LanguageProvider');
  }
  return context;
}

export default LanguageContext;