import React, { createContext, useContext, useReducer } from 'react';

// Initial state based on the structure from react-refactoring-plan.md
const initialState = {
  language: {
    current: "python",
    available: ["python", "javascript", "typescript", "react"],
    preferences: {
      // Language-specific settings
      python: { tabSize: 4, insertSpaces: true },
      javascript: { tabSize: 2, insertSpaces: true },
      typescript: { tabSize: 2, insertSpaces: true },
      react: { tabSize: 2, insertSpaces: true }
    }
  },
  models: {
    availableProviders: [],
    availableModels: {},
    codeEvalProvider: null,
    codeEvalModel: null,
    questionGenProvider: null,
    questionGenModel: null
  },
  practiceQuestion: {
    current: null,
    topic: "",
    skillLevel: "Beginner",
    topicSuggestions: []
  },
  codeEvaluation: {
    code: "",
    result: null,
    loading: false
  },
  assistant: {
    messages: [],
    loading: false
  },
  ui: {
    activeView: "playground",
    sidebarCollapsed: false,
    assistantPanelWidth: 30, // percentage
    showResults: true
  }
};

// Action types
const ActionTypes = {
  SET_MODELS: 'SET_MODELS',
  SET_ACTIVE_MODELS: 'SET_ACTIVE_MODELS',
  SET_CODE: 'SET_CODE',
  SET_CODE_RESULT: 'SET_CODE_RESULT',
  SET_LOADING: 'SET_LOADING',
  SET_UI_STATE: 'SET_UI_STATE',
  UPDATE_ASSISTANT: 'UPDATE_ASSISTANT',
  RESET_STATE: 'RESET_STATE',
};

// Reducer function
function appReducer(state, action) {
  switch (action.type) {
    case ActionTypes.SET_MODELS:
      return {
        ...state,
        models: {
          ...state.models,
          availableProviders: action.payload.providers || state.models.availableProviders,
          availableModels: action.payload.models || state.models.availableModels,
        }
      };
    
    case ActionTypes.SET_ACTIVE_MODELS:
      return {
        ...state,
        models: {
          ...state.models,
          codeEvalProvider: action.payload.codeEvalProvider || state.models.codeEvalProvider,
          codeEvalModel: action.payload.codeEvalModel || state.models.codeEvalModel,
          questionGenProvider: action.payload.questionGenProvider || state.models.questionGenProvider,
          questionGenModel: action.payload.questionGenModel || state.models.questionGenModel,
        }
      };
    
    case ActionTypes.SET_CODE:
      return {
        ...state,
        codeEvaluation: {
          ...state.codeEvaluation,
          code: action.payload
        }
      };
    
    case ActionTypes.SET_CODE_RESULT:
      return {
        ...state,
        codeEvaluation: {
          ...state.codeEvaluation,
          result: action.payload,
          loading: false
        }
      };
    
    case ActionTypes.SET_LOADING:
      return {
        ...state,
        codeEvaluation: {
          ...state.codeEvaluation,
          loading: action.payload.codeEvaluation || state.codeEvaluation.loading
        },
        assistant: {
          ...state.assistant,
          loading: action.payload.assistant || state.assistant.loading
        }
      };
    
    case ActionTypes.SET_UI_STATE:
      return {
        ...state,
        ui: {
          ...state.ui,
          ...action.payload
        }
      };
    
    case ActionTypes.UPDATE_ASSISTANT:
      return {
        ...state,
        assistant: {
          ...state.assistant,
          messages: action.payload.messages || state.assistant.messages,
          loading: action.payload.loading !== undefined 
            ? action.payload.loading 
            : state.assistant.loading
        }
      };
    
    case ActionTypes.RESET_STATE:
      return {
        ...initialState,
        // Preserve models configuration and UI preferences
        models: state.models,
        ui: state.ui
      };
    
    default:
      console.warn(`Unknown action type: ${action.type}`);
      return state;
  }
}

// Create context
const AppContext = createContext();

// Provider component
export function AppProvider({ children }) {
  const [state, dispatch] = useReducer(appReducer, initialState);
  
  // Value object with state and helper functions
  const value = {
    state,
    dispatch,
    
    // Helper methods
    setModels: (providers, models) => {
      dispatch({
        type: ActionTypes.SET_MODELS,
        payload: { providers, models }
      });
    },
    
    setActiveModels: (codeEvalProvider, codeEvalModel, questionGenProvider, questionGenModel) => {
      dispatch({
        type: ActionTypes.SET_ACTIVE_MODELS,
        payload: { 
          codeEvalProvider, 
          codeEvalModel, 
          questionGenProvider, 
          questionGenModel 
        }
      });
    },
    
    setCode: (code) => {
      dispatch({
        type: ActionTypes.SET_CODE,
        payload: code
      });
    },
    
    setCodeResult: (result) => {
      dispatch({
        type: ActionTypes.SET_CODE_RESULT,
        payload: result
      });
    },
    
    setLoading: (isLoading, type = 'codeEvaluation') => {
      dispatch({
        type: ActionTypes.SET_LOADING,
        payload: { [type]: isLoading }
      });
    },
    
    updateAssistant: (messages, loading) => {
      dispatch({
        type: ActionTypes.UPDATE_ASSISTANT,
        payload: { messages, loading }
      });
    },
    
    setUIState: (uiState) => {
      dispatch({
        type: ActionTypes.SET_UI_STATE,
        payload: uiState
      });
    },
    
    resetState: () => {
      dispatch({ type: ActionTypes.RESET_STATE });
    }
  };
  
  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

// Custom hook for using the app context
export function useAppContext() {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useAppContext must be used within an AppProvider');
  }
  return context;
}

export default AppContext;