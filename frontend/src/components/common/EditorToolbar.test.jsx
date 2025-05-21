import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import EditorToolbar from './EditorToolbar';
import { useLanguage } from '../../context/LanguageContext'; // Path to the hook

// Mock the useLanguage hook
vi.mock('../../context/LanguageContext', () => ({
  useLanguage: vi.fn(),
}));

const mockSetLanguage = vi.fn();
const mockOnRun = vi.fn();
const mockOnFormat = vi.fn();
const mockOnToggleTheme = vi.fn();
const mockOnToggleFullscreen = vi.fn();

const supportedLanguagesMock = {
  python: 'Python',
  javascript: 'JavaScript',
  java: 'Java',
};

describe('EditorToolbar', () => {
  beforeEach(() => {
    // Reset mocks before each test
    mockSetLanguage.mockClear();
    mockOnRun.mockClear();
    mockOnFormat.mockClear();
    mockOnToggleTheme.mockClear();
    mockOnToggleFullscreen.mockClear();
    
    // Default mock return value for useLanguage
    useLanguage.mockReturnValue({
      language: 'python',
      setLanguage: mockSetLanguage,
      supportedLanguages: supportedLanguagesMock,
    });
  });

  it('renders language dropdown correctly with initial language selected', () => {
    render(
      <EditorToolbar
        onRun={mockOnRun}
        onFormat={mockOnFormat}
        isDarkMode={false}
        onToggleTheme={mockOnToggleTheme}
        onToggleFullscreen={mockOnToggleFullscreen}
      />
    );

    // Check if the select element is there (getByRole 'combobox' is appropriate for select)
    const selectElement = screen.getByRole('combobox', { name: 'Select Programming Language' });
    expect(selectElement).toBeInTheDocument();

    // Check if the options are rendered correctly
    // and that their values are the language IDs
    const pythonOption = screen.getByRole('option', { name: 'Python' });
    expect(pythonOption).toBeInTheDocument();
    expect(pythonOption.value).toBe('python');

    const jsOption = screen.getByRole('option', { name: 'JavaScript' });
    expect(jsOption).toBeInTheDocument();
    expect(jsOption.value).toBe('javascript');
    
    const javaOption = screen.getByRole('option', { name: 'Java' });
    expect(javaOption).toBeInTheDocument();
    expect(javaOption.value).toBe('java');

    // Check if the correct language is selected by default
    expect(selectElement.value).toBe('python');
    
    // Check total number of options (including the selected one)
    const options = screen.getAllByRole('option');
    expect(options.length).toBe(3);
  });

  it('calls setLanguage when a new language is selected', () => {
    render(
      <EditorToolbar
        onRun={mockOnRun}
        onFormat={mockOnFormat}
        isDarkMode={false}
        onToggleTheme={mockOnToggleTheme}
        onToggleFullscreen={mockOnToggleFullscreen}
      />
    );

    const selectElement = screen.getByRole('combobox', { name: 'Select Programming Language' });
    
    // Simulate user changing the language to JavaScript
    fireEvent.change(selectElement, { target: { value: 'javascript' } });
    
    // Check if setLanguage was called with the new language ID
    expect(mockSetLanguage).toHaveBeenCalledTimes(1);
    expect(mockSetLanguage).toHaveBeenCalledWith('javascript');
  });

  it('updates the selected language in the dropdown when context value changes', () => {
    // Initial render with Python
    const { rerender } = render(
      <EditorToolbar
        onRun={mockOnRun}
        onFormat={mockOnFormat}
        isDarkMode={false}
        onToggleTheme={mockOnToggleTheme}
        onToggleFullscreen={mockOnToggleFullscreen}
      />
    );
    const selectElement = screen.getByRole('combobox', { name: 'Select Programming Language' });
    expect(selectElement.value).toBe('python');

    // Update the mock to return a new language
    useLanguage.mockReturnValue({
      language: 'java', // Current language is now Java
      setLanguage: mockSetLanguage,
      supportedLanguages: supportedLanguagesMock,
    });

    // Rerender the component with the new context value
    // EditorToolbar is memoized, so props need to change or the context it consumes needs to change.
    // Here, the hook `useLanguage` will return a new value, triggering a re-render.
    rerender(
      <EditorToolbar
        onRun={mockOnRun}
        onFormat={mockOnFormat}
        isDarkMode={false} // Keep other props the same
        onToggleTheme={mockOnToggleTheme}
        onToggleFullscreen={mockOnToggleFullscreen}
      />
    );
    
    // Check if the select element's value has updated to 'java'
    expect(selectElement.value).toBe('java');
  });
});
