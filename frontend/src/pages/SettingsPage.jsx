import React from 'react';
import LanguageSelector from '../components/common/LanguageSelector';
import ModelSelector from '../components/common/ModelSelector';

const SettingsPage = () => {
  return (
    <div className="settings-page">
      <h1>Settings</h1>
      
      <div className="settings-section">
        <h2>Preferences</h2>
        
        <div className="setting-item">
          <h3>Default Language</h3>
          <LanguageSelector />
        </div>
        
        <div className="setting-item">
          <h3>AI Assistant Model</h3>
          <ModelSelector />
        </div>
        
        {/* Additional settings could be added here */}
        <div className="setting-item">
          <h3>Editor Theme</h3>
          <select>
            <option value="light">Light</option>
            <option value="dark">Dark</option>
            <option value="system">System Default</option>
          </select>
        </div>
      </div>
    </div>
  );
};

export default SettingsPage;