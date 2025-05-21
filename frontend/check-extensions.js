#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const { globSync } = require('glob');

console.log('Checking file extension consistency...');

// Find all JavaScript files
const jsFiles = globSync('src/**/*.js', { cwd: __dirname });

// Potential issues
const jsxInJsFiles = [];

// Check .js files that might contain JSX
jsFiles.forEach(filePath => {
  const fullPath = path.join(__dirname, filePath);
  const content = fs.readFileSync(fullPath, 'utf8');
  
  // Simple check for JSX syntax in .js files (not perfect but gives an indication)
  if (content.includes('<') && content.includes('/>') || 
      content.includes('</') && content.includes('>') ||
      content.includes('React.') && (content.includes('<') || content.includes('>'))) {
    jsxInJsFiles.push(filePath);
  }
});

// Results
if (jsxInJsFiles.length > 0) {
  console.log('\n⚠️  The following .js files may contain JSX and should have a .jsx extension:');
  jsxInJsFiles.forEach(file => console.log(`  - ${file}`));
  console.log('\nConsider renaming these files to .jsx for consistency.');
} else {
  console.log('\n✅ No .js files with JSX content detected. File extensions appear consistent.');
}

console.log('\nNote: This is a basic check and might not catch all cases.');
console.log('For more reliable detection, consider setting up an ESLint rule.');