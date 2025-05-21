const fs = require('fs');
const path = require('path');

function fixDuplicateImports() {
  const srcDir = path.join(__dirname, 'src');
  
  // Get all JSX files recursively
  const jsxFiles = getAllFiles(srcDir, '.jsx');
  
  jsxFiles.forEach(file => {
    console.log(`Processing: ${file}`);
    let content = fs.readFileSync(file, 'utf8');
    
    // Remove duplicate import lines
    const lines = content.split('\n');
    const uniqueLines = [];
    const reactImportPattern = /import\s+React\b/;
    let foundReactImport = false;
    
    lines.forEach(line => {
      if (reactImportPattern.test(line)) {
        if (!foundReactImport) {
          uniqueLines.push(line);
          foundReactImport = true;
        }
      } else {
        uniqueLines.push(line);
      }
    });
    
    const newContent = uniqueLines.join('\n');
    if (content !== newContent) {
      fs.writeFileSync(file, newContent);
      console.log(`Fixed: ${file}`);
    }
  });
}

function getAllFiles(dir, ext) {
  let files = [];
  const dirContents = fs.readdirSync(dir);
  
  dirContents.forEach(item => {
    const fullPath = path.join(dir, item);
    if (fs.statSync(fullPath).isDirectory()) {
      // Recursively get files from subdirectories
      files = files.concat(getAllFiles(fullPath, ext));
    } else if (item.endsWith(ext)) {
      files.push(fullPath);
    }
  });
  
  return files;
}

// Create vite.config.js
function createViteConfig() {
  const viteConfigPath = path.join(__dirname, 'vite.config.js');
  
  if (!fs.existsSync(viteConfigPath)) {
    const viteConfig = `import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [
    react({
      jsxRuntime: 'automatic'
    })
  ],
  resolve: {
    extensions: ['.js', '.jsx', '.json']
  },
  esbuild: {
    loader: 'jsx',
    include: /\\.jsx?$/,
    exclude: []
  }
});
`;
    fs.writeFileSync(viteConfigPath, viteConfig);
    console.log('Created vite.config.js');
  } else {
    console.log('vite.config.js already exists');
  }
}

// Execute the functions
createViteConfig();
fixDuplicateImports();
console.log('React import fix completed');