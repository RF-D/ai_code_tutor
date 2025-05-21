const fs = require('fs');
const path = require('path');

function findAllJsxFiles(dir) {
  let results = [];
  const items = fs.readdirSync(dir);
  
  for (const item of items) {
    const itemPath = path.join(dir, item);
    const stat = fs.statSync(itemPath);
    
    if (stat.isDirectory()) {
      results = results.concat(findAllJsxFiles(itemPath));
    } else if (item.endsWith('.jsx')) {
      results.push(itemPath);
    }
  }
  
  return results;
}

function fixFile(filePath) {
  console.log(`Processing ${filePath}`);
  
  try {
    let content = fs.readFileSync(filePath, 'utf8');
    
    // Find and replace duplicate React imports
    const fixedContent = content
      // First remove all double React imports entirely
      .replace(/^import React.*from ['"]react['"];.*?import React.*from ['"]react['"];/ms, (match) => {
        // Extract all hooks from the import statements
        const hooks = [];
        const regex = /import React, {([^}]*)} from ['"]react['"];/g;
        let m;
        while ((m = regex.exec(match)) !== null) {
          if (m[1]) {
            hooks.push(...m[1].split(',').map(h => h.trim()).filter(h => h));
          }
        }
        
        // Create a new import with all hooks
        if (hooks.length > 0) {
          return `import React, { ${[...new Set(hooks)].join(', ')} } from 'react';`;
        } else {
          return `import React from 'react';`;
        }
      })
      
      // Then handle any remaining React imports
      .replace(/^import React from ['"]react['"];.*?import React, {([^}]*)} from ['"]react['"];/ms, 
        (match, hooks) => `import React, { ${hooks} } from 'react';`)
      .replace(/^import React, {([^}]*)} from ['"]react['"];.*?import React from ['"]react['"];/ms, 
        (match, hooks) => `import React, { ${hooks} } from 'react';`);
    
    if (content !== fixedContent) {
      fs.writeFileSync(filePath, fixedContent, 'utf8');
      console.log(`Fixed ${filePath}`);
    }
  } catch (error) {
    console.error(`Error processing ${filePath}:`, error);
  }
}

const srcDir = path.join(__dirname, 'src');
const jsxFiles = findAllJsxFiles(srcDir);

jsxFiles.forEach(fixFile);
console.log('Done fixing React imports manually.');

// Now create a proper Vite config
const viteConfigContent = `
import { defineConfig } from 'vite';
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
    include: /\\.[jt]sx?$/,
    exclude: []
  }
});
`;

fs.writeFileSync(path.join(__dirname, 'vite.config.js'), viteConfigContent, 'utf8');
console.log('Updated vite.config.js');