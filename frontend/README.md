# Frontend Setup

This directory contains the React frontend for the AI Code Tutor project.

## Getting Started

1. Install dependencies:
   ```bash
   npm install
   ```

2. Fix React import issues (if this is your first time setting up):
   ```bash
   npm run fix-imports
   ```
   
3. Run the development server:
   ```bash
   npm run dev
   ```

The app will be available at `http://localhost:5173` by default when using Vite.

## React Import Fix

If you encounter duplicate React import errors, the project includes a script to fix this issue. 
The error occurs due to both explicit imports in components and automatic injection via Vite configuration.

```bash
npm run fix-imports
```

For more details about this fix, see [REACT-IMPORTS-FIX.md](REACT-IMPORTS-FIX.md).

## File Extensions

For consistency across the project:
- Use `.jsx` for React component files
- Use `.js` for utility files without JSX
- If you add TypeScript, use `.tsx` for TS components and `.ts` for TS utilities

You can check for file extension consistency with:
```bash
npm run check-extensions
```

This will scan the codebase and identify `.js` files that appear to contain JSX code, which should be renamed to `.jsx` for consistency.
