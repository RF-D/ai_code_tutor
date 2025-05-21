# Tailwind CSS and Shadcn/UI Setup

This document outlines the setup and configuration of Tailwind CSS and Shadcn/UI in the AI Code Tutor application.

## Tailwind CSS Setup

### Installation and Configuration

The project uses Tailwind CSS for utility-first styling, with a customized configuration to match the application's design system.

1. **Installation**:
   ```bash
   npm install -D tailwindcss postcss autoprefixer
   npx tailwindcss init -p
   ```

2. **Configuration** (`tailwind.config.js`):
   ```js
   module.exports = {
     content: [
       "./index.html",
       "./src/**/*.{js,jsx,ts,tsx}",
     ],
     darkMode: 'class', // Enables dark mode with class strategy
     theme: {
       extend: {
         colors: {
           // Extended color palette
           primary: {
             DEFAULT: 'var(--color-primary)',
             light: 'var(--color-primary-light)',
             dark: 'var(--color-primary-dark)',
           },
           secondary: {
             DEFAULT: 'var(--color-secondary)',
           },
           background: {
             light: 'var(--color-background-light)',
             dark: 'var(--color-background-dark)',
           },
           text: {
             primary: 'var(--color-text-primary)',
             secondary: 'var(--color-text-secondary)',
           }
         },
         fontFamily: {
           sans: ['Inter', 'system-ui', 'sans-serif'],
           mono: ['JetBrains Mono', 'Menlo', 'monospace'],
         },
         fontSize: {
           // Custom font sizes
           'code-xs': '0.75rem',
           'code-sm': '0.875rem',
           'code-base': '1rem',
         },
         spacing: {
           // Custom spacing values
           '0.5': '0.125rem',
           '1.5': '0.375rem',
         },
         borderRadius: {
           'sm': '0.25rem',
           'md': '0.375rem',
           'lg': '0.5rem',
         },
       },
     },
     plugins: [
       require('@tailwindcss/typography'),
       require('@tailwindcss/forms'),
     ],
   }
   ```

3. **CSS Variables** (`src/styles/theme.css`):
   ```css
   :root {
     /* Light mode variables */
     --color-primary: #4f46e5;
     --color-primary-light: #818cf8;
     --color-primary-dark: #4338ca;
     --color-secondary: #0ea5e9;
     --color-background-light: #ffffff;
     --color-background-dark: #f3f4f6;
     --color-text-primary: #111827;
     --color-text-secondary: #4b5563;
   }

   .dark {
     /* Dark mode variables */
     --color-primary: #818cf8;
     --color-primary-light: #a5b4fc;
     --color-primary-dark: #6366f1;
     --color-secondary: #38bdf8;
     --color-background-light: #1f2937;
     --color-background-dark: #111827;
     --color-text-primary: #f9fafb;
     --color-text-secondary: #d1d5db;
   }
   ```

4. **Project CSS Setup** (`src/index.css`):
   ```css
   @import 'tailwindcss/base';
   @import './styles/theme.css';
   @import 'tailwindcss/components';
   @import 'tailwindcss/utilities';
   
   /* Custom component styles */
   @layer components {
     .btn {
       @apply px-4 py-2 rounded-md font-medium transition-colors;
     }
     .btn-primary {
       @apply bg-primary text-white hover:bg-primary-dark;
     }
     /* ... more component styles */
   }
   ```

## Shadcn/UI Integration

Shadcn/UI provides accessible, customizable components built on Radix UI and styled with Tailwind CSS.

### Installation and Setup

1. **Initial Setup**:
   ```bash
   npx shadcn-ui@latest init
   ```

2. **Configuration** (`components.json`):
   ```json
   {
     "$schema": "https://ui.shadcn.com/schema.json",
     "style": "default",
     "rsc": false,
     "tsx": false,
     "tailwind": {
       "config": "tailwind.config.js",
       "css": "src/index.css",
       "baseColor": "slate",
       "cssVariables": true
     },
     "aliases": {
       "components": "@/components",
       "utils": "@/lib/utils"
     }
   }
   ```

3. **Utility Function** (`src/lib/utils.js`):
   ```js
   import { clsx } from 'clsx';
   import { twMerge } from 'tailwind-merge';

   export function cn(...inputs) {
     return twMerge(clsx(inputs));
   }
   ```

### Core Components Integration

The following Shadcn/UI components have been integrated:

1. **Button Component**:
   ```bash
   npx shadcn-ui@latest add button
   ```
   - Customized with application-specific styles and variants
   - Enhanced with loading states and icon support

2. **Dialog/Modal Component**:
   ```bash
   npx shadcn-ui@latest add dialog
   ```
   - Used for confirmations, settings panels, and information displays
   - Responsive behavior for different screen sizes

3. **Form Components**:
   ```bash
   npx shadcn-ui@latest add form input checkbox select textarea
   ```
   - Integrated with React Hook Form for validation
   - Styled consistently with application theme

4. **Navigation Components**:
   ```bash
   npx shadcn-ui@latest add navigation-menu dropdown-menu sheet
   ```
   - Used for main navigation and mobile responsive menu
   - Enhanced with custom animations and transitions

5. **Accordion Component**:
   ```bash
   npx shadcn-ui@latest add accordion
   ```
   - Used for FAQ sections and collapsible content
   - Custom styling to match application design

## Custom Styling Extensions

1. **Theme Extension**:
   - Extended Shadcn component styles to support both light and dark modes
   - Added custom animation transitions for improved UX

2. **Component Customization**:
   - Modified default component styles to match application design system
   - Created custom variants for specific use cases

3. **Responsive Adaptations**:
   - Enhanced components with mobile-first responsive design
   - Added touch-friendly adaptations for mobile users

## Usage Examples

```jsx
// Button component example
import { Button } from "@/components/ui/button";

export function SaveButton({ isLoading, onClick }) {
  return (
    <Button 
      variant="primary" 
      size="sm" 
      onClick={onClick}
      disabled={isLoading}
    >
      {isLoading ? (
        <>
          <Spinner className="mr-2 h-4 w-4" />
          Saving...
        </>
      ) : (
        "Save Changes"
      )}
    </Button>
  );
}

// Dialog component example
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";

export function SettingsDialog() {
  return (
    <Dialog>
      <DialogTrigger>Open Settings</DialogTrigger>
      <DialogContent className="sm:max-w-[425px]">
        <DialogHeader>
          <DialogTitle>Application Settings</DialogTitle>
          <DialogDescription>
            Configure your preferences for the AI Code Tutor application.
          </DialogDescription>
        </DialogHeader>
        {/* Settings form content */}
      </DialogContent>
    </Dialog>
  );
}
```

## Performance Considerations

1. **CSS Optimization**:
   - Configured Tailwind's PurgeCSS to remove unused styles in production
   - Minimized custom CSS outside of Tailwind utilities

2. **Component Lazy Loading**:
   - Implemented dynamic imports for larger Shadcn components
   - Used code splitting to improve initial load time

3. **Responsive Optimizations**:
   - Used responsive utility classes to avoid media query duplication
   - Leveraged Tailwind's JIT compiler for optimal CSS output

## Further Resources

- [Tailwind CSS Documentation](https://tailwindcss.com/docs)
- [Shadcn/UI Documentation](https://ui.shadcn.com)
- [Radix UI Primitives](https://www.radix-ui.com/)