// theme-provider.jsx
// Shadcn/UI theme provider component for React

"use client"

import * as React from "react"
import { ThemeProvider as NextThemesProvider } from "next-themes"

/**
 * ThemeProvider component from Shadcn/UI
 * This provider enables theme support for Shadcn/UI components
 * It handles both light and dark themes using the 'next-themes' library
 */
export function ThemeProvider({ 
  children, 
  defaultTheme = "system", 
  storageKey = "ui-theme",
  ...props 
}) {
  return (
    <NextThemesProvider
      attribute="class"
      defaultTheme={defaultTheme}
      enableSystem
      storageKey={storageKey}
      {...props}
    >
      {children}
    </NextThemesProvider>
  )
}