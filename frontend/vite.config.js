
import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import { splitVendorChunkPlugin } from 'vite';
import { visualizer } from 'rollup-plugin-visualizer';
import { createHtmlPlugin } from 'vite-plugin-html';
import { compression } from 'vite-plugin-compression';

export default defineConfig(({ command, mode }) => {
  // Load env variables
  const env = loadEnv(mode, process.cwd(), '');
  
  return {
    plugins: [
      react({
        jsxRuntime: 'automatic',
        babel: {
          plugins: [
            ['@emotion/babel-plugin', { sourceMap: true }]
          ]
        }
      }),
      splitVendorChunkPlugin(),
      // HTML optimization plugin
      createHtmlPlugin({
        minify: mode === 'production',
        inject: {
          data: {
            title: 'AI Code Tutor',
            injectScript: mode === 'production' ? `<script src="./injectScript.js"></script>` : '',
          },
        },
      }),
      // Generate bundle analyzer report in stats.html (only for production)
      mode === 'production' && visualizer({
        filename: 'stats.html',
        open: false,
        gzipSize: true,
        brotliSize: true,
      }),
      // Compress assets for production
      mode === 'production' && compression({
        algorithm: 'brotliCompress',
        ext: '.br',
      }),
      // Gzip fallback for browsers that don't support Brotli
      mode === 'production' && compression({
        algorithm: 'gzip',
        ext: '.gz',
      })
    ].filter(Boolean),
    resolve: {
      extensions: ['.js', '.jsx', '.json']
    },
    esbuild: {
      loader: 'jsx',
      include: /\.[jt]sx?$/,
      exclude: []
    },
    build: {
      // Enable minification for production builds
      minify: 'terser',
      terserOptions: {
        compress: {
          drop_console: true, // Remove console.log in production
          drop_debugger: true, // Remove debugger statements
          pure_funcs: ['console.log', 'console.debug', 'console.info']
        },
        mangle: {
          safari10: true, // Fix Safari 10 bugs
        },
        format: {
          comments: false // Remove comments in production
        }
      },
      // Enable source map in development but not in production
      sourcemap: mode !== 'production',
      // Split chunks for better caching
      rollupOptions: {
        output: {
          // Hash file names for better cache invalidation
          entryFileNames: mode === 'production' ? 'assets/[name].[hash].js' : 'assets/[name].js',
          chunkFileNames: mode === 'production' ? 'assets/[name].[hash].js' : 'assets/[name].js',
          assetFileNames: mode === 'production' ? 'assets/[name].[hash].[ext]' : 'assets/[name].[ext]',
          manualChunks: (id) => {
            // Create more granular chunks for better caching
            if (id.includes('node_modules')) {
              if (id.includes('react')) {
                return 'vendor-react';
              }
              if (id.includes('monaco-editor')) {
                return 'vendor-monaco';
              }
              if (id.includes('react-icons')) {
                return 'vendor-icons';
              }
              if (id.includes('react-syntax-highlighter') || id.includes('react-markdown')) {
                return 'vendor-markdown';
              }
              // Group other third-party dependencies
              return 'vendor-deps';
            }
            // Group by feature for app code
            if (id.includes('/components/')) {
              if (id.includes('/ui/')) {
                return 'ui-components';
              }
              if (id.includes('/CodePlayground/')) {
                return 'feature-playground';
              }
              if (id.includes('/PracticeQuestion/')) {
                return 'feature-practice';
              }
              return 'app-components';
            }
            if (id.includes('/context/')) {
              return 'app-context';
            }
            if (id.includes('/hooks/') || id.includes('/utils/')) {
              return 'app-utils';
            }
          }
        }
      },
      // Optimize chunk size warning limits
      chunkSizeWarningLimit: 1000,
      // Enable css code splitting
      cssCodeSplit: true,
      // Reduce asset file size warnings threshold
      assetsInlineLimit: 4096,
      // Target modern browsers for better tree-shaking and smaller bundle
      target: 'es2015',
      // Avoid empty chunks
      emptyOutDir: true,
    },
    server: {
      // Optimize dev server
      hmr: {
        overlay: true
      },
      // Improve cold-start performance
      fs: {
        strict: true
      },
      // Add middleware for potential proxy needs
      proxy: {
        '/api': {
          target: env.VITE_API_URL || 'http://localhost:8000',
          changeOrigin: true,
          secure: false
        }
      }
    },
    // Performance optimizations
    optimizeDeps: {
      include: [
        'react',
        'react-dom',
        'react-router-dom',
        '@monaco-editor/react',
        'react-icons'
      ],
      esbuildOptions: {
        target: 'es2020'
      }
    }
  };
});
