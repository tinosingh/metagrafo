import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0', // Listen on all network interfaces
    port: 9000,
    strictPort: true,
    open: true,
  },
  build: {
    minify: false, // Disable minification for debugging
    sourcemap: true, // Generate source maps
  },
  logLevel: 'info', // Set log level to 'info' (options: 'info', 'warn', 'error', 'silent')
});
