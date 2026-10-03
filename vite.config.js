import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // Forward API calls to the Flask recommendation server (backend/app.py).
    proxy: {
      '/api': 'http://127.0.0.1:5000',
      '/images': 'http://127.0.0.1:5000',
    },
  },
})
