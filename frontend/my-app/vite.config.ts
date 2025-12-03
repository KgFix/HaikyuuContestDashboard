import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // bind to all network interfaces so the dev server is reachable
    // at your machine IP (e.g. 10.0.0.3)
    host: '0.0.0.0',
  },
})
