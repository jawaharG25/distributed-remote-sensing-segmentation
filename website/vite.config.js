import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  base: '/distributed-remote-sensing-segmentation/',
  plugins: [react()]
});
