/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'dark-bg': '#080c12',
        'card-bg': '#0f1824',
        'accent-green': '#3ef5c8',
        'accent-red': '#ff6b6b',
        'accent-yellow': '#ffd166',
        'accent-blue': '#58a6ff',
        'accent-purple': '#bc8cff',
        'text-primary': '#dde4f0',
        'text-muted': '#4e6080',
        'border-color': '#1e2d45',
      },
      fontFamily: {
        'syne': ['Syne', 'sans-serif'],
        'mono': ['JetBrains Mono', 'monospace'],
      },
    },
  },
  plugins: [],
}
