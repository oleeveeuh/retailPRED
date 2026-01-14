/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#3A3A6C',
          50: '#E8E8F0',
          100: '#D1D1E1',
          200: '#A3A3C3',
          300: '#7575A5',
          400: '#474787',
          500: '#3A3A6C',
          600: '#2F2F5A',
          700: '#242448',
          800: '#1A1A36',
          900: '#0F0F24',
        },
        accent: {
          DEFAULT: '#81C1AC',
          50: '#F0F8F5',
          100: '#E0F1EB',
          200: '#C1E3D7',
          300: '#A2D5C3',
          400: '#91CFBA',
          500: '#81C1AC',
          600: '#67AB94',
          700: '#4D957C',
          800: '#337F64',
          900: '#19694C',
        },
      },
    },
  },
  plugins: [],
}
