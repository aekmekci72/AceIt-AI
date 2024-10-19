/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [
    function({ addUtilities }) {
      const newUtilities = {
        '.cyan-filter': {
          filter: 'brightness(10) saturate(100%) invert(48%) sepia(90%) saturate(2000%) hue-rotate(160deg) brightness(97%) contrast(100%)',
        },
      }
      addUtilities(newUtilities)
    }
  ],
}
