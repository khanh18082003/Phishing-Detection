/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    maxWidth: {
      "1/6": "16.6667%",
      "5/6": "83.333%",
      "1/2": "50%",
      "1/3": "33.333%",
    },

    extend: {},
  },
  plugins: [],
};
