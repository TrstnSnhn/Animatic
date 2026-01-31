/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          950: "#05060A",
          900: "#0A0B12",
          850: "#0D1020",
        },
        neon: {
          cyan: "#00C2FF",
          violet: "#8B5CFF",
          pink: "#FF47B7",
        },
      },
      boxShadow: {
        glass:
          "0 0 0 1px rgba(255,255,255,0.08), 0 24px 70px rgba(0,0,0,0.55)",
        soft:
          "0 0 0 1px rgba(255,255,255,0.06), 0 14px 38px rgba(0,0,0,0.45)",
      },
      backgroundImage: {
        "anim-vignette":
          "radial-gradient(ellipse at center, rgba(255,255,255,0.10) 0%, rgba(0,0,0,0.82) 55%, rgba(0,0,0,1) 100%)",
        "anim-glow":
          "linear-gradient(90deg, rgba(0,194,255,0.85) 0%, rgba(139,92,255,0.35) 42%, rgba(255,71,183,0.85) 100%)",
        "anim-badge":
          "linear-gradient(135deg, #00C2FF 0%, #8B5CFF 55%, #FF47B7 100%)",
      },
      borderRadius: {
        "2xl": "1.25rem",
      },
    },
  },
  plugins: [],
};
