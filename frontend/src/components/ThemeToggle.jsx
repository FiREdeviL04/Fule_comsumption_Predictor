function ThemeToggle({ theme, onToggle }) {
  return (
    <button className="secondary" onClick={onToggle}>
      Switch to {theme === "dark" ? "Light" : "Dark"} Theme
    </button>
  );
}

export default ThemeToggle;
