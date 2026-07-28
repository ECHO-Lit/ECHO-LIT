import { useEffect, useState } from "react";

/**
 * Tracks Tailwind's class-based dark mode (`darkMode: ["class"]` in tailwind.config.ts).
 *
 * Charts need a resolved colour value, not a CSS variable -- Plotly writes hex into
 * SVG presentation attributes, where `var(...)` does not resolve. So chart colour
 * choices have to branch on the theme in JS.
 */
export function useIsDarkMode(): boolean {
  const [isDark, setIsDark] = useState(
    () => typeof document !== "undefined" && document.documentElement.classList.contains("dark")
  );

  useEffect(() => {
    const root = document.documentElement;
    const observer = new MutationObserver(() => setIsDark(root.classList.contains("dark")));
    observer.observe(root, { attributes: true, attributeFilter: ["class"] });
    return () => observer.disconnect();
  }, []);

  return isDark;
}
