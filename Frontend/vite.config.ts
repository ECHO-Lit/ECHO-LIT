/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";
import { componentTagger } from "lovable-tagger";

// https://vitejs.dev/config/
//
// `defineConfig` is imported from "vite", not "vitest/config", deliberately.
// Importing it from vitest would make every vite command -- the dev server and
// `npm run build`, not only the tests -- load vitest at config-resolution time.
// The triple-slash reference above is types-only and is erased at transpile,
// so it types the `test` block below without any runtime cost.
export default defineConfig(({ mode }) => ({
  server: {
    host: "::",
    port: 8080,
  },
  plugins: [
    react(),
    mode === 'development' &&
    componentTagger(),
  ].filter(Boolean),
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  test: {
    environment: "jsdom",
    // Fixes the document origin so history.pushState() route cases are
    // deterministic and match the dev server. API_BASE stays cross-origin
    // (http://localhost:8000), which is what the app really does.
    environmentOptions: { jsdom: { url: "http://localhost:8080/" } },
    // Explicit imports in every module: no tsconfig `types` entry needed, and
    // no dependence on whether eslint's no-undef is off for TS files.
    globals: false,
    setupFiles: ["./src/tests/setup.ts"],
    include: ["src/tests/**/*.test.{ts,tsx}"],
    // Tailwind is not processed under test. Consequence: `data-[state=...]`
    // variants never apply, so visibility assertions are banned in the tab
    // families and axe's color-contrast rule is disabled. See the plan doc.
    css: false,
    clearMocks: true,
    restoreMocks: true,
    unstubGlobals: true,
    unstubEnvs: true,
    // Mounting the whole dashboard under jsdom (four panels, each fetching on
    // mount, plus PanelGroup) costs 1-5s on its own, and the cost varies with
    // how many files are running in parallel. 10s proved too tight under
    // --sequence.shuffle; 30s leaves headroom without hiding a real hang.
    testTimeout: 30_000,
  },
}));
