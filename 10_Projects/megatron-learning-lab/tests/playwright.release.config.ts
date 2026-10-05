import { defineConfig, devices } from "@playwright/test";
import { fileURLToPath } from "node:url";
const root = fileURLToPath(new URL("../", import.meta.url));
export default defineConfig({
  testDir: "./browser",
  testMatch: "release.spec.ts",
  outputDir: "../runs/production-results",
  workers: 1,
  retries: 0,
  use: {
    baseURL: "http://127.0.0.1:5174",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [
    { name: "chromium-production", use: { ...devices["Desktop Chrome"] } },
  ],
  webServer: {
    command:
      "node node_modules/vite/bin/vite.js preview --host 127.0.0.1 --port 5174 --strictPort",
    cwd: root,
    url: "http://127.0.0.1:5174",
    reuseExistingServer: false,
    timeout: 30000,
  },
});
