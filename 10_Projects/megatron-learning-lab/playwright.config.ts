import { defineConfig, devices } from "@playwright/test";
export default defineConfig({
  testDir: "./tests/browser",
  outputDir: "./runs/browser-results",
  reporter: [
    ["list"],
    ["json", { outputFile: "runs/browser-results-summary.json" }],
    ["html", { outputFolder: "runs/browser-report", open: "never" }],
  ],
  use: {
    baseURL: "http://127.0.0.1:5173",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: "npm run dev",
    url: "http://127.0.0.1:5173",
    reuseExistingServer: !process.env.CI,
    timeout: 60000,
  },
});
