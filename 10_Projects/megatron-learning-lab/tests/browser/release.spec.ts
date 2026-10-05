import { test, expect } from "@playwright/test";

test("[G09] integrated sample to model, update, parallel, MLA/MoE and RL route", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  await expect(page.getByRole("heading", { level: 1 })).toHaveText(
    "Qwen3-0.6B 整模漫游",
  );
  const nav = page.getByRole("navigation", { name: "学习视图" });
  await nav.getByRole("button", { name: /样本与监督/ }).click();
  await expect(
    page.getByLabel("SFT 数据演算").getByTestId("sft-summary"),
  ).toContainText("N =");
  await page.getByRole("button", { name: "last_turn", exact: true }).click();
  await expect(page.locator('[data-supervised="true"]')).toHaveCount(1);
  await nav.getByRole("button", { name: /整模漫游/ }).click();
  const flow = page.getByLabel("完整计算流程");
  await flow.getByRole("button", { name: /Decoder/ }).click();
  const decoder = page.getByLabel("完整微型 decoder");
  await decoder
    .getByLabel("微型模型子步骤")
    .getByRole("button", { name: "Update / resume" })
    .click();
  await expect(decoder).toContainText("冻结参数不变：true");
  const parallel = page.getByLabel("TP DP rank inspector");
  await parallel.getByLabel("TP", { exact: true }).selectOption("2");
  await parallel.getByLabel("DP", { exact: true }).selectOption("2");
  await parallel.getByLabel("Global rank", { exact: true }).selectOption("3");
  await expect(parallel.getByTestId("parallel-groups")).toContainText(
    "TP group=[2,3]；DP group=[1,3]",
  );
  await page
    .getByLabel("PP SP CP sequence journey")
    .getByLabel("PP", { exact: true })
    .selectOption("2");
  await expect(page.getByTestId("pipeline-timeline")).toContainText(
    "L1 + final norm/head",
  );
  await page
    .getByLabel("当前模型", { exact: true })
    .selectOption("deepseek-v3");
  await flow.getByRole("button", { name: /Decoder/ }).click();
  await page.getByRole("button", { name: "Layer 3 MoE", exact: true }).click();
  await expect(page.getByLabel("当前步骤")).toContainText("Shared experts: 1");
  await page.getByRole("button", { name: "MLA decode 视图" }).click();
  await expect(page.getByTestId("mla-graph")).toContainText("Latent cache");
  await expect(page.getByLabel("MoE MLA 完整微型模型")).toBeVisible();
  await page.getByRole("button", { name: "RL Cycle", exact: true }).click();
  const rl = page.getByLabel("RL 数值参考闭环");
  await expect(rl.getByTestId("rl-scope")).toContainText("非真实 rollout");
  await rl.getByLabel("RL algorithm", { exact: true }).selectOption("ppo");
  await expect(rl.getByTestId("rl-value-loss")).toBeVisible();
  await flow.getByRole("button", { name: /Export \/ refit/ }).click();
  await page.getByRole("button", { name: "查看当前步骤源码 ↗" }).click();
  await expect(page.getByRole("dialog")).toContainText("R-WORKER");
  await page.getByLabel("来源通道").selectOption("runtime");
  await expect(page.getByRole("dialog")).toContainText("运行来源尚未建立");
  await expect(
    page.getByRole("link", { name: "打开完整源码文件 ↗" }),
  ).toHaveCount(0);
  await page.keyboard.press("Escape");
  await nav.getByRole("button", { name: /运行对照/ }).click();
  await expect(page.getByLabel("Trace 比较结果")).toContainText(
    "先载入两份 trace",
  );
  await expect(page.getByTestId("trace-communication-compare")).toContainText(
    "不可推导",
  );
  expect(errors).toEqual([]);
});

test("[G09] direct nested routes load built assets, local math/fonts and work offline", async ({
  page,
  context,
  baseURL,
  browser,
}, info) => {
  const external: string[] = [],
    fonts: string[] = [],
    errors: string[] = [];
  const origin = new URL(baseURL!).origin;
  await context.route("**/*", (route) => {
    const url = new URL(route.request().url());
    if (url.origin !== origin) {
      external.push(url.href);
      return route.abort();
    }
    return route.continue();
  });
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("response", (r) => {
    if (/\.woff2?(\?|$)/.test(r.url()) && r.ok()) fonts.push(r.url());
  });
  await page.goto(
    "/learning/session/#model=qwen3-06b&view=course&step=decoder",
  );
  await page.getByText("独立数学符号表 · 随时查阅", { exact: true }).click();
  await expect(page.locator(".course .katex").first()).toBeVisible();
  await page.evaluate(() => document.fonts.ready);
  expect(fonts.length).toBeGreaterThan(0);
  expect(fonts.every((f) => new URL(f).origin === origin)).toBe(true);
  const scripts = await page
    .locator("script[src]")
    .evaluateAll((nodes) => nodes.map((n) => n.getAttribute("src")));
  if (info.project.name === "chromium-production") {
    expect(
      scripts.some((s) => s?.startsWith("/assets/") && s.endsWith(".js")),
    ).toBe(true);
    expect(
      scripts.some((s) => s?.includes("@vite") || s?.includes("/web/")),
    ).toBe(false);
  }
  await page.reload();
  await expect(page.getByLabel("当前模型", { exact: true })).toHaveValue(
    "qwen3-06b",
  );
  await context.setOffline(true);
  await page.getByRole("button", { name: /基础速览/ }).click();
  await page.getByRole("button", { name: "跳过基础，进入整模核心 →" }).focus();
  await page.keyboard.press("Enter");
  await expect(page.getByLabel("当前步骤")).toContainText("Decoder");
  await page.getByRole("button", { name: /运行对照/ }).click();
  const a = page.getByRole("article", { name: "Trace A", exact: true });
  await a.getByRole("button", { name: "载入 PPO CPU 样例" }).click();
  await expect(a.getByTestId("trace-provenance")).toContainText("reference");
  await expect(a.getByTestId("trace-provenance")).toContainText(
    "imported_claim",
  );
  await page.locator(".runtime-course summary").click();
  await expect(page.locator(".runtime-course .prose")).toContainText(
    "独立数学符号表",
  );
  await expect(page.locator(".katex-error")).toHaveCount(0);
  await page.evaluate(() => document.fonts.ready);
  expect(
    await page.evaluate(() =>
      Array.from(document.fonts)
        .filter((f) => f.family.includes("KaTeX") && f.status === "error")
        .map((f) => f.family),
    ),
  ).toEqual([]);
  expect(external).toEqual([]);
  expect(errors).toEqual([]);
  await info.attach("browser-and-fonts", {
    body: JSON.stringify({
      browser: browser.version(),
      fonts,
      scripts,
      external,
    }),
    contentType: "application/json",
  });
});

test("[G09] narrow production comparison supports keyboard and missing evidence", async ({
  page,
}, info) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/review/#view=runtime");
  const a = page.getByRole("article", { name: "Trace A", exact: true });
  await a.getByRole("button", { name: "载入 SFT CPU 样例" }).focus();
  await page.keyboard.press("Enter");
  await expect(a.getByTestId("trace-result")).toBeVisible();
  await page
    .getByRole("article", { name: "Trace B", exact: true })
    .getByRole("button", { name: "载入 SFT CPU 样例" })
    .click();
  await expect(page.getByTestId("trace-metric-compare")).toContainText(
    "未采集可比较的选定梯度",
  );
  await expect(page.getByTestId("trace-communication-compare")).toContainText(
    "未采集",
  );
  const region = page.getByRole("region", {
    name: "Loss 和梯度差值",
    exact: true,
  });
  await region.focus();
  await expect(region).toBeFocused();
  await page.keyboard.press("ArrowRight");
  await expect
    .poll(() => region.evaluate((e) => e.scrollLeft))
    .toBeGreaterThan(0);
  await region.evaluate((e) => {
    e.scrollLeft = 0;
  });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth + 1,
    ),
  ).toBe(true);
  await region.scrollIntoViewIfNeeded();
  await page.screenshot({ path: info.outputPath("comparison-mobile.png") });
  await info.attach("comparison-mobile", {
    path: info.outputPath("comparison-mobile.png"),
    contentType: "image/png",
  });
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page
    .getByLabel("Trace 比较结果", { exact: true })
    .scrollIntoViewIfNeeded();
  await page.screenshot({ path: info.outputPath("comparison-desktop.png") });
  await info.attach("comparison-desktop", {
    path: info.outputPath("comparison-desktop.png"),
    contentType: "image/png",
  });
});
