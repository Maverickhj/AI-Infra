import { test, expect } from "@playwright/test";

test.beforeEach(async ({ page }) => {
  await page.goto("/");
});

test("Qwen3: full sample → model → loss → update with source and math", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await expect(page.getByRole("heading", { level: 1 })).toHaveText(
    "Qwen3-0.6B 整模漫游",
  );
  await expect(
    page.getByLabel("Decoder 层选择").getByRole("button"),
  ).toHaveCount(28);
  await expect(page.getByTestId("attention-facts")).toContainText("4096");
  await expect(page.locator(".katex").first()).toBeVisible();
  const flow = page.getByLabel("完整计算流程");
  await flow.getByRole("button", { name: /样本与目标/ }).click();
  await expect(page.getByTestId("message-span")).toHaveCount(4);
  await expect(page.locator('[data-supervised="true"]')).toHaveCount(2);
  await page.getByRole("button", { name: "last_turn", exact: true }).click();
  await expect(page.locator('[data-supervised="true"]')).toHaveCount(1);
  await page.getByRole("button", { name: "full", exact: true }).click();
  await expect(page.locator('[data-supervised="true"]')).toHaveCount(4);
  await page.getByLabel("移位反例").selectOption("twice");
  await expect(page.getByRole("status")).toContainText("错误");
  for (const step of [
    "Embedding",
    "Decoder",
    "Final norm",
    "LM head",
    "Masked loss",
    "Backward",
    "更新与恢复",
  ]) {
    await flow.getByRole("button", { name: new RegExp(step) }).click();
    await expect(
      page.getByLabel("当前步骤").getByRole("heading", { level: 2 }),
    ).toContainText(step);
  }
  await expect(page.getByLabel("当前步骤")).toContainText("optimizer");
  await page.getByRole("button", { name: "查看当前步骤源码 ↗" }).click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await expect(
    page.getByRole("link", { name: "打开完整源码文件 ↗" }),
  ).toHaveAttribute("href", /\/blob\/[a-f0-9]{40}\//);
  await expect(page.getByRole("dialog")).toContainText("train_step");
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog")).not.toBeVisible();
  await expect(
    page.getByRole("button", { name: "查看当前步骤源码 ↗" }),
  ).toBeFocused();
  expect(errors).toEqual([]);
});

test("Model comparison changes QKV, norm and MoE instead of only title", async ({
  page,
}) => {
  await page.getByLabel("当前模型", { exact: true }).selectOption("qwen25-05b");
  await expect(
    page.getByLabel("Decoder 层选择").getByRole("button"),
  ).toHaveCount(24);
  const facts = page.getByTestId("attention-facts");
  await expect(facts).toContainText("1152");
  await expect(
    facts.locator(".fact").filter({ hasText: "QKV bias" }),
  ).toContainText("开启");
  await expect(
    facts.locator(".fact").filter({ hasText: "QK norm" }),
  ).toContainText("关闭");
  await page.getByLabel("当前模型", { exact: true }).selectOption("qwen3-06b");
  await expect(facts).toContainText("4096");
  await expect(
    facts.locator(".fact").filter({ hasText: "QKV bias" }),
  ).toContainText("关闭");
  await page.getByLabel("Q head", { exact: true }).fill("3");
  await expect(page.getByText("Q head 3 → KV group 1")).toBeVisible();
  await page.getByRole("button", { name: /模型对照/ }).click();
  await page.getByRole("button", { name: /Qwen3 MoE Qwen3-30B-A3B/ }).click();
  await expect(
    page.getByLabel("Decoder 层选择").getByRole("button"),
  ).toHaveCount(48);
  await expect(page.getByLabel("当前步骤")).toContainText("Shared experts: 0");
  await expect(page.getByRole("table").first()).toContainText(
    "DeepSeek-R1-Distill-Qwen",
  );
});

test("DeepSeek MLA, dense/MoE boundary and complete RL cycle", async ({
  page,
}) => {
  await page
    .getByLabel("当前模型", { exact: true })
    .selectOption("deepseek-v3");
  await expect(
    page.getByLabel("Decoder 层选择").getByRole("button"),
  ).toHaveCount(61);
  await expect(page.getByTestId("mla-graph")).toContainText(
    "q_down → norm → q_up",
  );
  await expect(page.getByTestId("mla-graph")).toContainText("kv_down");
  await page
    .getByRole("button", { name: "Layer 2 Dense", exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name: /Layer 2 · SwiGLU/ }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Layer 3 MoE", exact: true }).click();
  await expect(page.getByLabel("当前步骤")).toContainText("Shared experts: 1");
  await page.getByRole("button", { name: "MLA decode 视图" }).click();
  await expect(page.getByTestId("mla-graph")).toContainText("Latent cache");
  await expect(page.getByTestId("mla-graph")).not.toContainText("q_up");
  await page
    .getByLabel("当前模型", { exact: true })
    .selectOption("deepseek-v2-lite");
  await page.getByRole("button", { name: "MLA 训练视图" }).click();
  await expect(page.getByTestId("mla-graph")).toContainText(
    "直接 Q projection",
  );
  await page.getByRole("button", { name: "RL Cycle", exact: true }).click();
  await expect(page.locator(".logprob")).toHaveCount(4);
  for (const name of [
    "generation_logprobs",
    "prev_logprobs",
    "current_logprobs",
    "reference_policy_logprobs",
  ])
    await expect(
      page.locator(".logprob code").filter({ hasText: name }),
    ).toBeVisible();
  for (const step of [
    "Reward",
    "四类 logprob",
    "Advantage",
    "Policy update",
    "Export / refit",
  ])
    await page
      .getByLabel("完整计算流程")
      .getByRole("button", { name: new RegExp(step) })
      .click();
  await expect(page.getByLabel("当前步骤")).toContainText(
    "training_weight_version",
  );
  await expect(page.getByLabel("当前步骤")).toContainText("not_run");
  await page.getByRole("button", { name: "查看当前步骤源码 ↗" }).click();
  await expect(page.getByRole("dialog")).toContainText("R-WORKER");
  await page.getByLabel("来源通道").selectOption("runtime");
  await expect(page.getByRole("dialog")).toContainText("运行来源尚未建立");
  await expect(
    page.getByRole("link", { name: "打开完整源码文件 ↗" }),
  ).toHaveCount(0);
});

test("Skippable foundations preserve core course and independent symbols", async ({
  page,
}) => {
  await page.getByRole("button", { name: /基础速览/ }).click();
  const skip = page.getByRole("button", { name: "跳过基础，进入整模核心 →" });
  await skip.focus();
  await page.keyboard.press("Enter");
  await expect(page.getByTestId("attention-facts")).toContainText("2048");
  await page.getByRole("button", { name: /完整课程/ }).click();
  await expect(
    page.getByText("核心精讲：Attention 不是“1024 除以16”", { exact: false }),
  ).toBeVisible();
  await page.getByText("独立数学符号表 · 随时查阅").click();
  await expect(page.locator(".symbols")).toContainText(
    "每个 Q/K/V head 的维度",
  );
  await expect(page.locator(".symbols .katex").first()).toBeVisible();
  await expect(page.locator(".katex-error")).toHaveCount(0);
  await expect(page.locator(".course")).toContainText("两个必要的反例");
});

test("Deep links reload complete state and invalid inputs normalize safely", async ({
  page,
}) => {
  await page.goto(
    "/#model=deepseek-v3&scenario=sft&layer=5&step=decoder&sourceLane=reference&view=walkthrough&mla=decode",
  );
  await expect(
    page.getByRole("button", { name: "Layer 5 MoE", exact: true }),
  ).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByTestId("mla-graph")).toContainText("Latent cache");
  await page.reload();
  await expect(page.getByLabel("当前模型", { exact: true })).toHaveValue(
    "deepseek-v3",
  );
  await expect(
    page.getByRole("button", { name: "Layer 5 MoE", exact: true }),
  ).toHaveAttribute("aria-pressed", "true");
  await page.goto("/#model=invalid&layer=99999&step=invalid&view=invalid");
  await expect(page.getByLabel("当前模型", { exact: true })).toHaveValue(
    "qwen3-06b",
  );
  await expect(
    page.getByLabel("当前步骤").getByRole("heading", { level: 2 }),
  ).toContainText("样本与目标");
});

test("Narrow screen navigation, dialogs and tables stay usable", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole("button", { name: "查看当前步骤源码 ↗" }).click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await page.getByRole("button", { name: "关闭源码" }).click();
  await page.getByRole("button", { name: /模型对照/ }).click();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await expect(page.locator(".table-wrap")).toBeVisible();
  await page.getByRole("button", { name: /样本与监督/ }).click();
  await expect(page.getByTestId("message-span")).toHaveCount(4);
  await page.getByRole("button", { name: /整模漫游/ }).click();
  await page.screenshot({
    path: "runs/screenshots/mobile-qwen3.png",
    fullPage: true,
  });
});

test("All lessons render math without errors and capture review screenshots", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.screenshot({
    path: "runs/screenshots/desktop-qwen3.png",
    fullPage: true,
  });
  await page
    .getByLabel("当前模型", { exact: true })
    .selectOption("deepseek-v3");
  await page.getByRole("button", { name: "Layer 3 MoE", exact: true }).click();
  await page.screenshot({
    path: "runs/screenshots/desktop-deepseek.png",
    fullPage: true,
  });
  for (const scenario of ["sft", "rl"]) {
    await page.goto(
      `/#model=deepseek-v3&scenario=${scenario}&view=course&step=${scenario === "rl" ? "logprobs" : "decoder"}`,
    );
    await page.getByText("独立数学符号表 · 随时查阅").click();
    await expect(page.locator(".katex-error")).toHaveCount(0);
    expect(await page.locator(".katex").count()).toBeGreaterThan(10);
  }
  await page.getByRole("button", { name: /整模漫游/ }).click();
  await page.screenshot({
    path: "runs/screenshots/desktop-rl.png",
    fullPage: true,
  });
});
