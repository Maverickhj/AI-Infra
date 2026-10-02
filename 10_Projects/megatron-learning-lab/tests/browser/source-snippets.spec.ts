import { test, expect } from "@playwright/test";
import fs from "node:fs";
const snippets = JSON.parse(
  fs.readFileSync(
    new URL("../../content/source-snippets.json", import.meta.url),
    "utf8",
  ),
);

test("All curated snippets remain verbatim and readable without network", async ({
  page,
  context,
}) => {
  await page.goto("/");
  await page.getByRole("button", { name: "查看当前步骤源码 ↗" }).waitFor();
  await context.setOffline(true);
  await page.getByRole("button", { name: "查看当前步骤源码 ↗" }).click();
  for (const entry of snippets.entries) {
    const ids = entry.excerpts.map((excerpt: { id: string }) => excerpt.id);
    expect(
      ids.every((id: string) => typeof id === "string" && id.length > 0),
    ).toBe(true);
    expect(new Set(ids).size).toBe(ids.length);
    await page
      .getByRole("combobox", { name: "证据条目", exact: true })
      .selectOption(entry.source_id);
    for (let i = 0; i < entry.excerpts.length; i++) {
      const excerpt = entry.excerpts[i];
      await page
        .getByRole("combobox", { name: "代码片段", exact: true })
        .selectOption(excerpt.id);
      await expect(page.getByTestId("source-range")).toHaveText(
        `L${excerpt.start_line}–L${excerpt.end_line}`,
      );
      expect(
        (await page.locator(".source-text").allTextContents()).join(""),
      ).toBe(excerpt.code);
      await expect(page.getByLabel("中文讲解注释")).toContainText(
        excerpt.annotations[0].text,
      );
    }
  }
  await page.getByLabel("来源通道").selectOption("runtime");
  await expect(page.locator(".source-code")).toHaveCount(0);
  await expect(page.getByRole("dialog")).toContainText("运行来源尚未建立");
});

test("Annotation anchors, license, full-file fallback and narrow-screen code scrolling", async ({
  page,
}) => {
  await page.goto("/#model=qwen3-06b&scenario=sft&step=loss");
  await page.getByRole("button", { name: "查看当前步骤源码 ↗" }).click();
  await page
    .getByRole("combobox", { name: "证据条目", exact: true })
    .selectOption("B-LOSS");
  await page.getByRole("button", { name: "定位 L68", exact: true }).click();
  await expect(page.locator(".code-line.highlighted")).toHaveAttribute(
    "data-line",
    "68",
  );
  await expect(
    page.getByRole("link", { name: "打开完整源码文件 ↗" }),
  ).toHaveAttribute("href", /\/blob\/[a-f0-9]{40}\/src\/.*losses.py$/);
  await expect(
    page.getByRole("link", { name: "在完整文件中定位此片段 ↗" }),
  ).toHaveAttribute("href", /#L62-L68$/);
  await page.getByText("上游版权与许可证", { exact: true }).click();
  await expect(page.locator(".source-license")).toContainText("Apache License");
  await page.getByText("上游版权与许可证", { exact: true }).click();
  await page.screenshot({
    path: "runs/screenshots/source-excerpts-desktop.png",
    fullPage: false,
  });
  await page.setViewportSize({ width: 390, height: 844 });
  await page
    .getByRole("combobox", { name: "证据条目", exact: true })
    .selectOption("B-Q3");
  await page
    .getByRole("combobox", { name: "代码片段", exact: true })
    .selectOption("b-q3-l70");
  await expect(page.locator(".source-code")).toBeVisible();
  expect(
    await page
      .getByRole("dialog")
      .evaluate((el) => el.scrollWidth <= el.clientWidth),
  ).toBe(true);
  expect(
    await page
      .locator(".source-code")
      .evaluate((el) => el.scrollWidth > el.clientWidth),
  ).toBe(true);
  await page.locator(".source-code").focus();
  await page.keyboard.press("ArrowRight");
  await page.screenshot({
    path: "runs/screenshots/source-excerpts-mobile.png",
    fullPage: false,
  });
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog")).not.toBeVisible();
});

test("Step routes select semantic excerpts and disclose missing implementations", async ({
  page,
}) => {
  await page.goto("/");
  const flow = page.getByLabel("完整计算流程");
  for (const [step, id, range, code] of [
    [
      "Embedding",
      "gpt-embedding",
      "L336–L345",
      "self.embedding(input_ids=input_ids",
    ],
    [
      "LM head",
      "gpt-output-projection",
      "L762–L767",
      "logits, _ = self.output_layer(",
    ],
    ["Masked loss", "c-gpt-l790", "L790–L796", "labels"],
  ]) {
    await flow.getByRole("button", { name: new RegExp(step) }).click();
    await page.getByRole("button", { name: "查看当前步骤源码 ↗" }).click();
    await expect(page.getByLabel("关键源码片段")).toHaveAttribute(
      "data-snippet-id",
      id,
    );
    await expect(page.getByTestId("source-range")).toHaveText(range);
    await expect(page.locator(".source-code")).toContainText(code);
    await page.keyboard.press("Escape");
  }
  await flow.getByRole("button", { name: /Final norm/ }).click();
  await page.getByRole("button", { name: "查看当前步骤源码 ↗" }).click();
  await expect(page.getByLabel("关键源码片段")).toContainText("实现片段待补");
  await expect(page.locator(".source-code")).toHaveCount(0);
  await page
    .getByRole("combobox", { name: "代码片段", exact: true })
    .selectOption("b-q3-l70");
  await expect(page.locator(".source-code")).toBeVisible();
});

test("MLA train/decode routes select the matching fixed source branch", async ({
  page,
}) => {
  await page.goto("/");
  await page
    .getByLabel("当前模型", { exact: true })
    .selectOption("deepseek-v3");
  for (const [mode, id, range, code] of [
    ["MLA decode 视图", "c-mla-l321", "L321–L335", "cache"],
    ["MLA 训练视图", "c-mla-l638", "L638–L663", "linear_q_down_proj"],
  ]) {
    await page.getByRole("button", { name: mode }).click();
    await page.getByRole("button", { name: "C-MLA · 查阅条件分支 ↗" }).click();
    await expect(page.getByLabel("关键源码片段")).toHaveAttribute(
      "data-snippet-id",
      id,
    );
    await expect(page.getByTestId("source-range")).toHaveText(range);
    await expect(page.locator(".source-code")).toContainText(code);
    await page.keyboard.press("Escape");
  }
});

test("Display formulas retain block semantics while symbols stay inline", async ({
  page,
}) => {
  await page.goto("/");
  // These previously single-line generated formulas must render as display math.
  await expect(
    page.getByLabel("当前步骤").locator(".katex-display"),
  ).toHaveCount(2);
  expect(await page.locator(".prose p > .katex").count()).toBeGreaterThan(0);
  for (const [scenario, step] of [
    ["sft", "loss"],
    ["rl", "advantage"],
    ["rl", "logprobs"],
  ]) {
    await page.goto(`/#scenario=${scenario}&step=${step}`);
    await expect(
      page.getByLabel("当前步骤").locator(".katex-display"),
    ).toHaveCount(1);
    await expect(page.locator(".katex-error")).toHaveCount(0);
  }
  await page.goto("/#view=course&scenario=sft&model=qwen3-06b");
  expect(await page.locator(".course .katex-display").count()).toBeGreaterThan(
    0,
  );
  await expect(page.locator(".katex-error")).toHaveCount(0);
});

test("Q head rejects invalid values and remains valid across model switches", async ({
  page,
}) => {
  await page.goto("/");
  const head = page.getByLabel("Q head", { exact: true });
  for (const value of ["0", "15", "3"]) {
    await head.fill(value);
    await expect(
      page.getByText(
        `Q head ${value} → KV group ${Math.floor(Number(value) / 2)}`,
        { exact: true },
      ),
    ).toBeVisible();
  }
  for (const value of ["3.5", "-1", "16", "", "1e309"]) {
    await head.fill(value);
    await expect(head).toHaveAttribute("aria-invalid", "true");
    await expect(page.getByRole("alert")).toContainText("保留上一个合法 head");
    await expect(
      page.getByText("Q head 3 → KV group 1", { exact: true }),
    ).toBeVisible();
  }
  await head.fill("15");
  await expect(head).toHaveAttribute("aria-invalid", "false");
  await page.getByLabel("当前模型", { exact: true }).selectOption("qwen25-05b");
  await expect(head).toHaveValue("13");
  await page.getByLabel("当前模型", { exact: true }).selectOption("qwen3-06b");
  await expect(head).toHaveValue("13");
});
