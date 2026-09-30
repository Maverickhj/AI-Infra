import { test, expect } from "@playwright/test";
import fs from "node:fs";
const snippets = JSON.parse(
  fs.readFileSync(
    new URL("../../content/source-snippets.json", import.meta.url),
    "utf8",
  ),
);

test("All 45 snippets remain verbatim and readable without network", async ({
  page,
  context,
}) => {
  await page.goto("/");
  await page.getByRole("button", { name: "查看当前步骤源码 ↗" }).waitFor();
  await context.setOffline(true);
  await page.getByRole("button", { name: "查看当前步骤源码 ↗" }).click();
  for (const entry of snippets.entries) {
    await page
      .getByRole("combobox", { name: "证据条目", exact: true })
      .selectOption(entry.source_id);
    for (let i = 0; i < entry.excerpts.length; i++) {
      const excerpt = entry.excerpts[i];
      await page
        .getByRole("combobox", { name: "代码片段", exact: true })
        .selectOption(String(i));
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
    .selectOption("1");
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
