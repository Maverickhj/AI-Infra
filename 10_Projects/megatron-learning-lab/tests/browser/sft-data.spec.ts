import { test, expect } from "@playwright/test";

test("[G02] authored target/mask table, truncation and padding preserve messages", async ({
  page,
}) => {
  await page.goto("/#step=input&view=sample");
  const journey = page.getByLabel("SFT 数据演算"),
    table = journey.getByTestId("sft-tokens");
  await expect(page.getByTestId("message-span")).toHaveCount(4);
  await expect(journey).toContainText("authored token fixture");
  await expect(table.locator("tbody tr")).toHaveCount(20);
  await expect(table.locator('tr[data-mask="1"]')).toHaveCount(6);
  await expect(table.locator("tbody tr").nth(7)).toContainText("<assistant>");
  await expect(table.locator("tbody tr").nth(7)).toContainText("5");
  await expect(journey.getByTestId("sft-summary")).toContainText("N = 6");
  await page.getByRole("button", { name: "last_turn", exact: true }).click();
  await expect(table.locator('tr[data-mask="1"]')).toHaveCount(3);
  await page.getByRole("button", { name: "full", exact: true }).click();
  await expect(table.locator('tr[data-mask="1"]')).toHaveCount(19);
  await page.getByLabel("对齐 padding").selectOption("8");
  await expect(table.locator("tbody tr")).toHaveCount(24);
  await expect(table.locator('tr[data-valid="false"]')).toHaveCount(4);
  await page.getByRole("button", { name: "last_turn", exact: true }).click();
  await page.getByLabel("保留 token 上限").fill("12");
  await expect(journey.getByTestId("sft-summary")).toContainText("N = 0");
  await expect(journey).toContainText("no_supervision");
  await expect(journey.getByTestId("sft-summary")).not.toContainText(
    /NaN|Infinity/,
  );
  await expect(table).not.toContainText(/NaN|Infinity/);
  await page.getByLabel("移位反例").selectOption("twice");
  await expect(page.getByRole("status")).toContainText("错误");
  await expect(page.getByTestId("message-span").last()).toContainText("10。");
});

test("[G02] packed/unpacked loss, block boundaries, export and deep links", async ({
  page,
}) => {
  await page.goto(
    "/#step=input&view=sample&sftLayout=packed&sftPadding=8&mode=assistant",
  );
  const journey = page.getByLabel("SFT 数据演算");
  await expect(journey.getByTestId("sft-layout")).toContainText(
    "microbatch=1，包含 2",
  );
  await expect(journey.getByTestId("sft-metadata")).toContainText(
    "[0, 11, 34]",
  );
  await expect(journey.getByTestId("sft-metadata")).toContainText(
    "[0, 16, 40]",
  );
  const packedSummary = await journey.getByTestId("sft-summary").innerText();
  await journey.getByLabel("查看 query 16", { exact: true }).click();
  await expect(
    journey.getByTestId("sft-keys").locator('[data-allowed="true"]'),
  ).toHaveCount(1);
  await expect(
    journey.getByTestId("sft-keys").locator("span").first(),
  ).toHaveAttribute("data-allowed", "false");
  await page.getByLabel("移位反例").selectOption("leak");
  await expect(page.getByRole("status")).toContainText("跨样本 key 泄漏");
  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: "导出数据参考 JSON" }).click();
  const artifact = await download;
  const stream = await artifact.createReadStream();
  const chunks: Buffer[] = [];
  for await (const chunk of stream!) chunks.push(chunk);
  const exported = JSON.parse(Buffer.concat(chunks).toString());
  expect(exported.provenance).toBe("authored token fixture");
  expect(exported.cuSeqlensPadded).toEqual([0, 16, 40]);
  expect(exported.loss.count).toBe(14);
  await page.getByLabel("数据布局").selectOption("unpacked");
  await expect(journey.getByTestId("sft-layout")).toContainText("[2,24]");
  expect(await journey.getByTestId("sft-summary").innerText()).toBe(
    packedSummary,
  );
  await expect(
    journey.getByTestId("sft-tokens").locator("tbody tr"),
  ).toHaveCount(48);
  await page.reload();
  await expect(page.getByLabel("数据布局")).toHaveValue("unpacked");
  await expect(page.getByLabel("对齐 padding")).toHaveValue("8");
  await page.goto(
    "/#step=input&sftLayout=bad&sftPadding=3&sftLimit=-1&sftQuery=999",
  );
  await expect(page.getByLabel("数据布局")).toHaveValue("single");
  await expect(page.getByLabel("保留 token 上限")).toHaveValue("128");
});

test("[G02] exact source chain, offline math, trace validation and narrow keyboard use", async ({
  page,
  context,
}) => {
  await page.goto("/#step=input&view=sample");
  for (const [name, id, needle] of [
    ["Dataset → collator ↗", "sft-dataset-collate", "self.collate_fn"],
    [
      "collator → shift → pack ↗",
      "sft-collate-shift",
      "build_shifted_labels_and_loss_mask",
    ],
    ["单次移位实现 ↗", "sft-shift", "mask[:, 1:]"],
    ["THD 边界实现 ↗", "sft-pack-boundaries", "cu_seqlens_q_padded"],
    ["gpt_step 调用 ↗", "b-step-l482", "loss_mask"],
    ["loss sum/count ↗", "b-loss-l62", "torch.sum"],
  ]) {
    await page.getByRole("button", { name, exact: true }).click();
    await expect(page.getByLabel("关键源码片段")).toHaveAttribute(
      "data-snippet-id",
      id,
    );
    await expect(
      page.getByRole("dialog").locator(".source-code"),
    ).toContainText(needle);
    await page.keyboard.press("Escape");
  }
  await context.setOffline(true);
  await page
    .getByText("精讲：监督、packing 与独立数学符号表", { exact: true })
    .click();
  await expect(page.locator(".sft-course .katex-error")).toHaveCount(0);
  expect(
    await page.locator(".sft-course .katex-display").count(),
  ).toBeGreaterThanOrEqual(4);
  await page
    .getByText("读取外部 tokenizer trace（待实测核验）", { exact: true })
    .click();
  await page.getByLabel("Tokenizer trace JSON").fill('{"schema_version":1}');
  await page
    .getByRole("button", { name: "读取 tokenizer trace", exact: true })
    .click();
  await expect(page.getByRole("alert")).toContainText("缺少");
  const trace = {
    schema_version: 1,
    kind: "tokenizer_trace",
    model_id: "synthetic/browser-contract",
    tokenizer_revision: "a".repeat(40),
    template_sha256: "b".repeat(64),
    loss_mode: "assistant",
    rendered_text: "<img src=x onerror=alert(1)>",
    messages: [{ role: "assistant", content: "a" }],
    tokens: [
      { id: 1, text: "p" },
      { id: 2, text: "a" },
    ],
    loss_mask: [0, 1],
  };
  await page.getByLabel("Tokenizer trace JSON").fill(JSON.stringify(trace));
  await page
    .getByRole("button", { name: "读取 tokenizer trace", exact: true })
    .click();
  await expect(page.getByTestId("tokenizer-trace")).toContainText(
    "external_unverified",
  );
  await expect(page.getByTestId("tokenizer-trace").locator("img")).toHaveCount(
    0,
  );
  await expect(
    page.getByTestId("tokenizer-trace").locator("tbody tr").first(),
  ).toContainText("a");
  await page.getByLabel("Tokenizer trace JSON").fill("{}");
  await expect(page.getByTestId("tokenizer-trace")).toHaveCount(0);
  await page.setViewportSize({ width: 390, height: 844 });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  const table = page.getByLabel("逐 token 对齐表，可横向滚动").first();
  await table.focus();
  await page.keyboard.press("ArrowRight");
  await expect(table).toBeFocused();
  await page.screenshot({
    path: "runs/screenshots/g02-mobile.png",
    fullPage: true,
  });
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.screenshot({
    path: "runs/screenshots/g02-desktop.png",
    fullPage: true,
  });
});
