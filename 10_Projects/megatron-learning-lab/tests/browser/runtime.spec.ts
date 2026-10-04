import { test, expect } from "@playwright/test";
import fs from "node:fs";
import { createHash } from "node:crypto";
const traces = JSON.parse(
  fs.readFileSync(
    new URL("../../content/fixtures/runtime-reference.json", import.meta.url),
    "utf8",
  ),
).traces;
const hash = (s: string) => createHash("sha256").update(s).digest("hex");
const slot = (page: any, name = "A") =>
  page.getByRole("article", { name: "Trace " + name, exact: true });
const upload = async (panel: any, value: any) =>
  panel.getByLabel("选择 trace JSON", { exact: false }).setInputFiles({
    name: "trace.json",
    mimeType: "application/json",
    buffer: Buffer.from(
      typeof value === "string" ? value : JSON.stringify(value),
    ),
  });
test("[G08] SFT trace keeps target mask and provenance, compares and exports validated data", async ({
  page,
}) => {
  await page.goto("/#view=runtime");
  await expect(page.getByRole("heading", { level: 1 })).toHaveText(
    "运行证据，逐项核对",
  );
  const a = slot(page),
    b = slot(page, "B"),
    r = traces[0],
    d = JSON.parse(r.input_json);
  await a.getByRole("button", { name: "载入 SFT CPU 样例" }).click();
  await expect(a.getByTestId("trace-provenance")).toContainText("reference");
  await expect(a.getByTestId("trace-provenance")).toContainText(
    "imported_claim",
  );
  await expect(a.getByTestId("trace-loss")).toContainText(
    r.measurements.loss_mean.toPrecision(9),
  );
  const rows = a.getByTestId("trace-tokens").locator("tbody tr");
  await expect(rows).toHaveCount(d.input_ids[0].length);
  for (let j = 0; j < d.input_ids[0].length; j++)
    await expect(rows.nth(j).locator("td")).toHaveText([
      String(j),
      String(d.input_ids[0][j]),
      String(d.labels[0][j]),
      String(d.loss_mask[0][j]),
      r.measurements.token_logprobs[0][j].toPrecision(9),
    ]);
  await b.getByRole("button", { name: "载入 SFT CPU 样例" }).click();
  await expect(page.getByTestId("trace-difference")).toContainText(
    "0.000000e+0",
  );
  await expect(page.getByLabel("Trace 比较结果")).toContainText(
    "不证明运行身份或跨引擎等价",
  );
  const downloaded = page.waitForEvent("download");
  await a.getByRole("button", { name: "导出已校验 trace" }).click();
  const dl = await downloaded;
  expect(JSON.parse(fs.readFileSync((await dl.path())!, "utf8"))).toEqual(r);
  await page.reload();
  await expect(page).toHaveURL(/view=runtime/);
  await expect(slot(page).getByTestId("trace-result")).toHaveCount(0);
});
test("[G08] PPO import remains read-only, preserves policy versions and refuses mismatched tasks", async ({
  page,
  context,
}) => {
  await page.goto("/#view=runtime");
  await context.setOffline(true);
  const a = slot(page),
    b = slot(page, "B"),
    r = structuredClone(traces[1]);
  r.run_id = '<img src=x onerror="window.traceExecuted=true">';
  r.manifest.execution.command = [
    "javascript:window.traceExecuted=true",
    "python",
    "-c",
    "raise RuntimeError('do not run')",
  ];
  await upload(a, r);
  await expect(a.getByTestId("trace-result")).toBeVisible();
  await expect(a.locator("img")).toHaveCount(0);
  expect(
    await page.evaluate(() => Object.hasOwn(window, "traceExecuted")),
  ).toBe(false);
  await expect(a.getByTestId("trace-versions")).toContainText(
    "generation=0 / previous=0 / current=1 / after=2",
  );
  await expect(a.getByTestId("trace-versions")).toContainText("refit=not_run");
  await a.getByLabel("查看序列", { exact: true }).selectOption("3");
  await expect(
    a
      .getByTestId("trace-tokens")
      .locator("tbody tr")
      .nth(10)
      .locator("td")
      .nth(3),
  ).toHaveText("0");
  await b.getByRole("button", { name: "载入 SFT CPU 样例" }).click();
  await expect(page.getByLabel("Trace 比较结果")).toContainText(
    "不同任务不能逐 token 比较",
  );
  await b.getByRole("button", { name: "载入 PPO CPU 样例" }).click();
  await expect(page.getByTestId("trace-difference")).toContainText(
    "0.000000e+0",
  );
  await a.getByRole("button", { name: "载入 PPO CPU 样例" }).click();
  await expect(a.getByTestId("trace-result")).toContainText(
    "authored-ppo-cpu-v1",
  );
  await page.screenshot({
    path: "runs/program-v1/g08-desktop.png",
    fullPage: true,
  });
});
test("[G08] invalid provenance, masks, losses, mappings and JSON fail closed without stale results", async ({
  page,
}) => {
  await page.goto("/#view=runtime");
  const a = slot(page);
  for (const [mutate, message] of [
    [(r: any) => (r.manifest.adapter = "unknown_v9"), "unknown adapter"],
    [
      (r: any) => {
        r.provenance = "observed_bridge";
        r.manifest.evidence_kind = "synthetic_contract";
      },
      "immutable revision",
    ],
    [(r: any) => (r.measurements.loss_mean += 1), "loss reduction"],
    [
      (r: any) => {
        const d = JSON.parse(r.input_json);
        delete d.loss_mask;
        r.input_json = JSON.stringify(d);
        r.input_sha256 = hash(r.input_json);
      },
      "mask",
    ],
    [(r: any) => (r.input_sha256 = "0".repeat(64)), "SHA256"],
  ] as const) {
    await a.getByRole("button", { name: "载入 SFT CPU 样例" }).click();
    await expect(a.getByTestId("trace-result")).toBeVisible();
    const bad = structuredClone(traces[0]);
    mutate(bad);
    await upload(a, bad);
    await expect(a.getByRole("alert")).toContainText(message);
    await expect(a.getByTestId("trace-result")).toHaveCount(0);
  }
  for (const raw of ['{"x":1,"x":2}', '{"__proto__":{}}', '{"x":NaN}']) {
    await a.getByLabel("粘贴 trace JSON").fill(raw);
    await a.getByRole("button", { name: "校验并载入" }).click();
    await expect(a.getByRole("alert")).toContainText("拒绝导入");
  }
  await upload(a, " ".repeat(1048577));
  await expect(a.getByRole("alert")).toContainText("1 MiB");
  await a.getByRole("button", { name: "载入 SFT CPU 样例" }).click();
  await a.getByLabel("粘贴 trace JSON").fill("{broken");
  await a.getByRole("button", { name: "校验并载入" }).click();
  await expect(a.getByRole("alert")).toContainText("拒绝导入");
  await expect(a.getByTestId("trace-result")).toHaveCount(0);
  await a.getByRole("button", { name: "清空", exact: true }).click();
  await expect(a.getByRole("alert")).toHaveCount(0);
});
test("[G08] source links, offline mathematics, keyboard and narrow trace tables work", async ({
  page,
  context,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/#view=runtime");
  await context.setOffline(true);
  const panel = page.getByLabel("只读运行对照"),
    a = slot(page);
  await a.getByRole("button", { name: "载入 PPO CPU 样例" }).focus();
  await page.keyboard.press("Enter");
  await expect(a.getByTestId("trace-result")).toBeVisible();
  await panel.locator(".runtime-course summary").focus();
  await page.keyboard.press("Enter");
  await expect(panel.locator(".runtime-course .prose")).toContainText(
    "独立数学符号表",
  );
  expect(await panel.locator(".runtime-course .katex").count()).toBeGreaterThan(
    10,
  );
  await expect(panel.locator(".katex-error")).toHaveCount(0);
  await panel.getByRole("button", { name: "查看 RL loss 参考源码" }).click();
  await expect(page.getByLabel("关键源码片段")).toHaveAttribute(
    "data-snippet-id",
    "rl-ratio-clip",
  );
  await page.keyboard.press("Escape");
  await panel.locator(".runtime-course summary").click();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth + 1,
    ),
  ).toBe(true);
  await expect(a.getByTestId("trace-tokens").locator("td").first()).toHaveCSS(
    "white-space",
    "nowrap",
  );
  await page.screenshot({
    path: "runs/program-v1/g08-mobile.png",
    fullPage: true,
  });
});
