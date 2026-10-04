import { test, expect } from "@playwright/test";
import { spawnSync } from "node:child_process";
let cpu: any[];
test.beforeAll(() => {
  const env = { ...process.env, CUDA_VISIBLE_DEVICES: "" };
  delete env.PYTHONPATH;
  const p = spawnSync("python", ["-m", "experiments.sequence_reference"], {
    input: JSON.stringify([
      { kind: "sp", layer: 0 },
      { kind: "cp", layout: "thd", padding: 8, cp: 1 },
    ]),
    encoding: "utf8",
    env,
    maxBuffer: 8 * 1024 * 1024,
  });
  expect(p.status, p.stderr).toBe(0);
  cpu = JSON.parse(p.stdout);
});
test("[G05] 1F1B timeline follows two-layer allocation, dependencies and activation lifetime", async ({
  page,
}) => {
  await page.goto("/#pp=2&microbatches=2&pipelineMicrobatch=0");
  const p = page.getByLabel("PP SP CP sequence journey");
  await expect(p.getByTestId("pipeline-cost")).toContainText("makespan=6");
  await expect(p.getByTestId("pipeline-cost")).toContainText(
    "idle stage-slots=4",
  );
  await expect(p.getByTestId("pipeline-timeline")).toContainText(
    "embedding + L0",
  );
  await expect(p.getByTestId("pipeline-timeline")).toContainText(
    "L1 + final norm/head",
  );
  const rows = p.getByTestId("pipeline-timeline").locator("tbody tr");
  await expect(rows.nth(0).locator("td")).toHaveText([
    "F0",
    "F1",
    "·",
    "B0",
    "·",
    "B1",
  ]);
  await expect(rows.nth(1).locator("td")).toHaveText([
    "·",
    "F0",
    "B0",
    "F1",
    "B1",
    "·",
  ]);
  await p.getByLabel("选定 microbatch", { exact: true }).selectOption("1");
  await expect(p.getByTestId("pipeline-lifetime")).toContainText(
    "stage0 activation [1,6)",
  );
  await expect(p.getByTestId("pipeline-lifetime")).toContainText(
    "峰值保留数=[2,1]",
  );
  await p.getByLabel("PP", { exact: true }).selectOption("1");
  await expect(p.getByTestId("pipeline-cost")).toContainText("makespan=8");
  await expect(p.getByTestId("pipeline-cost")).toContainText("bubble=0.00%");
  await p.getByLabel("Microbatches", { exact: true }).fill("3");
  await expect(p.getByTestId("pipeline-cost")).toContainText("makespan=12");
  await page.reload();
  await expect(p.getByLabel("Microbatches", { exact: true })).toHaveValue("3");
  await p.screenshot({ path: "runs/program-v1/g05-desktop.png" });
});
test("[G05] SP output matches complete CPU layer and remains distinct from CP layout", async ({
  page,
}) => {
  await page.goto("/#tp=2&tinyLayer=0");
  const p = page.getByLabel("PP SP CP sequence journey");
  await expect(p.getByTestId("sp-layout")).toContainText("[10,8]");
  await expect(p.getByTestId("sp-layout")).toContainText("[20,6]");
  await expect(p.getByTestId("sp-result")).toContainText(
    cpu[0].output[0].map((v: number) => v.toFixed(8)).join(", "),
  );
  await page
    .getByLabel("TP DP rank inspector")
    .getByLabel("TP", { exact: true })
    .selectOption("1");
  await expect(p.getByTestId("sp-layout")).toContainText("[20,12]");
  await expect(p.getByTestId("sp-result")).toContainText(
    cpu[0].output[0].map((v: number) => v.toFixed(8)).join(", "),
  );
});
test("[G05] THD boundaries, remote KV, leakage and padding queries change the actual attention", async ({
  page,
}) => {
  await page.goto(
    "/#sequenceLayout=thd&cp=2&sftPadding=1&sequenceRank=0&sequenceQuery=38",
  );
  const p = page.getByLabel("PP SP CP sequence journey");
  await expect(p.getByRole("alert")).toContainText("整除 2×CP");
  await p.getByLabel("序列 padding", { exact: true }).selectOption("8");
  await expect(p.getByRole("alert")).toHaveCount(0);
  await expect(p.getByTestId("cp-metadata")).toContainText(
    "cu_seqlens=[0,11,34]",
  );
  await expect(p.getByTestId("cp-metadata")).toContainText(
    "cu_seqlens_padded=[0,16,40]",
  );
  await expect(p.getByTestId("cp-metadata")).toContainText("逐文档 zigzag");
  await expect(p.getByTestId("cp-result")).toContainText(
    cpu[1].outputs[38][0].map((v: number) => v.toFixed(8)).join(", "),
  );
  await expect(
    p.getByTestId("cp-keys").locator("tbody tr").nth(1),
  ).toContainText("[22,23,24,25,26,27,28,29,30,31,32,33]");
  const correct = await p.getByTestId("cp-result").textContent();
  await p.getByLabel("CP 反例", { exact: true }).selectOption("local_kv");
  await expect(p.getByTestId("cp-result")).not.toHaveText(correct!);
  await expect(
    p.getByTestId("cp-keys").locator("tbody tr").nth(1),
  ).toContainText("[]");
  await p.getByLabel("CP 反例", { exact: true }).selectOption("leak");
  await expect(
    p.getByTestId("cp-keys").locator("tbody tr").nth(1),
  ).toContainText("4,5,6,7");
  await p.getByLabel("CP 反例", { exact: true }).selectOption("none");
  await p.getByLabel("CP query", { exact: true }).selectOption("16");
  await expect(p.getByTestId("cp-query")).toContainText("position=0");
  await p.getByLabel("CP query", { exact: true }).selectOption("12");
  await expect(p.getByTestId("cp-query")).toContainText(
    "padding query 不计算 softmax",
  );
  await expect(p.getByTestId("cp-result")).not.toContainText("NaN");
  await p.getByLabel("CP query", { exact: true }).selectOption("38");
  const pending = page.waitForEvent("download");
  await p.getByRole("button", { name: "导出 sequence 参考" }).click();
  const d = await pending,
    stream = await d.createReadStream(),
    chunks: Buffer[] = [];
  for await (const c of stream!) chunks.push(c);
  const exportData = JSON.parse(Buffer.concat(chunks).toString());
  expect(exportData.provenance).toBe("reference_simulation");
  expect(exportData.cp.cu_seqlens_padded).toEqual([0, 16, 40]);
  expect(exportData.cp.max_error).toBeLessThan(1e-10);
  expect(exportData.timing).toBe("logical_units_not_wall_time");
});
test("[G05] offline source branches, keyboard course and narrow sequence tables remain usable", async ({
  page,
  context,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/#sequenceLayout=thd&sftPadding=8");
  const p = page.getByLabel("PP SP CP sequence journey");
  await expect(p).toBeVisible();
  await context.setOffline(true);
  for (const [button, id, text] of [
    ["查看 1F1B 调度源码 ↗", "pp-1f1b", "send_forward_recv_backward"],
    [
      "查看 SP gather / reduce-scatter 源码 ↗",
      "sp-gather",
      "_reduce_scatter_along_first_dim",
    ],
    ["查看 CP 分片源码 ↗", "cp-document-zigzag", "thd_get_partitioned_indices"],
  ]) {
    await p.getByRole("button", { name: button, exact: true }).click();
    await expect(page.getByLabel("关键源码片段")).toHaveAttribute(
      "data-snippet-id",
      id,
    );
    await expect(
      page.getByRole("dialog").locator(".source-code"),
    ).toContainText(text);
    await page.keyboard.press("Escape");
  }
  await p
    .getByLabel("Sequence layout", { exact: true })
    .selectOption("ordinary");
  await p.getByRole("button", { name: "查看 CP 分片源码 ↗" }).click();
  await expect(page.getByLabel("关键源码片段")).toHaveAttribute(
    "data-snippet-id",
    "cp-sequence-zigzag",
  );
  await page.keyboard.press("Escape");
  await p.locator("summary").focus();
  await page.keyboard.press("Enter");
  await expect(p.locator("details")).toHaveAttribute("open", "");
  await expect(p.locator(".prose")).toContainText("独立数学符号表");
  await expect(p.locator(".katex-error")).toHaveCount(0);
  expect(await p.locator(".katex").count()).toBeGreaterThan(30);
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await p.locator("summary").click();
  await p.screenshot({ path: "runs/program-v1/g05-mobile.png" });
});
