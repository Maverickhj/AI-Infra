import { test, expect } from "@playwright/test";
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
let evidence: any;
test.beforeAll(() => {
  evidence = JSON.parse(
    readFileSync("content/generated/tp-dp-cpu.json", "utf8"),
  );
  expect(evidence.provenance).toBe("reference_simulation");
  for (const [path, hash] of Object.entries(evidence.source_hashes))
    expect(createHash("sha256").update(readFileSync(path)).digest("hex")).toBe(
      hash,
    );
});
const fmt = (v: number | string) => (typeof v === "number" ? v.toFixed(8) : v);
test("[G04] global ranks, shards, input/output/gradient slices and layer links match CPU", async ({
  page,
}) => {
  await page.goto("/#step=decoder&tp=2&dp=2&parallelRank=3&sample=1");
  const p = page.getByLabel("TP DP rank inspector");
  await expect(p.getByTestId("parallel-groups")).toContainText(
    "TP group=[2,3]；DP group=[1,3]",
  );
  const c = evidence.cases.find(
    (c: any) => c.tp === 2 && c.sample === 1 && c.tied,
  );
  for (const op of [
    "qkv",
    "attention_output",
    "ffn_pair",
    "ffn_output",
    "vocab",
  ]) {
    await p.getByLabel("TP 算子", { exact: true }).selectOption(op);
    const event = c.rank_events.find(
        (e: any) => e.name === op && (e.layer === 0 || op === "vocab"),
      ),
      r = event.ranks[1];
    for (const [name, values] of [
      ["rank input", r.input_token7],
      ["rank output / partial", r.token7],
      ["global output", event.global_token7],
    ] as const) {
      await expect(
        p.getByRole("rowheader", { name, exact: true }).locator(".."),
      ).toContainText(values.map(fmt).join(", "));
    }
    await expect(p.getByTestId("parallel-shape")).toContainText(
      "rank-local [" + r.weight_shape.join(",") + "]",
    );
    await expect(
      p
        .getByRole("rowheader", {
          name: "rank parameter gradient",
          exact: true,
        })
        .locator(".."),
    ).toContainText(r.gradient_preview[0].map(fmt).join(", "));
  }
  await expect(p.getByTestId("vocab-reductions")).toContainText("物理 V=28");
  await expect(p.getByTestId("parallel-ranges")).toContainText(
    "有效词表 [14,27]",
  );
  await p.getByLabel("TP 算子", { exact: true }).selectOption("ffn_pair");
  await expect(p.getByTestId("parallel-ranges")).toContainText(
    "gate 行 [6,12]；up 行 [18,24]",
  );
  await p.getByLabel("TP 层", { exact: true }).selectOption("1");
  const e = c.rank_events.find(
    (e: any) => e.name === "ffn_pair" && e.layer === 1,
  );
  await expect(
    p
      .getByRole("rowheader", { name: "rank output / partial", exact: true })
      .locator(".."),
  ).toContainText(e.ranks[1].token7.map(fmt).join(", "));
  await page.reload();
  await expect(p.getByLabel("TP 层", { exact: true })).toHaveValue("1");
  await expect(p.getByLabel("Global rank", { exact: true })).toHaveValue("3");
  await expect(p.getByTestId("parallel-bytes")).toContainText(
    "发送量 2(TP−1)/TP × payload=704 bytes",
  );
  await p.screenshot({ path: "runs/program-v1/g04-desktop.png" });
});
test("[G04] DP accumulation exposes incorrect local means and scoped exports", async ({
  page,
}) => {
  await page.goto("/#step=backward&mode=full&decoderTied=untied");
  const p = page.getByLabel("TP DP rank inspector");
  await expect(p).toContainText("固定 assistant、token 7");
  await expect(p).toContainText("上方监督模式为 full");
  await expect(
    p.getByTestId("dp-counts").locator("tbody tr").nth(0),
  ).toContainText("4");
  await expect(
    p.getByTestId("dp-counts").locator("tbody tr").nth(1),
  ).toContainText("18");
  await expect(p.getByTestId("dp-loss")).toContainText("global_count=22");
  await expect(p.getByTestId("dp-loss")).toContainText(
    "所选 loss=" + fmt(evidence.dp.global_loss),
  );
  await p.getByLabel("DP 归约", { exact: true }).selectOption("wrong");
  await expect(p.getByTestId("dp-loss")).toContainText(
    "所选 loss=" + fmt(evidence.dp.mean_of_means),
  );
  await expect(p.getByTestId("dp-loss")).toContainText("错误");
  expect(evidence.wrong_dp.max_gradient_error).toBeGreaterThan(1e-5);
  const pending = page.waitForEvent("download");
  await p.getByRole("button", { name: "导出 TP DP 切片" }).click();
  const d = await pending,
    stream = await d.createReadStream(),
    parts: Buffer[] = [];
  for await (const part of stream!) parts.push(part);
  const exported = JSON.parse(Buffer.concat(parts).toString());
  expect(exported.provenance).toBe("reference_simulation");
  expect(exported.distributed_backend).toBe("not_run");
  expect(exported.mode).toBe("assistant");
  expect(exported.tied).toBe(false);
  expect(exported.token).toBe(7);
  expect(exported.max_gradient_error).toBeLessThan(1e-10);
  expect(exported.source_hashes).toEqual(evidence.source_hashes);
});
test("[G04] fixed source routes remain verbatim offline and disclose runtime boundary", async ({
  page,
  context,
}) => {
  await page.goto("/#step=decoder");
  const p = page.getByLabel("TP DP rank inspector");
  await expect(p).toBeVisible();
  await context.setOffline(true);
  for (const [op, id, code] of [
    ["qkv", "tp-column-gather", "gather_output"],
    ["ffn_output", "tp-row-sum", "reduce_from_tensor_model_parallel_region"],
    ["vocab", "tp-vocab-ce", "ReduceOp.MAX"],
  ]) {
    await p.getByLabel("TP 算子", { exact: true }).selectOption(op);
    await p.getByRole("button", { name: /查看 TP 算子源码/ }).click();
    await expect(page.getByLabel("关键源码片段")).toHaveAttribute(
      "data-snippet-id",
      id,
    );
    await expect(
      page.getByRole("dialog").locator(".source-code"),
    ).toContainText(code);
    await page.keyboard.press("Escape");
  }
  await p.getByRole("button", { name: "查看 token 归一化源码 ↗" }).click();
  await expect(page.getByLabel("关键源码片段")).toHaveAttribute(
    "data-snippet-id",
    "dp-token-normalize",
  );
  await expect(page.getByRole("dialog").locator(".source-code")).toContainText(
    "model_chunk.scale_gradients(scaling)",
  );
  await page.getByRole("dialog").getByLabel("来源通道").selectOption("runtime");
  await expect(page.getByRole("dialog")).toContainText("运行来源尚未建立");
  await expect(page.getByRole("dialog").locator(".source-code")).toHaveCount(0);
});
test("[G04] illegal dimensions reject, TP1 resets rank, and narrow view is keyboard accessible", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/#tp=2&dp=2&parallelRank=3");
  const p = page.getByLabel("TP DP rank inspector");
  await p.getByRole("button", { name: "验证配置" }).click();
  await expect(p.getByRole("alert")).toContainText("非法 TP");
  await expect(p.getByLabel("TP", { exact: true })).toHaveValue("2");
  await p.getByLabel("试算 TP", { exact: true }).fill("1");
  await p.getByRole("button", { name: "验证配置" }).click();
  await expect(p.getByRole("alert")).toHaveCount(0);
  await expect(p.getByTestId("parallel-groups")).toContainText("world=2");
  await expect(p.getByLabel("Global rank", { exact: true })).toHaveValue("0");
  await expect(p.getByTestId("parallel-shape")).toContainText(
    "rank-local [32,8]",
  );
  await p.getByLabel("DP", { exact: true }).selectOption("1");
  await expect(p.getByTestId("parallel-groups")).toContainText("world=1");
  await expect(p.getByTestId("parallel-bytes")).toContainText(
    "payload=0 bytes",
  );
  await p.locator("summary").focus();
  await page.keyboard.press("Enter");
  await expect(p.locator("details")).toHaveAttribute("open", "");
  await expect(p.locator(".prose")).toContainText("独立符号表");
  expect(await p.locator(".katex").count()).toBeGreaterThan(20);
  await expect(p.locator(".katex-error")).toHaveCount(0);
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBe(true);
  await p.locator("summary").click();
  await p.screenshot({ path: "runs/program-v1/g04-mobile.png" });
});
