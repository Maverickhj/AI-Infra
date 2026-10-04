import { test, expect } from "@playwright/test";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
let references: any[];
test.beforeAll(() => {
  const env = { ...process.env, CUDA_VISIBLE_DEVICES: "" };
  delete env.PYTHONPATH;
  const p = spawnSync(
    "python",
    ["-m", "experiments.moe_mla_reference", "--forward"],
    { env, encoding: "utf8", maxBuffer: 24 * 1024 * 1024 },
  );
  expect(p.status, p.stderr).toBe(0);
  references = JSON.parse(p.stdout);
});
const cpu = (family: string, padding = 0) =>
  references.find(
    (r) => r.family === family && r.sample === 1 && r.padding === padding,
  ).trace;
const vector = (a: number[]) => a.map((v) => v.toFixed(8)).join(", ");

test("[G06] complete Qwen MoE output, dispatch and expert groups match independent CPU", async ({
  page,
}) => {
  await page.goto(
    "/#model=qwen3-30ba3b&step=decoder&sample=1&tinyLayer=1&familyToken=7",
  );
  const panel = page.getByLabel("MoE MLA 完整微型模型"),
    ref = cpu("qwen3-moe"),
    r = ref.layers[1].moe;
  await expect(panel.getByTestId("family-architecture")).toContainText(
    "L0/L1 均为 MoE",
  );
  await expect(panel.getByTestId("family-loss")).toContainText(
    "CE=" + ref.loss.toFixed(8),
  );
  await expect(panel.getByTestId("family-backbone")).toContainText(
    vector(ref.final_norm[7]),
  );
  const rows = panel.getByTestId("family-router").locator("tbody tr");
  for (let e = 0; e < 4; e++)
    await expect(rows.nth(e).locator("td").last()).toHaveText(
      r.routing[7][e].toFixed(8),
    );
  await panel.getByLabel("Family EP", { exact: true }).selectOption("2");
  await panel.getByLabel("Family ETP", { exact: true }).selectOption("2");
  await expect(panel.getByTestId("family-loss")).toContainText(
    "CE=" + ref.loss.toFixed(8),
  );
  await panel
    .getByText("已验证的 EP / ETP / EDP 参考组", { exact: true })
    .click();
  await expect(
    panel.getByTestId("family-groups").locator("tbody tr"),
  ).toHaveCount(4);
  await expect(
    panel.getByTestId("family-groups").locator("tbody tr").nth(3).locator("td"),
  ).toHaveText(["3", "2,3", "1,3", "2,3", "3", "[2,4)"]);
  await panel
    .getByRole("button", { name: "Dispatch / experts", exact: true })
    .click();
  await panel.getByLabel("Family expert", { exact: true }).selectOption("2");
  await expect(
    panel.getByTestId("family-dispatch").locator("tbody tr"),
  ).toHaveCount(r.dispatch.filter((d: any) => d.expert === 2).length);
  await panel
    .getByRole("button", { name: "Routed + shared", exact: true })
    .click();
  await expect(panel.getByTestId("family-combine")).toContainText(
    vector(r.combined[7]),
  );
  await panel.getByLabel("Family token", { exact: true }).fill("0");
  await expect(panel.getByTestId("family-combine")).toContainText(
    vector(r.combined[0]),
  );
  await page.reload();
  await expect(panel.getByLabel("Family token", { exact: true })).toHaveValue(
    "0",
  );
});

test("[G06] padding, auxiliary counts, bias and dispatch counterexamples remain explicit", async ({
  page,
}) => {
  await page.goto(
    "/#model=deepseek-v3&step=decoder&sample=1&tinyLayer=1&familyToken=7",
  );
  const panel = page.getByLabel("MoE MLA 完整微型模型"),
    ref = cpu("deepseek-v3");
  await panel.getByRole("button", { name: "Bias 更新", exact: true }).click();
  await expect(panel.getByTestId("family-bias")).toContainText(
    "next=[" + vector(ref.layers[1].moe.updated_bias) + "]",
  );
  await panel
    .getByRole("button", { name: "Auxiliary loss", exact: true })
    .click();
  await expect(panel.getByTestId("family-aux")).toContainText(
    ref.layers[1].moe.aux_loss.toFixed(8),
  );
  await panel.getByLabel("Family padding", { exact: true }).selectOption("2");
  await expect(panel.getByTestId("family-loss")).toContainText(
    "CE=" + ref.loss.toFixed(8),
  );
  await panel.getByLabel("Family token", { exact: true }).fill("12");
  await expect(panel.getByTestId("family-routing-scope")).toContainText(
    "是 padding，已排除",
  );
  await panel
    .getByRole("button", { name: "Routed + shared", exact: true })
    .click();
  await expect(panel.getByTestId("family-combine")).toContainText(
    vector(Array(8).fill(0)),
  );
  await panel
    .getByLabel("Family fault", { exact: true })
    .selectOption("missing_shared");
  await expect(panel.getByTestId("family-loss")).not.toContainText(
    "ΔCE=0.00000000",
  );
  await panel
    .getByLabel("Family fault", { exact: true })
    .selectOption("duplicate_dispatch");
  await expect(panel.getByRole("alert")).toContainText("duplicate or missing");
  await expect(
    panel.getByRole("button", { name: "导出 family reference 切片" }),
  ).toBeDisabled();
  await panel.getByLabel("Family fault", { exact: true }).selectOption("none");
  await panel.getByLabel("Family token", { exact: true }).fill("7");
  const download = page.waitForEvent("download");
  await panel
    .getByRole("button", { name: "导出 family reference 切片" })
    .click();
  const payload = JSON.parse(
    fs.readFileSync((await (await download).path())!, "utf8"),
  );
  expect(payload.provenance).toBe("reference");
  expect(payload.runtime_training).toBe("not_run");
  expect(payload.moe.combined[0]).toBeCloseTo(
    ref.layers[1].moe.combined[7][0],
    10,
  );
  expect(payload.decode_cache).toBeNull();
});

test("[G06] V2 direct Q and V3 latent Q preserve separate train and decode cache views", async ({
  page,
}) => {
  await page.goto("/#model=deepseek-v2-lite&step=decoder&sample=1&tinyLayer=0");
  const panel = page.getByLabel("MoE MLA 完整微型模型");
  await expect(panel.getByTestId("family-dense")).toContainText("dense SwiGLU");
  await panel.getByRole("button", { name: "进入 L1 MoE", exact: true }).click();
  await panel
    .getByRole("button", { name: "MLA Q/KV norm", exact: true })
    .click();
  const v2 = cpu("deepseek-v2-lite").mla[1];
  await expect(panel.getByTestId("family-architecture")).toContainText(
    "直接 Q，无 Q latent norm",
  );
  await expect(panel.getByTestId("family-mla-train")).toContainText(
    vector(v2.q_raw[7]),
  );
  await expect(panel.getByTestId("family-mla-train")).toContainText(
    vector(v2.kv_latent[7]),
  );
  await page.locator("#model-select").selectOption("deepseek-v3");
  const v3 = cpu("deepseek-v3").mla[1];
  await expect(panel.getByTestId("family-architecture")).toContainText(
    "Q down 8→4",
  );
  await expect(panel.getByTestId("family-mla-train")).toContainText(
    vector(v3.q_raw[7]),
  );
  await panel
    .getByRole("button", { name: "MLA latent/cache", exact: true })
    .click();
  await expect(panel.getByTestId("family-mla-train")).toHaveCount(0);
  await expect(panel.getByTestId("family-mla-cache")).toContainText(
    vector(v3.cache[7]),
  );
  await expect(panel.getByTestId("family-mla-cache")).toContainText(
    "payload=320 bytes",
  );
  await panel.getByLabel("Family MLA head", { exact: true }).selectOption("1");
  await expect(panel.getByTestId("family-mla-cache")).toContainText(
    vector(v3.cached_context[7][1]),
  );
  const download = page.waitForEvent("download");
  await panel
    .getByRole("button", { name: "导出 family reference 切片" })
    .click();
  const payload = JSON.parse(
    fs.readFileSync((await (await download).path())!, "utf8"),
  );
  expect(payload.training_activation).toBeNull();
  expect(payload.decode_cache.prefix_length).toBe(8);
  expect(payload.decode_cache.row).toEqual(
    v3.cache[7].map((_: any, i: number) => expect.closeTo(v3.cache[7][i], 10)),
  );
  for (const fault of ["wrong_norm", "wrong_scale", "missing_rope"]) {
    await panel.getByLabel("Family fault", { exact: true }).selectOption(fault);
    const text = await panel.getByTestId("family-mla-error").innerText();
    expect(Number(text.match(/吸收误差=([0-9.e+-]+)/)![1])).toBeGreaterThan(
      1e-5,
    );
  }
  await panel.getByLabel("Family fault", { exact: true }).selectOption("none");
  await panel.screenshot({ path: "runs/program-v1/g06-desktop.png" });
});

test("[G06] fixed family source branches, offline math, keyboard and narrow tables work", async ({
  page,
  context,
}) => {
  await page.goto(
    "/#model=deepseek-v2-lite&step=decoder&sample=1&tinyLayer=1&familyOp=mla-cache&mla=decode",
  );
  const panel = page.getByLabel("MoE MLA 完整微型模型");
  await context.setOffline(true);
  await page.locator(".skip-link").focus();
  await expect(page.locator(".skip-link")).toHaveCSS("opacity", "1");
  await page.keyboard.press("Enter");
  await expect(page.locator("#main-content")).toBeFocused();
  await panel
    .getByRole("button", { name: "查看 family 子步骤源码", exact: true })
    .click();
  await expect(page.getByLabel("关键源码片段")).toHaveAttribute(
    "data-snippet-id",
    "mla-cached-latent",
  );
  await expect(page.getByRole("dialog").locator(".source-code")).toContainText(
    "kv_cached = torch.cat",
  );
  await page.keyboard.press("Escape");
  await panel
    .getByRole("button", { name: "查看 V2 直接 Q", exact: true })
    .click();
  await expect(page.getByLabel("关键源码片段")).toHaveAttribute(
    "data-snippet-id",
    "mla-direct-q",
  );
  await expect(page.getByRole("dialog").locator(".source-code")).toContainText(
    "q_compressed = hidden_states",
  );
  await page.keyboard.press("Escape");
  await panel
    .getByRole("button", { name: "查看 family 配置分支", exact: true })
    .click();
  await expect(page.getByLabel("关键源码片段")).toHaveAttribute(
    "data-snippet-id",
    "ds2-provider",
  );
  await page.keyboard.press("Escape");
  const summary = panel.locator(".family-course summary");
  await summary.focus();
  await page.keyboard.press("Enter");
  await expect(panel.locator(".family-course .katex")).not.toHaveCount(0);
  await expect(panel.locator(".family-course")).toContainText("独立数学符号表");
  await expect(page.locator(".katex-error")).toHaveCount(0);
  await summary.focus();
  await page.keyboard.press("Space");
  await expect(panel.locator(".family-course .prose")).toHaveCount(0);
  await page.setViewportSize({ width: 390, height: 844 });
  await panel.screenshot({ path: "runs/program-v1/g06-mobile.png" });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth + 1,
    ),
  ).toBe(true);
  await expect(panel.getByTestId("family-mla-cache")).toBeVisible();
});
