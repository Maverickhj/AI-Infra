import { test, expect } from "@playwright/test";
import { spawnSync } from "node:child_process";
let cpu: any;
test.beforeAll(() => {
  const env = { ...process.env, CUDA_VISIBLE_DEVICES: "" };
  delete env.PYTHONPATH;
  const p = spawnSync(
    "python",
    ["-m", "experiments.decoder_reference", "--forward"],
    { env, encoding: "utf8", maxBuffer: 16 * 1024 * 1024 },
  );
  expect(p.status, p.stderr).toBe(0);
  cpu = JSON.parse(p.stdout)[0].trace;
});
test("[G03] two-layer token slices match CPU and norm/FFN/head sources", async ({
  page,
}) => {
  await page.goto(
    "/#step=decoder&decoderOp=ffn_norm&decoderToken=7&tinyLayer=0",
  );
  const panel = page.getByLabel("完整微型 decoder");
  const ops = panel.getByLabel("微型模型子步骤");
  const output = panel.getByTestId("decoder-values").locator("tbody tr").nth(1);
  for (const [label, key] of [
    ["输入 RMSNorm", "input_norm"],
    ["GQA + residual", "attention"],
    ["FFN RMSNorm", "ffn_norm"],
    ["Gate", "gate"],
    ["Up", "up"],
    ["SiLU", "silu"],
    ["Gate × Up", "product"],
    ["Down", "down"],
    ["FFN residual", "residual"],
  ]) {
    await ops.getByRole("button", { name: label, exact: true }).click();
    await expect(output).toContainText(
      cpu.layers[0][key][7]
        .slice(0, 8)
        .map((x: number) => x.toFixed(8))
        .join(", "),
    );
  }
  await panel.getByLabel("微型层").selectOption("1");
  await expect(output).toContainText(
    cpu.layers[1].residual[7]
      .slice(0, 8)
      .map((x: number) => x.toFixed(8))
      .join(", "),
  );
  await panel.getByLabel("Decoder token", { exact: true }).fill("9");
  await expect(output).toContainText(
    cpu.layers[1].residual[9]
      .slice(0, 8)
      .map((x: number) => x.toFixed(8))
      .join(", "),
  );
  for (const [label, id, code] of [
    ["FFN RMSNorm", "decoder-pre-ffn-norm", "pre_mlp_layernorm"],
    ["Gate × Up", "decoder-swiglu", "torch.chunk(x, 2"],
    ["FFN residual", "decoder-ffn-residual", "mlp_output_with_bias, residual"],
    [
      "Final RMSNorm",
      "decoder-final-norm",
      "apply_module(self.final_layernorm)",
    ],
    ["词表 head", "gpt-output-projection", "self.output_layer"],
  ]) {
    await ops.getByRole("button", { name: label, exact: true }).click();
    await panel.getByRole("button", { name: /查看此子步骤源码/ }).click();
    await expect(page.getByLabel("关键源码片段")).toHaveAttribute(
      "data-snippet-id",
      id,
    );
    await expect(
      page.getByRole("dialog").locator(".source-code"),
    ).toContainText(code);
    await page.keyboard.press("Escape");
  }
  await ops.getByRole("button", { name: "Masked CE", exact: true }).click();
  await expect(panel.getByTestId("decoder-loss")).toContainText(
    "loss=" + cpu.loss.toFixed(8),
  );
  await expect(panel).toContainText("authored");
});

test("[G03] real CPU gradient/update evidence and export remain honestly scoped", async ({
  page,
}) => {
  await page.goto("/#step=backward&decoderTied=untied&sample=1&mode=full");
  const panel = page.getByLabel("完整微型 decoder");
  await expect(panel.getByTestId("decoder-cpu")).toContainText(
    "arithmetic-multiturn / assistant / tied",
  );
  await expect(panel.getByTestId("prompt-gradient")).toContainText(
    "loss mask=0",
  );
  await expect(panel.getByTestId("decoder-resume")).toContainText(
    "最大参数误差=0",
  );
  await expect(panel).toContainText("不是 observed_bridge");
  await panel
    .getByLabel("微型模型子步骤")
    .getByRole("button", { name: "Update / resume" })
    .click();
  await expect(panel).toContainText("冻结参数不变：true");
  const pending = page.waitForEvent("download");
  await panel
    .getByRole("button", { name: "导出 decoder 切片与 CPU 证据" })
    .click();
  const d = await pending,
    stream = await d.createReadStream(),
    parts: Buffer[] = [];
  for await (const chunk of stream!) parts.push(chunk);
  const result = JSON.parse(Buffer.concat(parts).toString());
  expect(result.slice).toBeNull();
  expect(result.cpu_evidence.device).toBe("cpu");
  expect(Object.keys(result.cpu_evidence.source_hashes)).toContain(
    "experiments/gqa_reference.py",
  );
  expect(
    result.cpu_evidence.selected_parameters.every(
      (p: any) =>
        Number.isFinite(p.gradient) && Number.isFinite(p.finite_difference),
    ),
  ).toBe(true);
  expect(result.cpu_evidence.resume.continuous_next_loss).toBe(
    result.cpu_evidence.resume.resumed_next_loss,
  );
});

test("[G03] deep links, keyboard, offline math and small screen", async ({
  page,
  context,
}) => {
  await page.goto(
    "/#step=decoder&decoderOp=product&tinyLayer=1&decoderToken=8&decoderTied=untied",
  );
  const panel = page.getByLabel("完整微型 decoder");
  await page.reload();
  await expect(panel.getByLabel("微型层")).toHaveValue("1");
  await expect(panel.getByLabel("词表权重")).toHaveValue("untied");
  await expect(panel.getByTestId("decoder-op")).toContainText("Gate × Up");
  await context.setOffline(true);
  await panel
    .getByText("精讲：norm、SwiGLU、共享权重与更新 · 符号表", { exact: true })
    .click();
  await expect(panel.locator(".katex-error")).toHaveCount(0);
  expect(await panel.locator(".katex-display").count()).toBeGreaterThanOrEqual(
    8,
  );
  await page.setViewportSize({ width: 390, height: 844 });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  const final = panel.getByRole("button", {
    name: "Final RMSNorm",
    exact: true,
  });
  await final.focus();
  await page.keyboard.press("Enter");
  await expect(panel.getByTestId("decoder-op")).toContainText("Final RMSNorm");
  await panel.screenshot({ path: "runs/screenshots/g03-mobile.png" });
  await page.setViewportSize({ width: 1440, height: 1000 });
  await panel.screenshot({ path: "runs/screenshots/g03-desktop.png" });
  await context.setOffline(false);
  await page.goto(
    "/#step=decoder&decoderOp=bad&decoderToken=-1&tinyLayer=99&decoderTied=bad",
  );
  await expect(panel.getByLabel("微型层")).toHaveValue("0");
  await expect(panel.getByLabel("词表权重")).toHaveValue("tied");
  await expect(panel.getByLabel("Decoder token", { exact: true })).toHaveValue(
    "7",
  );
});
