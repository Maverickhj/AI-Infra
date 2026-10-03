import { test, expect } from "@playwright/test";
import { readFileSync } from "node:fs";
const fixture = JSON.parse(
  readFileSync(
    new URL("../../content/fixtures/gqa-reference.json", import.meta.url),
    "utf8",
  ),
);
import { compute } from "../../web/gqa/compute";

const route =
  "/#model=qwen3-06b&scenario=sft&step=decoder&operator=scores&query=2&gqaHead=3&layer=4";

test("[G01-GQA] token/head/substep arrays follow the shared computation and model branch", async ({
  page,
}) => {
  await page.goto(route);
  const unit = page.getByLabel("GQA 精讲", { exact: true });
  await expect(unit.getByTestId("gqa-selection")).toHaveText(
    "query 2 · Q head 3 → KV group 1",
  );
  await expect(unit.getByTestId("gqa-real-config")).toContainText("Q 宽=2048");
  await expect(unit).toContainText("Layer 4 / 27");
  const expected = compute(fixture);
  const routes: [string, number[][]][] = [
    ["Attention 输入", [expected.input[2]]],
    ["RMSNorm", [expected.norm[2]]],
    [
      "QKV / grouped split",
      [expected.q[2][3], expected.k[2][1], expected.v[2][1]],
    ],
    ["Q/K head norm", [expected.qnorm[2][3], expected.knorm[2][1]]],
    ["RoPE", [expected.qrope[2][3], expected.krope[2][1]]],
    ["QK 打分", [expected.scores[2][3]]],
    ["缩放", [expected.scaled[2][3]]],
    ["Causal mask", [expected.masked[2][3]]],
    ["Softmax", [expected.probabilities[2][3]]],
    ["加权 V", [expected.heads[2][3]]],
    ["合并 heads", [expected.merged[2]]],
    ["输出投影", [expected.projected[2]]],
    ["Residual add", [expected.residual[2]]],
  ];
  for (const [i, [title, rows]] of routes.entries()) {
    await unit
      .getByRole("button", { name: `${i + 1}. ${title}`, exact: true })
      .click();
    await expect(unit.getByTestId("gqa-values").locator("td")).toHaveText(
      rows.flat().map((v) => (v === -Infinity ? "−∞" : v.toFixed(6))),
    );
    await expect(unit.getByLabel("GQA 当前子步骤")).toContainText("验证：");
  }
  await unit.getByLabel("教学 Q head", { exact: true }).fill("0");
  await unit.getByLabel("教学 query token", { exact: true }).fill("0");
  await unit.getByRole("button", { name: "10. 加权 V", exact: true }).click();
  await expect(unit.getByTestId("gqa-values").locator("td")).toHaveText(
    expected.v[0][0].map((v) => v.toFixed(6)),
  );
  await expect(
    unit.getByTestId("gqa-probabilities").locator('td[data-masked="true"]'),
  ).toHaveText(Array(6).fill("0.000000"));
  for (const value of ["3.5", "-1", "4", "", "1e309"]) {
    await unit.getByLabel("教学 Q head", { exact: true }).fill(value);
    await expect(unit.getByRole("alert")).toContainText("保留上一个合法选择");
    await expect(unit.getByTestId("gqa-selection")).toHaveText(
      "query 0 · Q head 0 → KV group 0",
    );
  }
  for (const value of ["2.5", "-1", "4", ""]) {
    await unit.getByLabel("教学 query token", { exact: true }).fill(value);
    await expect(unit.getByRole("alert")).toContainText("保留上一个合法选择");
    await expect(unit.getByTestId("gqa-selection")).toHaveText(
      "query 0 · Q head 0 → KV group 0",
    );
  }
  await page.getByLabel("当前模型", { exact: true }).selectOption("qwen25-05b");
  await unit
    .getByRole("button", { name: "4. Q/K head norm", exact: true })
    .click();
  const q25 = compute(fixture, "qwen25");
  await expect(unit.getByTestId("gqa-values").locator("td")).toHaveText(
    [...q25.q[0][0], ...q25.k[0][0]].map((v) => v.toFixed(6)),
  );
  await expect(unit.getByTestId("gqa-real-config")).toContainText(
    "QKV bias 开启，QK norm 关闭",
  );
  await unit.getByLabel("教学 query token", { exact: true }).fill("2");
  for (const fault of ["omit_scale", "wrong_group"]) {
    await unit.getByLabel("GQA 反例", { exact: true }).selectOption(fault);
    await expect(unit.getByTestId("gqa-counterexample")).not.toContainText(
      "0.00000000000",
    );
    await expect(
      unit
        .getByTestId("gqa-counterexample-values")
        .locator("tr")
        .last()
        .locator("td"),
    ).not.toHaveText(q25.residual[2].map((v) => v.toFixed(6)));
  }
});

test("[G01-SOURCE] every substep maps to a pinned excerpt and switches without stale defaults", async ({
  page,
}) => {
  await page.goto(route);
  const unit = page.getByLabel("GQA 精讲", { exact: true });
  const cases = [
    [
      "1. Attention 输入",
      "gqa-input-norm",
      "L616–L637",
      "residual = hidden_states",
    ],
    ["2. RMSNorm", "gqa-input-norm", "L616–L637", "self.input_layernorm"],
    ["3. QKV / grouped split", "c-attn-l1887", "L1887–L1906", "split"],
    ["4. Q/K head norm", "c-attn-l1920", "L1920–L1924", "self.q_layernorm"],
    ["5. RoPE", "gqa-rope-layout", "L82–L89", "torch.chunk"],
    ...[
      "6. QK 打分",
      "7. 缩放",
      "8. Causal mask",
      "9. Softmax",
      "10. 加权 V",
    ].map((name) => [
      name,
      "gqa-core-boundary",
      "L1555–L1567",
      "self.core_attention",
    ]),
    [
      "11. 合并 heads",
      "gqa-output-projection",
      "L1613–L1621",
      "self.linear_proj",
    ],
    [
      "12. 输出投影",
      "gqa-output-projection",
      "L1613–L1621",
      "self.linear_proj",
    ],
    [
      "13. Residual add",
      "gqa-residual",
      "L676–L687",
      "attention_output_with_bias, residual",
    ],
  ];
  for (const [title, id, range, code] of cases) {
    await unit.getByRole("button", { name: title, exact: true }).click();
    await unit.getByRole("button", { name: "打开 GQA 子步骤源码 ↗" }).click();
    await expect(page.getByLabel("关键源码片段")).toHaveAttribute(
      "data-snippet-id",
      id,
    );
    await expect(page.getByTestId("source-range")).toHaveText(range);
    await expect(page.locator(".source-code")).toContainText(code);
    await expect(page.getByTestId("source-relationship")).toContainText(
      "query 2 · head 3",
    );
    await expect(
      page.getByRole("link", { name: "在完整文件中定位此片段 ↗" }),
    ).toHaveAttribute("href", /a0f793dfa4e776d99a8aa63bed74e0fafc91b0db/);
    await page.keyboard.press("Escape");
  }
  await unit.getByRole("button", { name: "5. RoPE", exact: true }).click();
  await unit.getByRole("button", { name: "查看 RoPE 调用边界 ↗" }).click();
  await expect(page.getByLabel("关键源码片段")).toHaveAttribute(
    "data-snippet-id",
    "gqa-rope-call",
  );
  await page.evaluate(() => {
    location.hash =
      "model=qwen3-06b&step=decoder&operator=qknorm&query=1&gqaHead=0&layer=3";
  });
  await expect(page.getByLabel("关键源码片段")).toHaveAttribute(
    "data-snippet-id",
    "c-attn-l1920",
  );
  await expect(page.getByTestId("source-relationship")).toContainText(
    "Layer 3 · query 1 · head 0",
  );
  await page.keyboard.press("Escape");
  await page.getByLabel("当前模型", { exact: true }).selectOption("qwen25-05b");
  await unit.getByRole("button", { name: "打开 GQA 子步骤源码 ↗" }).click();
  await expect(page.getByLabel("关键源码片段")).toHaveAttribute(
    "data-snippet-id",
    "b-q2-l46",
  );
  await expect(page.locator(".source-code")).toContainText(
    "provider.add_qkv_bias = True",
  );
  await page.keyboard.press("Escape");
  await unit.getByRole("button", { name: "查看 QK norm 条件实现 ↗" }).click();
  await expect(page.getByLabel("关键源码片段")).toHaveAttribute(
    "data-snippet-id",
    "c-attn-l1920",
  );
  await expect(page.locator(".source-code")).toContainText(
    "if self.q_layernorm is not None:",
  );
});

test("[G01-REGRESSION] deep links, reference export, course and narrow screen remain usable", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto(route);
  await page.reload();
  const unit = page.getByLabel("GQA 精讲", { exact: true });
  await expect(
    unit.getByLabel("教学 query token", { exact: true }),
  ).toHaveValue("2");
  await expect(unit.getByLabel("教学 Q head", { exact: true })).toHaveValue(
    "3",
  );
  await unit.getByRole("button", { name: "GQA 下一步 →" }).focus();
  await page.keyboard.press("Enter");
  await expect(unit.getByLabel("GQA 当前子步骤")).toHaveAttribute(
    "data-substep",
    "scale",
  );
  await unit
    .getByText("完整 GQA 精讲与独立数学符号表", { exact: true })
    .click();
  await expect(unit.locator(".gqa-course")).toContainText("独立数学符号表");
  await expect(
    unit.locator(".gqa-course .katex-display").first(),
  ).toBeVisible();
  await expect(page.locator(".katex-error")).toHaveCount(0);
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  const downloadPromise = page.waitForEvent("download");
  await unit.getByRole("button", { name: "导出 reference 演算" }).click();
  const download = await downloadPromise;
  const stream = await download.createReadStream();
  const chunks: Buffer[] = [];
  for await (const chunk of stream!) chunks.push(chunk);
  const exported = JSON.parse(Buffer.concat(chunks).toString());
  expect(exported.architecture_origin).toBe("authored_scaled_gqa");
  expect(exported.weights_origin).toBe("authored_fixture");
  expect(exported.training_execution).toBe("not_run");
  expect(exported.fixture).toEqual(fixture);
  expect(exported.result.residual).toEqual(compute(fixture).residual);
  await unit.screenshot({ path: "runs/screenshots/g01-mobile.png" });
  await page.setViewportSize({ width: 1440, height: 1000 });
  await unit
    .getByText("完整 GQA 精讲与独立数学符号表", { exact: true })
    .click();
  await unit.scrollIntoViewIfNeeded();
  await page.evaluate(() =>
    document.querySelector(".gqa-unit")!.scrollIntoView({ block: "start" }),
  );
  await page.screenshot({ path: "runs/screenshots/g01-desktop.png" });
  await page
    .getByLabel("当前模型", { exact: true })
    .selectOption("deepseek-v3");
  await expect(unit).toHaveCount(0);
  await expect(page.getByTestId("mla-graph")).toBeVisible();
});
