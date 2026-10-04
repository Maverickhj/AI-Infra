import { test, expect } from "@playwright/test";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
let references: any[];
const fixture = JSON.parse(
  fs.readFileSync(
    new URL("../../content/fixtures/rl-reference.json", import.meta.url),
    "utf8",
  ),
);
const vocabulary = JSON.parse(
  fs.readFileSync(
    new URL("../../content/fixtures/sft-data.json", import.meta.url),
    "utf8",
  ),
).vocabulary;
test.beforeAll(() => {
  const env = { ...process.env, CUDA_VISIBLE_DEVICES: "" };
  delete env.PYTHONPATH;
  const p = spawnSync(
    "python",
    ["-m", "experiments.rl_reference", "--forward"],
    { env, encoding: "utf8", maxBuffer: 32 * 1024 * 1024 },
  );
  expect(p.status, p.stderr).toBe(0);
  references = JSON.parse(p.stdout);
});
const cpu = (
  algorithm = "grpo",
  reduction = "token",
  kl = true,
  force = false,
) =>
  references.find(
    (r) =>
      r.algorithm === algorithm &&
      r.reduction === reduction &&
      r.kl_enabled === kl &&
      r.force_on_policy === force,
  );
const fmt = (v: number) => v.toFixed(8),
  vector = (v: number[]) => v.map(fmt).join(", ");
async function tokensMatch(panel: any, r: any, i: number) {
  const rows = panel.getByTestId("rl-token-logprobs").locator("tbody tr");
  await expect(rows).toHaveCount(r.ids[i].length);
  for (let t = 0; t < r.ids[i].length; t++) {
    await expect(rows.nth(t).locator("td")).toHaveText([
      t + " · " + vocabulary[r.ids[i][t]],
      String(r.mask[i][t]),
      ...[
        r.generation_logprobs[i][t],
        r.previous_logprobs[i][t],
        r.current_logprobs[i][t],
        r.reference_logprobs[i][t],
        r.advantages[i][t],
        r.terms.ratio[i][t],
        r.terms.clipped_ratio[i][t],
        r.terms.pg_token[i][t],
        r.logprob_gradient[i][t],
      ].map(fmt),
    ]);
  }
}
test("[G07] authored trajectories, four action-aligned logprobs and GRPO groups match CPU", async ({
  page,
}) => {
  await page.goto("/#scenario=rl&step=rollout&rlKl=off");
  const panel = page.getByLabel("RL 数值参考闭环"),
    r = cpu("grpo", "token", false);
  await expect(panel.getByTestId("rl-scope")).toContainText("非真实 rollout");
  await expect(
    panel.getByTestId("rl-trajectories").locator("tbody tr"),
  ).toHaveCount(4);
  await expect(panel.getByTestId("rl-trajectories")).toContainText("1,0,0,1");
  await tokensMatch(panel, r, 0);
  await panel.getByLabel("RL trajectory", { exact: true }).selectOption("3");
  await tokensMatch(panel, r, 3);
  const groups = panel.getByTestId("rl-group").locator("tbody tr");
  for (let i = 0; i < 4; i++)
    await expect(groups.nth(i).locator("td")).toHaveText([
      fixture.trajectories[i].id,
      ...[
        r.rewards[i],
        r.group.baseline[i],
        r.group.std[i],
        r.group.sequence_advantages[i],
      ].map(fmt),
    ]);
  await expect(panel.getByTestId("rl-critic-update")).toHaveCount(0);
  await panel.getByLabel("RL token", { exact: true }).fill("0");
  await page.reload();
  await expect(panel.getByLabel("RL token", { exact: true })).toHaveValue("0");
  await expect(panel.getByTestId("rl-selection")).toContainText(
    "prefix 结尾=dummy",
  );
});
test("[G07] PPO GAE, masked carry, returns and separate critic update match CPU", async ({
  page,
}) => {
  await page.goto(
    "/#scenario=rl&step=advantage&rlAlgorithm=ppo&rlTrajectory=3&rlToken=12",
  );
  const panel = page.getByLabel("RL 数值参考闭环"),
    r = cpu("ppo");
  await expect(panel.getByTestId("rl-value-loss")).toContainText(
    fmt(r.value.value_loss),
  );
  const rows = panel.getByTestId("rl-gae").locator("tbody tr");
  for (let t = 0; t < 13; t++)
    await expect(rows.nth(t).locator("td")).toHaveText([
      String(t),
      String(r.mask[3][t]),
      ...[
        r.gae.token_rewards[3][t],
        r.value.old_values[3][t],
        r.gae.delta[3][t],
        r.advantages[3][t],
        r.value.returns[3][t],
        r.value.values[3][t],
        r.value.clipped_values[3][t],
        r.value.value_token[3][t],
      ].map(fmt),
    ]);
  await panel.locator(".rl-head summary").click();
  await expect(panel.getByTestId("rl-critic-update")).toContainText(
    "critic gradient=[" + vector(r.critic_gradient) + "]",
  );
  await expect(panel.getByTestId("rl-critic-update")).toContainText(
    "critic after=[" + vector(r.critic_after) + "]",
  );
  await expect(
    panel
      .getByTestId("rl-head-update")
      .locator("tbody tr")
      .first()
      .locator("td"),
  ).toHaveText([
    "0",
    fmt(fixture.heads.current[4][0]),
    fmt(r.head_gradient[4][0]),
    fmt(r.head_after[4][0]),
  ]);
  await panel
    .getByLabel("RL reduction", { exact: true })
    .selectOption("sequence");
  const seq = cpu("ppo", "sequence");
  await expect(panel.getByTestId("rl-losses")).toContainText(
    "policy=" + fmt(seq.terms.policy_loss),
  );
  await expect(panel.getByTestId("rl-value-loss")).toContainText(
    fmt(r.value.value_loss),
  );
  await panel.getByLabel("RL KL", { exact: true }).selectOption("off");
  await expect(panel.getByTestId("rl-losses")).toContainText(
    "policy=" + fmt(cpu("ppo", "sequence", false).terms.policy_loss),
  );
  await panel
    .getByRole("button", { name: "查看 RL 当前子步骤源码", exact: true })
    .click();
  await expect(page.getByLabel("关键源码片段")).toHaveAttribute(
    "data-snippet-id",
    "rl-gae-carry",
  );
  await page.keyboard.press("Escape");
  await panel
    .getByRole("button", { name: "查看 value loss 源码", exact: true })
    .click();
  await expect(page.getByRole("dialog").locator(".source-code")).toContainText(
    "vf_losses_clipped",
  );
  await page.keyboard.press("Escape");
  await panel.screenshot({ path: "runs/program-v1/g07-desktop.png" });
});
test("[G07] force ratio preserves gradient, equal rewards zero PG and wrong gradients are exposed", async ({
  page,
}) => {
  await page.goto("/#scenario=rl&step=policy_update&rlForce=on&rlKl=off");
  const panel = page.getByLabel("RL 数值参考闭环"),
    r = cpu("grpo", "token", false, true);
  await tokensMatch(panel, r, 0);
  await expect(panel.getByTestId("rl-losses")).toContainText(
    "PG=" + fmt(r.terms.actor_loss),
  );
  await panel
    .getByLabel("RL fault", { exact: true })
    .selectOption("constant_ratio");
  await expect(panel.getByTestId("rl-losses")).toContainText(
    "head gradient norm=0.00000000",
  );
  await expect(
    panel.getByRole("button", { name: "导出 RL reference 切片" }),
  ).toBeDisabled();
  await expect(
    panel.getByRole("button", { name: "应用一次参考 SGD" }),
  ).toBeDisabled();
  const difference = await panel.getByTestId("rl-counterexample").innerText();
  expect(Number(difference.match(/最大差=([0-9.]+)/)![1])).toBeGreaterThan(
    1e-5,
  );
  await panel.getByLabel("RL fault", { exact: true }).selectOption("none");
  await panel
    .getByLabel("RL equal rewards", { exact: true })
    .selectOption("on");
  await expect(panel.getByTestId("rl-losses")).toContainText("PG=0.00000000");
  await expect(panel.getByTestId("rl-losses")).toContainText(
    "head gradient norm=0.00000000",
  );
  await panel
    .getByLabel("RL equal rewards", { exact: true })
    .selectOption("off");
  await panel.getByLabel("RL force", { exact: true }).selectOption("off");
  await panel
    .getByLabel("RL fault", { exact: true })
    .selectOption("wrong_clip");
  expect(
    Number(
      (await panel.getByTestId("rl-counterexample").innerText()).match(
        /最大差=([0-9.]+)/,
      )![1],
    ),
  ).toBeGreaterThan(1e-5);
  await panel
    .getByLabel("RL fault", { exact: true })
    .selectOption("detach_kl_weight");
  expect(
    Number(
      (await panel.getByTestId("rl-counterexample").innerText()).match(
        /最大差=([0-9.]+)/,
      )![1],
    ),
  ).toBeGreaterThan(1e-5);
});
test("[G07] one update exports weights and refit rejects bad version/hash/ack before real reference copy", async ({
  page,
}) => {
  await page.goto("/#scenario=rl&step=policy_update&rlAlgorithm=ppo");
  const panel = page.getByLabel("RL 数值参考闭环"),
    r = cpu("ppo");
  await expect(
    panel.getByRole("button", { name: "执行参考 refit" }),
  ).toBeDisabled();
  await panel.getByRole("button", { name: "应用一次参考 SGD" }).click();
  await expect(panel.getByTestId("rl-versions")).toContainText(
    "policy=2 · generation=0",
  );
  await expect(
    panel.getByRole("button", { name: "应用一次参考 SGD" }),
  ).toBeDisabled();
  await page
    .getByLabel("完整计算流程")
    .getByRole("button", { name: /Export \/ refit/ })
    .click();
  const dl = page.waitForEvent("download");
  await panel.getByRole("button", { name: "导出 policy 快照" }).click();
  const payload = JSON.parse(
    fs.readFileSync((await (await dl).path())!, "utf8"),
  );
  expect(payload.provenance).toBe("reference");
  expect(payload.policy_version).toBe(2);
  expect(payload.runtime_training).toBe("not_run");
  for (let v = 0; v < 27; v++)
    for (let h = 0; h < 8; h++)
      expect(payload.weights[v][h]).toBeCloseTo(r.head_after[v][h], 10);
  for (const [fault, error] of [
    ["version", "policy/export version mismatch"],
    ["hash", "export weight hash mismatch"],
    ["ack", "refit not completed"],
  ]) {
    await panel
      .getByLabel("RL refit fault", { exact: true })
      .selectOption(fault);
    await panel.getByRole("button", { name: "执行参考 refit" }).click();
    await expect(panel.getByRole("alert")).toContainText(error);
    await expect(panel.getByTestId("rl-versions")).toContainText(
      "generation=0",
    );
    await expect(panel.getByTestId("rl-refit-status")).not.toContainText(
      "reference_contract_synchronized",
    );
  }
  await panel
    .getByLabel("RL refit fault", { exact: true })
    .selectOption("none");
  await panel.getByRole("button", { name: "执行参考 refit" }).click();
  await expect(panel.getByTestId("rl-refit-status")).toContainText(
    "reference_contract_synchronized",
  );
  await expect(panel.getByTestId("rl-versions")).toContainText(
    "policy=2 · generation=2",
  );
  await expect(panel.getByTestId("rl-refit-hash")).toContainText(
    "ack hash=" + payload.weights_sha256,
  );
  await expect(panel.getByTestId("rl-refit-error")).toContainText(
    "最大差=0.00000000",
  );
  for (let i = 0; i < 4; i++)
    await expect(
      panel
        .getByTestId("rl-generation-lp")
        .locator("tbody tr")
        .nth(i)
        .locator("td")
        .last(),
    ).toHaveText(
      vector(
        r.logprobs_after[i].filter((_: number, t: number) => r.mask[i][t]),
      ),
    );
  await panel.getByLabel("RL algorithm", { exact: true }).selectOption("grpo");
  await expect(panel.getByTestId("rl-versions")).toContainText("generation=0");
  await expect(
    panel.getByRole("button", { name: "执行参考 refit" }),
  ).toBeDisabled();
});
test("[G07] selected reference export retains masks, provenance, gradients and before/after versions", async ({
  page,
}) => {
  await page.goto(
    "/#scenario=rl&step=logprobs&rlAlgorithm=ppo&rlReduction=sequence&rlTrajectory=3&rlToken=12",
  );
  const panel = page.getByLabel("RL 数值参考闭环"),
    r = cpu("ppo", "sequence");
  const dl = page.waitForEvent("download");
  await panel.getByRole("button", { name: "导出 RL reference 切片" }).click();
  const p = JSON.parse(fs.readFileSync((await (await dl).path())!, "utf8"));
  expect(p.provenance).toBe("reference");
  expect(p.actual_rollout).toBe("not_run");
  expect(p.algorithm).toBe("ppo");
  expect(p.versions).toEqual({
    generation: 0,
    previous: 0,
    current: 1,
    next: 2,
  });
  expect(p.mask).toEqual(r.mask[3]);
  expect(p.losses.policy).toBeCloseTo(r.terms.policy_loss, 10);
  expect(p.losses.value).toBeCloseTo(r.value.value_loss, 10);
  for (let t = 0; t < 13; t++) {
    expect(p.logprob_gradient[t]).toBeCloseTo(r.logprob_gradient[3][t], 10);
    expect(p.advantages[t]).toBeCloseTo(r.advantages[3][t], 10);
  }
  expect(p.head_row.action).toBe(4);
  expect(p.head_row.after[0]).toBeCloseTo(r.head_after[4][0], 10);
  await page.goto(
    "/#scenario=rl&step=advantage&rlAlgorithm=invalid&rlTrajectory=99&rlToken=-1&rlReduction=nope",
  );
  await expect(panel.getByLabel("RL algorithm", { exact: true })).toHaveValue(
    "grpo",
  );
  await expect(panel.getByLabel("RL trajectory", { exact: true })).toHaveValue(
    "0",
  );
  await expect(panel.getByLabel("RL token", { exact: true })).toHaveValue("8");
});
test("[G07] offline sources, full mathematics, keyboard and narrow layout remain usable", async ({
  page,
  context,
}) => {
  await page.goto("/#scenario=rl&step=advantage&rlAlgorithm=ppo");
  const panel = page.getByLabel("RL 数值参考闭环");
  await expect(panel).toBeVisible();
  await context.setOffline(true);
  for (const [button, id, code] of [
    ["查看 RL 当前子步骤源码", "rl-gae-carry", "last_gae_lam"],
    ["查看组标准差源码", "rl-group-std", "num_valid / (num_valid - 1)"],
    [
      "查看 KL 梯度源码",
      "rl-kl-score-gradient",
      "curr_logprobs_unfiltered.detach()",
    ],
  ]) {
    await panel.getByRole("button", { name: button, exact: true }).click();
    await expect(page.getByLabel("关键源码片段")).toHaveAttribute(
      "data-snippet-id",
      id,
    );
    await expect(
      page.getByRole("dialog").locator(".source-code"),
    ).toContainText(code);
    await page.keyboard.press("Escape");
  }
  await panel.locator(".rl-course summary").focus();
  await page.keyboard.press("Enter");
  await expect(panel.locator(".rl-course")).toContainText("独立数学符号表");
  await expect(panel.locator(".rl-course .katex")).not.toHaveCount(0);
  await expect(page.locator(".katex-error")).toHaveCount(0);
  await expect(panel.locator(".rl-course")).toContainText("Bessel");
  await expect(panel.locator(".rl-course")).toContainText("有限差分");
  await panel.locator(".rl-course summary").focus();
  await page.keyboard.press("Space");
  await page.setViewportSize({ width: 390, height: 844 });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  expect(
    await panel
      .getByTestId("rl-token-logprobs")
      .locator("..")
      .evaluate((e) => e.scrollWidth > e.clientWidth),
  ).toBe(true);
  await panel.getByLabel("RL algorithm", { exact: true }).focus();
  await page.keyboard.press("g");
  await page.keyboard.press("Enter");
  await expect(panel.getByLabel("RL algorithm", { exact: true })).toHaveValue(
    "grpo",
  );
  await expect(
    panel.getByTestId("rl-token-logprobs").locator("td").nth(2),
  ).toHaveCSS("white-space", "nowrap");
  await panel.screenshot({ path: "runs/program-v1/g07-mobile.png" });
});
