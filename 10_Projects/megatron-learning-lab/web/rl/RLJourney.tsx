import { memo, useMemo, useState } from "react";
import { CourseDetails } from "../CourseText";
import type { State } from "../data";
import base from "../../content/fixtures/decoder-reference.json";
import fixture from "../../content/fixtures/rl-reference.json";
import tokens from "../../content/fixtures/sft-data.json";
import course from "../../content/cases/10_rl_reference.md?raw";
import { computeRL, policyLogprobs, refitContract, stateHash } from "./compute";
type Trace = ReturnType<typeof computeRL>;
const fmt = (v: number) => v.toFixed(8);
const vector = (v: number[]) => v.map(fmt).join(", ");
function download(value: unknown, name: string) {
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(value, null, 2)], { type: "application/json" }),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}
function VersionDemo({
  trace,
  visible,
  disabled,
  onSource,
}: {
  trace: Trace;
  visible: boolean;
  disabled: boolean;
  onSource: (id: string, excerpt?: string) => void;
}) {
  const initial = trace.force_on_policy
    ? fixture.heads.previous
    : fixture.heads.current;
  const [policy, setPolicy] = useState(initial.map((r) => r.slice()));
  const [version, setVersion] = useState(trace.current_version);
  const [generation, setGeneration] = useState(
    fixture.heads.generation.map((r) => r.slice()),
  );
  const [generationVersion, setGenerationVersion] = useState(0);
  const [applied, setApplied] = useState(false);
  const [snapshot, setSnapshot] = useState<{
    version: number;
    weights: number[][];
    hash: string;
  } | null>(null);
  const [fault, setFault] = useState("none");
  const [message, setMessage] = useState(
    "reference generation version 0；真实 rollout / refit not_run",
  );
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [ack, setAck] = useState<string | null>(null);
  const lp = useMemo(
    () => policyLogprobs(trace.features, generation, trace.ids).logprobs,
    [trace, generation],
  );
  const mismatch = Math.max(
    ...lp.flat().map((v, i) => Math.abs(v - trace.logprobs_after.flat()[i])),
  );
  async function exportPolicy() {
    setBusy(true);
    setError("");
    try {
      const weights = policy.map((r) => r.slice()),
        hash = await stateHash(weights);
      const next = { version, weights, hash };
      setSnapshot(next);
      download(
        {
          schema_version: 1,
          provenance: "reference",
          policy_version: version,
          weights_sha256: hash,
          weights,
          encoding: "f64le:27x8",
          actions_origin: "authored fixed trajectories",
          runtime_training: "not_run",
        },
        "rl-policy-reference.json",
      );
      setMessage("参考 policy 快照已导出；generation 仍需完成 refit。");
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }
  async function refit() {
    if (!snapshot) return;
    setBusy(true);
    setError("");
    setAck(null);
    try {
      const candidate = snapshot.weights.map((r) => r.slice());
      const candidateHash = await stateHash(candidate);
      const result = await refitContract(
        version,
        policy,
        fault === "version" ? snapshot.version - 1 : snapshot.version,
        fault === "hash" ? "0".repeat(64) : snapshot.hash,
        snapshot.version,
        candidateHash,
        fault !== "ack",
      );
      setGeneration(candidate);
      setGenerationVersion(snapshot.version);
      setAck(result.weights_sha256);
      setMessage(
        "reference_contract_synchronized；同一固定 action 上重算，不代表新 rollout。",
      );
    } catch (e) {
      setError(String(e));
      setMessage("拒绝本次同步；generation 快照保持原值。");
    } finally {
      setBusy(false);
    }
  }
  return (
    <section
      className="rl-version"
      aria-label="参考更新与 refit"
      hidden={!visible}
    >
      <h3>一次参考 SGD → export → refit</h3>
      <p>
        每批只应用一次。切换算法或计算配置会重置本流程；选择 token
        不重置。导出的都是 authored 参数。
      </p>
      <div className="rl-actions">
        <button
          disabled={applied || disabled || busy}
          onClick={() => {
            setPolicy(trace.head_after.map((r) => r.slice()));
            setVersion(trace.next_version);
            setApplied(true);
            setSnapshot(null);
            setAck(null);
            setError("");
            setMessage(
              "policy 已应用一次参考 SGD；generation 仍为旧快照，尚未同步。",
            );
          }}
        >
          应用一次参考 SGD
        </button>
        <button disabled={!applied || disabled || busy} onClick={exportPolicy}>
          导出 policy 快照
        </button>
        <label>
          Refit 反例
          <select
            aria-label="RL refit fault"
            value={fault}
            onChange={(e) => setFault(e.target.value)}
          >
            <option value="none">正确版本与完成 ack</option>
            <option value="version">错误 export version</option>
            <option value="hash">错误 weight hash</option>
            <option value="ack">refit 未完成</option>
          </select>
        </label>
        <button disabled={!snapshot || disabled || busy} onClick={refit}>
          执行参考 refit
        </button>
        <button onClick={() => onSource("R-GRPO", "rl-refit-ack")}>
          查看 refit 完成检查源码
        </button>
      </div>
      <p data-testid="rl-versions">
        policy={version} · generation={generationVersion} · exported=
        {snapshot?.version ?? "none"} · applied={String(applied)}
      </p>
      <p role="status" data-testid="rl-refit-status">
        {message}
      </p>
      {error && <p role="alert">{error}</p>}
      <p className="small" data-testid="rl-refit-hash">
        export hash={snapshot?.hash ?? "未导出"}
        <br />
        ack hash={ack ?? "未确认"}
      </p>
      <p data-testid="rl-refit-error">
        generation 对更新后 policy 的固定 action logprob 最大差={fmt(mismatch)}
      </p>
      <div className="table-wrap">
        <table data-testid="rl-generation-lp">
          <caption>
            当前 in-memory generation 重新前向；仅有效 response 位置
          </caption>
          <thead>
            <tr>
              <th>trajectory</th>
              <th>action 位置</th>
              <th>generation logprobs</th>
            </tr>
          </thead>
          <tbody>
            {lp.map((row, i) => (
              <tr key={i}>
                <td>{fixture.trajectories[i].id}</td>
                <td>
                  {trace.mask[i].flatMap((m, j) => (m ? [j] : [])).join(", ")}
                </td>
                <td>{vector(row.filter((_, j) => trace.mask[i][j]))}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
export const RLJourney = memo(function RLJourney({
  state,
  patch,
  onSource,
}: {
  state: State;
  patch: (p: Partial<State>) => void;
  onSource: (id: string, excerpt?: string) => void;
}) {
  const algorithm = state.rlAlgorithm,
    reduction = state.rlReduction,
    kl = state.rlKl === "on",
    force = state.rlForce === "on",
    equal = state.rlEqual === "on",
    fault = state.rlFault;
  const normal = useMemo(
    () => computeRL(base, fixture, algorithm, reduction, kl, force, equal),
    [algorithm, reduction, kl, force, equal],
  );
  const trace = useMemo(
    () =>
      fault === "none"
        ? normal
        : computeRL(
            base,
            fixture,
            algorithm,
            reduction,
            kl,
            force,
            equal,
            fault,
          ),
    [normal, algorithm, reduction, kl, force, equal, fault],
  );
  const i = state.rlTrajectory,
    j = Math.min(state.rlToken, trace.ids[i].length - 1),
    item = fixture.trajectories[i],
    action = trace.ids[i][j];
  const before = force ? fixture.heads.previous : fixture.heads.current;
  const gradientDelta = Math.max(
    ...trace.head_gradient
      .flat()
      .map((v, k) => Math.abs(v - normal.head_gradient.flat()[k])),
  );
  const gradientNorm = Math.sqrt(
    trace.head_gradient.flat().reduce((s, v) => s + v * v, 0),
  );
  const source: Record<string, [string, string]> = {
    rollout: ["R-LOSS", "r-loss-l310"],
    reward: ["R-ADV", "rl-terminal-reward"],
    logprobs: ["R-LOSS", "r-loss-l310"],
    advantage:
      algorithm === "grpo"
        ? ["R-ADV", "rl-grpo-advantage"]
        : ["R-ADV", "rl-gae-carry"],
    policy_update: ["R-LOSS", "rl-ratio-clip"],
    refit: ["R-GRPO", "rl-refit-ack"],
  };
  const open = source[state.step] ?? source.logprobs;
  const identity = [algorithm, reduction, kl, force, equal, fault].join("/");
  function exportTrace() {
    download(
      {
        schema_version: 1,
        provenance: "reference",
        scope: trace.scope,
        actions_origin: fixture.actions_origin,
        runtime_training: "not_run",
        actual_rollout: "not_run",
        algorithm,
        reduction,
        kl_enabled: kl,
        force_on_policy: force,
        trajectory: item.id,
        token_index: j,
        ids: trace.ids[i],
        mask: trace.mask[i],
        reward: trace.rewards[i],
        versions: {
          generation: 0,
          previous: 0,
          current: trace.current_version,
          next: trace.next_version,
        },
        generation_logprobs: trace.generation_logprobs[i],
        previous_logprobs: trace.previous_logprobs[i],
        current_logprobs: trace.current_logprobs[i],
        reference_logprobs: trace.reference_logprobs[i],
        advantages: trace.advantages[i],
        ratio: trace.terms.ratio[i],
        clipped_ratio: trace.terms.clipped_ratio[i],
        logprob_gradient: trace.logprob_gradient[i],
        losses: {
          actor: trace.terms.actor_loss,
          kl: trace.terms.kl_loss,
          policy: trace.terms.policy_loss,
          value: algorithm === "ppo" ? trace.value.value_loss : null,
        },
        critic:
          algorithm === "ppo"
            ? {
                old_values: trace.value.old_values[i],
                values: trace.value.values[i],
                returns: trace.value.returns[i],
                gradient: trace.critic_gradient,
                after: trace.critic_after,
              }
            : null,
        head_row: {
          action,
          before: before[action],
          gradient: trace.head_gradient[action],
          after: trace.head_after[action],
        },
        logprobs_after: trace.logprobs_after[i],
      },
      "rl-reference-slice.json",
    );
  }
  return (
    <section className="panel rl-reference" aria-label="RL 数值参考闭环">
      <div className="section-header">
        <div>
          <span className="eyebrow">G07 · REFERENCE</span>
          <h2>固定轨迹上的 {algorithm.toUpperCase()} 更新</h2>
        </div>
        <button disabled={fault !== "none"} onClick={exportTrace}>
          导出 RL reference 切片
        </button>
      </div>
      <p className="notice" data-testid="rl-scope">
        四条 authored trajectories，非真实 rollout。复用 G03 完整两层
        decoder，backbone 冻结；训练 LM head [27,8]
        {algorithm === "ppo"
          ? " 和独立 critic head [8]"
          : "，本路线不训练 critic"}
        。所有数值为 reference，真实 runtime not_run。
      </p>
      <div className="controls rl-controls">
        <label>
          算法
          <select
            aria-label="RL algorithm"
            value={algorithm}
            onChange={(e) =>
              patch({ rlAlgorithm: e.target.value as State["rlAlgorithm"] })
            }
          >
            <option value="grpo">GRPO · 组优势</option>
            <option value="ppo">PPO · GAE + critic</option>
          </select>
        </label>
        <label>
          轨迹
          <select
            aria-label="RL trajectory"
            value={i}
            onChange={(e) => {
              const n = Number(e.target.value);
              patch({
                rlTrajectory: n,
                rlToken: fixture.trajectories[n].prompt.length,
              });
            }}
          >
            {fixture.trajectories.map((t, k) => (
              <option key={t.id} value={k}>
                {t.id}
              </option>
            ))}
          </select>
        </label>
        <label>
          Action 位置
          <input
            aria-label="RL token"
            type="number"
            min="0"
            max="12"
            value={j}
            onChange={(e) => patch({ rlToken: Number(e.target.value) })}
          />
        </label>
        <label>
          Actor 归约
          <select
            aria-label="RL reduction"
            value={reduction}
            onChange={(e) =>
              patch({ rlReduction: e.target.value as State["rlReduction"] })
            }
          >
            <option value="token">token mean</option>
            <option value="sequence">sequence mean</option>
          </select>
        </label>
        <label>
          Reference KL
          <select
            aria-label="RL KL"
            value={state.rlKl}
            onChange={(e) => patch({ rlKl: e.target.value as State["rlKl"] })}
          >
            <option value="on">β=0.02</option>
            <option value="off">β=0</option>
          </select>
        </label>
        <label>
          Force-on-policy
          <select
            aria-label="RL force"
            value={state.rlForce}
            onChange={(e) =>
              patch({ rlForce: e.target.value as State["rlForce"] })
            }
          >
            <option value="off">current version 1</option>
            <option value="on">version 0 · 一次更新</option>
          </select>
        </label>
        <label>
          奖励
          <select
            aria-label="RL equal rewards"
            value={state.rlEqual}
            onChange={(e) =>
              patch({ rlEqual: e.target.value as State["rlEqual"] })
            }
          >
            <option value="off">原始 authored rewards</option>
            <option value="on">全部同分 0.5</option>
          </select>
        </label>
        <label>
          梯度反例
          <select
            aria-label="RL fault"
            value={fault}
            onChange={(e) => {
              const f = e.target.value as State["rlFault"];
              patch({
                rlFault: f,
                ...(f === "constant_ratio"
                  ? { rlForce: "on", rlKl: "off" }
                  : f === "detach_kl_weight"
                    ? { rlKl: "on" }
                    : {}),
              });
            }}
          >
            <option value="none">正确参考</option>
            <option value="wrong_clip">max 错写 min</option>
            <option value="constant_ratio">ratio 错写常数 1</option>
            <option value="detach_kl_weight">错误 detach KL weight</option>
          </select>
        </label>
      </div>
      <p data-testid="rl-losses">
        PG={fmt(trace.terms.actor_loss)} · KL={fmt(trace.terms.kl_loss)} ·
        policy={fmt(trace.terms.policy_loss)} · head gradient norm=
        {fmt(gradientNorm)}
      </p>
      <p data-testid="rl-selection">
        trajectory={item.id} · j={j} · token={tokens.vocabulary[action]} · ID=
        {action} · mask={trace.mask[i][j]} · prefix 结尾=
        {j === 0 ? "dummy" : j - 1}
      </p>
      {fault !== "none" && (
        <p role="alert" data-testid="rl-counterexample">
          错误实现仅作反例，禁止导出/应用更新。全 head 梯度相对正确参考的最大差=
          {fmt(gradientDelta)}；loss 相同不代表梯度相同。
        </p>
      )}
      <div className="rl-actions">
        <button onClick={() => onSource(...open)}>
          查看 RL 当前子步骤源码
        </button>
        <button onClick={() => onSource("R-UTIL", "rl-group-std")}>
          查看组标准差源码
        </button>
        <button onClick={() => onSource("R-LOSS", "rl-kl-score-gradient")}>
          查看 KL 梯度源码
        </button>
        {algorithm === "ppo" && (
          <button onClick={() => onSource("R-LOSS", "rl-value-loss")}>
            查看 value loss 源码
          </button>
        )}
      </div>
      {(state.step === "rollout" || state.step === "reward") && (
        <>
          <h3>固定 prompt / response / reward</h3>
          <p>
            可见词表是教学 IDs，未加载 HF tokenizer；reward 为固定输入，没有执行
            reward model。
          </p>
          <div className="table-wrap">
            <table data-testid="rl-trajectories">
              <thead>
                <tr>
                  <th>ID / group</th>
                  <th>Prompt</th>
                  <th>Authored response</th>
                  <th>Mask</th>
                  <th>Reward</th>
                </tr>
              </thead>
              <tbody>
                {fixture.trajectories.map((t, k) => (
                  <tr key={t.id}>
                    <td>
                      {t.id}
                      <br />
                      group={t.group}
                    </td>
                    <td>
                      {t.prompt.map((v) => tokens.vocabulary[v]).join("")}
                    </td>
                    <td>
                      {t.response.map((v) => tokens.vocabulary[v]).join("")}
                    </td>
                    <td>{t.response_mask.join(",")}</td>
                    <td>{fmt(trace.rewards[k])}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
      <div className="table-wrap">
        <table data-testid="rl-token-logprobs">
          <caption>
            同一个 action 的四类 logprob 与策略梯度；mask=0 的行不进入 loss
          </caption>
          <thead>
            <tr>
              <th>j / token</th>
              <th>mask</th>
              <th>generation</th>
              <th>previous</th>
              <th>current</th>
              <th>reference</th>
              <th>A</th>
              <th>ratio</th>
              <th>clipped ratio</th>
              <th>PG token</th>
              <th>dL/d logp</th>
            </tr>
          </thead>
          <tbody>
            {trace.ids[i].map((v, t) => (
              <tr key={t} data-selected={t === j ? "true" : "false"}>
                <td>
                  <button
                    aria-label={"选择 RL token " + t}
                    aria-pressed={t === j}
                    onClick={() => patch({ rlToken: t })}
                  >
                    {t} · {tokens.vocabulary[v]}
                  </button>
                </td>
                <td>{trace.mask[i][t]}</td>
                {[
                  trace.generation_logprobs[i][t],
                  trace.previous_logprobs[i][t],
                  trace.current_logprobs[i][t],
                  trace.reference_logprobs[i][t],
                  trace.advantages[i][t],
                  trace.terms.ratio[i][t],
                  trace.terms.clipped_ratio[i][t],
                  trace.terms.pg_token[i][t],
                  trace.logprob_gradient[i][t],
                ].map((n, k) => (
                  <td key={k}>{fmt(n)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="small">
        generation/previous/reference 无当前梯度；current
        参与。generation=previous 是此参考设定。token/sequence 切换仅改变 loss
        归约，ratio 仍逐 token。
      </p>
      {algorithm === "grpo" ? (
        <div className="table-wrap">
          <table data-testid="rl-group">
            <caption>
              GRPO：Bessel 标准差，关闭 leave-one-out；无 critic 更新
            </caption>
            <thead>
              <tr>
                <th>trajectory</th>
                <th>reward</th>
                <th>baseline</th>
                <th>sample std</th>
                <th>sequence advantage</th>
              </tr>
            </thead>
            <tbody>
              {fixture.trajectories.map((t, k) => (
                <tr key={t.id}>
                  <td>{t.id}</td>
                  {[
                    trace.rewards[k],
                    trace.group.baseline[k],
                    trace.group.std[k],
                    trace.group.sequence_advantages[k],
                  ].map((v, n) => (
                    <td key={n}>{fmt(v)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <>
          <h3>PPO：旧 value → GAE / returns → 当前 critic loss</h3>
          <p data-testid="rl-value-loss">
            value loss={fmt(trace.value.value_loss)} · critic 始终 token mean ·
            actor/critic 图独立
          </p>
          <div className="table-wrap">
            <table data-testid="rl-gae">
              <thead>
                <tr>
                  <th>j</th>
                  <th>mask</th>
                  <th>terminal reward</th>
                  <th>old value</th>
                  <th>δ</th>
                  <th>A</th>
                  <th>return</th>
                  <th>current value</th>
                  <th>clipped value</th>
                  <th>value token loss</th>
                </tr>
              </thead>
              <tbody>
                {trace.ids[i].map((_, t) => (
                  <tr key={t}>
                    <td>{t}</td>
                    <td>{trace.mask[i][t]}</td>
                    {[
                      trace.gae.token_rewards[i][t],
                      trace.value.old_values[i][t],
                      trace.gae.delta[i][t],
                      trace.advantages[i][t],
                      trace.value.returns[i][t],
                      trace.value.values[i][t],
                      trace.value.clipped_values[i][t],
                      trace.value.value_token[i][t],
                    ].map((v, k) => (
                      <td key={k}>{fmt(v)}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p>
            mask=0 时保留 GAE carry，不额外折扣；最终 advantage 清零，returns
            可非零且不计入 value loss。
          </p>
          <button onClick={() => onSource("R-ADV", "rl-gae-output-mask")}>
            查看 GAE 最终 mask
          </button>
        </>
      )}
      <details className="rl-head">
        <summary>查看选定 action 的参数行与完整批次更新</summary>
        <p data-testid="rl-prefix-feature">
          prefix feature=[{vector(trace.features[i][j])}]
        </p>
        <p>
          action ID={action}；head 行梯度累计全部轨迹的贡献。SGD lr=
          {fixture.config.actor_lr}。
        </p>
        <div className="table-wrap">
          <table data-testid="rl-head-update">
            <thead>
              <tr>
                <th>h</th>
                <th>before</th>
                <th>gradient</th>
                <th>after</th>
              </tr>
            </thead>
            <tbody>
              {before[action].map((v, h) => (
                <tr key={h}>
                  <td>{h}</td>
                  <td>{fmt(v)}</td>
                  <td>{fmt(trace.head_gradient[action][h])}</td>
                  <td>{fmt(trace.head_after[action][h])}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p data-testid="rl-after-logprob">
          更新后同一 action logprob={fmt(trace.logprobs_after[i][j])}
        </p>
        {algorithm === "ppo" && (
          <div data-testid="rl-critic-update">
            <p>critic before=[{vector(fixture.critic_current)}]</p>
            <p>critic gradient=[{vector(trace.critic_gradient)}]</p>
            <p>critic after=[{vector(trace.critic_after)}]</p>
          </div>
        )}
      </details>
      <VersionDemo
        key={identity}
        trace={trace}
        visible={state.step === "policy_update" || state.step === "refit"}
        disabled={fault !== "none"}
        onSource={onSource}
      />
      <CourseDetails
        text={course}
        summary="展开 GRPO / PPO 精讲与独立数学符号表"
        className="rl-course"
      />
    </section>
  );
});
