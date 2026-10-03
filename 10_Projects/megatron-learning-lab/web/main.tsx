import { GqaWalkthrough } from "./gqa/GqaWalkthrough";
import { SourceExcerpts } from "./SourceExcerpts";
import React, {
  useCallback,
  useMemo,
  useEffect,
  useReducer,
  useRef,
  useState,
} from "react";
import { createRoot } from "react-dom/client";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";
import "./style.css";
import {
  models,
  cases,
  sources,
  sourceLock,
  samples,
  lessons,
  clean,
  sftSteps,
  rlSteps,
  labels,
  readState,
  reducer,
  sourceIds,
  type State,
  type Model,
} from "./data";

const Prose = React.memo(function Prose({ text }: { text: string }) {
  return (
    <div className="prose">
      <Markdown
        components={{ h1: ({ children }) => <h2>{children}</h2> }}
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
      >
        {clean(text)}
      </Markdown>
    </div>
  );
});
function Badge({ children }: { children: React.ReactNode }) {
  return <span className="badge">{children}</span>;
}
function Fact({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="fact">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}
function App() {
  const [state, dispatch] = useReducer(reducer, undefined, readState);
  const [sourceOpen, setSourceOpen] = useState(false);
  const [sourceId, setSourceId] = useState("");
  const sourceDialog = useRef<HTMLDialogElement>(null);
  const sourceTrigger = useRef<HTMLButtonElement>(null);
  const model = models.find((m) => m.id === state.model)!;
  const steps: readonly string[] = state.scenario === "rl" ? rlSteps : sftSteps;
  const index = steps.indexOf(state.step);
  const ids = useMemo(() => sourceIds(state), [state]);
  const patch = useCallback((value: Partial<State>) => dispatch(value), []);
  useEffect(() => {
    const onHash = () => dispatch(readState());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  useEffect(() => {
    const hash = new URLSearchParams(
      Object.entries(state).map(([k, v]) => [k, String(v)]),
    ).toString();
    history.replaceState(null, "", `#${hash}`);
  }, [state]);
  useEffect(() => {
    if (sourceOpen) {
      sourceDialog.current?.showModal();
    } else {
      sourceDialog.current?.close();
    }
  }, [sourceOpen]);
  const openSource = useCallback(
    (id?: string) => {
      setSourceId(id || ids[0] || "B-Q3");
      setSourceOpen(true);
    },
    [ids],
  );
  const currentSource =
    sources.find((s) => s.id === sourceId) ||
    sources.find((s) => s.id === ids[0]) ||
    sources[0];
  const selectedRepo =
    sourceLock.repositories[
      currentSource.repo_key as keyof typeof sourceLock.repositories
    ];
  const move = (delta: number) =>
    patch({
      step: steps[Math.max(0, Math.min(steps.length - 1, index + delta))],
    });
  const chapter =
    state.scenario === "rl"
      ? "rl"
      : model.attention === "mla"
        ? "deepseek"
        : "qwen";
  return (
    <>
      <a
        className="skip-link"
        href="#main-content"
        onClick={(e) => {
          e.preventDefault();
          document.getElementById("main-content")?.focus();
        }}
      >
        跳至主内容
      </a>
      <header className="topbar">
        <a
          className="brand"
          href="#"
          onClick={(e) => {
            e.preventDefault();
            patch({
              model: "qwen3-06b",
              scenario: "sft",
              view: "walkthrough",
              step: "decoder",
              layer: 0,
            });
          }}
        >
          <span className="logo">
            M<span>↗</span>
          </span>
          <span>
            MEGATRON<span className="brand-sub">LEARNING LAB</span>
          </span>
        </a>
        <div className="top-note">从一条样本，看见完整模型</div>
        <span className="environment">
          <i /> E0 · 本地学习
        </span>
      </header>
      <div className="workspace">
        <aside className="sidebar">
          <div className="eyebrow">学习工作台 / 01</div>
          <label htmlFor="model-select">当前模型</label>
          <select
            id="model-select"
            value={state.model}
            onChange={(e) => patch({ model: e.target.value, layer: 0 })}
          >
            {models.map((m) => (
              <option key={m.id} value={m.id}>
                {m.hf_id.split("/")[1]}
              </option>
            ))}
          </select>
          <p className="muted small">完整架构 · 配置推导</p>
          <nav aria-label="学习视图">
            {(
              [
                ["walkthrough", "01", "整模漫游"],
                ["atlas", "02", "模型对照"],
                ["sample", "03", "样本与监督"],
                ["course", "04", "完整课程"],
                ["basics", "00", "基础速览"],
              ] as const
            ).map(([id, num, label]) => (
              <button
                key={id}
                className={state.view === id ? "nav-item active" : "nav-item"}
                aria-current={state.view === id ? "page" : undefined}
                onClick={() =>
                  patch(
                    id === "sample"
                      ? { view: id, scenario: "sft", step: "input" }
                      : { view: id },
                  )
                }
              >
                <span>{num}</span>
                {label}
                <span className="nav-arrow">↗</span>
              </button>
            ))}
          </nav>
          <div className="sidebar-bottom">
            <span className="eyebrow">证据边界</span>
            <p>
              <span className="dot teal" /> derived · 结构与 shape
            </p>
            <p>
              <span className="dot amber" /> static_source_read · 源码记录
            </p>
            <p>
              <span className="dot gray" /> observed · 尚未运行
            </p>
            <p className="small muted">
              课程包含 AI 整理内容，尚待人工
              review。参考源码与当前运行环境分别记录。
            </p>
          </div>
        </aside>
        <main id="main-content" tabIndex={-1}>
          <div className="breadcrumb">
            学习实验室 <span>/</span>{" "}
            {state.scenario === "rl" ? "RL 全周期" : "完整模型与 SFT"}{" "}
            <span>/</span> {model.family}
          </div>
          <section className="page-heading">
            <div>
              <div className="eyebrow">WHOLE MODEL · SOURCE CONNECTED</div>
              <h1>
                {state.view === "atlas"
                  ? "同一骨架，不同计算"
                  : state.view === "sample"
                    ? "一条对话，哪些位置参与监督？"
                    : state.view === "course"
                      ? "沿着源码，读懂完整链路"
                      : state.view === "basics"
                        ? "基础速览：先定位，再深入"
                        : state.scenario === "rl"
                          ? "一次更新，还不是一次 RL 迭代"
                          : `${model.hf_id.split("/")[1]} 整模漫游`}
              </h1>
              <p>
                从输入、层内计算到参数更新。每一步都保留数学、实现分支与证据边界。
              </p>
            </div>
            <Badge>derived · 非实测</Badge>
          </section>
          <div className="context-bar">
            <div className="segmented" aria-label="训练场景">
              <button
                aria-pressed={state.scenario === "sft"}
                onClick={() => patch({ scenario: "sft" })}
              >
                SFT 全链路
              </button>
              <button
                aria-pressed={state.scenario === "rl"}
                onClick={() => patch({ scenario: "rl", view: "walkthrough" })}
              >
                RL Cycle
              </button>
            </div>
            <span className="context-model">
              {model.family} · {model.layers} 层
            </span>
            <span className="run-state">
              运行状态 <code>not_run</code>
            </span>
          </div>
          {state.view === "basics" ? (
            <section className="panel basics">
              <h2>你在完整训练链路的哪里？</h2>
              <p>
                Tensor 的 shape 描述数据布局；Linear 改变最后一维；embedding 把
                token 索引映射为向量；autograd 沿计算图汇集梯度。Transformer 将
                attention 与 FFN 堆叠，SFT 与 RL
                则提供不同的目标和状态生命周期。
              </p>
              <p>
                Bridge 负责模型配置与权重转换，Megatron Core
                负责模型及分布式执行，backend
                实现具体算子。结构、权重、一次运行是三个不同对象。
              </p>
              <button
                className="primary"
                onClick={() =>
                  patch({
                    view: "walkthrough",
                    scenario: "sft",
                    step: "decoder",
                  })
                }
              >
                跳过基础，进入整模核心 →
              </button>
            </section>
          ) : null}
          {state.view === "atlas" ? (
            <Atlas
              model={model}
              onSelect={(id) => patch({ model: id, layer: 0 })}
            />
          ) : null}

          {state.view === "walkthrough" ||
          state.view === "atlas" ||
          state.view === "sample" ? (
            <>
              <section className="pipeline panel" aria-label="完整计算流程">
                <div className="section-header">
                  <div>
                    <span className="eyebrow">
                      {state.scenario === "rl"
                        ? "TRAJECTORY → REFIT"
                        : "MESSAGES → MODEL → UPDATE"}
                    </span>
                    <h2>
                      {state.scenario === "rl"
                        ? "同步 RL 周期"
                        : "完整计算链路"}
                    </h2>
                  </div>
                  <span className="small muted">选择步骤深入 · 不自动播放</span>
                </div>
                <div className="step-list">
                  {steps.map((step, i) => (
                    <button
                      key={step}
                      aria-current={state.step === step ? "step" : undefined}
                      onClick={() => patch({ step })}
                    >
                      <span className="step-index">
                        {String(i + 1).padStart(2, "0")}
                      </span>
                      <span>{labels[step]}</span>
                      {step === "decoder" ? (
                        <small>× {model.layers} layers</small>
                      ) : (
                        <small>
                          {state.scenario === "rl" ? "语义路线" : "逻辑计算"}
                        </small>
                      )}
                    </button>
                  ))}
                </div>
              </section>
              <section className="panel detail" aria-label="当前步骤">
                <div className="section-header">
                  <div>
                    <span className="eyebrow">
                      STEP {String(index + 1).padStart(2, "0")} / {steps.length}
                    </span>
                    <h2>
                      {labels[state.step]}
                      <span className="heading-note">
                        {state.step === "decoder"
                          ? ` · Layer ${state.layer}`
                          : ""}
                      </span>
                    </h2>
                  </div>
                  <button
                    className="outline"
                    ref={sourceTrigger}
                    onClick={() => openSource()}
                  >
                    查看当前步骤源码 ↗
                  </button>
                </div>
                {state.scenario === "rl" ? (
                  <RLStep step={state.step} />
                ) : state.step === "decoder" ? (
                  <Decoder
                    model={model}
                    state={state}
                    patch={patch}
                    onSource={openSource}
                  />
                ) : state.step === "input" ? (
                  <SampleInspector state={state} patch={patch} />
                ) : (
                  <SFTStep model={model} step={state.step} />
                )}
                <div className="step-footer">
                  <span>
                    来源：
                    {ids.map((id) => (
                      <button
                        className="source-chip"
                        key={id}
                        onClick={() => openSource(id)}
                      >
                        {id}
                      </button>
                    ))}
                  </span>
                  <div>
                    <button
                      className="text-button"
                      disabled={index === 0}
                      onClick={() => move(-1)}
                    >
                      ← 上一步
                    </button>
                    <button
                      className="primary"
                      disabled={index === steps.length - 1}
                      onClick={() => move(1)}
                    >
                      下一步 →
                    </button>
                  </div>
                </div>
              </section>
              {state.scenario === "sft" &&
                state.step === "decoder" &&
                model.attention === "gqa" && (
                  <GqaWalkthrough
                    model={model}
                    state={state}
                    patch={patch}
                    onSource={openSource}
                  />
                )}
            </>
          ) : null}
          {state.view !== "basics" ? (
            <section className="panel course">
              <div className="section-header">
                <div>
                  <span className="eyebrow">READ · DERIVE · VERIFY</span>
                  <h2>
                    {state.view === "course"
                      ? "完整课程与独立符号表"
                      : "深入阅读当前案例"}
                  </h2>
                </div>
                <Badge>AI 整理 · 待人工 review</Badge>
              </div>
              <p className="muted">
                {chapter === "qwen"
                  ? "正文以 Qwen3-0.6B 为具体演算基线；当前模型的不同配置见上方模型图与对照。"
                  : chapter === "deepseek"
                    ? "正文以 DeepSeek-V3 为主，并列出 V2-Lite 的直接 Q 路径差异。"
                    : "正文描述完整 RL 数据合约；当前选择的模型尚未执行 RL 验证。"}
              </p>
              <Lesson
                text={lessons[chapter]}
                full={state.view === "course"}
                onSource={openSource}
              />
            </section>
          ) : null}
          <section className="runtime-status" aria-label="执行证据">
            <div>
              <span className="eyebrow">RUNTIME EVIDENCE</span>
              <h3>学习可继续，实测待采集</h3>
            </div>
            <p>
              HF reference / observed_bridge / observed_rl：<code>not_run</code>
              <br />
              没有 tokenizer trace、loss 曲线、GPU 耗时或激活热力图。
            </p>
            <button
              className="text-button"
              onClick={() => {
                patch({ sourceLane: "runtime" });
                openSource();
              }}
            >
              检查运行来源 →
            </button>
          </section>
          <footer>
            MEGATRON LEARNING LAB{" "}
            <span>参考档案 v2.1 · E0 课程与交互 · 不执行训练</span>
          </footer>
        </main>
      </div>
      <dialog
        ref={sourceDialog}
        className="source-dialog"
        onCancel={() => setSourceOpen(false)}
        onClick={(e) => {
          if (e.target === e.currentTarget) setSourceOpen(false);
        }}
        onClose={() => {
          setSourceOpen(false);
          sourceTrigger.current?.focus();
        }}
      >
        <div className="source-content">
          <div className="section-header">
            <div>
              <span className="eyebrow">SOURCE READER</span>
              <h2>源码与证据</h2>
            </div>
            <button
              autoFocus
              aria-label="关闭源码"
              onClick={() => setSourceOpen(false)}
            >
              关闭
            </button>
          </div>
          <p>
            {model.family} · {labels[state.step]} · layer {state.layer}
          </p>
          <label>
            来源通道
            <select
              value={state.sourceLane}
              onChange={(e) =>
                patch({ sourceLane: e.target.value as State["sourceLane"] })
              }
            >
              <option value="reference">固定参考源码</option>
              <option value="runtime">实际运行源码</option>
            </select>
          </label>
          {state.sourceLane === "runtime" ? (
            <div className="notice">
              <h3>运行来源尚未建立</h3>
              <p>
                没有该模型与场景的运行 manifest 或 runtime source
                mapping。当前环境版本不能代替实际 trace。
              </p>
              <code>not_checked / not_run</code>
            </div>
          ) : (
            <>
              <label>
                证据条目
                <select
                  value={currentSource.id}
                  onChange={(e) => setSourceId(e.target.value)}
                >
                  {sources.map((s) => (
                    <option key={s.id} value={s.id}>
                      {s.id} · {s.symbols[0]}
                    </option>
                  ))}
                </select>
              </label>
              <Badge>static_source_read</Badge>
              <h3>{currentSource.id}</h3>
              <p>{currentSource.review_scope}</p>
              <SourceExcerpts
                key={`${currentSource.id}:${state.model}:${state.scenario}:${state.step}:${state.layer}:${state.mla}:${state.operator}:${state.query}:${state.gqaHead}`}
                sourceId={currentSource.id}
                repoKey={currentSource.repo_key}
                url={currentSource.url}
                state={state}
              />
              <details className="source-metadata">
                <summary>完整来源与历史审阅信息</summary>
                <dl>
                  <dt>Repository</dt>
                  <dd>{selectedRepo.repository}</dd>
                  <dt>Commit</dt>
                  <dd>
                    <code>{selectedRepo.commit}</code>
                  </dd>
                  <dt>Path</dt>
                  <dd>
                    <code>{currentSource.path}</code>
                  </dd>
                  <dt>Symbols</dt>
                  <dd>
                    {currentSource.symbols.map((s) => (
                      <code key={s}>
                        {s}
                        <br />
                      </code>
                    ))}
                  </dd>
                  <dt>Git blob</dt>
                  <dd>
                    <code>{currentSource.git_blob_sha}</code>
                  </dd>
                </dl>
              </details>
              <p className="notice">
                关键片段已保存在本地。完整文件仍通过固定 commit
                链接查看；静态阅读不代表实际执行。
              </p>
              <a
                className="primary link-button"
                href={currentSource.url}
                target="_blank"
                rel="noreferrer"
              >
                打开完整源码文件 ↗
              </a>
            </>
          )}
        </div>
      </dialog>
    </>
  );
}

function Atlas({
  model,
  onSelect,
}: {
  model: Model;
  onSelect: (id: string) => void;
}) {
  return (
    <section className="panel">
      <div className="section-header">
        <h2>Model Atlas</h2>
        <Badge>config → derived</Badge>
      </div>
      <div className="model-cards">
        {models.map((m) => (
          <button
            key={m.id}
            className={m.id === model.id ? "model-card selected" : "model-card"}
            aria-pressed={m.id === model.id}
            onClick={() => onSelect(m.id)}
          >
            <span>{m.family}</span>
            <strong>{m.hf_id.split("/")[1]}</strong>
            <small>
              {m.layers} layers · {m.attention.toUpperCase()} ·{" "}
              {m.routed_experts ? `${m.routed_experts} experts` : "Dense"}
            </small>
          </button>
        ))}
      </div>
      <div className="table-wrap">
        <table>
          <caption>配置差异 · 不是实测对齐结果</caption>
          <thead>
            <tr>
              <th>模型</th>
              <th>Attention</th>
              <th>QKV bias</th>
              <th>QK norm</th>
              <th>Dense / MoE 层</th>
              <th>Routed / Top-k / Shared</th>
            </tr>
          </thead>
          <tbody>
            {models.map((m) => (
              <tr
                key={m.id}
                className={m.id === model.id ? "selected-row" : ""}
              >
                <th>{m.family}</th>
                <td>{m.attention.toUpperCase()}</td>
                <td>
                  {m.attention === "mla"
                    ? "MLA 独立投影"
                    : m.qkv_bias
                      ? "开启"
                      : "关闭"}
                </td>
                <td>
                  {m.attention === "mla"
                    ? "latent norm"
                    : m.qk_norm
                      ? "开启"
                      : "关闭"}
                </td>
                <td>
                  {m.routed_experts
                    ? `${m.first_dense_layers} / ${m.layers - (m.first_dense_layers || 0)}`
                    : `${m.layers} / 0`}
                </td>
                <td>
                  {m.routed_experts || 0} / {m.experts_per_token || 0} /{" "}
                  {m.shared_experts}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="notice">
        R1-Distill-Qwen 使用 Qwen2 dense 架构，不能作为 DeepSeek MLA/MoE
        的替代。A3B 活跃参数也不等于常驻权重显存。
      </p>
      <p className="small">
        当前 HF revision：
        <code>
          {model.revision ||
            "未锁定 · 仅供结构学习，不可标称 reproducible observed run"}
        </code>
      </p>
    </section>
  );
}

function Decoder({
  model: m,
  state,
  patch,
  onSource,
}: {
  model: Model;
  state: State;
  patch: (p: Partial<State>) => void;
  onSource: (id?: string) => void;
}) {
  const c = cases.find((c) => c.id === m.id)!;
  const layer = c.layers[state.layer];
  const isMoE = layer.feed_forward === "moe";
  const [head, setHead] = useState(0);
  const [headError, setHeadError] = useState(false);
  const currentHead = Math.min(head, m.heads - 1);
  useEffect(() => {
    setHead((previous) => Math.min(previous, m.heads - 1));
    setHeadError(false);
  }, [m.id, m.heads]);
  return (
    <>
      <p className="lead">
        {m.attention === "gqa"
          ? "Q 的投影宽度，由 head_dim 决定。"
          : "MLA 的低秩表示，如何还原为 attention？"}{" "}
        <span className="muted">选择任意一层，查看它实际采用的结构。</span>
      </p>
      <div className="layer-label">
        <span>完整层栈 · 从 0 编号</span>
        <span>
          <i className="dot teal" /> Dense <i className="dot amber" /> MoE
        </span>
      </div>
      <div className="layer-strip" aria-label="Decoder 层选择">
        {c.layers.map((l) => (
          <button
            key={l.layer_index}
            aria-label={`Layer ${l.layer_index} ${l.feed_forward === "moe" ? "MoE" : "Dense"}`}
            aria-pressed={state.layer === l.layer_index}
            className={l.feed_forward === "moe" ? "moe" : ""}
            onClick={() => patch({ layer: l.layer_index })}
          >
            {l.layer_index}
          </button>
        ))}
      </div>
      <div className="facts">
        <Fact label="Residual width H" value={m.hidden_size} />
        <Fact label="Attention" value={m.attention.toUpperCase()} />
        <Fact label="当前 FFN" value={isMoE ? "MoE" : "Dense SwiGLU"} />
        <Fact
          label="共享 embedding / head"
          value={m.tie_word_embeddings ? "是" : "否"}
        />
      </div>
      <p className="small muted">
        逻辑图采用 [B,S,…]；真实 MCore 可能使用 [S,B,H] 或 packed
        THD。TP=PP=CP=EP=1，SP 关闭；padded vocab 待 provider 确认。
      </p>
      {m.attention === "gqa" ? (
        <>
          <div className="flow-graph">
            <div className="flow-node">
              Residual<span>[B,S,{m.hidden_size}]</span>
            </div>
            <span className="flow-arrow">→</span>
            <button className="flow-node" onClick={() => onSource("C-ATTN")}>
              Norm + QKV
              <span>
                宽度 {m.heads * m.head_dim! + 2 * m.kv_heads! * m.head_dim!}
              </span>
            </button>
            <span className="flow-arrow">→</span>
            <div className="flow-node accent">
              GQA
              <span>
                {m.heads} Q / {m.kv_heads} KV
              </span>
            </div>
            <span className="flow-arrow">→</span>
            <div className="flow-node">
              Output + residual<span>[B,S,{m.hidden_size}]</span>
            </div>
          </div>
          <div className="facts" data-testid="attention-facts">
            <Fact label="Q 总宽度" value={m.heads * m.head_dim!} />
            <Fact label="K / V 各自宽度" value={m.kv_heads! * m.head_dim!} />
            <Fact
              label="Fused QKV 宽度"
              value={m.heads * m.head_dim! + 2 * m.kv_heads! * m.head_dim!}
            />
            <Fact label="head_dim" value={m.head_dim!} />
            <Fact label="QKV bias" value={m.qkv_bias ? "开启" : "关闭"} />
            <Fact label="QK norm" value={m.qk_norm ? "开启" : "关闭"} />
          </div>
          <Prose
            text={`\n\n$$\nD_Q=n_q d=${m.heads}\\times ${m.head_dim}=${m.heads * m.head_dim!},\\qquad D_K=D_V=n_{kv}d=${m.kv_heads! * m.head_dim!}.\n$$\n\n\nQ: \`[B,S,${m.heads},${m.head_dim}]\`；K/V: \`[B,S,${m.kv_heads},${m.head_dim}]\`。$n_q,n_{kv}$ 是 Q/KV head 数，$d$ 是每头维度。${m.head_dim_origin}。`}
          />
          <div className="exercise">
            <label>
              选择 Q head{" "}
              <input
                aria-label="Q head"
                type="number"
                min="0"
                max={m.heads - 1}
                value={currentHead}
                step="1"
                aria-invalid={headError}
                aria-describedby={headError ? "head-error" : undefined}
                onChange={(e) => {
                  const value = e.target.valueAsNumber;
                  const valid =
                    Number.isInteger(value) && value >= 0 && value < m.heads;
                  setHeadError(!valid);
                  if (valid) setHead(value);
                }}
              />
            </label>
            {headError && (
              <p id="head-error" role="alert">
                请输入 0–{m.heads - 1} 的整数；保留上一个合法 head。
              </p>
            )}
            <strong>
              Q head {currentHead} → KV group{" "}
              {Math.floor(currentHead / (m.heads / m.kv_heads!))}
            </strong>
            <p>
              每组 {m.heads / m.kv_heads!} 个 Q heads 共享一组 K/V。权重映射按
              GQA group 排列，不能简单拼接全部 Q、K、V。
            </p>
          </div>
          <p>
            计算路线：RMSNorm → grouped QKV split →{" "}
            {m.qk_norm ? "Q/K norm → " : ""}Q/K RoPE → causal attention → output
            projection → residual。TE fused kernel 未必暴露独立 norm 或完整
            score 矩阵。
          </p>
          <p className="notice">
            反例：把 head_dim 一律写成 H / heads，会错误改变 Qwen3-0.6B 的 Q
            投影。验证时应检查 config 显式字段、QKV mapping 和真实输入输出切片。
          </p>
        </>
      ) : (
        <>
          <div className="segmented">
            <button
              aria-pressed={state.mla === "train"}
              onClick={() => patch({ mla: "train" })}
            >
              MLA 训练视图
            </button>
            <button
              aria-pressed={state.mla === "decode"}
              onClick={() => patch({ mla: "decode" })}
            >
              MLA decode 视图
            </button>
          </div>
          {state.mla === "train" ? (
            <div className="mla-graph" data-testid="mla-graph">
              <div className="flow-node">
                Hidden states <span>[B,S,{m.hidden_size}]</span>
              </div>
              <div className="mla-branches">
                <div className="flow-node accent">
                  {m.q_lora_rank
                    ? `q_down → norm → q_up`
                    : "直接 Q projection（无 q_down）"}
                  <span>
                    {m.q_lora_rank
                      ? `${m.hidden_size} → ${m.q_lora_rank} → ${m.heads * (m.qk_nope_head_dim! + m.qk_rope_head_dim!)}`
                      : `${m.hidden_size} → ${m.heads * (m.qk_nope_head_dim! + m.qk_rope_head_dim!)}`}
                  </span>
                  <small>
                    每头 Q = no-PE {m.qk_nope_head_dim} + RoPE{" "}
                    {m.qk_rope_head_dim}
                  </small>
                </div>
                <div className="flow-node accent">
                  kv_down → 分离位置 K → norm → kv_up
                  <span>
                    压缩 KV {m.kv_lora_rank} + 位置 K {m.qk_rope_head_dim}
                  </span>
                  <small>展开内容 K / V · V 每头 {m.v_head_dim}</small>
                </div>
              </div>
              <div className="flow-node">
                QK 打分 → causal attention → V → output projection
                <span>训练展开 Q/K/V，不能直接套用 decode cache 大小</span>
              </div>
            </div>
          ) : (
            <div className="mla-graph" data-testid="mla-graph">
              <div className="flow-node">
                新 token → kv_down<span>每步追加 latent KV 与位置 K</span>
              </div>
              <div className="flow-node accent">
                Latent cache
                <span>
                  压缩 KV {m.kv_lora_rank} + 位置 K {m.qk_rope_head_dim}
                </span>
              </div>
              <div className="flow-node">
                条件 absorption → attention → output
                <span>
                  cached-latent inference 分支；未执行，不是训练激活图
                </span>
              </div>
            </div>
          )}
          <p>
            Q/K 每头打分维度 {m.qk_nope_head_dim! + m.qk_rope_head_dim!}，V 每头{" "}
            {m.v_head_dim}。KV rank {m.kv_lora_rank} 是压缩表示宽度，不是每个
            head 的 K 宽度。
          </p>
          <button className="source-chip" onClick={() => onSource("C-MLA")}>
            C-MLA · 查阅条件分支 ↗
          </button>
        </>
      )}
      <div className="ffn-block">
        <h3>
          Layer {state.layer} ·{" "}
          {isMoE ? "路由、专家与组合" : "SwiGLU 与第二次残差"}
        </h3>
        {isMoE ? (
          <>
            <div className="flow-graph">
              <button className="flow-node" onClick={() => onSource("C-MOE")}>
                Route
                <span>
                  top-{m.experts_per_token} / {m.routed_experts}
                </span>
              </button>
              <span>→</span>
              <div className="flow-node">
                Dispatch<span>expert assignment</span>
              </div>
              <span>→</span>
              <div className="flow-node accent">
                Routed experts<span>各自 gate / up / down</span>
              </div>
              <span>→</span>
              <div className="flow-node">
                Combine + residual
                <span>Shared experts: {m.shared_experts}</span>
              </div>
            </div>
            <p>
              shared expert 分支：
              {m.shared_experts
                ? `${m.shared_experts} 个共享专家作用于 token，与 routed 输出相加；实际 overlap 以 backend 为准。`
                : "该配置没有 shared expert，不能套用 DeepSeek 的共享分支。"}
            </p>
            <p>
              MoE 路由顺序：route → preprocess → dispatch →
              routed_experts_compute → combine → postprocess。未采集 token
              分配，不展示专家负载热力图。
            </p>
          </>
        ) : (
          <>
            <Prose
              text={
                "\n\n$$\ng=XW_g,\\quad u=XW_u,\\quad h=\\operatorname{SiLU}(g)\\odot u,\\quad o=hW_d.\n$$\n\n\n$X$ 为归一化输入；$W_g,W_u,W_d$ 为 gate/up/down 权重；$g,u,h,o$ 为相应中间结果。"
              }
            />
            <p>
              gate / up 各 {m.ffn_hidden_size}，融合宽度{" "}
              <strong>{2 * m.ffn_hidden_size}</strong>；down 回到{" "}
              {m.hidden_size}。实际 Linear 权重按 [out,in] 保存。第二次 residual
              add 后进入下一层。
            </p>
          </>
        )}
        {m.mtp_layers_in_hf_config ? (
          <p className="notice">
            可选 MTP：配置中 {m.mtp_layers_in_hf_config} 层；未包含在 {m.layers}{" "}
            层标准 decoder 主图中，未执行。
          </p>
        ) : null}
      </div>
      <details>
        <summary>实现分支、反例与验证方法</summary>
        <p>
          语义算子与 nn.Module 不一一对应：norm 可以融合到 TE Linear，hook
          看不到独立模块不代表未执行。先按映射还原权重布局，再比较 HF 与 Bridge
          的少量 tensor。当前没有运行数据，以上只描述验证方法。
        </p>
        <p>
          MoE 辅助目标、expert bias 与 padding
          排除需要按配置核验；同权重下的训推路由也不保证相同。MLA decode cache
          不能替代训练显存估算。
        </p>
      </details>
    </>
  );
}

function SampleInspector({
  state,
  patch,
}: {
  state: State;
  patch: (p: Partial<State>) => void;
}) {
  const sample = samples[state.sample];
  const last = sample.messages.map((m) => m.role).lastIndexOf("assistant");
  const [shift, setShift] = useState<"once" | "twice">("once");
  return (
    <div className="sample-inspector">
      <div className="section-header">
        <h3>Sample Journey · 原始 messages</h3>
        <Badge>authored sample · 非 tokenizer trace</Badge>
      </div>
      <label>
        课程样本
        <select
          value={state.sample}
          onChange={(e) => patch({ sample: Number(e.target.value) })}
        >
          {samples.map((s, i) => (
            <option key={s.id} value={i}>
              {s.id}
            </option>
          ))}
        </select>
      </label>
      <div className="segmented mask-modes" aria-label="监督模式">
        {(["assistant", "last_turn", "full"] as const).map((mode) => (
          <button
            key={mode}
            aria-pressed={state.mode === mode}
            onClick={() => patch({ mode })}
          >
            {mode}
          </button>
        ))}
      </div>
      <p>
        {state.mode === "assistant"
          ? "监督所有 assistant 回复，user 内容仍参与上下文计算。"
          : state.mode === "last_turn"
            ? "只监督最后一轮 assistant 回复，前面的回复仍保留在上下文。"
            : "监督完整文本序列中的有效目标；实际 special token 与截断边界以预处理为准。"}
      </p>
      <div className="messages">
        {sample.messages.map((m, i) => {
          const supervised =
            state.mode === "full" ||
            (m.role === "assistant" &&
              (state.mode === "assistant" || i === last));
          return (
            <div
              className={supervised ? "message supervised" : "message"}
              data-testid="message-span"
              data-supervised={supervised}
              key={i}
            >
              <span className="role">
                {m.role}
                <small>span {i + 1}</small>
              </span>
              <p>{m.content}</p>
              <span className="mask">{supervised ? "监督范围" : "上下文"}</span>
            </div>
          );
        })}
      </div>
      <p className="notice">
        这是角色跨度示意，不是逐 token mask。Rendered template、token
        IDs、EOS/特殊符号边界尚未采集，不能把一个 span 当成一个 token。真实 loss
        mask 跟随目标位置。
      </p>
      <div className="exercise">
        <div className="section-header">
          <h3>只做一次 next-token 移位</h3>
          <select
            aria-label="移位反例"
            value={shift}
            onChange={(e) => setShift(e.target.value as "once" | "twice")}
          >
            <option value="once">正确：一次 shift</option>
            <option value="twice">反例：重复 shift</option>
          </select>
        </div>
        <p>符号序列 x₀…xₛ：仅演示位置关系，不表示真实 token ID。</p>
        <div className="shift-grid">
          <span>Input</span>
          <code>x₀</code>
          <code>x₁</code>
          <code>x₂</code>
          <span>Target</span>
          <code>{shift === "once" ? "x₁" : "x₂"}</code>
          <code>{shift === "once" ? "x₂" : "x₃"}</code>
          <code>{shift === "once" ? "x₃" : "x₄"}</code>
        </div>
        <p role="status">
          {shift === "once"
            ? "目标与输入相差一个位置；mask 对齐 target。dataset/collator 已移位时，后续不能再移位。"
            : "错误：跳过了紧邻目标，监督语义发生改变。检查 dataset/collator 与模型是否重复移位。"}
        </p>
      </div>
      <details>
        <summary>Padding、packing 与验证边界</summary>
        <p>
          padding 不参与有效目标统计。packing 后需保留样本边界、position 与
          cu_seqlens；不得让前一个样本的末尾预测后一个样本的开头。本轮未生成
          packed token trace；默认短序列、无 packing、CP=1。
        </p>
        <p>
          验证方法：导出少量真实 input/label/mask 对照，检查 assistant
          模式、截断边界及全局有效 token 数。prompt 不被监督不意味着它不参与
          forward 或其表示没有梯度。
        </p>
      </details>
    </div>
  );
}

const sftCopy: Record<string, string> = {
  embedding:
    "### 从索引进入 residual stream\n输入 tokens 为 `[B,S]`，embedding 输出逻辑 `[B,S,H]`。实际 MCore storage 可能为 `[S,B,H]`；packed THD 是另一条路径。\n\n结构构建与权重加载必须分开：AutoBridge 根据 architecture 选择 family bridge，再由 provider/spec 构建 GPTModel。只有 checkpoint 导入及映射检查，才能证明使用了预训练权重。\n\n**验证：** 固定 tokenizer/checkpoint revision，核对少量 embedding 行及 input IDs；不将随机初始化 smoke test 称为 SFT。",
  final_norm:
    "### 最后一层之后还有一次归一化\n最后一个 decoder 的 residual 输出进入 final RMSNorm，输出仍为 `[B,S,H]`，再送入词表投影。\n\n**实现分支：** PP 下的 post-process stage 决定 final norm/head 的实际归属。首轮仅解释 TP=PP=CP=1。\n\n**反例与验证：** 不要把最后一层内的 pre-FFN norm 当作 final norm；按 GPTModel 与 provider 的 stage 配置核对实际模块。当前没有对应 activation trace。",
  lm_head:
    "### 从 hidden states 到词表目标\n逻辑 logits 是 `[B,S,V_pad]`。$V_{pad}$ 为 provider padding 后的词表大小；HF 的 $V$ 不能直接替代它。\n\n带 labels 的 MCore GPT 默认后处理可以直接返回 **token loss**。概念上存在 logits，不意味着 forward 返回值就是 logits。\n\n**反例：** 对 token loss 再做一次 cross entropy，会改变含义。\n\n**验证：** 检查 labels、output_processor、MTP 分支与实际返回结构，仅捕获少量所需输出。",
  loss: "### Mask、sum/count 与全局归一化\n\n\n$$\nL_{SFT}=\\frac{\\sum_t m_t\\ell_t}{N},\\quad N=\\sum_t m_t.\n$$\n\n\n$t$ 为目标位置，$m_t$ 为目标 mask，$\\ell_t$ 为该位置 NLL，$N$ 为全局有效目标数，$L_{SFT}$ 为此教学基线的目标。\n\nBridge callback 返回 loss sum 与 num_tokens；调度、累积和梯度 finalize 的缩放仍需结合实际配置验证。**不能从 callback 独自推断最终训练目标。**\n\n**演算反例（教学数字，非实测）：** rank A 的 loss sum=2、count=1；rank B 的 sum=12、count=3。局部均值再平均是 (2+4)/2=3，全局 sum/count 是 14/4=3.5。\n\n**验证：** 保持同一组有效 tokens，比较单卡与拆分 batch 的 loss/梯度；全 mask rank 的零分母路径单独检查。",
  backward:
    "### 梯度从目标流向整模\nhead 接收 masked loss 的梯度；每次 residual add 把梯度送往恒等分支与子层分支，随后在共享上游表示汇集。attention 与 FFN 分别产生输入和权重梯度。\n\n共享 embedding/head 时，同一份参数汇集输入查表与输出投影两条路径的贡献。不能把 backward 简单理解为 forward 动画反放。\n\n**实现分支：** Full SFT 与 LoRA 的 trainable 参数集合不同；分布式归约、累积与 loss scaling 也会影响梯度。\n\n**验证：** 记录选定层 grad norm、trainable names 和少量参数更新；冻结参数保持不变。fused/custom autograd 内部未暴露的边界标记不可直接观测。",
  update:
    "### 更新、保存与恢复是不同状态\n优化器根据梯度更新可训练参数，scheduler 推进学习率状态。必须核对真实参数变化，不能仅凭有限 loss 宣布更新成功。\n\n| 操作 | 目的 | 必须核对 |\n|---|---|---|\n| 预训练权重导入 | 开始 SFT | checkpoint revision、映射与加载证据 |\n| Training checkpoint resume | 延续训练 | global step、optimizer、scheduler、RNG 等所需状态 |\n| HF export | 推理或跨引擎使用 | 权重映射、格式及输出一致性 |\n\n**反例：** HF 权重导出成功不证明 optimizer 状态可以恢复。\n\n**验证：** 在固定输入与 RNG 条件下比较连续训练与 save/resume 后的下一步；声明容差与 backend。当前三项都为 `not_run`。",
};
function SFTStep({ model, step }: { model: Model; step: string }) {
  return (
    <>
      <div className="facts">
        <Fact label="完整 decoder" value={`${model.layers} 层`} />
        <Fact label="Hidden size" value={model.hidden_size} />
        <Fact
          label="HF vocab / padded vocab"
          value={`${model.vocab_size} / 待确认`}
        />
        <Fact label="权重与执行" value="not_run" />
      </div>
      <Prose text={sftCopy[step] || ""} />
    </>
  );
}
const rlCopy: Record<string, string> = {
  rollout:
    "### 固定 policy 版本，保存同一条 trajectory\n渲染 prompt → rollout backend 采样 → 保存实际 action token IDs、结束原因与 response mask。prompt_id、group_id、trajectory_id 和 rollout_weight_version 必须保留。\n\n首个实验建议 temperature=1、无 top-k/p 截断，减少采样分布定义差异；这只是待验证配置。训练读取同一份 token IDs，不重新 tokenize 格式化后的回复。\n\n**反例：** 将不同 prompt 的回复混成一个 GRPO group，会破坏组内基线。当前没有实际生成的回复，课程样本不冒充 rollout。",
  reward:
    "### 奖励函数与训练循环分别验证\n算术题可使用可检查的正确性奖励。奖励函数单元测试不证明模型真的生成过这些答案。reward 必须绑定 trajectory、prompt group 与有效 sample mask。\n\n**反例：** 组内丢弃样本却继续使用旧组基线，会改变 advantage。全组同分可能没有有用的组内优势；reward 未上升也不自动说明系统错误。",
  logprobs:
    "### 两种差异回答不同问题\n$\\ell^{gen}-\\ell^{prev}$ 比较生成与训练后端；$\\ell^{cur}-\\ell^{prev}$ 比较当前与更新前 policy。$\\ell$ 表示 action token 的 logprob，上标分别对应下方四种来源。\n\n\n\n$$\nr_{i,t}=\\exp(\\ell^{cur}_{i,t}-\\ell^{prev}_{i,t}).\n$$\n\n\n$i,t$ 是 trajectory 和 token 位置，$r$ 是更新 ratio。训推校正与 PPO ratio 不是同一概念。\n\n**验证：** 固定同一 token 序列、权重版本和 sampling config，检查 response mask/旧 logprob 的目标位置偏移。不同 backend 或 MoE 路由可能带来差异。",
  advantage:
    "### 同一个 prompt 内比较 responses\n\n\n$$\nA_i=\\frac{R_i-\\mu_R}{\\sigma_R+\\delta}.\n$$\n\n\n$i$ 为组内样本，$R_i$ 是 reward，$\\mu_R,\\sigma_R$ 为组内均值、标准差，$\\delta$ 为稳定项，$A_i$ 为教学 advantage。\n\n这只是概念式。实际 estimator 的 normalize、leave-one-out、标准差定义与同分行为需按正式实现核验，不能把该公式视为所有 GRPO 分支。\n\n**验证：** 对固定 reward 向量逐项比较 estimator 输出；PPO 则还需要 value predictions、returns 与 GAE。",
  policy_update:
    "### Actor loss 不是把所有回复再做一次 SFT\n使用 action logprob、advantage、ratio、clipping 与可选 reference KL；token_mask × sample_mask 决定有效贡献。token-level 和 sequence-level 归约对长短回复的权重不同。\n\n**关键反例：** force_on_policy_ratio 可让 ratio 前向值为 1，同时通过 detach 保留当前梯度。直接替换为字面常数 1 会改变计算图。\n\n实现应复用正式 ClippedPGLossFn 的选定分支；用固定微型输入核对数值与梯度，随后确认 optimizer 真正改变参数。\n\n**PPO 分支：** actor 和 critic 各有梯度路径；增加 value predictions → returns/GAE → value loss 与独立 optimizer，不能只把 GRPO 改个标题。",
  refit:
    "### Optimizer step 之后，生成端还需要新权重\n训练端更新 → export/refit → 验证 generation backend 的权重版本 → 下一批 rollout。training_weight_version 与 generation_weight_version 必须独立记录。\n\nTraining checkpoint resume、HF export 和 in-memory refit 是三种不同能力。保存成功不证明推理端已经换权重。\n\n**验证：** refit 后用固定输入检查 token logprob，确认 rollout 使用新版本，至少两次同步迭代追踪版本闭环。两次迭代仅证明流程，不证明能力提升。",
};
function RLStep({ step }: { step: string }) {
  return (
    <>
      <p className="notice">
        同步 RL 教学路线 · 真实 rollout / update / refit 全部{" "}
        <code>not_run</code>。未生成 trajectory、奖励或 logprob 数值。
      </p>
      <div className="logprob-grid">
        {[
          [
            "generation_logprobs",
            "生成后端",
            "不参与当前梯度",
            "实际采样时记录",
          ],
          [
            "prev_logprobs",
            "训练后端 · 更新前",
            "不参与当前梯度",
            "固定 trajectory 的重算",
          ],
          [
            "current_logprobs",
            "训练后端 · 当前 policy",
            "参与当前梯度",
            "当前可导 forward",
          ],
          [
            "reference_policy_logprobs",
            "固定 reference",
            "不参与当前梯度",
            "选定 KL 分支需要时使用",
          ],
        ].map(([name, origin, grad, desc]) => (
          <article className="logprob" key={name}>
            <code>{name}</code>
            <h3>{origin}</h3>
            <p>{desc}</p>
            <small>{grad}</small>
            <span className="run-state">not_run</span>
          </article>
        ))}
      </div>
      <Prose text={rlCopy[step]} />
      <div className="facts">
        <Fact label="Training weight version" value="未采集" />
        <Fact label="Generation weight version" value="未采集" />
        <Fact label="Refit 状态" value="not_run" />
      </div>
    </>
  );
}

function Lesson({
  text,
  full,
  onSource,
}: {
  text: string;
  full: boolean;
  onSource: (id: string) => void;
}) {
  const parts = clean(text).split(/(?=^## )/m);
  const symbol = parts.find((p) => p.startsWith("## 数学符号")) || "";
  const body = parts.filter(
    (p) =>
      !p.startsWith("## 数学符号") &&
      p.replace(/^#+ [^\n]+\n?/, "").trim().length > 0,
  );
  return (
    <>
      <details className="symbols">
        <summary>独立数学符号表 · 随时查阅</summary>
        <Prose text={symbol} />
      </details>
      {body.map((p, i) =>
        full ? (
          <Prose key={i} text={p} />
        ) : (
          <details key={i} className="lesson-section" open={i === 0}>
            <summary>
              {p
                .split("\n")
                .find((line) => line.startsWith("#"))
                ?.replace(/^#+ /, "") || "学习目标"}
            </summary>
            <Prose text={p.replace(/^#+ .+\n/, "")} />
          </details>
        ),
      )}
      <div className="source-index">
        <span className="small muted">本章源码证据</span>
        {[
          ...new Set(
            [...text.matchAll(/\[((?:B|C|R)-[A-Z0-9]+)\]/g)].map((m) => m[1]),
          ),
        ]
          .filter((id) => sources.some((s) => s.id === id))
          .map((id) => (
            <button
              className="source-chip"
              key={id}
              onClick={() => onSource(id)}
            >
              {id}
            </button>
          ))}
      </div>
    </>
  );
}
createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
