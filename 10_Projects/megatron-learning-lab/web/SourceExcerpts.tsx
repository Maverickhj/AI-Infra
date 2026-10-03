import { stepFor } from "./gqa/steps";
import { useRef, useState } from "react";
import { sourceExcerptId, type State } from "./data";
import snippets from "../content/source-snippets.json";
import licenses from "../research/source-licenses.json";

export function SourceExcerpts({
  sourceId,
  repoKey,
  url,
  state,
}: {
  sourceId: string;
  repoKey: string;
  url: string;
  state: State;
}) {
  const entry = snippets.entries.find((e) => e.source_id === sourceId);
  const defaultExcerptId = sourceExcerptId(state, sourceId);
  const [excerptId, setExcerptId] = useState(() => defaultExcerptId || "");
  const [activeLine, setActiveLine] = useState<number | null>(null);
  const codeRef = useRef<HTMLPreElement>(null);
  if (!entry)
    return (
      <p className="notice">
        此条目尚未摘录，可通过完整源码链接查看。没有用伪代码代替原文。
      </p>
    );
  const excerpt = entry.excerpts.find((item) => item.id === excerptId);
  const picker = (
    <label>
      代码片段
      <select
        value={excerptId}
        onChange={(e) => {
          setExcerptId(e.target.value);
          setActiveLine(null);
        }}
      >
        <option value="" disabled>
          请选择相关片段
        </option>
        {entry.excerpts.map((item) => (
          <option key={item.id} value={item.id}>
            {item.title} · L{item.start_line}–L{item.end_line}
          </option>
        ))}
      </select>
    </label>
  );
  if (!excerpt)
    return (
      <section className="source-excerpts" aria-label="关键源码片段">
        <p className="notice">
          相关调用入口 / 实现片段待补：当前模型、步骤或模式尚无匹配摘录。
          可手动选择下列相关片段；它们不代表本步骤实现。
        </p>
        {picker}
      </section>
    );
  const lines = excerpt.code.replace(/\n$/, "").split("\n");
  const license = licenses.licenses.find((l) => l.repo_key === repoKey);
  function focusLine(line: number) {
    setActiveLine(line);
    codeRef.current?.querySelector(`[data-line="${line}"]`)?.scrollIntoView({
      block: "nearest",
      inline: "nearest",
      behavior: "instant",
    });
  }
  return (
    <section
      className="source-excerpts"
      aria-label="关键源码片段"
      data-snippet-id={excerpt.id}
    >
      <div className="snippet-heading">
        <span className="badge">逐字摘录 · 本地可读</span>
        <span className="small muted">{entry.excerpts.length} 个关键片段</span>
      </div>
      {picker}
      {state.operator !== "overview" &&
        defaultExcerptId &&
        excerptId === defaultExcerptId && (
          <p data-testid="source-relationship">
            {sourceId === "B-Q2" ||
            (sourceId === "C-ATTN" && state.operator === "rope")
              ? "调用入口"
              : stepFor(state.operator).kind}
            {" · "}Layer {state.layer} · query {state.query} · head{" "}
            {state.gqaHead}；静态源码不是该次模型运行。
          </p>
        )}
      {excerptId !== defaultExcerptId && (
        <p className="notice">
          当前为手动浏览的相关片段，不代表当前步骤或模式的实现。
        </p>
      )}
      <p className="snippet-boundary">
        仅展示下列连续行；前后代码未展示，不能当作独立可执行程序。静态讲解尚待人工
        review，不代表运行验证。
      </p>
      <div className="snippet-layout">
        <div className="snippet-code-wrap">
          <div className="code-toolbar">
            <span>原始源码 · {sourceId}</span>
            <span data-testid="source-range">
              L{excerpt.start_line}–L{excerpt.end_line}
            </span>
          </div>
          <pre
            className="source-code"
            ref={codeRef}
            tabIndex={0}
            aria-label="可横向滚动的源码原文"
          >
            <code>
              {lines.map((line, i) => (
                <span
                  className={`code-line${activeLine === excerpt.start_line + i ? " highlighted" : ""}`}
                  key={i}
                  data-line={excerpt.start_line + i}
                >
                  <span className="line-number" aria-hidden="true">
                    {excerpt.start_line + i}
                  </span>
                  <span
                    className={`source-text${line.trimStart().startsWith("#") ? " code-comment" : ""}`}
                  >
                    {line}
                    {"\n"}
                  </span>
                </span>
              ))}
            </code>
          </pre>
        </div>
        <aside className="code-notes" aria-label="中文讲解注释">
          <h3>读这一段时，留意什么？</h3>
          <p className="small muted">以下为 lab 新增讲解，未写入上游代码。</p>
          {excerpt.annotations.map((note) => (
            <div className="code-note" key={note.line}>
              <button
                className="source-chip"
                onClick={() => focusLine(note.line)}
              >
                定位 L{note.line}
              </button>
              <p>{note.text}</p>
            </div>
          ))}
          <p className="small muted">
            片段范围：{excerpt.title}
            。可能超出旧证据记录中的历史审阅范围；本轮仅补充该片段阅读，不扩写整文件审计结论。
          </p>
        </aside>
      </div>
      <p className="small muted">
        摘录校验：固定 commit 的完整 Git blob
        与证据清单匹配；本地片段保留原始空白及行号。
      </p>
      <a
        className="source-chip"
        href={`${url}#L${excerpt.start_line}-L${excerpt.end_line}`}
        target="_blank"
        rel="noreferrer"
      >
        在完整文件中定位此片段 ↗
      </a>
      <details className="source-license">
        <summary>上游版权与许可证</summary>
        <p>源文件：{entry.path}。片段内容未修改，中文讲解独立于原文。</p>
        {entry.copyright_header ? (
          <pre>{entry.copyright_header}</pre>
        ) : (
          <p>此文件没有单独版权头，保留仓库许可证。</p>
        )}
        {license ? (
          <>
            <a href={license.url} target="_blank" rel="noreferrer">
              固定版本 LICENSE
            </a>
            <pre>{license.text}</pre>
          </>
        ) : null}
      </details>
    </section>
  );
}
