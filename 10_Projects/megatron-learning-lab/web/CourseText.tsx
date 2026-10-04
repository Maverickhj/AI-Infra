import { memo, useState } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
// Static course text stays identical while token, rank or source controls change.
export const CourseText = memo(function CourseText({ text }: { text: string }) {
  return (
    <Markdown
      remarkPlugins={[remarkGfm, remarkMath]}
      rehypePlugins={[rehypeKatex]}
    >
      {text.replace(/^---\n[\s\S]*?\n---\n/, "")}
    </Markdown>
  );
});

// Mount long formula trees only while the reader has opened the course.
// Native summary click also covers Enter/Space keyboard activation.
export const CourseDetails = memo(function CourseDetails({
  text,
  summary,
  className,
}: {
  text: string;
  summary: string;
  className: string;
}) {
  const [open, setOpen] = useState(false);
  return (
    <details className={className} open={open}>
      <summary
        onClick={(event) => {
          event.preventDefault();
          setOpen((value) => !value);
        }}
      >
        {summary}
      </summary>
      {open && (
        <div className="prose">
          <CourseText text={text} />
        </div>
      )}
    </details>
  );
});
