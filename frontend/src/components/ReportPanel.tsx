import ReactMarkdown from "react-markdown";

type Props = {
  markdown: string;
  highlight?: boolean;
};

export function ReportPanel({ markdown, highlight }: Props) {
  return (
    <section className={`report-panel ${highlight ? "pulse" : ""}`}>
      <header className="section-head">
        <div>
          <h2>证据链研究报告</h2>
          <p>
            含方法演进、Limitation Lifecycle、External Critique 与 Evidence
            Boundary。KEEP 仅相对当前检索范围成立。
          </p>
        </div>
      </header>
      <article className="report-body markdown-body">
        <ReactMarkdown>{markdown}</ReactMarkdown>
      </article>
    </section>
  );
}
