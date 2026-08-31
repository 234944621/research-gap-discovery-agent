type Props = {
  topic: string;
  onTopicChange: (value: string) => void;
  loading: boolean;
  error: string;
  onSubmit: () => void;
  onCancel: () => void;
};

export function HomeCompose({
  topic,
  onTopicChange,
  loading,
  error,
  onSubmit,
  onCancel,
}: Props) {
  return (
    <section className="home-compose">
      <div className="home-copy">
        <p className="brand-mark">科研空白发现智能体</p>
        <h1>
          从文献证据链里，
          <br />
          找出仍未闭合的研究空白
        </h1>
        <p className="lede">
          记忆召回 → 检索规划 → 全文与引用链 → 局限生命周期 →
          工具调用验证。帮助发现候选科研空白，不宣称自动发现创新点。
        </p>
      </div>

      <form
        className="topic-form"
        onSubmit={(e) => {
          e.preventDefault();
          onSubmit();
        }}
      >
        <label htmlFor="topic">研究主题</label>
        <textarea
          id="topic"
          value={topic}
          onChange={(e) => onTopicChange(e.target.value)}
          placeholder="例如：跨链桥智能合约漏洞检测"
          rows={4}
          required
          disabled={loading}
        />
        <div className="form-row">
          <button type="submit" className="primary-btn" disabled={loading}>
            {loading ? "证据链构建中…" : "开始发现空白"}
          </button>
          {loading && (
            <button type="button" className="ghost-btn" onClick={onCancel}>
              取消
            </button>
          )}
        </div>
        {error ? <p className="error-text">{error}</p> : null}
        <p className="fineprint">
          证据边界：OA 全文 / abstract_only / AI inferred 将在报告中显式标注。
        </p>
      </form>
    </section>
  );
}
