import { useMemo, useState } from "react";

export type StateDiff = {
  field: string;
  before: unknown;
  after: unknown;
};

export type StatePatch = {
  id: string;
  node: string;
  diff: StateDiff[];
  state: Record<string, unknown>;
};

type Props = {
  currentState: Record<string, unknown> | null;
  patches: StatePatch[];
  activeNode: string | null;
};

function preview(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

export function StateInspector({ currentState, patches, activeNode }: Props) {
  const [selectedPatchId, setSelectedPatchId] = useState<string | null>(null);

  const selected = useMemo(() => {
    if (!patches.length) return null;
    if (selectedPatchId) {
      return patches.find((p) => p.id === selectedPatchId) || patches[patches.length - 1];
    }
    return patches[patches.length - 1];
  }, [patches, selectedPatchId]);

  return (
    <section className="state-panel">
      <header className="section-head">
        <div>
          <h2>Shared ResearchState</h2>
          <p>
            节点写入的状态字段会在此汇总；点选左侧 patch 可查看该步 diff
            {activeNode ? ` · 当前运行：${activeNode}` : ""}
          </p>
        </div>
      </header>

      <div className="state-grid">
        <aside className="patch-list">
          <h3>State updates</h3>
          {patches.length === 0 ? (
            <p className="muted">尚无节点完成更新</p>
          ) : (
            <ul>
              {patches.map((patch) => (
                <li key={patch.id}>
                  <button
                    type="button"
                    className={selected?.id === patch.id ? "active" : ""}
                    onClick={() => setSelectedPatchId(patch.id)}
                  >
                    <strong>{patch.node}</strong>
                    <span>
                      {patch.diff.length
                        ? patch.diff.map((d) => d.field).join(", ")
                        : "无字段变化"}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </aside>

        <div className="state-detail">
          <div className="diff-block">
            <h3>
              {selected
                ? `由节点 ${selected.node} 更新的字段`
                : "等待节点更新"}
            </h3>
            {selected && selected.diff.length === 0 ? (
              <p className="muted">该步未改变受监控字段（可能只追加了 events）。</p>
            ) : null}
            {selected?.diff.map((d) => (
              <article key={`${selected.id}-${d.field}`} className="diff-card">
                <header>
                  <code>{d.field}</code>
                  <span>updated by {selected.node}</span>
                </header>
                <div className="diff-cols">
                  <pre>{preview(d.before)}</pre>
                  <pre>{preview(d.after)}</pre>
                </div>
              </article>
            ))}
          </div>

          <div className="snapshot-block">
            <h3>当前 State 快照</h3>
            <pre className="state-json">
              {currentState ? preview(currentState) : "null"}
            </pre>
          </div>
        </div>
      </div>
    </section>
  );
}
