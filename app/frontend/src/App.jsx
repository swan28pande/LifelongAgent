import { useState, useEffect } from 'react';
import {
  MessageSquare, Database, Layers, FileText, CheckSquare,
  RefreshCw, Loader2, Send, AlertTriangle, Users, BookOpen,
} from 'lucide-react';

const API = 'http://localhost:8000/api';

const VIEWS = [
  { id: 'dataset',       icon: <FileText size={16} />,      label: 'Dataset' },
  { id: 'memories',      icon: <Database size={16} />,      label: 'Preferences' },
  { id: 'chunks',        icon: <Layers size={16} />,        label: 'RAG Store' },
  { id: 'summaries',     icon: <FileText size={16} />,      label: 'Summaries' },
  { id: 'qa',            icon: <CheckSquare size={16} />,   label: 'Evaluation' },
  { id: 'locomo',        icon: <Users size={16} />,         label: 'LoCoMo' },
  { id: 'locomo_eval',   icon: <BookOpen size={16} />,      label: 'LoCoMo Eval' },
  { id: 'chat',          icon: <MessageSquare size={16} />, label: 'Chat' },
];

const get = (path) =>
  fetch(`${API}${path}`).then(async (r) => {
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
    return r.json();
  });

export default function App() {
  const [view, setView]       = useState('dataset');
  const [status, setStatus]   = useState(null);
  const [stats, setStats]     = useState(null);
  const [data, setData]       = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState('');

  const [entityFilter, setEntityFilter] = useState('all');
  const [dayFilter, setDayFilter]       = useState('all');
  const [messages, setMessages]         = useState([]);
  const [input, setInput]               = useState('');
  const [thinking, setThinking]         = useState(false);
  const [locomoConv, setLocomoConv]     = useState(0);
  const [locomoFilter, setLocomoFilter] = useState('all');

  useEffect(() => { get('/status').then(setStatus).catch(() => {}); }, []);

  // Each view owns one endpoint; cache so switching tabs doesn't refetch.
  useEffect(() => {
    const endpoint = {
      dataset: '/dataset', memories: '/memories',
      chunks: '/chunks', summaries: '/summaries', qa: '/qa',
      locomo: '/locomo/dataset', locomo_eval: '/locomo/qa',
    }[view];
    if (!endpoint || data[view]) return;

    setLoading(true); setError('');
    get(endpoint)
      .then((d) => setData((prev) => ({ ...prev, [view]: d })))
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [view]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (view !== 'dataset' || stats) return;
    get('/stats').then(setStats).catch(() => {});
  }, [view]); // eslint-disable-line react-hooks/exhaustive-deps

  const reload = () => {
    setData((prev) => ({ ...prev, [view]: undefined }));
    setStats(null);
    setView(view);
    const endpoint = {
      dataset: '/dataset', memories: '/memories',
      chunks: '/chunks', summaries: '/summaries', qa: '/qa',
      locomo: '/locomo/dataset', locomo_eval: '/locomo/qa',
    }[view];
    if (!endpoint) return;
    setLoading(true); setError('');
    get(endpoint)
      .then((d) => setData((prev) => ({ ...prev, [view]: d })))
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  };

  const send = () => {
    const text = input.trim();
    if (!text || thinking) return;
    setMessages((m) => [...m, { role: 'user', text }]);
    setInput(''); setThinking(true);

    fetch(`${API}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text }),
    })
      .then((r) => r.json())
      .then((d) => setMessages((m) => [...m, { role: 'ai', text: d.response || d.detail }]))
      .catch((e) => setMessages((m) => [...m, { role: 'ai', text: `Error: ${e.message}` }]))
      .finally(() => setThinking(false));
  };

  const memories = data.memories || [];
  const entities = [...new Set(memories.map((m) => m.entity))].sort();
  const shownMemories = entityFilter === 'all'
    ? memories : memories.filter((m) => m.entity === entityFilter);

  const days = data.dataset?.days || [];
  const dates = [...new Set(days.map((d) => d.date))];
  const shownDays = dayFilter === 'all' ? days : days.filter((d) => d.date === dayFilter);

  return (
    <div className="app-shell">
      <div className="top-bar">
        <div className="top-bar-nav">
          {VIEWS.map(({ id, icon, label }) => (
            <button key={id}
              className={`nav-btn ${view === id ? 'active' : ''}`}
              onClick={() => setView(id)}>
              {icon} {label}
            </button>
          ))}
        </div>
        <div className="dataset-badge" title={status?.store || ''}>
          <div className="pulse-dot" />
          {status ? `${status.system} · ${status.run}` : 'connecting…'}
        </div>
      </div>

      <div className="main-content">
        {error && (
          <div className="empty-state">
            <AlertTriangle size={32} />
            <p>{error}</p>
          </div>
        )}

        {loading && !error && (
          <div className="empty-state">
            <Loader2 size={32} className="spin" />
            <p>Loading… (first call opens the store, which takes a moment)</p>
          </div>
        )}

        {/* ===== DATASET ===== */}
        {view === 'dataset' && !loading && !error && (
          <div className="data-view fade-in">
            <div className="data-view-header">
              <div style={{ flex: 1 }}>
                <h2>Source conversations</h2>
                <p>{days.length} sessions from {status?.dataset}</p>
              </div>
              <button className="refresh-btn" onClick={reload}>
                <RefreshCw size={14} /> Refresh
              </button>
            </div>

            {stats && (
              <div className="filter-bar">
                <span className="bucket-pill">{stats.preferences} preferences</span>
                <span className="bucket-pill">{stats.chunks} chunks</span>
                <span className="bucket-pill">{stats.summaries} summaries</span>
                <span className="bucket-pill">
                  {stats.date_range?.[0]} → {stats.date_range?.[1]}
                </span>
              </div>
            )}

            <div className="filter-bar">
              {['all', ...dates].map((d) => (
                <button key={d}
                  className={`filter-pill ${dayFilter === d ? 'active' : ''}`}
                  onClick={() => setDayFilter(d)}>
                  {d === 'all' ? `All ${dates.length} days` : d}
                </button>
              ))}
            </div>

            <div className="data-view-body">
              {shownDays.map((day, i) => (
                <div key={i} className="rag-card">
                  <div className="rag-card-header">
                    <span className="rag-title">{day.date}</span>
                    <span className="rag-card-meta">
                      {day.time_of_day} · {day.turns.length} turns
                    </span>
                  </div>
                  <div className="rag-card-content">
                    {day.turns.map((t, j) => (
                      <div key={j} style={{ marginBottom: 6 }}>
                        <b>{t.speaker}:</b> {t.text}
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ===== PREFERENCES ===== */}
        {view === 'memories' && !loading && !error && (
          <div className="data-view fade-in">
            <div className="data-view-header">
              <div style={{ flex: 1 }}>
                <h2>Structured preferences</h2>
                <p>{memories.length} rows · entity and value discovered by the model</p>
              </div>
              <button className="refresh-btn" onClick={reload}>
                <RefreshCw size={14} /> Refresh
              </button>
            </div>

            <div className="filter-bar">
              {['all', ...entities].map((e) => (
                <button key={e}
                  className={`filter-pill ${entityFilter === e ? 'active' : ''}`}
                  onClick={() => setEntityFilter(e)}>
                  {e === 'all'
                    ? `All (${memories.length})`
                    : `${e} (${memories.filter((m) => m.entity === e).length})`}
                </button>
              ))}
            </div>

            <div className="data-view-body">
              {shownMemories.length === 0 ? (
                <div className="empty-state"><Database size={32} /><p>No preferences stored</p></div>
              ) : (
                <table className="sql-table">
                  <thead>
                    <tr>
                      <th>Date</th><th>Entity</th><th>Value</th><th>Speaker</th>
                    </tr>
                  </thead>
                  <tbody>
                    {shownMemories.map((m) => (
                      <tr key={m.id}>
                        <td className="date-cell">{m.date}</td>
                        <td><span className="category-badge">{m.entity}</span></td>
                        <td>{m.content}</td>
                        <td className="rag-card-meta">{m.speaker}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </div>
        )}

        {/* ===== RAG STORE ===== */}
        {view === 'chunks' && !loading && !error && (
          <div className="data-view fade-in">
            <div className="data-view-header">
              <div style={{ flex: 1 }}>
                <h2>RAG store</h2>
                <p>{(data.chunks || []).length} indexed conversation chunks</p>
              </div>
              <button className="refresh-btn" onClick={reload}>
                <RefreshCw size={14} /> Refresh
              </button>
            </div>
            <div className="data-view-body">
              <div className="rag-grid">
                {(data.chunks || []).map((c, i) => (
                  <div key={i} className="rag-card">
                    <div className="rag-card-header">
                      <span className="rag-title">{c.date}</span>
                    </div>
                    <div className="rag-card-content">{c.content}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* ===== SUMMARIES ===== */}
        {view === 'summaries' && !loading && !error && (
          <div className="data-view fade-in">
            <div className="data-view-header">
              <div style={{ flex: 1 }}>
                <h2>Summary hierarchy</h2>
                <p>{(data.summaries || []).length} summaries · weekly → monthly → yearly → lifetime</p>
              </div>
              <button className="refresh-btn" onClick={reload}>
                <RefreshCw size={14} /> Refresh
              </button>
            </div>
            <div className="data-view-body">
              <div className="rag-grid">
                {(data.summaries || []).map((s, i) => (
                  <div key={i} className="rag-card">
                    <div className="rag-card-header">
                      <span className="rag-title">{s.title || s.identifier}</span>
                      <span className="category-badge">{s.level}</span>
                    </div>
                    <div className="rag-card-content">{s.content}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* ===== EVALUATION ===== */}
        {view === 'qa' && !loading && !error && (
          <div className="data-view fade-in">
            <div className="data-view-header">
              <div style={{ flex: 1 }}>
                <h2>Evaluation</h2>
                <p>
                  {data.qa?.summary
                    ? `${data.qa.summary.n} scored · F1 ${data.qa.summary.overall_f1.toFixed(3)} · judge ${data.qa.summary.overall_llm.toFixed(3)}`
                    : 'Questions not yet scored — run scripts/run_memory_v3.py'}
                </p>
              </div>
              <button className="refresh-btn" onClick={reload}>
                <RefreshCw size={14} /> Refresh
              </button>
            </div>

            {data.qa?.summary?.by_type && (
              <div className="filter-bar">
                {Object.entries(data.qa.summary.by_type).map(([t, v]) => (
                  <span key={t} className="bucket-pill">
                    {t}: {v.llm.toFixed(2)} (n={v.n})
                  </span>
                ))}
              </div>
            )}

            <div className="data-view-body">
              <table className="sql-table">
                <thead>
                  <tr>
                    <th>Type</th><th>Question</th><th>Expected</th>
                    <th>Answered</th><th>Judge</th>
                  </tr>
                </thead>
                <tbody>
                  {(data.qa?.pairs || []).map((p, i) => (
                    <tr key={i}>
                      <td><span className="category-badge">{p.type}</span></td>
                      <td>{p.question}</td>
                      <td>{p.answer}</td>
                      <td>{p.response ?? <em style={{ opacity: 0.5 }}>not run</em>}</td>
                      <td>
                        {p.llm === undefined ? '—'
                          : p.llm === 1 ? '✓' : '✗'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* ===== LOCOMO DATASET ===== */}
        {view === 'locomo' && !loading && !error && (
          <div className="data-view fade-in">
            <div className="data-view-header">
              <div style={{ flex: 1 }}>
                <h2>LoCoMo Dataset</h2>
                <p>{(data.locomo?.conversations || []).length} conversations · multi-speaker life events</p>
              </div>
              <button className="refresh-btn" onClick={reload}>
                <RefreshCw size={14} /> Refresh
              </button>
            </div>

            <div className="filter-bar">
              {(data.locomo?.conversations || []).map((c) => (
                <button key={c.index}
                  className={`filter-pill ${locomoConv === c.index ? 'active' : ''}`}
                  onClick={() => setLocomoConv(c.index)}>
                  {c.sample_id} ({c.speaker_a} & {c.speaker_b})
                </button>
              ))}
            </div>

            {(() => {
              const conv = (data.locomo?.conversations || [])[locomoConv];
              if (!conv) return null;
              return (
                <div className="data-view-body">
                  <div className="filter-bar">
                    <span className="bucket-pill">{conv.sessions.length} sessions</span>
                    <span className="bucket-pill">{conv.n_questions} questions</span>
                    <span className="bucket-pill">{conv.speaker_a} & {conv.speaker_b}</span>
                  </div>
                  {conv.sessions.map((s, i) => (
                    <div key={i} className="rag-card">
                      <div className="rag-card-header">
                        <span className="rag-title">{s.key}</span>
                        <span className="rag-card-meta">
                          {s.date_raw} · {s.turns.length} turns
                        </span>
                      </div>
                      <div className="rag-card-content">
                        {s.turns.map((t, j) => (
                          <div key={j} style={{ marginBottom: 6 }}>
                            <b>{t.speaker}:</b> {t.text}
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              );
            })()}
          </div>
        )}

        {/* ===== LOCOMO EVAL ===== */}
        {view === 'locomo_eval' && !loading && !error && (
          <div className="data-view fade-in">
            <div className="data-view-header">
              <div style={{ flex: 1 }}>
                <h2>LoCoMo Evaluation</h2>
                <p>
                  {data.locomo_eval?.summary?.n
                    ? `${data.locomo_eval.summary.n} scored · overall ${data.locomo_eval.summary.overall?.toFixed(3)} · ${data.locomo_eval.run}`
                    : 'No LoCoMo results found — run scripts/run_locomo_v3.py'}
                </p>
              </div>
              <button className="refresh-btn" onClick={reload}>
                <RefreshCw size={14} /> Refresh
              </button>
            </div>

            {data.locomo_eval?.summary?.by_category && (
              <div className="filter-bar">
                {Object.entries(data.locomo_eval.summary.by_category).map(([cat, v]) => (
                  <span key={cat} className="bucket-pill">
                    {cat}: {v.score?.toFixed(3)} (n={v.n})
                  </span>
                ))}
              </div>
            )}

            {data.locomo_eval?.records && (
              <>
                <div className="filter-bar">
                  {['all', ...new Set(data.locomo_eval.records.map((r) => r.category_name))].map((f) => (
                    <button key={f}
                      className={`filter-pill ${locomoFilter === f ? 'active' : ''}`}
                      onClick={() => setLocomoFilter(f)}>
                      {f === 'all' ? `All (${data.locomo_eval.records.length})` : f}
                    </button>
                  ))}
                </div>

                <div className="data-view-body">
                  <table className="sql-table">
                    <thead>
                      <tr>
                        <th>Conv</th><th>Category</th><th>Question</th>
                        <th>Expected</th><th>Response</th><th>F1</th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.locomo_eval.records
                        .filter((r) => locomoFilter === 'all' || r.category_name === locomoFilter)
                        .map((r, i) => (
                          <tr key={i} style={{
                            background: r.score > 0.5 ? 'rgba(34,197,94,0.08)'
                              : r.score > 0.2 ? 'rgba(234,179,8,0.08)'
                              : 'rgba(239,68,68,0.06)',
                          }}>
                            <td className="date-cell">{r.sample_id}</td>
                            <td><span className="category-badge">{r.category_name}</span></td>
                            <td>{r.question}</td>
                            <td>{String(r.answer)}</td>
                            <td>{r.response}</td>
                            <td style={{ fontVariantNumeric: 'tabular-nums' }}>
                              {r.score?.toFixed(3)}
                            </td>
                          </tr>
                        ))}
                    </tbody>
                  </table>
                </div>
              </>
            )}
          </div>
        )}

        {/* ===== CHAT ===== */}
        {view === 'chat' && (
          <div className="chat-view fade-in">
            <div className="chat-messages">
              {messages.length === 0 && (
                <div className="empty-state">
                  <MessageSquare size={32} />
                  <p>Ask the agent about the stored memory.</p>
                </div>
              )}
              {messages.map((m, i) => (
                <div key={i} className={`message-row ${m.role}`}>
                  <div className="bubble">{m.text}</div>
                </div>
              ))}
              {thinking && (
                <div className="message-row ai">
                  <div className="bubble"><Loader2 size={14} className="spin" /> retrieving…</div>
                </div>
              )}
            </div>
            <div className="chat-input-area">
              <input
                value={input}
                placeholder="Ask about a preference, a date, or a pattern…"
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && send()}
              />
              <button className="generate-btn" onClick={send} disabled={thinking}>
                <Send size={14} /> Send
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
