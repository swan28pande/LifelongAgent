import { useState, useEffect } from 'react';
import {
  MessageSquare, Database, Layers, FileText, CheckSquare,
  RefreshCw, Loader2, Send, AlertTriangle, ChevronRight,
} from 'lucide-react';

const API = 'http://localhost:8000/api';
const CATEGORY_NAMES = {1:'multi-hop',2:'temporal',3:'commonsense',4:'single-hop',5:'adversarial'};

const get = (path) =>
  fetch(`${API}${path}`).then(async (r) => {
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
    return r.json();
  });

export default function App() {
  const [section, setSection] = useState('benchmarks');
  const [subView, setSubView] = useState({});
  const [status, setStatus]   = useState(null);
  const [stats, setStats]     = useState(null);
  const [data, setData]       = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState('');

  const [entityFilter, setEntityFilter] = useState('all');
  const [dayFilter, setDayFilter]       = useState('all');
  const [locomoConv, setLocomoConv]     = useState(0);
  const [locomoFilter, setLocomoFilter] = useState('all');

  const [messages, setMessages] = useState([]);
  const [input, setInput]       = useState('');
  const [thinking, setThinking] = useState(false);

  const sub = (sec) => subView[sec] || { datasets: 'synthetic_data', store: 'prefs' }[sec] || null;
  const setSub = (sec, v) => setSubView((p) => ({ ...p, [sec]: v }));

  useEffect(() => { get('/status').then(setStatus).catch(() => {}); }, []);
  useEffect(() => { get('/stats').then(setStats).catch(() => {}); }, []);
  useEffect(() => {
    get('/qa').then((d) => setData((p) => ({ ...p, synthetic_eval: d }))).catch(() => {});
    get('/locomo/qa').then((d) => setData((p) => ({ ...p, locomo_eval: d }))).catch(() => {});
  }, []);

  const ENDPOINTS = {
    synthetic_data: '/dataset',
    synthetic_eval: '/qa',
    locomo_data: '/locomo/dataset',
    locomo_eval: '/locomo/qa',
    prefs: '/memories',
    chunks: '/chunks',
    summaries: '/summaries',
  };

  const activeKey = section === 'datasets' || section === 'store' ? sub(section) : null;

  useEffect(() => {
    if (!activeKey || !ENDPOINTS[activeKey] || data[activeKey]) return;
    setLoading(true); setError('');
    get(ENDPOINTS[activeKey])
      .then((d) => setData((p) => ({ ...p, [activeKey]: d })))
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [activeKey]); // eslint-disable-line react-hooks/exhaustive-deps

  const reload = () => {
    if (!activeKey || !ENDPOINTS[activeKey]) return;
    setData((p) => ({ ...p, [activeKey]: undefined }));
    setLoading(true); setError('');
    get(ENDPOINTS[activeKey])
      .then((d) => setData((p) => ({ ...p, [activeKey]: d })))
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

  const memories = data.prefs || [];
  const entities = [...new Set(memories.map((m) => m.entity))].sort();
  const shownMemories = entityFilter === 'all' ? memories : memories.filter((m) => m.entity === entityFilter);

  const days = data.synthetic_data?.days || [];
  const dates = [...new Set(days.map((d) => d.date))];
  const shownDays = dayFilter === 'all' ? days : days.filter((d) => d.date === dayFilter);

  return (
    <div className="app-shell">
      {/* ── Top bar ── */}
      <header className="top-bar">
        <span className="logo">Lifelong Agent</span>
        <nav className="top-nav">
          {['benchmarks', 'datasets', 'store', 'chat'].map((id) => (
            <button key={id} className={`nav-btn${section === id ? ' active' : ''}`}
              onClick={() => setSection(id)}>
              {id.charAt(0).toUpperCase() + id.slice(1)}
            </button>
          ))}
        </nav>
        <div className="status-badge" title={status?.store || ''}>
          <span className="status-dot" />
          {status ? `${status.run}` : '...'}
        </div>
      </header>

      <div className="main-content">
        {error && <div className="empty-state"><AlertTriangle size={24} /><p>{error}</p></div>}
        {loading && !error && <div className="empty-state"><Loader2 size={24} className="spin" /><p>Loading...</p></div>}

        {/* ═══════ BENCHMARKS ═══════ */}
        {section === 'benchmarks' && !loading && !error && (
          <div className="data-view fade-in">
            <div className="view-header">
              <div><h2>Benchmarks</h2><p className="subtitle">Performance across evaluation datasets</p></div>
            </div>
            <div className="view-body" style={{ padding: '24px 32px' }}>
              <div className="bench-grid">
                {/* Synthetic */}
                <div className="bench-card" onClick={() => { setSection('datasets'); setSub('datasets', 'synthetic_eval'); }}>
                  <div className="bench-head">
                    <span className="bench-name">Synthetic</span>
                    <ChevronRight size={14} className="bench-arrow" />
                  </div>
                  <p className="bench-desc">60 days, 67 QA pairs &middot; 7-day cycle, 3-day alternation, day-of-week rule</p>
                  {data.synthetic_eval?.summary ? (
                    <div className="bench-metrics">
                      <div className="metric"><span className="metric-val">{data.synthetic_eval.summary.overall_f1?.toFixed(3)}</span><span className="metric-lbl">Token F1</span></div>
                      <div className="metric"><span className="metric-val">{data.synthetic_eval.summary.overall_llm?.toFixed(3)}</span><span className="metric-lbl">LLM Judge</span></div>
                      <div className="metric"><span className="metric-val">{data.synthetic_eval.summary.n}</span><span className="metric-lbl">Questions</span></div>
                      {data.synthetic_eval.summary.by_type && Object.entries(data.synthetic_eval.summary.by_type).map(([t, v]) => (
                        <div key={t} className="metric sm"><span className="metric-val">{v.llm?.toFixed(2)}</span><span className="metric-lbl">{t} (n={v.n})</span></div>
                      ))}
                    </div>
                  ) : <p className="bench-empty">No results yet</p>}
                </div>

                {/* LoCoMo */}
                <div className="bench-card" onClick={() => { setSection('datasets'); setSub('datasets', 'locomo_eval'); }}>
                  <div className="bench-head">
                    <span className="bench-name">LoCoMo</span>
                    <ChevronRight size={14} className="bench-arrow" />
                  </div>
                  <p className="bench-desc">10 conversations, 1986 QA pairs &middot; multi-hop, temporal, single-hop, commonsense, adversarial</p>
                  {data.locomo_eval?.summary?.n ? (
                    <div className="bench-metrics">
                      <div className="metric"><span className="metric-val">{data.locomo_eval.summary.overall?.toFixed(3)}</span><span className="metric-lbl">Overall F1</span></div>
                      <div className="metric"><span className="metric-val">{data.locomo_eval.summary.n}</span><span className="metric-lbl">Questions</span></div>
                      {data.locomo_eval.summary.by_category && Object.entries(data.locomo_eval.summary.by_category).map(([cat, v]) => (
                        <div key={cat} className="metric sm"><span className="metric-val">{(v.score ?? v.overall)?.toFixed(3)}</span><span className="metric-lbl">{cat}{v.n != null ? ` (${v.n})` : ''}</span></div>
                      ))}
                    </div>
                  ) : <p className="bench-empty">No results yet</p>}
                  <div className="bench-baseline">
                    <span className="baseline-lbl">GPT-3.5 baseline:</span>
                    <span className="pill">overall ~0.27</span>
                    <span className="pill">single-hop ~0.35</span>
                    <span className="pill">temporal ~0.22</span>
                    <span className="pill">multi-hop ~0.19</span>
                  </div>
                </div>

                {/* Store stats */}
                {stats && (
                  <div className="bench-card no-click">
                    <div className="bench-head"><span className="bench-name">Current Store</span></div>
                    <p className="bench-desc">{status?.store || 'No store loaded'}</p>
                    <div className="bench-metrics">
                      <div className="metric"><span className="metric-val">{stats.preferences}</span><span className="metric-lbl">Preferences</span></div>
                      <div className="metric"><span className="metric-val">{stats.chunks}</span><span className="metric-lbl">Chunks</span></div>
                      <div className="metric"><span className="metric-val">{stats.summaries}</span><span className="metric-lbl">Summaries</span></div>
                      <div className="metric"><span className="metric-val">{stats.date_range?.[0]} &rarr; {stats.date_range?.[1]}</span><span className="metric-lbl">Date range</span></div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* ═══════ DATASETS ═══════ */}
        {section === 'datasets' && !loading && !error && (() => {
          const s = sub('datasets') || 'synthetic_data';
          return (
            <div className="data-view fade-in">
              <div className="view-header">
                <div>
                  <h2>Datasets</h2>
                  <div className="sub-nav">
                    {[
                      ['synthetic_data', 'Synthetic'],
                      ['synthetic_eval', 'Synthetic Eval'],
                      ['locomo_data', 'LoCoMo'],
                      ['locomo_eval', 'LoCoMo Eval'],
                    ].map(([id, label]) => (
                      <button key={id} className={`sub-btn${s === id ? ' active' : ''}`}
                        onClick={() => setSub('datasets', id)}>{label}</button>
                    ))}
                  </div>
                </div>
                <button className="refresh-btn" onClick={reload}><RefreshCw size={14} /></button>
              </div>

              {/* Synthetic conversations */}
              {s === 'synthetic_data' && (
                <>
                  <div className="filter-bar">
                    {['all', ...dates].map((d) => (
                      <button key={d} className={`filter-pill${dayFilter === d ? ' active' : ''}`}
                        onClick={() => setDayFilter(d)}>
                        {d === 'all' ? `All (${dates.length})` : d}
                      </button>
                    ))}
                  </div>
                  <div className="view-body">
                    {shownDays.map((day, i) => (
                      <div key={i} className="card" style={{ margin: '10px 24px' }}>
                        <div className="card-head">
                          <span className="card-title">{day.date}</span>
                          <span className="card-meta">{day.time_of_day} &middot; {day.turns.length} turns</span>
                        </div>
                        <div className="card-body">
                          {day.turns.map((t, j) => (
                            <div key={j} className="turn"><b>{t.speaker}:</b> {t.text}</div>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </>
              )}

              {/* Synthetic eval */}
              {s === 'synthetic_eval' && (
                <>
                  {data.synthetic_eval?.summary?.by_type && (
                    <div className="filter-bar">
                      {Object.entries(data.synthetic_eval.summary.by_type).map(([t, v]) => (
                        <span key={t} className="pill">{t}: {v.llm?.toFixed(2)} (n={v.n})</span>
                      ))}
                    </div>
                  )}
                  <div className="view-body">
                    <table className="tbl">
                      <thead><tr><th>Type</th><th>Question</th><th>Expected</th><th>Answered</th><th>Judge</th></tr></thead>
                      <tbody>
                        {(data.synthetic_eval?.pairs || []).map((p, i) => (
                          <tr key={i}>
                            <td><span className="badge">{p.type}</span></td>
                            <td>{p.question}</td>
                            <td>{p.answer}</td>
                            <td>{p.response ?? <em className="muted">not run</em>}</td>
                            <td>{p.llm === undefined ? '—' : p.llm === 1 ? '✓' : '✗'}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </>
              )}

              {/* LoCoMo conversations */}
              {s === 'locomo_data' && (() => {
                const convs = data.locomo_data?.conversations || [];
                const conv = convs[locomoConv];
                return (
                  <>
                    <div className="filter-bar">
                      {convs.map((c) => (
                        <button key={c.index} className={`filter-pill${locomoConv === c.index ? ' active' : ''}`}
                          onClick={() => setLocomoConv(c.index)}>
                          {c.sample_id}
                        </button>
                      ))}
                    </div>
                    {conv && (
                      <div className="view-body">
                        <div className="filter-bar" style={{ margin: '10px 24px' }}>
                          <span className="pill">{conv.sessions.length} sessions</span>
                          <span className="pill">{conv.n_questions} questions</span>
                          <span className="pill">{conv.speaker_a} & {conv.speaker_b}</span>
                        </div>
                        {conv.sessions.map((ses, i) => (
                          <div key={i} className="card" style={{ margin: '10px 24px' }}>
                            <div className="card-head">
                              <span className="card-title">{ses.key}</span>
                              <span className="card-meta">{ses.date_raw} &middot; {ses.turns.length} turns</span>
                            </div>
                            <div className="card-body">
                              {ses.turns.map((t, j) => (
                                <div key={j} className="turn"><b>{t.speaker}:</b> {t.text}</div>
                              ))}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </>
                );
              })()}

              {/* LoCoMo eval */}
              {s === 'locomo_eval' && (() => {
                const le = data.locomo_eval;
                if (!le) return null;
                const records = le.records || [];
                const isV3 = records[0]?.response !== undefined;
                const systems = !isV3 && records[0]
                  ? Object.keys(records[0]).filter((k) =>
                      !['conv_id','category','question','answer','sample_id','category_name'].includes(k)
                      && typeof records[0][k] === 'object')
                  : [];
                const catSummary = isV3 ? le.summary?.by_category : le.summary?.[systems[systems.length - 1]]?.by_category;
                const catNames = [...new Set(records.map((r) => r.category_name || CATEGORY_NAMES[r.category] || String(r.category)))];

                return (
                  <>
                    {catSummary && (
                      <div className="filter-bar">
                        {Object.entries(catSummary).map(([cat, v]) => (
                          <span key={cat} className="pill">{cat}: {(v.score ?? v.overall)?.toFixed?.(3) ?? v}{v.n != null ? ` (${v.n})` : ''}</span>
                        ))}
                      </div>
                    )}
                    <div className="filter-bar">
                      {['all', ...catNames].map((f) => (
                        <button key={f} className={`filter-pill${locomoFilter === f ? ' active' : ''}`}
                          onClick={() => setLocomoFilter(f)}>
                          {f === 'all' ? `All (${records.length})` : f}
                        </button>
                      ))}
                    </div>
                    <div className="view-body">
                      <table className="tbl">
                        <thead>
                          <tr>
                            <th>Conv</th><th>Category</th><th>Question</th><th>Expected</th>
                            {isV3 ? <><th>Response</th><th>F1</th></> : systems.map((sy) => <th key={sy}>{sy}</th>)}
                          </tr>
                        </thead>
                        <tbody>
                          {records
                            .filter((r) => {
                              const cn = r.category_name || CATEGORY_NAMES[r.category] || String(r.category);
                              return locomoFilter === 'all' || cn === locomoFilter;
                            })
                            .map((r, i) => {
                              const cn = r.category_name || CATEGORY_NAMES[r.category] || String(r.category);
                              const best = isV3 ? r.score : Math.max(...systems.map((sy) => r[sy]?.score ?? 0));
                              return (
                                <tr key={i} className={best > 0.5 ? 'row-good' : best > 0.2 ? 'row-mid' : 'row-bad'}>
                                  <td className="mono">{r.sample_id || r.conv_id}</td>
                                  <td><span className="badge">{cn}</span></td>
                                  <td>{r.question}</td>
                                  <td>{String(r.answer)}</td>
                                  {isV3 ? (
                                    <>
                                      <td>{r.response}</td>
                                      <td className="mono">{r.score?.toFixed(3)}</td>
                                    </>
                                  ) : systems.map((sy) => (
                                    <td key={sy} title={r[sy]?.response || ''} className="mono">{r[sy]?.score?.toFixed(3) ?? '—'}</td>
                                  ))}
                                </tr>
                              );
                            })}
                        </tbody>
                      </table>
                    </div>
                  </>
                );
              })()}
            </div>
          );
        })()}

        {/* ═══════ STORE ═══════ */}
        {section === 'store' && !loading && !error && (() => {
          const s = sub('store') || 'prefs';
          return (
            <div className="data-view fade-in">
              <div className="view-header">
                <div>
                  <h2>Memory Store</h2>
                  <div className="sub-nav">
                    {[['prefs','Preferences'],['chunks','Chunks'],['summaries','Summaries']].map(([id, label]) => (
                      <button key={id} className={`sub-btn${s === id ? ' active' : ''}`}
                        onClick={() => setSub('store', id)}>{label}</button>
                    ))}
                  </div>
                </div>
                <button className="refresh-btn" onClick={reload}><RefreshCw size={14} /></button>
              </div>

              {s === 'prefs' && (
                <>
                  <div className="filter-bar">
                    {['all', ...entities].map((e) => (
                      <button key={e} className={`filter-pill${entityFilter === e ? ' active' : ''}`}
                        onClick={() => setEntityFilter(e)}>
                        {e === 'all' ? `All (${memories.length})` : `${e} (${memories.filter((m) => m.entity === e).length})`}
                      </button>
                    ))}
                  </div>
                  <div className="view-body">
                    {shownMemories.length === 0
                      ? <div className="empty-state"><Database size={24} /><p>No preferences stored</p></div>
                      : (
                        <table className="tbl">
                          <thead><tr><th>Date</th><th>Entity</th><th>Value</th><th>Speaker</th></tr></thead>
                          <tbody>
                            {shownMemories.map((m) => (
                              <tr key={m.id}>
                                <td className="mono">{m.date}</td>
                                <td><span className="badge">{m.entity}</span></td>
                                <td>{m.content}</td>
                                <td className="muted">{m.speaker}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      )}
                  </div>
                </>
              )}

              {s === 'chunks' && (
                <div className="view-body">
                  <div className="card-grid">
                    {(data.chunks || []).map((c, i) => (
                      <div key={i} className="card">
                        <div className="card-head"><span className="card-title">{c.date}</span></div>
                        <div className="card-body">{c.content}</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {s === 'summaries' && (
                <div className="view-body">
                  <div className="card-grid">
                    {(data.summaries || []).map((sm, i) => (
                      <div key={i} className="card">
                        <div className="card-head">
                          <span className="card-title">{sm.title || sm.identifier}</span>
                          <span className="badge">{sm.level}</span>
                        </div>
                        <div className="card-body">{sm.content}</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          );
        })()}

        {/* ═══════ CHAT ═══════ */}
        {section === 'chat' && (
          <div className="chat-view fade-in">
            <div className="chat-messages">
              {messages.length === 0 && (
                <div className="empty-state"><MessageSquare size={24} /><p>Ask the agent about stored memory</p></div>
              )}
              {messages.map((m, i) => (
                <div key={i} className={`msg ${m.role}`}><div className="bubble">{m.text}</div></div>
              ))}
              {thinking && <div className="msg ai"><div className="bubble"><Loader2 size={14} className="spin" /> retrieving...</div></div>}
            </div>
            <div className="chat-input">
              <input value={input} placeholder="Ask about a preference, a date, or a pattern..."
                onChange={(e) => setInput(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && send()} />
              <button className="send-btn" onClick={send} disabled={thinking}><Send size={14} /></button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
