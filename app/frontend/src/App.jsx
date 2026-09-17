import React, { useState, useEffect, useRef } from 'react';
import { MessageSquare, Database, Layers, Bot, User, Calendar, Clock, Loader2, Trash2, RefreshCw } from 'lucide-react';

const API_BASE = 'http://localhost:8000/api';

export default function App() {
  const [activeView, setActiveView]       = useState('chat');
  const [conversations, setConversations] = useState([]);
  const [memories, setMemories]           = useState([]);
  const [memoryTypes, setMemoryTypes]     = useState([]);
  const [memTypeFilter, setMemTypeFilter] = useState('all');
  const [ragEntries, setRagEntries]       = useState([]);
  const [ragFilter, setRagFilter]         = useState('all');
  const [loading, setLoading]             = useState(false);
  const [isLiveChatActive, setIsLiveChatActive] = useState(false);
  const [userInput, setUserInput]         = useState('');
  const [isAiThinking, setIsAiThinking]   = useState(false);
  const [lifetimeSummary, setLifetimeSummary] = useState('');
  const scrollRef = useRef(null);

  const scrollToBottom = () => {
    if (scrollRef.current)
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  };

  // Load conversations on mount
  useEffect(() => {
    fetch(`${API_BASE}/conversations`).then(r => r.json()).then(setConversations).catch(console.error);
    fetch(`${API_BASE}/summaries/lifetime`).then(r => r.json()).then(d => setLifetimeSummary(d.summary || '')).catch(() => {});
  }, []);

  useEffect(() => { setTimeout(scrollToBottom, 300); }, [activeView, conversations]);

  // Load SQL memories when tab opens
  useEffect(() => {
    if (activeView === 'sql') refreshMemories();
    if (activeView === 'rag') refreshRag();
  }, [activeView]);

  const refreshConversations = () =>
    fetch(`${API_BASE}/conversations`).then(r => r.json()).then(setConversations).catch(console.error);

  const refreshMemories = () => {
    setLoading(true);
    Promise.all([
      fetch(`${API_BASE}/sql/memories`).then(r => r.json()),
      fetch(`${API_BASE}/sql/types`).then(r => r.json()),
    ]).then(([mems, types]) => {
      setMemories(mems);
      setMemoryTypes(['all', ...types]);
    }).catch(console.error).finally(() => setLoading(false));
  };

  const refreshRag = () => {
    setLoading(true);
    fetch(`${API_BASE}/rag/entries`).then(r => r.json()).then(setRagEntries).catch(console.error).finally(() => setLoading(false));
  };

  const handleDeleteDay = (date) => {
    if (!window.confirm(`Delete all data for ${date}?`)) return;
    setLoading(true);
    fetch(`${API_BASE}/conversations/${date}`, { method: 'DELETE' })
      .then(() => { setConversations(prev => prev.filter(c => c.date !== date)); })
      .catch(console.error).finally(() => setLoading(false));
  };

  const handleStartConversation = () => {
    setLoading(true);
    fetch(`${API_BASE}/conversation/start`, { method: 'POST' })
      .then(r => r.json())
      .then(() => { setIsLiveChatActive(true); refreshConversations(); })
      .catch(console.error).finally(() => setLoading(false));
  };

  const handleSend = () => {
    const msg = userInput.trim();
    if (!msg || isAiThinking) return;
    setUserInput('');

    // Optimistic UI update
    setConversations(prev => {
      const updated = [...prev];
      if (updated.length > 0) {
        const last = { ...updated[updated.length - 1] };
        const inters = [...last.interactions];
        if (inters.length > 0) {
          const lastInter = { ...inters[inters.length - 1] };
          lastInter.turns = [...lastInter.turns, { speaker: 'User', text: msg }];
          inters[inters.length - 1] = lastInter;
        }
        last.interactions = inters;
        updated[updated.length - 1] = last;
      }
      return updated;
    });

    setIsAiThinking(true);
    fetch(`${API_BASE}/conversation/respond`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: msg }),
    })
      .then(r => r.json())
      .then(() => refreshConversations())
      .catch(console.error)
      .finally(() => setIsAiThinking(false));
  };

  // ── RAG identifier → readable label ──
  const ragLabel = (identifier = '') => {
    if (identifier.startsWith('week:'))     return 'weekly';
    if (identifier.startsWith('month:'))    return 'monthly';
    if (identifier.startsWith('year:'))     return 'yearly';
    if (identifier === 'lifetime')          return 'lifetime';
    if (identifier.startsWith('traj-'))     return 'trajectory';
    return 'raw';
  };

  const filteredMemories = memTypeFilter === 'all'
    ? memories
    : memories.filter(m => m.type === memTypeFilter);

  const filteredRag = ragFilter === 'all'
    ? ragEntries
    : ragEntries.filter(e => {
        const id  = e.metadata?.identifier || '';
        const lbl = ragLabel(id);
        if (ragFilter === 'raw') return e.metadata?.type === 'raw';
        return lbl === ragFilter;
      });

  return (
    <div className="app-shell">

      {/* ── TOP BAR ── */}
      <div className="top-bar">
        <div className="top-bar-nav">
          {[
            { id: 'chat', icon: <MessageSquare size={16} />, label: 'Conversations' },
            { id: 'sql',  icon: <Database size={16} />,      label: 'Memories' },
            { id: 'rag',  icon: <Layers size={16} />,        label: 'RAG Store' },
          ].map(({ id, icon, label }) => (
            <button key={id} className={`nav-btn ${activeView === id ? 'active' : ''}`} onClick={() => setActiveView(id)}>
              {icon} {label}
            </button>
          ))}
        </div>
        <div className="dataset-badge">
          <div className="pulse-dot" />
          Memory v2
        </div>
      </div>

      {/* ── MAIN ── */}
      <div className="main-content">

        {/* ===== CHAT ===== */}
        {activeView === 'chat' && (
          <div className="chat-view fade-in">

            {/* Lifetime summary banner */}
            {lifetimeSummary && (
              <div className="lifetime-banner">
                <span className="lifetime-label">Global Memory</span>
                <p>{lifetimeSummary}</p>
              </div>
            )}

            <div className="chat-messages" ref={scrollRef}>
              {conversations.map(dayData => {
                const d = new Date(dayData.date + 'T00:00:00');
                const dateLabel = d.toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric', year: 'numeric' });
                return (
                  <React.Fragment key={dayData.date}>
                    <div className="date-divider">
                      <div className="divider-line" />
                      <span className="divider-label"><Calendar size={14} /> {dateLabel}</span>
                      <button className="delete-day-btn" onClick={() => handleDeleteDay(dayData.date)} title={`Delete ${dayData.date}`}>
                        <Trash2 size={14} />
                      </button>
                      <div className="divider-line" />
                    </div>

                    {dayData.interactions.map((inter, ii) => (
                      <React.Fragment key={ii}>
                        <div className="time-divider">
                          <Clock size={12} /><span>{inter.time_of_day}</span>
                        </div>
                        {inter.turns.map((turn, ti) => {
                          const isUser = turn.speaker === 'User';
                          return (
                            <div key={ti} className={`message-row ${isUser ? 'user' : 'ai'}`}>
                              <div className={`avatar ${isUser ? 'user' : 'ai'}`}>
                                {isUser ? <User size={18} color="#0891b2" /> : <Bot size={18} color="#6366f1" />}
                              </div>
                              <div className={`bubble ${isUser ? 'user' : 'ai'}`}>{turn.text}</div>
                            </div>
                          );
                        })}
                      </React.Fragment>
                    ))}
                  </React.Fragment>
                );
              })}

              {!isLiveChatActive ? (
                <div className="integrated-start-section">
                  <div className="divider-line" />
                  <button className="start-button mini" disabled={loading} onClick={handleStartConversation}>
                    {loading ? <Loader2 size={16} className="spin" /> : <><MessageSquare size={16} /> Start Fresh Conversation</>}
                  </button>
                  <div className="divider-line" />
                </div>
              ) : (
                isAiThinking && (
                  <div className="message-row ai">
                    <div className="avatar ai"><Bot size={18} color="#6366f1" /></div>
                    <div className="bubble ai typing">
                      <span className="dot" /><span className="dot" /><span className="dot" />
                    </div>
                  </div>
                )
              )}
            </div>

            {isLiveChatActive && (
              <div className="chat-input-area sticky">
                <input
                  type="text"
                  autoFocus
                  placeholder="Continue the conversation..."
                  value={userInput}
                  onChange={e => setUserInput(e.target.value)}
                  onKeyDown={e => { if (e.key === 'Enter') handleSend(); }}
                />
              </div>
            )}
          </div>
        )}

        {/* ===== SQL MEMORIES ===== */}
        {activeView === 'sql' && (
          <div className="data-view fade-in">
            <div className="data-view-header">
              <div style={{ flex: 1 }}>
                <h2>Structured Memories</h2>
                <p>Extracted from conversations — type and entity assigned by LLM</p>
              </div>
              <button className="refresh-btn" onClick={refreshMemories} disabled={loading}>
                <RefreshCw size={14} className={loading ? 'spin' : ''} /> Refresh
              </button>
            </div>

            {/* Type filter pills */}
            <div className="filter-bar">
              {memoryTypes.map(t => (
                <button key={t} className={`filter-pill ${memTypeFilter === t ? 'active' : ''}`} onClick={() => setMemTypeFilter(t)}>
                  {t}
                </button>
              ))}
            </div>

            <div className="data-view-body">
              {loading ? (
                <div className="empty-state"><Loader2 size={32} className="spin" /><p>Loading memories...</p></div>
              ) : filteredMemories.length === 0 ? (
                <div className="empty-state"><Database size={32} /><p>No memories found</p></div>
              ) : (
                <table className="sql-table">
                  <thead>
                    <tr>
                      <th>Type</th>
                      <th>Entity</th>
                      <th>Memory</th>
                      <th style={{ textAlign: 'right' }}>Date</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredMemories.map((m, i) => (
                      <tr key={i}>
                        <td><span className="category-badge">{m.type}</span></td>
                        <td className="entity-cell">{m.entity}</td>
                        <td>{m.content}</td>
                        <td className="date-cell" style={{ textAlign: 'right' }}>{m.date}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </div>
        )}

        {/* ===== RAG STORE ===== */}
        {activeView === 'rag' && (
          <div className="data-view fade-in">
            <div className="data-view-header">
              <div style={{ flex: 1 }}>
                <h2>RAG Store</h2>
                <p>Hierarchical summaries and raw conversation chunks</p>
              </div>
              <button className="refresh-btn" onClick={refreshRag} disabled={loading}>
                <RefreshCw size={14} className={loading ? 'spin' : ''} /> Refresh
              </button>
            </div>

            <div className="filter-bar">
              {['all', 'weekly', 'monthly', 'yearly', 'lifetime', 'trajectory', 'raw'].map(f => (
                <button key={f} className={`filter-pill ${ragFilter === f ? 'active' : ''}`} onClick={() => setRagFilter(f)}>
                  {f}
                </button>
              ))}
            </div>

            <div className="data-view-body">
              {loading ? (
                <div className="empty-state"><Loader2 size={32} className="spin" style={{ color: 'var(--primary)' }} /><p>Loading RAG store...</p></div>
              ) : filteredRag.length === 0 ? (
                <div className="empty-state"><Layers size={32} /><p>No entries found</p></div>
              ) : (
                <div className="rag-grid">
                  {filteredRag.map((entry, idx) => {
                    const id    = entry.metadata?.identifier || '';
                    const label = ragLabel(id);
                    const date  = entry.metadata?.source_date || id.replace(/^(week:|month:|year:)/, '') || '';
                    return (
                      <div key={idx} className={`rag-card ${entry.metadata?.type === 'summary' ? 'summary-card' : ''}`}>
                        <div className="rag-card-header">
                          <span className="rag-date">{date}</span>
                          {label !== 'raw' && <span className="insight-badge">{label}</span>}
                        </div>
                        {entry.metadata?.title && <p className="rag-title">{entry.metadata.title}</p>}
                        <p className="rag-content">{entry.content}</p>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
