import React, { useState, useEffect, useRef } from 'react';
import { MessageSquare, Database, Layers, Bot, User, Calendar, Clock, Sparkles, Loader2, Trash2 } from 'lucide-react';

const API_BASE = 'http://localhost:8000/api';

export default function App() {
  const [activeView, setActiveView] = useState('chat');
  const [conversations, setConversations] = useState([]);
  const [sqlPrefs, setSqlPrefs] = useState([]);
  const [sqlTasks, setSqlTasks] = useState([]);
  const [sqlTab, setSqlTab] = useState('preferences');
  const [ragEntries, setRagEntries] = useState([]);
  const [ragFilter, setRagFilter] = useState('all');
  const [loading, setLoading] = useState(false);
  const [isLiveChatActive, setIsLiveChatActive] = useState(false);
  const [userInput, setUserInput] = useState('');
  const [isAiThinking, setIsAiThinking] = useState(false);
  const [pendingBucket, setPendingBucket] = useState([]);
  const scrollRef = useRef(null);

  const scrollToBottom = () => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  };

  useEffect(() => {
    fetch(`${API_BASE}/conversations`)
      .then(r => r.json())
      .then(setConversations)
      .catch(console.error);
  }, []);

  useEffect(() => {
    if (activeView === 'chat' || conversations.length > 0) {
      // Small delay to ensure DOM has rendered
      setTimeout(scrollToBottom, 300);
    }
  }, [activeView, conversations]);

  useEffect(() => {
    if (activeView === 'sql') {
      setLoading(true);
      Promise.all([
        fetch(`${API_BASE}/sql/preferences`).then(r => r.json()),
        fetch(`${API_BASE}/sql/tasks`).then(r => r.json())
      ]).then(([prefs, tasks]) => {
        setSqlPrefs(prefs);
        setSqlTasks(tasks);
      }).catch(console.error).finally(() => setLoading(false));
    }
    if (activeView === 'rag') {
      refreshRag();
    }
  }, [activeView]);

  const refreshSql = () => {
    setLoading(true);
    Promise.all([
      fetch(`${API_BASE}/sql/preferences`).then(r => r.json()),
      fetch(`${API_BASE}/sql/tasks`).then(r => r.json())
    ]).then(([prefs, tasks]) => {
      setSqlPrefs(prefs);
      setSqlTasks(tasks);
    }).catch(console.error).finally(() => setLoading(false));
  };

  const refreshRag = () => {
    setLoading(true);
    fetch(`${API_BASE}/rag/entries`)
      .then(r => r.json())
      .then(setRagEntries)
      .catch(console.error)
      .finally(() => setLoading(false));
  };

  const handleDeleteDay = (date) => {
    if (window.confirm(`Are you sure you want to delete all data for ${date}? This will remove conversations, preferences, and RAG context.`)) {
      setLoading(true);
      fetch(`${API_BASE}/conversations/${date}`, { method: 'DELETE' })
        .then(r => r.json())
        .then(() => {
          // Remove from local state
          setConversations(prev => prev.filter(c => c.date !== date));
          // Refresh background memory
          refreshSql();
          refreshRag();
        })
        .catch(console.error)
        .finally(() => setLoading(false));
    }
  };

  return (
    <div className="app-shell">
      {/* ===== TOP BAR ===== */}
      <div className="top-bar">

        <div className="top-bar-nav">
          <button
            className={`nav-btn ${activeView === 'chat' ? 'active' : ''}`}
            onClick={() => setActiveView('chat')}
          >
            <MessageSquare size={16} /> Conversations
          </button>
          <button
            className={`nav-btn ${activeView === 'sql' ? 'active' : ''}`}
            onClick={() => setActiveView('sql')}
          >
            <Database size={16} /> SQL Memory
          </button>
          <button
            className={`nav-btn ${activeView === 'rag' ? 'active' : ''}`}
            onClick={() => setActiveView('rag')}
          >
            <Layers size={16} /> RAG Memory
          </button>
        </div>

        <div className="dataset-badge">
          <div className="pulse-dot" />
          Dataset 3
        </div>
      </div>

      {/* ===== MAIN CONTENT ===== */}
      <div className="main-content">
        {activeView === 'chat' && (
          <div className="chat-view fade-in">
            {/* All messages in one scrollable feed */}
            <div className="chat-messages" ref={scrollRef}>
              {conversations.map((dayData) => {
                const d = new Date(dayData.date + 'T00:00:00');
                const dateLabel = d.toLocaleDateString('en-US', {
                  weekday: 'long', month: 'long', day: 'numeric', year: 'numeric'
                });
                return (
                  <React.Fragment key={dayData.date}>
                    {/* Date divider */}
                    <div className="date-divider">
                      <div className="divider-line" />
                      <span className="divider-label">
                        <Calendar size={14} /> {dateLabel}
                      </span>
                      <button
                        className="delete-day-btn"
                        onClick={() => handleDeleteDay(dayData.date)}
                        title={`Delete all data for ${dayData.date}`}
                      >
                        <Trash2 size={14} />
                      </button>
                      <div className="divider-line" />
                    </div>

                    {dayData.interactions.map((inter, ii) => (
                      <React.Fragment key={ii}>
                        {/* Time-of-day sub-header */}
                        <div className="time-divider">
                          <Clock size={12} />
                          <span>{inter.time_of_day}</span>
                        </div>

                        {inter.turns.map((turn, ti) => {
                          const isUser = turn.speaker === 'User';
                          return (
                            <div key={ti} className={`message-row ${isUser ? 'user' : 'ai'}`}>
                              <div className={`avatar ${isUser ? 'user' : 'ai'}`}>
                                {isUser
                                  ? <User size={18} color="#0891b2" />
                                  : <Bot size={18} color="#6366f1" />
                                }
                              </div>
                              <div className={`bubble ${isUser ? 'user' : 'ai'}`}>
                                {turn.text}
                              </div>
                            </div>
                          );
                        })}
                      </React.Fragment>
                    ))}
                  </React.Fragment>
                );
              })}

              {/* ===== INTEGRATED LIVE CHAT ===== */}
              {!isLiveChatActive ? (
                <div className="integrated-start-section">
                  <div className="divider-line" />
                  <button
                    className="start-button mini"
                    disabled={loading}
                    onClick={() => {
                      setLoading(true);
                      fetch(`${API_BASE}/conversation/start`, { method: 'POST' })
                        .then(r => r.json())
                        .then(data => {
                          setPendingBucket(data.pending_bucket || []);
                          setIsLiveChatActive(true);
                          // Refresh conversations list to show "Today" immediately
                          fetch(`${API_BASE}/conversations`)
                            .then(r => r.json())
                            .then(setConversations);
                        })
                        .catch(console.error)
                        .finally(() => setLoading(false));
                    }}
                  >
                    {loading ? <Loader2 size={16} className="spin" /> : <><MessageSquare size={16} /> Start Fresh Conversation</>}
                  </button>
                  <div className="divider-line" />
                </div>
              ) : (
                <div className="live-session-wrapper">
                  {/* BUCKET VISUALIZATION */}
                  {pendingBucket.length > 0 && (
                    <div className="bucket-area animate-in">
                      <div className="bucket-header">
                        <Layers size={14} />
                        <span>Preference Bucket</span>
                      </div>
                      <div className="bucket-pills">
                        {pendingBucket.map((item, idx) => (
                          <div key={idx} className="bucket-pill">
                            {item}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {isAiThinking && (
                    <div className="message-row ai">
                      <div className="avatar ai"><Bot size={18} color="#6366f1" /></div>
                      <div className="bubble ai typing">
                        <span className="dot" />
                        <span className="dot" />
                        <span className="dot" />
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Sticky Input Bar when active */}
            {isLiveChatActive && (
              <div className="chat-input-area sticky">
                <input
                  type="text"
                  autoFocus
                  placeholder="Continue the conversation..."
                  value={userInput}
                  onChange={(e) => setUserInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && userInput.trim() && !isAiThinking) {
                      const msg = userInput;
                      setUserInput('');
                      
                      // Optimistically update the UI timeline with user's message
                      setConversations(prev => {
                        const newConvs = [...prev];
                        if (newConvs.length > 0) {
                          const lastDay = { ...newConvs[newConvs.length - 1] };
                          if (lastDay.interactions && lastDay.interactions.length > 0) {
                            lastDay.interactions = [...lastDay.interactions]; // CLONE ARRAY FIRST
                            const lastInter = { ...lastDay.interactions[lastDay.interactions.length - 1] };
                            lastInter.turns = [...lastInter.turns, { speaker: 'User', text: msg }];
                            lastDay.interactions[lastDay.interactions.length - 1] = lastInter;
                            newConvs[newConvs.length - 1] = lastDay;
                          }
                        }
                        return newConvs;
                      });

                      setIsAiThinking(true);

                      fetch(`${API_BASE}/conversation/respond`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ message: msg })
                      })
                        .then(r => r.json())
                        .then(data => {
                          setPendingBucket(data.pending_bucket || []);
                          // Refresh main conversation list from source-of-truth JSON
                          fetch(`${API_BASE}/conversations`)
                            .then(r => r.json())
                            .then(setConversations);
                          refreshSql();
                          refreshRag();
                        })
                        .catch(console.error)
                        .finally(() => setIsAiThinking(false));
                    }
                  }}
                />
              </div>
            )}
          </div>
        )}

        {activeView === 'sql' && (
          <div className="data-view fade-in">
            <div className="data-view-header">
              <h2>Structured Memory</h2>
              <p>Extracted preferences and tasks from the SQL database</p>
            </div>
            <div className="sql-tab-bar">
              <button
                className={`sql-tab-btn ${sqlTab === 'preferences' ? 'active' : ''}`}
                onClick={() => setSqlTab('preferences')}
              >
                Preferences ({sqlPrefs.length})
              </button>
              <button
                className={`sql-tab-btn ${sqlTab === 'tasks' ? 'active' : ''}`}
                onClick={() => setSqlTab('tasks')}
              >
                Tasks ({sqlTasks.length})
              </button>
            </div>
            <div className="data-view-body">
              {sqlTab === 'preferences' ? (
                <table className="sql-table">
                  <thead>
                    <tr>
                      <th>Entity</th>
                      <th>Preference</th>
                      <th>Category</th>
                      <th>Time</th>
                      <th style={{ textAlign: 'right' }}>Source Date</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sqlPrefs.map((p, i) => (
                      <tr key={i}>
                        <td className="entity-cell">{p.entity}</td>
                        <td>{p.preference}</td>
                        <td><span className="category-badge">{p.category}</span></td>
                        <td>{p.time_of_day}</td>
                        <td className="date-cell" style={{ textAlign: 'right' }}>{p.source_date}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <table className="sql-table">
                  <thead>
                    <tr>
                      <th>Task</th>
                      <th>Type</th>
                      <th>Status</th>
                      <th>Time</th>
                      <th style={{ textAlign: 'right' }}>Source Date</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sqlTasks.map((t, i) => (
                      <tr key={i}>
                        <td className="entity-cell">{t.task_description}</td>
                        <td><span className="category-badge">{t.task_type}</span></td>
                        <td>{t.status}</td>
                        <td>{t.time_of_day}</td>
                        <td className="date-cell" style={{ textAlign: 'right' }}>{t.source_date}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </div>
        )}
        {activeView === 'rag' && (
          <div className="data-view fade-in">
            <div className="data-view-header">
              <div style={{ flex: 1 }}>
                <h2> RAG Memory</h2>
              </div>

              <div className="filter-bar">
                {['all', 'weekly-summary', 'monthly-summary', 'routine', 'trajectory', 'raw'].map(f => (
                  <button
                    key={f}
                    className={`filter-pill ${ragFilter === f ? 'active' : ''}`}
                    onClick={() => setRagFilter(f)}
                  >
                    {f.replace('-', ' ')}
                  </button>
                ))}
              </div>
            </div>

            <div className="data-view-body">
              {loading ? (
                <div className="empty-state">
                  <Loader2 size={32} className="spin" style={{ color: 'var(--primary)' }} />
                  <p>Retrieving semantic memory...</p>
                </div>
              ) : ragEntries.length === 0 ? (
                <div className="empty-state">
                  <Layers size={32} />
                  <p>No semantic entries found</p>
                </div>
              ) : (
                <div className="rag-grid">
                  {ragEntries
                    .filter(entry => {
                      if (ragFilter === 'all') return true;
                      if (ragFilter === 'raw') return !entry.metadata?.type || entry.metadata.type !== 'summary_document';
                      return entry.metadata?.insight_type === ragFilter;
                    })
                    .map((entry, idx) => (
                      <div key={idx} className={`rag-card ${entry.metadata?.type === 'summary_document' ? 'summary-card' : ''}`}>
                        <div className="rag-card-header">
                          <div className="rag-card-meta">
                            <span className="rag-date">{entry.metadata?.date || entry.metadata?.generated_at || 'Unknown Date'}</span>
                            <span className="rag-time">{entry.metadata?.time_of_day || ''}</span>
                          </div>
                          {entry.metadata?.type === 'summary_document' && (
                            <span className="insight-badge">{entry.metadata?.insight_type}</span>
                          )}
                        </div>
                        <p className="rag-content">{entry.content}</p>
                      </div>
                    ))}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
