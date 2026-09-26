# Inspector frontend

React + Vite single-page app. It talks to the backend at `http://localhost:8000/api`
(`API` in `src/App.jsx`), so start `app/backend.py` first.

```bash
npm install
npm run dev      # dev server with hot reload
npm run build    # production build into dist/
```

## Source

| File | What it is |
|---|---|
| `src/App.jsx` | The whole UI. Four top-level tabs: **Benchmarks** (score cards for the synthetic eval and LoCoMo; click through to per-question results), **Datasets** (synthetic conversations and QA, LoCoMo conversations and QA results), **Store** (preferences, conversation chunks, summaries of the open store), **Chat** (talk to the open store). |
| `src/index.css` | Styles: design tokens on `:root` (single indigo accent), tables (`.tbl`), sub-navigation (`.sub-nav`), benchmark cards (`.bench-*`), score-coloured rows (`.row-good/mid/bad`). |
| `src/main.jsx` | React entry point. |
| `index.html`, `vite.config.js`, `eslint.config.js` | Vite and lint config. |

The LoCoMo views handle both result formats: current (`response` and `score` on each
record) and the older v2 format (one nested object per system).
