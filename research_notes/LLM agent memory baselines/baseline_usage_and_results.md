# Standard baselines and reported results for conversational long-term memory (2025–2026)

Method note: 25 arXiv PDFs were downloaded and converted with `pdftotext`. Baseline lists and tables below come from that extracted text. Venues come from the paper's own "Published as..." header or from the arXiv comment field (export.arxiv.org API), unless marked otherwise. When a paper is marked "arXiv-only", no venue statement was found in the PDF or in the arXiv metadata. The paper may still have been accepted somewhere since.

## Q1. Which baseline systems appear most often in recent papers?

### Takeaway
Across 16 memory-system and benchmark papers from 2025–2026, **Mem0** is the most common baseline, used by 12 of the 15 other papers. The three that omit it (Zep, A-Mem, MemoryOS) were written before or at the same time as Mem0. Next come **Zep** (10), **A-Mem** (10) and a **full-context / long-context LLM** (9). **LangMem** and **MemGPT/Letta** follow with 7 each. A second tier includes naive RAG, MemoryBank, the LoCoMo paper's own RAG baseline, MemoryOS, LightMem and Nemori (4–5 each). The newest 2026 papers add SimpleMem, MIRIX, MemOS/EverMemOS, Memobase, Hindsight and Supermemory. A new paper in 2026 would be expected to include at least Full-context, RAG, Mem0 (optionally Mem0g), Zep, A-Mem and LangMem. It should also include one or two of the 2025–26 "efficient" systems (LightMem, Nemori, SimpleMem, MemoryOS).

### Cited Findings: per-paper baseline lists
1. **Mem0** (ECAI 2025; IOS Press FAIA251160). Baselines on LoCoMo: LoCoMo, ReadAgent, MemoryBank, MemGPT, A-Mem, LangMem, RAG (chunk sizes 128–8192, k in {1,2}), Full-context, OpenAI Memory, Zep. Backbone: GPT-4o-mini. — [PDF](https://arxiv.org/pdf/2504.19413); venue: [IOS Press](https://ebooks.iospress.nl/doi/10.3233/FAIA251160), [ECAI accepted papers](https://ecai2025.org/accepted-papers/)
2. **A-MEM** (NeurIPS 2025, per the arXiv comment). Baselines: LoCoMo, ReadAgent, MemoryBank, MemGPT. Six backbones (GPT-4o-mini, GPT-4o, Qwen2.5-1.5B/3B, Llama 3.2 1B/3B). — [PDF](https://arxiv.org/pdf/2502.12110)
3. **MemoryOS** (EMNLP 2025 main, Oral). Baselines: TiM, MemoryBank, MemGPT, A-Mem, on GVD and LoCoMo (F1/BLEU-1). Backbones: GPT-4o-mini and Qwen2.5-7B. — [PDF](https://arxiv.org/pdf/2506.06326); venue: [ACL Anthology 2025.emnlp-main.1318](https://aclanthology.org/2025.emnlp-main.1318.pdf), [GitHub](https://github.com/BAI-LAB/MemoryOS)
4. **LightMem** (ICLR 2026, per the PDF header). Baselines: Full Text, Naive RAG, LangMem, A-MEM, MemoryOS, Mem0. Backbones: GPT-4o-mini, Qwen3-30B-A3B-Instruct-2507, GLM-4.6. Judge: GPT-4o-mini. — [PDF](https://arxiv.org/pdf/2510.18866)
5. **Memory-R1** (arXiv-only). Baselines: LoCoMo (RAG), A-Mem, Mem0, MemoryOS, Memory-SFT. All were re-implemented on LLaMA-3.1-8B-Instruct and Qwen-2.5-7B-Instruct. Also evaluated on MSC and LongMemEval. — [PDF](https://arxiv.org/pdf/2508.19828)
6. **MIRIX** (arXiv-only). LoCoMo baselines: A-Mem, LangMem, Zep, Mem0, Memobase, RAG-500, Full-Context. OpenAI Memory is quoted from Mem0. All were re-run with gpt-4.1-mini, and GPT-4.1 was the judge. — [PDF](https://arxiv.org/pdf/2507.07957)
7. **SimpleMem** (arXiv-only). Baselines: LoCoMo, ReadAgent, MemoryBank, MemGPT, A-Mem, LightMem, Mem0. Full-context appears on LongMemEval. Backbones: GPT-4o, GPT-4.1-mini, Qwen-Plus, Qwen2.5 1.5B/3B, Qwen3 1.7B/8B. — [PDF](https://arxiv.org/pdf/2601.02553)
8. **EMem, "A Simple Yet Strong Baseline..."** (arXiv, marked "Work in progress"). It follows the Nemori setup: full-context, LangMem, Zep, Mem0, Nemori. — [PDF](https://arxiv.org/pdf/2511.17208)
9. **ES-Mem** (arXiv-only). Ten baselines: MemGPT, MemoryBank, A-Mem, MemoryOS, H-Mem, Zep, Mem0, LangMem, Nemori, LightMem. Evaluated on LoCoMo and LongMemEval with GPT-4o-mini and Qwen2.5-3B. — [PDF](https://arxiv.org/pdf/2601.07582)
10. **MemIR, "Mitigating Provenance-Role Collapse..."** (arXiv-only, May 2026). Baselines: Zep, LangMem, A-Mem, MemoryOS, Mem0, LightMem, Nemori, SimpleMem, ReadAgent, LoCoMo, SwiftMem, HiMem. Benchmarks: LoCoMo and BEAM-100K. — [PDF](https://arxiv.org/pdf/2605.25869)
11. **RippleMem** (arXiv-only, Aug 2026). LoCoMo baselines: Full-Context, Mem0, Mem0g, Zep, MemGAS, M-Flow, REMem, SimpleMem, RF-Mem. LongMemEval-S adds reported numbers from LightMem, MemU, MemOS and EverMemOS. — [PDF](https://arxiv.org/pdf/2608.13334)
12. **REALM, "Retrieval-Driven Memory Reconsolidation..."** (arXiv-only, Sep 2026). Baselines: Mem0, Zep, MIRIX, A-Mem, Nemori, MAGMA. GPT-4o-mini is both backbone and judge. — [PDF](https://arxiv.org/pdf/2609.16053)
13. **Hindsight** (arXiv tech report, Dec 2025). LongMemEval baselines: Full-context (GPT-4o and OSS-20B), Zep, and Supermemory, all taken from Supermemory's report. LoCoMo baselines: Backboard, Memobase, Zep, Mem0, Mem0-Graph, LangMem, OpenAI, all "as claimed on the official Backboard LoCoMo benchmark results" and not reproduced. — [PDF](https://arxiv.org/pdf/2512.12818)
14. **Agent Zero Memory** (arXiv-only, Aug 2026). Baselines: Zep, Mem0, Mastra, Hindsight, EmergenceMem, Supermemory, ByteRover, Memobase. "All competitor numbers are the..." publicly reported ones, not re-runs. — [PDF](https://arxiv.org/pdf/2608.29606)
15. **Zep** (arXiv-only, Jan 2025). DMR baselines: MemGPT, recursive summarization, conversation summaries, full-conversation. LongMemEval baseline: full-context. — [PDF](https://arxiv.org/pdf/2501.13956)
16. **MemoryAgentBench** (ICLR 2026, per the PDF header). Evaluates long-context agents, RAG agents (e.g., BM25, RAPTOR, HippoRAG-v2, Cognee, Zep, MemoRAG, Mem0) and agentic memory (e.g., MemGPT, Self-RAG, A-MEM, MIRIX). Main table rows include MemoRAG, HippoRAG-v2 and Mem0. — [PDF](https://arxiv.org/pdf/2507.05257)
- Older-style personalization papers use generic baselines rather than named systems. **RMM** (ACL 2025, per the arXiv comment) compares against No History, Long Context, and RAG at turn and session level. — [PDF](https://arxiv.org/pdf/2503.08026). **SeCom** (ICLR 2025) compares against Turn-Level, Session-Level, Zero History, Full History, SumMem, RecurSum, ConditionMem and MemoChat. — [PDF](https://arxiv.org/pdf/2502.05589)

### Occurrence tally (papers 1–16; a system's own paper is not counted)
| Baseline | Count | Papers (#) |
|---|---|---|
| Mem0 (incl. Mem0g) | 12 | 4,5,6,7,8,9,10,11,12,13,14,16 |
| Zep / Graphiti | 10 | 1,6,8,9,10,11,12,13,14,16 |
| A-Mem | 10 | 1,3,4,5,6,7,9,10,12,16 |
| Full-context / long-context | 9 | 1,4,6,7,8,11,13,15,16 (plus RMM, SeCom) |
| LangMem | 7 | 1,4,6,8,9,10,13 |
| MemGPT / Letta | 7 | 1,2,3,7,9,15,16 |
| Naive RAG | 5 | 1,4,5(LoCoMo-RAG),6,16 (plus RMM, SeCom) |
| MemoryBank | 5 | 1,2,3,7,9 |
| LoCoMo paper baseline | 5 | 1,2,5,7,10 |
| MemoryOS | 4 | 4,5,9,10 |
| LightMem | 4 | 7,9,10,11 |
| Nemori | 4 | 8,9,10,12 |
| ReadAgent | 4 | 1,2,7,10 |
| OpenAI Memory | 3 | 1,6,13 |
| Memobase | 3 | 6,13,14 |
| MIRIX, SimpleMem, Supermemory | 2 each | 12,16 / 10,11 / 13,14 |
| MemOS, EverMemOS, MAGMA, Hindsight, MemU, Mastra, ByteRover | 1 each | 11 / 11 / 12 / 14 / 11 / 14 / 14 |
| Memory-R1 | 0 in this sample | — |

### Inferences
- The "canonical set" shifted over time. Early-2025 papers (A-Mem, MemoryOS) used the LoCoMo-paper baselines: LoCoMo, ReadAgent, MemoryBank, MemGPT, TiM. After Mem0 (Apr 2025), the standard became Mem0 + Zep + LangMem + A-Mem + Full-context + RAG. 2026 papers add the efficiency-oriented systems (LightMem, Nemori, SimpleMem, MemoryOS) and, increasingly, vendor systems (Memobase, Supermemory, Hindsight, EverMemOS).
- Many 2026 arXiv papers copy numbers instead of re-running systems. Hindsight, Agent Zero, RippleMem's LongMemEval section, REALM ("aligned literature") and SimpleMem (A-Mem/LoCoMo rows copied from the A-Mem paper) all do this. Reviewers increasingly treat a same-backbone re-run as more credible, as MIRIX, LightMem and Memory-R1 did.
- MemGPT/Letta is cited often but is usually only a LoCoMo F1/BLEU row copied from the A-Mem paper. It is rarely re-run on LongMemEval. Zep's paper also says MemGPT could not easily be evaluated on LongMemEval ([Zep PDF](https://arxiv.org/pdf/2501.13956)).

### Gaps
- Nemori (arXiv 2508.03341, now titled "What Deserves Memory: Adaptive Memory Distillation for LLM Agents") did not yield extractable text, so its baseline list is not tallied.
- Only one 2026 paper with a peer-reviewed venue was confirmed (LightMem / MemoryAgentBench / BEAM at ICLR 2026). Most 2026 memory-system papers in this sample are arXiv-only. The tally therefore leans toward preprints.
- Memory-R1's venue was not verified. It has no arXiv comment.

## Q2. Reported scores on LoCoMo and LongMemEval, and where papers disagree

### Takeaway
Scores for the same system vary widely between papers. Mem0 on LoCoMo (LLM-judge, overall) ranges from about 43% to 67% in third-party re-runs, against a vendor claim of 92.5%. Zep ranges from about 59% to 79% in papers, against vendor claims of 75.1% and 94.7%. Full-context on LongMemEval-S ranges from 35% to 60%. The spread comes from the backbone, the judge model and prompt, per-category label conventions, and whether numbers were re-run or copied. Only numbers from the same paper, same backbone and same judge are comparable.

### Cited Findings: LoCoMo (J = LLM-as-judge %, categories 1–4, adversarial excluded)
**Mem0 paper, GPT-4o-mini** (Single-hop / Multi-hop / Open-domain / Temporal; J with F1 in parentheses; overall J from its Table 2) — [PDF](https://arxiv.org/pdf/2504.19413)
| System | Single-hop J (F1) | Multi-hop J (F1) | Open-domain J (F1) | Temporal J (F1) | Overall J |
|---|---|---|---|---|---|
| A-Mem* (re-run) | 39.79 (20.76) | 18.85 (9.22) | 54.05 (33.34) | 49.91 (35.40) | 48.38 |
| LangMem | 62.23 (35.51) | 47.92 (26.04) | 71.12 (40.91) | 23.43 (30.75) | 58.10 |
| Zep | 61.70 (35.74) | 41.35 (19.37) | 76.60 (49.56) | 49.31 (42.00) | 65.99 |
| OpenAI Memory | 63.79 (34.30) | 42.92 (20.09) | 62.29 (39.31) | 21.71 (14.04) | 52.90 |
| Mem0 | 67.13 (38.72) | 51.15 (28.64) | 72.93 (47.65) | 55.51 (48.93) | 66.88 |
| Mem0g | 65.71 (38.09) | 47.19 (24.32) | 75.71 (49.27) | 58.13 (51.55) | 68.44 |
| Full-context (26k tok) | – | – | – | – | 72.90 |
| Best RAG (k=2, 256 tok) | – | – | – | – | 60.97 |

BLEU-1 is also given. For example, Mem0 scores 27.13 / 21.58 / 38.72 / 40.51. — [PDF](https://arxiv.org/pdf/2504.19413)

**MIRIX re-run, gpt-4.1-mini, GPT-4.1 judge** (Single / Multi / Open / Temporal / Overall): RAG-500 37.94/37.69/48.96/61.83/51.62; Zep 79.43/69.16/73.96/83.33/79.09; Mem0 62.41/57.32/44.79/66.47/62.47; MIRIX 85.11/83.70/65.62/88.39/85.38; Full-Context 88.53/77.70/71.88/92.70/87.52. The same table's GPT-4o-mini block lists Memobase 70.91, Zep 75.14 and LangMem 78.05 overall. — [PDF](https://arxiv.org/pdf/2507.07957)

**LightMem, overall accuracy (GPT-4o-mini judge).** With GPT-4o-mini: FullText 71.83, NaiveRAG 63.64, LangMem 57.20, A-MEM 64.16, MemoryOS 58.25 (LoCoMo-tuned) / 54.87 (regular), Mem0 61.69, LightMem 70.26–72.99. With Qwen3-30B-A3B: FullText 74.87, NaiveRAG 66.95, LangMem 60.53, A-MEM 56.10, MemoryOS 61.04/51.30, Mem0 43.31, LightMem 71.36–72.60. — [PDF](https://arxiv.org/pdf/2510.18866)

**REALM, GPT-4o-mini backbone and judge** (Multi-hop / Temporal / Open / Single / Avg): MIRIX 54.26/68.54/46.88/68.22/64.33; Mem0 58.75/52.34/45.83/73.33/64.57; Zep 52.12/54.82/33.33/66.23/59.22; MAGMA 52.80/65.00/51.70/77.60/68.80; Nemori 56.90/64.90/48.50/76.40/68.70; A-Mem 53.55/50.16/41.67/61.83/56.62; REALM 64.54/76.64/58.33/81.57/75.97. — [PDF](https://arxiv.org/pdf/2609.16053)

**RippleMem, J per category** (Multi-hop / Temporal / Open / Single / Avg J; Avg F1 in brackets): Full-Context 61.70/50.78/53.13/81.57/69.74 [44.17]; Mem0 62.06/64.49/53.13/62.54/62.27 [38.65]; Mem0g 68.44/64.80/57.29/66.59/65.97 [41.28]; Zep 66.31/70.72/60.42/84.30/76.69 [43.98]; SimpleMem 78.01/76.01/63.54/89.42/82.92 [50.48]; RippleMem 77.67/85.67/70.83/92.75/87.14 [52.49]. The backbone was not captured in the extraction. — [PDF](https://arxiv.org/pdf/2608.13334)

**MemIR, GPT-4.1-mini, J per category** (Single / Multi / Temporal / Open): Zep 66.9/53.7/60.2/43.8; A-Mem 64.0/55.7/66.7/37.5; LightMem 72.5/69.6/68.1/52.4; MemoryOS 68.9/62.4/37.7/60.4; Mem0 71.4/68.2/56.9/47.9; LangMem 84.5/71.0/50.8/59.0; Nemori 87.0/74.8/77.3/56.3; SimpleMem 87.4/73.8/80.7/57.3; MemIR 89.5/70.2/84.6/61.9. — [PDF](https://arxiv.org/pdf/2605.25869)

**EMem (Nemori protocol), overall LLM score** for two backbones: LangMem 0.513 / 0.734; Mem0 0.613 / 0.663; Zep 0.585 / 0.616; Nemori 0.744 / 0.794. Full-context scores 0.806 and EMem-G 0.853 on the second backbone. — [PDF](https://arxiv.org/pdf/2511.17208)

**Hindsight, as-reported leaderboard numbers** (Single / Multi / Open / Temporal / Overall): Backboard 89.36/75.00/91.20/91.90/90.00; Memobase 70.92/46.88/77.17/85.05/75.78; Zep 74.11/66.04/67.71/79.79/75.14; Mem0 66.88 overall; Hindsight OSS-20B 83.18, OSS-120B 85.67, Gemini-3 89.61. Hindsight's own judge was GPT-OSS-120B, but the baseline numbers were copied. — [PDF](https://arxiv.org/pdf/2512.12818)

**F1/BLEU-only lines of work:** A-Mem GPT-4o-mini LoCoMo F1 (Multi / Temporal / Open / Single / Adversarial as labelled by A-Mem): A-Mem 27.02/45.85/12.14/44.65/50.03 vs MemGPT 26.65/25.52/9.15/41.04/43.29 vs LoCoMo 25.02/18.41/12.04/40.36/69.23. — [PDF](https://arxiv.org/pdf/2502.12110). SimpleMem with GPT-4.1-mini reports average F1: A-Mem 32.58, LightMem 24.63, Mem0 34.20, SimpleMem 43.24. — [PDF](https://arxiv.org/pdf/2601.02553)

**Vendor claims (2026, self-reported):** Mem0's "new algorithm" reports LoCoMo 92.5% and LongMemEval 94.4% at about 6.7–7K tokens per retrieval. Zep claims 94.7% on LoCoMo, while "third-party tests put Zep at 75.1%". ByteRover publishes 92.2% or 96.1%. — [Mem0 blog](https://mem0.ai/blog/ai-memory-benchmarks-in-2026). Agent Zero's LoCoMo table lists Agent Zero 93.6, Mem0 92.5, ByteRover 92.2, Hindsight 89.6, Memobase 75.8, Zep 75.1. — [PDF](https://arxiv.org/pdf/2608.29606)

### Cited Findings: LongMemEval-S (500 Qs; J %; ability columns)
**Zep paper, GPT-4o judge.** Overall: Full-context 55.4 (4o-mini) / 60.2 (4o); Zep 63.8 (4o-mini) / 71.2 (4o). Zep with 4o-mini → 4o per ability: SS-pref 53.3→56.7, SS-asst 75.0→80.4, temporal 54.1→62.4, multi-session 47.4→57.9, knowledge-update 74.4→83.3, SS-user 92.9→92.9. Full-context 4o scores knowledge-update 78.2 and temporal 45.1. — [PDF](https://arxiv.org/pdf/2501.13956)

**LightMem, GPT-4o-mini** (Overall; Temporal / Multi-session / KU / SS-user / SS-asst / SS-pref):
| System | Overall | TR | MS | KU | SSU | SSA | SSP |
|---|---|---|---|---|---|---|---|
| Full Text | 56.80 | 31.58 | 45.45 | 76.92 | 87.14 | 89.29 | 36.67 |
| Naive RAG | 61.00 | 39.85 | 48.48 | 67.95 | 90.00 | 98.21 | 53.33 |
| LangMem | 37.20 | 15.79 | 20.30 | 66.67 | 60.00 | 46.43 | 60.00 |
| A-MEM | 62.60 | 47.36 | 48.87 | 64.11 | 92.86 | 96.43 | 46.67 |
| MemoryOS | 44.80 | 32.33 | 31.06 | 48.72 | 80.00 | 64.29 | 30.00 |
| Mem0 | 53.61 | 40.15 | 46.21 | 70.12 | 81.43 | 41.07 | 60.00 |
| LightMem | 64.29–68.64 | 67.18 | 71.74 | 83.12 | 87.14 | 32.14 | 68.18 |

With Qwen3-30B, overall scores are FullText 54.80, NaiveRAG 60.80, LangMem 50.80, A-MEM 65.20, MemoryOS 49.60 and Mem0 39.51. — [PDF](https://arxiv.org/pdf/2510.18866)

**SimpleMem, GPT-4.1-mini judge** (TR / MS / KU / SSU / SSA / SSP / Avg). With GPT-4.1-mini: Full-context 27.06/30.08/41.03/47.14/32.14/60.00/39.57; Mem0 40.60/50.37/69.23/87.14/48.21/63.33/59.81; LightMem 85.71/47.37/92.30/88.57/21.43/76.67/68.67; SimpleMem 83.46/60.92/79.48/85.71/75.00/76.67/76.87. With GPT-4.1: Full-context 56.72, Mem0 58.51, LightMem 76.86, SimpleMem 83.97. — [PDF](https://arxiv.org/pdf/2601.02553)

**REALM, GPT-4o-mini** (SS-pref / SS-asst / TR / MS / KU / SS-user / Avg): MIRIX 53.3/63.6/25.6/30.1/52.6/72.9/43.49; Zep 53.3/75.0/54.1/47.4/74.4/92.9/63.80, which reproduces Zep's own numbers; MAGMA 73.3/83.9/45.1/50.4/66.7/72.9/61.20; Nemori 62.7/73.2/43.0/51.4/52.6/77.7/56.20; REALM 36.66/82.14/56.69/46.28/88.89/89.06/65.11. — [PDF](https://arxiv.org/pdf/2609.16053)

**RippleMem, "EverMemOS-aligned" group** (SSU / MS / SSP / TR / KU / SSA / Overall): MemU 38.40 overall; Zep 63.80; Mem0 82.86/63.15/90.00/72.18/66.67/26.78/66.40; EverMemOS 97.14/73.68/93.33/77.44/89.74/85.71/83.00; RippleMem 86.60. In the SimpleMem-aligned group, Full-Context scores 35.40 overall and RippleMem 84.80. — [PDF](https://arxiv.org/pdf/2608.13334)

**Leaderboard-style claims:** Hindsight reports 83.6% (OSS-20B), 89.0% (OSS-120B) and 91.4% (Gemini-3 Pro), against Supermemory at 81.6% (GPT-4o) and 84.6% (GPT-5), Zep+GPT-4o at 71.2% and Full-context GPT-4o at 60.2%. — [PDF](https://arxiv.org/pdf/2512.12818). Agent Zero lists Agent Zero 95.60, Mastra 94.87, Hindsight 91.40, EmergenceMem 86.00, Supermemory 85.20 and Zep 71.20. — [PDF](https://arxiv.org/pdf/2608.29606). Mem0 reports KU 93.6% and multi-session 88.0% for its 2026 algorithm. — [Mem0 blog](https://mem0.ai/blog/ai-memory-benchmarks-in-2026)

**Memory-R1 on LongMemEval (overall F1 / B1 / J).** With LLaMA-3.1-8B: LoCoMo-RAG 20.55/15.17/21.00, A-Mem 38.36/33.30/54.20, Mem0 31.41/21.69/41.20, Memory-R1-GRPO 45.20/39.30/55.40. With Qwen-2.5-7B: A-Mem 54.80 J, Mem0 46.80 J, Memory-R1-GRPO 57.80 J. Its main LoCoMo table is an image and was not extractable. — [PDF](https://arxiv.org/pdf/2508.19828)

**MSC / DMR:** Zep reports 94.8% with gpt-4-turbo against MemGPT's 93.4%, and 98.2% with gpt-4o-mini against 98.0% for full-conversation. Zep argues DMR is saturated and small. — [PDF](https://arxiv.org/pdf/2501.13956)

### Notable disagreements (same system, different papers)
- **Mem0 on LoCoMo, overall J:** 66.88 (Mem0 paper, 4o-mini); 61.69 (LightMem, 4o-mini); 43.31 (LightMem, Qwen3-30B); 62.47 (MIRIX, 4.1-mini); 64.57 (REALM); 62.27 (RippleMem); 92.5 (Mem0's 2026 self-report).
- **Zep on LoCoMo:** 65.99 (Mem0 paper); 75.14 (Zep rebuttal, also used by MIRIX and Hindsight); 79.09 (MIRIX re-run, 4.1-mini); 59.22 (REALM); 76.69 (RippleMem); 94.7 (Zep claim quoted by Mem0). — sources above plus [Zep blog](https://blog.getzep.com/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory/)
- **LangMem on LoCoMo:** 58.10 (Mem0 paper) vs 78.05 (GPT-4o-mini block in MIRIX's table). **LangMem on LongMemEval:** 37.20 (LightMem, 4o-mini) vs 50.80 (Qwen3).
- **Full-context on LongMemEval-S:** 55.4/60.2 (Zep); 56.80 (LightMem); 39.57 (SimpleMem, 4.1-mini, macro-average of abilities); 35.40 (RippleMem, same raw numbers but a different averaging). On the temporal ability, full-context scores 31.6 (LightMem), 27.06 (SimpleMem) and 45.1 (Zep, 4o).
- **LoCoMo category labels are inconsistent.** A-Mem's own GPT-4o-mini row reads Multi-Hop 27.02 / Temporal 45.85 / Open-Domain 12.14 / Single-Hop 44.65. Mem0 reprints the same numbers as Single-Hop 27.02 / Multi-Hop 12.14 / Open-Domain 44.65 / Temporal 45.85. The "single-hop", "multi-hop" and "open-domain" names are therefore permuted between the two lines of papers. SimpleMem and RippleMem follow A-Mem's ordering, while Mem0, MIRIX and Hindsight follow Mem0's. — [A-Mem PDF](https://arxiv.org/pdf/2502.12110); [Mem0 PDF](https://arxiv.org/pdf/2504.19413)

### Inferences
- Per-category LoCoMo comparisons across papers are unsafe unless the category mapping is stated. A new paper should state its mapping explicitly (category ID → name) and report overall J, F1 and BLEU-1.
- Mem0 and LangMem scores drop sharply with open-weight backbones (LightMem, Qwen3-30B). Results on open models may not transfer from GPT-4o-mini numbers.
- Full-context is a strong baseline on LoCoMo (72–88% J, often beating memory systems). It is much weaker on LongMemEval-S (about 115k tokens per history). LongMemEval is therefore the more discriminative test for memory.

### Gaps
- The Hindsight and Memory-R1 per-category LongMemEval and LoCoMo tables were images, so their numbers were not captured.
- RippleMem's and EMem's backbones were not confirmed from the extracted text.
- No official LongMemEval leaderboard with standardized judge settings was found. Leaderboard numbers are vendor self-reports.

## Q3. Public leaderboards, reproducibility studies and critiques

### Takeaway
There is no neutral, standardized leaderboard. Numbers circulate as vendor claims (Backboard, Mem0, Zep, ByteRover, Supermemory) that later papers copy. Documented critiques are: Zep's rebuttal of Mem0's Zep configuration; Letta's filesystem result showing LoCoMo can be solved without a memory system; the Penfield Labs audit finding 6.4% wrong gold answers and a lenient judge; and Zep's argument that LoCoMo and DMR are too short to test memory.

### Cited Findings
- Zep says Mem0 mis-configured Zep. Mem0 assigned the user role to both speakers, appended timestamps to message text instead of using the timestamp field, and ran searches sequentially, which inflated latency. With a corrected setup Zep reports 75.14 ± 0.17 J and 0.632 s p95 search latency. — [Zep blog](https://blog.getzep.com/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory/)
- Zep also argues that LoCoMo conversations (16k–26k tokens) fit in context, and that full-context (about 73%) beats Mem0's best (about 68%). It adds that LoCoMo has no knowledge-update questions and has data-quality problems: category 5 lacks gold answers, image-dependent questions, speaker-attribution errors and ambiguous questions. — [Zep blog](https://blog.getzep.com/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory/)
- Letta reports that an agent storing conversation history in files reaches 74.0% on LoCoMo with gpt-4o-mini, above Mem0's reported 68.5% for Mem0g. It concludes that current memory benchmarks "may not be very meaningful". — [Letta blog](https://www.letta.com/blog/benchmarking-ai-agent-memory/)
- The Penfield Labs audit found 99 score-corrupting errors in 1,540 questions (6.4%), so the realistic ceiling is about 93.6%. The gpt-4o-mini judge with the published prompt accepted 62.81% of deliberately wrong but topical answers. The audit also notes that LoCoMo-Plus keeps the original 1,540 questions unchanged. — [Penfield Labs](https://penfieldlabs.substack.com/p/we-audited-locomo-64-of-the-answer); [audit repo](https://github.com/dial481/locomo-audit)
- Zep argues DMR (MemGPT's MSC subset) is too small and saturated: 98.2% for Zep vs 98.0% for full-conversation with gpt-4o-mini. Zep says it could not reproduce MemGPT's DMR numbers with gpt-4o-mini. — [Zep PDF](https://arxiv.org/pdf/2501.13956)
- Mem0's own guide says the judge model, answering model and use of reranking all change the score: "two labs running 'the same benchmark'... will not land on the same number". It labels most leaderboard entries as self-reported. — [Mem0 blog](https://mem0.ai/blog/ai-memory-benchmarks-in-2026)
- Hindsight explicitly says its LoCoMo baselines are "as claimed on the official Backboard LoCoMo benchmark results... reported reference points rather than our independently reproduced baselines", and that Backboard's numbers "could not be independently reproduced". — [Hindsight PDF](https://arxiv.org/pdf/2512.12818)
- MIRIX re-ran Mem0, LangMem and RAG-500 with Mem0's evaluation code, and Zep with Zep's official repo, all on gpt-4.1-mini. Baselines were run once, while MIRIX and Full-Context were run three times. — [MIRIX PDF](https://arxiv.org/pdf/2507.07957)
- Search surfaced newer holistic re-evaluations that were not read in detail: "Harness the Memory: A Holistic Evaluation of Memory Substrates in Memory Agents" ([PDF](https://arxiv.org/pdf/2608.15008)) and an open reproducible harness ([agent-memory-bench](https://github.com/hamza-dev-tech/agent-memory-bench)).

### Inferences
- A credible new paper should re-run key baselines (Mem0, Zep, A-Mem, LangMem, Full-context) with the same backbone and judge, and report the judge model and prompt. It should also report a human-verified subset or a stricter judge, and ideally exclude or correct the audited LoCoMo errors.

### Gaps
- The Penfield audit is a blog/Substack post, not peer-reviewed. Its figures were not independently verified.
- No peer-reviewed reproducibility paper on LoCoMo or LongMemEval was confirmed. The "Harness the Memory" paper was not read.

## Q4. Newer benchmarks for lifelong or evolving memory

### Takeaway
LoCoMo (2024) and LongMemEval (ICLR 2025) remain the two de facto benchmarks. Almost every 2025–26 paper in the sample reports LoCoMo, and most also report LongMemEval-S. Several newer benchmarks target knowledge updates, contradictions, preference change and hallucinated memory, and a few 2026 papers now report them: BEAM (ICLR 2026) used by MemIR, MemoryAgentBench (ICLR 2026), PersonaMem (COLM 2025), PrefEval (ICLR 2025), MemBench (ACL 2025 Findings), HaluMem and LoCoMo-Plus (arXiv).

### Cited Findings
| Benchmark | Focus | Venue (verified how) | PDF |
|---|---|---|---|
| LoCoMo | Very long multi-session dialogues (about 300 turns, 9k–26k tokens), QA in 5 categories, event summarization | ACL 2024 per the task brief. The arXiv comment gives no venue, and the Anthology page was not re-checked here | [arXiv PDF](https://arxiv.org/pdf/2402.17753) |
| LongMemEval | 500 questions, abilities: information extraction, multi-session reasoning, knowledge updates, temporal reasoning, abstention; S (about 115k tokens) and M settings | ICLR 2025 (PDF header) | [PDF](https://arxiv.org/pdf/2410.10813) |
| PrefEval | Following stated or implicit user preferences over long conversations | ICLR 2025 (PDF header) | [PDF](https://arxiv.org/pdf/2502.09597) |
| MemoryAgentBench | Four competencies: accurate retrieval, test-time learning, long-range understanding, selective forgetting / conflict resolution. New EventQA and FactConsolidation data | ICLR 2026 (PDF header) | [PDF](https://arxiv.org/pdf/2507.05257) |
| BEAM ("Beyond a Million Tokens") | Conversations up to 10M tokens, 2,000 probing questions over 10 abilities incl. contradiction resolution, knowledge update, temporal reasoning, abstention | ICLR 2026 (PDF header) | [PDF](https://arxiv.org/pdf/2510.27246) |
| PersonaMem ("Know Me, Respond to Me") | Evolving user profiles and preferences across sessions | COLM 2025 (arXiv comment) | [PDF](https://arxiv.org/pdf/2504.14225) |
| MemBench | Factual and reflective memory, participation vs observation | ACL 2025 Findings (arXiv comment) | [PDF](https://arxiv.org/pdf/2506.21605) |
| HaluMem | Operation-level hallucination in memory extraction, updating and QA | arXiv-only | [PDF](https://arxiv.org/pdf/2511.03506) |
| LoCoMo-Plus | Beyond-factual, "cognitive" memory questions added to LoCoMo | arXiv-only (Feb 2026) | [PDF](https://arxiv.org/pdf/2602.10715) |
| MSC / DMR | Multi-Session Chat, and MemGPT's 500-conversation Deep Memory Retrieval subset | MSC: ACL 2022; DMR: from the MemGPT paper (arXiv). Venues not re-verified | Discussed in [Zep PDF](https://arxiv.org/pdf/2501.13956) |

- In MemIR's BEAM-100K results (Avg over ABS, CR, EO, IE, IF, KU, MSR, PF, SUM, TR), MemoryOS scores 28.64, LightMem 32.46, Mem0 37.05, Nemori 40.45 and SimpleMem 40.68. Knowledge-update (KU) scores are Mem0 45.1, Nemori 52.1 and SimpleMem 55.0. — [MemIR PDF](https://arxiv.org/pdf/2605.25869)
- Mem0 reports BEAM-1M 64.1% and BEAM-10M 48.6% for its 2026 algorithm (self-reported). — [Mem0 blog](https://mem0.ai/blog/ai-memory-benchmarks-in-2026)
- In MemoryAgentBench's main table, Mem0 averages 21.1, below HippoRAG-v2 (41.6) and MemoRAG (30.9). — [PDF](https://arxiv.org/pdf/2507.05257)
- Other 2026 benchmarks surfaced by search but not read: MemOps (lifecycle memory operations; [PDF](https://arxiv.org/pdf/2607.12893)), Supersede (memory-update gap; [PDF](https://arxiv.org/pdf/2606.27472)), AlpsBench (real-dialogue preference alignment; [HTML](https://arxiv.org/html/2603.26680v2)), and "Ask Now, Use Later" (proactivity in long-lived agents; [PDF](https://arxiv.org/pdf/2605.28108)). A preference-aware memory update paper (PAMU) appears in ACL 2026 Findings. — [ACL Anthology](https://aclanthology.org/2026.findings-acl.38/)
- Surveys: "Memory in the Age of AI Agents: A Survey — Forms, Functions and Dynamics" (arXiv, Dec 2025) — [PDF](https://arxiv.org/pdf/2512.13564); "Always-On Agents: A Survey of Persistent Memory, State, and Governance in LLM Agents" (arXiv 2026, not read) — [PDF](https://arxiv.org/pdf/2606.30306).

### Inferences
- For a paper on evolving or lifelong memory, knowledge updates and preference change are best tested with the LongMemEval KU and temporal abilities, BEAM (contradiction resolution and KU), MemoryAgentBench (selective forgetting / FactConsolidation) and PersonaMem or PrefEval. LoCoMo has no knowledge-update category.

### Gaps
- LoCoMo's ACL 2024 venue and the MSC venue were not re-verified in this session.
- The venues of HaluMem, LoCoMo-Plus, MemOps and Supersede were not found.
- The survey contents were not read in detail, so no claims from them are made.
