# Academic long-term / lifelong conversational memory methods for LLM agents (2024-2026)

Scope: research methods only (not company frameworks). Company systems such as Mem0, Zep, LangMem and EverMemOS appear only where papers use them as baselines. Checked 2026-09-28. Venue status comes from primary sources: ACL Anthology, NeurIPS/ICLR/ICML proceedings or virtual-site pages, AAAI/DOI, and author "Accepted to X" comments on arXiv. The source used is noted each time.

## Q1. Exact title, authors, venue and year (verified)

### Takeaway
Of the 17 requested systems, 12 appeared at a peer-reviewed venue. Four are arXiv-only: MemoChat, Think-in-Memory, Nemori and Mem-alpha. Mem-alpha was submitted to ICLR 2026, but I could not confirm an acceptance. The core set of 2025-2026 "standard" conversational-memory methods with confirmed venues is A-MEM (NeurIPS 2025), MemoryOS (EMNLP 2025), SeCom and MemTree (ICLR 2025), RMM (ACL 2025), LightMem (ICLR 2026) and Memory-R1 (ACL 2026).

### Cited Findings

Requested methods:

| Method | Exact title | Authors | Venue / year (status) | PDF |
|---|---|---|---|---|
| A-MEM | A-MEM: Agentic Memory for LLM Agents | Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, Yongfeng Zhang | **NeurIPS 2025** (poster) | https://arxiv.org/pdf/2502.12110 |
| MemoryOS | Memory OS of AI Agent | Jiazheng Kang, Mingming Ji, Zhe Zhao, Ting Bai | **EMNLP 2025 main**, pp. 25961-25970 (repo says Oral) | https://aclanthology.org/2025.emnlp-main.1318.pdf |
| LightMem (ZJU) | LightMem: Lightweight and Efficient Memory-Augmented Generation | Jizhan Fang, Xinle Deng, Haoming Xu, Ziyan Jiang, Yuqi Tang, Ziwen Xu, Shumin Deng, et al. (12 authors) | **ICLR 2026** (arXiv comment and repo) | https://arxiv.org/pdf/2510.18866 |
| Memory-R1 | Memory-R1: Enhancing Large Language Model Agents to Manage and Utilize Memories via Reinforcement Learning | Sikuan Yan, Xiufeng Yang, Zuchao Huang, Ercong Nie, Zifeng Ding, Zonggen Li, Xiaowen Ma, et al. (13) | **ACL 2026 long** (2026.acl-long.583) | https://aclanthology.org/2026.acl-long.583.pdf (also https://arxiv.org/pdf/2508.19828) |
| SeCom | SeCom: On Memory Construction and Retrieval for Personalized Conversational Agents (arXiv title omits "SeCom:") | Zhuoshi Pan, Qianhui Wu, Huiqiang Jiang, Xufang Luo, Hao Cheng, Dongsheng Li, Yuqing Yang, et al. (11; Microsoft) | **ICLR 2025** | https://proceedings.iclr.cc/paper_files/paper/2025/file/e56f394bbd4f0ec81393d767caa5a31b-Paper-Conference.pdf |
| RMM | In Prospect and Retrospect: Reflective Memory Management for Long-term Personalized Dialogue Agents | Zhen Tan, Jun Yan, I-Hung Hsu, Rujun Han, Zifeng Wang, Long Le, Yiwen Song, Yanfei Chen, Hamid Palangi, George Lee, Anand Rajan Iyer, Tianlong Chen, Huan Liu, Chen-Yu Lee, Tomas Pfister (Google) | **ACL 2025 long**, pp. 8416-8439 | https://aclanthology.org/2025.acl-long.413.pdf |
| Nemori | v4 title: "What Deserves Memory: Adaptive Memory Distillation for LLM Agents" (v1 title: "Nemori: Self-Organizing Agent Memory Inspired by Cognitive Science") | Wenquan Ma, Jiayan Nan, Wenlong Wu, Yize Chen | **arXiv only** (2508.03341, v4 dated Apr 2026); no venue found | https://arxiv.org/pdf/2508.03341 |
| MemTree | From Isolated Conversations to Hierarchical Schemas: Dynamic Tree Memory Representation for LLMs | Alireza Rezazadeh, Zichao Li, Wei Wei, Yujia Bao | **ICLR 2025** | https://proceedings.iclr.cc/paper_files/paper/2025/file/0382cb76309820f71c6eacd47b36ce71-Paper-Conference.pdf |
| LD-Agent | Hello Again! LLM-powered Personalized Agent for Long-term Dialogue | Hao Li, Chenghao Yang, An Zhang, Yang Deng, Xiang Wang, Tat-Seng Chua | **NAACL 2025** (arXiv comment and repo) | https://arxiv.org/pdf/2406.05925 |
| MemoryBank (older, 2023/24) | MemoryBank: Enhancing Large Language Models with Long-Term Memory | Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, Yanlin Wang | **AAAI 2024**, vol. 38(17), pp. 19724-19731, DOI 10.1609/aaai.v38i17.29946 | https://arxiv.org/pdf/2305.10250 |
| HippoRAG (older, 2024) | HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models | Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, Yu Su | **NeurIPS 2024** (arXiv comment) | https://arxiv.org/pdf/2405.14831 |
| HippoRAG 2 | From RAG to Memory: Non-Parametric Continual Learning for Large Language Models | Bernal Jiménez Gutiérrez, Yiheng Shu, Weijian Qi, Sizhe Zhou, Yu Su | **ICML 2025**, PMLR 267:21497-21515 | https://arxiv.org/pdf/2502.14802 |
| MemoChat (older, 2023) | MemoChat: Tuning LLMs to Use Memos for Consistent Long-Range Open-Domain Conversation | Junru Lu, Siyu An, Mingbao Lin, Gabriele Pergola, Yulan He, Di Yin, Xing Sun, Yunsheng Wu | **arXiv only** (2308.08239) | https://arxiv.org/pdf/2308.08239 |
| Think-in-Memory (TiM) (older, 2023) | Think-in-Memory: Recalling and Post-thinking Enable LLMs with Long-Term Memory | Lei Liu, Xiaoyan Yang, Yue Shen, Binbin Hu, Zhiqiang Zhang, Jinjie Gu, Guannan Zhang | **arXiv only** (2311.08719) | https://arxiv.org/pdf/2311.08719 |
| COMEDY | Compress to Impress: Unleashing the Potential of Compressive Memory in Real-World Long-Term Conversations | Nuo Chen, Hongguang Li, Jianhui Chang, Juhua Huang, Baoyuan Wang, Jia Li | **COLING 2025**, pp. 755-773 | https://aclanthology.org/2025.coling-main.51.pdf |
| MemInsight | MemInsight: Autonomous Memory Augmentation for LLM Agents | Rana Salama, Jason Cai, Michelle Yuan, Anna Currey, Monica Sunkara, Yi Zhang, Yassine Benajiba (Amazon) | **EMNLP 2025 main**, pp. 33136-33152 | https://aclanthology.org/2025.emnlp-main.1683.pdf |
| Mem-alpha | Mem-α: Learning Memory Construction via Reinforcement Learning | Yu Wang, Ryuichi Takanobu, Zhiqi Liang, Yuzhen Mao, Yuanzhe Hu, Julian McAuley, Xiaojian Wu | **ICLR 2026 submission** on OpenReview (id dm42omwep1); acceptance **not verified**; treat as arXiv preprint | https://arxiv.org/pdf/2509.25911 |

Other notable 2025-2026 methods I found, all at a verified venue unless marked:

| Method | Title | Authors | Venue | PDF |
|---|---|---|---|---|
| MemGAS | From Single to Multi-Granularity: Toward Long-Term Memory Association and Selection of Conversational Agents | Derong Xu, Yi Wen, Pengyue Jia, Yingyi Zhang, Wenlin Zhang, Yichao Wang, Huifeng Guo, et al. (11) | **ICLR 2026** poster | https://arxiv.org/pdf/2505.19549 |
| MemAgent (RL, long-context) | MemAgent: Reshaping Long-Context LLM with Multi-Conv RL-based Memory Agent | Hongli Yu, Tinghong Chen, Jiangtao Feng, Jiangjie Chen, Weinan Dai, Qiying Yu, Ya-Qin Zhang, et al. (11) | **ICLR 2026 Oral** | https://arxiv.org/pdf/2507.02259 |
| MEM1 (RL, long-horizon agents) | MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents | Zijian Zhou, Ao Qu, Zhaoxuan Wu, Sunghwan Kim, Alok Prakash, Daniela Rus, Jinhua Zhao, et al. | **ICLR 2026** poster | https://arxiv.org/pdf/2506.15841 |
| PREMem | Pre-Storage Reasoning for Episodic Memory: Shifting Inference Burden to Memory for Personalized Dialogue | Sangyeop Kim, Yohan Lee, Sanghwa Kim, Hyunjong Kim, Sungzoon Cho | **Findings of EMNLP 2025** | https://aclanthology.org/2025.findings-emnlp.1204.pdf |
| THEANINE | Towards Lifelong Dialogue Agents via Timeline-based Memory Management | Kai Tzu-iunn Ong, Namyoung Kim, Minju Gwak, Hyungjoo Chae, Taeyoon Kwon, Yohan Jo, Seung-won Hwang, et al. | **NAACL 2025 long**, pp. 8631-8661 | https://arxiv.org/pdf/2406.10996 |
| APEX-MEM | APEX-MEM: Agentic Semi-Structured Memory with Temporal Reasoning for Long-Term Conversational AI | Pratyay Banerjee, Masud Moshtaghi, Shivashankar Subramanian, Amita Misra, Ankit Chadha | **ACL 2026 long**, pp. 16470-16489 | https://aclanthology.org/2026.acl-long.749.pdf |
| Chain-of-Memory (CoM) | Chain-of-Memory: Lightweight Memory Construction with Dynamic Evolution for LLM Agents | Xiucheng Xu, Bingbing Xu, Xueyun Tian, Zihe Huang, Rongxin Chen, Yunfan Li, Huawei Shen | **ACL 2026 long**, pp. 11618-11631 | https://aclanthology.org/2026.acl-long.534.pdf |
| "LightMem" (SLM variant; a different paper from ZJU LightMem) | Lightweight LLM Agent Memory with Small Language Models | Jiaquan Zhang, Chaoning Zhang, Shuxu Chen, et al. (12) | **ACL 2026 long**, pp. 12914-12929 | https://aclanthology.org/2026.acl-long.588.pdf |
| TSM | Beyond Dialogue Time: Temporal Semantic Memory for Personalized LLM Agents | Miao Su, Yucan Guo, Zhongni Hou, Long Bai, Zixuan Li, et al. (11) | **Findings of ACL 2026**, pp. 29935-29951 | https://aclanthology.org/2026.findings-acl.1496.pdf |
| H-MEM | H-MEM: Hierarchical Memory for High-Efficiency Long-Term Reasoning in LLM Agents | Haoran Sun, Shaoning Zeng, Bob Zhang | **EACL 2026 long**, pp. 341-350 | https://aclanthology.org/2026.eacl-long.15.pdf |
| Mnemis | Mnemis: Dual-Route Retrieval on Hierarchical Graphs for Long-Term LLM Memory | Zihao Tang, Xin Yu, et al. (12) | "Accepted to ACL2026" (arXiv comment only; not checked on the Anthology) | https://arxiv.org/pdf/2602.15313 |
| G-Memory (multi-agent, not conversational) | G-Memory: Tracing Hierarchical Memory for Multi-Agent Systems | Guibin Zhang, Muxin Fu, Guancheng Wan, Miao Yu, Kun Wang, Shuicheng Yan | **NeurIPS 2025** poster | https://papers.neurips.cc/paper_files/paper/2025/file/136a45cd9b841bf785625709a19c6508-Paper-Conference.pdf |
| EMem (event-centric baseline) | A Simple Yet Strong Baseline for Long-Term Conversational Memory of LLM Agents | Sizhe Zhou, Jiawei Han | **arXiv only** ("Work in progress") | https://arxiv.org/pdf/2511.17208 |
| MIRIX | MIRIX: Multi-Agent Memory System for LLM-Based Agents | Yu Wang, Xi Chen | **arXiv only**; startup-linked | https://arxiv.org/pdf/2507.07957 |
| MemGPT (older, 2023; common baseline) | MemGPT: Towards LLMs as Operating Systems | Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G. Patil, Ion Stoica, Joseph E. Gonzalez | arXiv (2310.08560); no venue checked | https://arxiv.org/pdf/2310.08560 |

Benchmarks and surveys (for context):
- LoCoMo: "Evaluating Very Long-Term Conversational Memory of LLM Agents" (Maharana et al.), ACL 2024 — https://aclanthology.org/2024.acl-long.747.pdf
- LongMemEval (Di Wu et al.), ICLR 2025 — https://arxiv.org/pdf/2410.10813
- LoCoMo-Plus, ACL 2026 — https://aclanthology.org/2026.acl-long.1150/
- Surveys: "Memory in the Age of AI Agents" (Hu et al., arXiv 2512.13564; 47 authors) — https://arxiv.org/pdf/2512.13564; "From Human Memory to AI Memory" (arXiv 2504.15965); "Rethinking Memory in LLM based Agents" (Du et al., arXiv 2505.00675); a survey at Findings of ACL 2026 (https://aclanthology.org/2026.findings-acl.2069.pdf)

Sources for the venue claims:
- arXiv API metadata (titles, authors, comments) for all arXiv IDs — [arXiv API](https://export.arxiv.org/api/query?id_list=2502.12110,2506.06326,2510.18866,2508.19828,2502.05589,2503.08026,2508.03341,2410.14052,2406.05925,2305.10250,2405.14831,2502.14802,2308.08239,2311.08719,2402.11975,2503.21760,2509.25911)
- A-MEM NeurIPS 2025 — [NeurIPS poster](https://neurips.cc/virtual/2025/poster/119020); [arXiv comment](https://arxiv.org/abs/2502.12110)
- MemoryOS EMNLP 2025 — [ACL Anthology](https://aclanthology.org/2025.emnlp-main.1318/)
- LightMem ICLR 2026 — [arXiv abs](https://arxiv.org/abs/2510.18866); [GitHub zjunlp/LightMem](https://github.com/zjunlp/LightMem)
- Memory-R1 ACL 2026 — [ACL Anthology](https://aclanthology.org/2026.acl-long.583/)
- SeCom ICLR 2025 — [OpenReview](https://openreview.net/forum?id=xKDZAW0He3); [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/e56f394bbd4f0ec81393d767caa5a31b-Abstract-Conference.html)
- RMM ACL 2025 — [ACL Anthology](https://aclanthology.org/2025.acl-long.413/)
- MemTree ICLR 2025 — [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/0382cb76309820f71c6eacd47b36ce71-Abstract-Conference.html); [OpenReview](https://openreview.net/forum?id=moXtEmCleY)
- LD-Agent NAACL 2025 — [arXiv abs](https://arxiv.org/abs/2406.05925)
- MemoryBank AAAI 2024 — [dblp](https://dblp.org/rec/conf/aaai/ZhongGGYW24.html); [ACM DL / DOI](https://dl.acm.org/doi/abs/10.1609/aaai.v38i17.29946)
- HippoRAG 2 ICML 2025 — [PMLR](https://proceedings.mlr.press/v267/gutierrez25a.html); [ICML poster](https://icml.cc/virtual/2025/poster/45585)
- COMEDY COLING 2025 — [ACL Anthology](https://aclanthology.org/2025.coling-main.51/)
- MemInsight EMNLP 2025 — [ACL Anthology](https://aclanthology.org/2025.emnlp-main.1683/)
- Mem-alpha ICLR 2026 submission — [OpenReview](https://openreview.net/forum?id=dm42omwep1)
- MemGAS ICLR 2026 — [ICLR poster](https://iclr.cc/virtual/2026/poster/10008001)
- MEM1 ICLR 2026 — [ICLR poster](https://iclr.cc/virtual/2026/poster/10008961)
- MemAgent ICLR 2026 Oral — [arXiv abs / journal-ref](https://arxiv.org/abs/2507.02259)
- PREMem Findings of EMNLP 2025 — [ACL Anthology](https://aclanthology.org/2025.findings-emnlp.1204/)
- THEANINE NAACL 2025 — [ACL Anthology](https://aclanthology.org/2025.naacl-long.435/)
- APEX-MEM — [ACL Anthology](https://aclanthology.org/2026.acl-long.749/)
- Chain-of-Memory — [ACL Anthology](https://aclanthology.org/2026.acl-long.534/)
- SLM LightMem — [ACL Anthology](https://aclanthology.org/2026.acl-long.588/)
- TSM — [ACL Anthology](https://aclanthology.org/2026.findings-acl.1496/)
- H-MEM — [ACL Anthology](https://aclanthology.org/2026.eacl-long.15/)
- G-Memory NeurIPS 2025 — [NeurIPS poster](https://neurips.cc/virtual/2025/poster/116187)
- MemoChat and TiM have no peer-reviewed venue (search found only arXiv/CoRR records) — [search aggregate: Survey_Memory_in_AI list](https://github.com/elvin-yiming-du/survey_memory_in_ai)
- Nemori has no venue — [arXiv v4 HTML](https://arxiv.org/html/2508.03341v4); [HF paper page](https://huggingface.co/papers/2508.03341)

### Inferences
- Name collisions: two different papers are called "LightMem" (ZJU, ICLR 2026, and the ACL 2026 small-language-model paper). Nemori was retitled in v4. Cite by arXiv ID or venue to avoid confusion.
- 2026 venue activity is concentrated at ACL 2026: Memory-R1, APEX-MEM, Chain-of-Memory, SLM-LightMem, TSM, Mnemis and LoCoMo-Plus.

### Gaps
- Mem-alpha: OpenReview would not render behind a verification wall, and I found no iclr.cc poster page. Its ICLR 2026 outcome is unknown.
- I found no AAAI 2026 conversational-memory method in the time available. The searches returned only arXiv preprints.
- Mnemis's ACL 2026 status rests only on the author comment. LD-Agent's NAACL 2025 status and HippoRAG's NeurIPS 2024 status also come from arXiv comments, though the LD-Agent repo title matches.
- MemGPT's venue status was not checked; it is usually cited as an arXiv preprint.

## Q2. Core idea of each method: memory unit, organisation, update/forgetting, retrieval, time handling

### Takeaway
The field splits into five design families:
1. Flat stores built at a chosen granularity: SeCom, MemInsight, PREMem, EMem.
2. Hierarchical or tiered "OS-like" stores: MemoryOS, LightMem, MemTree, H-MEM, MemGPT.
3. Graph or linked-note stores: A-MEM, HippoRAG 2, APEX-MEM, THEANINE, Mnemis.
4. Compressive or single-model memory: COMEDY, MemoChat.
5. RL-learned memory managers: Memory-R1, Mem-alpha, MEM1, MemAgent, and RMM's RL reranker.

Explicit time and changing-fact modelling is rare. The ones that do it are MemoryBank (Ebbinghaus decay), LD-Agent (time decay), MemoryOS (recency heat), THEANINE (timelines), APEX-MEM (append-only temporal graph), TSM (occurrence-time semantic timelines), Memory-R1 and Mem-alpha (UPDATE/DELETE operations), and Mem-alpha's timestamped episodic memory.

### Cited Findings
- **A-MEM**: each memory note stores the content, a timestamp, LLM-generated keywords, tags, a contextual description, an embedding and links. When a note is added, *link generation* (embedding similarity plus LLM analysis) connects it to related notes. *Memory evolution* then rewrites the contextual attributes of related older notes. This is a Zettelkasten-style graph with no explicit forgetting. — [arXiv HTML](https://arxiv.org/html/2502.12110v11)
- **MemoryOS**: three tiers.
  - STM holds "dialogue pages" {Q, R, timestamp, chain meta} in a fixed-length FIFO queue (7 pages).
  - MTM groups pages into topic "segments" (segmented paging). A page joins a segment when cosine similarity plus keyword Jaccard exceeds θ=0.6. Segments carry Heat = α·N_visit + β·L_interaction + γ·exp(−Δt/μ), and the lowest-heat segments are evicted.
  - LPM is long-term persona memory: a static user profile, a User KB, 90-dimension User Traits, and agent profile/traits, with KB and traits held in FIFO queues of 100. A segment moves from MTM to LPM when its heat exceeds τ=5.
  - Retrieval takes all of STM, a two-stage search over MTM (top-m segments, then top-k pages), and the top-10 LPM entries.
  - Recency is modelled through heat. — [EMNLP 2025 PDF](https://aclanthology.org/2025.emnlp-main.1318.pdf)
- **LightMem (ZJU)**: three stages modelled on human memory.
  - Light1 (sensory): pre-compresses input with LLMLingua-2 and segments it by topic.
  - Light2 (short-term): buffers topic segments and summarizes only when the buffer fills.
  - Light3 (long-term): "soft" updates at test time, with offline parallel consolidation during "sleep" periods.
  - Updates are decoupled from online inference. — [arXiv HTML](https://arxiv.org/html/2510.18866v4)
- **Memory-R1**: a Memory Manager LLM learns the ADD / UPDATE / DELETE / NOOP operations over an external memory bank. An Answer Agent "distills" 60 retrieved candidates down to the relevant ones before answering. Both agents are trained with outcome-reward RL (PPO or GRPO) on answer correctness. Changing facts are handled through UPDATE/DELETE. — [arXiv HTML](https://arxiv.org/html/2508.19828v5); [ACL Anthology](https://aclanthology.org/2026.acl-long.583/)
- **SeCom**: the memory unit is a *topical segment*, produced by an LLM-based (GPT-4) conversation segmentation model. Before retrieval, segments are denoised with LLMLingua-2 compression. Retrieval uses BM25 or MPNet over the compressed segments. The store is flat, with no forgetting or time model. — [arXiv HTML](https://arxiv.org/html/2502.05589v3); [OpenReview](https://openreview.net/forum?id=xKDZAW0He3)
- **RMM**:
  - *Prospective reflection* builds a topic-based memory bank by summarizing the dialogue at utterance, turn and session granularity.
  - *Retrospective reflection* trains a lightweight MLP reranker online with REINFORCE, using the LLM's citations of retrieved memories as ±1 rewards.
  - Existing memories are merged or updated as new topics arrive. — [arXiv HTML](https://arxiv.org/html/2503.08026v2); [ACL Anthology](https://aclanthology.org/2025.acl-long.413/)
- **Nemori**: (1) Episodic Memory Integration segments the conversational stream into coherent episodes (grounded in Event Segmentation Theory) and rewrites them as narrative representations. (2) Semantic Knowledge Distillation uses predict-calibrate: it extracts only what existing knowledge fails to predict (prediction error). — [arXiv v4 HTML](https://arxiv.org/html/2508.03341v4); [HF page](https://huggingface.co/papers/2508.03341)
- **MemTree**: a dynamic tree. Each node holds aggregated text, an embedding and an abstraction level that depends on depth. A new item is routed down by embedding similarity and parent summaries are re-aggregated, with O(log N) insertion. This is presented as an online alternative to RAPTOR/GraphRAG. — [arXiv HTML](https://arxiv.org/html/2410.14052v3); [OpenReview search summary](https://openreview.net/forum?id=moXtEmCleY)
- **LD-Agent**: an event memory with a long-term bank of timestamped event summaries and a short-term session cache. Retrieval combines semantic similarity, noun overlap and exponential time decay e^(−t/τ). A persona-extraction module (LoRA-tuned or zero-shot CoT) tracks user and agent traits. The design is model-agnostic. — [arXiv HTML](https://arxiv.org/html/2406.05925v2)
- **MemoryBank** (2023/AAAI 2024): writer / retriever / reader modules. Memory strength follows the Ebbinghaus forgetting curve: memories are forgotten or reinforced according to elapsed time and significance. It also builds a user portrait. The demo is the SiliconFriend companion. — [arXiv abstract via API](https://arxiv.org/abs/2305.10250)
- **HippoRAG / HippoRAG 2**:
  - OpenIE builds a knowledge graph of phrase nodes, and Personalized PageRank retrieves over it (hippocampal-indexing analogy).
  - HippoRAG 2 adds passage nodes (dense-sparse integration), "query-to-triple" linking, and an LLM "recognition memory" filter on the PPR seed triples.
  - It is framed as non-parametric continual learning over documents. It is **not** evaluated on conversational-memory benchmarks in the paper. — [arXiv HTML](https://arxiv.org/html/2502.14802v2); [arXiv abstract](https://arxiv.org/abs/2502.14802)
- **MemoChat** (2023): instruction-tunes LLMs through a "memorization-retrieval-response" cycle using self-composed structured memos. — [arXiv abstract](https://arxiv.org/abs/2308.08239)
- **Think-in-Memory** (2023): stores post-response "thoughts" rather than raw turns. Insert, forget and merge operations maintain them, and Locality-Sensitive Hashing retrieves them. — [arXiv abstract](https://arxiv.org/abs/2311.08719)
- **COMEDY**: no retrieval module or memory database. A single LLM does memory generation, compression into "compressive memory" (session summaries, user-bot dynamics, past events) and response generation. It comes with the Dolphin Chinese instruction-tuning dataset built from real user-chatbot logs. — [arXiv abstract](https://arxiv.org/abs/2402.11975); [ACL Anthology](https://aclanthology.org/2025.coling-main.51/)
- **MemInsight**: an LLM autonomously augments historical interactions to improve their semantic representation and retrieval. The finer details (attribute/annotation types) were not verified beyond the abstract. — [arXiv abstract](https://arxiv.org/abs/2503.21760); [ACL Anthology](https://aclanthology.org/2025.emnlp-main.1683/)
- **Mem-alpha**: a three-part memory. Core memory is an in-context summary of at most 512 tokens. Semantic memory holds discrete facts. Episodic memory holds timestamped events. Each part has insert/update/delete tools. A Qwen3-4B agent is trained with GRPO on a composite reward: QA correctness, tool-format validity, compression, and memory quality judged by an LM. — [arXiv HTML](https://arxiv.org/html/2509.25911v1)
- **MemGAS**: multi-granularity units (turn, session, LLM summary, keywords). A Gaussian-mixture-model step associates new memories with accept/reject sets of old ones. An entropy-based router weights the granularities for each query, and an LLM filters out redundancy. — [arXiv HTML](https://arxiv.org/html/2505.19549v2)
- **PREMem**: moves reasoning to storage time. It extracts factual, experiential and subjective fragments, and links them across sessions with evolution relations (extension, transformation, implication). — [ACL Anthology](https://aclanthology.org/2025.findings-emnlp.1204/)
- **THEANINE**: never removes memories. It links them by temporal and cause-effect relations into "memory timelines" that condition generation. It also proposes the TeaFarm counterfactual evaluation. — [ACL Anthology](https://aclanthology.org/2025.naacl-long.435/)
- **APEX-MEM**: a property graph with a domain-agnostic ontology over temporally grounded events, append-only storage that preserves how facts evolve, and a multi-tool retrieval agent that resolves conflicting or changing information at query time. — [ACL Anthology](https://aclanthology.org/2026.acl-long.749/)
- **TSM**: organizes memories by *event occurrence time* rather than dialogue time. It builds semantic timelines, consolidates them into durative memories, and uses temporal intent at retrieval. — [ACL Anthology](https://aclanthology.org/2026.findings-acl.1496/)
- **Chain-of-Memory**: uses lightweight construction plus richer use at query time. Retrieved items are organized into inference chains through "dynamic evolution" with adaptive truncation. — [ACL Anthology](https://aclanthology.org/2026.acl-long.534/)
- **H-MEM**: multi-level memory organized by semantic abstraction. Each vector carries positional indices that point to its sub-memories, so retrieval is routed layer by layer through the index. — [arXiv abstract](https://arxiv.org/abs/2507.22925); [ACL Anthology](https://aclanthology.org/2026.eacl-long.15/)
- **EMem (Simple Yet Strong Baseline)**: splits sessions into enriched elementary discourse units (event-like propositions with normalized entities and turn attribution). These go into a heterogeneous graph with dense retrieval, LLM filtering and optional graph propagation. The design is deliberately non-compressive. — [arXiv abstract](https://arxiv.org/abs/2511.17208)
- **G-Memory**: a three-tier graph (insight, query, interaction) for *multi-agent* systems, with bidirectional traversal. It is out of scope for dialogue but NeurIPS 2025. — [NeurIPS poster](https://neurips.cc/virtual/2025/poster/116187)

### Inferences
- For a new method that targets *changing facts and preferences*, the most directly comparable baselines are Memory-R1 (learned UPDATE/DELETE), APEX-MEM and TSM (temporal validity), MemoryOS (persona traits plus recency heat) and Mem-alpha (episodic timestamps). SeCom and A-MEM have no explicit handling of superseded facts.
- LLMLingua-2 compression recurs as a component (SeCom, LightMem), so it makes sense as a shared ablation.

### Gaps
- MemTree's update rule details (thresholds, re-aggregation frequency) come from summaries, not a full read.
- For MemInsight, COMEDY, TiM and MemoChat, the detailed mechanisms come from abstracts only.
- For Mnemis and SLM-LightMem, only the abstract level was checked.

## Q3. Benchmarks and headline results (metric, backbone)

### Takeaway
LoCoMo, scored with F1/BLEU-1 and increasingly an LLM judge, is the default testbed. LongMemEval_S, scored by LLM-judge accuracy, is the second standard. MSC, GVD, Long-MT-Bench+ and DialSim appear in older or specific papers. Numbers are **not comparable across papers**: judges, categories (whether adversarial questions are included), backbones and re-implementations all differ. MemoryOS, for example, reports both A-Mem's published numbers and its own re-run ("A-Mem*"), which is much lower on some categories.

### Cited Findings
- **A-MEM** (LoCoMo, GPT-4o-mini, F1): multi-hop 27.02 vs 26.65 (MemGPT); temporal 45.85 vs 25.52 (MemGPT). About 1,200 tokens per memory operation vs about 16,900 for the LoCoMo/MemGPT baselines, an 85-93% reduction. Also evaluated on DialSim. Backbones span GPT-4o/4o-mini, Qwen2.5 1.5B/3B, Llama 3.2 1B/3B, DeepSeek-R1-32B and Claude 3/3.5 Haiku. — [arXiv HTML](https://arxiv.org/html/2502.12110v11)
- **MemoryOS** (LoCoMo, GPT-4o-mini): F1 single-hop 35.27, multi-hop 41.15, temporal 20.02, open-domain 48.62. The best-baseline improvements it reports are 32.35%, 23.83%, 118.80% and 18.47%. The paper text gives average gains of 49.11% F1 and 46.18% BLEU-1, while the abstract says 48.36% F1. Efficiency: 3,874 tokens and 4.9 LLM calls per response, vs A-Mem* at 2,712 tokens and 13.0 calls, and MemGPT at 16,977 tokens. On GVD with GPT-4o-mini: retrieval accuracy 93.3, correctness 91.2, coherence 92.3, vs A-Mem at 90.4/86.5/91.4. It also reports Qwen2.5-3B and 7B. — [EMNLP PDF](https://aclanthology.org/2025.emnlp-main.1318.pdf)
- **LightMem (ZJU)**: on LongMemEval-S with GPT-4o-mini, accuracy is +2.09 to 6.40% over baselines, with 10-38× fewer tokens, 3.6-30× fewer API calls and 2.9-12.4× faster runtime. On LoCoMo with GPT-4o-mini, accuracy is +6.10 to 18.12%. It is also run on Qwen3-30B-A3B-Instruct-2507 and GLM-4.6. — [arXiv HTML v4](https://arxiv.org/html/2510.18866v4)
- **Memory-R1**: trained on only 152 QA pairs. On LoCoMo with LLaMA-3.1-8B, the GRPO variant beats MemoryOS by 28% relative F1, 34% BLEU-1 and 30% LLM-as-judge. It is evaluated on LoCoMo, MSC and LongMemEval, with Qwen-2.5 3B/7B/14B also reported. — [arXiv HTML v5](https://arxiv.org/html/2508.19828v5)
- **SeCom**: on LoCoMo with a GPT-3.5-Turbo generator and BM25 retrieval, GPT4Score is 71.57 vs 65.58 for turn-level and 65.10 for MemoChat. On Long-MT-Bench+ with MPNet retrieval, it is 88.81 vs 85.14 for MemoChat. Robustness is checked with a Mistral-7B generator. — [arXiv HTML](https://arxiv.org/html/2502.05589v3)
- **RMM** (Gemini-1.5-Flash generator): on MSC with GTE retrieval, METEOR is 33.4 and BERTScore 57.1, vs 27.5 and 52.1 for RAG. On LongMemEval, Recall@5 is 69.8 and accuracy 70.4, vs 62.4 and 63.6 for RAG. The abstract claims a >10% accuracy gain over no memory management. — [arXiv HTML](https://arxiv.org/html/2503.08026v2); [ACL Anthology](https://aclanthology.org/2025.acl-long.413/)
- **Nemori** (v4): on LoCoMo, the LLM-judge score is 80.8 with gpt-4.1-mini, vs 80.6 for Full Context. Memory-construction tokens drop 38.7% vs baselines, and generation needs 95-96% fewer tokens. On LongMemEval_S it is +16.7% vs Full Context. Integrated into A-MEM or MemoryOS, it cuts their storage by 45-64%. Backbones are GPT-4o-mini and GPT-4.1-mini. — [arXiv HTML v4](https://arxiv.org/html/2508.03341v4)
- **MemTree** (GPT-4o and Llama-3.1-70B; text-embedding-3-large or E5-Mistral):
  - MSC-E (200 rounds): 82.5% vs 80.7% for MemoryStream.
  - QuALITY: 59.8 vs 59.0 for RAPTOR.
  - MultiHop-RAG: 80.5 vs 81.0 for RAPTOR (temporal subset 68.4 vs 66.0).
  - Also evaluated on MSC. — [arXiv HTML](https://arxiv.org/html/2410.14052v3)
- **LD-Agent**: on MSC and Conversation Chronicles (CC), with ChatGLM, ChatGPT, BlenderBot and BART as backbones. On MSC session 2 with tuned ChatGLM, BLEU-2 is 10.70 vs 5.48 and ROUGE-L 23.31 vs 17.65. — [arXiv HTML](https://arxiv.org/html/2406.05925v2)
- **HippoRAG 2** (backbone and retriever not re-verified here): evaluated on NQ, PopQA, MuSiQue, 2Wiki, HotpotQA, LV-Eval and NarrativeQA. Its associative-memory gain is "7% ... over the state-of-the-art embedding model". The WebFetch summary said "77 point", which I treat as a parsing error; the abstract says 7%. — [arXiv abstract](https://arxiv.org/abs/2502.14802); [arXiv HTML](https://arxiv.org/html/2502.14802v2)
- **MemInsight**: on LLM-REDIAL, recommendation persuasiveness is up to +14%. On LoCoMo, retrieval recall is +34% over a RAG baseline. The repo lists Claude-3 Sonnet/Haiku as the models. — [arXiv abstract](https://arxiv.org/abs/2503.21760); [GitHub README](https://github.com/amazon-science/MemInsight)
- **Mem-alpha**: trained on sequences of at most 30k tokens, it generalizes to more than 400k (about 13×). On MemoryAgentBench (RULER-QA, LongMemEval, TREC, NLU, Banking77, Clinic150, InfBench-Sum), its validation average is 0.592 vs 0.461-0.567 for baselines, with about 50% memory savings vs dense retrieval. — [arXiv HTML](https://arxiv.org/html/2509.25911v1)
- **MemGAS** (GPT-4o-mini, Contriever): on LongMemEval-s, GPT-4o-judge 60.20, F1 20.38 and Recall@3 78.51, vs 55.40, 13.78 and 71.06 for Contriever. Also evaluated on LongMemEval-m, LoCoMo and Long-MT-Bench+. — [arXiv HTML](https://arxiv.org/html/2505.19549v2)
- **PREMem** (Qwen2.5 3B/14B/72B, Gemma3 4B/12B/27B, GPT-4.1 nano/mini/base): on LongMemEval with GPT-4.1, LLM-judge 71.4 vs 55.9 for A-Mem. On LoCoMo with Qwen2.5-72B, 71.0 vs 45.6 for A-Mem. — [arXiv HTML](https://arxiv.org/html/2509.10852v1)
- **APEX-MEM**: 88.88% accuracy on LoCoMo and 86.2% on LongMemEval (backbone not in abstract). — [ACL Anthology](https://aclanthology.org/2026.acl-long.749/)
- **TSM**: up to 12.2% absolute accuracy gain on LongMemEval and LoCoMo. — [ACL Anthology](https://aclanthology.org/2026.findings-acl.1496/)
- **Chain-of-Memory**: accuracy +7.5 to 10.4%, using about 2.7% of the tokens and 6.0% of the latency of "complex memory architectures". — [ACL Anthology](https://aclanthology.org/2026.acl-long.534/)
- **SLM-LightMem** (ACL 2026): about +2.5 F1 over A-MEM on LoCoMo, with 83 ms retrieval and 581 ms end-to-end latency. — [ACL Anthology](https://aclanthology.org/2026.acl-long.588/)
- **COMEDY**: evaluated on the Dolphin (Chinese) data against retrieval-based methods; the abstract gives no headline number. — [arXiv abstract](https://arxiv.org/abs/2402.11975)

### Inferences
- LoCoMo plus LongMemEval_S, with GPT-4o-mini or GPT-4.1-mini as the backbone and an LLM judge, is the de facto 2025-2026 protocol. A new method should report both. Using F1/BLEU-1 as well keeps it comparable with A-MEM, MemoryOS and Memory-R1.
- Very high 2026 LoCoMo accuracies (APEX-MEM 88.9) and Nemori's result against Full Context (80.8 vs 80.6) suggest LoCoMo is close to saturation. This helps explain the new benchmarks: LoCoMo-Plus ([ACL 2026](https://aclanthology.org/2026.acl-long.1150/)), LongMemEval-V2 ([arXiv](https://arxiv.org/html/2605.12493v1)) and BEAM (mentioned in a [Mem0 blog](https://mem0.ai/blog/ai-memory-benchmarks-in-2026), a vendor source).

### Gaps
- Several backbones for the ACL 2026 papers (APEX-MEM, TSM, CoM) are not in their abstracts; the full PDFs were not read.
- I did not check HippoRAG 2's exact per-dataset F1 numbers.

## Q4. Which baselines each paper compares against

### Takeaway
The "standard baseline set" has shifted over time:
- 2023-24 papers compared against MemoryBank, MemGPT, TiM, ReadAgent and MemoChat.
- Mid-2025 papers use A-Mem, MemoryOS, Mem0, LangMem, Zep, full context and naive RAG.
- Late 2025 and 2026 papers add SeCom, HippoRAG 2 and RAPTOR on the retrieval side, and MemAgent and MEM1 for RL memory agents.

A-MEM is the single most-used academic baseline. Mem0 is the most-used industry one.

### Cited Findings
- A-MEM compares with LoCoMo (no memory), ReadAgent, MemoryBank and MemGPT. — [arXiv HTML](https://arxiv.org/html/2502.12110v11)
- MemoryOS compares with TiM, MemoryBank, MemGPT, A-Mem (published numbers) and A-Mem* (re-run). — [EMNLP PDF](https://aclanthology.org/2025.emnlp-main.1318.pdf)
- LightMem (ZJU) compares with Full Text, Naive RAG, LangMem, A-MEM, MemoryOS and Mem0. — [arXiv HTML](https://arxiv.org/html/2510.18866v4)
- Memory-R1 compares with LoCoMo RAG, A-Mem, Mem0, MemoryOS and Memory-SFT. — [arXiv HTML](https://arxiv.org/html/2508.19828v5)
- SeCom compares with Zero History, Full History, Turn-level, Session-level, SumMem, RecurSum, ConditionMem and MemoChat. — [arXiv HTML](https://arxiv.org/html/2502.05589v3)
- RMM compares with No History, Long Context, RAG variants, MemoryBank and LD-Agent. — [arXiv HTML](https://arxiv.org/html/2503.08026v2)
- Nemori compares with Full Context, RAG-4096, LangMem, Zep, Mem0, A-MEM and MemoryOS. — [arXiv HTML](https://arxiv.org/html/2508.03341v4)
- MemTree compares with MemoryStream, MemGPT, RAPTOR, GraphRAG and naive full history. — [arXiv HTML](https://arxiv.org/html/2410.14052v3)
- LD-Agent compares with HAHT and backbones without the LD-Agent modules. — [arXiv HTML](https://arxiv.org/html/2406.05925v2)
- Mem-alpha compares with Long-Context (Qwen3-32B), RAG-Top2, MemAgent (14B) and MEM1 (7B). — [arXiv HTML](https://arxiv.org/html/2509.25911v1)
- MemGAS compares with MPNet, Contriever, MPC, RecurSum, SeCom, HippoRAG 2, RAPTOR, A-Mem and full history. — [arXiv HTML](https://arxiv.org/html/2505.19549v2)
- PREMem compares with Turn-level, Session-level, SeCom, HippoRAG-2 and A-Mem. — [arXiv HTML](https://arxiv.org/html/2509.10852v1)
- HippoRAG 2 compares with standard dense RAG (e.g. NV-Embed-v2) and structure-augmented RAG. Per the abstract, KG/structured RAG methods fall below standard RAG on factual tasks. — [arXiv abstract](https://arxiv.org/abs/2502.14802)
- H-MEM compares with "five baseline methods" on LoCoMo; they are not named in the abstract. — [ACL Anthology](https://aclanthology.org/2026.eacl-long.15/)
- SLM-LightMem compares with A-MEM, at least. — [ACL Anthology](https://aclanthology.org/2026.acl-long.588/)

### Inferences
- A defensible baseline suite for a new conversational-memory method in 2026:
  - Full Context and Naive RAG (turn- or session-level).
  - SeCom, A-MEM and MemoryOS.
  - Mem0, as the industry reference.
  - LightMem (ZJU), for efficiency.
  - Nemori or PREMem, for episodic and pre-storage reasoning.
  - Memory-R1 or Mem-alpha, if the method is learned.
  - HippoRAG 2 or MemGAS, for a graph or multi-granularity retrieval angle.
- Re-running baselines matters: MemoryOS's A-Mem* re-run differs sharply from A-Mem's reported numbers (temporal F1 8.04 vs 45.85).

### Gaps
- I did not extract baseline lists for APEX-MEM, TSM, Chain-of-Memory, Mnemis, MemInsight or COMEDY (full PDFs not read).

## Q5. Code availability, license, and ease of use with Gemini/Vertex or OpenAI-compatible endpoints

### Takeaway
Most 2025 methods have MIT or Apache-2.0 code that reads an OpenAI-style API key and a base_url, so any OpenAI-compatible server works (vLLM, OpenRouter, or Gemini's OpenAI-compatible endpoint at `https://generativelanguage.googleapis.com/v1beta/openai/`). The easiest to point at Gemini are MemoryOS, LightMem, SeCom, Nemori, HippoRAG 2 and A-mem-sys (via OpenRouter). RMM and MemTree have no public code. Mem-alpha and COMEDY have no license file.

### Cited Findings

| Method | Repo | License | Stars (Sep 2026) | LLM backend notes |
|---|---|---|---|---|
| A-MEM | https://github.com/agiresearch/A-mem ; system version https://github.com/WujiangXu/A-mem-sys (paper-eval repo WujiangXu/AgenticMemory redirects) | MIT / MIT | 1187 / 397 | A-mem: `llm_backend` openai or ollama. A-mem-sys: openai, ollama, sglang, openrouter; its README lists `google/gemini-2.0-flash-001` via OpenRouter |
| MemoryOS | https://github.com/BAI-LAB/MemoryOS | Apache-2.0 | 1590 | `openai_api_key` plus `openai_base_url` ("if using a custom OpenAI endpoint"); README lists DeepSeek, Qwen, vLLM and an MCP server |
| LightMem (ZJU) | https://github.com/zjunlp/LightMem | MIT | 1180 | MemoryManagerConfig backends: openai, deepseek, ollama, vllm, transformers; configurable `BASE_URL`; needs a local LLMLingua-2 model |
| Memory-R1 | https://github.com/yansikuan/memory-r1 (first author's account; official status inferred) | Apache-2.0 | 126 | RL training needs local open-weight models (LLaMA/Qwen) |
| SeCom | https://github.com/microsoft/SeCom | MIT | 63 | `OPENAI_API_KEY` plus `OPENAI_API_BASE` in an env file; uses LLMLingua-2 |
| Nemori | https://github.com/nemori-ai/nemori | MIT | 211 | `LLM_BASE_URL` / `EMBEDDING_BASE_URL`; README example uses OpenRouter with `google/gemini-embedding-001` |
| Mem-alpha | https://github.com/wangyu-ustc/Mem-alpha ; model https://huggingface.co/YuWangX/Memalpha-4B ; data https://huggingface.co/datasets/YuWangX/Memalpha | none detected | 227 | Memory server runs Qwen3-32B on vLLM; includes MEM1 baseline scripts |
| HippoRAG / HippoRAG 2 | https://github.com/OSU-NLP-Group/HippoRAG | MIT | 4027 | `llm_base_url` / `embedding_base_url` for OpenAI-compatible servers; vLLM; Bedrock via LiteLLM |
| LD-Agent | https://github.com/leolee99/LD-Agent | MIT | 86 | README does not mention backend configuration |
| MemoryBank | https://github.com/zhongwanjun/MemoryBank-SiliconFriend | MIT | 451 | Last push May 2023 (stale) |
| MemoChat | https://github.com/LuJunru/MemoChat | MIT | 30 | Tuned open models |
| COMEDY | https://github.com/nuochenpku/COMEDY | none detected | 25 | Fine-tuned 7B/13B-style models; Chinese data |
| MemInsight | https://github.com/amazon-science/MemInsight | NOASSERTION (custom) | 9 | README uses Claude-3 Sonnet/Haiku (Bedrock-style) |
| MemGAS | https://github.com/quqxui/MemGAS | not checked | n/a | GPT-4o-mini in the paper |
| PREMem | https://github.com/sangyeop-kim/PREMem | not checked | n/a | Multiple open and GPT backbones |
| MemAgent | https://github.com/BytedTsinghua-SIA/MemAgent | Apache-2.0 | 1110 | RL training framework |
| MEM1 | https://github.com/MIT-MI/MEM1 | MIT | 335 | vLLM inference |
| G-Memory | https://github.com/bingreeky/GMemory | none detected | 280 | n/a |
| MIRIX | https://github.com/Mirix-AI/MIRIX | Apache-2.0 | 3447 | Product-style app |
| EMem | https://github.com/KevinSRR/EMem (announced) | n/a | n/a | "will be released" |
| RMM | none found | n/a | n/a | Paper names only dependencies (PyTorch 2.4.1, Transformers 4.44.2) |
| MemTree | none found | n/a | n/a | Paper: code "will be made publicly available upon acceptance" |
| APEX-MEM, TSM, CoM, H-MEM, SLM-LightMem | no code link in the Anthology abstract pages | n/a | n/a | n/a |

Sources:
- Licenses, stars and push dates — [GitHub REST API](https://api.github.com/repos/BAI-LAB/MemoryOS) (queried for each repo above)
- Backend support came from each repo's README (fetched through the GitHub API): [A-mem-sys](https://github.com/WujiangXu/A-mem-sys), [MemoryOS](https://github.com/BAI-LAB/MemoryOS), [LightMem](https://github.com/zjunlp/LightMem), [SeCom](https://github.com/microsoft/SeCom), [Nemori](https://github.com/nemori-ai/nemori), [HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG), [Mem-alpha](https://github.com/wangyu-ustc/Mem-alpha)
- The Gemini API has an OpenAI-compatible base URL, `https://generativelanguage.googleapis.com/v1beta/openai/`, supporting chat completions and embeddings; switching needs only the API key, base URL and model name. — [Google AI docs](https://ai.google.dev/gemini-api/docs/openai)
- RMM has no repository — [arXiv HTML](https://arxiv.org/html/2503.08026v2)
- MemTree code was promised "upon acceptance" — [arXiv HTML](https://arxiv.org/html/2410.14052v3)
- Mem-alpha resources — [arXiv HTML](https://arxiv.org/html/2509.25911v1)
- MemGAS repo — [arXiv HTML](https://arxiv.org/html/2505.19549v2)
- PREMem repo — [arXiv HTML](https://arxiv.org/html/2509.10852v1)

### Inferences
- Any repo that exposes an OpenAI `base_url` should run on Gemini through the endpoint above with only config changes. This covers MemoryOS, LightMem, SeCom, Nemori, HippoRAG 2 and A-mem-sys via OpenRouter. Caveats I have not tested:
  - Embedding dimensions differ from OpenAI's.
  - Some repos hard-code OpenAI embedding model names or local sentence-transformers.
  - JSON-mode and function-calling behaviour differs across providers.
- Vertex AI is most easily reached through LiteLLM (HippoRAG already routes through LiteLLM for Bedrock) or through an OpenAI-compatible proxy.
- RMM is a Google paper evaluated with Gemini-1.5-Flash, but it has no code, so it would need to be re-implemented.
- RL methods (Memory-R1, Mem-alpha, MEM1, MemAgent) need local GPU training on open-weight models. Gemini can serve only as a frozen component or judge.

### Gaps
- Vertex AI's own OpenAI-compatibility endpoint was not verified; the Google AI page does not cover it.
- Licenses for MemGAS and PREMem, and the exact Mem-alpha license status (no LICENSE detected by the API), should be re-checked.
- I did not run any repo, so "ease of running" is inferred from READMEs only.

## Q6. Direct PDF links

### Takeaway
Every method in the catalogue has a working PDF URL. They are listed in the Q1 tables and repeated here for convenience.

### Cited Findings
- A-MEM — https://arxiv.org/pdf/2502.12110 — [arXiv](https://arxiv.org/abs/2502.12110)
- MemoryOS — https://aclanthology.org/2025.emnlp-main.1318.pdf (fetched successfully) — [ACL Anthology](https://aclanthology.org/2025.emnlp-main.1318/)
- LightMem (ZJU) — https://arxiv.org/pdf/2510.18866 — [arXiv](https://arxiv.org/abs/2510.18866)
- Memory-R1 — https://aclanthology.org/2026.acl-long.583.pdf ; https://arxiv.org/pdf/2508.19828 — [ACL Anthology](https://aclanthology.org/2026.acl-long.583/)
- SeCom — https://proceedings.iclr.cc/paper_files/paper/2025/file/e56f394bbd4f0ec81393d767caa5a31b-Paper-Conference.pdf ; https://arxiv.org/pdf/2502.05589 — [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/e56f394bbd4f0ec81393d767caa5a31b-Abstract-Conference.html)
- RMM — https://aclanthology.org/2025.acl-long.413.pdf — [ACL Anthology](https://aclanthology.org/2025.acl-long.413/)
- Nemori — https://arxiv.org/pdf/2508.03341 — [arXiv](https://arxiv.org/abs/2508.03341)
- MemTree — https://proceedings.iclr.cc/paper_files/paper/2025/file/0382cb76309820f71c6eacd47b36ce71-Paper-Conference.pdf — [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/0382cb76309820f71c6eacd47b36ce71-Abstract-Conference.html)
- LD-Agent — https://arxiv.org/pdf/2406.05925 — [arXiv](https://arxiv.org/abs/2406.05925)
- MemoryBank — https://arxiv.org/pdf/2305.10250 — [arXiv](https://arxiv.org/abs/2305.10250)
- HippoRAG — https://arxiv.org/pdf/2405.14831 — [arXiv](https://arxiv.org/abs/2405.14831)
- HippoRAG 2 — https://arxiv.org/pdf/2502.14802 — [PMLR](https://proceedings.mlr.press/v267/gutierrez25a.html)
- MemoChat — https://arxiv.org/pdf/2308.08239 — [arXiv](https://arxiv.org/abs/2308.08239)
- Think-in-Memory — https://arxiv.org/pdf/2311.08719 — [arXiv](https://arxiv.org/abs/2311.08719)
- COMEDY — https://aclanthology.org/2025.coling-main.51.pdf — [ACL Anthology](https://aclanthology.org/2025.coling-main.51/)
- MemInsight — https://aclanthology.org/2025.emnlp-main.1683.pdf — [ACL Anthology](https://aclanthology.org/2025.emnlp-main.1683/)
- Mem-alpha — https://arxiv.org/pdf/2509.25911 ; OpenReview PDF https://openreview.net/pdf/84b195754f5a425454f70a545ce1e22ee38834db.pdf — [arXiv](https://arxiv.org/abs/2509.25911)
- MemGAS — https://arxiv.org/pdf/2505.19549 — [ICLR poster](https://iclr.cc/virtual/2026/poster/10008001)
- MemAgent — https://arxiv.org/pdf/2507.02259 — [arXiv](https://arxiv.org/abs/2507.02259)
- MEM1 — https://arxiv.org/pdf/2506.15841 — [ICLR poster](https://iclr.cc/virtual/2026/poster/10008961)
- PREMem — https://aclanthology.org/2025.findings-emnlp.1204.pdf — [ACL Anthology](https://aclanthology.org/2025.findings-emnlp.1204/)
- THEANINE — https://arxiv.org/pdf/2406.10996 — [ACL Anthology](https://aclanthology.org/2025.naacl-long.435/)
- APEX-MEM — https://aclanthology.org/2026.acl-long.749.pdf — [ACL Anthology](https://aclanthology.org/2026.acl-long.749/)
- Chain-of-Memory — https://aclanthology.org/2026.acl-long.534.pdf — [ACL Anthology](https://aclanthology.org/2026.acl-long.534/)
- SLM-LightMem — https://aclanthology.org/2026.acl-long.588.pdf — [ACL Anthology](https://aclanthology.org/2026.acl-long.588/)
- TSM — https://aclanthology.org/2026.findings-acl.1496.pdf — [ACL Anthology](https://aclanthology.org/2026.findings-acl.1496/)
- H-MEM — https://aclanthology.org/2026.eacl-long.15.pdf — [ACL Anthology](https://aclanthology.org/2026.eacl-long.15/)
- Mnemis — https://arxiv.org/pdf/2602.15313 — [arXiv](https://arxiv.org/abs/2602.15313)
- G-Memory — https://papers.neurips.cc/paper_files/paper/2025/file/136a45cd9b841bf785625709a19c6508-Paper-Conference.pdf — [NeurIPS proceedings](https://proceedings.neurips.cc//paper_files/paper/2025/hash/136a45cd9b841bf785625709a19c6508-Abstract-Conference.html)
- EMem — https://arxiv.org/pdf/2511.17208 — [arXiv](https://arxiv.org/abs/2511.17208)
- MemGPT — https://arxiv.org/pdf/2310.08560 — [arXiv](https://arxiv.org/abs/2310.08560)
- Survey "Memory in the Age of AI Agents" — https://arxiv.org/pdf/2512.13564 — [arXiv](https://arxiv.org/abs/2512.13564)

### Inferences
- arXiv PDF URLs of the form https://arxiv.org/pdf/<id> resolve to the latest version. Pin a version (e.g. 2508.03341v1) where the title or content has changed, as with Nemori.

### Gaps
- I did not fetch every PDF URL individually. The arXiv IDs were confirmed through the arXiv API, and the Anthology and proceedings PDF URLs came from search results or Anthology pages. Only the MemoryOS Anthology PDF was downloaded directly.
