# Prepare and validate TiMem locally

Status: resolved
Type: task

The earlier pilot timed out twice during L3 generation at low concurrency, and
separate diagnostics recorded HTTP 504 responses. The existing wrapper only
times out the whole operation; streaming failures are not retried at the node
boundary. Inspect the installed transport and implement lean bounded recovery,
then run focused offline checks and one real monthly checkpoint with the judge.

Deliver a local command with explicit concurrency, project-local caches/logs,
the correct interpreter, and documented resume behavior.

## Comments

The local Python environment satisfies the TiMem requirements and its 111 installed
packages are compatible. ADC refresh succeeded for `stb-prj-symb-reg-ai-rfbe`.
The cached Nomic model produces 768-dimensional vectors with network access disabled.
64 offline checks passed with local IPC enabled; restricted sandbox IPC prevented
async callback threads from waking their event loop during initial test attempts.

The first real monthly check used a 45-second transport deadline. One L3 API failure
recovered automatically, but another L3 operation exhausted three attempts with
HTTP 504 responses. The exact saved prompt then completed in 31.90 seconds using
a fresh client and a 120-second deadline. This does not establish timeout causality;
use the more generous 120-second request limit while retaining the native layer
deadline and bounded retries. A fresh unchanged-CLI monthly check is required.

## Answer

The fresh CLI run with the 120-second request deadline completed u5/March:
13 L1, 13 L2, 13 L3, six L4 and one L5 node, followed by six answers and six
successful LLM grades. Accuracy was 6/6; no retry warnings occurred. Recorded
stage time was 713.02 seconds. A second invocation skipped the completed checkpoint
without any model calls or changed user files.

Final transport inspection confirmed the installed SDK uses aiohttp. Added bounded
recovery for its connection/payload errors and Google auth network errors, while
invalid credentials still fail immediately. All 70 offline tests passed after that
addition. The validation report preserves the live implementation hashes and notes
this later error-path coverage; successful generation settings were unchanged.

The local launcher reuses the validated environment/cache and defaults to five
parallel users with one answer and one judge request per user. The full dataset run
is left for the user's command. No other methods, agent code or datasets changed.
