"""Two bounded calls with the same stalled prompt through fresh sync/stream clients."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

from pilot import Recorder, Usage, config, save
from setup_2.baselines._common import make_llm
from langchain_core.messages import HumanMessage


directory=Path(__file__).parent/'stream_diagnostic'
directory.mkdir(exist_ok=True)
text=(Path(__file__).parent/'u5_march_20261004_b/timem/store/logs/prompts_l3.jsonl').read_text()
decoder=json.JSONDecoder()
while text.strip():
    record,end=decoder.raw_decode(text.lstrip())
    text=text.lstrip()[end:]
prompt=record['prompt']
recorder=Recorder(directory)
models={name:make_llm(config.MODEL,Usage(recorder)) for name in ('sync','stream')}
for name,llm in models.items():
    llm.timeout=45
    llm.max_retries=1
    recorder.attach(llm,name)

def synchronous():
    return recorder.call({'stage':'sync','condition':'diagnostic'},
                         lambda:models['sync'].invoke([HumanMessage(content=prompt)]))

async def streaming():
    async def collect():
        async for chunk in models['stream'].astream([HumanMessage(content=prompt)]):
            pass
    return await recorder.acall({'stage':'stream','condition':'diagnostic'},
                               lambda:asyncio.wait_for(collect(),timeout=60))

start=time.monotonic()
with ThreadPoolExecutor(max_workers=1) as pool:
    future=pool.submit(synchronous)
    try: asyncio.run(streaming())
    except Exception as error: print('STREAM_ERROR',type(error).__name__,flush=True)
    try: future.result()
    except Exception as error: print('SYNC_ERROR',type(error).__name__,flush=True)
save(directory/'summary.json',{
    'model':config.MODEL,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
    'prompt_chars':len(prompt),'timeout_seconds':45,'retry_attempts':1,
    'wall_seconds':time.monotonic()-start,
    'logical_requests':[event for event in recorder.events if event['kind']=='logical'],
})
for model in models.values(): model.client.close()
print((directory/'summary.json').read_text(),flush=True)
