"""Short final probe after the user requested an ASAP answer; partial TiMem context is explicit."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
from types import SimpleNamespace

from pilot import Recorder, Usage, config, create, Instance, load, LLM, save, summarize
from experiments.core.grading import JUDGE_SYSTEM, judge_prompt


source=Path(__file__).parent/'u5_march_20261004_c'
output=Path(__file__).parent/'quick_api'
output.mkdir(exist_ok=True)
recorder=Recorder(output)
timeline=load(users=['u5'])[0]
question=timeline.checkpoints[0].questions[0]
instance=Instance(timeline.id,[],[],timeline.name)
names=('timem','naive_rag','full_context')
methods,judges,prompts,phases={}, {}, {}, []
loop=asyncio.new_event_loop()
start=time.monotonic()

class Capture:
    def invoke(self,messages):
        self.messages=messages
        return SimpleNamespace(content='')

for name in names:
    method=create(name,source/name/'store',instance,config.MODEL,Usage(recorder))
    methods[name]=method
    llm,was_pending=method.llm,getattr(method,'pending',False)
    capture=Capture()
    # Capture real retrieval messages from completed nodes, without claiming
    # the interrupted TiMem hierarchy is finalized or evaluating its accuracy.
    method.llm=capture
    if name=='timem': method.pending=False
    try: method.answer(question)
    finally:
        method.llm=llm
        if name=='timem': method.pending=was_pending
    prompts[name]=capture.messages
    llm.timeout,llm.max_retries=20,1
    recorder.attach(llm,name)
    judges[name]=LLM(config.JUDGE_MODEL,0.0,output/f'{name}_judge_log.jsonl')
    judges[name].chat.callbacks=[Usage(recorder)]
    judges[name].chat.timeout,judges[name].chat.max_retries=20,1
    recorder.attach(judges[name].chat,name)

manifest=json.loads((source/'manifest.json').read_text())
manifest.update({'purpose':'ASAP short API probe; one real prompt per stream',
    'source_run':source.name,'rounds':1,'combined_concurrency':3,
    'question_id':question.id,'timeout_seconds':20,'sdk_attempts':1,
    'timem_preparation_complete':False,
    'limitation':'Partial TiMem nodes; warmed-order bias; three requests, not 120/240; no accuracy benchmark'})
save(output/'manifest.json',manifest)

def meta(name,stage,condition):
    return {'method':name,'stage':stage,'condition':condition,'round':1,'question_id':question.id}

def answer(name,condition):
    message=recorder.call(meta(name,'answer',condition),lambda:methods[name].llm.invoke(prompts[name]))
    content=message.content
    if isinstance(content,list): content=' '.join(block.get('text','') for block in content if isinstance(block,dict))
    recorder.emit('answer',**meta(name,'answer',condition),response=content)
    return str(content)

frozen={}

def phase(selected,stage,condition):
    before=time.monotonic()
    if stage=='answer':
        with ThreadPoolExecutor(max_workers=len(selected)) as pool:
            futures={name:pool.submit(answer,name,condition) for name in selected}
            for name,future in futures.items():
                try:
                    response=future.result()
                    if condition=='solitary': frozen[name]=response
                except Exception: pass
    else:
        async def one(name):
            if name not in frozen: return
            try:
                await recorder.acall(meta(name,'judge',condition),
                    lambda:judges[name].json(JUDGE_SYSTEM,judge_prompt(question,frozen[name]),condition))
            except Exception: pass
        async def batch(): await asyncio.gather(*(one(name) for name in selected))
        loop.run_until_complete(batch())
    elapsed=time.monotonic()-before
    for name in selected:
        row=summarize(recorder.events,meta(name,stage,condition),elapsed)
        phases.append(row)
        print('PHASE',json.dumps(row),flush=True)
    save(output/'phases.json',phases)

try:
    for name in names: phase([name],'answer','solitary')
    phase(names,'answer','concurrent')
    for name in names: phase([name],'judge','solitary')
    phase(names,'judge','concurrent')
finally:
    summary={'wall_seconds':time.monotonic()-start,'phases':phases,
        'http_statuses':{},'complete':len(phases)==12}
    from collections import Counter
    summary['http_statuses']=dict(Counter(str(e['status']) for e in recorder.events if e['kind']=='http'))
    save(output/'summary.json',summary)
    for method in methods.values():
        if hasattr(method,'close'): method.close()
        method.llm.client.close()
    for judge in judges.values():
        loop.run_until_complete(judge.chat.client.aio.aclose())
        judge.chat.client.close()
    loop.close()
    print('QUICK_PROBE_FINISHED',output,flush=True)
