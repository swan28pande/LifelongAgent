"""Replay actual prompts at five-user-equivalent API concurrency, without CPU retrieval contention."""

import argparse
import asyncio
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
from types import SimpleNamespace

from pilot import Recorder, Usage, config, create, Instance, load, LLM, save, summarize
from experiments.core.grading import JUDGE_SYSTEM, judge_prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-run', required=True)
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    for name in (args.source_run,args.run):
        if Path(name).name != name:
            parser.error('Run names must be directory names')
    source = Path(__file__).parent / args.source_run
    if not json.loads((source / 'run_summary.json').read_text())['complete']:
        raise ValueError('The source pilot must finish successfully first')
    output = Path(__file__).parent / args.run
    output.mkdir(exist_ok=True)
    if (output / 'events.jsonl').exists():
        raise ValueError('Use a fresh run name')
    recorder = Recorder(output)
    timeline = load(users=['u5'])[0]
    questions = timeline.checkpoints[0].questions
    instance = Instance(timeline.id,[],[],timeline.name)
    names = ('timem','naive_rag','full_context')
    methods, judges, prompts, phases = {}, {}, {}, []
    frozen = {(row['method'],row['question_id']): row['response']
              for row in json.loads((source / 'fixed_judge_inputs.json').read_text())}
    loop = asyncio.new_event_loop()
    start = time.monotonic()

    class CapturePrompt:
        def invoke(self,messages):
            self.messages = messages
            return SimpleNamespace(content='')

    for name in names:
        method = create(name,source/name/'store',instance,config.MODEL,Usage(recorder))
        methods[name] = method
        llm, capture = method.llm, CapturePrompt()
        method.llm = capture
        prompts[name] = []
        try:
            for question in questions:
                method.answer(question)
                prompts[name].append(capture.messages)
        finally:
            method.llm = llm
        recorder.attach(llm,name)
        judges[name] = LLM(config.JUDGE_MODEL,0.0,output/f'{name}_judge_log.jsonl')
        judges[name].chat.callbacks = [Usage(recorder)]
        recorder.attach(judges[name].chat,name)
    manifest = json.loads((source/'manifest.json').read_text())
    manifest.update({
        'purpose': 'API-only prompt replay at five-user-equivalent concurrency',
        'source_run': args.source_run, 'rounds': 1,
        'answer_requests_per_method':40,'judge_requests_per_method':80,
        'peak_answer_calls':120,'peak_judge_calls':240,
        'planned_measured_requests':720,
        'prompt_capture': 'actual baseline retrieval once per original question; no fake answers used',
        'limitation': 'repeated first-month prompts; one burst per condition, not sustained or 15 real workers',
    })
    save(output/'manifest.json',manifest)

    def meta(name,stage,condition,index):
        slots = 8 if stage=='answer' else 16
        return {'method':name,'stage':stage,'condition':condition,'round':1,
                'question_id':questions[index%len(questions)].id,'replica':index,
                'simulated_user_slot':index//slots}

    def answers(selected,condition):
        def one(name,index):
            recorder.call(meta(name,'answer',condition,index),
                          lambda:methods[name].llm.invoke(prompts[name][index%len(questions)]))
        with ThreadPoolExecutor(max_workers=40*len(selected)) as pool:
            futures=[pool.submit(one,name,index) for name in selected for index in range(40)]
            errors=[]
            for future in futures:
                try: future.result()
                except Exception as error: errors.append(type(error).__name__)
        return errors

    async def grades(selected,condition):
        async def one(name,index):
            question=questions[index%len(questions)]
            verdict,_=await recorder.acall(meta(name,'judge',condition,index),
                lambda:judges[name].json(JUDGE_SYSTEM,judge_prompt(question,frozen[name,question.id]),
                                        f'{condition}/{index}/{question.id}'))
            if verdict.get('verdict') not in ('RIGHT','WRONG'):
                raise ValueError('Invalid judge verdict')
        results=await asyncio.gather(*(one(name,index) for name in selected for index in range(80)),
                                     return_exceptions=True)
        return [type(result).__name__ for result in results if isinstance(result,BaseException)]

    def phase(selected,stage,condition):
        before=time.monotonic()
        errors=answers(selected,condition) if stage=='answer' else loop.run_until_complete(grades(selected,condition))
        elapsed=time.monotonic()-before
        for name in selected:
            row=summarize(recorder.events,{'method':name,'stage':stage,'condition':condition,'round':1},elapsed)
            phases.append(row)
            print('PHASE',json.dumps(row),flush=True)
        save(output/'phases.json',phases)
        if errors:
            raise RuntimeError(f'Exhausted failures at {stage}/{condition}: {Counter(errors)}; stopping higher load')

    failure=None
    try:
        for name in names:
            recorder.call(meta(name,'answer','warmup',0),lambda:methods[name].llm.invoke(prompts[name][0]))
            loop.run_until_complete(recorder.acall(meta(name,'judge','warmup',0),
                lambda:judges[name].json(JUDGE_SYSTEM,judge_prompt(questions[0],frozen[name,questions[0].id]),'warmup')))
        for name in names:
            phase([name],'answer','solitary')
        phase(names,'answer','concurrent')
        for name in reversed(names):
            phase([name],'judge','solitary')
        phase(names,'judge','concurrent')
    except BaseException as error:
        failure={'type':type(error).__name__,'message':str(error)[:500]}
        raise
    finally:
        save(output/'run_summary.json',{
            'complete':failure is None,'failure':failure,'wall_seconds':time.monotonic()-start,
            'http_statuses':dict(Counter(str(e['status']) for e in recorder.events if e['kind']=='http')),
            'input_tokens':sum(e.get('input_tokens',0) for e in recorder.events if e['kind']=='tokens'),
            'output_tokens':sum(e.get('output_tokens',0) for e in recorder.events if e['kind']=='tokens'),
        })
        for method in methods.values():
            if hasattr(method,'close'): method.close()
            method.llm.client.close()
        for judge in judges.values():
            loop.run_until_complete(judge.chat.client.aio.aclose())
            judge.chat.client.close()
        loop.close()
        print('SCALED_PILOT_FINISHED',output,flush=True)


if __name__=='__main__':
    main()
