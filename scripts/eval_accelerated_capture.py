#!/usr/bin/env python3
"""Ordered, resumable capture: persistent OCR processes overlap one GPU batch queue."""
from __future__ import annotations
import argparse, concurrent.futures as cf, copy, hashlib, importlib.util, json, multiprocessing as mp, os, shutil, sys, time
from pathlib import Path
from datetime import datetime, timezone
import httpx

def now(): return datetime.now(timezone.utc).isoformat(timespec='seconds')
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def atomic(path,data,strict=True):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.'+str(os.getpid())+'.'+str(time.time_ns())+'.tmp')
    try:
        tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        for attempt in range(6):
            try:os.replace(tmp,path);return
            except PermissionError:
                if attempt==5:raise
                time.sleep(.05)
    except OSError:
        if strict:raise
    finally:
        if tmp.exists():tmp.unlink(missing_ok=True)
def contract():
    name='_accelerated_capture_contract'
    if name in sys.modules:return sys.modules[name]
    p=Path(__file__).with_name('eval_country_ablation.py')
    spec=importlib.util.spec_from_file_location(name,p);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

_WORKER=None
_WORKER_STATUS=None

def initialize_worker(config):
    global _WORKER,_WORKER_STATUS
    os.environ['DEEPSEEK_API_KEY']='';os.environ['GEO_AI_MODE']='mock';os.environ['M_PHOTO_GEOCODER_DISABLED']='1'
    os.environ['OMP_NUM_THREADS']=str(config['cpu_threads']);os.environ['ORT_NUM_THREADS']=str(config['cpu_threads'])
    c=contract();args=argparse.Namespace(**config['context_args']);ctx=c._load_capture_context(args)
    import backend.app.domain.geolocation.analysis.ocr as ocr
    from backend.app.services.model_gateway import clean_image
    original_params=ocr.RapidOCREngine._build_params
    def cpu_params(self):
        result=original_params(self);result.update({'EngineConfig.onnxruntime.intra_op_num_threads':config['cpu_threads'],'EngineConfig.onnxruntime.inter_op_num_threads':1});return result
    ocr.RapidOCREngine._build_params=cpu_params
    client=httpx.Client(timeout=httpx.Timeout(config['ocr_timeout'],connect=5),trust_env=False,follow_redirects=False)
    state={'last':None}
    if config['ocr_device']=='cuda':
        def remote_recognize(self,path):
            payload=Path(path).read_bytes();r=client.post(config['ocr_url']+'/recognize',content=payload,headers={'Authorization':'Bearer '+config['api_key'],'Content-Type':'application/octet-stream'});r.raise_for_status();body=r.json()
            if body.get('image_sha256') != hashlib.sha256(payload).hexdigest():raise RuntimeError('GPU OCR image identity mismatch')
            status=body.get('ocr_status') or body.get('status') or {}
            if status.get('effective_device',status.get('device'))!='cuda' or status.get('execution_provider')!='CUDAExecutionProvider':raise RuntimeError('GPU OCR did not report CUDA execution')
            actual=status.get('execution_providers') or {}
            if any(not actual.get(k) or actual[k][0]!='CUDAExecutionProvider' for k in ('det','cls','rec')):raise RuntimeError('GPU OCR session provider mismatch')
            state['last']=copy.deepcopy(status)
            result=[]
            for row in body.get('lines') or []:
                result.append(ocr.OCRLine(text=str(row['text']),confidence=float(row.get('confidence',row.get('score'))),bbox=row.get('bbox',row.get('box')),provider='rapidocr',det_model=row.get('det_model',self.profile.det_model),rec_model=row.get('rec_model',self.profile.rec_model),cls_model=row.get('cls_model',self.profile.cls_model),model_type='server',model_version='PP-OCRv5'))
            return result
        ocr.RapidOCREngine.recognize=remote_recognize
    _WORKER=(ctx,config,clean_image,state,client)

def analyze_image(index):
    ctx,config,clean,state,client=_WORKER
    entry=ctx.entries[index];started=time.perf_counter();raw_hash=sha(entry.image_path)
    clues=ctx.pipeline.analyze(str(entry.image_path));status=ctx.pipeline.ocr_status()
    if status.get('state')!='ok':raise RuntimeError('OCR failed for '+entry.id)
    if config['ocr_device']=='cuda':
        remote=state.get('last') or {};versions=remote.get('runtime')
        status={**status,**remote,'runtime_versions':versions,'runtime':'gpu','device':'cuda','requested_device':'cuda','effective_device':'cuda','execution_provider':'CUDAExecutionProvider','identity_source':'gpu_service'}
    else:status={**status,'runtime':'cpu','device':'cpu','execution_provider':'CPUExecutionProvider','identity_source':'cpu_worker','cpu_intra_op_threads':config['cpu_threads']}
    return {'index':index,'id':entry.id,'image_ref':entry.image_ref,'raw_image_sha256':raw_hash,'payload':clean(entry.image_path),'clues':clues,'ocr_status':status,'analyze_ms':round((time.perf_counter()-started)*1000,3)}

def rank_batch(gateway,url,key,rows,timeout):
    started=time.perf_counter()
    files=[('files',(r['id']+'.jpg',r['payload'],'image/jpeg')) for r in rows]
    with httpx.Client(timeout=httpx.Timeout(timeout,connect=5),trust_env=False,follow_redirects=False) as client:
        response=client.post(url+'/predict-batch',files=files,headers={'Authorization':'Bearer '+key});response.raise_for_status();results=response.json().get('results')
    if not isinstance(results,list) or len(results)!=len(rows):raise RuntimeError('StreetCLIP batch cardinality mismatch')
    return [gateway.validate(item) for item in results],round((time.perf_counter()-started)*1000,3)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source-root','manifest','image-root','catalog-db','env-file','resume-from','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--streetclip-url',default='http://127.0.0.1:8001');p.add_argument('--ocr-url',default='http://127.0.0.1:8003');p.add_argument('--ocr-device',choices=('cpu','cuda'),default='cuda');p.add_argument('--workers',type=int,default=2);p.add_argument('--cpu-threads',type=int,default=2);p.add_argument('--batch-size',type=int,default=4);p.add_argument('--ocr-timeout',type=float,default=90);p.add_argument('--batch-timeout',type=float,default=120)
    a=p.parse_args();out=Path(a.output).resolve()
    if out.exists():raise RuntimeError('Output already exists; use a new directory')
    if not 1<=a.workers<=8 or not 1<=a.batch_size<=32:raise RuntimeError('Invalid concurrency limits')
    out.mkdir(parents=True)
    os.environ['DEEPSEEK_API_KEY']='';os.environ['GEO_AI_MODE']='mock';os.environ['M_PHOTO_GEOCODER_DISABLED']='1';os.environ['M_PHOTO_EVAL_MODE']='1'
    c=contract();ctxargs={'source_root':a.source_root,'manifest':a.manifest,'image_root':a.image_root,'catalog_db':a.catalog_db,'env_file':a.env_file,'ocr_provider':'rapidocr','ocr_model_type':'server','ai_mode':'mock','streetclip_url':a.streetclip_url,'container':False,'ids':None,'limit':None}
    ctx=c._load_capture_context(argparse.Namespace(**ctxargs))
    records,oldmeta=c.load_cache(a.resume_from);c._validate_resume_records(ctx,records)
    total=len(ctx.entries);reused=len(records);key=ctx.gateway.settings.streetclip_api_key if hasattr(ctx.gateway,'settings') else ''
    if not key:
        from dotenv import dotenv_values
        vals=dotenv_values(a.env_file);key=str(vals.get('STREETCLIP_API_KEY') or '')
    from dotenv import dotenv_values
    gpu_key=str(dotenv_values(a.env_file).get('GPUOCR_API_KEY') or key)
    with httpx.Client(timeout=15,trust_env=False) as client:
        health=client.get(a.streetclip_url+'/health').json()
        if not health.get('ready') or not health.get('country_batch') or a.batch_size>int(health.get('batch_max_size',0)):raise RuntimeError('Batched model service not ready')
        ocrhealth=client.get(a.ocr_url+'/health').json() if a.ocr_device=='cuda' else {}
        if a.ocr_device=='cuda' and (not ocrhealth.get('ready') or ocrhealth.get('execution_provider')!='CUDAExecutionProvider'):raise RuntimeError('GPU OCR service not ready')
    atomic(out/'preflight.json',{'model':health,'ocr':ocrhealth,'observed_at':now()})
    source=Path(a.source_root).resolve();before=c._capture_input_snapshot(argparse.Namespace(**ctxargs),source)
    original=Path(a.resume_from)/'cache.jsonl';partial=out/'cache.jsonl.partial';shutil.copyfile(original,partial)
    if sha(partial)!=oldmeta['cache_sha256']:raise RuntimeError('Copied prefix mismatch')
    config={'context_args':ctxargs,'ocr_device':a.ocr_device,'ocr_url':a.ocr_url.rstrip('/'),'cpu_threads':a.cpu_threads,'ocr_timeout':a.ocr_timeout,'api_key':gpu_key}
    started=time.perf_counter();started_at=now();written=0;next_submit=reused;pending={};status='running';error=None
    def progress():
        elapsed=time.perf_counter()-started;completed=reused+written;rate=written/elapsed if elapsed>0 else 0
        data={'schema_version':1,'kind':'accelerated_country_capture','status':status,'started_at':started_at,'updated_at':now(),'pid':os.getpid(),'total':total,'reused':reused,'new_completed':written,'completed':completed,'remaining':total-completed,'elapsed_seconds':round(elapsed,3),'new_samples_per_second':round(rate,5),'new_seconds_per_sample':round(elapsed/written,4) if written else None,'eta_minutes':round((total-completed)/rate/60,1) if rate else None,'ocr_device':a.ocr_device,'workers':a.workers,'cpu_threads':a.cpu_threads,'batch_size':a.batch_size,'error_type':error,'partial':str(partial)}
        atomic(out/'progress.json',data,strict=False)
        return data
    progress();atomic(out/'resume-checkpoint.json',{'status':'running','reused':reused,'source_cache':str(original),'source_cache_sha256':sha(original),'snapshot_before':before})
    try:
        with cf.ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('spawn'),initializer=initialize_worker,initargs=(config,)) as pool,cf.ThreadPoolExecutor(max_workers=1) as gpu,partial.open('a',encoding='utf-8',newline='\n') as handle:
            window=max(a.batch_size*2,a.workers*2)
            def fill():
                nonlocal next_submit
                while next_submit<total and len(pending)<window:
                    pending[next_submit]=pool.submit(analyze_image,next_submit);next_submit+=1
            fill();gpu_work=None
            for start in range(reused,total,a.batch_size):
                indices=list(range(start,min(total,start+a.batch_size)))
                rows=[pending.pop(i).result() for i in indices];fill()
                # GPU inference overlaps the next window of persistent OCR workers.
                predictions,batch_ms=gpu.submit(rank_batch,ctx.gateway,a.streetclip_url.rstrip('/'),key,rows,a.batch_timeout).result()
                for item,evidence in zip(rows,predictions):
                    evidence={**evidence,'schema_version':1,'provider_status':'ok','prompt_version':ctx.version_info['prompt_version'],'image_sha256':hashlib.sha256(item['payload']).hexdigest()}
                    record={'schema_version':1,'kind':'country_ablation_sample','id':item['id'],'image_ref':item['image_ref'],'raw_image_sha256':item['raw_image_sha256'],'model_image_sha256':evidence['image_sha256'],'model_id':ctx.version_info['model_id'],'model_revision':ctx.version_info['model_revision'],'prompt_version':ctx.version_info['prompt_version'],'prompt_templates':ctx.version_info['prompts'],'catalog_revision':ctx.version_info['catalog_revision'],'catalog_revision_stored':ctx.version_info['catalog_revision_stored'],'legacy_semantic_sha256':ctx.provenance['legacy_semantic_sha256'],'legacy_scorer_sha256':ctx.provenance['legacy_scorer_sha256'],'legacy_engine_sha256':ctx.provenance['legacy_engine_sha256'],'ocr_status':item['ocr_status'],'evidence':evidence,'clues':item['clues'],'capture_timing_ms':{'analyze_ms':item['analyze_ms'],'batch_model_ms':batch_ms,'batch_size':len(rows)}}
                    record=c.seal_record(record);handle.write(json.dumps(record,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n');records.append(record);written+=1
                handle.flush();os.fsync(handle.fileno());progress();print(f'capture {reused+written}/{total} reused={reused} new={written}',flush=True)
        after=c._capture_input_snapshot(argparse.Namespace(**ctxargs),source)
        comparison=c._compare_capture_snapshots(before,after)
        if before['source_files']!=after['source_files'] or before['catalog']!=after['catalog'] or before['manifest']!=after['manifest']:raise RuntimeError('Capture inputs changed')
        if [r['id'] for r in records]!=[e.id for e in ctx.entries]:raise RuntimeError('Final sample order mismatch')
        if len(records)!=total or len({r['id'] for r in records})!=total:raise RuntimeError('Final sample count mismatch')
        final=out/'cache.jsonl';os.replace(partial,final)
        resume=c._resume_info(source_dir=Path(a.resume_from).resolve(),source_sha256=oldmeta['cache_sha256'],source_meta=oldmeta,context=ctx,reused_count=reused,total_count=total)
        resume['capture_segments']=[{'first_index':0,'count':reused,'ocr_device':'per_record','source':'reused_prefix'},{'first_index':reused,'count':written,'ocr_device':a.ocr_device,'source':'accelerated_capture'}]
        snapshot_path=out/c.CAPTURE_SNAPSHOT_FILENAME
        atomic(snapshot_path,{'schema_version':1,'kind':'country_ablation_capture_snapshot','started_at':started_at,'finished_at':now(),'before':before,'after':after,'comparison':comparison})
        meta=c._build_capture_meta(ctx,records,sha(final),resume=resume,snapshot_info={'file':c.CAPTURE_SNAPSHOT_FILENAME,'sha256':sha(snapshot_path)})
        meta['config'].update({'applies_to_records':'new_records_only','ocr_config_scope':'per_record','ocr_device':a.ocr_device,'workers':a.workers,'cpu_threads':a.cpu_threads,'model_batch_size':a.batch_size,'mixed_runtime':a.ocr_device!='cpu'})
        meta['provenance']['accelerated_capture_sha256']=sha(__file__);atomic(out/'meta.json',meta)
        status='completed';progress();atomic(out/'resume-checkpoint.json',{'status':status,'completed':total,'reused':reused,'new_completed':written,'cache_sha256':meta['cache_sha256'],'snapshot_before':before,'snapshot_after':after})
        print('Capture completed; ready for offline replay',flush=True)
    except BaseException as exc:
        status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed';error=type(exc).__name__;data=progress();atomic(out/'resume-checkpoint.json',{**data,'snapshot_before':before},strict=False);raise

if __name__=='__main__':
    try:main()
    except Exception as exc:print('capture failed: '+type(exc).__name__+': '+str(exc),file=sys.stderr,flush=True);raise SystemExit(1)
