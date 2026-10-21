#!/usr/bin/env python3
"""Freeze inputs and supervise accelerated three-arm capture/replay with heartbeat."""
from __future__ import annotations
import argparse,hashlib,json,os,shutil,subprocess,sys,threading,time
from pathlib import Path
from datetime import datetime,timezone

def now():return datetime.now(timezone.utc).isoformat(timespec='seconds')
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def write(path,data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
 tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
 for attempt in range(8):
  try:os.replace(tmp,path);return
  except PermissionError:
   if attempt==7:raise
   time.sleep(.05)
def copy_source(src,dst):
 dst.mkdir(parents=True)
 for rel in ('scripts','backend/app','services'):
  shutil.copytree(src/rel,dst/rel,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.pyo'))
 for rel in ('backend/__init__.py','catalog_db.py','config.py'):
  if (src/rel).is_file():(dst/rel).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src/rel,dst/rel)
def hashes(root):
 return {p.relative_to(root).as_posix():sha(p) for p in root.rglob('*.py') if '__pycache__' not in p.parts}
def main():
 p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--repo-root',default=str(Path(__file__).resolve().parents[1]));p.add_argument('--resume-from',default=None);p.add_argument('--prepare',action='store_true');p.add_argument('--execute',action='store_true');p.add_argument('--replay-only',action='store_true');a=p.parse_args()
 root=Path(a.repo_root).resolve();run=Path(a.run_dir).resolve();control=run/'control';jobfile=control/'job.json'
 if a.prepare:
  if run.exists():raise RuntimeError('Run directory already exists')
  control.mkdir(parents=True);(run/'inputs').mkdir();(run/'frozen').mkdir()
  legacy=run/'frozen/legacy';revised=run/'frozen/revised'
  copy_source(root.parent/'m_photo-eval-frozen',legacy);copy_source(root.parent/'m_photo-ocr-evidence',revised)
  for name in ('eval_country_ablation.py','eval_accelerated_capture.py'):shutil.copyfile(root/'scripts'/name,legacy/'scripts'/name)
  shutil.copyfile(root/'scripts/eval_country_ablation.py',revised/'scripts/eval_country_ablation.py')
  resume_source=Path(a.resume_from).resolve() if a.resume_from else root/'reports/eval-acceleration-20261021/recovered1283'
  resume_meta=json.loads((resume_source/'meta.json').read_text(encoding='utf-8'));reused=int(resume_meta['sample_count'])
  shutil.copytree(resume_source,run/'inputs/reused')
  for src,name in ((root/'data/eval/manifest.img2gps3k.jsonl','manifest.jsonl'),(root/'data/geo_catalog.sqlite3','catalog.sqlite3')):shutil.copyfile(src,run/'inputs'/name)
  shutil.copyfile(__file__,control/'runner.py')
  files={str(run/'inputs/manifest.jsonl'):sha(run/'inputs/manifest.jsonl'),str(run/'inputs/catalog.sqlite3'):sha(run/'inputs/catalog.sqlite3')}
  for directory in (legacy,revised):
   files.update({str(directory/rel):value for rel,value in hashes(directory).items()})
  live={str(root/rel):sha(root/rel) for rel in ('inference/streetclip/predictor.py','inference/streetclip/service.py','inference/streetclip/app.py','inference/streetclip/config.py','inference/gpuocr/runtime.py')}
  write(control/'snapshot.json',{'created_at':now(),'frozen_files':files,'live_inference_files':live})
  job={'schema_version':1,'kind':'accelerated_three_arm_eval','status':'prepared','repo_root':str(root),'run_dir':str(run),'prepared_at':now(),'expected_samples':2997,'reused_samples':reused,'new_samples':2997-reused,'ocr_device_new':'cuda','ocr_device_reused':'per_record','mixed_runtime':True,'workers':2,'batch_size':4,'geocoder_enabled':False,'llm_enabled':False,'capture':str(run/'capture'),'replay':str(replay_dir),'automatic_restart':False}
  write(jobfile,job);print('Prepared '+str(run));return
 if not a.execute:raise RuntimeError('Use --prepare or --execute')
 job=json.loads(jobfile.read_text(encoding='utf-8'));reused=int(job['reused_samples']);snapshot=json.loads((control/'snapshot.json').read_text(encoding='utf-8'))
 if job['status']!='prepared' and not (a.replay_only and job['status']=='failed' and job['stage']=='replay'):raise RuntimeError('Job is not prepared or a failed replay')
 attempt=int(job.get('replay_retry_count',0))+(1 if a.replay_only else 0)
 replay_dir=run/('replay-retry-'+str(attempt) if a.replay_only else 'replay')
 if a.replay_only:
  write(control/('job.before-replay-retry-'+str(attempt)+'.json'),job)
  job.update({'replay_retry_count':attempt,'replay_restarted_at':now(),'replay':str(replay_dir)})
  for field in ('error','error_type','finished_at'):job.pop(field,None)
 for name,value in snapshot['frozen_files'].items():
  if sha(name)!=value:raise RuntimeError('Frozen input changed: '+name)
 for name,value in snapshot['live_inference_files'].items():
  if not a.replay_only and sha(name)!=value:raise RuntimeError('Inference source changed before launch: '+name)
 env=os.environ.copy();env.update({'GEO_AI_MODE':'mock','DEEPSEEK_API_KEY':'','M_PHOTO_GEOCODER_DISABLED':'1','M_PHOTO_EVAL_MODE':'1','M_PHOTO_OCR_MODEL_DIR':str(root/'backend/storage/ocr-models'),'PYTHONDONTWRITEBYTECODE':'1','PYTHONUTF8':'1'})
 py=str(root/'.venv/Scripts/python.exe');legacy=run/'frozen/legacy';revised=run/'frozen/revised';progress=run/'capture/progress.json'
 command=[py,'-B',str(legacy/'scripts/eval_accelerated_capture.py'),'--source-root',str(legacy),'--manifest',str(run/'inputs/manifest.jsonl'),'--image-root',str(root/'data/eval'),'--catalog-db',str(run/'inputs/catalog.sqlite3'),'--env-file',str(root/'.env'),'--resume-from',str(run/'inputs/reused'),'--output',str(run/'capture'),'--streetclip-url','http://127.0.0.1:8001','--ocr-url','http://127.0.0.1:8003','--ocr-device','cuda','--workers','2','--cpu-threads','2','--batch-size','4']
 replay=[py,'-B',str(revised/'scripts/eval_country_ablation.py'),'replay','--cache',str(run/'capture'),'--manifest',str(run/'inputs/manifest.jsonl'),'--image-root',str(root/'data/eval'),'--catalog-db',str(run/'inputs/catalog.sqlite3'),'--high-margin','0.02','--medium-margin','0.005','--output',str(replay_dir)]
 job.update({'status':'running','stage':'replay' if a.replay_only else 'capture','started_at':job.get('started_at',now()) if a.replay_only else now(),'supervisor_pid':os.getpid(),'supervisor_script_sha256':sha(__file__),'capture_command':command,'replay_command':replay,'progress':str(progress),'updated_at':now()})
 (control/'supervisor.pid').write_text(str(os.getpid())+'\n',encoding='ascii');write(jobfile,job)
 pointer={'schema_version':1,'kind':'accelerated_three_arm_eval','status':'running','run_dir':str(run),'job':str(jobfile),'progress':str(progress),'capture_log':str(control/'capture.stdout.log'),'stderr_log':str(control/'capture.stderr.log'),'report':str(replay_dir/'report.md'),'supervisor_pid':os.getpid(),'updated_at':now(),'reused_samples':reused,'mixed_runtime':True}
 pointerfile=root/'reports/ocr-evidence-ablation-20261020T205803/ACTIVE_RUN.json';write(pointerfile,pointer);(pointerfile.parent/'ACTIVE_RUN.txt').write_text(str(run)+'\n',encoding='utf-8')
 stop=threading.Event();lock=threading.Lock()
 def update():
  while not stop.wait(5):
   try:
    data=json.loads(progress.read_text(encoding='utf-8')) if progress.is_file() else {}
    with lock:
     job.update({'updated_at':now(),'completed_samples':data.get('completed',reused),'new_completed':data.get('new_completed',0),'eta_minutes':data.get('eta_minutes'),'new_seconds_per_sample':data.get('new_seconds_per_sample')});write(jobfile,job)
     pointer.update({'status':job['status'],'updated_at':job['updated_at'],'stage':job['stage'],'completed_samples':job.get('completed_samples'),'eta_minutes':job.get('eta_minutes')});write(pointerfile,pointer)
   except (OSError,ValueError):pass
 thread=threading.Thread(target=update,daemon=True);thread.start()
 try:
  if not a.replay_only:
   with (control/'capture.stdout.log').open('w',encoding='utf-8') as out,(control/'capture.stderr.log').open('w',encoding='utf-8') as err:
    proc=subprocess.Popen(command,cwd=str(legacy),env=env,stdout=out,stderr=err)
    with lock:job['capture_pid']=proc.pid;write(jobfile,job)
    code=proc.wait()
   if code:raise RuntimeError('Capture exited '+str(code))
  cachemeta=json.loads((run/'capture/meta.json').read_text(encoding='utf-8'))
  if cachemeta.get('sample_count')!=2997 or cachemeta.get('resume',{}).get('reused_count')!=reused:raise RuntimeError('Final cache counts mismatch')
  with lock:job.update({'stage':'replay','capture_exitcode':0,'updated_at':now()});write(jobfile,job)
  with (control/('replay-retry-'+str(attempt)+'.stdout.log' if a.replay_only else 'replay.stdout.log')).open('w',encoding='utf-8') as out,(control/('replay-retry-'+str(attempt)+'.stderr.log' if a.replay_only else 'replay.stderr.log')).open('w',encoding='utf-8') as err:code=subprocess.call(replay,cwd=str(revised),env=env,stdout=out,stderr=err)
  if code:raise RuntimeError('Replay exited '+str(code))
  report=json.loads((replay_dir/'report.json').read_text(encoding='utf-8'))
  if report.get('cache',{}).get('sample_count')!=2997 or report.get('failures'):raise RuntimeError('Replay report incomplete or contains arm failures')
  for name,value in snapshot['frozen_files'].items():
   if sha(name)!=value:raise RuntimeError('Frozen input changed during run')
  with lock:job.update({'status':'completed','stage':'complete','completed_samples':2997,'report':str(replay_dir/'report.md'),'report_sha256':sha(replay_dir/'report.json'),'finished_at':now(),'updated_at':now()})
 except BaseException as exc:
  with lock:job.update({'status':'interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed','error_type':type(exc).__name__,'error':str(exc),'finished_at':now(),'updated_at':now()})
  raise
 finally:
  stop.set();thread.join(timeout=2)
  try:
   final_progress=json.loads(progress.read_text(encoding='utf-8'));job.update({'completed_samples':final_progress['completed'],'new_completed':final_progress['new_completed'],'eta_minutes':final_progress.get('eta_minutes'),'new_seconds_per_sample':final_progress.get('new_seconds_per_sample')})
  except (OSError,ValueError,KeyError):pass
  with lock:
   write(jobfile,job);pointer.update({'status':job['status'],'stage':job['stage'],'updated_at':now(),'completed_samples':job.get('completed_samples'),'error_type':job.get('error_type')});write(pointerfile,pointer)
if __name__=='__main__':
 try:main()
 except Exception as exc:print(type(exc).__name__+': '+str(exc),file=sys.stderr,flush=True);raise SystemExit(1)
