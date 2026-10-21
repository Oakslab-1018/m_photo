"""Independent, authenticated GPU OCR runtime; CPU backend dependencies stay isolated."""
from __future__ import annotations
import os, sys, threading, time, hashlib
from pathlib import Path
from contextlib import asynccontextmanager
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
DLLS = ROOT/'inference/streetclip/.venv/Lib/site-packages/torch/lib'
_DLL_HANDLE = os.add_dll_directory(str(DLLS)) if os.name=='nt' else None
os.environ['PATH']=str(DLLS)+os.pathsep+os.environ.get('PATH','')
import onnxruntime as ort
ort.preload_dlls(cuda=True,cudnn=True,msvc=False,directory=str(DLLS))
from rapidocr.inference_engine.onnxruntime.main import OrtInferSession
RUN_OPTIONS=ort.RunOptions()
RUN_OPTIONS.add_run_config_entry('memory.enable_memory_arena_shrinkage','gpu:0')
def _ort_call_with_shrinkage(self,input_content):
    input_dict=dict(zip(self.get_input_names(),[input_content]))
    return self.session.run(self.get_output_names(),input_dict,RUN_OPTIONS)[0]
OrtInferSession.__call__=_ort_call_with_shrinkage
from fastapi import FastAPI,Request,HTTPException
from starlette.concurrency import run_in_threadpool
from backend.app.domain.geolocation.analysis.ocr import RapidOCREngine,PPOCR_VERSION
from importlib.metadata import version
import hmac

def env_secret(name):
    value=os.environ.get(name,'').strip()
    if value:return value
    path=ROOT/'.env'
    if path.is_file():
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            key,sep,raw=line.partition('=')
            if sep and key.strip()==name:return raw.strip().strip('\"\'')
    return ''
KEY=env_secret('GPUOCR_API_KEY') or env_secret('STREETCLIP_API_KEY')
MAX_BYTES=16*1024*1024
GPU_MEM_LIMIT_BYTES=3*1024*1024*1024
class CudaEngine(RapidOCREngine):
    def _build_params(self):
        params=super()._build_params()
        params.update({'EngineConfig.onnxruntime.use_cuda':True,
          'EngineConfig.onnxruntime.intra_op_num_threads':2,
          'EngineConfig.onnxruntime.inter_op_num_threads':1,
          'EngineConfig.onnxruntime.cuda_ep_cfg.arena_extend_strategy':'kSameAsRequested',
          'EngineConfig.onnxruntime.cuda_ep_cfg.cudnn_conv_algo_search':'HEURISTIC',
          'EngineConfig.onnxruntime.cuda_ep_cfg.gpu_mem_limit':GPU_MEM_LIMIT_BYTES,
          'EngineConfig.onnxruntime.cuda_ep_cfg.cudnn_conv_use_max_workspace':False,
          'Rec.rec_batch_num':4,'Cls.cls_batch_num':4})
        return params
ENGINE=CudaEngine(device='cuda',model_type='server',model_dir=ROOT/'backend/storage/ocr-models')
LOCK=threading.Lock()
STATE={'ready':False,'detail':'loading','device':'cuda','execution_provider':'CUDAExecutionProvider'}
def providers():
    raw=ENGINE.load()
    found={}
    for short,attr in (('det','_load_det_model'),('cls','_load_cls_model'),('rec','_load_rec_model')):
        model=getattr(raw,attr)()
        session=getattr(model,'infer',None)
        if session is None:session=getattr(model,'infer_session',None)
        if session is None:
            for val in vars(model).values():
                if hasattr(val,'session') and hasattr(val.session,'get_providers'):session=val;break
        actual=session.session.get_providers() if session is not None else []
        if not actual or actual[0]!='CUDAExecutionProvider':raise RuntimeError('CUDA provider not active for '+short)
        found[short]=actual
    return found
@asynccontextmanager
async def lifespan(app):
    if 'CUDAExecutionProvider' not in ort.get_available_providers():raise RuntimeError('CUDAExecutionProvider is unavailable')
    actual=providers()
    identity=ENGINE.model_identity()
    STATE.update({'ready':True,'detail':None,'provider':'rapidocr','state':'ok','enabled':True,'available':True,
      'requested_device':'cuda','effective_device':'cuda','transport':'http','engine':'onnxruntime',
      'execution_providers':actual,'model_version':PPOCR_VERSION,'model_type':'server',
      'models':{'det':identity['det_model'],'cls':identity['cls_model'],'rec':identity['rec_model']},
      'runtime':{'onnxruntime':ort.__version__,'rapidocr':version('rapidocr'),'python':sys.version.split()[0],'cuda_dll_source':'StreetCLIP torch/lib','gpu_mem_limit_bytes':GPU_MEM_LIMIT_BYTES,'memory_arena_shrinkage':'gpu:0'}})
    yield
app=FastAPI(title='MPhoto GPU OCR',lifespan=lifespan)
@app.get('/health')
def health():return dict(STATE)
def auth(request):
    if KEY and not hmac.compare_digest(request.headers.get('authorization',''),'Bearer '+KEY):raise HTTPException(401,'invalid bearer token')
def recognize(payload):
    started=time.perf_counter()
    with LOCK:
        output=ENGINE.load()(payload)
        lines=ENGINE._parse_output(output)
    status={**STATE,'line_count':len(lines)}
    values=[{'text':x.text,'confidence':x.confidence,'score':x.confidence,'bbox':x.bbox,'box':x.bbox,'provider':x.provider,'det_model':x.det_model,'rec_model':x.rec_model,'cls_model':x.cls_model,'model_type':x.model_type,'model_version':x.model_version} for x in lines]
    return {'lines':values,'ocr_status':status,'status':status,'elapsed_ms':round((time.perf_counter()-started)*1000,3),'image_sha256':hashlib.sha256(payload).hexdigest()}
@app.post('/recognize')
async def recognize_route(request:Request):
    auth(request)
    payload=await request.body()
    if not payload or len(payload)>MAX_BYTES:raise HTTPException(413,'invalid image size')
    try:return await run_in_threadpool(recognize,payload)
    except Exception as exc:
        import logging
        logging.getLogger('gpuocr').exception('GPU OCR recognition failed')
        raise HTTPException(503,'GPU OCR unavailable: '+type(exc).__name__) from None



