"""Reproduce an unrecorded external spatial preprocessing change across resume."""
import json,tempfile
from dataclasses import replace
from pathlib import Path
import torch
from torch.nn import functional as F
from cosmos_framework.scripts.physword_m3_probe import synthetic_training,SYNTHETIC_REVISIONS
from cosmos_framework.model.generator.physword.checkpoint import save_checkpoint,load_checkpoint,training_protocol
out=Path(__file__).resolve().parent
net,processor,_,cfg,opt,batch,step,_=synthetic_training('A')
protocol=training_protocol(step,{'frozen_source_windows_sha256':'same underlying source'})
with tempfile.TemporaryDirectory(prefix='pw-review-preprocess-') as tmp:
 path=Path(tmp)/'bundle'
 save_checkpoint(path,net,processor,opt,cfg,step.config,step=0,revisions=SYNTHETIC_REVISIONS,experiment=protocol)
 restored,_,_,cfg2,opt2,_,step2,_=synthetic_training('A')
 proc2,_=load_checkpoint(path,restored,opt2,cfg2,step2.config,expected_revisions=SYNTHETIC_REVISIONS,experiment=protocol)
 step2.processor=proc2
 def resize(x):return F.interpolate(x.float(),size=(96,96),mode='bilinear',align_corners=False).round().byte()
 changed=replace(batch,history=replace(batch.history,frames=resize(batch.history.frames)),future_rgb=resize(batch.future_rgb))
 with torch.no_grad():
  a=step2(batch);b=step2(changed)
 result={'test':'external resize protocol guard','status':'FAIL','expected':'reject changed preprocessing on strict experiment resume','observed':'checkpoint load accepted same manifest; both 64x64 and 96x96 batches accepted','original_hw':list(batch.history.frames.shape[-2:]),'changed_hw':[96,96],'loss_original':{k:float(v) for k,v in a['losses'].items()},'loss_changed':{k:float(v) for k,v in b['losses'].items()},'manifest_experiment':protocol,'scope':'Tiny; external transform not represented by current training_protocol. Does not invalidate same-transform numerical resume PASS.'}
 (out/'preprocessing_resume.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
