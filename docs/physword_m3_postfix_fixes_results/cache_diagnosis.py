"""Independent real-weight, one-batch structural audit; no optimizer or training."""
import json
from pathlib import Path
from dataclasses import replace
import torch
from torch.nn import functional as F
from cosmos_framework.data.generator.physword.dataset import TrainingWindowDataset,to_m3_batch
from cosmos_framework.data.generator.physword.task import config_from_manifest
from cosmos_framework.model.generator.physword.loading import load_local_edge
from cosmos_framework.model.generator.physword.multimodal import build_reasoner_prefix, replay_physical_words
from cosmos_framework.model.generator.physword.training import forward_teacher_words
from cosmos_framework.inference.physword import generate_physical_words,repack_physical_words
from cosmos_framework.inference.physword_diagnostics import capture_und,compare_ar_repack
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean
OUT=Path(__file__).resolve().parent
result={}
def save(key,value):
 result[key]=value; (OUT/'cache_diagnosis.json').write_text(json.dumps(result,indent=2)+'\n');print(key,value.get('status','PASS'),flush=True)
task=Path('/aifs/lijiawang/physword/pre_m4_fixes_results/badminton_left_release')
torch.manual_seed(20261101)
cfg=replace(config_from_manifest(json.loads((task/'vocabulary.json').read_text())),decode_mode='cached')
net,proc,report=load_local_edge('/aifs/lijiawang/physword/Cosmos3-Edge',cfg)
# Independently verify every private row, not just provenance or summary statistics.
rows={}
for i,w in enumerate(cfg.vocabulary):
 ids=proc.tokenizer.encode(w,add_special_tokens=False)
 for kind,source,actual in [('input',net.language_model.model.embed_tokens.weight,net.physword.embedding.weight[i+1]),('head',net.language_model.lm_head.weight,net.physword.word_head.weight[i])]:
  expected=source[ids].float().mean(0)
  rows[w+'.'+kind]=float((expected-actual.detach().float()).abs().max())
assert all(v==0 for v in rows.values())
assert torch.equal(net.physword.embedding.weight[0],net.language_model.model.embed_tokens.weight[10].float())
buffers={n:str(b.dtype) for n,b in net.named_buffers() if b.is_floating_point() and any(s in n for s in ['rotary','freq'])}
save('semantic_initialization',{'status':'PASS','all_rows_max_abs':rows,'start_max_abs':0,'buffers':buffers,'loading':report})
batch=to_m3_batch(TrainingWindowDataset(task/'windows.jsonl')[0],pw_config=cfg)
def resize(x): return F.interpolate(x.float(),size=(64,64),mode='bilinear',align_corners=False).round().byte()
batch=replace(batch,history=replace(batch.history,frames=resize(batch.history.frames)),future_rgb=resize(batch.future_rgb))
net.cuda().eval()
special={'eos_token_id':proc.tokenizer.eos_token_id,'start_of_generation':proc.tokenizer.convert_tokens_to_ids('<|vision_start|>'),'end_of_generation':proc.tokenizer.convert_tokens_to_ids('<|vision_end|>')}
# Synthetic fixed GEN latents isolate UND/attention from VAE; history RGB and labels are real.
data=GenerationDataClean(batch_size=1,is_image_batch=False,x0_tokens_vision=[torch.randn(1,48,6,4,4,device='cuda')],fps_vision=torch.tensor([10.]))
import cosmos_framework.model.generator.mot.unified_mot as native
from contextlib import contextmanager
@contextmanager
def pad_decode_attention():
 original=native.imaginaire_attention
 def attention(*,query,key,value,is_causal,causal_type,**kw):
  if query.shape[1]==1 and key.shape[1]>1:
   padded=torch.cat((torch.zeros(query.shape[0],key.shape[1]-1,*query.shape[2:],device=query.device,dtype=query.dtype),query),dim=1)
   result=original(query=padded,key=key,value=value,is_causal=True,causal_type=native.CausalType.DontCare,**kw)
   return result[:,-1:]
  return original(query=query,key=key,value=value,is_causal=is_causal,causal_type=causal_type,**kw)
 native.imaginaire_attention=attention
 try:yield
 finally:native.imaginaire_attention=original
@contextmanager
def pad_decode_linears():
 originals=[]
 for layer in net.language_model.model.layers:
  for part in [layer.self_attn.q_proj,layer.self_attn.k_proj,layer.self_attn.v_proj,layer.self_attn.o_proj,*layer.mlp.modules()]:
   if not isinstance(part,torch.nn.Linear):continue
   original=part.forward;originals.append((part,original))
   def forward(x,fn=original):
    if x.numel()//x.shape[-1]==1:
     padded=x.reshape(1,-1).expand(128,-1).contiguous()
     return fn(padded)[:1].reshape(*x.shape[:-1],-1)
    return fn(x)
   part.forward=forward
 try:yield
 finally:
  for part,original in originals:part.forward=original
with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
 prefix=build_reasoner_prefix(net,proc,batch.history,batch.instruction)
 pred=generate_physical_words(net,prefix)
 replay=replay_physical_words(net,pred.condition)[:,pred.prediction_positions.to('cuda')]
 with pad_decode_attention():
  padded=generate_physical_words(net,prefix)
 with pad_decode_linears():
  linear_padded=generate_physical_words(net,prefix)
 # Explicit full-prefix K=4 greedy and final flush, without any KV reuse.
 classes=[]; states=[]
 for k in range(4):
  cond=prefix.with_words(tuple(classes),cfg.token_id_base)
  h=replay_physical_words(net,cond)
  z=h[:,-1:];states.append(z);classes.append(int(net.physword.word_head(z[:,0]).argmax(-1)))
 cond=prefix.with_words(tuple(classes),cfg.token_id_base)
 full=replay_physical_words(net,cond)[:,cond.prediction_positions.to('cuda')]
 recompute=torch.cat(states,1)
 def diff(a,b):
  d=a.float()-b.float();return {'max_abs':float(d.abs().max()),'rms':float(d.square().mean().sqrt()),'relative_rms':float(d.square().mean().sqrt()/b.float().square().mean().sqrt())}
 save('cache_diagnosis',{'cached_ids':pred.class_ids.tolist(),'padded_attention_ids':padded.class_ids.tolist(),'recompute_ids':classes,'cached_vs_full':diff(pred.prediction_hidden_states,replay),'padded_vs_full':diff(padded.prediction_hidden_states,replay),'padded_linears_vs_full':diff(linear_padded.prediction_hidden_states,replay),'recompute_vs_full':diff(recompute,full),'note':'padded attention is a temporary diagnostic only; production native attention is unchanged'})
