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
 result[key]=value; (OUT/'real_structural.json').write_text(json.dumps(result,indent=2)+'\n');print(key,value.get('status','PASS'),flush=True)
task=Path('/aifs/lijiawang/physword/pre_m4_fixes_results/badminton_left_release')
torch.manual_seed(20261101)
cfg=config_from_manifest(json.loads((task/'vocabulary.json').read_text()))
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
with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
 prefix=build_reasoner_prefix(net,proc,batch.history,batch.instruction)
 gt=forward_teacher_words(net,prefix,batch.word_labels,batch.valid_word)
 errors=[]
 for k in range(4):
  labels=batch.word_labels.clone();labels[0,k]=(labels[0,k]+1)%7
  swapped=forward_teacher_words(net,prefix,labels,batch.valid_word)
  err=float((gt['hidden'][:,:k+1]-swapped['hidden'][:,:k+1]).abs().max());errors.append(err);assert err==0
 save('T03_shift',{'status':'PASS','prefix_state_max_abs_per_target':errors,'real_rgb':True,'size':[64,64]})
 with capture_und(net) as ar: pred=generate_physical_words(net,prefix,capture_cache=True)
 pack=repack_physical_words(net,pred.condition,data,torch.tensor([374.6]),special);pack.to_cuda()
 with capture_und(net) as joint: output=net(pack)
 # Keep the existing tolerance; collect ALL failures instead of stopping at layer 6.
 failures=[]
 original_assert=torch.testing.assert_close
 def collect(a,b,**kw):
  try: original_assert(a,b,**kw)
  except AssertionError:
   delta=(a.float()-b.float()).abs();bound=kw['atol']+kw['rtol']*b.float().abs()
   failures.append({'name':kw.get('msg'),'outside_count':int((delta>bound).sum()),'numel':a.numel(),'max_bound_ratio':float((delta/bound).max()),'relative_rms':float(delta.square().mean().sqrt()/b.float().square().mean().sqrt())})
 torch.testing.assert_close=collect
 try: metrics=compare_ar_repack(net,pred,ar,joint,output)
 finally: torch.testing.assert_close=original_assert
 replay=replay_physical_words(net,pred.condition)
 z_replay=replay[:,pred.prediction_positions.to('cuda')]
 z_joint=output['last_hidden_state'][pred.prediction_positions.to('cuda')].unsqueeze(0)
 logits_ar=net.physword.word_head(pred.prediction_hidden_states).float()
 logits_replay=net.physword.word_head(z_replay).float()
 logits_joint=net.physword.word_head(z_joint).float()
 save('T06',{'status':'FAIL' if failures else 'PASS','atol':.02,'rtol':.02,'metrics':metrics,'failures':failures,'replay_joint_hidden_max_abs':float((z_replay-z_joint).abs().max()),'ar_replay_hidden_max_abs':float((pred.prediction_hidden_states-z_replay).abs().max()),'logits_ar':logits_ar.tolist(),'logits_replay':logits_replay.tolist(),'logits_joint':logits_joint.tolist(),'argmax_ar':logits_ar.argmax(-1).tolist(),'argmax_replay':logits_replay.argmax(-1).tolist(),'argmax_joint':logits_joint.argmax(-1).tolist(),'gen_latents':'synthetic fixed; real UND/GEN weights and historical RGB'})

save('memory',{'peak_bytes':torch.cuda.max_memory_allocated()})
