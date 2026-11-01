"""Read-only independent review probes; no production code changes."""
import ast
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace, MethodType
import torch
from cosmos_framework.model.generator.diffusion.rectified_flow import RectifiedFlow
from cosmos_framework.data.generator.sequence_packing.runtime import from_all_seq, get_all_seq_unpadded
from cosmos_framework.model.generator.mot.attention import build_packed_sequence, dispatch_attention

OUT = Path(__file__).resolve().parent

def method(revision):
    path='cosmos_framework/model/generator/omni_mot_model.py'
    src=Path(path).read_text() if revision is None else subprocess.check_output(['git','show',revision+':'+path],text=True)
    cls=next(n for n in ast.parse(src).body if isinstance(n,ast.ClassDef) and n.name=='OmniMoTModel')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_get_train_noise_level_vision')
    tree=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),fn],type_ignores=[])
    ns={'torch':torch}; exec(compile(ast.fix_missing_locations(tree),str(revision),'exec'),ns)
    return ns[fn.name]

old,new=method('55f0554'),method(None)
checks=[]
for dist in ['waver','uniform','logitnormal','ltx2']:
 for strategy in ['none','diffusion_forcing']:
  for mode in ['video','image','mixed','dynamic','lidar_only']:
   rf=SimpleNamespace(use_discrete_rf=False,shift={'256':3,'480':5,'720':10},shift_image=2,shift_lidar=7)
   kwargs=dict(batch_size=3,is_image_batch=mode=='image',num_vision_latent_frames=[1,4,6],resolutions=['256','480','720'])
   if mode=='mixed': kwargs['sample_is_image']=[True,False,True]
   if mode=='dynamic':
    rf.shift={'dynamic_shift_base_num_tokens_video':64}; kwargs['num_tokens']=[64,256,576]
   if mode=='lidar_only': kwargs['sequence_plans']=[SimpleNamespace(has_lidar=True,has_vision=False)]*3
   values=[]
   for fn in [old,new]:
    obj=SimpleNamespace(config=SimpleNamespace(rectified_flow_training_config=rf,causal_training_strategy=strategy),tensor_kwargs_fp32={'device':'cpu','dtype':torch.float32},rectified_flow_video=RectifiedFlow(None,train_time_distribution=dist),rectified_flow_image=RectifiedFlow(None,train_time_distribution='logitnormal'))
    obj._get_train_noise_level_vision=MethodType(fn,obj)
    torch.manual_seed(20261101); values.append(obj._get_train_noise_level_vision(**kwargs))
   diffs=[float((a-b).abs().max()) for a,b in zip(*values)]
   assert diffs==[0,0]
   checks.append(dict(distribution=dist,strategy=strategy,mode=mode,t_max_abs=diffs[0],sigma_max_abs=diffs[1]))

# Native one-layer varlen two-way attention vs explicitly masked FP32 reference,
# including Q/K/V backwards. Same BF16 inputs, upstream gradient and GQA repeats.
gen=torch.Generator(device='cuda').manual_seed(73)
values=[torch.randn(26,h,32,generator=gen,device='cuda',dtype=torch.bfloat16,requires_grad=True) for h in (4,2,2)]
und=torch.tensor([*range(5),*range(12,18)],device='cuda'); gn=torch.tensor([*range(5,12),*range(18,26)],device='cuda')
pack,meta,_=build_packed_sequence('two_way',packed_sequence=values[0],attn_modes=['causal','full']*2,split_lens=[5,7,6,8],sample_lens=[12,14],packed_und_token_indexes=und,packed_gen_token_indexes=gn,num_heads=4,head_dim=32,num_layers=1)
packs=[pack,*[from_all_seq(v,pack) for v in values[1:]]]
actual=get_all_seq_unpadded(dispatch_attention(*packs,meta)[0]).float()
ref=[v.detach().float().requires_grad_() for v in values]
q,k,v=ref; k,v=[x.repeat_interleave(2,dim=1) for x in [k,v]]
expected=[]
for start,u,g in [(0,5,7),(12,6,8)]:
 size=u+g; scores=torch.einsum('qhd,khd->hqk',q[start:start+size],k[start:start+size])/32**0.5
 mask=torch.ones(size,size,dtype=torch.bool,device='cuda'); mask[:u]=False; mask[:u,:u]=torch.ones(u,u,device='cuda',dtype=torch.bool).tril()
 expected.append(torch.einsum('hqk,khd->qhd',scores.masked_fill(~mask,-torch.inf).softmax(-1),v[start:start+size]).flatten(1))
expected=torch.cat(expected)
up=torch.randn(actual.shape,generator=gen,device='cuda')
ga=torch.autograd.grad((actual*up).sum(),values); gb=torch.autograd.grad((expected*up).sum(),ref)
metrics={'forward_max_abs':float((actual-expected).abs().max()),'gradients':{},'atol':0.02,'rtol':0.02}
torch.testing.assert_close(actual,expected,atol=.02,rtol=.02)
for name,a,b in zip(['q','k','v'],ga,gb):
 metrics['gradients'][name]={'max_abs':float((a.float()-b).abs().max()),'rms':float((a.float()-b).square().mean().sqrt())}
 torch.testing.assert_close(a.float(),b,atol=.02,rtol=.02)
(OUT/'independent_checks.json').write_text(json.dumps({'native_sampler_cases':checks,'T04':metrics},indent=2)+'\n')
print('PASS: 40 original/current native sampler cases, T04 forward and Q/K/V gradients',metrics)
