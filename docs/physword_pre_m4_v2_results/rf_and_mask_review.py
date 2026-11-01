"""Read-only audit probes. No production source changes or optimizer updates."""
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import torch

from cosmos_framework.model.generator.diffusion.rectified_flow import RectifiedFlow
from cosmos_framework.model.generator.algorithm.loss.flow_matching import compute_flow_matching_loss
from cosmos_framework.model.generator.physword.multimodal import build_reasoner_prefix
from cosmos_framework.model.generator.physword.training import forward_teacher_words
from cosmos_framework.model.generator.physword.losses import word_ce, latent_cosine
from cosmos_framework.inference.physword import generate_physical_words, repack_physical_words
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean
from cosmos_framework.scripts.physword_m3_probe import synthetic_training
from cosmos_framework.scripts.physword_probe import SPECIAL_TOKENS

ROOT = Path('/aifs/lijiawang/physword/cosmos-framework')
OUT = Path('/tmp/physword-pre-m4-v2')
source = ROOT / 'cosmos_framework/model/generator/omni_mot_model.py'
tree = ast.parse(source.read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'OmniMoTModel')
node = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_get_train_noise_level_vision'))
module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
namespace = {'torch': torch}
exec(compile(ast.fix_missing_locations(module), str(source), 'exec'), namespace)
native_time = namespace[node.name]
report = {'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
          'native_time_method': 'unmodified AST from OmniMoTModel; only self/config supplied by fixture',
          'real_pretrained_weights': False, 'formal_training': False}
kwcpu = {'device': 'cpu', 'dtype': torch.float32}
sampling = []
for distribution, resolution, shift in [(d,r,s) for d in ['waver','logitnormal'] for r,s in [('256',3),('480',5),('720',10)]]:
    flow = RectifiedFlow(None, train_time_distribution=distribution, shift=shift, device='cpu')
    cfg = NS(use_discrete_rf=False, shift={'256':3, '480':5, '720':10}, shift_image=None, shift_lidar=None)
    obj = NS(rectified_flow_video=flow, config=NS(rectified_flow_training_config=cfg, causal_training_strategy='none'), tensor_kwargs_fp32=kwcpu)
    torch.manual_seed(20261101)
    tn, sn = native_time(obj, 8, False, [6]*8, resolutions=[resolution]*8)
    torch.manual_seed(20261101)
    u = flow.sample_train_time(8)
    tp = flow.get_discrete_timestamp(u, kwcpu)
    sp = flow.get_sigmas(tp, kwcpu)
    sampling.append({'distribution':distribution,'resolution':resolution,'shift':shift,'native_sigma':sn.flatten().tolist(),'pw_sigma':sp.tolist(),
                     'sigma_max_abs':(sn.flatten()-sp).abs().max().item(),
                     'sigma_rms':(sn.flatten()-sp).square().mean().sqrt().item(),
                     'native_t_equals_sigma_N_max_abs':(tn-sn*1000).abs().max().item(),
                     'pw_t_equals_sigma_N_max_abs':(tp-sp*1000).abs().max().item()})
report['same_seed_sampling'] = sampling

net, processor, vae, cfg, optimizer, batch, step, kwargs = synthetic_training()
net.eval()
kwcuda = {'device': 'cuda', 'dtype': torch.float32}
continuous = []
with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
    prefix = build_reasoner_prefix(net, processor, batch.history, batch.instruction)
    prediction = generate_physical_words(net, prefix)
    observed = batch.history.frames.permute(1,0,2,3).unsqueeze(0).float()/127.5-1
    future = batch.future_rgb.permute(1,0,2,3).unsqueeze(0).float()/127.5-1
    history = step.encode_video(observed).float()
    full = step.encode_video(torch.cat((observed,future),2)).float()
    clean = torch.cat((history,full[:,:,2:]),2)
    noise = kwargs['noise'].to(clean)
    mask = torch.tensor([1,1,0,0,0,0],device='cuda',dtype=torch.float32).reshape(6,1,1)
    for sigma in [0.12345, 0.3746, 0.82731]:
        s = torch.tensor([sigma],**kwcuda)
        t = s*step.flow.noise_scheduler.config.num_train_timesteps
        row = {'sigma_requested':sigma,'timestep_float32':t.item()}
        # Full, UNMODIFIED PW training forward: failure must remain visible.
        try:
            actual = step(batch,timesteps=t,noise=kwargs['noise'])
            row['unmodified_pw_step'] = 'PASS'
            row['loss_video'] = actual['losses']['video'].item()
        except Exception as exc:
            row['unmodified_pw_step'] = 'FAIL'
            row['error'] = f'{type(exc).__name__}: {exc}'
        net.eval()
        row['lookup_shape'] = list(step.flow.get_sigmas(t,kwcuda).shape)
        # Isolate already-shared native components; do NOT pretend this bypass fixes the step.
        xn, vn = step.flow.get_interpolation([noise],[clean],[s.reshape(1,1,1)*(1-mask)])
        xp, vp = step.flow.get_interpolation([noise],[clean],[s.reshape(1,1,1,1,1)])
        xp = torch.cat((history,xp[0][:,:,2:]),2)
        expected_x = (1-s.reshape(1,1,1)*(1-mask))*clean + (s.reshape(1,1,1)*(1-mask))*noise
        row['native_vs_pw_noising_max_abs'] = (xn[0]-xp).abs().max().item()
        row['native_vs_formula_noising_max_abs'] = (xn[0]-expected_x).abs().max().item()
        row['velocity_target_max_abs'] = (vp[0]-(noise-clean)).abs().max().item()
        row['native_vs_pw_target_max_abs'] = (vn[0]-vp[0]).abs().max().item()
        row['history_clean_max_abs'] = (xp[:,:,:2]-history).abs().max().item()
        data = GenerationDataClean(batch_size=1,is_image_batch=False,x0_tokens_vision=[xp],fps_vision=torch.tensor([10.]))
        packed = repack_physical_words(net,prediction.condition,data,t.cpu(),SPECIAL_TOKENS)
        packed.to_cuda()
        captured=[]
        h = net.time_embedder.register_forward_pre_hook(lambda _, args: captured.append(args[0].detach().clone()))
        try: output = net(packed)
        finally: h.remove()
        emb_inputs = torch.cat(captured)
        row['timestep_embedder_class'] = type(net.time_embedder).__name__
        row['embedding_input_vs_t_scale_max_abs'] = (emb_inputs-t*net.timestep_scale).abs().max().item()
        with torch.autocast('cuda',enabled=False):
            embedded = net._embed_packed_timesteps(packed.vision.timesteps.float()*net.timestep_scale,packed,torch.float32)
            direct = net.time_embedder((t*net.timestep_scale)).expand_as(embedded)
        row['native_vs_packed_embedding_max_abs'] = (embedded-direct).abs().max().item()
        pred = output['preds_vision'][0].float()
        raw,_=compute_flow_matching_loss(pred=[pred],target=vp,condition_mask=packed.vision.condition_mask,
            timesteps=t.reshape(1,1),has_valid_tokens=True,rectified_flow=step.flow,tensor_kwargs_fp32=kwcuda,
            normalize_by_active=False)
        manual=((pred-(noise-clean)).square()*(1-mask)).mean()
        row['native_loss_vs_masked_mse_max_abs']=(raw-manual).abs().item()
        row['raw_loss']=raw.item()
        row['scaled_video_loss']=(raw*step.config.video_loss_scale).item()
        row['condition_mask']=packed.vision.condition_mask[0].flatten().tolist()
        row['component_forward_status']='PASS' if max(row[k] for k in row if k.endswith('max_abs'))<1e-6 else 'FAIL'
        continuous.append(row)
    # Partially invalid preceding GT; changing invalid values cannot affect later predictions.
    mask_word=torch.tensor([[True,False,True,True]])
    labels_a=torch.tensor([[2,-100,1,3]])
    labels_b=torch.tensor([[2,999,1,3]])
    a=forward_teacher_words(net,prefix,labels_a,mask_word)
    b=forward_teacher_words(net,prefix,labels_b,mask_word)
    ca,na=word_ce(a['logits'],labels_a,mask_word)
    cb,nb=word_ce(b['logits'],labels_b,mask_word)
    partial={'invalid_previous_feedback_id':a['input_ids'][prefix.context_length+2].item(),
             'expected_start_id':net.physword.config.token_id_base,
             'all_logits_max_abs':(a['logits']-b['logits']).abs().max().item(),
             'ce_max_abs':(ca-cb).abs().item(),'valid_count':na}
    assert partial['invalid_previous_feedback_id']==partial['expected_start_id']
    assert partial['all_logits_max_abs']==partial['ce_max_abs']==0 and na==nb==3
report['fractional_sigma'] = continuous
report['partial_invalid_feedback'] = partial
report['full_continuous_step_status'] = 'PASS' if all(x['unmodified_pw_step']=='PASS' for x in continuous) else 'FAIL'
(OUT/'rf_and_mask.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
