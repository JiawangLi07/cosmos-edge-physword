"""Same recipe, separate interpreter checkpoint acceptance, varying PYTHONHASHSEED."""
import json,sys
from pathlib import Path
from cosmos_framework.scripts.physword_m3_probe import synthetic_training,SYNTHETIC_REVISIONS
from cosmos_framework.model.generator.physword.checkpoint import save_checkpoint,load_checkpoint,training_protocol
net,proc,_,cfg,opt,_,step,_=synthetic_training('A')
protocol=training_protocol(step,{'data':'unchanged SYNTHETIC','split':'frozen'})
path=Path(sys.argv[2]);out=Path(__file__).resolve().parent
if sys.argv[1]=='save':
 save_checkpoint(path,net,proc,opt,cfg,step.config,step=0,revisions=SYNTHETIC_REVISIONS,experiment=protocol)
 print('save PASS')
else:
 expected=json.loads((path/'manifest.json').read_text())['experiment']
 try:
  load_checkpoint(path,net,opt,cfg,step.config,expected_revisions=SYNTHETIC_REVISIONS,experiment=protocol)
  result={'status':'PASS'}
 except ValueError as exc:
  result={'status':'FAIL','error':str(exc)}
 result.update(saved_protocol=expected,new_protocol=protocol,expected='identical recipe must resume across interpreter restart')
 (out/'cross_process_resume.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
