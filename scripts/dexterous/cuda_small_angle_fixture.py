#!/usr/bin/env python3
"""One tiny CUDA arithmetic probe; no physics simulation or sensor changes."""
import argparse,hashlib,json,time
from pathlib import Path
import torch,triton,triton.language as tl
from triton.language.extra import cuda as cuda_language

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    import numpy as np
    @triton.jit
    def evaluate(inputs,outputs,N:tl.constexpr):
        idx=tl.arange(0,32);x=tl.load(inputs+idx,idx<N,0.)
        fast=tl.inline_asm_elementwise('sin.approx.ftz.f32 $0, $1;',constraints='=f,f',args=[x],dtype=tl.float32,is_pure=True,pack=1)
        slow=cuda_language.libdevice.sin(x)
        tl.store(outputs+idx*2,fast,idx<N);tl.store(outputs+idx*2+1,slow,idx<N)
    # Same first-step angular magnitude and dt as the physical fixture.
    w=float(np.linalg.norm(np.array([.03,.01,.02],dtype=np.float32)))
    steps=[1,2,4,8,16,32,64,128]
    angles=np.asarray([np.float32(w)*np.float32(.5)*np.float32(.002/p) for p in steps],dtype=np.float32)
    x=torch.tensor(angles,device='cuda');y=torch.zeros((len(angles),2),device='cuda');start=time.monotonic();evaluate[(1,)](x,y,len(angles));torch.cuda.synchronize();values=y.cpu().numpy();truth=np.sin(angles.astype(np.float64))
    rows=[dict(position_iterations=p,argument_rad=float(v),ptx_sin_approx=float(z[0]),libdevice_sin=float(z[1]),float64_sin=float(t),approx_ratio=float(z[0]/t),libdevice_ratio=float(z[1]/t)) for p,v,z,t in zip(steps,angles,values,truth)]
    report=dict(schema='doorbench.cuda-small-angle-fixture.v1',scope=__doc__,completed=True,passed=None,rows=rows,wall_s=time.monotonic()-start,gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,triton_version=triton.__version__,cuda_version=torch.version.cuda,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),note='PTX intrinsic isolation supports an integrator hypothesis; it does not prove the installed PhysX binary call graph or change any physics')
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
