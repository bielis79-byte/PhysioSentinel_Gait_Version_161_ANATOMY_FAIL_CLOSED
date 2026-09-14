"""PhysioSentinel V109 research PoC.
Run only in a locally authorized SKEL environment with personally downloaded model files.
This script intentionally does NOT redistribute SKEL code or model data.
"""
import argparse, os
import numpy as np, pandas as pd, torch, trimesh
from skel.skel_model import SKEL

p=argparse.ArgumentParser()
p.add_argument('--target',required=True,help='V109_SKEL_target_frame_001.csv')
p.add_argument('--gender',choices=['male','female'],default='female')
p.add_argument('--out',default='V109_SKEL_TPOSE_skeleton.obj')
a=p.parse_args()
# Stage 1 PoC: deterministic forward pass only. No claim of patient fit yet.
# The target is loaded so the next optimizer can compare anatomical joints to V104/V107.
target=pd.read_csv(a.target)
model=SKEL(gender=a.gender).to('cpu')
pose=torch.zeros(1,model.num_q_params)
betas=torch.zeros(1,model.num_betas)
trans=torch.zeros(1,3)
out=model(pose,betas,trans)
mesh=trimesh.Trimesh(vertices=out.skel_verts.detach().cpu().numpy()[0],faces=model.skel_f.cpu(),process=False)
mesh.export(a.out)
print('SKEL forward pass OK:',a.out)
print('Target joints loaded:',len(target))
print('NOTE: this is T-pose geometry, not yet a fitted patient skeleton.')
