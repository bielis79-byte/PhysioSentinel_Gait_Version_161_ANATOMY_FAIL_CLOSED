
from __future__ import annotations
import numpy as np

def _unit(v, fallback):
    v=np.asarray(v,float)
    n=np.linalg.norm(v)
    if n<1e-9:
        return np.asarray(fallback,float)
    return v/n

def unified_body_frame(joints_frame, joint_names):
    """Canonical body frame from SKEL joints.
    Columns = AP, SUP, ML.
    """
    J=np.asarray(joints_frame,float)
    nm={str(n):i for i,n in enumerate(joint_names)}
    pel=nm.get("pelvis"); th=nm.get("thorax"); lum=nm.get("lumbar_body")
    hr=nm.get("femur_r"); hl=nm.get("femur_l")
    if pel is None:
        origin=np.nanmean(J,axis=0)
    else:
        origin=J[pel]
    if th is not None and pel is not None:
        sup=_unit(J[th]-J[pel],[0,1,0])
    elif lum is not None and pel is not None:
        sup=_unit(J[lum]-J[pel],[0,1,0])
    else:
        sup=np.array([0.,1.,0.])
    if hr is not None and hl is not None:
        ml=_unit(J[hr]-J[hl],[0,0,1])
    else:
        ml=np.array([0.,0.,1.])
    ap=_unit(np.cross(sup,ml),[1,0,0])
    ml=_unit(np.cross(ap,sup),ml)
    return origin, np.stack([ap,sup,ml],axis=1)

def canonicalize_points(points, joints_frame, joint_names):
    origin,A=unified_body_frame(joints_frame,joint_names)
    P=np.asarray(points,float)
    return ((P-origin)@A).astype(np.float32), origin, A

def decanonicalize_points(points_c, origin, A):
    P=np.asarray(points_c,float)
    return (origin + P@A.T).astype(np.float32)

def canonicalize_sequence(vertices_seq,joints_seq,joint_names):
    V=np.asarray(vertices_seq,np.float32)
    J=np.asarray(joints_seq,np.float32)
    Vout=np.empty_like(V); Jout=np.empty_like(J)
    origins=[]; frames=[]
    for t in range(len(J)):
        origin,A=unified_body_frame(J[t],joint_names)
        Vout[t]=(V[t]-origin)@A
        Jout[t]=(J[t]-origin)@A
        origins.append(origin); frames.append(A)
    return Vout,Jout,np.asarray(origins,np.float32),np.asarray(frames,np.float32)

def anthropometric_summary(joints_frame,joint_names):
    J=np.asarray(joints_frame,float); nm={str(n):i for i,n in enumerate(joint_names)}
    def dist(a,b):
        if a in nm and b in nm:
            return float(np.linalg.norm(J[nm[b]]-J[nm[a]]))
        return np.nan
    return {
        "Altura tronco pelvis-tórax":dist("pelvis","thorax"),
        "Anchura bicoxal":dist("femur_l","femur_r"),
        "Fémur D":dist("femur_r","tibia_r"),
        "Fémur I":dist("femur_l","tibia_l"),
        "Tibia D":dist("tibia_r","talus_r"),
        "Tibia I":dist("tibia_l","talus_l"),
        "Húmero D":dist("humerus_r","ulna_r"),
        "Húmero I":dist("humerus_l","ulna_l"),
        "Antebrazo D":dist("ulna_r","hand_r"),
        "Antebrazo I":dist("ulna_l","hand_l"),
        "Pie D":dist("calcn_r","toes_r"),
        "Pie I":dist("calcn_l","toes_l"),
    }
