
from __future__ import annotations
import numpy as np

MUSCLE_PATHS=[
    ("gluteus_r",["pelvis","femur_r"],0.035),
    ("gluteus_l",["pelvis","femur_l"],0.035),
    ("quadriceps_r",["femur_r","tibia_r"],0.025),
    ("quadriceps_l",["femur_l","tibia_l"],0.025),
    ("hamstrings_r",["femur_r","tibia_r"],0.022),
    ("hamstrings_l",["femur_l","tibia_l"],0.022),
    ("gastrosoleus_r",["tibia_r","calcn_r"],0.018),
    ("gastrosoleus_l",["tibia_l","calcn_l"],0.018),
    ("tibialis_ant_r",["tibia_r","talus_r"],0.012),
    ("tibialis_ant_l",["tibia_l","talus_l"],0.012),
    ("deltoid_r",["scapula_r","humerus_r"],0.018),
    ("deltoid_l",["scapula_l","humerus_l"],0.018),
    ("biceps_r",["humerus_r","radius_r"],0.014),
    ("biceps_l",["humerus_l","radius_l"],0.014),
    ("triceps_r",["humerus_r","ulna_r"],0.014),
    ("triceps_l",["humerus_l","ulna_l"],0.014),
]

def muscle_segments(joints_frame,joint_names):
    J=np.asarray(joints_frame,float)
    nm={str(n):i for i,n in enumerate(joint_names)}
    out=[]
    for name,path,r in MUSCLE_PATHS:
        if all(p in nm for p in path):
            pts=np.asarray([J[nm[p]] for p in path],float)
            out.append({"id":name,"points":pts,"radius":float(r)})
    return out
