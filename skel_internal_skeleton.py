from __future__ import annotations
import numpy as np

SEGMENTS = [
    ("pelvis","femur_r","Fémur D"),("femur_r","tibia_r","Tibia D"),
    ("tibia_r","talus_r","Tobillo D"),("talus_r","calcn_r","Retropié D"),
    ("calcn_r","toes_r","Pie D"),
    ("pelvis","femur_l","Fémur I"),("femur_l","tibia_l","Tibia I"),
    ("tibia_l","talus_l","Tobillo I"),("talus_l","calcn_l","Retropié I"),
    ("calcn_l","toes_l","Pie I"),
    ("pelvis","lumbar_body","Pelvis-Lumbar"),("lumbar_body","thorax","Tronco"),
    ("thorax","head","Cuello"),
    ("thorax","scapula_r","Cintura escapular D"),("scapula_r","humerus_r","Hombro D"),
    ("humerus_r","ulna_r","Húmero D"),("ulna_r","radius_r","Antebrazo D"),
    ("radius_r","hand_r","Mano D"),
    ("thorax","scapula_l","Cintura escapular I"),("scapula_l","humerus_l","Hombro I"),
    ("humerus_l","ulna_l","Húmero I"),("ulna_l","radius_l","Antebrazo I"),
    ("radius_l","hand_l","Mano I"),
]

def _basis(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    z=b-a; nz=np.linalg.norm(z)
    if nz<1e-9: z=np.array([0.,0.,1.]); nz=1.
    z/=nz
    h=np.array([1.,0.,0.])
    x=h-np.dot(h,z)*z
    if np.linalg.norm(x)<1e-6:
        h=np.array([0.,1.,0.]); x=h-np.dot(h,z)*z
    x/=max(np.linalg.norm(x),1e-9)
    y=np.cross(z,x); y/=max(np.linalg.norm(y),1e-9)
    return x,y,z

def _tube(a,b,r1,r2,n=12):
    a=np.asarray(a,float); b=np.asarray(b,float)
    x,y,_=_basis(a,b)
    ts=(0.,0.5,1.)
    rs=(r1,max(r1,r2)*0.72,r2)
    V=[]
    for t,r in zip(ts,rs):
        c=a+(b-a)*t
        for k in range(n):
            q=2*np.pi*k/n
            V.append(c+r*(np.cos(q)*x+np.sin(q)*y))
    F=[]
    for j in range(2):
        o0=j*n; o1=(j+1)*n
        for k in range(n):
            k2=(k+1)%n
            F += [[o0+k,o0+k2,o1+k],[o0+k2,o1+k2,o1+k]]
    return np.asarray(V,np.float32),np.asarray(F,np.int32)

def _sphere(c,r,nlat=6,nlon=10):
    c=np.asarray(c,float); V=[]
    for i in range(nlat+1):
        p=np.pi*i/nlat
        for j in range(nlon):
            t=2*np.pi*j/nlon
            V.append(c+r*np.array([np.sin(p)*np.cos(t),np.sin(p)*np.sin(t),np.cos(p)]))
    F=[]
    for i in range(nlat):
        for j in range(nlon):
            j2=(j+1)%nlon
            a=i*nlon+j; b=i*nlon+j2; c0=(i+1)*nlon+j; d=(i+1)*nlon+j2
            F += [[a,c0,b],[b,c0,d]]
    return np.asarray(V,np.float32),np.asarray(F,np.int32)

def _radius_from_skin(V,a,b):
    V=np.asarray(V,float); a=np.asarray(a,float); b=np.asarray(b,float)
    c=(a+b)*0.5; L=max(np.linalg.norm(b-a),1e-6)
    d=np.linalg.norm(V-c,axis=1)
    if not len(d): return 0.04*L
    k=max(8,min(64,max(8,len(d)//100)))
    near=np.partition(d,k-1)[:k]
    return float(np.clip(np.median(near)*0.20,0.02*L,0.10*L))

def build_internal_skeleton(vertices_frame,joints_frame,joint_names):
    V=np.asarray(vertices_frame,np.float32); J=np.asarray(joints_frame,np.float32)
    nm={str(n):i for i,n in enumerate(joint_names)}
    parts=[]
    for an,bn,label in SEGMENTS:
        ia,ib=nm.get(an),nm.get(bn)
        if ia is None or ib is None: continue
        a,b=J[ia],J[ib]; L=float(np.linalg.norm(b-a))
        if L<1e-5: continue
        r=_radius_from_skin(V,a,b)
        if any(x in label for x in ("Fémur","Tibia","Húmero","Antebrazo")): r*=0.58
        elif any(x in label for x in ("Tronco","Pelvis")): r*=1.15
        else: r*=0.75
        vv,ff=_tube(a,b,r*1.08,r*0.92)
        parts.append({"id":label,"V":vv,"F":ff,"kind":"shaft"})
    return parts

def joint_centres(joints_frame,joint_names,base_radius=0.018):
    J=np.asarray(joints_frame,np.float32)
    out=[]
    for i,name in enumerate(joint_names):
        vv,ff=_sphere(J[i],base_radius)
        out.append({"id":f"joint:{name}","V":vv,"F":ff,"kind":"joint"})
    return out

def segment_lengths(joints_frame,joint_names):
    J=np.asarray(joints_frame,float); nm={str(n):i for i,n in enumerate(joint_names)}
    rows=[]
    for an,bn,label in SEGMENTS:
        if an in nm and bn in nm:
            rows.append({"Segmento":label,"Longitud relativa 3D":float(np.linalg.norm(J[nm[bn]]-J[nm[an]]))})
    return rows
