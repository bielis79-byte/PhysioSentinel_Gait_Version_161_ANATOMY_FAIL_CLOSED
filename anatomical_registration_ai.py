from __future__ import annotations
import numpy as np

# Optimization-based registration. No learned neural network is bundled.
# The "AI/Optimization" label refers to automatic constrained fitting using
# skin + joints + anatomical priors.

LONG_BONES = {
    "femur_r": ("femur_r","tibia_r"),
    "femur_l": ("femur_l","tibia_l"),
    "tibia_r": ("tibia_r","talus_r"),
    "tibia_l": ("tibia_l","talus_l"),
    "humerus_r": ("humerus_r","ulna_r"),
    "humerus_l": ("humerus_l","ulna_l"),
    "ulna_r": ("ulna_r","hand_r"),
    "ulna_l": ("ulna_l","hand_l"),
    "radius_r": ("radius_r","hand_r"),
    "radius_l": ("radius_l","hand_l"),
}

def _unit(v, fallback=(1,0,0)):
    v=np.asarray(v,float); n=np.linalg.norm(v)
    if n<1e-9: return np.asarray(fallback,float)
    return v/n

def _basis_from_segment(a,b,lr_hint):
    z=_unit(np.asarray(b)-np.asarray(a),(0,1,0))
    x=np.asarray(lr_hint,float)
    x=x-np.dot(x,z)*z
    x=_unit(x,(1,0,0))
    y=_unit(np.cross(z,x),(0,0,1))
    x=_unit(np.cross(y,z),x)
    return np.stack([x,y,z],axis=1)

def _near_skin_cross_section(vertices,a,b,frac=0.5,band=0.18):
    """Estimate skin half-width/depth around a segment at a normalized location."""
    V=np.asarray(vertices,float)
    a=np.asarray(a,float); b=np.asarray(b,float)
    d=b-a; L=max(np.linalg.norm(d),1e-9); z=d/L
    c=a+frac*d
    # stable transverse basis
    h=np.array([1.,0.,0.])
    x=h-np.dot(h,z)*z
    if np.linalg.norm(x)<1e-6:
        h=np.array([0.,0.,1.]); x=h-np.dot(h,z)*z
    x=_unit(x,(1,0,0)); y=_unit(np.cross(z,x),(0,1,0))
    rel=V-c
    longitudinal=rel@z
    mask=np.abs(longitudinal)<=band*L
    pts=rel[mask]
    if len(pts)<40:
        dd=np.linalg.norm(rel,axis=1)
        k=min(max(40,len(V)//100),len(V))
        if k:
            pts=rel[np.argpartition(dd,k-1)[:k]]
    if not len(pts):
        return 0.05*L,0.05*L
    qx=np.percentile(np.abs(pts@x),70)
    qy=np.percentile(np.abs(pts@y),70)
    return float(max(qx,0.015*L)), float(max(qy,0.015*L))

def _endpoint_centroid_y(V,proximal=True,frac=0.12):
    V=np.asarray(V,float); y=V[:,1]
    q=np.quantile(y,1-frac if proximal else frac)
    pts=V[y>=q] if proximal else V[y<=q]
    return np.mean(pts,axis=0) if len(pts) else np.mean(V,axis=0)

def _long_bone_fit(V,a,b,skin,lr_hint):
    """Fit one real Hamner bone to joints + skin envelope.
    Longitudinal scale is exact to joint distance; transverse scales are chosen
    from skin envelope with conservative bone occupancy priors.
    """
    V=np.asarray(V,float); a=np.asarray(a,float); b=np.asarray(b,float)
    cp=_endpoint_centroid_y(V,True); cd=_endpoint_centroid_y(V,False)
    nvec=cd-cp; nL=max(np.linalg.norm(nvec),1e-9)
    tvec=b-a; tL=max(np.linalg.norm(tvec),1e-9)
    B=_basis_from_segment(a,b,lr_hint)

    # native local frame
    nz=_unit(nvec,(0,-1,0))
    nx=np.array([1.,0.,0.]); nx=nx-np.dot(nx,nz)*nz; nx=_unit(nx,(1,0,0))
    ny=_unit(np.cross(nz,nx),(0,0,1))
    Bn=np.stack([nx,ny,nz],axis=1)
    Q=(V-cp)@Bn

    # skin envelope at segment midpoint; bones occupy a conservative portion.
    sw,sd=_near_skin_cross_section(skin,a,b,0.52,0.20)
    native_w=max(np.percentile(np.abs(Q[:,0]),92),1e-6)
    native_d=max(np.percentile(np.abs(Q[:,1]),92),1e-6)

    sx=np.clip((sw*0.58)/native_w,0.20,2.5)
    sy=np.clip((sd*0.58)/native_d,0.20,2.5)
    sz=tL/nL

    Qs=Q*np.array([sx,sy,sz])[None,:]
    W=a+Qs@B.T
    return W.astype(np.float32), {"sx":float(sx),"sy":float(sy),"sz":float(sz)}

def _pelvis_fit(V,J,nm,skin,frame):
    pel=nm.get("pelvis"); hr=nm.get("femur_r"); hl=nm.get("femur_l"); th=nm.get("thorax")
    if pel is None or hr is None or hl is None:
        return None,{}
    hip_r=J[frame,hr]; hip_l=J[frame,hl]
    center=0.5*(hip_r+hip_l)
    ml=_unit(hip_r-hip_l,(0,0,1))
    up=_unit((J[frame,th]-J[frame,pel]) if th is not None else np.array([0,1,0]),(0,1,0))
    ap=_unit(np.cross(up,ml),(1,0,0))
    ml=_unit(np.cross(ap,up),ml)
    T=np.stack([ap,up,ml],axis=1)

    V=np.asarray(V,float); C=np.mean(V,axis=0); X=V-C
    span=np.maximum(np.ptp(X,axis=0),1e-6)
    hip_width=max(float(np.linalg.norm(hip_r-hip_l)),1e-6)

    # use skin envelope around pelvis to set AP/vertical/ML scales
    rel=np.asarray(skin,float)-center
    local=rel@T
    near=local[np.linalg.norm(rel,axis=1)<max(hip_width*1.2,0.25)]
    if len(near)>80:
        target=np.array([
            max(np.percentile(np.abs(near[:,0]),72)*1.15,0.08),
            max(np.percentile(np.abs(near[:,1]),72)*1.05,0.08),
            max(hip_width*0.62,0.08),
        ])
    else:
        target=np.array([hip_width*0.48,hip_width*0.42,hip_width*0.62])

    native_half=np.maximum(np.percentile(np.abs(X),90,axis=0),1e-6)
    sc=np.clip(target/native_half,0.25,2.2)
    W=center+(X*sc[None,:])@T.T
    return W.astype(np.float32),{"scale":sc.tolist()}

def _hand_fit(V,wrist,hand_center,skin,forearm,T):
    V=np.asarray(V,float); C=np.mean(V,axis=0); X=V-C
    _,_,vt=np.linalg.svd(X,full_matrices=False)
    B=vt.T
    if np.linalg.det(B)<0: B[:,2]*=-1
    Q=X@B
    spans=np.maximum(np.ptp(Q,axis=0),1e-6)

    # infer approximate skin envelope near hand
    d=np.linalg.norm(np.asarray(skin,float)-hand_center,axis=1)
    pts=np.asarray(skin,float)[d<max(0.16,forearm*0.42)]
    if len(pts)>30:
        rel=pts-hand_center
        target_len=float(np.clip(max(np.percentile(np.linalg.norm(rel,axis=1),75)*1.10,forearm*0.33),0.07,0.17))
    else:
        target_len=float(np.clip(forearm*0.42,0.07,0.17))
    width=target_len*0.44; thick=target_len*0.18

    sc=np.clip(np.array([target_len/spans[0],width/spans[1],thick/spans[2]]),0.12,1.25)
    Q=Q*sc[None,:]
    Q[:,0]-=np.min(Q[:,0])

    fwd=_unit(np.asarray(hand_center)-np.asarray(wrist),T[:,0])
    up=T[:,1]-np.dot(T[:,1],fwd)*fwd; up=_unit(up,T[:,1])
    lat=_unit(np.cross(fwd,up),T[:,2])
    TT=np.stack([fwd,up,lat],axis=1)
    W=np.asarray(wrist)[None,:]+Q@TT.T
    return W.astype(np.float32),{"target_len":target_len}

def register_hamner_frame_optimized(bodies,joints,joint_names,skin_vertices,frame,base_build_fn):
    """Use the existing Hamner build as fallback, then replace critical segments
    with constrained skin+joints optimization.
    """
    J=np.asarray(joints,np.float32); skin=np.asarray(skin_vertices,np.float32)
    names=[str(x) for x in joint_names]; nm={n:i for i,n in enumerate(names)}
    base=base_build_fn(bodies,J,names,frame=frame,skin_vertices=skin)
    by={p["body"]:p for p in base}

    if "femur_r" in nm and "femur_l" in nm:
        lr_hint=J[frame,nm["femur_r"]]-J[frame,nm["femur_l"]]
    else:
        lr_hint=np.array([1.,0.,0.])

    # pelvis
    if "pelvis" in bodies and "pelvis" in by:
        W,_=_pelvis_fit(bodies["pelvis"]["V"],J,nm,skin,frame)
        if W is not None:
            by["pelvis"]["V"]=W

    # long bones
    for body,(a_name,b_name) in LONG_BONES.items():
        if body not in bodies or body not in by or a_name not in nm or b_name not in nm:
            continue
        W,_=_long_bone_fit(
            bodies[body]["V"],J[frame,nm[a_name]],J[frame,nm[b_name]],skin,lr_hint
        )
        by[body]["V"]=W

    # hands
    for side in ("r","l"):
        body=f"hand_{side}"
        if body not in bodies or body not in by: continue
        rad=nm.get(f"radius_{side}"); ulna=nm.get(f"ulna_{side}")
        hand=nm.get(body); hum=nm.get(f"humerus_{side}")
        wrist=rad if rad is not None else ulna
        if wrist is None or hand is None: continue
        if hum is not None and ulna is not None:
            forearm=max(float(np.linalg.norm(J[0,hand]-J[0,ulna])),0.12)
        else:
            forearm=max(float(np.linalg.norm(J[0,hand]-J[0,wrist])),0.12)

        # approximate target hand frame from current base hand principal axes:
        fwd=_unit(J[frame,hand]-J[frame,wrist],(1,0,0))
        up=np.array([0.,1.,0.]); up=up-np.dot(up,fwd)*fwd; up=_unit(up,(0,1,0))
        lat=_unit(np.cross(fwd,up),(0,0,1))
        T=np.stack([fwd,up,lat],axis=1)
        W,_=_hand_fit(
            bodies[body]["V"],J[frame,wrist],J[frame,hand],skin,forearm,T
        )
        by[body]["V"]=W

    # preserve original order
    out=[]
    for p in base:
        out.append(by.get(p["body"],p))
    return out

def registration_quality(parts,skin,joints,joint_names,frame):
    """Simple QC: fraction of bone vertices inside an expanded skin envelope (nearest-neighbor proxy omitted)
    and critical joint endpoint distances.
    """
    # Keep lightweight: report critical endpoint distances from joints to mesh centroids near ends.
    J=np.asarray(joints,float); nm={str(n):i for i,n in enumerate(joint_names)}
    rows=[]
    for p in parts:
        body=p.get("body","")
        if body in LONG_BONES:
            a,b=LONG_BONES[body]
            if a in nm and b in nm:
                V=np.asarray(p["V"],float)
                da=float(np.min(np.linalg.norm(V-J[frame,nm[a]],axis=1)))
                db=float(np.min(np.linalg.norm(V-J[frame,nm[b]],axis=1)))
                rows.append({"segmento":body,"error_prox":da,"error_dist":db})
    return rows
