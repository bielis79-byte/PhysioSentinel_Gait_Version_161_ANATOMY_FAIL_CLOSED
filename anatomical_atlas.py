
from pathlib import Path
import json, struct, numpy as np

def _simple_name(x):
    return str(x).strip().lower().replace("-","_").replace(" ","_")

def load_obj(path):
    V=[]; F=[]
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("v "):
            p=line.split()
            if len(p)>=4: V.append([float(p[1]),float(p[2]),float(p[3])])
        elif line.startswith("f "):
            ids=[]
            for tok in line.split()[1:]:
                try: ids.append(int(tok.split("/")[0])-1)
                except: pass
            if len(ids)>=3:
                for i in range(1,len(ids)-1): F.append([ids[0],ids[i],ids[i+1]])
    return np.asarray(V,np.float32), np.asarray(F,np.int32)

def load_ascii_stl(path):
    V=[]; F=[]; lut={}
    def idx(p):
        k=tuple(round(float(x),7) for x in p)
        if k not in lut: lut[k]=len(V); V.append(list(k))
        return lut[k]
    tri=[]
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        s=line.strip().split()
        if len(s)==4 and s[0].lower()=="vertex":
            tri.append(idx([float(s[1]),float(s[2]),float(s[3])]))
            if len(tri)==3: F.append(tri); tri=[]
    return np.asarray(V,np.float32), np.asarray(F,np.int32)

def load_binary_stl(path):
    b=Path(path).read_bytes()
    if len(b)<84: return np.zeros((0,3),np.float32), np.zeros((0,3),np.int32)
    n=struct.unpack("<I",b[80:84])[0]; V=[]; F=[]; lut={}
    off=84
    def idx(p):
        k=tuple(round(float(x),7) for x in p)
        if k not in lut: lut[k]=len(V); V.append(list(k))
        return lut[k]
    for _ in range(n):
        if off+50>len(b): break
        vals=struct.unpack("<12fH",b[off:off+50]); off+=50
        tri=[idx(vals[3:6]),idx(vals[6:9]),idx(vals[9:12])]
        F.append(tri)
    return np.asarray(V,np.float32), np.asarray(F,np.int32)

def load_mesh(path):
    p=Path(path); ext=p.suffix.lower()
    if ext==".obj": return load_obj(p)
    if ext==".stl":
        head=p.read_bytes()[:80].lstrip().lower()
        return load_ascii_stl(p) if head.startswith(b"solid") else load_binary_stl(p)
    raise ValueError(f"Formato no soportado: {ext}. Use OBJ o STL.")

def _basis(a,b):
    d=np.asarray(b,float)-np.asarray(a,float); L=float(np.linalg.norm(d))
    if L<1e-8: return np.eye(3),1.0
    ez=d/L
    ref=np.array([0.,0.,1.]) if abs(ez[2])<.9 else np.array([0.,1.,0.])
    ex=np.cross(ez,ref); ex/=max(np.linalg.norm(ex),1e-9)
    ey=np.cross(ez,ex); ey/=max(np.linalg.norm(ey),1e-9)
    return np.stack([ex,ey,ez],axis=1),L

def register_mesh_to_segment(V, a, b):
    """Normaliza un atlas local y lo registra entre dos joints SKEL.
    Eje principal del atlas = eje de mayor varianza; se orienta proximal→distal.
    """
    V=np.asarray(V,float)
    if V.size==0: return V.astype(np.float32)
    c=V.mean(0); X=V-c
    _,_,vh=np.linalg.svd(X,full_matrices=False)
    ez=vh[0]/max(np.linalg.norm(vh[0]),1e-9)
    ref=vh[1] if len(vh)>1 else np.array([1.,0.,0.])
    ex=ref-np.dot(ref,ez)*ez; ex/=max(np.linalg.norm(ex),1e-9)
    ey=np.cross(ez,ex); ey/=max(np.linalg.norm(ey),1e-9)
    B0=np.stack([ex,ey,ez],axis=1)
    Q=X@B0
    zspan=float(np.ptp(Q[:,2])); zspan=max(zspan,1e-6)
    B,L=_basis(a,b)
    scale=L/zspan
    Q*=scale
    # proximal end to a
    Q[:,2]-=Q[:,2].min()
    return (np.asarray(a,float)+Q@B.T).astype(np.float32)

def load_atlas_registered(root, joints, joint_names):
    root=Path(root); man=json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    J=np.asarray(joints,np.float32); names=[str(x) for x in joint_names]
    nm={_simple_name(n):i for i,n in enumerate(names)}
    out={"bones":[],"muscles":[],"missing":[],"version":man.get("atlas_version","")}
    for group,key in [("bones","bones"),("muscles","muscles")]:
        for s in man.get(key,[]):
            f=root/s["file"]; a=nm.get(_simple_name(s["proximal"])); b=nm.get(_simple_name(s["distal"]))
            if not f.exists():
                out["missing"].append(s["file"]); continue
            if a is None or b is None:
                out["missing"].append(s["id"]+" [joint mapping]"); continue
            V,F=load_mesh(f)
            seq=np.stack([register_mesh_to_segment(V,J[t,a],J[t,b]) for t in range(J.shape[0])])
            out[group].append({"id":s["id"],"V":seq,"F":F})
    return out



# === V110.3.20.16 · BodyParts3D DIRECT FMA asset manager ===
# Corrección V20.15: elimina dependencia de GitHub API/tree discovery.
# Usa rutas RAW deterministas: assets/BodyParts3D_data/stl/FMA<ID>.stl
# No genera geometría procedural de respaldo.

import os as _os, tempfile as _tempfile, urllib.request as _urlreq

_BP3D_RAW_BASE="https://raw.githubusercontent.com/Kevin-Mattheus-Moerman/BodyParts3D/main/assets/BodyParts3D_data/stl"
_BP3D_CREDIT="BodyParts3D, © The Database Center for Life Science licensed under CC Attribution-Share Alike 2.1 Japan"



def _segment_transform_for_frame(V, a, b):
    """Return affine transform components for a base atlas mesh without materializing all frames."""
    V=np.asarray(V,float)
    if V.size==0:
        return np.eye(3,dtype=np.float32), np.zeros(3,dtype=np.float32), 1.0, np.zeros(3,dtype=np.float32)
    c=V.mean(0); X=V-c
    _,_,vh=np.linalg.svd(X,full_matrices=False)
    ez=vh[0]/max(np.linalg.norm(vh[0]),1e-9)
    ref=vh[1] if len(vh)>1 else np.array([1.,0.,0.])
    ex=ref-np.dot(ref,ez)*ez; ex/=max(np.linalg.norm(ex),1e-9)
    ey=np.cross(ez,ex); ey/=max(np.linalg.norm(ey),1e-9)
    B0=np.stack([ex,ey,ez],axis=1)
    Q=X@B0
    zmin=float(Q[:,2].min()); zspan=max(float(np.ptp(Q[:,2])),1e-6)
    B,L=_basis(a,b)
    scale=L/zspan
    # World = a + ((V-c) @ B0 * scale with z shifted to proximal) @ B.T
    # Encode as local basis/scale/translation.
    return B0.astype(np.float32), B.astype(np.float32), float(scale), np.asarray(a,float).astype(np.float32), float(zmin)

def load_atlas_transform_plan(root, joints, joint_names, groups=("bones","muscles")):
    """Load each REAL mesh once and compute only per-frame transform metadata.
    No 75x duplication of vertices.
    """
    root=Path(root)
    man=json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    J=np.asarray(joints,np.float32)
    names=[str(x) for x in joint_names]
    nm={_simple_name(n):i for i,n in enumerate(names)}
    out={"bones":[],"muscles":[],"missing":[],"version":man.get("atlas_version",""),"credit":man.get("credit","")}
    for key in groups:
        if key not in ("bones","muscles"): 
            continue
        for s in man.get(key,[]):
            f=root/s["file"]
            ai=nm.get(_simple_name(s["proximal"])); bi=nm.get(_simple_name(s["distal"]))
            if not f.exists():
                out["missing"].append(s["file"]); continue
            if ai is None or bi is None:
                out["missing"].append(s["id"]+" [joint mapping]"); continue
            V,F=load_mesh(f)
            if V.size==0 or F.size==0:
                out["missing"].append(s["id"]+" [empty mesh]"); continue

            # Precompute invariant atlas-local canonical coordinates once.
            c=V.mean(0); X=V-c
            _,_,vh=np.linalg.svd(X,full_matrices=False)
            ez=vh[0]/max(np.linalg.norm(vh[0]),1e-9)
            ref=vh[1] if len(vh)>1 else np.array([1.,0.,0.])
            ex=ref-np.dot(ref,ez)*ez; ex/=max(np.linalg.norm(ex),1e-9)
            ey=np.cross(ez,ex); ey/=max(np.linalg.norm(ey),1e-9)
            B0=np.stack([ex,ey,ez],axis=1)
            Q=(X@B0).astype(np.float32)
            zmin=float(Q[:,2].min()); zspan=max(float(np.ptp(Q[:,2])),1e-6)
            Q[:,2]-=zmin

            R=[]; S=[]; T=[]
            for t in range(J.shape[0]):
                B,L=_basis(J[t,ai],J[t,bi])
                R.append(B.astype(np.float32))
                S.append(float(L/zspan))
                T.append(np.asarray(J[t,ai],dtype=np.float32))
            out[key].append({
                "id":s["id"],
                "V0":Q.astype(np.float32),   # base canonical mesh ONCE
                "F":np.asarray(F,dtype=np.int32),
                "R":np.stack(R).astype(np.float32),  # 75x3x3
                "S":np.asarray(S,dtype=np.float32),  # 75
                "T":np.stack(T).astype(np.float32),  # 75x3
            })
    return out

def apply_atlas_transform(part, frame_index):
    """Materialize only ONE structure for ONE frame."""
    V0=np.asarray(part["V0"],dtype=np.float32)
    R=np.asarray(part["R"][frame_index],dtype=np.float32)
    s=float(part["S"][frame_index])
    T=np.asarray(part["T"][frame_index],dtype=np.float32)
    return (T + (V0*s)@R.T).astype(np.float32)
def _S(group,id,fma,proximal,distal,label):
    return dict(group=group,id=id,fma=str(fma),proximal=proximal,distal=distal,label=label)

# IDs confirmados contra parts_list_e.txt del mirror BodyParts3D.
_BP3D_DIRECT_TARGETS=[
    # --- HUESOS ---
    _S("bones","hip_bone_r",16586,"pelvis","femur_r","right hip bone"),
    _S("bones","hip_bone_l",16587,"pelvis","femur_l","left hip bone"),
    _S("bones","femur_r",24474,"femur_r","tibia_r","right femur"),
    _S("bones","femur_l",24475,"femur_l","tibia_l","left femur"),
    _S("bones","patella_r",24486,"femur_r","tibia_r","right patella"),
    _S("bones","patella_l",24487,"femur_l","tibia_l","left patella"),
    _S("bones","tibia_r",24477,"tibia_r","talus_r","right tibia"),
    _S("bones","tibia_l",24478,"tibia_l","talus_l","left tibia"),
    _S("bones","fibula_r",24480,"tibia_r","talus_r","right fibula"),
    _S("bones","fibula_l",24481,"tibia_l","talus_l","left fibula"),
    _S("bones","talus_r",24482,"talus_r","calcn_r","right talus"),
    _S("bones","talus_l",24483,"talus_l","calcn_l","left talus"),
    _S("bones","calcaneus_r",24497,"calcn_r","toes_r","right calcaneus"),
    _S("bones","calcaneus_l",24498,"calcn_l","toes_l","left calcaneus"),
    _S("bones","scapula_r",13395,"thorax","scapula_r","right scapula"),
    _S("bones","scapula_l",13396,"thorax","scapula_l","left scapula"),
    _S("bones","clavicle_r",13322,"thorax","scapula_r","right clavicle"),
    _S("bones","clavicle_l",13323,"thorax","scapula_l","left clavicle"),
    _S("bones","humerus_r",23130,"humerus_r","ulna_r","right humerus"),
    _S("bones","humerus_l",23131,"humerus_l","ulna_l","left humerus"),
    _S("bones","radius_r",23464,"radius_r","hand_r","right radius"),
    _S("bones","radius_l",23465,"radius_l","hand_l","left radius"),
    _S("bones","ulna_r",23467,"ulna_r","hand_r","right ulna"),
    _S("bones","ulna_l",23468,"ulna_l","hand_l","left ulna"),

    # --- MÚSCULOS MIEMBRO INFERIOR ---
    _S("muscles","gluteus_maximus_r",22328,"pelvis","femur_r","right gluteus maximus"),
    _S("muscles","gluteus_maximus_l",22329,"pelvis","femur_l","left gluteus maximus"),
    _S("muscles","gluteus_medius_r",22330,"pelvis","femur_r","right gluteus medius"),
    _S("muscles","gluteus_medius_l",22331,"pelvis","femur_l","left gluteus medius"),
    _S("muscles","iliacus_r",22322,"pelvis","femur_r","right iliacus"),
    _S("muscles","iliacus_l",22323,"pelvis","femur_l","left iliacus"),
    _S("muscles","psoas_major_r",22342,"pelvis","femur_r","right psoas major"),
    _S("muscles","psoas_major_l",22343,"pelvis","femur_l","left psoas major"),
    _S("muscles","rectus_femoris_r",38928,"femur_r","tibia_r","right rectus femoris"),
    _S("muscles","rectus_femoris_l",38929,"femur_l","tibia_l","left rectus femoris"),
    _S("muscles","vastus_lateralis_r",38930,"femur_r","tibia_r","right vastus lateralis"),
    _S("muscles","vastus_lateralis_l",38931,"femur_l","tibia_l","left vastus lateralis"),
    _S("muscles","vastus_medialis_r",38932,"femur_r","tibia_r","right vastus medialis"),
    _S("muscles","vastus_medialis_l",38933,"femur_l","tibia_l","left vastus medialis"),
    _S("muscles","semitendinosus_r",22358,"pelvis","tibia_r","right semitendinosus"),
    _S("muscles","semitendinosus_l",22359,"pelvis","tibia_l","left semitendinosus"),
    _S("muscles","semimembranosus_r",22448,"pelvis","tibia_r","right semimembranosus"),
    _S("muscles","semimembranosus_l",22449,"pelvis","tibia_l","left semimembranosus"),
    _S("muscles","gastrocnemius_medial_r",45957,"tibia_r","calcn_r","medial head right gastrocnemius"),
    _S("muscles","gastrocnemius_medial_l",45958,"tibia_l","calcn_l","medial head left gastrocnemius"),
    _S("muscles","gastrocnemius_lateral_r",45960,"tibia_r","calcn_r","lateral head right gastrocnemius"),
    _S("muscles","gastrocnemius_lateral_l",45961,"tibia_l","calcn_l","lateral head left gastrocnemius"),
    _S("muscles","soleus_r",22558,"tibia_r","calcn_r","right soleus"),
    _S("muscles","soleus_l",22559,"tibia_l","calcn_l","left soleus"),
    _S("muscles","tibialis_anterior_r",22544,"tibia_r","toes_r","right tibialis anterior"),
    _S("muscles","tibialis_anterior_l",22545,"tibia_l","toes_l","left tibialis anterior"),
    _S("muscles","fibularis_longus_r",22552,"tibia_r","toes_r","right fibularis longus"),
    _S("muscles","fibularis_longus_l",22553,"tibia_l","toes_l","left fibularis longus"),

    # --- MÚSCULOS MIEMBRO SUPERIOR ---
    _S("muscles","deltoid_clavicular_r",34680,"scapula_r","humerus_r","clavicular part right deltoid"),
    _S("muscles","deltoid_clavicular_l",34681,"scapula_l","humerus_l","clavicular part left deltoid"),
    _S("muscles","deltoid_acromial_r",34682,"scapula_r","humerus_r","acromial part right deltoid"),
    _S("muscles","deltoid_acromial_l",34683,"scapula_l","humerus_l","acromial part left deltoid"),
    _S("muscles","biceps_short_r",37684,"humerus_r","radius_r","short head right biceps brachii"),
    _S("muscles","biceps_short_l",37685,"humerus_l","radius_l","short head left biceps brachii"),
    _S("muscles","biceps_long_r",37686,"humerus_r","radius_r","long head right biceps brachii"),
    _S("muscles","biceps_long_l",37687,"humerus_l","radius_l","long head left biceps brachii"),
    _S("muscles","triceps_medial_r",37695,"humerus_r","ulna_r","medial head right triceps brachii"),
    _S("muscles","triceps_medial_l",37696,"humerus_l","ulna_l","medial head left triceps brachii"),
    _S("muscles","triceps_lateral_r",37697,"humerus_r","ulna_r","lateral head right triceps brachii"),
    _S("muscles","triceps_lateral_l",37698,"humerus_l","ulna_l","lateral head left triceps brachii"),
]

def _bp3d_cache_root():
    p=Path(_tempfile.gettempdir())/"physiosentinel_bodyparts3d_atlas_v2016"
    (p/"bones").mkdir(parents=True,exist_ok=True)
    (p/"muscles").mkdir(parents=True,exist_ok=True)
    return p

def _download(url,timeout=45):
    req=_urlreq.Request(url,headers={"User-Agent":"PhysioSentinel-Gait/110.3.20.16"})
    with _urlreq.urlopen(req,timeout=timeout) as r:
        return r.read()

def ensure_bodyparts3d_atlas(force=False):
    """Descarga directa por FMA ID, sin GitHub API. Fallo de una pieza no invalida todo el atlas."""
    root=_bp3d_cache_root()
    bones=[]; muscles=[]; missing=[]; downloaded=0; already=0
    for spec in _BP3D_DIRECT_TARGETS:
        rel=f'{spec["group"]}/{spec["id"]}.stl'
        local=root/rel
        url=f'{_BP3D_RAW_BASE}/FMA{spec["fma"]}.stl'
        ok=local.exists() and local.stat().st_size>84
        if force or not ok:
            try:
                raw=_download(url)
                if len(raw)<=84:
                    raise ValueError("archivo demasiado pequeño")
                tmp=local.with_suffix(".stl.part")
                tmp.write_bytes(raw)
                tmp.replace(local)
                downloaded+=1
                ok=True
            except Exception as exc:
                missing.append(f'{spec["id"]} · FMA{spec["fma"]} · {type(exc).__name__}: {exc}')
                ok=False
        else:
            already+=1
        if ok:
            item={
                "id":spec["id"],"file":rel,
                "proximal":spec["proximal"],"distal":spec["distal"],
                "source_fma":"FMA"+spec["fma"],"source_name":spec["label"],
            }
            (bones if spec["group"]=="bones" else muscles).append(item)

    man={
        "atlas_version":"BodyParts3D_directFMA_v2016",
        "mode":"STRICT_REAL_MESH",
        "source":"BodyParts3D / Anatomography",
        "source_url":"https://github.com/Kevin-Mattheus-Moerman/BodyParts3D",
        "license":"CC BY-SA 2.1 Japan",
        "credit":_BP3D_CREDIT,
        "bones":bones,"muscles":muscles,"missing_sources":missing,
    }
    (root/"manifest.json").write_text(json.dumps(man,indent=2,ensure_ascii=False),encoding="utf-8")
    return {
        "ok":bool(bones or muscles),"root":str(root),
        "bones":len(bones),"muscles":len(muscles),
        "available":len(bones)+len(muscles),
        "downloaded":downloaded,"cached":already,"missing":missing,
    }

def bodyparts3d_atlas_status():
    root=_bp3d_cache_root(); mp=root/"manifest.json"
    if not mp.exists():
        return {"ready":False,"root":str(root),"bones":0,"muscles":0,"missing":[]}
    try:
        man=json.loads(mp.read_text(encoding="utf-8"))
        b=sum(1 for s in man.get("bones",[]) if (root/s["file"]).exists())
        m=sum(1 for s in man.get("muscles",[]) if (root/s["file"]).exists())
        return {"ready":bool(b or m),"root":str(root),"bones":b,"muscles":m,
                "missing":man.get("missing_sources",[])}
    except Exception as exc:
        return {"ready":False,"root":str(root),"bones":0,"muscles":0,"missing":[str(exc)]}


# ---------------------------------------------------------------------------
# V110.3.20.19 · GLOBAL RIGID BODYParts3D SKELETON REGISTRATION
# ---------------------------------------------------------------------------
# Principle:
# 1) Preserve BodyParts3D geometry in its SHARED atlas coordinate system.
# 2) Estimate ONE global similarity transform (scale + rotation + translation)
#    from multiple anatomical structures to SKEL frame 0.
# 3) Apply only RIGID per-bone motion from SKEL frame 0 -> frame t.
# 4) Never PCA-stretch each bone between two joints and never scale a bone
#    frame-by-frame.
#
# This is still an atlas-to-SKEL registration, not patient CT/MRI anatomy.

_V2019_BONE_KIN = {
    # id -> (proximal joint, distal joint, target anchor mode)
    "hip_bone_r": ("pelvis","femur_r","prox"),
    "hip_bone_l": ("pelvis","femur_l","prox"),
    "femur_r": ("femur_r","tibia_r","mid"),
    "femur_l": ("femur_l","tibia_l","mid"),
    "patella_r": ("femur_r","tibia_r","dist"),
    "patella_l": ("femur_l","tibia_l","dist"),
    "tibia_r": ("tibia_r","talus_r","mid"),
    "tibia_l": ("tibia_l","talus_l","mid"),
    "fibula_r": ("tibia_r","talus_r","mid"),
    "fibula_l": ("tibia_l","talus_l","mid"),
    "talus_r": ("talus_r","calcn_r","prox"),
    "talus_l": ("talus_l","calcn_l","prox"),
    "calcaneus_r": ("calcn_r","toes_r","prox"),
    "calcaneus_l": ("calcn_l","toes_l","prox"),
    "scapula_r": ("thorax","scapula_r","dist"),
    "scapula_l": ("thorax","scapula_l","dist"),
    "clavicle_r": ("thorax","scapula_r","mid"),
    "clavicle_l": ("thorax","scapula_l","mid"),
    "humerus_r": ("humerus_r","ulna_r","mid"),
    "humerus_l": ("humerus_l","ulna_l","mid"),
    "radius_r": ("radius_r","hand_r","mid"),
    "radius_l": ("radius_l","hand_l","mid"),
    "ulna_r": ("ulna_r","hand_r","mid"),
    "ulna_l": ("ulna_l","hand_l","mid"),
}

def _v2019_part_id(s):
    x=str(s.get("id","")).strip().lower()
    # tolerate old manifest ids
    repl={
        "pelvis_r":"hip_bone_r","pelvis_l":"hip_bone_l",
        "right_hip_bone":"hip_bone_r","left_hip_bone":"hip_bone_l",
    }
    return repl.get(x,x)

def _v2019_anchor(a,b,mode):
    a=np.asarray(a,float); b=np.asarray(b,float)
    if mode=="prox": return a
    if mode=="dist": return b
    return 0.5*(a+b)

def _v2019_basis_from_chain(a,b,hint=None):
    """Stable segment frame. z = segment axis; x uses a global/hint vector."""
    a=np.asarray(a,float); b=np.asarray(b,float)
    z=b-a; nz=np.linalg.norm(z)
    if nz<1e-9:
        z=np.array([0.,0.,1.]); nz=1.
    z=z/nz
    h=np.asarray(hint if hint is not None else [1.,0.,0.],float)
    x=h-np.dot(h,z)*z
    if np.linalg.norm(x)<1e-6:
        h=np.array([0.,1.,0.]); x=h-np.dot(h,z)*z
    x=x/max(np.linalg.norm(x),1e-9)
    y=np.cross(z,x); y=y/max(np.linalg.norm(y),1e-9)
    x=np.cross(y,z); x=x/max(np.linalg.norm(x),1e-9)
    return np.stack([x,y,z],axis=1)

def _v2019_similarity(X,Y):
    """Umeyama-style similarity Y ~= s * X @ R.T + t."""
    X=np.asarray(X,float); Y=np.asarray(Y,float)
    if len(X)<3:
        return 1.0,np.eye(3),np.zeros(3)
    mx=X.mean(0); my=Y.mean(0)
    Xc=X-mx; Yc=Y-my
    C=(Yc.T@Xc)/len(X)
    U,D,Vt=np.linalg.svd(C)
    S=np.eye(3)
    if np.linalg.det(U@Vt)<0: S[-1,-1]=-1
    R=U@S@Vt
    var=np.mean(np.sum(Xc*Xc,axis=1))
    sc=float(np.trace(np.diag(D)@S)/max(var,1e-12))
    t=my-sc*(mx@R.T)
    return sc,R,t

def load_bodyparts3d_rigid_skeleton(root, joints, joint_names):
    """Real BodyParts3D bones registered with a shared atlas transform and rigid motion.
    Returns base registered bone meshes + per-frame rigid transforms.
    """
    root=Path(root)
    man=json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    J=np.asarray(joints,dtype=np.float32)
    names=[str(x) for x in joint_names]
    nm={_simple_name(n):i for i,n in enumerate(names)}
    bones=[]; missing=[]

    # Load real bone meshes in their ORIGINAL common BodyParts3D coordinates.
    raw=[]
    for s in man.get("bones",[]):
        pid=_v2019_part_id(s)
        if pid not in _V2019_BONE_KIN:
            continue
        f=root/s["file"]
        if not f.exists():
            missing.append(str(s.get("file",pid))); continue
        V,F=load_mesh(f)
        if V.size==0 or F.size==0:
            missing.append(pid+" [empty mesh]"); continue
        raw.append((pid,np.asarray(V,dtype=np.float32),np.asarray(F,dtype=np.int32)))

    if not raw:
        return {"bones":[],"missing":missing,"method":"global_rigid_v2019"}

    # Shared-atlas landmarks: mesh centroids.
    X=[]; Y=[]; valid=[]
    for pid,V,F in raw:
        p,d,mode=_V2019_BONE_KIN[pid]
        pi=nm.get(_simple_name(p)); di=nm.get(_simple_name(d))
        if pi is None or di is None:
            missing.append(pid+" [joint mapping]"); continue
        X.append(V.mean(0))
        Y.append(_v2019_anchor(J[0,pi],J[0,di],mode))
        valid.append((pid,V,F,pi,di,mode))

    if len(X)<3:
        return {"bones":[],"missing":missing+["insufficient landmarks"],"method":"global_rigid_v2019"}

    scale,Rg,tg=_v2019_similarity(np.asarray(X),np.asarray(Y))

    # Global left-right hint from SKEL pelvis/hips if possible.
    hipr=nm.get(_simple_name("femur_r")); hipl=nm.get(_simple_name("femur_l"))
    if hipr is not None and hipl is not None:
        lr0=np.asarray(J[0,hipr]-J[0,hipl],float)
    else:
        lr0=np.array([1.,0.,0.])
    if np.linalg.norm(lr0)<1e-8: lr0=np.array([1.,0.,0.])

    for pid,V,F,pi,di,mode in valid:
        # ONE global similarity for the actual BodyParts3D geometry.
        Vg=(scale*(V@Rg.T)+tg).astype(np.float32)

        # Static local placement correction only at reference frame:
        # align the bone centroid to the relevant SKEL segment anchor.
        # Geometry itself is not stretched/deformed.
        target0=_v2019_anchor(J[0,pi],J[0,di],mode)
        delta=(target0-Vg.mean(0)).astype(np.float32)
        Vref=(Vg+delta).astype(np.float32)

        # Bone reference frame at t0. Use the same global LR hint to avoid random PCA flips.
        B0=_v2019_basis_from_chain(J[0,pi],J[0,di],lr0)
        A0=_v2019_anchor(J[0,pi],J[0,di],mode)
        local=((Vref-A0)@B0).astype(np.float32)

        Rlist=[]; Tlist=[]
        for t in range(J.shape[0]):
            # Recompute a stable left-right hint from current hips when available.
            if hipr is not None and hipl is not None:
                hint=np.asarray(J[t,hipr]-J[t,hipl],float)
                if np.linalg.norm(hint)<1e-8: hint=lr0
            else:
                hint=lr0
            Bt=_v2019_basis_from_chain(J[t,pi],J[t,di],hint)
            At=_v2019_anchor(J[t,pi],J[t,di],mode)
            # local @ Bt.T + At
            Rlist.append(Bt.astype(np.float32))
            Tlist.append(np.asarray(At,dtype=np.float32))

        bones.append({
            "id":pid,
            "V0":local,               # rigid local bone geometry
            "F":F[:,:3].astype(np.int32),
            "R":np.stack(Rlist).astype(np.float32),
            "S":np.ones(J.shape[0],dtype=np.float32),  # explicitly NO frame scaling
            "T":np.stack(Tlist).astype(np.float32),
            "atlas_scale":float(scale),
            "registration":"BodyParts3D shared-frame global similarity + rigid SKEL articulation",
        })

    return {
        "bones":bones,
        "missing":missing,
        "method":"global_rigid_v2019",
        "global_scale":float(scale),
        "global_rotation":np.asarray(Rg,dtype=np.float32),
        "global_translation":np.asarray(tg,dtype=np.float32),
    }

