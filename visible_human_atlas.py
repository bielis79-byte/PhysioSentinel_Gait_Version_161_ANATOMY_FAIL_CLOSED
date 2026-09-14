from __future__ import annotations
from pathlib import Path
import hashlib, json, zipfile, re, functools, tempfile
import numpy as np
import trimesh

CACHE_ROOT=Path("/tmp/physiosentinel_visible_human_v3220")
MANIFEST_SCHEMA=5
SOURCE_PAGE="https://digitalcommons.du.edu/visiblehuman/"
ATTRIBUTION=(
    "Visible Human Male/Female lower-extremity musculoskeletal geometry, "
    "Center for Orthopaedic Biomechanics, University of Denver, CC BY 4.0; "
    "Andreassen et al., Scientific Data 10, 34 (2023)."
)

BONES=[
    "coccyx","sacrum","pelvis","femur","patella","tibia","fibula","talus",
    "calcaneus","navicular","cuboid","cuneiform","phalanges","metatars","tarsal"
]
MUSCLES=[
    "adductor brevis","adductor longus","adductor magnus","biceps femoris long",
    "biceps femoris short","extensor digitorum longus","extensor hallucis longus",
    "flexor digitorum longus","flexor hallucis longus","gastrocnemius lateral",
    "gastrocnemius medial","gluteus maximus","gluteus medius","gluteus minimus",
    "gracilis","iliacus","inferior gemellus","obturator externus","obturator internus",
    "pectineus","peroneus longus","fibularis longus","piriformis","plantaris","popliteus",
    "psoas major","quadratus femoris","rectus femoris","sartorius","semimembranosus",
    "semitendinosus","soleus","superior gemellus","tensor fasciae latae","tibialis anterior",
    "tibialis posterior","vastus intermedius","vastus lateralis","vastus medialis"
]

def _norm(s):
    s=str(s).lower().replace("lnferior","inferior").replace("lliacus","iliacus")
    s=s.replace("lnternus","internus").replace("lntermedius","intermedius")
    s=re.sub(r"[_\-.]+"," ",s)
    s=re.sub(r"\s+"," ",s).strip()
    return s

def _side(path_text):
    t=" "+_norm(path_text)+" "
    if any(x in t for x in (" right "," rt "," r side ","_r ")):
        return "r"
    if any(x in t for x in (" left "," lt "," l side ","_l ")):
        return "l"
    # folder/file suffixes
    low=str(path_text).lower()
    if re.search(r"(^|[/\\ _-])r($|[/\\ _-])",low): return "r"
    if re.search(r"(^|[/\\ _-])l($|[/\\ _-])",low): return "l"
    return None

def _camel_words(s):
    s=Path(str(s)).stem
    s=re.sub(r"(?<=[a-z])(?=[A-Z])"," ",s)
    s=re.sub(r"[_\-.]+"," ",s)
    return re.sub(r"\s+"," ",s).strip().lower()

def _canonical_muscle_name(raw):
    t=_camel_words(raw)
    # common Denver spelling variants / head suffixes
    t=t.replace("quadratis femoris","quadratus femoris")
    t=t.replace("semitendonosus","semitendinosus")
    t=t.replace("illiacus","iliacus")
    t=t.replace("biceps femoris long head","biceps femoris long")
    t=t.replace("biceps femoris short head","biceps femoris short")
    for x in MUSCLES:
        if x in t:
            return x
    aliases={
        "gastrocnemius":"gastrocnemius medial",
        "peroneus longus":"peroneus longus",
        "fibularis longus":"fibularis longus",
    }
    for a,b in aliases.items():
        if a in t: return b
    return None

def _structure(path_text):
    """Classify by Denver file type token, not by anatomical word alone.

    This prevents Cartilage_FemurHead etc. from being rendered as bones and
    recognizes CamelCase muscle filenames correctly.
    """
    raw=str(path_text)
    low=raw.lower()
    stem=_camel_words(raw)

    if "_bone_" in low:
        stem=stem.replace("calcaneous","calcaneus")
        for x in BONES:
            if x in stem:
                return "bone",x
        return None,None

    if "_muscle_" in low:
        x=_canonical_muscle_name(raw)
        return ("muscle",x) if x else (None,None)

    # cartilage and ligament remain available as registration landmarks,
    # but are not rendered as bone/muscle anatomy.
    return None,None

def _region(kind,name):
    n=_norm(name)
    if kind=="bone":
        if n in ("pelvis","sacrum","coccyx"): return "pelvis"
        if n in ("femur","patella"): return "thigh"
        if n in ("tibia","fibula"): return "shank"
        return "foot"
    hip=("gluteus","iliacus","psoas","piriformis","gemellus","obturator","pectineus",
         "quadratus femoris","tensor fasciae latae")
    thigh=("adductor","biceps femoris","gracilis","rectus femoris","sartorius",
           "semimembranosus","semitendinosus","vastus")
    if any(x in n for x in hip): return "pelvis"
    if any(x in n for x in thigh): return "thigh"
    return "shank"

def prepare_visible_human_zip(data: bytes, subject="Female"):
    sha=hashlib.sha256(data).hexdigest()[:16]
    root=CACHE_ROOT/f"{subject.lower()}_{sha}"
    root.mkdir(parents=True,exist_ok=True)
    marker=root/"manifest.json"
    if marker.exists():
        try:
            old=json.loads(marker.read_text(encoding="utf-8"))
            if int(old.get("schema_version",0))==MANIFEST_SCHEMA:
                return old
        except Exception:
            pass
    zpath=root/"source.zip"
    zpath.write_bytes(data)
    extract=root/"stl"
    extract.mkdir(exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        for n in z.namelist():
            if n.lower().endswith(".stl"):
                try: z.extract(n,extract)
                except Exception: pass
    parts=[]
    for p in extract.rglob("*.stl"):
        kind,name=_structure(str(p))
        if not kind: continue
        side=_side(str(p))
        parts.append({
            "path":str(p),"kind":kind,"name":name,"side":side,
            "region":_region(kind,name)
        })
    man={
        "ready":bool(parts),"root":str(root),"subject":subject,"sha":sha,
        "parts":parts,"bones":sum(x["kind"]=="bone" for x in parts),
        "muscles":sum(x["kind"]=="muscle" for x in parts),
        "license":"CC BY 4.0","attribution":ATTRIBUTION,
        "source_page":SOURCE_PAGE,
        "schema_version":MANIFEST_SCHEMA
    }
    marker.write_text(json.dumps(man,ensure_ascii=False,indent=2),encoding="utf-8")
    return man

@functools.lru_cache(maxsize=256)
def _mesh(path):
    m=trimesh.load_mesh(path,process=False)
    if isinstance(m,trimesh.Scene):
        m=trimesh.util.concatenate(tuple(g for g in m.geometry.values()))
    V=np.asarray(m.vertices,np.float32)
    F=np.asarray(m.faces,np.int32)
    return V,F

def _unit(v,fb):
    v=np.asarray(v,float); n=np.linalg.norm(v)
    return np.asarray(fb,float) if n<1e-9 else v/n

def _pca(V):
    X=np.asarray(V,float)-np.mean(V,axis=0)
    _,_,vt=np.linalg.svd(X,full_matrices=False)
    B=vt.T
    if np.linalg.det(B)<0: B[:,2]*=-1
    return B

def _target_frame(J,nm,frame,side,region):
    pel=nm.get("pelvis")
    hip=nm.get(f"femur_{side}")
    knee=nm.get(f"tibia_{side}")
    ankle=nm.get(f"talus_{side}")
    cal=nm.get(f"calcn_{side}")
    toe=nm.get(f"toes_{side}")
    other=nm.get(f"femur_{'l' if side=='r' else 'r'}")
    lr=(J[frame,hip]-J[frame,other]) if hip is not None and other is not None else np.array([1.,0.,0.])
    if region=="pelvis":
        a=J[frame,pel] if pel is not None else J[frame,hip]
        b=J[frame,hip] if hip is not None else a+np.array([0,-.2,0])
    elif region=="thigh":
        a=J[frame,hip]; b=J[frame,knee]
    elif region=="shank":
        a=J[frame,knee]; b=J[frame,ankle]
    else:
        a=J[frame,cal] if cal is not None else J[frame,ankle]
        b=J[frame,toe] if toe is not None else J[frame,ankle]+np.array([0,0,.15])
    z=_unit(b-a,[0,-1,0])
    x=np.asarray(lr,float); x=x-np.dot(x,z)*z; x=_unit(x,[1,0,0])
    y=_unit(np.cross(z,x),[0,0,1])
    x=_unit(np.cross(y,z),x)
    return np.asarray(a,float),np.stack([x,y,z],axis=1),float(np.linalg.norm(b-a))

def _donor_reference(man,side,region):
    # select key bone
    preferred={
        "pelvis":["pelvis","sacrum"],
        "thigh":["femur"],
        "shank":["tibia","fibula"],
        "foot":["calcaneus","phalanges","talus"]
    }[region]
    cand=[p for p in man["parts"] if p["kind"]=="bone" and p.get("side") in (side,None) and p["name"] in preferred]
    if not cand:
        cand=[p for p in man["parts"] if p["kind"]=="bone" and p.get("side") in (side,None)]
    if not cand: return None
    # use first preferred hit
    p=sorted(cand,key=lambda q: preferred.index(q["name"]) if q["name"] in preferred else 99)[0]
    V,_=_mesh(p["path"])
    C=np.mean(V,axis=0)
    B=_pca(V)
    # choose longest PCA axis as local Z
    spans=np.ptp((V-C)@B,axis=0)
    iz=int(np.argmax(spans))
    others=[i for i in range(3) if i!=iz]
    z=B[:,iz]; x=B[:,others[0]]; y=_unit(np.cross(z,x),B[:,others[1]])
    x=_unit(np.cross(y,z),x)
    Bb=np.stack([x,y,z],axis=1)
    L=max(float(spans[iz]),1e-6)
    # origin at proximal extreme along local z
    q=(V-C)@Bb
    origin=C+Bb[:,2]*float(np.max(q[:,2]))
    # flip z so geometry extends primarily in + local longitudinal from origin
    Bb[:,2]*=-1
    Bb[:,1]=_unit(np.cross(Bb[:,2],Bb[:,0]),Bb[:,1])
    return origin,Bb,L

def _skin_radius(skin,a,b):
    V=np.asarray(skin,float); c=.5*(a+b); L=max(np.linalg.norm(b-a),1e-6)
    d=np.linalg.norm(V-c,axis=1)
    k=min(max(50,len(V)//100),len(V))
    if k==0:return .05*L
    pts=V[np.argpartition(d,k-1)[:k]]
    return float(np.percentile(np.linalg.norm(pts-c,axis=1),65))

def build_visible_human_frame(man,joints,joint_names,skin_vertices,frame=0,
                              show_bones=True,show_muscles=True,max_faces_each=220):
    J=np.asarray(joints,np.float32); skin=np.asarray(skin_vertices,np.float32)
    nm={str(n):i for i,n in enumerate(joint_names)}
    out=[]
    refs={}
    for side in ("r","l"):
        for region in ("pelvis","thigh","shank","foot"):
            refs[(side,region)]=_donor_reference(man,side,region)
    for p in man["parts"]:
        if p["kind"]=="bone" and not show_bones: continue
        if p["kind"]=="muscle" and not show_muscles: continue
        side=p.get("side")
        if side not in ("r","l"):
            # central pelvis/sacrum shown once on right-frame convention
            side="r"
        ref=refs.get((side,p["region"]))
        if ref is None: continue
        donor_origin,donor_B,donor_L=ref
        try:
            target_origin,target_B,target_L=_target_frame(J,nm,frame,side,p["region"])
        except Exception:
            continue
        V,F=_mesh(p["path"])
        Q=(V-donor_origin)@donor_B
        scale=target_L/max(donor_L,1e-6)
        if p["kind"]=="bone":
            transverse=scale
        else:
            # preserve muscle volume better; only mild transverse adaptation to skin
            try:
                if p["region"]=="thigh":
                    a=J[frame,nm[f"femur_{side}"]]; b=J[frame,nm[f"tibia_{side}"]]
                elif p["region"]=="shank":
                    a=J[frame,nm[f"tibia_{side}"]]; b=J[frame,nm[f"talus_{side}"]]
                else:
                    a=target_origin; b=target_origin+target_B[:,2]*target_L
                skin_r=_skin_radius(skin,a,b)
                native_r=max(np.percentile(np.linalg.norm(Q[:,:2],axis=1),70),1e-6)
                transverse=float(np.clip((skin_r*.72)/native_r,scale*.65,scale*1.35))
            except Exception:
                transverse=scale
        Qs=Q*np.array([transverse,transverse,scale])[None,:]
        W=target_origin+Qs@target_B.T

        # display decimation by deterministic face sampling
        step=max(1,int(np.ceil(len(F)/float(max_faces_each))))
        Fs=np.asarray(F[::step,:3],np.int32)
        used=np.unique(Fs.reshape(-1))
        remap=np.full(len(W),-1,np.int32); remap[used]=np.arange(len(used),dtype=np.int32)
        out.append({
            "id":f"{p['kind']}:{p['name']}:{side}",
            "kind":p["kind"],"name":p["name"],"side":side,
            "V":np.asarray(W[used],np.float32),"F":remap[Fs]
        })
    return out

def atlas_counts(man):
    if not man:return {"bones":0,"muscles":0}
    return {"bones":int(man.get("bones",0)),"muscles":int(man.get("muscles",0))}



# ---------------------------------------------------------------------------
# V110.3.22.5 · MASTER ATLAS DENVER -> HAMNER/SKEL
# ---------------------------------------------------------------------------

MUSCLE_CHAIN = {
    # pelvis / hip
    "gluteus maximus":("pelvis","thigh"), "gluteus medius":("pelvis","thigh"),
    "gluteus minimus":("pelvis","thigh"), "iliacus":("pelvis","thigh"),
    "psoas major":("pelvis","thigh"), "piriformis":("pelvis","thigh"),
    "inferior gemellus":("pelvis","thigh"), "superior gemellus":("pelvis","thigh"),
    "obturator externus":("pelvis","thigh"), "obturator internus":("pelvis","thigh"),
    "pectineus":("pelvis","thigh"), "quadratus femoris":("pelvis","thigh"),
    "tensor fasciae latae":("pelvis","thigh"),
    # thigh / knee
    "adductor brevis":("pelvis","thigh"), "adductor longus":("pelvis","thigh"),
    "adductor magnus":("pelvis","thigh"), "biceps femoris long":("pelvis","shank"),
    "biceps femoris short":("thigh","shank"), "gracilis":("pelvis","shank"),
    "rectus femoris":("pelvis","shank"), "sartorius":("pelvis","shank"),
    "semimembranosus":("pelvis","shank"), "semitendinosus":("pelvis","shank"),
    "vastus intermedius":("thigh","shank"), "vastus lateralis":("thigh","shank"),
    "vastus medialis":("thigh","shank"), "popliteus":("thigh","shank"),
    # shank / ankle-foot
    "gastrocnemius lateral":("thigh","foot"), "gastrocnemius medial":("thigh","foot"),
    "soleus":("shank","foot"), "plantaris":("thigh","foot"),
    "tibialis anterior":("shank","foot"), "tibialis posterior":("shank","foot"),
    "extensor digitorum longus":("shank","foot"), "extensor hallucis longus":("shank","foot"),
    "flexor digitorum longus":("shank","foot"), "flexor hallucis longus":("shank","foot"),
    "peroneus longus":("shank","foot"), "fibularis longus":("shank","foot"),
}

def _clean_mesh_master(path):
    """Load and lightly clean STL without changing atlas coordinates."""
    m=trimesh.load_mesh(path,process=True)
    if isinstance(m,trimesh.Scene):
        m=trimesh.util.concatenate(tuple(g for g in m.geometry.values()))
    try:
        m.remove_unreferenced_vertices()
        m.fix_normals()
    except Exception:
        pass
    V=np.asarray(m.vertices,np.float64)
    F=np.asarray(m.faces,np.int32)
    return V,F

def _atlas_unit_scale(man):
    """Denver STL commonly uses millimetres. Infer one global unit scale only."""
    pts=[]
    for p in man.get("parts",[]):
        if p.get("kind")!="bone":
            continue
        try:
            V,_=_clean_mesh_master(p["path"])
            if len(V): pts.append(V[::max(1,len(V)//500)])
        except Exception:
            pass
    if not pts:
        return 1.0
    P=np.vstack(pts)
    span=float(np.max(np.ptp(P,axis=0)))
    # human lower limb should be O(1 m), not O(1000)
    return 0.001 if span>10.0 else 1.0


def _semantic_forward_from_joints(J,nm,t):
    """Estimate anatomical anterior from both heel->toe vectors in SKEL coordinates."""
    vals=[]
    for _s in ("r","l"):
        ih=nm.get(f"calcn_{_s}"); it=nm.get(f"toes_{_s}")
        if ih is not None and it is not None:
            v=np.asarray(J[t,it]-J[t,ih],float)
            if np.isfinite(v).all() and np.linalg.norm(v)>1e-8:
                vals.append(_unit(v,[0,0,1]))
    if not vals:
        return np.array([0.,0.,1.])
    f=np.sum(vals,axis=0)
    return _unit(f,vals[0])


def _body_frame_target(J,nm,t):
    """Body-fixed proper frame: columns = anatomical RIGHT, ANTERIOR, UP."""
    hr=np.asarray(J[t,nm["femur_r"]],float); hl=np.asarray(J[t,nm["femur_l"]],float)
    mid=(hr+hl)/2.0
    thor=nm.get("thorax",nm.get("lumbar_body"))
    up0=_unit(np.asarray(J[t,thor],float)-mid,[0,1,0]) if thor is not None else np.array([0.,1.,0.])
    right0=_unit(hr-hl,[1,0,0])
    f=[]
    for side in ("r","l"):
        ih=nm.get(f"calcn_{side}"); it=nm.get(f"toes_{side}")
        if ih is not None and it is not None:
            q=np.asarray(J[t,it]-J[t,ih],float)
            q=q-right0*np.dot(q,right0)-up0*np.dot(q,up0)
            if np.linalg.norm(q)>1e-8: f.append(_unit(q,[0,0,1]))
    ant0=_unit(np.sum(f,axis=0),[0,0,1]) if f else _unit(np.cross(up0,right0),[0,0,1])
    up=_unit(np.cross(right0,ant0),up0)
    if np.dot(up,up0)<0: up=-up; ant0=-ant0
    ant=_unit(np.cross(up,right0),ant0)
    right=_unit(np.cross(ant,up),right0)
    return right,ant,up,mid


def _semantic_target_frame(J,nm,t,side,segment):
    """V160 target frames using side-independent anatomical axes.

    Proper frames only (det=+1): pelvis [R,A,U], thigh/shank [R,A,proximal],
    foot [R,forward,dorsal].  Using anatomical RIGHT rather than 'medial' avoids
    the left/right handedness trap that could visually swap or invert feet.
    """
    right_b,ant_b,up_b,mid=_body_frame_target(J,nm,t)
    if segment=="pelvis":
        hr=np.asarray(J[t,nm["femur_r"]],float); hl=np.asarray(J[t,nm["femur_l"]],float)
        L=float(np.linalg.norm(hr-hl))
        return mid,np.stack([right_b,ant_b,up_b],axis=1),max(L,1e-6)
    if segment=="thigh":
        hip=np.asarray(J[t,nm[f"femur_{side}"]],float); knee=np.asarray(J[t,nm[f"tibia_{side}"]],float)
        prox=_unit(hip-knee,[0,1,0])
        right=right_b-prox*np.dot(right_b,prox); right=_unit(right,right_b)
        ant=_unit(np.cross(prox,right),ant_b)
        if np.dot(ant,ant_b)<0: right=-right; ant=-ant
        prox=_unit(np.cross(right,ant),prox)
        return hip,np.stack([right,ant,prox],axis=1),max(float(np.linalg.norm(knee-hip)),1e-6)
    if segment=="shank":
        knee=np.asarray(J[t,nm[f"tibia_{side}"]],float); ankle=np.asarray(J[t,nm[f"talus_{side}"]],float)
        prox=_unit(knee-ankle,[0,1,0])
        right=right_b-prox*np.dot(right_b,prox); right=_unit(right,right_b)
        ant=_unit(np.cross(prox,right),ant_b)
        if np.dot(ant,ant_b)<0: right=-right; ant=-ant
        prox=_unit(np.cross(right,ant),prox)
        return knee,np.stack([right,ant,prox],axis=1),max(float(np.linalg.norm(ankle-knee)),1e-6)
    if segment=="foot":
        ankle=np.asarray(J[t,nm[f"talus_{side}"]],float)
        heel=np.asarray(J[t,nm.get(f"calcn_{side}",nm[f"talus_{side}"])],float)
        toe=np.asarray(J[t,nm[f"toes_{side}"]],float)
        fwd=_unit(toe-heel,ant_b)
        right=right_b-fwd*np.dot(right_b,fwd); right=_unit(right,right_b)
        dorsal=_unit(np.cross(right,fwd),up_b)
        if np.dot(dorsal,up_b)<0: right=-right; dorsal=-dorsal
        fwd=_unit(np.cross(dorsal,right),fwd)
        Ls=[np.linalg.norm(toe-ankle),np.linalg.norm(heel-ankle),np.linalg.norm(toe-heel)]
        return ankle,np.stack([right,fwd,dorsal],axis=1),max(float(np.median(Ls)),1e-6)
    return _segment_frame_from_joints(J,nm,t,side,segment)

def _segment_frame_from_joints(J,nm,t,side,segment):
    """Rigid segment frame in SKEL coordinates."""
    if segment=="pelvis":
        pel=nm.get("pelvis"); hr=nm.get("femur_r"); hl=nm.get("femur_l")
        thor=nm.get("lumbar_body",nm.get("thorax"))
        o=J[t,pel]
        up=_unit(J[t,thor]-o,[0,1,0]) if thor is not None else np.array([0,1.,0])
        ml=_unit(J[t,hr]-J[t,hl],[1,0,0]) if hr is not None and hl is not None else np.array([1,0,0])
        ap=_unit(np.cross(ml,up),[0,0,1])
        ml=_unit(np.cross(up,ap),ml)
        B=np.stack([ml,ap,up],axis=1)
        L=float(np.linalg.norm(J[t,hr]-J[t,hl])) if hr is not None and hl is not None else .25
        return o,B,max(L,1e-6)
    if segment=="thigh":
        a=J[t,nm[f"femur_{side}"]]; b=J[t,nm[f"tibia_{side}"]]
    elif segment=="shank":
        a=J[t,nm[f"tibia_{side}"]]; b=J[t,nm[f"talus_{side}"]]
    else:
        ankle=J[t,nm[f"talus_{side}"]]
        heel=J[t,nm.get(f"calcn_{side}",nm[f"talus_{side}"])]
        toe=J[t,nm[f"toes_{side}"]]
        other=nm.get(f"femur_{'l' if side=='r' else 'r'}")
        hip=nm.get(f"femur_{side}")
        lr=(J[t,hip]-J[t,other]) if hip is not None and other is not None else np.array([1,0,0])
        z=_unit(toe-heel,[0,0,1])
        x=_unit(lr-np.dot(lr,z)*z,[1,0,0])
        y=_unit(np.cross(z,x),[0,1,0])
        x=_unit(np.cross(y,z),x)
        return np.asarray(ankle,float),np.stack([x,y,z],axis=1),max(float(np.linalg.norm(toe-ankle)),1e-6)
    z=_unit(b-a,[0,-1,0])
    other=nm.get(f"femur_{'l' if side=='r' else 'r'}")
    hip=nm.get(f"femur_{side}")
    lr=(J[t,hip]-J[t,other]) if hip is not None and other is not None else np.array([1,0,0])
    x=lr-np.dot(lr,z)*z
    x=_unit(x,[1,0,0])
    y=_unit(np.cross(z,x),[0,0,1])
    x=_unit(np.cross(y,z),x)
    return np.asarray(a,float),np.stack([x,y,z],axis=1),max(float(np.linalg.norm(b-a)),1e-6)

def _find_raw_stl(man,side,token):
    root=Path(man["root"])/"stl"
    side_word="Left" if side=="l" else "Right"
    token_low=token.lower()
    cand=[]
    for p in root.rglob("*.stl"):
        s=p.name.lower()
        if side_word.lower() in s and token_low in s:
            cand.append(p)
    return sorted(cand)[0] if cand else None

@functools.lru_cache(maxsize=128)
def _stl_center_cached(path_str,unit_scale):
    V,_=_clean_mesh_master(path_str)
    return np.mean(V,axis=0)*float(unit_scale)

def _denver_landmarks(man,unit_scale):
    """Anatomical landmarks from Denver's own cartilage/bone STL files."""
    out={}
    for side in ("l","r"):
        specs={
            "hip":"Cartilage_FemurHead",
            "acetabulum":"Cartilage_PelvisAcetabulum",
            "knee":"Cartilage_FemurDistal",
            "tibia_medial":"Cartilage_TibiaMedial",
            "tibia_lateral":"Cartilage_TibiaLateral",
            "ankle":"Cartilage_TibiaDistal",
            "heel":"Bone_Calcaneous",
            "toe":"Bone_Phalanges",
            "patella":"Bone_Patella",
            "medial_cuneiform":"Bone_MedialCuneiform",
            "lateral_cuneiform":"Bone_LateralCuneiform",
            "sacrum":"Bone_Sacrum",
            "coccyx":"Bone_Coccyx",
        }
        for key,tok in specs.items():
            p=_find_raw_stl(man,side,tok)
            if p is not None:
                out[f"{key}_{side}"]=_stl_center_cached(str(p),float(unit_scale))
    return out

def _frame_from_axis(a,b,lateral):
    z=_unit(np.asarray(b)-np.asarray(a),[0,-1,0])
    x=np.asarray(lateral,float)-np.dot(lateral,z)*z
    x=_unit(x,[1,0,0])
    y=_unit(np.cross(z,x),[0,0,1])
    x=_unit(np.cross(y,z),x)
    return np.stack([x,y,z],axis=1)

def _body_frame_donor(L):
    """Denver proper body frame: columns = anatomical RIGHT, ANTERIOR, UP."""
    mid=(L["hip_r"]+L["hip_l"])/2.0; km=(L["knee_r"]+L["knee_l"])/2.0
    right0=_unit(L["hip_r"]-L["hip_l"],[1,0,0]); up0=_unit(mid-km,[0,0,1])
    post=[]
    for k in ("sacrum_l","sacrum_r","coccyx_l","coccyx_r"):
        if k in L: post.append(np.asarray(L[k],float))
    if post:
        ant0=-(np.mean(post,axis=0)-mid)
        ant0=ant0-right0*np.dot(ant0,right0)-up0*np.dot(ant0,up0)
        ant0=_unit(ant0,[0,1,0])
    else:
        ant0=np.array([0.,1.,0.])
    up=_unit(np.cross(right0,ant0),up0)
    if np.dot(up,up0)<0: up=-up; ant0=-ant0
    ant=_unit(np.cross(up,right0),ant0); right=_unit(np.cross(ant,up),right0)
    return right,ant,up,mid


def _master_donor_frame(man,side,segment,unit_scale):
    """V160 Denver donor frame with explicit proper anatomical semantics."""
    L=_denver_landmarks(man,float(unit_scale))
    needed=[f"hip_{side}",f"knee_{side}",f"ankle_{side}",f"toe_{side}"]
    if any(k not in L for k in needed) or "hip_l" not in L or "hip_r" not in L: return None
    hip=L[f"hip_{side}"]; knee=L[f"knee_{side}"]; ankle=L[f"ankle_{side}"]
    toe=L[f"toe_{side}"]; heel=L.get(f"heel_{side}",ankle)
    right_b,ant_b,up_b,mid=_body_frame_donor(L)
    if segment=="pelvis":
        return mid,np.stack([right_b,ant_b,up_b],axis=1),max(float(np.linalg.norm(L["hip_r"]-L["hip_l"])),1e-6)
    if segment=="thigh":
        prox=_unit(hip-knee,[0,0,1])
        right=right_b-prox*np.dot(right_b,prox); right=_unit(right,right_b)
        ant=_unit(np.cross(prox,right),ant_b)
        pat=L.get(f"patella_{side}")
        if pat is not None:
            pv=np.asarray(pat,float)-knee; pv=pv-prox*np.dot(pv,prox)
            if np.dot(ant,pv)<0: right=-right; ant=-ant
        prox=_unit(np.cross(right,ant),prox)
        return hip,np.stack([right,ant,prox],axis=1),max(float(np.linalg.norm(knee-hip)),1e-6)
    if segment=="shank":
        prox=_unit(knee-ankle,[0,0,1])
        right=right_b-prox*np.dot(right_b,prox); right=_unit(right,right_b)
        pt=_find_raw_stl(man,side,"Bone_Tibia"); pf=_find_raw_stl(man,side,"Bone_Fibula")
        if pt is not None and pf is not None:
            tv=_stl_center_cached(str(pt),float(unit_scale)); fv=_stl_center_cached(str(pf),float(unit_scale))
            latvec=np.asarray(fv)-np.asarray(tv)
            expected=1.0 if side=="r" else -1.0
            if expected*np.dot(latvec,right)<0: right=-right
        ant=_unit(np.cross(prox,right),ant_b)
        if np.dot(ant,ant_b)<0: right=-right; ant=-ant
        prox=_unit(np.cross(right,ant),prox)
        return knee,np.stack([right,ant,prox],axis=1),max(float(np.linalg.norm(ankle-knee)),1e-6)
    fwd=_unit(toe-heel,ant_b)
    right=right_b-fwd*np.dot(right_b,fwd); right=_unit(right,right_b)
    dorsal=_unit(np.cross(right,fwd),up_b)
    if np.dot(dorsal,up_b)<0: right=-right; dorsal=-dorsal
    fwd=_unit(np.cross(dorsal,right),fwd)
    Ls=[np.linalg.norm(toe-ankle),np.linalg.norm(heel-ankle),np.linalg.norm(toe-heel)]
    return ankle,np.stack([right,fwd,dorsal],axis=1),max(float(np.median(Ls)),1e-6)

def _simplify_connected_mesh(V,F,target_faces):
    """Cloud-safe simplification preserving surface topology as far as practical.

    First uses trimesh quadric decimation when available. If that backend is not
    present, a pure-numpy vertex-clustering fallback is used. It never samples
    isolated triangles with F[::step].
    """
    V=np.asarray(V,np.float64); F=np.asarray(F,np.int32)
    m=trimesh.Trimesh(vertices=V,faces=F,process=True)
    if len(F)<=int(target_faces):
        try: m.remove_unreferenced_vertices(); m.fix_normals()
        except Exception: pass
        return np.asarray(m.vertices,np.float32),np.asarray(m.faces,np.int32)
    try:
        m2=m.simplify_quadric_decimation(face_count=int(target_faces))
        if m2 is not None and len(getattr(m2,"faces",[]))>0:
            m=m2
            try: m.remove_unreferenced_vertices(); m.fix_normals()
            except Exception: pass
            return np.asarray(m.vertices,np.float32),np.asarray(m.faces,np.int32)
    except Exception:
        pass

    # Pure numpy vertex clustering fallback.
    V=np.asarray(m.vertices,np.float64); F=np.asarray(m.faces,np.int32)
    mn=V.min(axis=0); span=max(float(np.max(np.ptp(V,axis=0))),1e-9)
    bestV,bestF=V,F; besterr=abs(len(F)-int(target_faces))
    lo=max(span/2200.0,1e-8); hi=max(span/3.0,lo*2.0)
    for _ in range(18):
        cell=(lo+hi)/2.0
        keys=np.floor((V-mn)/cell).astype(np.int64)
        _,inv=np.unique(keys,axis=0,return_inverse=True)
        nv=int(inv.max())+1
        cnt=np.bincount(inv,minlength=nv).astype(float)
        V2=np.column_stack([
            np.bincount(inv,weights=V[:,a],minlength=nv)/np.maximum(cnt,1.0)
            for a in range(3)
        ])
        F2=inv[F]
        good=(F2[:,0]!=F2[:,1])&(F2[:,1]!=F2[:,2])&(F2[:,0]!=F2[:,2])
        F2=F2[good]
        if len(F2):
            sf=np.sort(F2,axis=1)
            _,keep=np.unique(sf,axis=0,return_index=True)
            F2=F2[np.sort(keep)]
        err=abs(len(F2)-int(target_faces))
        if len(F2)>20 and err<besterr:
            bestV,bestF,besterr=V2,F2,err
        if len(F2)>int(target_faces): lo=cell
        else: hi=cell
    mm=trimesh.Trimesh(vertices=bestV,faces=bestF,process=True)
    try: mm.remove_unreferenced_vertices(); mm.fix_normals()
    except Exception: pass
    return np.asarray(mm.vertices,np.float32),np.asarray(mm.faces,np.int32)

def _smoothstep01(x):
    x=np.clip(np.asarray(x,float),0.0,1.0)
    return x*x*(3.0-2.0*x)

def build_master_atlas(man,max_faces_each=60):
    """Preprocess Denver ONCE while preserving its shared anatomical coordinates."""
    sha=str(man.get("sha","unknown"))
    root=Path(man["root"])
    cache=root/f"master_atlas_v110_3_22_13_{max_faces_each}.npz"
    meta_path=root/f"master_atlas_v110_3_22_13_{max_faces_each}.json"
    if cache.exists() and meta_path.exists():
        d=np.load(cache,allow_pickle=True)
        return {
            "V":d["V"],"F":d["F"],"part_index":d["part_index"],
            "part_names":d["part_names"].tolist(),"part_sides":d["part_sides"].tolist(),
            "part_regions":d["part_regions"].tolist(),"unit_scale":float(d["unit_scale"][0]),
            "donor_frames":json.loads(meta_path.read_text(encoding="utf-8"))
        }

    us=_atlas_unit_scale(man)
    V_all=[]; F_all=[]; pidx=[]; names=[]; sides=[]; regions=[]; off=0
    muscle_parts=[p for p in man.get("parts",[]) if p.get("kind")=="muscle"]
    muscle_parts=sorted(muscle_parts,key=lambda p:(str(p.get("side")),p.get("name",""),p.get("path","")))
    for ip,p in enumerate(muscle_parts):
        V,F=_clean_mesh_master(p["path"])
        V=V*us
        if len(V)==0 or len(F)==0: continue
        Vs,Fs=_simplify_connected_mesh(V,F,max_faces_each)
        V_all.append(Vs.astype(np.float32)); F_all.append(Fs.astype(np.int32)+off)
        pidx.extend([len(names)]*len(Vs))
        names.append(p["name"]); sides.append(p.get("side") or "r"); regions.append(p.get("region") or "shank")
        off+=len(Vs)

    if not V_all:
        return None
    donor_frames={}
    for side in ("r","l"):
        for seg in ("pelvis","thigh","shank","foot"):
            ref=_master_donor_frame(man,side,seg,us)
            if ref is not None:
                o,B,L=ref
                donor_frames[f"{side}:{seg}"]={"o":np.asarray(o).tolist(),"B":np.asarray(B).tolist(),"L":float(L)}

    np.savez_compressed(
        cache,V=np.vstack(V_all).astype(np.float32),F=np.vstack(F_all).astype(np.int32),
        part_index=np.asarray(pidx,np.int16),
        part_names=np.asarray(names,dtype=object),part_sides=np.asarray(sides,dtype=object),
        part_regions=np.asarray(regions,dtype=object),unit_scale=np.asarray([us],np.float64)
    )
    meta_path.write_text(json.dumps(donor_frames,indent=2),encoding="utf-8")
    return {
        "V":np.vstack(V_all).astype(np.float32),"F":np.vstack(F_all).astype(np.int32),
        "part_index":np.asarray(pidx,np.int16),"part_names":names,"part_sides":sides,
        "part_regions":regions,"unit_scale":us,"donor_frames":donor_frames
    }

def _map_master_frame0(master,J,nm,skin0):
    """Register the master atlas ONCE to SKEL frame 0, segment-wise but coherently."""
    V0=np.asarray(master["V"],float)
    out=np.zeros_like(V0)
    seg_cache={}
    # target transform per side/segment
    for side in ("r","l"):
        for seg in ("pelvis","thigh","shank","foot"):
            key=f"{side}:{seg}"
            d=master["donor_frames"].get(key)
            if not d: continue
            do=np.asarray(d["o"],float); dB=np.asarray(d["B"],float); dL=float(d["L"])
            to,tB,tL=_segment_frame_from_joints(J,nm,0,side,seg)
            scale=tL/max(dL,1e-6)
            seg_cache[key]=(do,dB,to,tB,scale)

    # each muscle remains in the common Denver coordinate system and is transformed
    # using its anatomical chain; no independent PCA per STL.
    for ip,name in enumerate(master["part_names"]):
        side=master["part_sides"][ip] if master["part_sides"][ip] in ("r","l") else "r"
        region=master["part_regions"][ip]
        chain=MUSCLE_CHAIN.get(name,(region,region))
        idx=np.where(master["part_index"]==ip)[0]
        if len(idx)==0: continue
        P=V0[idx]
        k0=f"{side}:{chain[0]}"; k1=f"{side}:{chain[1]}"
        if k0 not in seg_cache: k0=f"{side}:{region}"
        if k1 not in seg_cache: k1=k0
        d0,B0,t0,T0,s0=seg_cache[k0]
        d1,B1,t1,T1,s1=seg_cache[k1]
        # map with both endpoint segment transforms and blend along muscle principal direction
        A=t0+((P-d0)@B0)*s0@T0.T if False else None
        P0=t0 + (((P-d0)@B0)*s0) @ T0.T
        P1=t1 + (((P-d1)@B1)*s1) @ T1.T
        if k0==k1:
            W=P0
        else:
            # weights derived from proximity to donor segment origins, preserving attachment ends
            dprox=np.linalg.norm(P-d0,axis=1)
            ddist=np.linalg.norm(P-d1,axis=1)
            w=dprox/(dprox+ddist+1e-9)
            w=np.clip(w,0.05,0.95)[:,None]
            W=(1-w)*P0+w*P1
        out[idx]=W
    return out,seg_cache

def _rigid_target_delta(J,nm,t,side,seg,base_frame):
    o0,B0,_=_segment_frame_from_joints(J,nm,0,side,seg)
    ot,Bt,_=_segment_frame_from_joints(J,nm,t,side,seg)
    R=Bt@B0.T
    return o0,ot,R


def _target_rigid_frame0_to_t(J,nm,t,side,seg,P):
    o0,B0,_=_segment_frame_from_joints(J,nm,0,side,seg)
    ot,Bt,_=_segment_frame_from_joints(J,nm,t,side,seg)
    R=Bt@B0.T
    return ot+(P-o0)@R.T

def _principal_coordinate(V):
    C=np.mean(V,axis=0)
    X=V-C
    try:
        _,_,vh=np.linalg.svd(X,full_matrices=False)
        axis=vh[0]
    except Exception:
        axis=np.array([0.0,0.0,1.0])
    q=X@axis
    return (q-np.min(q))/max(float(np.max(q)-np.min(q)),1e-12)

def _bounded_muscle_frame(V0,Vden,side,seg0,seg1,host,J,nm,t,donor_frames):
    Ph=_target_rigid_frame0_to_t(J,nm,t,side,host,V0)
    Pa=_target_rigid_frame0_to_t(J,nm,t,side,seg0,V0)
    Pb=_target_rigid_frame0_to_t(J,nm,t,side,seg1,V0)

    q=_principal_coordinate(Vden)
    da=donor_frames.get(f"{side}:{seg0}")
    db=donor_frames.get(f"{side}:{seg1}")
    if da and db:
        oa=np.asarray(da["o"],float)
        ob=np.asarray(db["o"],float)
        e0=np.mean(Vden[q<0.10],axis=0) if np.any(q<0.10) else Vden[np.argmin(q)]
        e1=np.mean(Vden[q>0.90],axis=0) if np.any(q>0.90) else Vden[np.argmax(q)]
        if np.linalg.norm(e0-oa)+np.linalg.norm(e1-ob) > np.linalg.norm(e1-oa)+np.linalg.norm(e0-ob):
            q=1.0-q

    # Central belly ~56% quasi-rigid; only terminal zones deform.
    wa=1.0-_smoothstep01(q/0.22)
    wb=_smoothstep01((q-0.78)/0.22)
    wh=np.clip(1.0-wa-wb,0.0,1.0)
    W=wh[:,None]*Ph + wa[:,None]*Pa + wb[:,None]*Pb

    # Shape preservation safeguard.
    dW=np.linalg.norm(np.ptp(W,axis=0))
    dH=np.linalg.norm(np.ptp(Ph,axis=0))
    ratio=dW/max(dH,1e-12)
    if ratio>1.18 or ratio<0.85:
        alpha=min(0.85,max(0.25,abs(ratio-1.0)))
        cW=np.mean(W,axis=0); cH=np.mean(Ph,axis=0)
        W=(1.0-alpha)*W + alpha*(Ph+(cW-cH))
    return W

def build_visible_human_muscle_sequence(man,joints,joint_names,skin_sequence,max_faces_each=900):
    """V110.3.22.13 — connected Denver muscle meshes + bounded skinning."""
    J=np.asarray(joints,np.float32)
    nm={str(n):i for i,n in enumerate(joint_names)}
    master=build_master_atlas(man,max_faces_each=max_faces_each)
    if master is None:
        return None,None

    Vden=np.asarray(master["V"],np.float32)
    F=np.asarray(master["F"],np.int32)
    V0=np.zeros_like(Vden,dtype=np.float32)
    host_by_part={}

    # Frame 0 morphology registration to one host segment, preserving each whole muscle.
    for ip,name in enumerate(master["part_names"]):
        side=master["part_sides"][ip] if master["part_sides"][ip] in ("r","l") else "r"
        region=master["part_regions"][ip]
        seg0,seg1=MUSCLE_CHAIN.get(name,(region,region))
        idx=np.where(master["part_index"]==ip)[0]
        if len(idx)==0:
            continue

        c=np.mean(Vden[idx],axis=0)
        def _dist(seg):
            d=master["donor_frames"].get(f"{side}:{seg}")
            return np.inf if not d else float(np.linalg.norm(c-np.asarray(d["o"],float)))
        host=seg0 if _dist(seg0)<=_dist(seg1) else seg1
        d=master["donor_frames"].get(f"{side}:{host}") or master["donor_frames"].get(f"{side}:{region}")
        if not d:
            V0[idx]=Vden[idx]
            host_by_part[ip]=region
            continue
        host_by_part[ip]=host

        do=np.asarray(d["o"],float)
        dB=np.asarray(d["B"],float)
        dL=max(float(d["L"]),1e-6)
        to,tB,tL=_segment_frame_from_joints(J,nm,0,side,host)
        sc=float(np.clip(tL/dL,0.35,3.0))
        V0[idx]=(to+(((Vden[idx]-do)@dB)*sc)@tB.T).astype(np.float32)

    frames=[V0.copy()]
    for t in range(1,len(J)):
        Vt=np.zeros_like(V0,dtype=np.float32)
        for ip,name in enumerate(master["part_names"]):
            side=master["part_sides"][ip] if master["part_sides"][ip] in ("r","l") else "r"
            region=master["part_regions"][ip]
            seg0,seg1=MUSCLE_CHAIN.get(name,(region,region))
            host=host_by_part.get(ip,region)
            idx=np.where(master["part_index"]==ip)[0]
            if len(idx)==0:
                continue
            try:
                if seg0==seg1:
                    Vt[idx]=_target_rigid_frame0_to_t(J,nm,t,side,host,V0[idx]).astype(np.float32)
                else:
                    Vt[idx]=_bounded_muscle_frame(
                        V0[idx],Vden[idx],side,seg0,seg1,host,J,nm,t,master["donor_frames"]
                    ).astype(np.float32)
            except Exception:
                Vt[idx]=_target_rigid_frame0_to_t(J,nm,t,side,host,V0[idx]).astype(np.float32)
        frames.append(Vt)

    return np.stack(frames,axis=0).astype(np.float32),F


# V110.3.22.13 · DENVER REAL BONE ATLAS
# The visible lower-limb skeleton uses Denver STL meshes.
# OpenSim/Hamner remains the invisible kinematic rig.
# ---------------------------------------------------------------------------

def _bone_segment(name):
    n=_norm(name)
    if n in ("pelvis","sacrum","coccyx"):
        return "pelvis"
    if n in ("femur","patella"):
        return "thigh"
    if n in ("tibia","fibula"):
        return "shank"
    return "foot"

def build_denver_bone_master(man,max_faces_each=70):
    """Build one compact REAL bone master atlas while preserving Denver shared coordinates."""
    root=Path(man["root"])
    cache=root/f"denver_real_bones_master_v160_sideproper_{max_faces_each}.npz"
    meta=root/f"denver_real_bones_master_v160_sideproper_{max_faces_each}.json"
    if cache.exists() and meta.exists():
        d=np.load(cache,allow_pickle=True)
        return {
            "V":d["V"],"F":d["F"],"part_index":d["part_index"],
            "part_names":d["part_names"].tolist(),
            "part_sides":d["part_sides"].tolist(),
            "part_segments":d["part_segments"].tolist(),
            "donor_frames":json.loads(meta.read_text(encoding="utf-8")),
            "unit_scale":float(d["unit_scale"][0])
        }

    us=_atlas_unit_scale(man)
    parts=[p for p in man.get("parts",[]) if p.get("kind")=="bone"]
    parts=sorted(parts,key=lambda p:(str(p.get("side")),p.get("region",""),p.get("name",""),p.get("path","")))
    Vs=[]; Fs=[]; pidx=[]; names=[]; sides=[]; segs=[]; off=0

    for p in parts:
        V,F=_clean_mesh_master(p["path"])
        V=V*us
        if len(V)==0 or len(F)==0:
            continue
        Vd,Fd=_simplify_connected_mesh(V,F,max_faces_each)
        Vs.append(Vd.astype(np.float32))
        Fs.append(Fd.astype(np.int32)+off)
        pidx.extend([len(names)]*len(Vd))
        names.append(p["name"])
        sides.append(p.get("side") or "r")
        segs.append(_bone_segment(p["name"]))
        off+=len(Vd)

    if not Vs:
        return None

    donor={}
    for side in ("r","l"):
        for seg in ("pelvis","thigh","shank","foot"):
            ref=_master_donor_frame(man,side,seg,us)
            if ref is not None:
                o,B,L=ref
                donor[f"{side}:{seg}"]={
                    "o":np.asarray(o,float).tolist(),
                    "B":np.asarray(B,float).tolist(),
                    "L":float(L)
                }

    V=np.vstack(Vs).astype(np.float32)
    F=np.vstack(Fs).astype(np.int32)
    np.savez_compressed(
        cache,V=V,F=F,part_index=np.asarray(pidx,np.int16),
        part_names=np.asarray(names,dtype=object),
        part_sides=np.asarray(sides,dtype=object),
        part_segments=np.asarray(segs,dtype=object),
        unit_scale=np.asarray([us],np.float64)
    )
    meta.write_text(json.dumps(donor,indent=2),encoding="utf-8")
    return {
        "V":V,"F":F,"part_index":np.asarray(pidx,np.int16),
        "part_names":names,"part_sides":sides,"part_segments":segs,
        "donor_frames":donor,"unit_scale":us
    }


def _semantic_target_delta(J,nm,t,side,seg):
    o0,B0,_=_semantic_target_frame(J,nm,0,side,seg)
    ot,Bt,_=_semantic_target_frame(J,nm,t,side,seg)
    return o0,ot,Bt@B0.T

def _similarity_no_reflection(A,B):
    """Least-squares uniform similarity A->B with det(R)=+1."""
    A=np.asarray(A,float); B=np.asarray(B,float)
    ca=A.mean(axis=0); cb=B.mean(axis=0); X=A-ca; Y=B-cb
    U,S,Vt=np.linalg.svd(X.T@Y); R=Vt.T@U.T
    if np.linalg.det(R)<0:
        Vt[-1,:]*=-1.0; R=Vt.T@U.T
    den=float(np.sum(X*X)); sc=float(np.sum(S)/den) if den>1e-12 else 1.0
    t=cb-sc*(R@ca)
    return R,sc,t

def _axis_rotation(axis,angle):
    axis=_unit(np.asarray(axis,float),[1,0,0]); x,y,z=axis
    c=float(np.cos(angle)); s=float(np.sin(angle)); C=1.0-c
    return np.asarray([[c+x*x*C,x*y*C-z*s,x*z*C+y*s],
                       [y*x*C+z*s,c+y*y*C,y*z*C-x*s],
                       [z*x*C-y*s,z*y*C+x*s,c+z*z*C]],float)


def _foot_candidate(L,J,nm,target,donor):
    A=np.stack([L[f"ankle_{donor}"],L[f"heel_{donor}"],L[f"toe_{donor}"]],axis=0)
    B=np.stack([J[0,nm[f"talus_{target}"]],J[0,nm[f"calcn_{target}"]],J[0,nm[f"toes_{target}"]]],axis=0)
    R,sc,t=_similarity_no_reflection(A,B); sc=float(np.clip(sc,0.35,3.0))
    mapped=(sc*(R@A.T)).T+t
    rmse=float(np.sqrt(np.mean(np.sum((mapped-B)**2,axis=1))))
    right,ant,up,_=_body_frame_target(J,nm,0)
    med=sc*(R@L[f"medial_cuneiform_{donor}"])+t
    lat=sc*(R@L[f"lateral_cuneiform_{donor}"])+t
    inward=(-1.0 if target=="r" else 1.0)*float(np.dot(med-lat,right))
    length=max(float(np.linalg.norm(B[2]-B[1])),1e-6)
    return {"target":target,"donor":donor,"R":R,"scale":sc,"t":t,
            "rmse":rmse,"medial_score":inward/length}


def _choose_foot_assignment(L,J,nm):
    choices=[]
    for mapping in ({"r":"r","l":"l"},{"r":"l","l":"r"}):
        cand={t:_foot_candidate(L,J,nm,t,d) for t,d in mapping.items()}
        positive=sum(cand[s]["medial_score"]>0 for s in ("r","l"))
        score=sum(cand[s]["medial_score"] for s in ("r","l"))-0.25*sum(cand[s]["rmse"] for s in ("r","l"))
        choices.append((positive,score,cand))
    choices.sort(key=lambda x:(x[0],x[1]),reverse=True)
    positive,score,best=choices[0]
    if positive<2:
        raise RuntimeError("QC anatómico bloqueado: no existe asignación bilateral con hallux medial")
    return best


def _neutral_foot_offset(J,nm,side):
    vals=[]
    for t in range(len(J)):
        right,ant,up,_=_body_frame_target(J,nm,t)
        f=np.asarray(J[t,nm[f"toes_{side}"]]-J[t,nm[f"calcn_{side}"]],float)
        f=f-right*np.dot(f,right); f=_unit(f,ant)
        vals.append(float(np.arcsin(np.clip(np.dot(f,up),-1,1))))
    return -float(np.median(vals))


def neutralize_skel_foot_skin(vertices,joints,joint_names,axis_map=None):
    """Viewer-only smooth neutral offset for the SKEL skin feet.

    Vertex ownership is frozen at frame 0 and assigned to only one side. The
    correction fades to zero at the ankle, so shank/knee/hip vertices are not
    altered. It never changes the saved fitting, joints, poses or q(t).
    """
    V=np.asarray(vertices,np.float32); J=np.asarray(joints,np.float32)
    nm={str(n):i for i,n in enumerate(joint_names)}
    if V.ndim!=3 or J.ndim!=3 or len(V)!=len(J):
        raise ValueError("SKEL skin/joints sequence incompatible")
    offsets={s:_neutral_foot_offset(J,nm,s) for s in ("r","l")}
    if max(abs(offsets["r"]),abs(offsets["l"]))>np.deg2rad(80) or abs(offsets["r"]-offsets["l"])>np.deg2rad(25):
        raise RuntimeError("QC piel-pie bloqueado: offset neutro implausible o discordante")
    raw={}; nearest={}
    for side in ("r","l"):
        ankle=J[0,nm[f"talus_{side}"]]; heel=J[0,nm[f"calcn_{side}"]]; toe=J[0,nm[f"toes_{side}"]]
        L=max(float(np.linalg.norm(toe-heel)),1e-5)
        d=np.minimum(np.linalg.norm(V[0]-heel,axis=1),np.linalg.norm(V[0]-toe,axis=1))
        da=np.linalg.norm(V[0]-ankle,axis=1)
        raw[side]=np.clip((1.05*L-d)/(.38*L),0,1)*np.clip((1.15*L-da)/(.35*L),0,1)
        nearest[side]=np.minimum(d,da)
    owner=np.where(nearest["r"]<=nearest["l"],"r","l")
    weights={s:np.where(owner==s,raw[s],0.0).astype(np.float32) for s in ("r","l")}
    out=V.copy()
    for t in range(len(J)):
        for side in ("r","l"):
            ankle=J[t,nm[f"talus_{side}"]]
            _,B,_=_semantic_target_frame(J,nm,t,side,"foot")
            R=_axis_rotation(B[:,0],offsets[side])
            rot=ankle+(out[t]-ankle)@R.T; w=weights[side][:,None]
            out[t]=(1.0-w)*out[t]+w*rot
    return out,{"neutral_offset_deg":{s:float(np.degrees(offsets[s])) for s in offsets},"pass":True}


def _register_denver_bones_frame0(master,J,nm,man=None,return_qc=False,axis_map=None):
    """V161 anatomy-directed registration with fail-closed QC.

    Pelvis AP is corrected only when the registered sacrum is not posterior.
    Foot identity/swap is selected from bilateral hallux-medial tests. A fixed
    side-specific neutral offset is then composed with every dynamic foot frame.
    """
    V=np.asarray(master["V"],float); out=np.zeros_like(V)
    L=_denver_landmarks(man,float(master.get("unit_scale",1.0))) if man is not None else None
    foot_similarity={}
    if L is not None:
        foot_similarity=_choose_foot_assignment(L,J,nm)
    neutral={s:_neutral_foot_offset(J,nm,s) for s in ("r","l")}
    if max(abs(neutral["r"]),abs(neutral["l"]))>np.deg2rad(80) or abs(neutral["r"]-neutral["l"])>np.deg2rad(25):
        raise RuntimeError("QC anatómico bloqueado: offset neutro del tobillo implausible o discordante")
    target_by_part=[]
    for ip,name in enumerate(master["part_names"]):
        side=master["part_sides"][ip] if master["part_sides"][ip] in ("r","l") else "r"
        seg=master["part_segments"][ip]; idx=np.where(master["part_index"]==ip)[0]
        if len(idx)==0:
            target_by_part.append((side,seg)); continue
        d=master["donor_frames"].get(f"{side}:{seg}")
        if d is None:
            target_by_part.append((side,seg)); continue
        # Foot-specific similarity is precomputed in master if available.
        target_side=side
        fs=None
        if seg=="foot":
            for _target,_fs in foot_similarity.items():
                if _fs.get("donor")==side:
                    target_side=_target; fs=_fs; break
        if seg=="foot" and fs is not None:
            R=np.asarray(fs["R"],float); sc=float(fs["scale"]); tt=np.asarray(fs["t"],float)
            W=(sc*(R@V[idx].T)).T+tt
            ankle=np.asarray(J[0,nm[f"talus_{target_side}"]],float)
            _,FB,_=_semantic_target_frame(J,nm,0,target_side,"foot")
            Rn=_axis_rotation(FB[:,0],neutral[target_side])
            out[idx]=ankle+(W-ankle)@Rn.T
            target_by_part.append((target_side,seg))
            continue
        do=np.asarray(d["o"],float); dB=np.asarray(d["B"],float); dL=max(float(d["L"]),1e-6)
        to,tB,tL=_semantic_target_frame(J,nm,0,side,seg); sc=float(np.clip(tL/dL,0.35,3.0))
        out[idx]=to+(((V[idx]-do)@dB)*sc)@tB.T
        target_by_part.append((target_side,seg))

    # Pelvis AP fail-safe: sacrum must lie posterior to the hip midpoint.
    sac=[]
    for ip,name in enumerate(master["part_names"]):
        if master["part_segments"][ip]=="pelvis" and _norm(name) in ("sacrum","coccyx"):
            sac.extend(np.where(master["part_index"]==ip)[0].tolist())
    right,ant,up,mid=_body_frame_target(J,nm,0)
    parity=float(np.linalg.det(np.asarray(axis_map,float))) if axis_map is not None else 1.0
    display_ant=ant if parity>=0 else -ant
    pelvis_flipped=False
    if sac:
        ap=float(np.dot(np.mean(out[np.asarray(sac,int)],axis=0)-mid,display_ant))
        if ap>=0:
            for ip,(_side,_seg) in enumerate(target_by_part):
                if _seg!="pelvis": continue
                idx=np.where(master["part_index"]==ip)[0]
                rel=out[idx]-mid; out[idx]=out[idx]-2.0*(rel@display_ant)[:,None]*display_ant[None,:]
            pelvis_flipped=True
            ap=-ap
        if ap>=-1e-5:
            raise RuntimeError("QC anatómico bloqueado: no puede garantizarse sacro posterior")
    else:
        raise RuntimeError("QC anatómico bloqueado: atlas sin sacro/cóccix identificable")
    qc={"version":"161-anatomy-fail-closed","pelvis_ap_corrected":pelvis_flipped,
        "sacrum_posterior_margin":float(-ap),
        "foot_assignment":{t:foot_similarity[t]["donor"] for t in foot_similarity},
        "hallux_medial_score":{t:float(foot_similarity[t]["medial_score"]) for t in foot_similarity},
        "neutral_offset_deg":{s:float(np.degrees(neutral[s])) for s in neutral},
        "axis_map_determinant":parity,
        "pass":True}
    return (out,target_by_part,qc) if return_qc else out

def build_visible_human_bone_sequence(man,joints,joint_names,skin_sequence,max_faces_each=70,axis_map=None):
    """Real Denver bone STL sequence driven by the SKEL/OpenSim kinematic rig.

    The STL anatomy is visible.
    Hamner/OpenSim is only the hidden rig.
    """
    J=np.asarray(joints,np.float32)
    nm={str(n):i for i,n in enumerate(joint_names)}
    master=build_denver_bone_master(man,max_faces_each=max_faces_each)
    if master is None:
        return None,None

    V0,target_by_part,qc=_register_denver_bones_frame0(master,J,nm,man,return_qc=True,axis_map=axis_map)
    V0=V0.astype(np.float32)
    frames=[V0]
    for t in range(1,len(J)):
        Vt=np.zeros_like(V0)
        for ip,name in enumerate(master["part_names"]):
            side,seg=target_by_part[ip]
            idx=np.where(master["part_index"]==ip)[0]
            if len(idx)==0:
                continue
            try:
                o0,ot,R=_semantic_target_delta(J,nm,t,side,seg)
                Vt[idx]=ot+(V0[idx]-o0)@R.T
            except Exception:
                Vt[idx]=V0[idx]
        frames.append(Vt.astype(np.float32))
    return np.stack(frames,axis=0),np.asarray(master["F"],np.int32)


# ---------------------------------------------------------------------------
# V110.3.22.13 · COMPACT DENVER RIG
# One anatomical mesh + compact transforms instead of 75 full mesh copies.
# ---------------------------------------------------------------------------

def _segment_keys_compact():
    return [(s,g) for s in ("r","l") for g in ("pelvis","thigh","shank","foot")]

def _segment_affines_compact(J,nm):
    """Return [T,8,12] affine transforms from frame0 to each SKEL frame.
    Each transform stores R(row-major 9) + translation b(3), with p'=R@p+b.
    """
    keys=_segment_keys_compact()
    out=np.zeros((len(J),len(keys),12),dtype=np.float32)
    for t in range(len(J)):
        for si,(side,seg) in enumerate(keys):
            o0,B0,_=_semantic_target_frame(J,nm,0,side,seg)
            ot,Bt,_=_semantic_target_frame(J,nm,t,side,seg)
            R=Bt@B0.T
            b=ot-R@o0
            out[t,si,:9]=R.reshape(-1)
            out[t,si,9:]=b
    return keys,out

def _muscle_frame0_compact(master,J,nm):
    Vden=np.asarray(master["V"],np.float32)
    V0=np.zeros_like(Vden,dtype=np.float32)
    keys=_segment_keys_compact(); key_to_idx={k:i for i,k in enumerate(keys)}
    seg0_idx=np.zeros(len(Vden),np.int16)
    seg1_idx=np.zeros(len(Vden),np.int16)
    host_idx=np.zeros(len(Vden),np.int16)
    wa=np.zeros(len(Vden),np.float32); wb=np.zeros(len(Vden),np.float32); wh=np.ones(len(Vden),np.float32)

    for ip,name in enumerate(master["part_names"]):
        side=master["part_sides"][ip] if master["part_sides"][ip] in ("r","l") else "r"
        region=master["part_regions"][ip]
        seg0,seg1=MUSCLE_CHAIN.get(name,(region,region))
        idx=np.where(master["part_index"]==ip)[0]
        if len(idx)==0: continue
        c=np.mean(Vden[idx],axis=0)
        def dd(seg):
            d=master["donor_frames"].get(f"{side}:{seg}")
            return np.inf if not d else float(np.linalg.norm(c-np.asarray(d["o"],float)))
        host=seg0 if dd(seg0)<=dd(seg1) else seg1
        d=master["donor_frames"].get(f"{side}:{host}") or master["donor_frames"].get(f"{side}:{region}")
        if d is None:
            V0[idx]=Vden[idx]; host=region
        else:
            do=np.asarray(d["o"],float); dB=np.asarray(d["B"],float); dL=max(float(d["L"]),1e-6)
            to,tB,tL=_segment_frame_from_joints(J,nm,0,side,host)
            sc=float(np.clip(tL/dL,0.35,3.0))
            V0[idx]=(to+(((Vden[idx]-do)@dB)*sc)@tB.T).astype(np.float32)

        s0=key_to_idx.get((side,seg0),key_to_idx[(side,region)])
        s1=key_to_idx.get((side,seg1),s0)
        sh=key_to_idx.get((side,host),key_to_idx[(side,region)])
        seg0_idx[idx]=s0; seg1_idx[idx]=s1; host_idx[idx]=sh

        if seg0!=seg1:
            q=_principal_coordinate(Vden[idx])
            da=master["donor_frames"].get(f"{side}:{seg0}"); db=master["donor_frames"].get(f"{side}:{seg1}")
            if da and db:
                oa=np.asarray(da["o"],float); ob=np.asarray(db["o"],float)
                e0=np.mean(Vden[idx][q<.10],axis=0) if np.any(q<.10) else Vden[idx][np.argmin(q)]
                e1=np.mean(Vden[idx][q>.90],axis=0) if np.any(q>.90) else Vden[idx][np.argmax(q)]
                if np.linalg.norm(e0-oa)+np.linalg.norm(e1-ob)>np.linalg.norm(e1-oa)+np.linalg.norm(e0-ob): q=1.0-q
            waa=1.0-_smoothstep01(q/0.22)
            wbb=_smoothstep01((q-0.78)/0.22)
            whh=np.clip(1.0-waa-wbb,0.0,1.0)
            wa[idx]=waa.astype(np.float32); wb[idx]=wbb.astype(np.float32); wh[idx]=whh.astype(np.float32)
    return V0,seg0_idx,seg1_idx,host_idx,wa,wb,wh

def build_visible_human_compact_payload(man,joints,joint_names,max_bone_faces_each=700,max_muscle_faces_each=300,axis_map=None):
    """Memory-safe Denver representation for Streamlit Cloud.

    Returns ONE bone mesh and ONE muscle mesh plus compact segment transforms.
    It intentionally does not allocate vertices[75,...] for Denver anatomy.
    """
    J=np.asarray(joints,np.float32); nm={str(n):i for i,n in enumerate(joint_names)}
    keys,aff=_segment_affines_compact(J,nm)
    key_to_idx={k:i for i,k in enumerate(keys)}

    bm=build_denver_bone_master(man,max_faces_each=max_bone_faces_each)
    mm=build_master_atlas(man,max_faces_each=max_muscle_faces_each)
    if bm is None or mm is None: return None

    BV0,target_by_part,anatomy_qc=_register_denver_bones_frame0(bm,J,nm,man,return_qc=True,axis_map=axis_map)
    BV0=BV0.astype(np.float32)
    BSEG=np.zeros(len(BV0),np.int16)
    for ip in range(len(bm["part_names"])):
        side,seg=target_by_part[ip]
        idx=np.where(bm["part_index"]==ip)[0]
        BSEG[idx]=key_to_idx[(side,seg)]

    MV0,MS0,MS1,MH,MWA,MWB,MWH=_muscle_frame0_compact(mm,J,nm)
    return {
        "segment_keys":[f"{a}:{b}" for a,b in keys],
        "anatomy_qc":anatomy_qc,
        "affines":aff,
        "bone_vertices0":BV0,"bone_faces":np.asarray(bm["F"],np.int32),"bone_seg":BSEG,
        "muscle_vertices0":MV0,"muscle_faces":np.asarray(mm["F"],np.int32),
        "muscle_seg0":MS0,"muscle_seg1":MS1,"muscle_host":MH,
        "muscle_wa":MWA,"muscle_wb":MWB,"muscle_wh":MWH,
    }
