from __future__ import annotations
from pathlib import Path
import json, xml.etree.ElementTree as ET
import numpy as np
import requests
HAMNER_MODEL_URL="https://raw.githubusercontent.com/opensim-org/opensim-models/master/Models/Hamner/FullBodyModel_Hamner2010_v2_0.osim"
HAMNER_GEOM_BASES=["https://raw.githubusercontent.com/opensim-org/opensim-models/master/Geometry","https://raw.githubusercontent.com/opensim-org/opensim-models/master/Models/Hamner/Geometry"]
CACHE_ROOT=Path("/tmp/physiosentinel_opensim_hamner_v3218")
BODY_MAP={
"pelvis":{"kind":"pelvis"},
"femur_r":{"kind":"long","a":"femur_r","b":"tibia_r"},"femur_l":{"kind":"long","a":"femur_l","b":"tibia_l"},
"tibia_r":{"kind":"long","a":"tibia_r","b":"talus_r"},"tibia_l":{"kind":"long","a":"tibia_l","b":"talus_l"},
"talus_r":{"kind":"block","a":"talus_r","b":"calcn_r"},"talus_l":{"kind":"block","a":"talus_l","b":"calcn_l"},
"calcn_r":{"kind":"foot","a":"calcn_r","b":"toes_r"},"calcn_l":{"kind":"foot","a":"calcn_l","b":"toes_l"},
"toes_r":{"kind":"foot","a":"calcn_r","b":"toes_r","distal":True},"toes_l":{"kind":"foot","a":"calcn_l","b":"toes_l","distal":True},
"torso":{"kind":"torso"},
"humerus_r":{"kind":"long","a":"humerus_r","b":"ulna_r"},"humerus_l":{"kind":"long","a":"humerus_l","b":"ulna_l"},
"ulna_r":{"kind":"long","a":"ulna_r","b":"hand_r"},"ulna_l":{"kind":"long","a":"ulna_l","b":"hand_l"},
"radius_r":{"kind":"long","a":"radius_r","b":"hand_r"},"radius_l":{"kind":"long","a":"radius_l","b":"hand_l"},
"hand_r":{"kind":"hand","a":"radius_r","b":"hand_r"},"hand_l":{"kind":"hand","a":"radius_l","b":"hand_l"}}
def _strip_ns(root):
    for e in root.iter():
        if '}' in e.tag: e.tag=e.tag.split('}',1)[1]
    return root
def _get(url,dst,timeout=45):
    dst=Path(dst); dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists() and dst.stat().st_size>32:return 'cached'
    r=requests.get(url,timeout=timeout); r.raise_for_status(); dst.write_bytes(r.content); return 'downloaded'
def _get_geom(fn,dst,force=False):
    dst=Path(dst)
    if force and dst.exists(): dst.unlink()
    if dst.exists() and dst.stat().st_size>32:return 'cached',None
    errs=[]
    for base in HAMNER_GEOM_BASES:
        try:return _get(f"{base}/{fn}",dst),base
        except Exception as e:
            errs.append(f"{base}: {type(e).__name__}: {e}")
            if dst.exists():
                try:dst.unlink()
                except:pass
    raise RuntimeError(' | '.join(errs))
def _parse_model(path):
    root=_strip_ns(ET.parse(path).getroot()); model=root.find('.//Model')
    meta={'model_name':model.attrib.get('name','3DGaitModelwithSimpleArms') if model is not None else '', 'bodies':{}, 'mesh_files':[]}
    for body in root.findall('.//BodySet/objects/Body'):
        bn=body.attrib.get('name',''); meshes=[]
        for mesh in body.findall('./attached_geometry/Mesh'):
            mf=mesh.find('mesh_file'); sf=mesh.find('scale_factors')
            if mf is None or not (mf.text or '').strip():continue
            fn=(mf.text or '').strip(); sc=[1.,1.,1.]
            if sf is not None and (sf.text or '').strip():
                try:sc=[float(x) for x in sf.text.split()[:3]]
                except:pass
            meshes.append({'file':fn,'scale':sc}); meta['mesh_files'].append(fn)
        meta['bodies'][bn]={'meshes':meshes}
    meta['mesh_files']=sorted(set(meta['mesh_files'])); return meta
def prepare_hamner(force=False):
    root=CACHE_ROOT; root.mkdir(parents=True,exist_ok=True); mp=root/'FullBodyModel_Hamner2010_v2_0.osim'
    if force and mp.exists():mp.unlink()
    try:_get(HAMNER_MODEL_URL,mp); meta=_parse_model(mp)
    except Exception as e:return {'ok':False,'error':f'{type(e).__name__}: {e}','root':str(root)}
    gdir=root/'Geometry'; gdir.mkdir(exist_ok=True); downloaded=cached=failed=0; errs=[]
    for fn in meta['mesh_files']:
        try:
            st,_=_get_geom(fn,gdir/fn,force=force); downloaded+=st=='downloaded'; cached+=st=='cached'
        except Exception as e:failed+=1; errs.append(f'{fn}: {type(e).__name__}: {e}')
    man={'meta':meta,'downloaded':downloaded,'cached':cached,'failed':failed,'errors':errs}
    (root/'manifest.json').write_text(json.dumps(man,ensure_ascii=False,indent=2),encoding='utf-8')
    return {'ok':True,'root':str(root),'model':meta['model_name'],'mesh_count':len(meta['mesh_files']),'downloaded':downloaded,'cached':cached,'failed':failed,'errors':errs}
def hamner_status():
    mp=CACHE_ROOT/'manifest.json'
    if not mp.exists():return {'ready':False,'root':str(CACHE_ROOT)}
    try:
        man=json.loads(mp.read_text(encoding='utf-8')); files=man.get('meta',{}).get('mesh_files',[])
        present=sum((CACHE_ROOT/'Geometry'/f).exists() for f in files)
        return {'ready':present>0,'root':str(CACHE_ROOT),'present':present,'mesh_count':len(files),'model':man.get('meta',{}).get('model_name','')}
    except:return {'ready':False,'root':str(CACHE_ROOT)}
def _parse_vtp_ascii(path):
    try:root=_strip_ns(ET.parse(path).getroot())
    except:return np.empty((0,3),np.float32),np.empty((0,3),np.int32)
    piece=root.find('.//Piece')
    if piece is None:return np.empty((0,3),np.float32),np.empty((0,3),np.int32)
    da=piece.find('./Points/DataArray')
    if da is None or da.attrib.get('format','ascii').lower()!='ascii':return np.empty((0,3),np.float32),np.empty((0,3),np.int32)
    vals=np.fromstring(da.text or '',sep=' ',dtype=np.float32)
    if vals.size%3:return np.empty((0,3),np.float32),np.empty((0,3),np.int32)
    V=vals.reshape(-1,3); polys=piece.find('./Polys')
    if polys is None:return V,np.empty((0,3),np.int32)
    conn=offs=None
    for x in polys.findall('./DataArray'):
        nm=(x.attrib.get('Name','') or '').lower(); arr=np.fromstring(x.text or '',sep=' ',dtype=np.int64)
        if 'connect' in nm:conn=arr
        elif 'offset' in nm:offs=arr
    if conn is None or offs is None:return V,np.empty((0,3),np.int32)
    faces=[]; st=0
    for en in offs:
        poly=conn[st:int(en)]; st=int(en)
        if len(poly)==3:faces.append(poly.tolist())
        elif len(poly)>3:
            for k in range(1,len(poly)-1):faces.append([int(poly[0]),int(poly[k]),int(poly[k+1])])
    return V,np.asarray(faces,dtype=np.int32)
def load_hamner_bodies():
    st=hamner_status()
    if not st.get('ready'):return {'bodies':{},'unsupported':[]}
    man=json.loads((CACHE_ROOT/'manifest.json').read_text(encoding='utf-8')); out={}; unsupported=[]
    for body,bmeta in man.get('meta',{}).get('bodies',{}).items():
        if body not in BODY_MAP:continue
        VV=[];FF=[];off=0
        for mm in bmeta.get('meshes',[]):
            V,F=_parse_vtp_ascii(CACHE_ROOT/'Geometry'/mm['file'])
            if V.size==0 or F.size==0:unsupported.append(mm['file']);continue
            V=(V*np.asarray(mm.get('scale',[1,1,1]),np.float32)).astype(np.float32); VV.append(V);FF.append(F+off);off+=len(V)
        if VV:out[body]={'V':np.vstack(VV).astype(np.float32),'F':np.vstack(FF).astype(np.int32),**BODY_MAP[body]}
    return {'bodies':out,'unsupported':unsupported}
def _frame(a,b,lr):
    z=np.asarray(b,float)-np.asarray(a,float); z=z/max(np.linalg.norm(z),1e-9); x=np.asarray(lr,float)-np.dot(lr,z)*z
    if np.linalg.norm(x)<1e-6:x=np.array([0.,0.,1.])-np.dot(np.array([0.,0.,1.]),z)*z
    x=x/max(np.linalg.norm(x),1e-9); y=np.cross(z,x); y=y/max(np.linalg.norm(y),1e-9); return x,y,z
def _safe_unit(v,fallback):
    v=np.asarray(v,float)
    n=np.linalg.norm(v)
    if n<1e-9:
        return np.asarray(fallback,float)
    return v/n

def _pelvic_frame(J,nm,frame):
    """Return AP, UP, ML axes in SKEL coordinates.
    OpenSim convention: +X anterior, +Y superior, +Z right.
    """
    pel=nm.get("pelvis"); hr=nm.get("femur_r"); hl=nm.get("femur_l")
    lum=nm.get("lumbar_body"); th=nm.get("thorax")
    if pel is None:
        return np.eye(3)
    ml=_safe_unit(J[frame,hr]-J[frame,hl],[0,0,1]) if hr is not None and hl is not None else np.array([0,0,1.])
    if lum is not None:
        up=_safe_unit(J[frame,lum]-J[frame,pel],[0,1,0])
    elif th is not None:
        up=_safe_unit(J[frame,th]-J[frame,pel],[0,1,0])
    else:
        up=np.array([0,1.,0])
    # Correct handedness: anterior = superior × right.
    ap=_safe_unit(np.cross(up,ml),[1,0,0])
    # re-orthogonalize
    ml=_safe_unit(np.cross(ap,up),ml)
    return np.stack([ap,up,ml],axis=1)

def _foot_frame(J,nm,frame,side):
    """Anatomical foot frame:
    +X forward = calcaneus -> toes
    +Y up = component of talus->tibia perpendicular to forward
    +Z right = X × Y (sign adjusted by side/global pelvis ML)
    """
    cal=nm.get(f"calcn_{side}"); toes=nm.get(f"toes_{side}")
    tal=nm.get(f"talus_{side}"); tib=nm.get(f"tibia_{side}")
    if cal is None or toes is None:
        return None
    fwd=_safe_unit(J[frame,toes]-J[frame,cal],[1,0,0])
    if tal is not None and tib is not None:
        up0=J[frame,tib]-J[frame,tal]
    else:
        up0=np.array([0,1.,0])
    up=up0-np.dot(up0,fwd)*fwd
    up=_safe_unit(up,[0,1,0])
    right=_safe_unit(np.cross(fwd,up),[0,0,1])
    # Match side orientation to pelvic right axis.
    P=_pelvic_frame(J,nm,frame)
    global_right=P[:,2]
    if np.dot(right,global_right)<0:
        right=-right
        up=-up
    return np.stack([fwd,up,right],axis=1)

def _hand_frame(J,nm,frame,side):
    """Hand frame from forearm/hand chain.
    +X = wrist/forearm -> hand, +Y dorsal-ish using elbow plane, +Z lateral.
    """
    rad=nm.get(f"radius_{side}"); hand=nm.get(f"hand_{side}")
    ulna=nm.get(f"ulna_{side}"); hum=nm.get(f"humerus_{side}")
    if hand is None:
        return None
    if rad is not None:
        origin=J[frame,rad]
    elif ulna is not None:
        origin=J[frame,ulna]
    else:
        return None
    fwd=_safe_unit(J[frame,hand]-origin,[1,0,0])
    if hum is not None and ulna is not None:
        ref=J[frame,hum]-J[frame,ulna]
    else:
        ref=np.array([0,1.,0])
    up=ref-np.dot(ref,fwd)*fwd
    up=_safe_unit(up,[0,1,0])
    right=_safe_unit(np.cross(fwd,up),[0,0,1])
    P=_pelvic_frame(J,nm,frame)
    gr=P[:,2]
    # right hand should follow +ML, left -ML
    desired=gr if side=="r" else -gr
    if np.dot(right,desired)<0:
        right=-right
        up=-up
    return np.stack([fwd,up,right],axis=1)

def _native_body_to_target(V, target_frame, center, scale, native_center=None):
    """Map OpenSim native axes directly:
       native +X -> target +X
       native +Y -> target +Y
       native +Z -> target +Z
    """
    V=np.asarray(V,float)
    if native_center is None:
        native_center=np.mean(V,axis=0)
    X=(V-native_center)*float(scale)
    return np.asarray(center,float)[None,:] + X @ np.asarray(target_frame,float).T


def _skin_local_extents(vertices, center, frame_axes, radius=0.30, min_points=80):
    V=np.asarray(vertices,float); c=np.asarray(center,float); A=np.asarray(frame_axes,float)
    d=np.linalg.norm(V-c,axis=1)
    pts=V[d<=float(radius)]
    if len(pts)<min_points:
        k=min(max(min_points,1),len(V))
        if k==0: return np.array([0.08,0.12,0.08],float)
        pts=V[np.argpartition(d,k-1)[:k]]
    Q=(pts-c)@A
    lo=np.percentile(Q,10,axis=0); hi=np.percentile(Q,90,axis=0)
    return np.maximum(0.5*(hi-lo),np.array([0.025,0.025,0.025]))

def _native_scaled_to_target(V,target_frame,center,scales_xyz,native_center=None):
    V=np.asarray(V,float)
    if native_center is None: native_center=np.mean(V,axis=0)
    X=(V-native_center)*np.asarray(scales_xyz,float)[None,:]
    return np.asarray(center,float)[None,:]+X@np.asarray(target_frame,float).T

def _find_head_component(V):
    V=np.asarray(V,float)
    return V[:,1]>=np.percentile(V[:,1],78)


def _endpoint_centroid_y(V, proximal=True, frac=0.10):
    """Centroid of native long-bone endpoint using OpenSim Y coordinate."""
    V=np.asarray(V,float)
    y=V[:,1]
    q=np.quantile(y,1.0-frac if proximal else frac)
    pts=V[y>=q] if proximal else V[y<=q]
    if len(pts)==0:
        return np.mean(V,axis=0)
    return np.mean(pts,axis=0)

def _register_long_bone_native(V, prox_target, dist_target, pelvic_frame):
    """Rigid-ish native long-bone registration.
    Uses actual proximal/distal endpoint centroids of the mesh rather than
    assuming max-Y itself is the joint center.
    """
    V=np.asarray(V,float)
    a=np.asarray(prox_target,float); b=np.asarray(dist_target,float)
    cp=_endpoint_centroid_y(V,True)
    cd=_endpoint_centroid_y(V,False)
    native_vec=cd-cp
    native_len=max(np.linalg.norm(native_vec),1e-9)
    target_vec=b-a
    target_len=max(np.linalg.norm(target_vec),1e-9)
    scale=target_len/native_len

    z=_safe_unit(target_vec,[0,1,0])
    # Target transverse reference: pelvic ML projected orthogonal to bone axis.
    x=np.asarray(pelvic_frame[:,2],float)
    x=x-np.dot(x,z)*z
    x=_safe_unit(x,pelvic_frame[:,0])
    y=_safe_unit(np.cross(z,x),pelvic_frame[:,1])

    # Native basis: long axis proximal->distal + explicit native transverse axes.
    nz=_safe_unit(native_vec,[0,-1,0])
    nx=np.array([1.,0.,0.])
    nx=nx-np.dot(nx,nz)*nz
    nx=_safe_unit(nx,[0,0,1])
    ny=_safe_unit(np.cross(nz,nx),[0,0,1])
    Bn=np.stack([nx,ny,nz],axis=1)
    Bt=np.stack([x,y,z],axis=1)

    local=(V-cp)@Bn
    W=a + (local*scale)@Bt.T
    return W.astype(np.float32), float(scale)

def _constrained_hand_registration(V, center, target_frame, target_len):
    """Orient hand morphology with PCA but constrain axes/sign to anatomical frame.
    Principal axis -> forearm/hand direction, second -> dorsal/palmar, third -> lateral.
    """
    V=np.asarray(V,float)
    C=np.mean(V,axis=0)
    X=V-C
    _,_,vt=np.linalg.svd(X,full_matrices=False)
    B=vt.T.copy()

    # Ensure right-handed native PCA basis.
    if np.linalg.det(B)<0:
        B[:,2]*=-1

    # PCA coordinates sorted by variance are already columns 0,1,2.
    Q=X@B
    spans=np.maximum(np.ptp(Q,axis=0),1e-6)

    # Principal axis is hand length.
    width=target_len*0.50
    thick=target_len*0.22
    scales=np.array([target_len/spans[0], width/spans[1], thick/spans[2]])
    scales=np.clip(scales,0.25,2.0)
    Qs=Q*scales[None,:]

    T=np.asarray(target_frame,float).copy()

    # Determine principal-axis sign from native long-axis endpoint relation.
    # Keep deterministic: distal end gets positive target +X.
    i_hi=np.argmax(Q[:,0]); i_lo=np.argmin(Q[:,0])
    # If native distal-like extreme is arbitrary, enforce consistent positive axis.
    if np.linalg.norm(V[i_hi]-C) < np.linalg.norm(V[i_lo]-C):
        Qs[:,0]*=-1

    return (np.asarray(center,float)[None,:] + Qs@T.T).astype(np.float32)


def _pelvis_center_and_scale(V,J,nm,frame):
    """Place pelvis from bilateral hip centers instead of generic pelvis centroid only."""
    pel=nm.get("pelvis"); hr=nm.get("femur_r"); hl=nm.get("femur_l")
    if pel is None:
        return None,1.0
    center=np.asarray(J[frame,pel],float)
    if hr is not None and hl is not None:
        hips=0.5*(J[frame,hr]+J[frame,hl])
        # blend SKEL pelvis and hip midpoint to reduce socket/head separation
        center=0.35*center+0.65*hips
        hip_width=max(float(np.linalg.norm(J[0,hr]-J[0,hl])),1e-6)
        zspan=max(float(np.ptp(V[:,2])),1e-6)
        scale=float(np.clip((hip_width*1.02)/zspan,0.35,2.5))
    else:
        scale=1.0
    return center,scale


def _endpoint_centroid_y_v32110(V, proximal=True, frac=0.12):
    V=np.asarray(V,float); y=V[:,1]
    q=np.quantile(y,1-frac if proximal else frac)
    pts=V[y>=q] if proximal else V[y<=q]
    return np.mean(pts,axis=0) if len(pts) else np.mean(V,axis=0)

def _register_long_bone_v32110(V, a, b, P):
    V=np.asarray(V,float); a=np.asarray(a,float); b=np.asarray(b,float)
    cp=_endpoint_centroid_y_v32110(V,True); cd=_endpoint_centroid_y_v32110(V,False)
    nv=cd-cp; tv=b-a
    nL=max(np.linalg.norm(nv),1e-9); tL=max(np.linalg.norm(tv),1e-9)
    nz=_safe_unit(nv,[0,-1,0]); tz=_safe_unit(tv,[0,1,0])
    nx=np.array([1.,0.,0.]); nx=nx-np.dot(nx,nz)*nz; nx=_safe_unit(nx,[0,0,1])
    ny=_safe_unit(np.cross(nz,nx),[0,0,1])
    tx=P[:,2]-np.dot(P[:,2],tz)*tz; tx=_safe_unit(tx,P[:,0])
    ty=_safe_unit(np.cross(tz,tx),P[:,1])
    Bn=np.stack([nx,ny,nz],axis=1); Bt=np.stack([tx,ty,tz],axis=1)
    Q=(V-cp)@Bn
    return (a + (Q*(tL/nL))@Bt.T).astype(np.float32)

def _hand_fit_v32110(V,wrist,hand_center,T,forearm):
    V=np.asarray(V,float); C=np.mean(V,axis=0); X=V-C
    _,_,vt=np.linalg.svd(X,full_matrices=False)
    B=vt.T
    if np.linalg.det(B)<0: B[:,2]*=-1
    Q=X@B
    spans=np.maximum(np.ptp(Q,axis=0),1e-6)

    # More conservative size than previous branch.
    target_len=float(np.clip(0.46*forearm,0.075,0.16))
    target_w=target_len*0.44
    target_t=target_len*0.18
    sc=np.array([target_len/spans[0],target_w/spans[1],target_t/spans[2]])
    sc=np.clip(sc,0.12,1.2)
    Q=Q*sc[None,:]

    # Place proximal extreme at wrist; point long axis toward hand center.
    Q[:,0]-=np.min(Q[:,0])
    fwd=_safe_unit(np.asarray(hand_center)-np.asarray(wrist),T[:,0])
    up=T[:,1]-np.dot(T[:,1],fwd)*fwd; up=_safe_unit(up,T[:,1])
    lat=_safe_unit(np.cross(fwd,up),T[:,2])
    TT=np.stack([fwd,up,lat],axis=1)
    return (np.asarray(wrist)[None,:]+Q@TT.T).astype(np.float32)

def build_hamner_fullbody_frame(bodies,joints,joint_names,frame=0,skin_vertices=None):
    """V110.3.21.10 orientation-corrected Hamner full body.

    Key change:
    - pelvis/sacrum, feet and hands no longer use unconstrained PCA orientation.
    - OpenSim native axes are mapped to anatomical SKEL frames with explicit handedness.
    - long bones remain segment-aligned with static reference scaling.
    """
    J=np.asarray(joints,np.float32)
    nm={str(n):i for i,n in enumerate(joint_names)}
    parts=[]

    P=_pelvic_frame(J,nm,frame)
    hr=nm.get("femur_r"); hl=nm.get("femur_l")
    hip_width=float(np.linalg.norm(J[0,hr]-J[0,hl])) if hr is not None and hl is not None else 0.20

    for body,p in bodies.items():
        V=np.asarray(p["V"],np.float32)
        F=np.asarray(p["F"],np.int32)
        kind=p["kind"]

        if kind=="pelvis":
            pel=nm.get("pelvis")
            if pel is None: continue
            # Center pelvis on bilateral hip midpoint to improve socket/head assembly.
            if hr is not None and hl is not None:
                hip_mid=0.5*(J[frame,hr]+J[frame,hl])
                hip_width=max(float(np.linalg.norm(J[0,hr]-J[0,hl])),1e-6)
                zspan=max(float(np.ptp(V[:,2])),1e-6)
                scale=np.clip((hip_width*0.92)/zspan,0.30,2.0)
                center=0.15*J[frame,pel]+0.85*hip_mid
            else:
                zspan=max(float(np.ptp(V[:,2])),1e-6)
                scale=1.0
                center=J[frame,pel]
            W=_native_body_to_target(V,P,center,scale)

        elif kind=="torso":
            th=nm.get("thorax"); lum=nm.get("lumbar_body"); head=nm.get("head")
            if th is None: continue
            lo=J[frame,lum] if lum is not None else J[frame,th]-P[:,1]*0.20
            tc=J[frame,th]
            hc=J[frame,head] if head is not None else tc+P[:,1]*0.25

            up=_safe_unit(tc-lo,P[:,1]); ml=P[:,2]
            ap=_safe_unit(np.cross(up,ml),P[:,0]); ml=_safe_unit(np.cross(ap,up),ml)
            T=np.stack([ap,up,ml],axis=1)

            head_mask=_find_head_component(V); trunk_mask=~head_mask
            if trunk_mask.sum()<50:
                trunk_mask=np.ones(len(V),dtype=bool); head_mask=np.zeros(len(V),dtype=bool)
            Vt=V[trunk_mask]; Vh=V[head_mask]

            native_ap=max(float(np.ptp(Vt[:,0])),1e-6)
            native_up=max(float(np.ptp(Vt[:,1])),1e-6)
            native_ml=max(float(np.ptp(Vt[:,2])),1e-6)
            target_up=max(float(np.linalg.norm(tc-lo))*1.10,0.18)

            if skin_vertices is not None:
                ext=_skin_local_extents(skin_vertices,tc,T,radius=max(target_up*0.75,0.22))
                target_ap=max(2*ext[0]*0.82,0.10)
                target_ml=max(2*ext[2]*0.88,0.12)
            else:
                target_ap=target_up*0.50; target_ml=target_up*0.65

            sc_t=np.array([
                np.clip(target_ap/native_ap,0.35,2.5),
                np.clip(target_up/native_up,0.35,2.5),
                np.clip(target_ml/native_ml,0.35,2.5)
            ])
            trunk_center=0.5*(lo+tc)
            Wt=_native_scaled_to_target(Vt,T,trunk_center,sc_t)

            if len(Vh):
                nh_ap=max(float(np.ptp(Vh[:,0])),1e-6)
                nh_up=max(float(np.ptp(Vh[:,1])),1e-6)
                nh_ml=max(float(np.ptp(Vh[:,2])),1e-6)
                if skin_vertices is not None:
                    hext=_skin_local_extents(skin_vertices,hc,T,radius=0.20,min_points=50)
                    targ_h_ap=max(2*hext[0]*0.82,0.10)
                    targ_h_up=max(2*hext[1]*0.88,0.12)
                    targ_h_ml=max(2*hext[2]*0.82,0.10)
                else:
                    neck_head=max(float(np.linalg.norm(hc-tc)),0.18)
                    targ_h_up=np.clip(neck_head*0.85,0.14,0.28)
                    targ_h_ap=targ_h_up*0.75; targ_h_ml=targ_h_up*0.72

                sc_h=np.array([
                    np.clip(targ_h_ap/nh_ap,0.30,2.2),
                    np.clip(targ_h_up/nh_up,0.30,2.2),
                    np.clip(targ_h_ml/nh_ml,0.30,2.2)
                ])
                Wh=_native_scaled_to_target(Vh,T,hc,sc_h)
                W=np.vstack([Wt,Wh]).astype(np.float32)

                idx_t=np.flatnonzero(trunk_mask); idx_h=np.flatnonzero(head_mask)
                map_t=np.full(len(V),-1,dtype=np.int32); map_t[idx_t]=np.arange(len(idx_t),dtype=np.int32)
                map_h=np.full(len(V),-1,dtype=np.int32); map_h[idx_h]=np.arange(len(idx_h),dtype=np.int32)+len(idx_t)
                outF=[]
                for tri in F:
                    tri=np.asarray(tri,dtype=np.int32)
                    if np.all(trunk_mask[tri]): outF.append(map_t[tri])
                    elif np.all(head_mask[tri]): outF.append(map_h[tri])
                if outF: F=np.asarray(outF,dtype=np.int32)
            else:
                W=Wt

        elif kind=="long":
            ai=nm.get(p["a"]); bi=nm.get(p["b"])
            if ai is None or bi is None: continue
            W=_register_long_bone_v32110(V,J[frame,ai],J[frame,bi],P)

        elif kind=="foot":
            side="r" if body.endswith("_r") else "l"
            cal=nm.get(f"calcn_{side}"); toes=nm.get(f"toes_{side}")
            if cal is None or toes is None: continue
            T=_foot_frame(J,nm,frame,side)
            if T is None: continue
            target_len=max(float(np.linalg.norm(J[0,toes]-J[0,cal])),0.08)
            # OpenSim foot native AP dimension is X.
            xspan=max(float(np.ptp(V[:,0])),1e-6)
            scale=np.clip(target_len/xspan,0.35,3.0)
            center=0.5*(J[frame,cal]+J[frame,toes])
            # For toes body, bias center distally.
            if p.get("distal"):
                center=J[frame,toes]
            W=_native_body_to_target(V,T,center,scale)

        elif kind=="block":
            side="r" if body.endswith("_r") else "l"
            tal=nm.get(f"talus_{side}"); cal=nm.get(f"calcn_{side}")
            if tal is None or cal is None: continue
            T=_foot_frame(J,nm,frame,side)
            if T is None: continue
            # small rigid ankle block, static scale from local size
            span=max(float(np.ptp(V[:,0])),float(np.ptp(V[:,1])),float(np.ptp(V[:,2])),1e-6)
            target=max(float(np.linalg.norm(J[0,tal]-J[0,cal]))*1.6,0.045)
            scale=np.clip(target/span,0.35,3.0)
            center=0.5*(J[frame,tal]+J[frame,cal])
            W=_native_body_to_target(V,T,center,scale)

        elif kind=="hand":
            side="r" if body.endswith("_r") else "l"
            rad=nm.get(f"radius_{side}"); ulna=nm.get(f"ulna_{side}")
            hand=nm.get(f"hand_{side}"); hum=nm.get(f"humerus_{side}")
            wrist_idx=rad if rad is not None else ulna
            if wrist_idx is None or hand is None: continue
            T=_hand_frame(J,nm,frame,side)
            if T is None: continue
            if hum is not None and ulna is not None:
                forearm=max(float(np.linalg.norm(J[0,hand]-J[0,ulna])),0.12)
            else:
                forearm=max(float(np.linalg.norm(J[0,hand]-J[0,wrist_idx])),0.12)
            W=_hand_fit_v32110(V,J[frame,wrist_idx],J[frame,hand],T,forearm)

        else:
            continue

        parts.append({
            "id":body,"body":body,
            "V":np.asarray(W,np.float32),
            "F":F
        })
    return parts

def render_hamner_walking_mp4(
    bodies, joints, joint_names, skin_vertices_seq=None,
    fps=25, max_faces_per_body=550, dpi=110
):
    """Render the already-calculated 75 SKEL frames as an actual walking MP4.

    Important:
    - does NOT rerun SKEL fitting;
    - does NOT regenerate q(t);
    - reuses existing joints/skin sequence;
    - constructs each Hamner body for each existing frame and writes frames sequentially,
      avoiding a giant browser payload.
    """
    import tempfile, os
    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import imageio.v2 as imageio

    J=np.asarray(joints,dtype=np.float32)
    T=int(J.shape[0])
    skin_seq=np.asarray(skin_vertices_seq,dtype=np.float32) if skin_vertices_seq is not None else None

    # Global plotting bounds from joints + optional skin.
    if skin_seq is not None and skin_seq.ndim==3 and skin_seq.shape[0]==T:
        pts=skin_seq.reshape(-1,3)
    else:
        pts=J.reshape(-1,3)
    lo=np.nanpercentile(pts,1,axis=0)
    hi=np.nanpercentile(pts,99,axis=0)
    cen=0.5*(lo+hi)
    span=max(float(np.max(hi-lo)),1e-3)
    half=0.58*span
    xlim=(cen[0]-half,cen[0]+half)
    ylim=(cen[1]-half,cen[1]+half)
    zlim=(cen[2]-half,cen[2]+half)

    tmpdir=Path(tempfile.mkdtemp(prefix="hamner_walk_"))
    out=tmpdir/"hamner_walk.mp4"

    with imageio.get_writer(str(out),fps=int(fps),codec="libx264",quality=7,macro_block_size=None) as writer:
        for t in range(T):
            skin_t=skin_seq[t] if skin_seq is not None and skin_seq.ndim==3 and t < len(skin_seq) else None
            parts=build_hamner_fullbody_frame(
                bodies,J,joint_names,frame=t,skin_vertices=skin_t
            )

            fig=plt.figure(figsize=(6.6,7.2),dpi=dpi)
            ax=fig.add_subplot(111,projection="3d")
            for p in parts:
                V=np.asarray(p["V"],dtype=np.float32)
                F=np.asarray(p["F"],dtype=np.int32)
                if len(F)==0 or len(V)==0:
                    continue
                step=max(1,int(np.ceil(len(F)/float(max_faces_per_body))))
                Fs=F[::step,:3]
                tris=V[Fs]
                pc=Poly3DCollection(tris,alpha=0.96,linewidths=0.05)
                pc.set_edgecolor("none")
                ax.add_collection3d(pc)

            ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)
            ax.set_box_aspect((1,1,1))
            ax.view_init(elev=8,azim=-78)
            ax.set_title(f"OpenSim Hamner · marcha · frame {t+1}/{T}")
            ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
            fig.tight_layout()

            fig.canvas.draw()
            rgba=np.asarray(fig.canvas.buffer_rgba())
            writer.append_data(rgba[:,:,:3])
            plt.close(fig)

    return str(out)

