from __future__ import annotations
import io, json, zipfile, re, sys, subprocess, importlib, tempfile, os, hashlib
from pathlib import Path
import numpy as np
from anatomical_atlas import load_atlas_registered, load_atlas_transform_plan, apply_atlas_transform, load_bodyparts3d_rigid_skeleton, ensure_bodyparts3d_atlas, bodyparts3d_atlas_status
from opensim_bridge import prepare_rajagopal, rajagopal_status, load_rajagopal_lower_body, build_rajagopal_skel_frame
from hamner_bridge import prepare_hamner, hamner_status, load_hamner_bodies, build_hamner_fullbody_frame, render_hamner_walking_mp4
from skel_internal_skeleton import build_internal_skeleton, joint_centres, segment_lengths
from unified_anatomical_frame import canonicalize_sequence, anthropometric_summary
from hamner_muscles import muscle_segments
from integrated_viewer import build_hamner_sequence, build_hamner_sequence_filtered, merge_mesh_sequences, muscle_paths_sequence, render_combined_mp4, render_layer_mp4, camera_preset, camera_from_angles, render_layer_mp4_manual_camera, render_layer_mp4_multiview
from anatomical_registration_ai import register_hamner_frame_optimized, registration_quality
from visible_human_atlas import prepare_visible_human_zip, build_visible_human_frame, atlas_counts, ATTRIBUTION
from denver_muscle_rig_v150 import build_all76_rig
import pandas as pd
import streamlit as st

# Landmarks mínimos que ya están presentes en la secuencia V104/V107 y son útiles
# para el ajuste articular inicial de SKEL.
JOINTS = [
    "LHip","RHip","LKnee","RKnee","LAnkle","RAnkle",
    "LHeel","RHeel","LBigToe","RBigToe","LSmallToe","RSmallToe",
    "LShoulder","RShoulder","LElbow","RElbow","LWrist","RWrist",
    "Neck","Head","Hip"
]

# V110.3.20.8 SEGMENT · GENTLE q14 STABILIZATION + SKEL SKELETON LAYER
# Base funcional V110.3.19.7 + solver inferior V110.3.19.9.
# Se eliminan deliberadamente persistencia/auto-recovery de secuencia/malla y migración NPZ.
# El único estado temporal de marcha vive en st.session_state durante la sesión activa.

# Alias tolerantes para no depender de una única convención de nombres.
_ALIASES = {
    "lhip": "LHip", "lefthip": "LHip", "left_hip": "LHip", "hipleft": "LHip",
    "rhip": "RHip", "righthip": "RHip", "right_hip": "RHip", "hipright": "RHip",
    "lknee": "LKnee", "leftknee": "LKnee", "left_knee": "LKnee", "kneeleft": "LKnee",
    "rknee": "RKnee", "rightknee": "RKnee", "right_knee": "RKnee", "kneeright": "RKnee",
    "lankle": "LAnkle", "leftankle": "LAnkle", "left_ankle": "LAnkle", "ankleleft": "LAnkle",
    "rankle": "RAnkle", "rightankle": "RAnkle", "right_ankle": "RAnkle", "ankleright": "RAnkle",
    "lheel": "LHeel", "leftheel": "LHeel", "left_heel": "LHeel",
    "rheel": "RHeel", "rightheel": "RHeel", "right_heel": "RHeel",
    "lbigtoe": "LBigToe", "leftbigtoe": "LBigToe", "left_big_toe": "LBigToe",
    "rbigtoe": "RBigToe", "rightbigtoe": "RBigToe", "right_big_toe": "RBigToe",
    "lsmalltoe": "LSmallToe", "leftsmalltoe": "LSmallToe", "left_small_toe": "LSmallToe",
    "rsmalltoe": "RSmallToe", "rightsmalltoe": "RSmallToe", "right_small_toe": "RSmallToe",
    "lshoulder": "LShoulder", "leftshoulder": "LShoulder", "left_shoulder": "LShoulder",
    "rshoulder": "RShoulder", "rightshoulder": "RShoulder", "right_shoulder": "RShoulder",
    "lelbow": "LElbow", "leftelbow": "LElbow", "left_elbow": "LElbow",
    "relbow": "RElbow", "rightelbow": "RElbow", "right_elbow": "RElbow",
    "lwrist": "LWrist", "leftwrist": "LWrist", "left_wrist": "LWrist",
    "rwrist": "RWrist", "rightwrist": "RWrist", "right_wrist": "RWrist",
    "neck": "Neck", "head": "Head", "nose": "Nose", "hip": "Hip",
}

def _norm_name(name: str) -> str:
    s = str(name).strip()
    compact = re.sub(r"[^a-z0-9]", "", s.lower())
    # Primero coincidencia exacta canónica.
    for canon in JOINTS + ["Nose"]:
        if compact == re.sub(r"[^a-z0-9]", "", canon.lower()):
            return canon
    # Después alias con y sin separadores.
    raw = s.lower().replace("-", "_").replace(" ", "_")
    return _ALIASES.get(raw, _ALIASES.get(compact, s))

def _xyz(v):
    try:
        if isinstance(v, dict):
            # Aceptar X/Y/Z y x/y/z.
            keys = {str(k).lower(): k for k in v.keys()}
            if all(k in keys for k in ("x","y","z")):
                a = [v[keys["x"]], v[keys["y"]], v[keys["z"]]]
            else:
                return None
        elif isinstance(v, (list, tuple, np.ndarray, pd.Series)) and len(v) >= 3:
            a = [v[0], v[1], v[2]]
        else:
            return None
        a = [float(x) for x in a]
        return a if np.isfinite(a).all() else None
    except Exception:
        return None

def _frame_points_from_frame(f):
    if not isinstance(f, dict):
        return {}
    # V104/V107 oficial usa `joints`; se mantienen los otros nombres como compatibilidad.
    src = f.get("joints")
    if not isinstance(src, dict) or not src:
        src = f.get("points")
    if not isinstance(src, dict) or not src:
        src = f.get("landmarks")
    if not isinstance(src, dict):
        return {}
    out = {}
    for k, v in src.items():
        p = _xyz(v)
        if p is not None:
            out[_norm_name(k)] = p

    # Centros derivados sólo para visualización/registro inicial. No sustituyen landmarks medidos.
    if "Hip" not in out and "LHip" in out and "RHip" in out:
        out["Hip"] = ((np.asarray(out["LHip"]) + np.asarray(out["RHip"])) / 2.0).tolist()
    if "Neck" not in out and "LShoulder" in out and "RShoulder" in out:
        out["Neck"] = ((np.asarray(out["LShoulder"]) + np.asarray(out["RShoulder"])) / 2.0).tolist()
    if "Head" not in out and "Nose" in out:
        out["Head"] = list(out["Nose"])
    return out

def _select_best_frame(motion):
    frames = list((motion or {}).get("frames") or [])
    best_i, best_pts, best_score = 0, {}, -1
    for i, f in enumerate(frames):
        pts = _frame_points_from_frame(f)
        score = sum(1 for j in JOINTS if j in pts)
        if score > best_score:
            best_i, best_pts, best_score = i, pts, score
        # 14+ ya es un frame excelente para este PoC; evitamos recorrer de más.
        if score >= 14:
            break
    return best_i, best_pts, max(0, best_score)

def _target_csv(points, frame_index=0):
    rows = []
    for j in JOINTS:
        if j in points:
            x, y, z = points[j]
            rows.append({"frame": int(frame_index)+1, "joint": j, "x": x, "y": y, "z": z,
                         "source": "derived" if j in ("Hip",) else "V104/V107"})
    return pd.DataFrame(rows)

def _inspect_private_bundle(data: bytes):
    info={"has_male":False,"has_female":False,"files":[],"valid_zip":False}
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            names=z.namelist(); info["valid_zip"]=True
            info["files"]=names[:80]
            low=[n.lower() for n in names]
            info["has_male"]=any(n.endswith("skel_male.pkl") for n in low)
            info["has_female"]=any(n.endswith("skel_female.pkl") for n in low)
    except Exception as e:
        info["error"]=str(e)
    return info

def _load_private_skel_bundle_auto():
    """V110.3.18 · Robust Backblaze B2 private loader + persistent runtime cache.

    Orden de resolución:
      1) ZIP válido ya presente en /tmp/physiosentinel_skel_b2_cache_v1_1 (sin red).
      2) Backblaze S3-compatible API usando el s3ApiUrl devuelto por authorize_account.
      3) Backblaze Native private download-by-name como fallback de red.
      4) Carga manual desde navegador (gestionada por el panel).

    Una caché válida nunca se elimina sólo porque B2 devuelva 401/403.
    """
    import hashlib
    from urllib.parse import quote
    keys=['B2_KEY_ID','B2_APPLICATION_KEY','B2_BUCKET','B2_FILE']
    try:
        cfg={k:str(st.secrets.get(k,'')).strip() for k in keys}
    except Exception:
        cfg={k:'' for k in keys}
    missing=[k for k,v in cfg.items() if not v]

    root=Path(tempfile.gettempdir())/'physiosentinel_skel_b2_cache_v1_1'
    root.mkdir(parents=True,exist_ok=True)
    expected=''
    try: expected=str(st.secrets.get('SKEL_ZIP_SHA256','')).strip().lower()
    except Exception: expected=''

    def sha256_file(path):
        h=hashlib.sha256()
        with open(path,'rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
        return h.hexdigest().lower()

    def validate_zip(path):
        path=Path(path)
        if not path.exists() or path.stat().st_size < 1024:
            return False,'archivo ausente o demasiado pequeño'
        if expected and sha256_file(path) != expected:
            return False,'SHA-256 distinto del esperado'
        try:
            with zipfile.ZipFile(path) as z:
                bad=z.testzip()
                if bad: return False,f'CRC ZIP inválido en {bad}'
                names=[n.lower() for n in z.namelist()]
                if not any(n.endswith('skel_male.pkl') for n in names): return False,'falta skel_male.pkl'
                if not any(n.endswith('skel_female.pkl') for n in names): return False,'falta skel_female.pkl'
            return True,None
        except Exception as exc:
            return False,f'ZIP inválido: {exc}'

    # 1) CACHE HIT: primero el nombre exacto de B2_FILE, después cualquier bundle SKEL válido.
    candidates=[]
    if cfg.get('B2_FILE'):
        candidates.append(root/(Path(cfg['B2_FILE']).name or 'skel_models_v1.1.zip'))
    candidates += [x for x in sorted(root.glob('*.zip')) if x not in candidates]
    for cached in candidates:
        ok,_=validate_zip(cached)
        if ok:
            return cached.read_bytes(),f"CACHE HIT · {cached.name} · /tmp persistente de la instancia",None

    if missing:
        return None,None,'Backblaze B2 no configurado: '+', '.join(missing)

    obj_name=Path(cfg['B2_FILE']).name or 'skel_models_v1.1.zip'
    local=root/obj_name
    meta=root/(obj_name+'.meta.json')
    diagnostics=[]

    try:
        import requests
        auth=requests.get('https://api.backblazeb2.com/b2api/v4/b2_authorize_account',
                          auth=(cfg['B2_KEY_ID'],cfg['B2_APPLICATION_KEY']),timeout=30)
        auth.raise_for_status()
        aj=auth.json()
        storage=((aj.get('apiInfo') or {}).get('storageApi') or {})
        token=aj.get('authorizationToken') or ''
        download_url=storage.get('downloadUrl') or ''
        s3_url=storage.get('s3ApiUrl') or ''
        allowed=storage.get('allowed') or {}
        caps=set(allowed.get('capabilities') or [])
        if 'readFiles' not in caps:
            raise PermissionError('La Application Key B2 no incluye la capacidad readFiles.')
        allowed_buckets=[b.get('name') for b in (allowed.get('buckets') or []) if isinstance(b,dict)]
        if allowed_buckets and cfg['B2_BUCKET'] not in allowed_buckets:
            raise PermissionError(f"La clave B2 no está autorizada para el bucket {cfg['B2_BUCKET']}.")
        diagnostics.append('B2 AUTH OK')

        # 2) S3-compatible: evita problemas de download-by-name/token vistos en V110.3.13.
        if s3_url:
            try:
                import boto3
                from urllib.parse import urlparse
                _host=(urlparse(s3_url).hostname or '')
                _parts=_host.split('.')
                _region=_parts[1] if len(_parts)>2 and _parts[0]=='s3' else 'us-east-1'
                cli=boto3.client('s3',endpoint_url=s3_url,
                    aws_access_key_id=cfg['B2_KEY_ID'],aws_secret_access_key=cfg['B2_APPLICATION_KEY'],
                    region_name=_region)
                tmp=local.with_suffix(local.suffix+'.part')
                tmp.unlink(missing_ok=True)
                # V110.3.18: NO usar download_fileobj. boto3 TransferManager ejecuta
                # HeadObject antes de GET y ciertas Application Keys B2 read-only devuelven
                # 403 en HEAD aunque GetObject esté autorizado. Se fuerza un GET firmado
                # directo y se transmite Body a disco.
                obj=cli.get_object(Bucket=cfg['B2_BUCKET'],Key=cfg['B2_FILE'])
                body=obj.get('Body')
                if body is None:
                    raise IOError('S3 GetObject no devolvió Body.')
                with open(tmp,'wb') as f:
                    while True:
                        chunk=body.read(1024*1024)
                        if not chunk: break
                        f.write(chunk)
                try: body.close()
                except Exception: pass
                ok,why=validate_zip(tmp)
                if not ok:
                    tmp.unlink(missing_ok=True)
                    raise IOError('S3 descargó el objeto pero no superó validación: '+str(why))
                tmp.replace(local)
                meta.write_text(json.dumps({'provider':'Backblaze B2 S3','bucket':cfg['B2_BUCKET'],
                    'object':cfg['B2_FILE'],'bytes':local.stat().st_size,'sha256':sha256_file(local)},
                    ensure_ascii=False,indent=2),encoding='utf-8')
                return local.read_bytes(),f"B2 AUTH OK · B2 DOWNLOAD OK (S3) · {cfg['B2_BUCKET']}/{cfg['B2_FILE']} · CACHE MISS → guardado",None
            except Exception as exc:
                diagnostics.append(f'S3 fallback: {type(exc).__name__}: {exc}')

        # 3) Native private download-by-name.
        if not download_url or not token:
            raise RuntimeError('b2_authorize_account no devolvió downloadUrl/token.')
        bq=quote(cfg['B2_BUCKET'],safe='')
        fq=quote(cfg['B2_FILE'],safe='/')
        url=f"{download_url}/file/{bq}/{fq}"
        tmp=local.with_suffix(local.suffix+'.part')
        tmp.unlink(missing_ok=True)
        with requests.get(url,headers={'Authorization':token},stream=True,timeout=(20,180)) as r:
            r.raise_for_status()
            remote_sha1=(r.headers.get('X-Bz-Content-Sha1') or '').strip().lower()
            h1=hashlib.sha1()
            with open(tmp,'wb') as f:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk); h1.update(chunk)
        if remote_sha1 and remote_sha1 not in ('none','do_not_verify') and len(remote_sha1)==40:
            if h1.hexdigest().lower()!=remote_sha1:
                tmp.unlink(missing_ok=True)
                raise IOError('La verificación SHA-1 de Backblaze B2 no coincide con el archivo descargado.')
        ok,why=validate_zip(tmp)
        if not ok:
            tmp.unlink(missing_ok=True)
            raise IOError('Native API descargó el objeto pero no superó validación: '+str(why))
        tmp.replace(local)
        meta.write_text(json.dumps({'provider':'Backblaze B2 Native','bucket':cfg['B2_BUCKET'],
            'object':cfg['B2_FILE'],'bytes':local.stat().st_size,'sha256':sha256_file(local),
            'remote_sha1':remote_sha1},ensure_ascii=False,indent=2),encoding='utf-8')
        return local.read_bytes(),f"B2 AUTH OK · B2 DOWNLOAD OK (Native) · {cfg['B2_BUCKET']}/{cfg['B2_FILE']} · CACHE MISS → guardado",None
    except Exception as exc:
        extra=(' · '+' | '.join(diagnostics)) if diagnostics else ''
        return None,None,f"Backblaze B2 Auto-Loader: {type(exc).__name__}: {exc}{extra}"


def _persist_manual_bundle_runtime_cache(raw: bytes, preferred_name: str=''):
    """V110.3.18: una carga manual válida alimenta la misma caché estable.
    Así un 401/403 posterior no obliga a seleccionar el ZIP en cada rerun.
    """
    try:
        name=Path(preferred_name).name if preferred_name else 'skel_models_v1.1_manual.zip'
        if not name.lower().endswith('.zip'): name += '.zip'
        root=Path(tempfile.gettempdir())/'physiosentinel_skel_b2_cache_v1_1'
        root.mkdir(parents=True,exist_ok=True)
        dst=root/name
        tmp=dst.with_suffix(dst.suffix+'.part')
        tmp.write_bytes(raw)
        info=_inspect_private_bundle(raw)
        if not (info.get('valid_zip') and info.get('has_male') and info.get('has_female')):
            tmp.unlink(missing_ok=True); return None
        tmp.replace(dst)
        return dst
    except Exception:
        return None


def _plot_frame(df):
    if df.empty:
        return
    try:
        import plotly.graph_objects as go
        links = [
            ("LShoulder","RShoulder"),("LShoulder","LElbow"),("LElbow","LWrist"),
            ("RShoulder","RElbow"),("RElbow","RWrist"),("LShoulder","LHip"),
            ("RShoulder","RHip"),("LHip","RHip"),("LHip","LKnee"),("LKnee","LAnkle"),
            ("RHip","RKnee"),("RKnee","RAnkle"),("Neck","LShoulder"),("Neck","RShoulder"),("Neck","Head")
        ]
        P={r.joint:(r.x,r.y,r.z) for r in df.itertuples()}
        fig=go.Figure()
        for a,b in links:
            if a in P and b in P:
                xa,ya,za=P[a]; xb,yb,zb=P[b]
                fig.add_trace(go.Scatter3d(x=[xa,xb],y=[ya,yb],z=[za,zb],mode="lines",showlegend=False,hoverinfo="skip"))
        fig.add_trace(go.Scatter3d(x=df.x,y=df.y,z=df.z,mode="markers+text",text=df.joint,
                                   textposition="top center",name="Landmarks objetivo"))
        fig.update_layout(height=520,margin=dict(l=0,r=0,t=35,b=0),title="Frame objetivo V110.1.3 · XYZ para ajuste SKEL",
                          scene=dict(aspectmode="data"))
        st.plotly_chart(fig,use_container_width=True)
    except Exception as exc:
        st.caption(f"Visualización 3D no disponible: {exc}")



# V110.3.13 · OFFICIAL SKEL24 JOINT MAP
# `SKEL.forward(...).joints` devuelve 24 joints en el orden cinemático interno de kin_skel.py.
# IMPORTANTE: el PKL expone `joints_name` con 22 nombres de otra capa semántica; no debe usarse
# para indexar directamente el tensor forward().joints de 24 elementos.
_SKEL24_FORWARD_JOINT_NAMES_140 = [
    "pelvis",
    "femur_r","tibia_r","talus_r","calcn_r","toes_r",
    "femur_l","tibia_l","talus_l","calcn_l","toes_l",
    "lumbar_body","thorax","head",
    "scapula_r","humerus_r","ulna_r","radius_r","hand_r",
    "scapula_l","humerus_l","ulna_l","radius_l","hand_l",
]

def _skel_forward_joint_names_140(model=None):
    return list(_SKEL24_FORWARD_JOINT_NAMES_140)

_SKEL24_TARGET_INDEX = {
    "Hip":0,
    "RHip":1,"RKnee":2,"RAnkle":3,
    "LHip":6,"LKnee":7,"LAnkle":8,
    "RShoulder":15,"RElbow":16,"RWrist":18,
    "LShoulder":20,"LElbow":21,"LWrist":23,
    "Neck":12,"Head":13,
}

# V110.1.8 · correspondencia anatómica entre landmarks V104/V107 y joints SKEL.
# Se resuelve por nombre real del modelo (`model.joints_name`), evitando índices rígidos.
_SKEL_TARGET_CANDIDATES = {
    "Hip": ["pelvis"],
    "RHip": ["right_hip", "femur_r", "hip_r", "rhip"],
    "RKnee": ["right_knee", "tibia_r", "knee_r", "rknee"],
    "RAnkle": ["right_ankle", "talus_r", "ankle_r", "rankle"],
    "LHip": ["left_hip", "femur_l", "hip_l", "lhip"],
    "LKnee": ["left_knee", "tibia_l", "knee_l", "lknee"],
    "LAnkle": ["left_ankle", "talus_l", "ankle_l", "lankle"],
    "RShoulder": ["right_shoulder", "humerus_r", "shoulder_r", "rshoulder"],
    "RElbow": ["right_elbow", "ulna_r", "elbow_r", "relbow"],
    "RWrist": ["right_wrist", "hand_r", "wrist_r", "rwrist"],
    "LShoulder": ["left_shoulder", "humerus_l", "shoulder_l", "lshoulder"],
    "LElbow": ["left_elbow", "ulna_l", "elbow_l", "lelbow"],
    "LWrist": ["left_wrist", "hand_l", "wrist_l", "lwrist"],
    "Neck": ["neck", "cervical", "c7"],
    "Head": ["head", "skull"],
}

def _simple_name(v):
    if isinstance(v, bytes):
        try: v=v.decode("utf-8")
        except Exception: v=str(v)
    return re.sub(r"[^a-z0-9]", "", str(v).lower())

def _resolve_skel_correspondence(model, target_df):
    """V110.3.13: correspondencia rígida contra los 24 joints reales de `forward().joints`.

    Nunca usa `model.joints_name` para indexar el tensor de 24 joints.
    """
    names=_skel_forward_joint_names_140(model)
    available=set(target_df["joint"].astype(str))
    rows=[]
    for target_name,jidx in _SKEL24_TARGET_INDEX.items():
        if target_name in available and 0 <= int(jidx) < len(names):
            rows.append((target_name,int(jidx),str(names[int(jidx)])))
    return rows,names

def _rodrigues_torch(torch, r):
    # Rotación 3D diferenciable desde vector axis-angle.
    theta=torch.sqrt(torch.sum(r*r)+1e-12)
    k=r/theta
    K=torch.stack([
        torch.stack([torch.zeros_like(k[0]),-k[2],k[1]]),
        torch.stack([k[2],torch.zeros_like(k[0]),-k[0]]),
        torch.stack([-k[1],k[0],torch.zeros_like(k[0])])
    ])
    I=torch.eye(3,dtype=r.dtype,device=r.device)
    return I + torch.sin(theta)*K + (1.0-torch.cos(theta))*(K@K)

def _similarity_to_target(torch, src, tgt, rotvec):
    # src/tgt: Nx3. Rotación libre + escala isotrópica + traslación analítica.
    R=_rodrigues_torch(torch,rotvec)
    src_mean=src.mean(dim=0,keepdim=True)
    tgt_mean=tgt.mean(dim=0,keepdim=True)
    src_c=src-src_mean
    tgt_c=tgt-tgt_mean
    src_r=src_c @ R.T
    src_rms=torch.sqrt(torch.mean(torch.sum(src_r*src_r,dim=1))+1e-12)
    tgt_rms=torch.sqrt(torch.mean(torch.sum(tgt_c*tgt_c,dim=1))+1e-12)
    scale=tgt_rms/src_rms
    pred=src_r*scale+tgt_mean
    trans=tgt_mean.squeeze(0)-scale*(src_mean.squeeze(0) @ R.T)
    return pred,R,scale,trans

def _plot_skel_fit(target_df, fit_rows, before_xyz, after_xyz, all_after=None, all_names=None):
    try:
        import plotly.graph_objects as go
        target_map={r.joint:np.array([r.x,r.y,r.z],float) for r in target_df.itertuples()}
        names=[r[0] for r in fit_rows]
        T=np.stack([target_map[n] for n in names])
        fig=go.Figure()
        fig.add_trace(go.Scatter3d(x=T[:,0],y=T[:,1],z=T[:,2],mode="markers+text",text=names,textposition="top center",name="XYZ objetivo"))
        fig.add_trace(go.Scatter3d(x=before_xyz[:,0],y=before_xyz[:,1],z=before_xyz[:,2],mode="markers",name="SKEL antes"))
        fig.add_trace(go.Scatter3d(x=after_xyz[:,0],y=after_xyz[:,1],z=after_xyz[:,2],mode="markers+text",text=names,textposition="bottom center",name="SKEL ajustado"))
        for i,n in enumerate(names):
            fig.add_trace(go.Scatter3d(x=[T[i,0],after_xyz[i,0]],y=[T[i,1],after_xyz[i,1]],z=[T[i,2],after_xyz[i,2]],mode="lines",showlegend=False,hoverinfo="skip"))
        fig.update_layout(height=620,margin=dict(l=0,r=0,t=45,b=0),title="V110.3.13 · Frame semilla · XYZ objetivo vs SKEL",scene=dict(aspectmode="data"))
        st.plotly_chart(fig,use_container_width=True)
    except Exception as exc:
        st.caption(f"Visualización del fit no disponible: {exc}")

# V110.3.13 · COORDINATE FRAME REGISTRATION + BONE-VECTOR RETARGETING
# El mapeo deja de depender de nombres biomecánicos supuestos. Se calibra el Jacobiano
# articular real q -> joints alrededor de neutro y se seleccionan DOF observables por rango.
_SKEL_Q_NAMES = [f"q{i:02d}" for i in range(46)]

# Parámetros de seguridad genéricos. La semántica concreta de cada q NO se presupone.
_EMPIRICAL_Q_ABS_LIMIT = 1.20
_EMPIRICAL_MAX_DOF = 18
_EMPIRICAL_REG = 1.5e-2


def _joint_index_map(model):
    names=_skel_forward_joint_names_140(model)
    return {_simple_name(n):i for i,n in enumerate(names)}


def _target_df_for_frame(frame, frame_index):
    pts=_frame_points_from_frame(frame)
    df=_target_csv(pts,frame_index)
    try:
        df.attrs['depth_weight']=float(frame.get('depth_weight',1.0))
        df.attrs['source_method']=str(frame.get('source_method',''))
    except Exception:
        pass
    return df


def _transform_joints_fixed(torch, joints, rotvec, scale, trans):
    R=_rodrigues_torch(torch,rotvec); return (joints @ R.T)*scale+trans,R


def _empirical_select_dofs(model, rows, jacobian, max_dofs=_EMPIRICAL_MAX_DOF):
    """V110.3.13: selección empírica por cadena anatómica, usada como espacio reducido antes del registro de marcos y el ajuste por vectores óseos.

    No presupone la semántica de q. Usa el Jacobiano medido para exigir que cada q
    seleccionado tenga efecto preferente en una cadena observable (tronco, pierna D/I,
    brazo D/I) y penaliza el movimiento fuera de esa cadena. Dentro de cada cadena se
    conserva independencia lineal mediante Gram-Schmidt.
    """
    J=np.asarray(jacobian,float)
    row_names=[str(r[0]) for r in rows]
    obs_idx=[int(r[1]) for r in rows]
    # Cadenas definidas sólo sobre landmarks realmente observados.
    chains=[
        ('axial', {'Hip','Neck','Head','RShoulder','LShoulder'}, 4),
        ('leg_R', {'Hip','RHip','RKnee','RAnkle'}, 4),
        ('leg_L', {'Hip','LHip','LKnee','LAnkle'}, 4),
        ('arm_R', {'Neck','RShoulder','RElbow','RWrist'}, 3),
        ('arm_L', {'Neck','LShoulder','LElbow','LWrist'}, 3),
    ]
    selected=[]; scores=[]; meta=[]
    all_obs=J[obs_idx,:,:]
    for cname, wanted, quota in chains:
        local_pos=[k for k,n in enumerate(row_names) if n in wanted]
        if not local_pos: continue
        local=all_obs[local_pos,:,:].reshape(len(local_pos)*3,J.shape[-1])
        other_pos=[k for k in range(len(row_names)) if k not in local_pos]
        other=all_obs[other_pos,:,:].reshape(len(other_pos)*3,J.shape[-1]) if other_pos else np.zeros((0,J.shape[-1]))
        Q=[]
        for _ in range(int(quota)):
            best=None; best_score=-1.; best_v=None; best_local=0.; best_off=0.
            for qi in range(J.shape[-1]):
                if qi<3 or qi in selected: continue
                v=local[:,qi].astype(float).copy()
                raw=float(np.linalg.norm(v))
                if raw<1e-6: continue
                for qv in Q: v-=qv*np.dot(qv,v)
                indep=float(np.linalg.norm(v))
                off=float(np.linalg.norm(other[:,qi])) if other.size else 0.0
                # Favorece efecto local independiente y penaliza fuerte influencia remota.
                specificity=raw/(off+0.35*raw+1e-9)
                score=indep*specificity
                if score>best_score:
                    best,best_score,best_v,best_local,best_off=qi,score,v,raw,off
            if best is None or best_score<1e-5: break
            selected.append(int(best)); scores.append(float(best_score))
            Q.append(best_v/max(float(np.linalg.norm(best_v)),1e-12))
            meta.append({'q_index':int(best),'q_label':f'q{best:02d}','chain':cname,
                         'score':float(best_score),'local_effect':float(best_local),'off_chain_effect':float(best_off)})
            if len(selected)>=int(max_dofs): break
        if len(selected)>=int(max_dofs): break
    # Relleno sólo si alguna cadena no pudo completar su cuota.
    if len(selected)<int(max_dofs):
        A=all_obs.reshape(len(obs_idx)*3,J.shape[-1]); Q=[]
        for qi in selected:
            v=A[:,qi].copy()
            for qv in Q: v-=qv*np.dot(qv,v)
            nr=float(np.linalg.norm(v))
            if nr>1e-8: Q.append(v/nr)
        while len(selected)<int(max_dofs):
            best=None; br=-1.; bv=None
            for qi in range(J.shape[-1]):
                if qi<3 or qi in selected: continue
                v=A[:,qi].copy()
                for qv in Q: v-=qv*np.dot(qv,v)
                nr=float(np.linalg.norm(v))
                if nr>br: best,br,bv=qi,nr,v
            if best is None or br<1e-5: break
            selected.append(int(best)); scores.append(float(br)); Q.append(bv/max(br,1e-12))
            meta.append({'q_index':int(best),'q_label':f'q{best:02d}','chain':'fallback','score':float(br),'local_effect':float(br),'off_chain_effect':np.nan})
    # Se conserva meta para auditoría sin cambiar la firma histórica.
    _empirical_select_dofs.last_meta=meta
    return selected,scores

def _empirical_seed_pose(model, rows, target_df, jacobian, base_joints, rotvec, scale, trans, selected):
    """Semilla lineal desde el Jacobiano calibrado. Convierte el target a coordenadas SKEL
    usando la similitud global actual y resuelve mínimos cuadrados sólo en DOF observables.
    """
    if not selected: return np.zeros((int(model.num_q_params),),np.float32)
    tmap={r.joint:np.array([r.x,r.y,r.z],float) for r in target_df.itertuples()}
    obs=[int(r[1]) for r in rows]
    tgt=np.stack([tmap[r[0]] for r in rows])
    rv=np.asarray(rotvec,float); th=np.linalg.norm(rv)
    if th<1e-12: R=np.eye(3)
    else:
        k=rv/th; K=np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]],float)
        R=np.eye(3)+np.sin(th)*K+(1-np.cos(th))*(K@K)
    sc=max(float(scale),1e-8); tr=np.asarray(trans,float)
    tgt_model=((tgt-tr)/sc)@R
    base=np.asarray(base_joints,float)[obs]
    y=(tgt_model-base).reshape(-1)
    A=np.asarray(jacobian,float)[obs,:,:][:,:,selected].reshape(len(obs)*3,len(selected))
    lam=0.08
    ATA=A.T@A + lam*np.eye(len(selected)); ATy=A.T@y
    try: q=np.linalg.solve(ATA,ATy)
    except Exception: q=np.linalg.lstsq(A,y,rcond=1e-4)[0]
    q=np.clip(q,-0.75,0.75)
    pose=np.zeros((int(model.num_q_params),),np.float32); pose[selected]=q.astype(np.float32)
    return pose


def _apply_empirical_constraints(torch, pose, selected):
    sel=set(int(i) for i in selected)
    with torch.no_grad():
        inactive=[i for i in range(int(pose.shape[1])) if i not in sel]
        if inactive: pose[:,inactive]=0.0
        if sel:
            ids=sorted(sel); pose[:,ids].clamp_(-float(_EMPIRICAL_Q_ABS_LIMIT),float(_EMPIRICAL_Q_ABS_LIMIT))


def _mask_pose_grad_empirical(pose, selected):
    if pose.grad is None: return
    mask=np.zeros((pose.shape[1],),np.float32); mask[list(selected)]=1.0
    pose.grad.mul_(pose.grad.new_tensor(mask).reshape(1,-1))


def _target_tensor_for_rows(torch, target_df, rows, names_subset=None):
    tmap={r.joint:np.array([r.x,r.y,r.z],np.float32) for r in target_df.itertuples()}
    rr=[r for r in rows if names_subset is None or r[0] in names_subset]
    idx=torch.tensor([r[1] for r in rr],dtype=torch.long,device='cpu')
    tgt=torch.tensor(np.stack([tmap[r[0]] for r in rr]).astype(np.float32),dtype=torch.float32,device='cpu')
    return rr,idx,tgt,tmap


# V110.3.13 · IK LOCAL POR CADENAS
# Cada q se asigna a una única cadena según el Jacobiano empírico. La optimización se
# realiza proximal→distal y penaliza explícitamente cualquier desplazamiento fuera de
# la cadena. Así un q que mejora un codo pero desplaza rodilla/pelvis deja de ser útil.
_CHAIN_SPECS_128 = [
    ('axial', ['Hip','Neck','Head','LShoulder','RShoulder'], [('Hip','Neck'),('Neck','Head'),('LShoulder','RShoulder')], 4),
    ('leg_R', ['Hip','RHip','RKnee','RAnkle'], [('Hip','RHip'),('RHip','RKnee'),('RKnee','RAnkle')], 4),
    ('leg_L', ['Hip','LHip','LKnee','LAnkle'], [('Hip','LHip'),('LHip','LKnee'),('LKnee','LAnkle')], 4),
    ('arm_R', ['Neck','RShoulder','RElbow','RWrist'], [('Neck','RShoulder'),('RShoulder','RElbow'),('RElbow','RWrist')], 3),
    ('arm_L', ['Neck','LShoulder','LElbow','LWrist'], [('Neck','LShoulder'),('LShoulder','LElbow'),('LElbow','LWrist')], 3),
]


def _chain_assign_dofs(rows, jacobian, max_total=_EMPIRICAL_MAX_DOF):
    J=np.asarray(jacobian,float)
    row_names=[str(r[0]) for r in rows]
    obs_idx=[int(r[1]) for r in rows]
    all_obs=J[obs_idx,:,:]
    assigned=set(); out={}; audit=[]
    for cname, landmarks, bones, quota in _CHAIN_SPECS_128:
        loc=[k for k,n in enumerate(row_names) if n in set(landmarks)]
        oth=[k for k in range(len(row_names)) if k not in loc]
        if not loc:
            out[cname]=[]; continue
        A=all_obs[loc,:,:].reshape(len(loc)*3,J.shape[-1])
        B=all_obs[oth,:,:].reshape(len(oth)*3,J.shape[-1]) if oth else np.zeros((0,J.shape[-1]))
        chosen=[]; Q=[]
        for _ in range(int(quota)):
            best=None; best_score=-1.; best_raw=0.; best_off=0.; best_v=None
            for qi in range(J.shape[-1]):
                if qi<3 or qi in assigned: continue
                v=A[:,qi].astype(float).copy(); raw=float(np.linalg.norm(v))
                if raw<1e-7: continue
                for qv in Q: v-=qv*np.dot(qv,v)
                indep=float(np.linalg.norm(v))
                off=float(np.linalg.norm(B[:,qi])) if B.size else 0.0
                # Especificidad local estricta: el efecto remoto tiene más penalización que en V110.2.7.
                specificity=raw/(raw+1.75*off+1e-9)
                score=indep*specificity
                if score>best_score:
                    best,best_score,best_raw,best_off,best_v=qi,score,raw,off,v
            if best is None or best_score<1e-6: break
            chosen.append(int(best)); assigned.add(int(best))
            Q.append(best_v/max(float(np.linalg.norm(best_v)),1e-12))
            audit.append({'q_index':int(best),'q_label':f'q{best:02d}','chain':cname,
                          'score':float(best_score),'local_effect':float(best_raw),
                          'off_chain_effect':float(best_off),
                          'specificity':float(best_raw/(best_raw+best_off+1e-9))})
            if len(assigned)>=int(max_total): break
        out[cname]=chosen
        if len(assigned)>=int(max_total): break
    for cname, *_ in _CHAIN_SPECS_128: out.setdefault(cname,[])
    return out,audit


def _chain_linear_seed(model, rows, target_df, jacobian, base_joints, rotvec, scale, trans, chain_map):
    pose=np.zeros((int(model.num_q_params),),np.float32)
    tmap={r.joint:np.array([r.x,r.y,r.z],float) for r in target_df.itertuples()}
    rv=np.asarray(rotvec,float); th=float(np.linalg.norm(rv))
    if th<1e-12: R=np.eye(3)
    else:
        k=rv/th; K=np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]],float)
        R=np.eye(3)+np.sin(th)*K+(1-np.cos(th))*(K@K)
    sc=max(float(scale),1e-8); tr=np.asarray(trans,float)
    J=np.asarray(jacobian,float); base=np.asarray(base_joints,float)
    for cname, landmarks, _bones, _quota in _CHAIN_SPECS_128:
        qids=list(chain_map.get(cname,[]))
        rr=[r for r in rows if r[0] in set(landmarks)]
        if not qids or len(rr)<2: continue
        tgt=np.stack([tmap[r[0]] for r in rr])
        tgt_model=((tgt-tr)/sc)@R
        obs=[int(r[1]) for r in rr]
        # Resta la contribución lineal ya sembrada por cadenas proximales.
        current=base[obs] + J[obs,:,:]@pose
        y=(tgt_model-current).reshape(-1)
        A=J[obs,:,:][:,:,qids].reshape(len(obs)*3,len(qids))
        lam=0.16
        try: dq=np.linalg.solve(A.T@A+lam*np.eye(len(qids)),A.T@y)
        except Exception: dq=np.linalg.lstsq(A,y,rcond=1e-4)[0]
        pose[qids]=np.clip(pose[qids]+dq,-0.60,0.60).astype(np.float32)
    return pose


def _chain_bone_loss_torch(torch, model, world_joints, target_df, bone_pairs):
    names=_skel_forward_joint_names_140(model); m={_simple_name(n):i for i,n in enumerate(names)}
    # V110.3.13: auditoría ósea contra el orden REAL de forward().joints (SKEL24).
    corr={'Hip':'pelvis','RHip':'femur_r','RKnee':'tibia_r','RAnkle':'talus_r','LHip':'femur_l','LKnee':'tibia_l','LAnkle':'talus_l',
          'Neck':'thorax','Head':'head','RShoulder':'humerus_r','RElbow':'ulna_r','RWrist':'hand_r','LShoulder':'humerus_l','LElbow':'ulna_l','LWrist':'hand_l'}
    t={r.joint:np.array([r.x,r.y,r.z],np.float32) for r in target_df.itertuples()}
    vals=[]
    for a,b in bone_pairs:
        ia=m.get(corr.get(a,'')); ib=m.get(corr.get(b,''))
        if ia is None or ib is None or a not in t or b not in t: continue
        pv=world_joints[ib]-world_joints[ia]
        tv=torch.as_tensor(t[b]-t[a],dtype=world_joints.dtype,device=world_joints.device)
        pn=torch.linalg.norm(pv)+1e-8; tn=torch.linalg.norm(tv)+1e-8
        vals.append(1.0-torch.clamp(torch.dot(pv,tv)/(pn*tn),-1.0,1.0))
    return torch.stack(vals).mean() if vals else torch.zeros((),dtype=world_joints.dtype,device=world_joints.device)


def _refine_per_chain_ik(torch, model, pose, rotvec, trans, fixed_scale, target_df, rows, chain_map,
                         betas, zero_trans, iterations_per_chain=26, lr=0.010,
                         reference_pose=None, similarity_mode=False):
    # Rotación global sólo se optimiza en la fase axial. En cadenas periféricas se mantiene
    # fija para que un brazo/pierna no 'arregle' su error girando todo el cuerpo.
    tmap={r.joint:np.array([r.x,r.y,r.z],np.float32) for r in target_df.itertuples()}
    ref=reference_pose.detach().clone() if reference_pose is not None else pose.detach().clone()
    all_selected=sorted({q for vals in chain_map.values() for q in vals})
    _apply_empirical_constraints(torch,pose,all_selected)
    for cname, landmarks, bones, _quota in _CHAIN_SPECS_128:
        qids=list(chain_map.get(cname,[]))
        rr=[r for r in rows if r[0] in set(landmarks)]
        if len(rr)<2: continue
        idx=torch.tensor([r[1] for r in rr],dtype=torch.long,device='cpu')
        tgt=torch.tensor(np.stack([tmap[r[0]] for r in rr]),dtype=torch.float32,device='cpu')
        outside=[r for r in rows if r[0] not in set(landmarks)]
        out_idx=torch.tensor([r[1] for r in outside],dtype=torch.long,device='cpu') if outside else None
        with torch.no_grad():
            Jbase=model(pose,betas,zero_trans,skelmesh=False).joints[0]
            if similarity_mode:
                _pred0,Rbase,sbase,tbase=_similarity_to_target(torch,Jbase.index_select(0,idx),tgt,rotvec)
                world_base=(Jbase@Rbase.T)*sbase+tbase
            else:
                world_base,_=_transform_joints_fixed(torch,Jbase,rotvec,fixed_scale,trans)
            outside_anchor=world_base.index_select(0,out_idx).detach().clone() if out_idx is not None and len(outside) else None
        pose.requires_grad_(True)
        optimize_rot=(cname=='axial')
        if optimize_rot: rotvec.requires_grad_(True)
        if (not similarity_mode) and cname=='axial': trans.requires_grad_(True)
        params=[pose] + ([rotvec] if optimize_rot else []) + ([trans] if ((not similarity_mode) and cname=='axial') else [])
        opt=torch.optim.Adam(params,lr=float(lr))
        for _ in range(int(iterations_per_chain)):
            opt.zero_grad(set_to_none=True)
            Jm=model(pose,betas,zero_trans,skelmesh=False).joints[0]
            if similarity_mode:
                pred,R,sc,tr2=_similarity_to_target(torch,Jm.index_select(0,idx),tgt,rotvec)
                world=(Jm@R.T)*sc+tr2
            else:
                pred,_=_transform_joints_fixed(torch,Jm.index_select(0,idx),rotvec,fixed_scale,trans)
                world,_=_transform_joints_fixed(torch,Jm,rotvec,fixed_scale,trans)
            pos=torch.mean(torch.sum((pred-tgt)**2,dim=1))
            bone=_chain_bone_loss_torch(torch,model,world,target_df,bones)
            if outside_anchor is not None:
                remote=torch.mean(torch.sum((world.index_select(0,out_idx)-outside_anchor)**2,dim=1))
            else: remote=torch.zeros((),dtype=pose.dtype)
            reg=8e-3*torch.mean((pose[:,qids]-ref[:,qids])**2) if qids else torch.zeros((),dtype=pose.dtype)
            neutral=4e-3*torch.mean(pose[:,qids]**2) if qids else torch.zeros((),dtype=pose.dtype)
            # Bone direction domina sobre XYZ dentro de la cadena; remote evita contaminación intercadena.
            loss=pos + 0.62*bone + 0.48*remote + reg + neutral
            loss.backward()
            # Sólo puede moverse el q perteneciente a esta cadena.
            _mask_pose_grad_empirical(pose,qids)
            torch.nn.utils.clip_grad_norm_(params,2.5)
            opt.step(); _apply_empirical_constraints(torch,pose,all_selected)
            with torch.no_grad(): rotvec.clamp_(-3.14159,3.14159)
        pose.requires_grad_(False); rotvec.requires_grad_(False); trans.requires_grad_(False)
    # V110.3.13 · reconciliación anatómica global corta. Corrige residuos entre cadenas
    # sin permitir que brazos/piernas compensen con grandes torsiones.
    ids=list(_DIRECT_ACTIVE_130)
    pose.requires_grad_(True)
    opt=torch.optim.Adam([pose],lr=0.0035 if temporal else 0.0045)
    for _ in range(18 if temporal else 28):
        opt.zero_grad(set_to_none=True)
        world,_,_=_direct_world_130(torch,model,pose,betas,zero_trans,rotvec,scale,trans)
        dl=[]; bl=[]
        for g in ('tronco','cabeza','pierna_D','pierna_I','brazo_D','brazo_I'):
            d,b=_direct_group_loss_130(torch,model,world,target_df,rows,g); dl.append(d); bl.append(b)
        loss=torch.mean(torch.stack(dl))+0.72*torch.mean(torch.stack(bl))
        lam=0.13 if temporal else 0.09
        loss=loss+lam*torch.mean((pose[:,ids]-ref[:,ids])**2)
        loss.backward()
        if pose.grad is not None:
            mask=torch.zeros_like(pose.grad); mask[:,ids]=1.0; pose.grad.mul_(mask)
        torch.nn.utils.clip_grad_norm_([pose],1.2)
        opt.step(); _clamp_direct_pose_130(torch,pose)
    pose.requires_grad_(False)
    return pose,rotvec,trans



# --- V110.3.13 · registro explícito de marco corporal y pérdidas por vectores óseos ---
_BONE_VECTOR_PAIRS = [
    ('Hip','RHip'),('RHip','RKnee'),('RKnee','RAnkle'),
    ('Hip','LHip'),('LHip','LKnee'),('LKnee','LAnkle'),
    ('Hip','Neck'),('Neck','Head'),
    ('Neck','RShoulder'),('RShoulder','RElbow'),('RElbow','RWrist'),
    ('Neck','LShoulder'),('LShoulder','LElbow'),('LElbow','LWrist'),
    ('LHip','RHip'),('LShoulder','RShoulder'),
]


def _safe_unit_np(v, eps=1e-9):
    v=np.asarray(v,float); n=float(np.linalg.norm(v))
    return v/max(n,eps)


def _body_frame_from_target_np(target_df):
    t={r.joint:np.array([r.x,r.y,r.z],float) for r in target_df.itertuples()}
    if not {'Hip','LHip','RHip','Neck'}.issubset(t): return None
    lateral=_safe_unit_np(t['RHip']-t['LHip'])
    up=_safe_unit_np(t['Neck']-t['Hip'])
    # Gram-Schmidt para evitar que lateral/up no sean exactamente ortogonales.
    lateral=_safe_unit_np(lateral-up*np.dot(lateral,up))
    forward=_safe_unit_np(np.cross(lateral,up))
    lateral=_safe_unit_np(np.cross(up,forward))
    return np.column_stack([lateral,up,forward])


def _body_frame_from_model_np(model, joints_np):
    names=_skel_forward_joint_names_140(model)
    m={_simple_name(n):i for i,n in enumerate(names)}; J=np.asarray(joints_np,float)
    need=['pelvis','lefthip','righthip','neck']
    if not all(k in m for k in need): return None
    lateral=_safe_unit_np(J[m['righthip']]-J[m['lefthip']])
    up=_safe_unit_np(J[m['neck']]-J[m['pelvis']])
    lateral=_safe_unit_np(lateral-up*np.dot(lateral,up))
    forward=_safe_unit_np(np.cross(lateral,up))
    lateral=_safe_unit_np(np.cross(up,forward))
    return np.column_stack([lateral,up,forward])


def _rotmat_to_rotvec_np(R):
    R=np.asarray(R,float)
    c=float(np.clip((np.trace(R)-1.0)/2.0,-1.0,1.0)); th=float(np.arccos(c))
    if th<1e-8: return np.zeros(3,np.float32)
    if abs(np.pi-th)<1e-4:
        A=(R+np.eye(3))/2.0
        axis=np.sqrt(np.maximum(np.diag(A),0.0))
        if R[2,1]-R[1,2]<0: axis[0]*=-1
        if R[0,2]-R[2,0]<0: axis[1]*=-1
        if R[1,0]-R[0,1]<0: axis[2]*=-1
        axis=_safe_unit_np(axis)
    else:
        axis=np.array([R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]],float)/(2*np.sin(th))
        axis=_safe_unit_np(axis)
    return (axis*th).astype(np.float32)


def _registered_rotvec_seed(model, joints_np, target_df):
    Bm=_body_frame_from_model_np(model,joints_np); Bt=_body_frame_from_target_np(target_df)
    if Bm is None or Bt is None: return np.zeros(3,np.float32)
    R=Bt@Bm.T
    if np.linalg.det(R)<0:
        Bt=Bt.copy(); Bt[:,2]*=-1; R=Bt@Bm.T
    return _rotmat_to_rotvec_np(R)


def _median_bone_scale(model, joints_np, target_df):
    names=_skel_forward_joint_names_140(model); m={_simple_name(n):i for i,n in enumerate(names)}
    t={r.joint:np.array([r.x,r.y,r.z],float) for r in target_df.itertuples()}
    # mapa target -> nombre SKEL normalizado
    corr={'Hip':'pelvis','RHip':'righthip','RKnee':'rightknee','RAnkle':'rightankle',
          'LHip':'lefthip','LKnee':'leftknee','LAnkle':'leftankle','Neck':'neck','Head':'head',
          'RShoulder':'rightshoulder','RElbow':'rightelbow','RWrist':'rightwrist',
          'LShoulder':'leftshoulder','LElbow':'leftelbow','LWrist':'leftwrist'}
    ratios=[]; J=np.asarray(joints_np,float)
    for a,b in _BONE_VECTOR_PAIRS:
        if a not in t or b not in t: continue
        ia=m.get(corr.get(a,'')); ib=m.get(corr.get(b,''))
        if ia is None or ib is None: continue
        lm=np.linalg.norm(J[ib]-J[ia]); lt=np.linalg.norm(t[b]-t[a])
        if lm>1e-6 and lt>1e-6: ratios.append(float(lt/lm))
    return float(np.median(ratios)) if ratios else 1.0


def _bone_vector_losses_torch(torch, model, world_joints, target_df):
    names=_skel_forward_joint_names_140(model); m={_simple_name(n):i for i,n in enumerate(names)}
    t={r.joint:np.array([r.x,r.y,r.z],np.float32) for r in target_df.itertuples()}
    corr={'Hip':'pelvis','RHip':'righthip','RKnee':'rightknee','RAnkle':'rightankle',
          'LHip':'lefthip','LKnee':'leftknee','LAnkle':'leftankle','Neck':'neck','Head':'head',
          'RShoulder':'rightshoulder','RElbow':'rightelbow','RWrist':'rightwrist',
          'LShoulder':'leftshoulder','LElbow':'leftelbow','LWrist':'leftwrist'}
    dir_terms=[]; len_terms=[]
    for a,b in _BONE_VECTOR_PAIRS:
        if a not in t or b not in t: continue
        ia=m.get(corr.get(a,'')); ib=m.get(corr.get(b,''))
        if ia is None or ib is None: continue
        pv=world_joints[ib]-world_joints[ia]
        tv=torch.as_tensor(t[b]-t[a],dtype=world_joints.dtype,device=world_joints.device)
        pn=torch.linalg.norm(pv)+1e-8; tn=torch.linalg.norm(tv)+1e-8
        cos=torch.clamp(torch.dot(pv,tv)/(pn*tn),-1.0,1.0)
        dir_terms.append(1.0-cos)
        # Longitud relativa, con peso muy pequeño: evita colapsos sin forzar antropometría exacta.
        len_terms.append(((pn-tn)/(tn+1e-6))**2)
    z=torch.zeros((),dtype=world_joints.dtype,device=world_joints.device)
    d=torch.stack(dir_terms).mean() if dir_terms else z
    l=torch.stack(len_terms).mean() if len_terms else z
    return d,l


def _frame_axis_loss_torch(torch, model, world_joints, target_df):
    names=_skel_forward_joint_names_140(model); m={_simple_name(n):i for i,n in enumerate(names)}
    t={r.joint:np.array([r.x,r.y,r.z],np.float32) for r in target_df.itertuples()}
    if not {'Hip','LHip','RHip','Neck'}.issubset(t): return torch.zeros((),dtype=world_joints.dtype)
    req=['pelvis','lefthip','righthip','neck']
    if not all(k in m for k in req): return torch.zeros((),dtype=world_joints.dtype)
    def cosloss(a,b):
        an=torch.linalg.norm(a)+1e-8; bn=torch.linalg.norm(b)+1e-8
        return 1.0-torch.clamp(torch.dot(a,b)/(an*bn),-1.0,1.0)
    lat_m=world_joints[m['righthip']]-world_joints[m['lefthip']]
    up_m=world_joints[m['neck']]-world_joints[m['pelvis']]
    lat_t=torch.as_tensor(t['RHip']-t['LHip'],dtype=world_joints.dtype,device=world_joints.device)
    up_t=torch.as_tensor(t['Neck']-t['Hip'],dtype=world_joints.dtype,device=world_joints.device)
    return 0.5*(cosloss(lat_m,lat_t)+cosloss(up_m,up_t))


def _bone_angle_audit(model, world_joints_np, target_df):
    names=_skel_forward_joint_names_140(model); m={_simple_name(n):i for i,n in enumerate(names)}
    t={r.joint:np.array([r.x,r.y,r.z],float) for r in target_df.itertuples()}
    # V110.3.13: auditoría ósea contra el orden REAL de forward().joints (SKEL24).
    corr={'Hip':'pelvis','RHip':'femur_r','RKnee':'tibia_r','RAnkle':'talus_r','LHip':'femur_l','LKnee':'tibia_l','LAnkle':'talus_l',
          'Neck':'thorax','Head':'head','RShoulder':'humerus_r','RElbow':'ulna_r','RWrist':'hand_r','LShoulder':'humerus_l','LElbow':'ulna_l','LWrist':'hand_l'}
    J=np.asarray(world_joints_np,float); ang=[]
    for a,b in _BONE_VECTOR_PAIRS:
        if a not in t or b not in t: continue
        ia=m.get(corr.get(a,'')); ib=m.get(corr.get(b,''))
        if ia is None or ib is None: continue
        u=_safe_unit_np(J[ib]-J[ia]); v=_safe_unit_np(t[b]-t[a]); ang.append(float(np.degrees(np.arccos(np.clip(np.dot(u,v),-1,1)))))
    return {'bone_angle_mean_deg':float(np.mean(ang)) if ang else np.nan,
            'bone_angle_max_deg':float(np.max(ang)) if ang else np.nan,
            'bone_vector_count':len(ang)}

def _empirical_refine(torch,model,pose,rotvec,trans,fixed_scale,target_df,rows,selected,betas,zero_trans,
                      iterations=45,lr=0.012,reference_pose=None,similarity_mode=False):
    rr,idx,tgt,_=_target_tensor_for_rows(torch,target_df,rows,None)
    pose.requires_grad_(True); rotvec.requires_grad_(True)
    params=[pose,rotvec]
    if not similarity_mode: trans.requires_grad_(True); params.append(trans)
    opt=torch.optim.Adam(params,lr=float(lr))
    ref=(reference_pose.detach().clone() if reference_pose is not None else torch.zeros_like(pose))
    ids=list(selected)
    for _ in range(int(iterations)):
        opt.zero_grad(set_to_none=True)
        J=model(pose,betas,zero_trans,skelmesh=False).joints[0]
        selj=J.index_select(0,idx)
        if similarity_mode:
            pred,R,sc,tr2=_similarity_to_target(torch,selj,tgt,rotvec)
            world=(J @ R.T)*sc+tr2
        else:
            pred,R=_transform_joints_fixed(torch,selj,rotvec,fixed_scale,trans)
            world=(J @ R.T)*fixed_scale+trans
        data=torch.mean(torch.sum((pred-tgt)**2,dim=1))
        bone_dir,bone_len=_bone_vector_losses_torch(torch,model,world,target_df)
        frame_loss=_frame_axis_loss_torch(torch,model,world,target_df)
        reg=_EMPIRICAL_REG*torch.mean((pose[:,ids]-ref[:,ids])**2) if ids else torch.zeros((),dtype=pose.dtype)
        # V110.3.13: la dirección de los segmentos y el marco corporal pesan de forma explícita.
        neutral=5e-3*torch.mean(pose[:,ids]**2) if ids else torch.zeros((),dtype=pose.dtype)
        loss=data + 0.42*bone_dir + 0.06*bone_len + 0.28*frame_loss + reg + neutral
        loss.backward(); _mask_pose_grad_empirical(pose,ids); torch.nn.utils.clip_grad_norm_(params,3.0)
        opt.step(); _apply_empirical_constraints(torch,pose,ids)
        with torch.no_grad(): rotvec.clamp_(-3.14159,3.14159)
    pose.requires_grad_(False); rotvec.requires_grad_(False); trans.requires_grad_(False)


def _pose_safety_audit_empirical(model,joints_np,pose_np,selected):
    names=_skel_forward_joint_names_140(model); m={_simple_name(n):i for i,n in enumerate(names)}
    J=np.asarray(joints_np,float); p=np.asarray(pose_np,float); sel=set(int(i) for i in selected)
    inactive=np.array([p[i] for i in range(min(len(p),46)) if i not in sel],float)
    leak=float(np.nanmax(np.abs(inactive))) if inactive.size else 0.0
    hits=[f'q{i:02d}' for i in sel if abs(abs(float(p[i]))-_EMPIRICAL_Q_ABS_LIMIT)<1e-3]
    flags=[]
    if leak>1e-5: flags.append(f'DOF no seleccionado fuera de neutro={leak:.3g}')
    if hits: flags.append('límite empírico: '+', '.join(hits[:6]))
    def ang(a,b,c):
        try: return float(np.degrees(_angle_np(J[m[_simple_name(a)]],J[m[_simple_name(b)]],J[m[_simple_name(c)]])))
        except Exception: return np.nan
    vals={'rodilla I':ang('left_hip','left_knee','left_ankle'),'rodilla D':ang('right_hip','right_knee','right_ankle'),
          'codo I':ang('left_shoulder','left_elbow','left_wrist'),'codo D':ang('right_shoulder','right_elbow','right_wrist')}
    # sólo alertas de geometría extrema; ya no se asume signo de q
    for lab,a in vals.items():
        if np.isfinite(a) and (a<10 or a>181): flags.append(f'{lab}={a:.0f}°')
    return {'ok':not flags,'flags':flags,'inactive_q_leak':leak,'active_dof_count':len(sel),
            'selected_q':[f'q{i:02d}' for i in sorted(sel)],**vals}



# V110.3.13 · SKEL FORWARD-KINEMATICS GROUND TRUTH
# Semántica q oficial de skel/kin_skel.py. Ya no se descubre qué hace cada q por ranking global.
_SKEL_Q_NAMES_130 = [
 'pelvis_tilt','pelvis_list','pelvis_rotation',
 'hip_flexion_r','hip_adduction_r','hip_rotation_r','knee_angle_r','ankle_angle_r','subtalar_angle_r','mtp_angle_r',
 'hip_flexion_l','hip_adduction_l','hip_rotation_l','knee_angle_l','ankle_angle_l','subtalar_angle_l','mtp_angle_l',
 'lumbar_bending','lumbar_extension','lumbar_twist','thorax_bending','thorax_extension','thorax_twist',
 'head_bending','head_extension','head_twist',
 'scapula_abduction_r','scapula_elevation_r','scapula_upward_rot_r','shoulder_r_x','shoulder_r_y','shoulder_r_z',
 'elbow_flexion_r','pro_sup_r','wrist_flexion_r','wrist_deviation_r',
 'scapula_abduction_l','scapula_elevation_l','scapula_upward_rot_l','shoulder_l_x','shoulder_l_y','shoulder_l_z',
 'elbow_flexion_l','pro_sup_l','wrist_flexion_l','wrist_deviation_l']

_DIRECT_GROUPS_130 = {
 'pelvis_global': [], # orientación rígida global: rotvec, no q0-q2 para evitar doble representación
 'tronco': [17,18,19,20,21,22],
 'cabeza': [23,24,25],
 'pierna_D': [3,4,5,6,7],
 'pierna_I': [10,11,12,13,14],
 'brazo_D': [26,27,28,29,30,31,32],
 'brazo_I': [36,37,38,39,40,41,42],
}
_DIRECT_ACTIVE_130=sorted({i for g in _DIRECT_GROUPS_130.values() for i in g})

# Límites conservadores: se prioriza una pose humana estable frente a bajar RMSE mediante torsión.
_DIRECT_LIMITS_130={
 3:(-1.35,1.35),4:(-0.85,0.85),5:(-0.85,0.85),6:(0.0,2.20),7:(-0.90,0.90),
 10:(-1.35,1.35),11:(-0.85,0.85),12:(-0.85,0.85),13:(0.0,2.20),14:(-0.90,0.90),
 17:(-0.55,0.55),18:(-0.65,0.65),19:(-0.55,0.55),20:(-0.55,0.55),21:(-0.65,0.65),22:(-0.55,0.55),
 23:(-0.55,0.55),24:(-0.55,0.55),25:(-0.55,0.55),
 26:(-0.65,0.65),27:(-0.40,0.40),28:(-0.35,0.35),29:(-1.20,1.20),30:(-1.00,1.00),31:(-1.20,1.20),32:(0.0,2.20),
 36:(-0.65,0.65),37:(-0.40,0.40),38:(-0.35,0.35),39:(-1.20,1.20),40:(-1.00,1.00),41:(-1.20,1.20),42:(0.0,2.20),
}

_DIRECT_BONES_BY_GROUP_130={
 'tronco':[('Hip','Neck'),('LHip','LShoulder'),('RHip','RShoulder')],
 'cabeza':[('Neck','Head')],
 'pierna_D':[('RHip','RKnee'),('RKnee','RAnkle')],
 'pierna_I':[('LHip','LKnee'),('LKnee','LAnkle')],
 'brazo_D':[('Neck','RShoulder'),('RShoulder','RElbow'),('RElbow','RWrist')],
 'brazo_I':[('Neck','LShoulder'),('LShoulder','LElbow'),('LElbow','LWrist')],
}
_DIRECT_LM_BY_GROUP_130={
 'tronco':['Hip','Neck','LShoulder','RShoulder'], 'cabeza':['Neck','Head'],
 'pierna_D':['RHip','RKnee','RAnkle'],'pierna_I':['LHip','LKnee','LAnkle'],
 'brazo_D':['RShoulder','RElbow','RWrist'],'brazo_I':['LShoulder','LElbow','LWrist']}

def _clamp_direct_pose_130(torch,pose):
    with torch.no_grad():
        pose[:,0:3]=0.0
        for i in range(int(pose.shape[1])):
            if i not in _DIRECT_ACTIVE_130: pose[:,i]=0.0
        for i,(lo,hi) in _DIRECT_LIMITS_130.items():
            if i<pose.shape[1]: pose[:,i].clamp_(float(lo),float(hi))
    return pose

def _direct_model_metadata_130(model):
    return {
      'q_names':list(_SKEL_Q_NAMES_130),
      'q_groups':{k:[_SKEL_Q_NAMES_130[i] for i in v] for k,v in _DIRECT_GROUPS_130.items()},
      'active_q_indices':list(_DIRECT_ACTIVE_130),
      'uses_private_model_metadata':['joints_name','per_joint_rot','parameter_mapping','osim_kintree_table','pose_params_name'],
      'strategy':'official q semantics + depth-safe local-chain fit + constrained global reconciliation; no global empirical q discovery',
      'private_pkl_audit':getattr(model,'_physiosentinel_private_meta',{})
    }

def _direct_world_130(torch,model,pose,betas,zero_trans,rotvec,scale,trans,axis_map=None):
    J=model(pose,betas,zero_trans,skelmesh=False).joints[0]
    P=_axis_map_tensor_134(torch,axis_map,J.dtype,J.device)
    Jm=J@P.T
    R=_rodrigues_torch(torch,rotvec)
    return (Jm@R.T)*scale+trans, J, R

def _direct_group_loss_130(torch,model,world,target_df,rows,group):
    # Posiciones sólo de la cadena actual, más direcciones óseas. Esto evita compensaciones remotas.
    tmap={r.joint:torch.tensor([r.x,r.y,r.z],dtype=world.dtype,device=world.device) for r in target_df.itertuples()}
    names=_skel_forward_joint_names_140(model); nm={_simple_name(n):i for i,n in enumerate(names)}
    # V110.3.13: auditoría ósea contra el orden REAL de forward().joints (SKEL24).
    corr={'Hip':'pelvis','RHip':'femur_r','RKnee':'tibia_r','RAnkle':'talus_r','LHip':'femur_l','LKnee':'tibia_l','LAnkle':'talus_l',
          'Neck':'thorax','Head':'head','RShoulder':'humerus_r','RElbow':'ulna_r','RWrist':'hand_r','LShoulder':'humerus_l','LElbow':'ulna_l','LWrist':'hand_l'}
    pos=[]
    # V110.3.13 · profundidad segura: en reconstrucción monocular la Z es inferida,
    # por lo que nunca debe tener el mismo peso que X/Y al retorcer una cadena.
    depth_w=float(target_df.attrs.get('depth_weight',1.0)) if hasattr(target_df,'attrs') else 1.0
    wxyz=torch.tensor([1.0,1.0,max(0.05,min(1.0,depth_w))],dtype=world.dtype,device=world.device)
    for lm in _DIRECT_LM_BY_GROUP_130.get(group,[]):
        ji=nm.get(corr.get(lm,''))
        if ji is not None and lm in tmap:
            d=(world[ji]-tmap[lm])*wxyz
            pos.append(torch.sum(d*d))
    data=torch.mean(torch.stack(pos)) if pos else torch.zeros((),dtype=world.dtype,device=world.device)
    def cosloss(a,b):
        a=a/(torch.linalg.norm(a)+1e-8); b=b/(torch.linalg.norm(b)+1e-8)
        return 1.0-torch.clamp(torch.sum(a*b),-1.0,1.0)
    dirs=[]
    for a,b in _DIRECT_BONES_BY_GROUP_130.get(group,[]):
        ia=nm.get(corr.get(a,'')); ib=nm.get(corr.get(b,''))
        if ia is not None and ib is not None and a in tmap and b in tmap:
            dirs.append(cosloss(world[ib]-world[ia],tmap[b]-tmap[a]))
    bdir=torch.mean(torch.stack(dirs)) if dirs else torch.zeros((),dtype=world.dtype,device=world.device)
    return data,bdir

def _direct_refine_130(torch,model,pose,rotvec,trans,scale,target_df,rows,betas,zero_trans,
                       iterations=90,lr=0.010,reference_pose=None,temporal=False,active_override=None,axis_map=None,freeze_global=False):
    # Fase 1: tronco/cabeza; Fase 2: piernas; Fase 3: brazos. Cada fase usa DOF semánticos fijos.
    stages=[['tronco','cabeza'],['pierna_D','pierna_I'],['brazo_D','brazo_I']]
    ref=reference_pose.detach().clone() if reference_pose is not None else torch.zeros_like(pose)
    for stage in stages:
        ids=sorted({i for g in stage for i in _DIRECT_GROUPS_130[g]})
        if active_override is not None:
            ids=[i for i in ids if i in set(active_override)]
        pose.requires_grad_(True)
        # rot/trans sólo se permiten en etapa axial inicial; brazos/piernas nunca recolocan todo el cuerpo.
        axial=('tronco' in stage)
        if axial and not freeze_global: rotvec.requires_grad_(True); trans.requires_grad_(True); params=[pose,rotvec,trans]
        else: params=[pose]
        opt=torch.optim.Adam(params,lr=float(lr))
        for _ in range(max(8,int(iterations)//len(stages))):
            opt.zero_grad(set_to_none=True)
            world,_,_=_direct_world_130(torch,model,pose,betas,zero_trans,rotvec,scale,trans,axis_map=axis_map)
            data=[]; bd=[]
            for g in stage:
                d,b=_direct_group_loss_130(torch,model,world,target_df,rows,g); data.append(d); bd.append(b)
            loss=torch.mean(torch.stack(data))+0.55*torch.mean(torch.stack(bd))
            # regularización fuerte a neutro/pose previa para impedir torsiones invisibles en landmarks.
            if ids:
                lam=0.075 if temporal else 0.045
                loss=loss+lam*torch.mean((pose[:,ids]-ref[:,ids])**2)
            if axial and not freeze_global:
                # Mantener pelvis global cerca de la semilla de registro explícito.
                loss=loss+0.008*torch.mean(rotvec**2)
            loss.backward()
            if pose.grad is not None:
                mask=torch.zeros_like(pose.grad); mask[:,ids]=1.0; pose.grad.mul_(mask)
            torch.nn.utils.clip_grad_norm_(params,2.5)
            opt.step(); _clamp_direct_pose_130(torch,pose)
            with torch.no_grad(): rotvec.clamp_(-3.14159,3.14159)
        pose.requires_grad_(False); rotvec.requires_grad_(False); trans.requires_grad_(False)
    return pose,rotvec,trans

def _acquisition_mode_gate_142(rmse_xy, rmse_z, audit, acquisition_mode, *, sanity_ok=True, unified_ok=True, xy_limit=0.38):
    """V110.3.19.3 · Puerta consistente por modalidad + avisos blandos de límites articulares.

    Restaura la lógica Depth-Safe validada en V110.3.11/V110.3.12 y evita que un
    único DoF no crítico (p. ej. cabeza/hombro/muñeca) en límite bloquee un frame
    con buen ajuste XY. En 3D calibrado se conserva la auditoría estricta.
    """
    mode=str(acquisition_mode or 'monocular_depth_safe').strip().lower()
    if mode not in {'monocular_depth_safe','biplanar_estimated','3d_strict'}:
        mode='monocular_depth_safe'
    ax=dict(audit or {})
    flags=[str(x) for x in (ax.get('flags',[]) or [])]
    angular_prefixes=('error angular medio=', 'error angular máximo=')
    angular_flags=[f for f in flags if f.startswith(angular_prefixes)]

    # Los límites se clasifican por semántica SKEL. En adquisición no métrica,
    # un único límite periférico/cabeza es aviso; cadera/rodilla siguen siendo duros.
    joint_limit_flags=[f for f in flags if f.endswith(' en límite')]
    def _joint_name(flag):
        return flag[:-len(' en límite')].strip() if flag.endswith(' en límite') else ''
    critical_prefixes=('hip_','knee_')
    critical_limit_flags=[f for f in joint_limit_flags if _joint_name(f).startswith(critical_prefixes)]
    soft_limit_flags=[f for f in joint_limit_flags if f not in critical_limit_flags]
    other_flags=[f for f in flags if f not in angular_flags and f not in joint_limit_flags]

    finite_xy=bool(np.isfinite(rmse_xy))
    finite_z=bool(np.isfinite(rmse_z))
    xy_ok=bool(finite_xy and float(rmse_xy) < float(xy_limit))
    strict=(mode=='3d_strict')
    depth_safe=(mode in {'monocular_depth_safe','biplanar_estimated'})

    warnings=[]
    if strict:
        z_ok=bool(finite_z and float(rmse_z) < float(xy_limit))
        hard_flags=list(flags)
        audit_ok=bool(ax.get('ok',False))
    else:
        z_ok=True
        # Un solo límite no crítico no bloquea. Tres o más límites simultáneos sí
        # se consideran señal de saturación general del ajuste.
        multi_limit_hard = len(joint_limit_flags) >= 3
        hard_flags=list(other_flags) + list(critical_limit_flags)
        if multi_limit_hard:
            hard_flags += [f for f in soft_limit_flags if f not in hard_flags]
        else:
            warnings += soft_limit_flags
        # Los errores angulares globales quedan como advertencia en Depth-Safe,
        # coherente con la puerta validada cuando Z no es métrica.
        warnings += angular_flags
        audit_ok=bool(len(hard_flags)==0)

    finite_ok=bool(finite_xy and (finite_z or depth_safe))
    ok=bool(finite_ok and xy_ok and z_ok and audit_ok and sanity_ok and unified_ok)
    reasons=[]
    if not finite_xy: reasons.append('RMSE XY inválido')
    if strict and not finite_z: reasons.append('RMSE Z inválido')
    if not xy_ok and finite_xy: reasons.append(f'RMSE XY={float(rmse_xy):.4f} ≥ {float(xy_limit):.2f}')
    if strict and not z_ok and finite_z: reasons.append(f'RMSE Z={float(rmse_z):.4f} ≥ {float(xy_limit):.2f}')
    for f in hard_flags:
        if f not in reasons: reasons.append(f)
    if not sanity_ok: reasons.append('SKEL24 sanity no superado')
    if not unified_ok: reasons.append('Unified Coordinate Frame no superado')
    labels={
        'monocular_depth_safe':'Depth-Safe monocular',
        'biplanar_estimated':'Biplanar 3D estimado',
        '3d_strict':'3D estricta calibrada',
    }
    return {
        'ok':ok,'mode':mode,'mode_label':labels[mode],
        'monocular':mode=='monocular_depth_safe','biplanar_estimated':mode=='biplanar_estimated','strict_3d':strict,
        'rmse_xy':float(rmse_xy),'rmse_z':float(rmse_z) if np.isfinite(rmse_z) else float('nan'),
        'xy_limit':float(xy_limit),'xy_ok':xy_ok,'z_ok':z_ok,'audit_ok':audit_ok,
        'hard_flags':hard_flags,'soft_warnings':warnings,
        'joint_limit_flags':joint_limit_flags,'critical_limit_flags':critical_limit_flags,
        'ignored_depth_flags':angular_flags if depth_safe else [],
        'sanity_ok':bool(sanity_ok),'unified_ok':bool(unified_ok),'reasons':reasons,
        'rule':(
            'Monocular: XY + límites críticos + coherencia; Z y avisos periféricos informativos' if mode=='monocular_depth_safe' else
            'Biplanar estimado: XY + límites críticos + coherencia; Z y avisos periféricos informativos' if mode=='biplanar_estimated' else
            '3D calibrado: XY + Z + auditoría ósea estricta'
        )
    }


def _direct_anatomical_audit_130(model,world_np,pose_np,target_df):
    ba=_bone_angle_audit(model,world_np,target_df)
    p=np.asarray(pose_np,float); flags=[]
    neutral_lower_ok={6,13,32,42}
    for i,(lo,hi) in _DIRECT_LIMITS_130.items():
        if i>=len(p):
            continue
        at_lo=abs(p[i]-lo)<1e-3
        at_hi=abs(p[i]-hi)<1e-3
        if at_lo and i in neutral_lower_ok and abs(float(p[i]))<1e-3:
            continue
        if at_lo or at_hi:
            flags.append(f'{_SKEL_Q_NAMES_130[i]} en límite')
    meanang=ba.get('bone_angle_mean_deg',np.nan); maxang=ba.get('bone_angle_max_deg',np.nan)
    if np.isfinite(meanang) and meanang>20: flags.append(f'error angular medio={meanang:.1f}°')
    if np.isfinite(maxang) and maxang>45: flags.append(f'error angular máximo={maxang:.1f}°')
    return {'ok':not flags,'flags':flags,**ba,
            'q_semantic_values':{_SKEL_Q_NAMES_130[i]:float(p[i]) for i in _DIRECT_ACTIVE_130 if i<len(p)}}


# V110.3.13 · SKELAnatomicalAutomata
# CMA-ES reducido para rescatar el Frame 1 monocular antes del refinamiento por gradiente.
# Se exploran sólo DOF observables/seguros en frontal; las rotaciones axiales y escápulas
# permanecen cerca de neutro para evitar compensar una Z inferida con torsiones anatómicas.
_AUTOMATA_SAFE_Q_133=[3,4,6,10,11,13,17,18,20,21,29,30,32,39,40,42]
_AUTOMATA_MAX_GEN_133=18
_AUTOMATA_POP_133=18
_AUTOMATA_TOPK_133=5

def _np_rodrigues_133(r):
    r=np.asarray(r,float).reshape(3); th=float(np.linalg.norm(r))
    if th<1e-12: return np.eye(3,dtype=float)
    k=r/th
    K=np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]],float)
    return np.eye(3)+np.sin(th)*K+(1.0-np.cos(th))*(K@K)


# V110.3.13 · calibración discreta del sistema de coordenadas SKEL ↔ V104/V107.
# Se prueban las 48 matrices de permutación/signo (incluidas reflexiones) y,
# para cada una, una similitud propia (rotación det=+1, escala, traslación).
# Esto permite resolver diferencias de handedness sin obligar a q anatómicos a retorcer el cuerpo.
def _signed_axis_maps_134():
    import itertools
    out=[]
    I=np.eye(3,dtype=float)
    for perm in itertools.permutations(range(3)):
        B=I[list(perm),:]
        for signs in itertools.product((-1.0,1.0),repeat=3):
            P=np.diag(signs)@B
            out.append((P,perm,signs,float(np.linalg.det(P))))
    return out

def _kabsch_similarity_rows_134(A,B):
    A=np.asarray(A,float); B=np.asarray(B,float)
    ca=A.mean(0); cb=B.mean(0); Ac=A-ca; Bc=B-cb
    H=Ac.T@Bc
    U,S,Vt=np.linalg.svd(H,full_matrices=False)
    R=Vt.T@U.T
    if np.linalg.det(R)<0:
        Vt[-1,:]*=-1.0; R=Vt.T@U.T
    Ar=Ac@R.T
    den=float(np.sum(Ac*Ac))
    sc=float(np.sum(Ar*Bc)/max(den,1e-12))
    sc=max(sc,1e-6)
    tr=cb-(ca@R.T)*sc
    pred=(A@R.T)*sc+tr
    return R,sc,tr,pred

def _rot_angle_deg_134(R):
    c=float(np.clip((np.trace(np.asarray(R,float))-1.0)/2.0,-1.0,1.0))
    return float(np.degrees(np.arccos(c)))

def _coordinate_system_calibration_134(model,joints_np,target_df,rows):
    J=np.asarray(joints_np,float)
    tmap={r.joint:np.array([r.x,r.y,r.z],float) for r in target_df.itertuples()}
    core_order=['Hip','LHip','RHip','Neck','LShoulder','RShoulder','Head']
    rowmap={r[0]:int(r[1]) for r in rows}
    use=[k for k in core_order if k in tmap and k in rowmap]
    if len(use)<4:
        use=[r[0] for r in rows if r[0] in tmap][:min(8,len(rows))]
    A0=np.stack([J[rowmap[k]] for k in use],axis=0)
    B=np.stack([tmap[k] for k in use],axis=0)
    span=np.nanmedian(np.linalg.norm(B-B.mean(0),axis=1)); span=float(span if np.isfinite(span) and span>1e-6 else 1.0)
    candidates=[]
    for P,perm,signs,detp in _signed_axis_maps_134():
        A=A0@P.T
        try:
            R,sc,tr,pred=_kabsch_similarity_rows_134(A,B)
        except Exception:
            continue
        d=pred-B
        rmse_xy=float(np.sqrt(np.mean(np.sum(d[:,:2]**2,axis=1))))
        rmse_z=float(np.sqrt(np.mean(d[:,2]**2)))
        # Preferimos una base que deje poca rotación residual: identifica la convención de ejes
        # en lugar de esconderla dentro de una rotación global arbitraria.
        residual_deg=_rot_angle_deg_134(R)
        # Coherencia lateral/vertical tras el registro, usando los ejes corporales observables.
        axis_err=0.0
        try:
            predmap={k:pred[i] for i,k in enumerate(use)}
            if all(k in predmap for k in ('LHip','RHip','Hip','Neck')):
                latp=_safe_unit_np(predmap['RHip']-predmap['LHip']); latt=_safe_unit_np(tmap['RHip']-tmap['LHip'])
                upp=_safe_unit_np(predmap['Neck']-predmap['Hip']); upt=_safe_unit_np(tmap['Neck']-tmap['Hip'])
                axis_err=0.5*((1-np.clip(np.dot(latp,latt),-1,1))+(1-np.clip(np.dot(upp,upt),-1,1)))
        except Exception:
            axis_err=0.0
        score=(rmse_xy/span) + 0.12*(rmse_z/span) + 0.22*float(axis_err) + 0.0018*residual_deg
        candidates.append({'score':float(score),'axis_map':P,'perm':tuple(int(x) for x in perm),'signs':tuple(int(x) for x in signs),
                           'det':float(detp),'R':R,'scale':float(sc),'trans':tr,'rmse_xy':rmse_xy,'rmse_z':rmse_z,
                           'residual_rotation_deg':float(residual_deg),'core_landmarks':list(use)})
    if not candidates:
        return {'axis_map':np.eye(3),'R':np.eye(3),'rotvec':np.zeros(3,np.float32),'scale':1.0,'trans':np.zeros(3,np.float32),
                'score':np.inf,'top_candidates':[],'det':1.0,'perm':(0,1,2),'signs':(1,1,1),'core_landmarks':use}
    candidates.sort(key=lambda x:x['score'])
    best=candidates[0]
    best['rotvec']=_rotmat_to_rotvec_np(best['R'])
    # V110.3.13: el RMSE mostrado como 'neutro' se calcula también sobre EXACTAMENTE las mismas correspondencias (15) usadas aguas abajo.
    Tbest=_make_unified_transform_136(best) if '_make_unified_transform_136' in globals() else None
    if Tbest is not None:
        Wall=_apply_unified_np_136(J,Tbest)
        best['rmse_xy_all_correspondences']=_rmse_xy_rows_136(Wall,target_df,rows)
    best['top_candidates']=[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in c.items() if k not in ('R','axis_map')} for c in candidates[:8]]
    return best

def _axis_map_tensor_134(torch,axis_map,dtype,device):
    if axis_map is None:
        return torch.eye(3,dtype=dtype,device=device)
    return torch.as_tensor(np.asarray(axis_map,float),dtype=dtype,device=device)

# V110.3.13 · UNIFIED COORDINATE FRAME
# A partir de la calibración discreta se congela una única similitud SKEL→V104.
# Ningún módulo posterior puede volver a estimar, reordenar o aplicar parcialmente esta transformación.
def _make_unified_transform_136(coord):
    return {
        'axis_map': np.asarray(coord.get('axis_map',np.eye(3)),dtype=np.float64),
        'R': np.asarray(coord.get('R',_np_rodrigues_133(coord.get('rotvec',[0,0,0]))),dtype=np.float64),
        'rotvec': np.asarray(coord.get('rotvec',[0,0,0]),dtype=np.float64),
        'scale': float(coord.get('scale',1.0)),
        'trans': np.asarray(coord.get('trans',[0,0,0]),dtype=np.float64),
        'frozen': True,
        'convention': 'row-vector: ((X @ axis_map.T) @ R.T) * scale + trans',
    }

def _apply_unified_np_136(X,T):
    X=np.asarray(X,float); P=np.asarray(T['axis_map'],float); R=np.asarray(T['R'],float)
    return ((X@P.T)@R.T)*float(T['scale'])+np.asarray(T['trans'],float)

def _apply_unified_torch_136(torch,X,T):
    P=torch.as_tensor(np.asarray(T['axis_map'],float),dtype=X.dtype,device=X.device)
    R=torch.as_tensor(np.asarray(T['R'],float),dtype=X.dtype,device=X.device)
    sc=torch.as_tensor(float(T['scale']),dtype=X.dtype,device=X.device)
    tr=torch.as_tensor(np.asarray(T['trans'],float),dtype=X.dtype,device=X.device)
    return ((X@P.T)@R.T)*sc+tr

def _rmse_xy_rows_136(world,target_df,rows):
    tmap={r.joint:np.array([r.x,r.y,r.z],float) for r in target_df.itertuples()}
    a=[]; b=[]
    for tname,jidx,_ in rows:
        if tname in tmap:
            a.append(np.asarray(world[int(jidx)],float)[:2]); b.append(tmap[tname][:2])
    if not a: return float('nan')
    d=np.asarray(a)-np.asarray(b)
    return float(np.sqrt(np.mean(np.sum(d*d,axis=1))))

def _unified_frame_invariance_136(J0,target_df,rows,T,tol=1e-5):
    W1=_apply_unified_np_136(J0,T)
    # deliberately reconstruct from serialized scalar components to detect convention drift
    T2={'axis_map':np.array(T['axis_map'],float),'R':_np_rodrigues_133(np.array(T['rotvec'],float)),
        'rotvec':np.array(T['rotvec'],float),'scale':float(T['scale']),'trans':np.array(T['trans'],float)}
    W2=_apply_unified_np_136(J0,T2)
    r1=_rmse_xy_rows_136(W1,target_df,rows); r2=_rmse_xy_rows_136(W2,target_df,rows)
    delta=abs(r1-r2) if np.isfinite(r1) and np.isfinite(r2) else float('inf')
    max_xyz=float(np.max(np.abs(W1-W2))) if W1.size else float('inf')
    return {'ok':bool(delta<=tol and max_xyz<=tol),'rmse_xy_reference':r1,'rmse_xy_reconstructed':r2,
            'delta_rmse_xy':float(delta),'max_coordinate_delta':max_xyz,'tolerance':float(tol)}

def _json_safe_134(obj):
    if isinstance(obj,np.ndarray): return obj.tolist()
    if isinstance(obj,(np.floating,)): return float(obj)
    if isinstance(obj,(np.integer,)): return int(obj)
    if isinstance(obj,(np.bool_,)): return bool(obj)
    if isinstance(obj,dict): return {str(k):_json_safe_134(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)): return [_json_safe_134(v) for v in obj]
    return obj


# V110.3.13 · CALIBRACIÓN AUTOMÁTICA DE CADA DoF LOCAL SKEL
# Perturbación central ±Δ alrededor de la pose neutra ya registrada. Mide qué joints mueve
# realmente cada q, su especificidad de cadena, sensibilidad XY/Z y un eje angular efectivo.
_DOF_CHAIN_JOINTS_135={
 'tronco':['spine1','spine2','spine3','neck','head','leftshoulder','rightshoulder','leftelbow','rightelbow','leftwrist','rightwrist'],
 'cabeza':['head'],
 'pierna_D':['rightknee','rightankle','rightfoot'],
 'pierna_I':['leftknee','leftankle','leftfoot'],
 'brazo_D':['rightshoulder','rightelbow','rightwrist'],
 'brazo_I':['leftshoulder','leftelbow','leftwrist'],
}
_DOF_CHAIN_BASIS_135={
 'tronco':('pelvis',['neck','head','leftshoulder','rightshoulder']),
 'cabeza':('neck',['head']),
 'pierna_D':('righthip',['rightknee','rightankle']),
 'pierna_I':('lefthip',['leftknee','leftankle']),
 'brazo_D':('rightshoulder',['rightelbow','rightwrist']),
 'brazo_I':('leftshoulder',['leftelbow','leftwrist']),
}

def _expected_chain_for_q_135(qi):
    for g,ids in _DIRECT_GROUPS_130.items():
        if qi in ids: return g
    return 'global_or_other'

def _skel_descendant_policy_137():
    """Expected downstream influence by anatomical chain.

    A proximal DoF is *supposed* to move distal descendants. We therefore score
    q by whether motion remains inside its legal kinematic subtree, not by spatial locality.
    """
    return {
      'tronco': {'allow':['spine1','spine2','spine3','neck','head','leftcollar','rightcollar','leftshoulder','rightshoulder','leftelbow','rightelbow','leftwrist','rightwrist'],
                 'forbid':['lefthip','righthip','leftknee','rightknee','leftankle','rightankle','leftfoot','rightfoot']},
      'cabeza': {'allow':['head'], 'forbid':['pelvis','lefthip','righthip','leftknee','rightknee','leftankle','rightankle','leftshoulder','rightshoulder','leftelbow','rightelbow','leftwrist','rightwrist']},
      'pierna_D': {'allow':['righthip','rightknee','rightankle','rightfoot'], 'forbid':['lefthip','leftknee','leftankle','leftfoot','leftshoulder','rightshoulder','leftelbow','rightelbow','leftwrist','rightwrist','head']},
      'pierna_I': {'allow':['lefthip','leftknee','leftankle','leftfoot'], 'forbid':['righthip','rightknee','rightankle','rightfoot','leftshoulder','rightshoulder','leftelbow','rightelbow','leftwrist','rightwrist','head']},
      'brazo_D': {'allow':['rightcollar','rightshoulder','rightelbow','rightwrist'], 'forbid':['leftcollar','leftshoulder','leftelbow','leftwrist','lefthip','righthip','leftknee','rightknee','leftankle','rightankle']},
      'brazo_I': {'allow':['leftcollar','leftshoulder','leftelbow','leftwrist'], 'forbid':['rightcollar','rightshoulder','rightelbow','rightwrist','lefthip','righthip','leftknee','rightknee','leftankle','rightankle']},
    }

def _calibrate_local_dofs_135(torch,model,betas,zero_trans,axis_map,rotvec,scale,delta=0.08,unified_transform=None):
    """V110.3.13 · Kinematic-tree aware finite-difference calibration.

    Distal propagation inside the legal subtree is rewarded rather than penalised.
    Cross-chain leakage is what makes a DoF unsafe.
    """
    nq=int(model.num_q_params)
    names=_skel_forward_joint_names_140(model)
    nm={_simple_name(n):i for i,n in enumerate(names)}
    policy=_skel_descendant_policy_137()
    T=unified_transform if unified_transform is not None else {'axis_map':np.asarray(axis_map,float) if axis_map is not None else np.eye(3),'R':_np_rodrigues_133(np.asarray(rotvec,float)),'rotvec':np.asarray(rotvec,float),'scale':float(scale),'trans':np.zeros(3,float)}
    pose0=np.zeros((1,nq),np.float32)
    with torch.no_grad():
        J0=model(torch.tensor(pose0),betas,zero_trans,skelmesh=False).joints[0].detach().cpu().numpy()
    W0=_apply_unified_np_136(J0,T)
    rows=[]; jac_world={}
    for qi in range(nq):
        qp=pose0.copy(); qm=pose0.copy(); qp[0,qi]=float(delta); qm[0,qi]=-float(delta)
        try:
            with torch.no_grad():
                Jp=model(torch.tensor(qp),betas,zero_trans,skelmesh=False).joints[0].detach().cpu().numpy()
                Jm=model(torch.tensor(qm),betas,zero_trans,skelmesh=False).joints[0].detach().cpu().numpy()
            Wp=_apply_unified_np_136(Jp,T); Wm=_apply_unified_np_136(Jm,T)
            dW=(Wp-Wm)/(2.0*float(delta)); jac_world[qi]=dW
            mag=np.linalg.norm(dW,axis=1); order=np.argsort(-mag)
            top=[names[j] for j in order[:6] if j<len(names)]
            expected=_expected_chain_for_q_135(qi)
            total=float(np.sum(mag**2))+1e-12
            allow_ids=[]; forbid_ids=[]
            if expected in policy:
                allow_ids=[nm.get(_simple_name(x)) for x in policy[expected]['allow']]; allow_ids=[j for j in allow_ids if j is not None]
                forbid_ids=[nm.get(_simple_name(x)) for x in policy[expected]['forbid']]; forbid_ids=[j for j in forbid_ids if j is not None]
            allowed_energy=float(np.sum(mag[allow_ids]**2))/total if allow_ids else 0.0
            forbidden_energy=float(np.sum(mag[forbid_ids]**2))/total if forbid_ids else 0.0
            subtree_score=allowed_energy/(allowed_energy+forbidden_energy+1e-9) if expected in policy else 0.0
            omega=[]
            if expected in _DOF_CHAIN_BASIS_135:
                prox,distals=_DOF_CHAIN_BASIS_135[expected]; ip=nm.get(_simple_name(prox))
                if ip is not None:
                    for dn in distals:
                        jd=nm.get(_simple_name(dn))
                        if jd is None: continue
                        r=W0[jd]-W0[ip]; dr=dW[jd]-dW[ip]; rr=float(np.dot(r,r))
                        if rr>1e-8:
                            om=np.cross(r,dr)/rr
                            if np.linalg.norm(om)>1e-8: omega.append(om)
            if omega:
                om=np.mean(np.stack(omega),axis=0); on=float(np.linalg.norm(om)); axis=om/on if on>1e-8 else np.zeros(3)
            else: axis=np.zeros(3); on=0.0
            sens_xy=float(np.sqrt(np.mean(np.sum(dW[:,:2]**2,axis=1)))); sens_z=float(np.sqrt(np.mean(dW[:,2]**2)))
            # Tree-aware criterion: legal distal propagation is expected; only cross-chain leakage is penalized.
            safe=bool(qi in _DIRECT_ACTIVE_130 and subtree_score>=0.72 and forbidden_energy<=0.24 and sens_xy>1e-5)
            rows.append({'q':int(qi),'name':_SKEL_Q_NAMES_130[qi] if qi<len(_SKEL_Q_NAMES_130) else f'q{qi:02d}',
                         'expected_chain':expected,'allowed_descendant_energy':allowed_energy,'forbidden_cross_chain_energy':forbidden_energy,
                         'kinematic_tree_score':float(subtree_score),'chain_specificity':float(subtree_score),'sensitivity_xy':sens_xy,
                         'sensitivity_z':sens_z,'axis_x':float(axis[0]),'axis_y':float(axis[1]),'axis_z':float(axis[2]),
                         'angular_gain':float(on),'top_moved_joints':top,'safe_for_fit':safe})
        except Exception as exc:
            rows.append({'q':int(qi),'name':_SKEL_Q_NAMES_130[qi] if qi<len(_SKEL_Q_NAMES_130) else f'q{qi:02d}',
                         'expected_chain':_expected_chain_for_q_135(qi),'allowed_descendant_energy':0.0,'forbidden_cross_chain_energy':1.0,
                         'kinematic_tree_score':0.0,'chain_specificity':0.0,'sensitivity_xy':0.0,'sensitivity_z':0.0,
                         'axis_x':0.0,'axis_y':0.0,'axis_z':0.0,'angular_gain':0.0,'top_moved_joints':[],'safe_for_fit':False,'error':f'{type(exc).__name__}: {exc}'})
    return {'version':'110.3.13','delta_rad':float(delta),'rows':rows,'jac_world':jac_world,
            'safe_q':[r['q'] for r in rows if r.get('safe_for_fit')],
            'strategy':'kinematic-tree aware central finite differences: legal descendant propagation rewarded; cross-chain leakage rejected'}



def _fk_ground_truth_calibration_138(torch,model,betas,zero_trans,unified_transform,delta=0.06):
    """Empirical ground truth from SKEL forward kinematics.

    No semantic assumption is used to decide what a q moves. Every q is perturbed
    ±delta from the neutral pose, transformed with the frozen T_SKEL→V104 and the
    resulting 24-joint displacement field is measured directly.
    """
    nq=int(model.num_q_params)
    names=_skel_forward_joint_names_140(model)
    T=unified_transform
    with torch.no_grad():
        z=torch.zeros((1,nq),dtype=torch.float32)
        J0=model(z,betas,zero_trans,skelmesh=False).joints[0].cpu().numpy()
    W0=_apply_unified_np_136(J0,T)
    rows=[]; jac_world={}; graph={}
    # target-chain sets use actual model joints only for reporting/assignment; influence itself is empirical.
    chain_sets={
      'tronco':{'lumbar_body','thorax','scapula_r','scapula_l'},
      'cabeza':{'head'},
      'pierna_D':{'femur_r','tibia_r','talus_r','calcn_r','toes_r'},
      'pierna_I':{'femur_l','tibia_l','talus_l','calcn_l','toes_l'},
      'brazo_D':{'scapula_r','humerus_r','ulna_r','radius_r','hand_r'},
      'brazo_I':{'scapula_l','humerus_l','ulna_l','radius_l','hand_l'},
    }
    simple=[_simple_name(n) for n in names]
    simple_sets={g:{_simple_name(x) for x in ss} for g,ss in chain_sets.items()}
    for qi in range(nq):
        try:
            pp=np.zeros((1,nq),np.float32); pm=np.zeros((1,nq),np.float32)
            pp[0,qi]=float(delta); pm[0,qi]=-float(delta)
            with torch.no_grad():
                Jp=model(torch.tensor(pp),betas,zero_trans,skelmesh=False).joints[0].cpu().numpy()
                Jm=model(torch.tensor(pm),betas,zero_trans,skelmesh=False).joints[0].cpu().numpy()
            Wp=_apply_unified_np_136(Jp,T); Wm=_apply_unified_np_136(Jm,T)
            dW=(Wp-Wm)/(2.0*float(delta)); jac_world[qi]=dW
            mag=np.linalg.norm(dW,axis=1); mx=float(np.max(mag)) if len(mag) else 0.0
            thr=max(1e-5,0.08*mx)
            moved=[names[i] for i,v in enumerate(mag) if float(v)>=thr]
            energies={}
            for g,ss in simple_sets.items():
                ids=[i for i,n in enumerate(simple) if n in ss]
                energies[g]=float(np.sum(mag[ids]**2)) if ids else 0.0
            total=float(np.sum(mag**2))+1e-12
            best_group=max(energies,key=energies.get) if energies else 'global_or_other'
            dominance=float(energies.get(best_group,0.0)/total)
            # q0-q2 remain reserved for the already-frozen global frame.
            usable=bool(qi>=3 and mx>1e-5 and dominance>=0.18)
            graph[int(qi)]={'moved_joints':moved,'influence':{names[i]:float(mag[i]) for i in range(len(names)) if float(mag[i])>=thr},
                            'assigned_group':best_group,'dominance':dominance}
            rows.append({'q':int(qi),'name':_SKEL_Q_NAMES_130[qi] if qi<len(_SKEL_Q_NAMES_130) else f'q{qi:02d}',
                         'assigned_group':best_group,'dominance':dominance,'max_joint_sensitivity':mx,
                         'moved_joint_count':len(moved),'moved_joints':moved,'usable':usable})
        except Exception as exc:
            rows.append({'q':int(qi),'name':f'q{qi:02d}','assigned_group':'error','dominance':0.0,'max_joint_sensitivity':0.0,
                         'moved_joint_count':0,'moved_joints':[],'usable':False,'error':f'{type(exc).__name__}: {exc}'})
    # V110.3.13: la dependencia FK se sigue midiendo empíricamente, pero la PROPIEDAD
    # de cada q se toma del mapa oficial SKEL. Un q lumbar puede mover ambos brazos y por
    # energía parecer 'brazo_D'; eso no cambia que pertenezca al tronco.
    q_by_group={g:[] for g in ['tronco','cabeza','pierna_D','pierna_I','brazo_D','brazo_I']}
    semantic_owner={}
    for g,ids in _DIRECT_GROUPS_130.items():
        if g in q_by_group:
            for qi in ids: semantic_owner[int(qi)]=g
    usable_measured={int(r['q']) for r in rows if r.get('usable')}
    for qi,g in semantic_owner.items():
        if qi in usable_measured:
            q_by_group[g].append(qi)
    # Reflejar también la propiedad semántica en la tabla, preservando el grupo empírico
    # como diagnóstico independiente.
    for r in rows:
        qi=int(r.get('q',-1)); r['empirical_dominant_group']=r.get('assigned_group')
        if qi in semantic_owner:
            r['assigned_group']=semantic_owner[qi]
    return {'version':'110.3.13','delta_rad':float(delta),'rows':rows,'jac_world':jac_world,'dependency_graph':graph,
            'q_by_group':q_by_group,'usable_q':sorted({q for v in q_by_group.values() for q in v}),
            'strategy':'forward-kinematics ground truth + official SKEL q ownership; empirical dependency graph retained for validation'}


def _official_skel24_sanity_140(fk_gt):
    """Valida lateralidad mínima del grafo FK antes de optimizar."""
    byq={int(r.get('q')):r for r in (fk_gt or {}).get('rows',[]) if isinstance(r,dict) and 'q' in r}
    tests=[]
    def one(q, expected_group, forbidden_tokens):
        r=byq.get(q,{})
        moved=[_simple_name(x) for x in r.get('moved_joints',[])]
        ok=bool(r.get('usable')) and r.get('assigned_group')==expected_group and not any(any(tok in m for tok in forbidden_tokens) for m in moved)
        tests.append({'q':q,'name':r.get('name',f'q{q:02d}'),'expected_group':expected_group,'assigned_group':r.get('assigned_group'),'moved_joints':r.get('moved_joints',[]),'ok':ok})
    one(3,'pierna_D',['_l','left'])
    one(10,'pierna_I',['_r','right'])
    one(29,'brazo_D',['_l','left'])
    one(32,'brazo_D',['_l','left'])
    one(39,'brazo_I',['_r','right'])
    one(42,'brazo_I',['_r','right'])
    return {'ok':all(t['ok'] for t in tests),'tests':tests,'joint_order':_SKEL24_FORWARD_JOINT_NAMES_140}

def _group_rmse_xy_138(world_np,target_df,rows,group):
    lm=set(_DIRECT_LM_BY_GROUP_130.get(group,[]))
    target={r.joint:np.array([r.x,r.y,r.z],float) for r in target_df.itertuples()}
    vals=[]
    for tname,jidx,_ in rows:
        if tname in lm and tname in target:
            d=np.asarray(world_np[int(jidx),:2],float)-target[tname][:2]
            vals.append(float(np.dot(d,d)))
    return float(np.sqrt(np.mean(vals))) if vals else float('nan')


def _monotonic_hierarchical_ik_138(torch,model,target_df,rows,betas,zero_trans,unified,pose_init,q_by_group,iterations_per_stage=55):
    """Hierarchical IK with rollback.

    A stage is committed only when global XY error does not increase and every
    previously frozen chain stays within a small tolerance. Otherwise the pose is rolled back.
    """
    pose=torch.tensor(np.asarray(pose_init,np.float32),dtype=torch.float32).reshape(1,-1)
    axis_map=np.asarray(unified['axis_map'],dtype=np.float32)
    rotvec=torch.tensor(np.asarray(unified['rotvec'],np.float32),dtype=torch.float32)
    scale=torch.tensor(float(unified['scale']),dtype=torch.float32)
    trans=torch.tensor(np.asarray(unified['trans'],np.float32),dtype=torch.float32)
    order=['tronco','pierna_D','pierna_I','brazo_D','brazo_I','cabeza']
    history=[]; frozen=[]
    def eval_world(pp):
        with torch.no_grad():
            w,_,_=_direct_world_130(torch,model,pp,betas,zero_trans,rotvec,scale,trans,axis_map=axis_map)
        wn=w.cpu().numpy(); return wn,_rmse_xy_rows_136(wn,target_df,rows)
    world0,global0=eval_world(pose)
    frozen_err={}
    for group in order:
        ids=[int(i) for i in q_by_group.get(group,[]) if int(i) in _DIRECT_ACTIVE_130]
        if not ids:
            history.append({'stage':group,'accepted':False,'reason':'no empirical q assigned','active_q':[],'rmse_xy_before':global0,'rmse_xy_after':global0}); continue
        before_pose=pose.detach().clone(); before_world,before_global=eval_world(before_pose)
        before_frozen={g:_group_rmse_xy_138(before_world,target_df,rows,g) for g in frozen}
        candidate=before_pose.clone().requires_grad_(True)
        opt=torch.optim.Adam([candidate],lr=0.008 if group in ('tronco','cabeza') else 0.010)
        for _ in range(int(iterations_per_stage)):
            opt.zero_grad(set_to_none=True)
            world,_,_=_direct_world_130(torch,model,candidate,betas,zero_trans,rotvec,scale,trans,axis_map=axis_map)
            data,bdir=_direct_group_loss_130(torch,model,world,target_df,rows,group)
            # weak global anchor prevents a local chain from improving itself by destroying the rest.
            _,idx_all,tgt_all,_=_target_tensor_for_rows(torch,target_df,rows,None)
            pred_all=world.index_select(0,idx_all)
            global_anchor=torch.mean(torch.sum((pred_all[:,:2]-tgt_all[:,:2])**2,dim=1))
            reg=0.02*torch.mean((candidate[:,ids]-before_pose[:,ids])**2)
            loss=data+0.55*bdir+0.22*global_anchor+reg
            loss.backward()
            if candidate.grad is not None:
                mask=torch.zeros_like(candidate.grad); mask[:,ids]=1.0; candidate.grad.mul_(mask)
            torch.nn.utils.clip_grad_norm_([candidate],2.0); opt.step(); _clamp_direct_pose_130(torch,candidate)
        candidate=candidate.detach(); after_world,after_global=eval_world(candidate)
        after_frozen={g:_group_rmse_xy_138(after_world,target_df,rows,g) for g in frozen}
        current_before=_group_rmse_xy_138(before_world,target_df,rows,group)
        current_after=_group_rmse_xy_138(after_world,target_df,rows,group)
        global_ok=bool(np.isfinite(after_global) and after_global<=before_global+1e-5)
        frozen_ok=all((not np.isfinite(before_frozen[g])) or (not np.isfinite(after_frozen[g])) or after_frozen[g]<=before_frozen[g]+0.015 for g in frozen)
        local_ok=bool((not np.isfinite(current_before)) or (np.isfinite(current_after) and current_after<=current_before+1e-5))
        accepted=bool(global_ok and frozen_ok and local_ok)
        if accepted:
            pose=candidate; frozen.append(group); world0,global0=after_world,after_global
            frozen_err[group]=current_after
        else:
            pose=before_pose; world0,global0=before_world,before_global
        history.append({'stage':group,'accepted':accepted,'active_q':ids,'rmse_xy_before':float(before_global),'rmse_xy_after':float(after_global),
                        'group_rmse_before':float(current_before) if np.isfinite(current_before) else None,
                        'group_rmse_after':float(current_after) if np.isfinite(current_after) else None,
                        'frozen_guard_ok':bool(frozen_ok),'global_guard_ok':bool(global_ok),'local_guard_ok':bool(local_ok)})
    return pose.cpu().numpy()[0].astype(np.float32), {'version':'110.3.13','history':history,'order':order,'accepted_stages':frozen,
            'rmse_xy_final':float(global0),'global_frame_frozen':True,'strategy':'monotonic hierarchical IK with rollback and frozen-chain guards'}

def _linear_seed_from_dof_calibration_135(target_df,rows_corr,calib,J0_world,active_q):
    if not active_q: return np.zeros(46,np.float32), {'ok':False,'reason':'no safe q'}
    # Build correspondence indices against SKEL joint names saved in rows_corr.
    target={r.joint:np.array([r.x,r.y,r.z],float) for r in target_df.itertuples()}
    # rows_corr tuples: target_name, skel_index, skel_name
    obs=[]; base=[]; jac_cols=[]
    for tname,jidx,_ in rows_corr:
        if tname not in target: continue
        obs.append(target[tname]); base.append(J0_world[int(jidx)])
    if not obs: return np.zeros(46,np.float32), {'ok':False,'reason':'no observations'}
    obs=np.asarray(obs,float); base=np.asarray(base,float); resid=(obs-base)[:,:2].reshape(-1)
    for qi in active_q:
        dW=calib['jac_world'].get(int(qi))
        if dW is None: jac_cols.append(np.zeros_like(resid)); continue
        col=[]
        for tname,jidx,_ in rows_corr:
            if tname not in target: continue
            col.extend(dW[int(jidx),:2].tolist())
        jac_cols.append(np.asarray(col,float))
    A=np.stack(jac_cols,axis=1)
    # Ridge: stable linearized anatomical seed, not final optimizer.
    lam=0.08
    try: dq=np.linalg.solve(A.T@A+lam*np.eye(A.shape[1]),A.T@resid)
    except Exception: dq=np.linalg.lstsq(np.vstack([A,np.sqrt(lam)*np.eye(A.shape[1])]),np.r_[resid,np.zeros(A.shape[1])],rcond=None)[0]
    pose=np.zeros(46,np.float32)
    for k,qi in enumerate(active_q):
        lo,hi=_DIRECT_LIMITS_130.get(int(qi),(-0.5,0.5)); pose[int(qi)]=np.float32(np.clip(dq[k],lo,hi))
    pred=base.copy()
    for k,qi in enumerate(active_q):
        dW=calib['jac_world'].get(int(qi))
        if dW is not None:
            for oi,(tname,jidx,_) in enumerate(rows_corr):
                if oi < len(pred): pred[oi]+=dW[int(jidx)]*float(pose[int(qi)])
    xy0=float(np.sqrt(np.mean(np.sum((base[:,:2]-obs[:,:2])**2,axis=1))))
    xy1=float(np.sqrt(np.mean(np.sum((pred[:,:2]-obs[:,:2])**2,axis=1))))
    return pose, {'ok':True,'rmse_xy_neutral':xy0,'rmse_xy_linear_seed':xy1,'improvement_pct':float((1-xy1/max(xy0,1e-12))*100),'active_q':list(map(int,active_q))}

def _hierarchical_ik_seed_137(torch,model,target_df,rows,betas,zero_trans,unified,pose_init,active_q,iterations_per_stage=45):
    """Proximal→distal IK in the frozen unified frame.

    Each stage optimizes only q belonging to that anatomical chain. Global similarity
    is immutable. The next stage starts from the previous stage pose.
    """
    pose=torch.tensor(np.asarray(pose_init,np.float32),dtype=torch.float32).reshape(1,-1)
    axis_map=np.asarray(unified['axis_map'],dtype=np.float32)
    rotvec=torch.tensor(np.asarray(unified['rotvec'],np.float32),dtype=torch.float32)
    scale=torch.tensor(float(unified['scale']),dtype=torch.float32)
    trans=torch.tensor(np.asarray(unified['trans'],np.float32),dtype=torch.float32)
    order=['tronco','pierna_D','pierna_I','brazo_D','brazo_I','cabeza']
    history=[]; aset=set(map(int,active_q))
    for group in order:
        ids=[i for i in _DIRECT_GROUPS_130.get(group,[]) if i in aset]
        if not ids: continue
        ref=pose.detach().clone(); pose.requires_grad_(True)
        opt=torch.optim.Adam([pose],lr=0.012 if group.startswith('pierna') or group.startswith('brazo') else 0.008)
        for _ in range(int(iterations_per_stage)):
            opt.zero_grad(set_to_none=True)
            world,_,_=_direct_world_130(torch,model,pose,betas,zero_trans,rotvec,scale,trans,axis_map=axis_map)
            data,bdir=_direct_group_loss_130(torch,model,world,target_df,rows,group)
            reg=0.025*torch.mean((pose[:,ids]-ref[:,ids])**2)
            loss=data+0.65*bdir+reg
            loss.backward()
            if pose.grad is not None:
                mask=torch.zeros_like(pose.grad); mask[:,ids]=1.0; pose.grad.mul_(mask)
            torch.nn.utils.clip_grad_norm_([pose],2.0); opt.step(); _clamp_direct_pose_130(torch,pose)
        pose.requires_grad_(False)
        with torch.no_grad():
            world,_,_=_direct_world_130(torch,model,pose,betas,zero_trans,rotvec,scale,trans,axis_map=axis_map)
            rxy=_rmse_xy_rows_136(world.cpu().numpy(),target_df,rows)
        history.append({'stage':group,'active_q':[int(i) for i in ids],'rmse_xy_all15':float(rxy)})
    return pose.detach().cpu().numpy()[0].astype(np.float32), {'history':history,'order':order,'global_frame_frozen':True}

def _active_for_target_133(target_df):
    dw=float(target_df.attrs.get('depth_weight',1.0)) if hasattr(target_df,'attrs') else 1.0
    return list(_AUTOMATA_SAFE_Q_133 if dw<0.75 else _DIRECT_ACTIVE_130)

def _automata_bounds_133(rv_seed,scale_seed,active_q,initial_pose=None,freeze_global=False):
    lo=[]; hi=[]; x0=[]
    # Global rotación: búsqueda local amplia pero no permite vueltas completas arbitrarias.
    for j in range(3):
        x0.append(float(rv_seed[j]))
        if freeze_global: lo.append(float(rv_seed[j])); hi.append(float(rv_seed[j]))
        else: lo.append(float(rv_seed[j])-0.65); hi.append(float(rv_seed[j])+0.65)
    # Escala positiva en log-space; V110.3.13 puede congelarla como parte de T_SKEL→V104.
    ls=float(np.log(max(float(scale_seed),1e-4))); x0.append(ls)
    if freeze_global: lo.append(ls); hi.append(ls)
    else: lo.append(ls+np.log(0.80)); hi.append(ls+np.log(1.25))
    for qi in active_q:
        qlo,qhi=_DIRECT_LIMITS_130.get(qi,(-0.5,0.5)); qseed=float(initial_pose[qi]) if initial_pose is not None and qi<len(initial_pose) else 0.0; x0.append(float(np.clip(qseed,qlo,qhi))); lo.append(float(qlo)); hi.append(float(qhi))
    return np.asarray(x0,float),np.asarray(lo,float),np.asarray(hi,float)

def _automata_eval_population_133(torch,model,X,active_q,betas,zero_trans,idx,tgt_np,target_df,axis_map=None,fixed_trans=None):
    """Evaluación vectorizada de población. Skin/skel mesh desactivados: sólo joints."""
    X=np.asarray(X,float); n=len(X)
    poses=np.zeros((n,int(model.num_q_params)),dtype=np.float32)
    for k,qi in enumerate(active_q): poses[:,qi]=X[:,4+k].astype(np.float32)
    with torch.no_grad():
        pt=torch.tensor(poses,dtype=torch.float32)
        bt=betas.repeat(n,1); zt=zero_trans.repeat(n,1)
        J=model(pt,bt,zt,skelmesh=False).joints.detach().cpu().numpy()
    tgt=np.asarray(tgt_np,float)
    names=_skel_forward_joint_names_140(model); nm={_simple_name(nm0):i for i,nm0 in enumerate(names)}
    # V110.3.13: auditoría ósea contra el orden REAL de forward().joints (SKEL24).
    corr={'Hip':'pelvis','RHip':'femur_r','RKnee':'tibia_r','RAnkle':'talus_r','LHip':'femur_l','LKnee':'tibia_l','LAnkle':'talus_l',
          'Neck':'thorax','Head':'head','RShoulder':'humerus_r','RElbow':'ulna_r','RWrist':'hand_r','LShoulder':'humerus_l','LElbow':'ulna_l','LWrist':'hand_l'}
    tmap={r.joint:np.array([r.x,r.y,r.z],float) for r in target_df.itertuples()}
    bones=[]
    for bl in _DIRECT_BONES_BY_GROUP_130.values(): bones.extend(bl)
    # unique ordered pairs
    seen=set(); bones=[b for b in bones if not (b in seen or seen.add(b))]
    depth_w=float(target_df.attrs.get('depth_weight',0.18)) if hasattr(target_df,'attrs') else 0.18
    depth_w=max(0.05,min(1.0,depth_w))
    body_scale=np.nanmedian([np.linalg.norm(tmap[b]-tmap[a]) for a,b in bones if a in tmap and b in tmap])
    if not np.isfinite(body_scale) or body_scale<1e-6: body_scale=1.0
    scores=np.full(n,np.inf,float); detail=[]
    for i in range(n):
        R=_np_rodrigues_133(X[i,:3]); sc=float(np.exp(X[i,3]))
        P=np.asarray(axis_map,float) if axis_map is not None else np.eye(3,dtype=float)
        world=((J[i]@P.T)@R.T)*sc
        sel=world[np.asarray(idx,dtype=int)]
        # V110.3.13: con Unified Coordinate Frame la traslación también se congela.
        tr=np.asarray(fixed_trans,float) if fixed_trans is not None else np.nanmean(tgt-sel,axis=0)
        world=world+tr; pred=world[np.asarray(idx,dtype=int)]
        d=pred-tgt
        exy=float(np.mean(np.sum(d[:,:2]**2,axis=1))/(body_scale**2+1e-9))
        ez=float(np.mean(d[:,2]**2)/(body_scale**2+1e-9))
        dirloss=[]; lenloss=[]; angle=[]
        for a,b in bones:
            ia=nm.get(corr.get(a,'')); ib=nm.get(corr.get(b,''))
            if ia is None or ib is None or a not in tmap or b not in tmap: continue
            va=world[ib]-world[ia]; vb=tmap[b]-tmap[a]
            na=np.linalg.norm(va); nb=np.linalg.norm(vb)
            if na<1e-8 or nb<1e-8: continue
            c=float(np.clip(np.dot(va,vb)/(na*nb),-1.0,1.0))
            dirloss.append(1.0-c); angle.append(np.degrees(np.arccos(c)))
            lenloss.append(((na-nb)/(nb+1e-8))**2)
        bdir=float(np.mean(dirloss)) if dirloss else 1.0
        blen=float(np.mean(lenloss)) if lenloss else 1.0
        prior=float(np.mean((X[i,4:]/0.8)**2)) if X.shape[1]>4 else 0.0
        # Coste monocular: fidelidad XY + Z protegida + anatomía + prior neutro.
        score=1.0*exy + depth_w*0.18*ez + 0.72*bdir + 0.08*blen + 0.055*prior
        # Penalización suave adicional al acercarse demasiado a los límites.
        margin_pen=0.0
        for k,qi in enumerate(active_q):
            qlo,qhi=_DIRECT_LIMITS_130.get(qi,(-1,1)); q=X[i,4+k]; span=max(qhi-qlo,1e-6)
            edge=min((q-qlo)/span,(qhi-q)/span)
            if edge<0.04: margin_pen += (0.04-edge)*4.0
        score += 0.12*margin_pen
        scores[i]=score
        detail.append({'xy':exy,'z':ez,'bone_dir':bdir,'bone_len':blen,'prior':prior,
                       'mean_angle_deg':float(np.mean(angle)) if angle else np.nan,
                       'max_angle_deg':float(np.max(angle)) if angle else np.nan,
                       'trans':tr.astype(float).tolist(),'scale':sc})
    return scores,detail

def _skel_anatomical_automata_133(torch,model,target_df,rows,betas,zero_trans,rv_seed,scale_seed,idx,tgt,
                                  generations=_AUTOMATA_MAX_GEN_133,popsize=_AUTOMATA_POP_133,seed=136,axis_map=None,active_q_override=None,initial_pose=None,freeze_global=False,fixed_trans=None):
    active_q=list(active_q_override) if active_q_override else _active_for_target_133(target_df)
    x0,lower,upper=_automata_bounds_133(rv_seed,scale_seed,active_q,initial_pose=initial_pose,freeze_global=freeze_global)
    n=len(x0); rng=np.random.default_rng(int(seed))
    # CMA-ES estándar reducido con proyección a límites. Inicialización centrada en registro neutro.
    lam=max(8,int(popsize)); mu=lam//2
    weights=np.log(mu+0.5)-np.log(np.arange(1,mu+1)); weights=weights/weights.sum(); mueff=1.0/np.sum(weights**2)
    cc=(4+mueff/n)/(n+4+2*mueff/n); cs=(mueff+2)/(n+mueff+5)
    c1=2/((n+1.3)**2+mueff); cmu=min(1-c1,2*(mueff-2+1/mueff)/((n+2)**2+mueff))
    damps=1+2*max(0,np.sqrt((mueff-1)/(n+1))-1)+cs
    sigma=0.24; C=np.eye(n); pc=np.zeros(n); ps=np.zeros(n); mean=x0.copy()
    # dimensión escala y global algo más contenida por escalado de covarianza inicial.
    C[3,3]=0.35
    chiN=np.sqrt(n)*(1-1/(4*n)+1/(21*n*n))
    best=[]; history=[]; tgt_np=tgt.detach().cpu().numpy(); idx_np=idx.detach().cpu().numpy()
    for g in range(int(generations)):
        vals,vecs=np.linalg.eigh((C+C.T)*0.5); vals=np.clip(vals,1e-8,1e3); A=vecs@np.diag(np.sqrt(vals))
        Z=rng.normal(size=(lam,n)); Y=Z@A.T; X=mean+sigma*Y; X=np.clip(X,lower,upper)
        scores,details=_automata_eval_population_133(torch,model,X,active_q,betas,zero_trans,idx_np,tgt_np,target_df,axis_map=axis_map,fixed_trans=fixed_trans if freeze_global else None)
        order=np.argsort(scores); elite=order[:mu]
        for oi in order[:min(_AUTOMATA_TOPK_133,lam)]:
            best.append((float(scores[oi]),X[oi].copy(),details[oi].copy(),g))
        best=sorted(best,key=lambda z:z[0])[:_AUTOMATA_TOPK_133]
        old=mean.copy(); mean=np.sum(weights[:,None]*X[elite],axis=0); y=(mean-old)/max(sigma,1e-9)
        invsqrt=vecs@np.diag(1/np.sqrt(vals))@vecs.T
        ps=(1-cs)*ps+np.sqrt(cs*(2-cs)*mueff)*(invsqrt@y)
        hsig=float(np.linalg.norm(ps)/np.sqrt(1-(1-cs)**(2*(g+1)))/chiN < (1.4+2/(n+1)))
        pc=(1-cc)*pc+hsig*np.sqrt(cc*(2-cc)*mueff)*y
        artmp=(X[elite]-old)/max(sigma,1e-9)
        C=(1-c1-cmu)*C+c1*(np.outer(pc,pc)+(1-hsig)*cc*(2-cc)*C)
        C+=cmu*sum(weights[j]*np.outer(artmp[j],artmp[j]) for j in range(mu))
        sigma*=np.exp((cs/damps)*(np.linalg.norm(ps)/chiN-1))
        history.append({'generation':g+1,'best':float(scores[order[0]]),'median':float(np.median(scores)),'sigma':float(sigma)})
        # aprobación temprana estocástica aproximada; la auditoría SKEL exacta se hace después.
        d=details[order[0]]
        if g>=5 and d.get('mean_angle_deg',999)<14.0 and d.get('max_angle_deg',999)<42.0 and d.get('xy',999)<0.12:
            break
    score,x,det,gen=best[0]
    pose=np.zeros(int(model.num_q_params),dtype=np.float32)
    for k,qi in enumerate(active_q): pose[qi]=np.float32(x[4+k])
    return {'pose':pose,'rotvec':np.asarray(x[:3],np.float32),'scale':float(np.exp(x[3])),
            'trans':np.asarray(det['trans'],np.float32),'score':float(score),'active_q':active_q,
            'top_k':[{'rank':i+1,'score':b[0],'generation':b[3]+1,**b[2]} for i,b in enumerate(best)],
            'history':history,'generations_run':len(history),'population':lam,
            'strategy':'bounded reduced-space CMA-ES + XY-first depth-safe anatomical energy','global_transform_frozen':bool(freeze_global)}


# V110.3.13 · PERIPHERAL LANDMARK ANATOMICAL REFINEMENT
# V110.3.9 queda congelada como primera base anatómicamente válida. Esta etapa sólo
# intenta reducir residuos distales sin permitir que tronco/cadenas ya correctas empeoren.
_PERIPHERAL_GROUPS_140 = {
    'pierna_D_distal': {'landmarks':['RKnee','RAnkle'], 'q':[3,4,5,6]},
    'pierna_I_distal': {'landmarks':['LKnee','LAnkle'], 'q':[10,11,12,13]},
    'brazo_D_distal':  {'landmarks':['RElbow','RWrist'], 'q':[26,27,28,29,30,31,32]},
    'brazo_I_distal':  {'landmarks':['LElbow','LWrist'], 'q':[36,37,38,39,40,41,42]},
}

def _landmark_axis_error_audit_140(world_np,target_df,rows):
    target={r.joint:np.array([r.x,r.y,r.z],float) for r in target_df.itertuples()}
    out=[]
    core={'Hip','LHip','RHip','Neck','LShoulder','RShoulder','Head'}
    for tname,jidx,sname in rows:
        if tname not in target: continue
        p=np.asarray(world_np[int(jidx)],float); t=target[tname]; d=p-t
        exy=float(np.linalg.norm(d[:2])); ez=float(abs(d[2])); e3=float(np.linalg.norm(d))
        out.append({'landmark':tname,'skel_joint':sname,'skel_index':int(jidx),'region':'core' if tname in core else 'periferico',
                    'error_x':float(d[0]),'error_y':float(d[1]),'error_z_signed':float(d[2]),
                    'error_xy':exy,'error_z_abs':ez,'error_3d':e3})
    out.sort(key=lambda r:r['error_3d'],reverse=True)
    for k,r in enumerate(out,1): r['rank_3d']=k
    return out

def _rmse_xy_subset_140(world_np,target_df,rows,names):
    names=set(names); target={r.joint:np.array([r.x,r.y,r.z],float) for r in target_df.itertuples()}; vals=[]
    for tname,jidx,_ in rows:
        if tname in names and tname in target:
            d=np.asarray(world_np[int(jidx),:2],float)-target[tname][:2]; vals.append(float(np.dot(d,d)))
    return float(np.sqrt(np.mean(vals))) if vals else float('nan')

def _peripheral_monotonic_refinement_140(torch,model,target_df,rows,betas,zero_trans,unified,pose_init,iterations=42):
    pose=torch.tensor(np.asarray(pose_init,np.float32),dtype=torch.float32).reshape(1,-1)
    axis_map=np.asarray(unified['axis_map'],dtype=np.float32); rotvec=torch.tensor(np.asarray(unified['rotvec'],np.float32),dtype=torch.float32)
    scale=torch.tensor(float(unified['scale']),dtype=torch.float32); trans=torch.tensor(np.asarray(unified['trans'],np.float32),dtype=torch.float32)
    core=['Hip','LHip','RHip','Neck','LShoulder','RShoulder','Head']; hist=[]
    def world_eval(pp):
        with torch.no_grad(): w,_,_=_direct_world_130(torch,model,pp,betas,zero_trans,rotvec,scale,trans,axis_map=axis_map)
        wn=w.cpu().numpy(); return wn,_rmse_xy_rows_136(wn,target_df,rows),_rmse_xy_subset_140(wn,target_df,rows,core)
    for group,spec in _PERIPHERAL_GROUPS_140.items():
        ids=[i for i in spec['q'] if i < pose.shape[1]]
        before=pose.detach().clone(); bw,bg,bc=world_eval(before); bl=_rmse_xy_subset_140(bw,target_df,rows,spec['landmarks'])
        cand=before.clone().requires_grad_(True); opt=torch.optim.Adam([cand],lr=0.006)
        for _ in range(int(iterations)):
            opt.zero_grad(set_to_none=True)
            world,_,_=_direct_world_130(torch,model,cand,betas,zero_trans,rotvec,scale,trans,axis_map=axis_map)
            target={r.joint:torch.tensor([r.x,r.y,r.z],dtype=world.dtype) for r in target_df.itertuples()}
            terms=[]
            for tname,jidx,_ in rows:
                if tname in spec['landmarks'] and tname in target:
                    terms.append(torch.sum((world[int(jidx),:2]-target[tname][:2])**2))
            if not terms: break
            local=torch.stack(terms).mean()
            _,idx_all,tgt_all,_=_target_tensor_for_rows(torch,target_df,rows,None); pred=world.index_select(0,idx_all)
            global_xy=torch.mean(torch.sum((pred[:,:2]-tgt_all[:,:2])**2,dim=1))
            reg=0.03*torch.mean((cand[:,ids]-before[:,ids])**2)
            loss=local+0.18*global_xy+reg; loss.backward()
            if cand.grad is not None:
                mask=torch.zeros_like(cand.grad); mask[:,ids]=1.0; cand.grad.mul_(mask)
            torch.nn.utils.clip_grad_norm_([cand],1.5); opt.step(); _clamp_direct_pose_130(torch,cand)
        cand=cand.detach(); aw,ag,ac=world_eval(cand); al=_rmse_xy_subset_140(aw,target_df,rows,spec['landmarks'])
        accepted=bool(np.isfinite(al) and al <= bl-1e-5 and ag <= bg+1e-5 and (not np.isfinite(bc) or not np.isfinite(ac) or ac <= bc+0.004))
        if accepted: pose=cand
        hist.append({'stage':group,'accepted':accepted,'active_q':ids,'landmarks':','.join(spec['landmarks']),
                     'group_rmse_xy_before':float(bl),'group_rmse_xy_after':float(al),
                     'global_rmse_xy_before':float(bg),'global_rmse_xy_after':float(ag),
                     'core_rmse_xy_before':float(bc),'core_rmse_xy_after':float(ac)})
    fw,fg,fc=world_eval(pose)
    return pose.cpu().numpy()[0].astype(np.float32),{'version':'110.3.13','history':hist,'rmse_xy_final':float(fg),'core_rmse_xy_final':float(fc),
            'strategy':'distal monotonic refinement; rollback unless peripheral improves without worsening global/core'}

def _fit_one_skel_frame(torch, model, target_df, max_iter=120, empirical_profile=None, acquisition_mode="monocular_depth_safe"):
    rows,joint_names=_resolve_skel_correspondence(model,target_df)
    if len(rows)<8: raise RuntimeError(f'Sólo se pudieron resolver {len(rows)} correspondencias SKEL↔XYZ; se requieren al menos 8.')
    _,idx,tgt,_=_target_tensor_for_rows(torch,target_df,rows,None)
    betas=torch.zeros((1,int(model.num_betas)),dtype=torch.float32); zero_trans=torch.zeros((1,3),dtype=torch.float32)
    pose0=torch.zeros((1,int(model.num_q_params)),dtype=torch.float32)
    with torch.no_grad():
        J0=model(pose0,betas,zero_trans,skelmesh=False).joints[0]
        # V110.3.13: resolver primero la convención espacial (permutación/signos/handedness).
        coord=_coordinate_system_calibration_134(model,J0.cpu().numpy(),target_df,rows)
        axis_map=np.asarray(coord.get('axis_map',np.eye(3)),dtype=np.float32)
        rv_seed=np.asarray(coord.get('rotvec',np.zeros(3)),dtype=np.float32)
        rv0=torch.tensor(rv_seed,dtype=torch.float32)
        s0=torch.tensor(float(coord.get('scale',1.0)),dtype=torch.float32)
        t0=torch.tensor(np.asarray(coord.get('trans',[0,0,0]),dtype=np.float32),dtype=torch.float32)
        unified=_make_unified_transform_136(coord)
        world0=_apply_unified_torch_136(torch,J0,unified)
        pred0=world0.index_select(0,idx)
        before=pred0.cpu().numpy(); err_before=np.linalg.norm(before-tgt.cpu().numpy(),axis=1)
        frame_invariance=_unified_frame_invariance_136(J0.cpu().numpy(),target_df,rows,unified,tol=1e-5)
        if not frame_invariance.get('ok',False):
            raise RuntimeError(f"Unified Coordinate Frame inconsistente antes de calibrar DoF: ΔRMSE={frame_invariance.get('delta_rmse_xy')} maxΔXYZ={frame_invariance.get('max_coordinate_delta')}")
    # V110.3.13: ground truth por forward real de SKEL. No se decide la dependencia de q por su nombre.
    fk_gt=_fk_ground_truth_calibration_138(torch,model,betas,zero_trans,unified,delta=0.06)
    skel24_sanity=_official_skel24_sanity_140(fk_gt)
    if not skel24_sanity.get('ok',False):
        bad=[t for t in skel24_sanity.get('tests',[]) if not t.get('ok')]
        raise RuntimeError(f'Official SKEL24 joint-map sanity FAIL: {bad}')
    safe_q=[q for q in fk_gt.get('usable_q',[]) if q in _active_for_target_133(target_df)]
    if len(safe_q)<6:
        safe_q=list(_active_for_target_133(target_df))
    # Reusar el Jacobiano empírico ground-truth en la semilla lineal.
    dof_calib={'version':'110.3.13','delta_rad':fk_gt.get('delta_rad',0.06),'rows':fk_gt.get('rows',[]),'jac_world':fk_gt.get('jac_world',{}),
               'safe_q':safe_q,'strategy':fk_gt.get('strategy')}
    J0w=_apply_unified_np_136(J0.cpu().numpy(),unified)
    pose_seed_linear,linear_seed=_linear_seed_from_dof_calibration_135(target_df,rows,dof_calib,J0w,safe_q)
    pose_seed,hierarchical_ik=_monotonic_hierarchical_ik_138(torch,model,target_df,rows,betas,zero_trans,unified,pose_seed_linear,fk_gt.get('q_by_group',{}),iterations_per_stage=48)
    neutral_delta=abs(float(linear_seed.get('rmse_xy_neutral',np.nan))-float(frame_invariance.get('rmse_xy_reference',np.nan)))
    unified_gate={'ok':bool(np.isfinite(neutral_delta) and neutral_delta<=1e-5),'delta_rmse_xy_modules':float(neutral_delta),
                  'coordinate_rmse_xy_all15':float(frame_invariance.get('rmse_xy_reference',np.nan)),
                  'local_dof_rmse_xy_all15':float(linear_seed.get('rmse_xy_neutral',np.nan)),'tolerance':1e-5}
    if not unified_gate['ok']:
        raise RuntimeError(f"Unified Coordinate Frame roto entre módulos: RMSE coordinate={unified_gate['coordinate_rmse_xy_all15']:.6f} vs LocalDoF={unified_gate['local_dof_rmse_xy_all15']:.6f}")
    # Estado estocástico de rescate: parte de la semilla anatómica medida, no de q=0.
    automata=_skel_anatomical_automata_133(torch,model,target_df,rows,betas,zero_trans,rv_seed,float(s0.cpu()),idx,tgt,axis_map=axis_map,active_q_override=safe_q,initial_pose=pose_seed,freeze_global=True,fixed_trans=t0.cpu().numpy())
    # V110.3.13: la búsqueda sólo cambia q. La similitud global queda INMUTABLE.
    automata['rotvec']=rv_seed.astype(float).tolist(); automata['scale']=float(s0.cpu()); automata['trans']=t0.cpu().numpy().astype(float).tolist(); automata['global_transform_frozen']=True
    pose=torch.tensor(automata['pose'],dtype=torch.float32).reshape(1,-1)
    rotvec=torch.tensor(rv_seed,dtype=torch.float32)
    scale=torch.tensor(float(s0.cpu()),dtype=torch.float32)
    trans=t0.detach().clone()
    active_q=list(automata['active_q'])
    # Gradiente sólo como refinador de la cuenca anatómica encontrada por CMA-ES.
    _direct_refine_130(torch,model,pose,rotvec,trans,scale,target_df,rows,betas,zero_trans,
                       iterations=min(int(max_iter),75),lr=0.006,reference_pose=pose.detach().clone(),temporal=False,
                       active_override=active_q,axis_map=axis_map,freeze_global=True)
    # V110.3.13: pase distal conservador sobre la base V110.3.9 ya válida.
    peripheral_pose,peripheral_refinement=_peripheral_monotonic_refinement_140(torch,model,target_df,rows,betas,zero_trans,unified,pose.detach().cpu().numpy()[0],iterations=40)
    pose=torch.tensor(peripheral_pose,dtype=torch.float32).reshape(1,-1)
    with torch.no_grad():
        world,J,R=_direct_world_130(torch,model,pose,betas,zero_trans,rotvec,scale,trans,axis_map=axis_map)
        pred=world.index_select(0,idx); after=pred.cpu().numpy(); tgt_arr=tgt.cpu().numpy(); err_after=np.linalg.norm(after-tgt_arr,axis=1)
    audit=_direct_anatomical_audit_130(model,world.cpu().numpy(),pose.cpu().numpy()[0],target_df)
    diff=after-tgt_arr
    rmse_xy=float(np.sqrt(np.mean(np.sum(diff[:,:2]**2,axis=1))))
    rmse_z=float(np.sqrt(np.mean(diff[:,2]**2)))
    landmark_error_audit=_landmark_axis_error_audit_140(world.cpu().numpy(),target_df,rows)
    depth_weight=float(target_df.attrs.get('depth_weight',1.0)) if hasattr(target_df,'attrs') else 1.0
    frame1_gate=_acquisition_mode_gate_142(
        rmse_xy,rmse_z,audit,acquisition_mode,
        sanity_ok=bool(skel24_sanity.get('ok',False)),
        unified_ok=bool(unified_gate.get('ok',False) and frame_invariance.get('ok',False)),
        xy_limit=0.38)
    return {'rows':rows,'joint_names':joint_names,'pose':pose.cpu().numpy()[0],'rotvec':rotvec.cpu().numpy(),
            'scale':float(scale.cpu()),'trans':trans.cpu().numpy(),'target':tgt_arr,'before':before,'after':after,
            'all_after':world.cpu().numpy(),'rmse_before':float(np.sqrt(np.mean(err_before**2))),
            'rmse_after':float(np.sqrt(np.mean(err_after**2))),'rmse_xy':rmse_xy,'rmse_z':rmse_z,
            'errors_before':err_before,'errors_after':err_after,
            'history':automata.get('history',[]),'automata':automata,'retargeting_mode':'official_skel24_peripheral_refinement',
            'selected_q':active_q,'active_q_names':[_SKEL_Q_NAMES_130[i] for i in active_q],
            'chain_dof_map':{k:[_SKEL_Q_NAMES_130[i] for i in v if i in active_q] for k,v in _DIRECT_GROUPS_130.items()},
            'chain_mapping':[], 'bone_vector_audit':audit,'anatomical_audit':audit,
            'coordinate_frame_seed_rotvec':rv_seed.tolist(),'axis_map':axis_map.astype(float).tolist(),'coordinate_system_calibration':_json_safe_134(coord),'unified_coordinate_transform':_json_safe_134(unified),'unified_frame_invariance':_json_safe_134(frame_invariance),'unified_module_gate':_json_safe_134(unified_gate),'dof_local_calibration':_json_safe_134({k:v for k,v in dof_calib.items() if k!='jac_world'}),'fk_ground_truth':_json_safe_134({k:v for k,v in fk_gt.items() if k!='jac_world'}),'skel24_joint_map_sanity':_json_safe_134(skel24_sanity),'linear_dof_seed':_json_safe_134(linear_seed),'hierarchical_ik':_json_safe_134(hierarchical_ik),'peripheral_refinement':_json_safe_134(peripheral_refinement),'landmark_error_audit':_json_safe_134(landmark_error_audit),'frame1_gate':_json_safe_134(frame1_gate),'depth_weight':float(depth_weight),'direct_model_metadata':_direct_model_metadata_130(model)}



def _safe_unit_112(v):
    v=np.asarray(v,dtype=np.float64).reshape(3)
    n=float(np.linalg.norm(v))
    return v/max(n,1e-9)


def _landmark_map_112(df):
    try:
        return {str(r.joint):np.asarray([r.x,r.y,r.z],dtype=np.float64) for r in df.itertuples()}
    except Exception:
        return {}



def _lower_segment_calibration_112(torch,model,reference_pose,rotvec,trans,scale,betas,zero_trans,axis_map):
    """Precalcula una sola vez la sensibilidad de orientación del muslo a q de cadera."""
    out={}
    names=_skel_forward_joint_names_140(model); nm={_simple_name(n):i for i,n in enumerate(names)}
    pref=reference_pose.detach().clone(); eps=0.025
    with torch.no_grad(): W0,_,_=_direct_world_130(torch,model,pref,betas,zero_trans,rotvec,scale,trans,axis_map=axis_map)
    W0=W0.detach().cpu().numpy().astype(np.float64)
    for side,hq,jns in [('R',[3,4,5],('femur_r','tibia_r')),('L',[10,11,12],('femur_l','tibia_l'))]:
        jh=nm.get(_simple_name(jns[0])); jk=nm.get(_simple_name(jns[1]))
        if jh is None or jk is None: continue
        J=np.zeros((3,len(hq)),dtype=np.float64)
        for col,qi in enumerate(hq):
            pp=pref.detach().clone(); pm=pref.detach().clone()
            with torch.no_grad():
                pp[0,qi]+=eps; pm[0,qi]-=eps; _clamp_direct_pose_130(torch,pp); _clamp_direct_pose_130(torch,pm)
                Wp,_,_=_direct_world_130(torch,model,pp,betas,zero_trans,rotvec,scale,trans,axis_map=axis_map)
                Wm,_,_=_direct_world_130(torch,model,pm,betas,zero_trans,rotvec,scale,trans,axis_map=axis_map)
            wpa=Wp.detach().cpu().numpy(); wma=Wm.detach().cpu().numpy()
            vp=_safe_unit_112(wpa[jk]-wpa[jh]); vm=_safe_unit_112(wma[jk]-wma[jh])
            J[:,col]=(vp-vm)/(2.0*eps)
        out[side]={'J':J,'skel_ref_thigh':_safe_unit_112(W0[jk]-W0[jh]),'hq':hq}
    return out


def _segment_relative_lower_seed_112(torch,model,pose,rotvec,trans,scale,target_df,reference_target_df,betas,zero_trans,reference_pose,axis_map,calibration=None):
    """V110.3.20.0 · Retargeting segmentario absoluto respecto al frame semilla.

    La cadera no se obtiene por acumulación frame-a-frame. Para cada lado se compara
    la orientación Hip→Knee del frame actual con el frame de referencia y se proyecta
    esa variación sobre la sensibilidad REAL de q de cadera del modelo SKEL alrededor
    de la pose de referencia. La rodilla se fija con el ángulo geométrico Hip-Knee-Ankle.
    El resultado se aplica directamente y no exige disminuir el error cartesiano absoluto.
    """
    out={'strategy':'V110.3.20.0 segment-relative hip + geometric knee','accepted':False,'chains':{}}
    try:
        cur=_landmark_map_112(target_df); ref=_landmark_map_112(reference_target_df)
        chains=[
            ('R',[3,4,5],6,('RHip','RKnee','RAnkle')),
            ('L',[10,11,12],13,('LHip','LKnee','LAnkle')),
        ]
        pref=reference_pose.detach().clone()
        calib=calibration or _lower_segment_calibration_112(torch,model,pref,rotvec,trans,scale,betas,zero_trans,axis_map)
        any_ok=False
        for side,hq,kq,lms in chains:
            h,k,a=lms
            if not all(x in cur and x in ref for x in (h,k,a)):
                out['chains'][side]={'accepted':False,'reason':'landmarks insuficientes'}; continue
            cc=calib.get(side)
            if not cc:
                out['chains'][side]={'accepted':False,'reason':'calibración SKEL de cadera ausente'}; continue
            tref=_safe_unit_112(ref[k]-ref[h]); tcur=_safe_unit_112(cur[k]-cur[h]); dt=tcur-tref
            J=np.asarray(cc['J'],dtype=np.float64); sk0=np.asarray(cc['skel_ref_thigh'],dtype=np.float64)
            try:
                dq=np.linalg.solve(J.T@J+0.015*np.eye(len(hq)),J.T@dt)
            except Exception:
                dq=np.linalg.lstsq(J,dt,rcond=1e-5)[0]
            dq=np.clip(dq,-0.75,0.75)
            with torch.no_grad():
                for col,qi in enumerate(hq): pose[0,qi]=pref[0,qi]+float(dq[col])
                kval=_target_knee_flexion_111(target_df,side)
                if kval is not None and np.isfinite(kval): pose[0,kq]=float(kval)
                _clamp_direct_pose_130(torch,pose)
            out['chains'][side]={'accepted':True,'target_thigh_delta_norm':float(np.linalg.norm(dt)),
                'hip_delta_q':[float(x) for x in dq], 'knee_rad':None if kval is None else float(kval),
                'skel_ref_thigh':sk0.tolist()}
            any_ok=True
        out['accepted']=any_ok
        return pose,out
    except Exception as exc:
        out['error']=f'{type(exc).__name__}: {exc}'; out['non_blocking']=True
        return pose,out


def _smooth_lower_q_sequence_112(torch,model,sequence,betas,zero_trans,axis_map,scale):
    """Suaviza sólo q inferiores y reconstruye joints. Filtro FIR simétrico, sin SciPy."""
    ok_idx=[i for i,f in enumerate(sequence) if f.get('status')=='ok' and f.get('pose') is not None]
    qids=[3,4,5,6,10,11,12,13]
    if len(ok_idx)<5: return {'applied':False,'reason':'menos de 5 frames'}
    P=np.asarray([sequence[i]['pose'] for i in ok_idx],dtype=np.float64)
    raw=P[:,qids].copy(); sm=raw.copy(); kernel=np.asarray([1,2,3,2,1],dtype=np.float64)/9.0
    # Two conservative zero-phase passes. Edge padding avoids endpoint collapse.
    for _ in range(2):
        pad=np.pad(sm,((2,2),(0,0)),mode='edge')
        sm=np.vstack([np.convolve(pad[:,j],kernel,mode='valid') for j in range(sm.shape[1])]).T
    # Keep frame-1 exactly fixed; blend to avoid over-smoothing amplitude.
    sm=0.72*sm+0.28*raw; sm[0]=raw[0]
    for r,i in enumerate(ok_idx):
        fr=sequence[i]; pp=np.asarray(fr['pose'],dtype=np.float32).copy(); pp[qids]=sm[r].astype(np.float32)
        pt=torch.tensor(pp,dtype=torch.float32).reshape(1,-1); _clamp_direct_pose_130(torch,pt)
        rv=torch.tensor(fr.get('rotvec',[0,0,0]),dtype=torch.float32); tr=torch.tensor(fr.get('trans',[0,0,0]),dtype=torch.float32)
        with torch.no_grad(): W,_,_=_direct_world_130(torch,model,pt,betas,zero_trans,rv,scale,tr,axis_map=axis_map)
        fr['pose']=pt.cpu().numpy()[0].astype(float).tolist(); fr['joints']=W.cpu().numpy().astype(float).tolist()
        fr['lower_limb_q']={_SKEL_Q_NAMES_130[q]:float(pt[0,q].cpu()) for q in [3,4,5,6,7,10,11,12,13,14]}
    return {'applied':True,'method':'2-pass symmetric FIR [1,2,3,2,1]/9 + 28% raw blend',
            'qids':qids,'raw_ranges':np.ptp(raw,axis=0).astype(float).tolist(),'smooth_ranges':np.ptp(sm,axis=0).astype(float).tolist()}

# V110.3.20.0 · FULL-BODY SEGMENT-DRIVEN RETARGETING
# Continúa la vía segmentaria de V110.3.19.12, pero corrige primero la postura
# de referencia y anima de forma explícita piernas y brazos desde la misma fuente.

def _target_flexion_200(target_df, a, b, c):
    """Flexión geométrica (0 = extensión) de tres landmarks."""
    m=_landmark_map_112(target_df)
    if not all(k in m for k in (a,b,c)): return None
    u=np.asarray(m[a]-m[b],dtype=np.float64); v=np.asarray(m[c]-m[b],dtype=np.float64)
    nu=float(np.linalg.norm(u)); nv=float(np.linalg.norm(v))
    if nu<1e-9 or nv<1e-9: return None
    ang=float(np.arccos(np.clip(np.dot(u,v)/(nu*nv),-1.0,1.0)))
    return float(np.clip(np.pi-ang,0.0,2.20))


def _segment_specs_200():
    return {
      'leg_R': {'q':[3,4,5,6,7], 'blocks':[([3,4,5],'femur_r','tibia_r','RHip','RKnee'),([7],'talus_r','toes_r','RAnkle','RBigToe')],
                'segments':[('femur_r','tibia_r','RHip','RKnee'),('tibia_r','talus_r','RKnee','RAnkle'),('talus_r','toes_r','RAnkle','RBigToe')],
                'hinge':(6,'RHip','RKnee','RAnkle')},
      'leg_L': {'q':[10,11,12,13,14], 'blocks':[([10,11,12],'femur_l','tibia_l','LHip','LKnee'),([14],'talus_l','toes_l','LAnkle','LBigToe')],
                'segments':[('femur_l','tibia_l','LHip','LKnee'),('tibia_l','talus_l','LKnee','LAnkle'),('talus_l','toes_l','LAnkle','LBigToe')],
                'hinge':(13,'LHip','LKnee','LAnkle')},
      'arm_R': {'q':[26,27,28,29,30,31,32],
                'blocks':[([26,27,28],'thorax','humerus_r','Neck','RShoulder'),([29,30,31],'humerus_r','ulna_r','RShoulder','RElbow')],
                'segments':[('thorax','humerus_r','Neck','RShoulder'),('humerus_r','ulna_r','RShoulder','RElbow'),('ulna_r','hand_r','RElbow','RWrist')],
                'hinge':(32,'RShoulder','RElbow','RWrist')},
      'arm_L': {'q':[36,37,38,39,40,41,42],
                'blocks':[([36,37,38],'thorax','humerus_l','Neck','LShoulder'),([39,40,41],'humerus_l','ulna_l','LShoulder','LElbow')],
                'segments':[('thorax','humerus_l','Neck','LShoulder'),('humerus_l','ulna_l','LShoulder','LElbow'),('ulna_l','hand_l','LElbow','LWrist')],
                'hinge':(42,'LShoulder','LElbow','LWrist')},
    }


def _frame1_segment_posture_fit_200(torch,model,pose_init,target_df,betas,zero_trans,rotvec,scale,trans,axis_map,iterations=88):
    """Corrige la POSTURA BASE usando direcciones segmentarias, no error XYZ absoluto.

    Es deliberadamente un ajuste de orientación: conserva pelvis/tronco/global de la
    solución validada y recoloca muslos, tibias y brazos para que el frame de referencia
    tenga la misma geometría direccional que la marcha fuente.
    """
    specs=_segment_specs_200(); pose=pose_init.detach().clone()
    names=_skel_forward_joint_names_140(model); nm={_simple_name(n):i for i,n in enumerate(names)}
    tmap={str(r.joint):np.asarray([r.x,r.y,r.z],dtype=np.float64) for r in target_df.itertuples()}
    report={'strategy':'orientation-only frame-reference posture alignment','groups':{},'accepted':False}
    # Semillas geométricas de bisagras: evita que rodilla/codo partan de una postura arbitraria.
    with torch.no_grad():
        for g,sp in specs.items():
            qi,a,b,c=sp['hinge']; val=_target_flexion_200(target_df,a,b,c)
            if val is not None and np.isfinite(val): pose[0,qi]=float(val)
        _clamp_direct_pose_130(torch,pose)
    # Ajuste por cadena. Sólo se optimizan los q de esa cadena; no se mueve global/pelvis/tronco.
    for g,sp in specs.items():
        segs=[]
        for ja,jb,ta,tb in sp['segments']:
            ia=nm.get(_simple_name(ja)); ib=nm.get(_simple_name(jb))
            if ia is None or ib is None or ta not in tmap or tb not in tmap: continue
            tv=_safe_unit_112(tmap[tb]-tmap[ta]); segs.append((ia,ib,tv,ta,tb))
        if not segs:
            report['groups'][g]={'accepted':False,'reason':'segmentos insuficientes'}; continue
        base=pose.detach().clone(); cand=base.clone().requires_grad_(True); opt=torch.optim.Adam([cand],lr=0.018)
        hist=[]
        for it in range(int(iterations)):
            opt.zero_grad(set_to_none=True)
            W,_,_=_direct_world_130(torch,model,cand,betas,zero_trans,rotvec,scale,trans,axis_map=axis_map)
            terms=[]
            for ia,ib,tv,_,_ in segs:
                sv=W[int(ib)]-W[int(ia)]; sv=sv/(torch.linalg.norm(sv)+1e-8)
                tt=torch.tensor(tv,dtype=W.dtype,device=W.device)
                terms.append(1.0-torch.clamp(torch.sum(sv*tt),-1.0,1.0))
            orient=torch.stack(terms).mean()
            qids=sp['q']; reg=0.0025*torch.mean((cand[:,qids]-base[:,qids])**2)
            loss=orient+reg; loss.backward()
            if cand.grad is not None:
                mask=torch.zeros_like(cand.grad); mask[:,qids]=1.0; cand.grad.mul_(mask)
            torch.nn.utils.clip_grad_norm_([cand],1.5); opt.step(); _clamp_direct_pose_130(torch,cand)
            if it in (0,int(iterations)-1): hist.append(float(loss.detach().cpu()))
        pose=cand.detach();
        with torch.no_grad():
            qi,a,b,c=sp['hinge']; val=_target_flexion_200(target_df,a,b,c)
            if val is not None and np.isfinite(val): pose[0,qi]=float(val)
            _clamp_direct_pose_130(torch,pose)
            W,_,_=_direct_world_130(torch,model,pose,betas,zero_trans,rotvec,scale,trans,axis_map=axis_map)
        angles=[]
        for ia,ib,tv,ta,tb in segs:
            sv=_safe_unit_112((W[int(ib)]-W[int(ia)]).detach().cpu().numpy())
            angles.append(float(np.degrees(np.arccos(np.clip(np.dot(sv,tv),-1.0,1.0)))))
        report['groups'][g]={'accepted':True,'qids':sp['q'],'mean_segment_angle_deg':float(np.mean(angles)) if angles else None,
                             'max_segment_angle_deg':float(np.max(angles)) if angles else None,'loss_edges':hist}
        report['accepted']=True
    report['pose_after']={_SKEL_Q_NAMES_130[i]:float(pose[0,i].cpu()) for i in [3,4,5,6,7,10,11,12,13,14,26,27,28,29,30,31,32,36,37,38,39,40,41,42]}
    return pose,report


def _segment_temporal_calibration_200(torch,model,reference_pose,rotvec,trans,scale,betas,zero_trans,axis_map):
    """Jacobianos locales de DIRECCIÓN para piernas y brazos en la postura base corregida."""
    names=_skel_forward_joint_names_140(model); nm={_simple_name(n):i for i,n in enumerate(names)}
    pref=reference_pose.detach().clone(); eps=0.025; out={}
    with torch.no_grad(): W0,_,_=_direct_world_130(torch,model,pref,betas,zero_trans,rotvec,scale,trans,axis_map=axis_map)
    W0=W0.detach().cpu().numpy().astype(np.float64)
    for g,sp in _segment_specs_200().items():
        blocks=[]
        for qids,ja,jb,ta,tb in sp['blocks']:
            ia=nm.get(_simple_name(ja)); ib=nm.get(_simple_name(jb))
            if ia is None or ib is None: continue
            J=np.zeros((3,len(qids)),dtype=np.float64)
            for col,qi in enumerate(qids):
                pp=pref.clone(); pm=pref.clone()
                with torch.no_grad():
                    pp[0,qi]+=eps; pm[0,qi]-=eps; _clamp_direct_pose_130(torch,pp); _clamp_direct_pose_130(torch,pm)
                    Wp,_,_=_direct_world_130(torch,model,pp,betas,zero_trans,rotvec,scale,trans,axis_map=axis_map)
                    Wm,_,_=_direct_world_130(torch,model,pm,betas,zero_trans,rotvec,scale,trans,axis_map=axis_map)
                vp=_safe_unit_112(Wp.detach().cpu().numpy()[ib]-Wp.detach().cpu().numpy()[ia])
                vm=_safe_unit_112(Wm.detach().cpu().numpy()[ib]-Wm.detach().cpu().numpy()[ia])
                J[:,col]=(vp-vm)/(2.0*eps)
            blocks.append({'qids':list(qids),'J':J,'skel_ref':_safe_unit_112(W0[ib]-W0[ia]),'target_pair':(ta,tb),'joint_pair':(ja,jb),
                           'rank':int(np.linalg.matrix_rank(J,tol=1e-5)),'sensitivity_norm':float(np.linalg.norm(J))})
        out[g]=blocks
    return out




def _angle_between_203(a,b):
    a=np.asarray(a,dtype=np.float64); b=np.asarray(b,dtype=np.float64)
    na=float(np.linalg.norm(a)); nb=float(np.linalg.norm(b))
    if na<1e-9 or nb<1e-9: return None
    return float(np.arccos(np.clip(np.dot(a,b)/(na*nb),-1.0,1.0)))


def _target_ankle_angle_203(target_df, side):
    """Ángulo geométrico 3D no métrico rodilla-tobillo-pie para q7/q14.

    Usa Knee-Ankle como vector proximal y Heel→BigToe como eje longitudinal del pie.
    Si Heel no está disponible, usa Ankle→BigToe. Sólo se usa la VARIACIÓN respecto
    al frame de referencia; no se interpreta como dorsiflexión clínica absoluta.
    """
    m=_landmark_map_112(target_df); side=str(side).upper()
    k,a,t=f'{side}Knee',f'{side}Ankle',f'{side}BigToe'; h=f'{side}Heel'
    if not all(x in m for x in (k,a,t)): return None
    shank=np.asarray(m[k]-m[a],dtype=np.float64)
    foot=np.asarray((m[t]-m[h]) if h in m else (m[t]-m[a]),dtype=np.float64)
    return _angle_between_203(shank,foot)


def _skel_ankle_angle_203(W, nm, side):
    """Análogo SKEL del ángulo proximal-pie empleado por el driver distal."""
    side=str(side).upper(); suf='r' if side=='R' else 'l'
    it=nm.get(_simple_name(f'tibia_{suf}')); ia=nm.get(_simple_name(f'talus_{suf}'))
    ih=nm.get(_simple_name(f'calcn_{suf}')); ito=nm.get(_simple_name(f'toes_{suf}'))
    if None in (it,ia,ito): return None
    arr=W.detach().cpu().numpy() if hasattr(W,'detach') else np.asarray(W)
    shank=np.asarray(arr[it]-arr[ia],dtype=np.float64)
    foot=np.asarray((arr[ito]-arr[ih]) if ih is not None else (arr[ito]-arr[ia]),dtype=np.float64)
    return _angle_between_203(shank,foot)


def _ankle_calibration_203(torch,model,reference_pose,rotvec,trans,scale,betas,zero_trans,axis_map):
    """Calibra signo/ganancia local q7/q14 → cambio angular del pie en SKEL."""
    names=_skel_forward_joint_names_140(model); nm={_simple_name(n):i for i,n in enumerate(names)}
    eps=0.035; out={}
    for side,qi in [('R',7),('L',14)]:
        p0=reference_pose.detach().clone(); pp=p0.clone(); pm=p0.clone()
        with torch.no_grad():
            pp[0,qi]+=eps; pm[0,qi]-=eps; _clamp_direct_pose_130(torch,pp); _clamp_direct_pose_130(torch,pm)
            W0,_,_=_direct_world_130(torch,model,p0,betas,zero_trans,rotvec,scale,trans,axis_map=axis_map)
            Wp,_,_=_direct_world_130(torch,model,pp,betas,zero_trans,rotvec,scale,trans,axis_map=axis_map)
            Wm,_,_=_direct_world_130(torch,model,pm,betas,zero_trans,rotvec,scale,trans,axis_map=axis_map)
        a0=_skel_ankle_angle_203(W0,nm,side); ap=_skel_ankle_angle_203(Wp,nm,side); am=_skel_ankle_angle_203(Wm,nm,side)
        slope=None if None in (ap,am) else float((ap-am)/(2.0*eps))
        out[side]={'q_index':qi,'skel_reference_angle_rad':a0,'slope_rad_per_qrad':slope,
                   'usable':bool(slope is not None and np.isfinite(slope) and abs(slope)>0.12)}
    return out


def _apply_distal_ankle_driver_203(torch,pose,target_df,reference_target_df,reference_pose,ankle_calib):
    """Activa q7/q14 desde el 3D estimado distal, con referencia frame-1 y sin deriva."""
    report={'strategy':'estimated3d distal foot relative-angle driver','sides':{},'accepted':False}
    for side,qi in [('R',7),('L',14)]:
        cur=_target_ankle_angle_203(target_df,side); ref=_target_ankle_angle_203(reference_target_df,side)
        cal=(ankle_calib or {}).get(side,{})
        slope=cal.get('slope_rad_per_qrad')
        if cur is None or ref is None:
            report['sides'][side]={'accepted':False,'reason':'Knee/Ankle/BigToe insuficientes'}; continue
        if not cal.get('usable') or slope is None:
            report['sides'][side]={'accepted':False,'reason':'sensibilidad SKEL de tobillo insuficiente','slope':slope}; continue
        delta=float(cur-ref)
        # El cambio angular objetivo se traduce por la sensibilidad real de q7/q14 en SKEL.
        dq=float(np.clip(delta/float(slope),-0.55,0.55))
        with torch.no_grad(): pose[0,qi]=reference_pose[0,qi]+dq
        report['sides'][side]={'accepted':True,'q_index':qi,'target_delta_deg':float(np.degrees(delta)),
                               'slope_rad_per_qrad':float(slope),'delta_q_deg':float(np.degrees(dq)),
                               'source':'Knee-Ankle + Heel→BigToe (fallback Ankle→BigToe)'}
        report['accepted']=True
    with torch.no_grad(): _clamp_direct_pose_130(torch,pose)
    return pose,report

def _segment_relative_fullbody_seed_200(torch,model,pose,target_df,reference_target_df,reference_pose,calibration):
    """Movimiento segmentario coordinado respecto al mismo frame de referencia.

    Piernas y brazos se derivan de la misma secuencia fuente y del mismo frame base.
    Cada bloque escribe q = q_ref + Δq; no hay acumulación frame-a-frame ni deriva.
    """
    cur=_landmark_map_112(target_df); ref=_landmark_map_112(reference_target_df); pref=reference_pose.detach().clone()
    out={'strategy':'frame-reference full-body segment-direction retargeting','groups':{},'accepted':False}
    for g,sp in _segment_specs_200().items():
        gout={'blocks':[],'accepted':False}
        for blk in calibration.get(g,[]):
            ta,tb=blk['target_pair']; qids=blk['qids']
            if ta not in cur or tb not in cur or ta not in ref or tb not in ref:
                gout['blocks'].append({'accepted':False,'reason':'landmarks insuficientes','target_pair':[ta,tb]}); continue
            tref=_safe_unit_112(ref[tb]-ref[ta]); tcur=_safe_unit_112(cur[tb]-cur[ta]); dt=tcur-tref
            J=np.asarray(blk['J'],dtype=np.float64)
            if float(np.linalg.norm(J))<1e-7:
                gout['blocks'].append({'accepted':False,'reason':'sensibilidad SKEL nula','qids':qids}); continue
            lam=0.012
            try: dq=np.linalg.solve(J.T@J+lam*np.eye(len(qids)),J.T@dt)
            except Exception: dq=np.linalg.lstsq(J,dt,rcond=1e-5)[0]
            dq=np.clip(dq,-0.70,0.70)
            with torch.no_grad():
                for c,qi in enumerate(qids): pose[0,qi]=pref[0,qi]+float(dq[c])
            gout['blocks'].append({'accepted':True,'qids':qids,'target_pair':[ta,tb],'delta_direction_norm':float(np.linalg.norm(dt)),
                                   'delta_q':[float(x) for x in dq],'rank':blk.get('rank')})
            gout['accepted']=True; out['accepted']=True
        qi,a,b,c=sp['hinge']; val=_target_flexion_200(target_df,a,b,c)
        if val is not None and np.isfinite(val):
            with torch.no_grad(): pose[0,qi]=float(val)
            gout['hinge_q']=int(qi); gout['hinge_rad']=float(val)
        out['groups'][g]=gout
    with torch.no_grad(): _clamp_direct_pose_130(torch,pose)
    return pose,out


def _refine_fullbody_q_sequence_202(torch,model,sequence,betas,zero_trans,axis_map,scale):
    """V110.3.20.2 · Refinamiento temporal final de q(t) preservando postura y amplitud.

    - filtro simétrico de 7 muestras para reducir jitter sin desplazar fase;
    - mezcla adaptada por familia articular;
    - preservación de >=92% de la excursión original cuando es estable;
    - limitador robusto de salto frame-a-frame, aplicado hacia delante y atrás;
    - el frame de referencia queda ANCLADO exactamente a la postura segmentaria corregida.

    No se suavizan vértices: se refinan los DOF y después se recalcula SKEL.forward().
    """
    ok_idx=[i for i,f in enumerate(sequence) if f.get('status')=='ok' and f.get('pose') is not None]
    qids=[3,4,5,6,7,10,11,12,13,14,26,27,28,29,30,31,32,36,37,38,39,40,41,42]
    if len(ok_idx)<7: return {'applied':False,'reason':'menos de 7 frames'}
    P=np.asarray([sequence[i]['pose'] for i in ok_idx],dtype=np.float64)
    raw=P[:,qids].copy(); n=raw.shape[0]
    kernel=np.asarray([1,2,3,4,3,2,1],dtype=np.float64)/16.0

    # Pesos de filtrado: rodilla/codo algo más suaves; tobillo y rotaciones pequeñas conservan más señal.
    blend_by_q={
      3:.70,4:.62,5:.58,6:.78,7:.62,10:.70,11:.62,12:.58,13:.78,14:.62,
      26:.66,27:.62,28:.58,29:.70,30:.66,31:.62,32:.76,
      36:.66,37:.62,38:.58,39:.70,40:.66,41:.62,42:.76,
    }
    # límites máximos conservadores por frame (radianes). Se combinan con un umbral robusto de la propia señal.
    hard_deg={3:4.5,4:3.5,5:3.0,6:5.0,7:4.0,10:4.5,11:3.5,12:3.0,13:5.0,14:4.0,
              26:4.0,27:3.5,28:3.0,29:4.5,30:4.0,31:4.0,32:5.0,
              36:4.0,37:3.5,38:3.0,39:4.5,40:4.0,41:4.0,42:5.0}

    # padding reflectivo: evita usar el final para alterar el frame 1; luego se restaura el anclaje exacto.
    pad=np.pad(raw,((3,3),(0,0)),mode='reflect')
    filt=np.vstack([np.convolve(pad[:,j],kernel,mode='valid') for j in range(raw.shape[1])]).T
    sm=np.empty_like(raw)
    per_q={}
    for j,qi in enumerate(qids):
        a=float(blend_by_q.get(qi,.68))
        x=a*filt[:,j]+(1.0-a)*raw[:,j]
        # Preservar excursión clínica: el filtrado no debe convertir movimiento real en una pose rígida.
        rr=float(np.ptp(raw[:,j])); xr=float(np.ptp(x))
        if rr>1e-5 and xr>1e-6 and xr < 0.92*rr:
            mu=float(np.mean(x)); x=mu+(x-mu)*(0.92*rr/xr)
        # Alinear la trayectoria a la postura base exacta del frame de referencia.
        x=x+(raw[0,j]-x[0])

        d=np.diff(x)
        med=float(np.median(np.abs(d))) if len(d) else 0.0
        robust=max(np.deg2rad(float(hard_deg.get(qi,4.5))), 3.25*med)
        # limitador bidireccional suave; sólo actúa sobre saltos, no impone una forma de onda.
        y=x.copy()
        for _ in range(2):
            for k in range(1,n):
                delta=y[k]-y[k-1]
                if abs(delta)>robust: y[k]=y[k-1]+np.sign(delta)*robust
            for k in range(n-2,-1,-1):
                delta=y[k]-y[k+1]
                if abs(delta)>robust: y[k]=y[k+1]+np.sign(delta)*robust
            y=y+(raw[0,j]-y[0])

        curvature_corrections=0
        curvature_threshold_deg=0.0

        # V110.3.20.8 SEGMENT: estabilización MUY SUAVE y exclusiva de q14.
        # No impone una velocidad máxima artificial ni copia el guard de la rama 3D estimated.
        # Sólo atenúa picos locales aislados de curvatura temporal, conservando >=99% del ROM
        # de la trayectoria segmentaria ya refinada.
        if qi == 14 and n >= 5:
            y_pre=y.copy()
            dd=np.diff(y_pre, n=2)
            abs_dd=np.abs(dd)
            if len(abs_dd):
                med_dd=float(np.median(abs_dd))
                mad_dd=float(np.median(np.abs(abs_dd-med_dd)))
                robust_curv=med_dd + 3.0*1.4826*mad_dd
                curv_thr=max(np.deg2rad(4.2), robust_curv)
                candidates=(np.where(abs_dd > curv_thr)[0] + 1).tolist()
                # Corrección parcial: sólo 25% hacia la interpolación de vecinos.
                # Equivale a amortiguar el vértice sin aplanar el movimiento.
                for kk in candidates:
                    local_interp=0.5*(y[kk-1]+y[kk+1])
                    y[kk]=0.75*y[kk]+0.25*local_interp
                if candidates:
                    rr_pre=float(np.ptp(y_pre)); rr_post=float(np.ptp(y))
                    if rr_pre>1e-6 and rr_post>1e-6 and rr_post < 0.99*rr_pre:
                        mu=float(np.mean(y))
                        y=mu+(y-mu)*(0.99*rr_pre/rr_post)
                    y=y+(raw[0,j]-y[0])
                curvature_corrections=len(candidates)
                curvature_threshold_deg=float(np.degrees(curv_thr))

        sm[:,j]=y
        per_q[_SKEL_Q_NAMES_130[qi]]={
            'raw_range_deg':float(np.degrees(rr)),
            'refined_range_deg':float(np.degrees(np.ptp(y))),
            'raw_max_step_deg':float(np.degrees(np.max(np.abs(np.diff(raw[:,j]))))) if n>1 else 0.0,
            'refined_max_step_deg':float(np.degrees(np.max(np.abs(np.diff(y))))) if n>1 else 0.0,
            'velocity_cap_deg_per_frame':float(np.degrees(robust)),
            'filter_weight':a,
            'curvature_corrections':int(curvature_corrections),
            'curvature_threshold_deg':float(curvature_threshold_deg),
        }

    for r,i in enumerate(ok_idx):
        fr=sequence[i]; pp=np.asarray(fr['pose'],dtype=np.float32).copy(); pp[qids]=sm[r].astype(np.float32)
        pt=torch.tensor(pp,dtype=torch.float32).reshape(1,-1); _clamp_direct_pose_130(torch,pt)
        rv=torch.tensor(fr.get('rotvec',[0,0,0]),dtype=torch.float32); tr=torch.tensor(fr.get('trans',[0,0,0]),dtype=torch.float32)
        with torch.no_grad(): W,_,_=_direct_world_130(torch,model,pt,betas,zero_trans,rv,scale,tr,axis_map=axis_map)
        fr['pose']=pt.cpu().numpy()[0].astype(float).tolist(); fr['joints']=W.cpu().numpy().astype(float).tolist()
        fr['segment_q']={_SKEL_Q_NAMES_130[q]:float(pt[0,q].cpu()) for q in qids}
    return {'applied':True,
            'method':'7-point symmetric FIR + adaptive blend + >=92% ROM preservation + robust bidirectional step limiter + q14 gentle local-curvature attenuation (25%, >=99% local ROM) + exact frame1 anchor',
            'qids':qids,'per_q':per_q,
            'raw_ranges':np.ptp(raw,axis=0).astype(float).tolist(),'refined_ranges':np.ptp(sm,axis=0).astype(float).tolist()}


def _coordination_audit_202(sequence):
    """Auditoría no bloqueante de coordinación y fluidez de la final candidate."""
    ok=[f for f in sequence if f.get('status')=='ok' and f.get('pose') is not None]
    if len(ok)<5: return {'ok':False,'reason':'secuencia insuficiente'}
    P=np.asarray([f['pose'] for f in ok],dtype=np.float64)
    ids=[3,4,5,6,7,10,11,12,13,14,26,27,28,29,30,31,32,36,37,38,39,40,41,42]
    def corr(a,b):
        a=np.asarray(a,float); b=np.asarray(b,float)
        if np.std(a)<1e-8 or np.std(b)<1e-8: return None
        return float(np.corrcoef(a,b)[0,1])
    qdiag={}
    for qi in ids:
        x=P[:,qi]; d=np.diff(x); dd=np.diff(x,n=2)
        qdiag[_SKEL_Q_NAMES_130[qi]]={
          'range_deg':float(np.degrees(np.ptp(x))),
          'max_step_deg':float(np.degrees(np.max(np.abs(d)))) if len(d) else 0.0,
          'max_accel_proxy_deg_frame2':float(np.degrees(np.max(np.abs(dd)))) if len(dd) else 0.0,
        }
    # Correlaciones descriptivas: no se fuerzan porque el signo depende de la convención SKEL.
    corrs={
      'hip_R_vs_L':corr(P[:,3],P[:,10]),
      'shoulder_swing_R_vs_L':corr(P[:,29],P[:,39]),
      'hip_R_vs_contralateral_shoulder_L':corr(P[:,3],P[:,39]),
      'hip_L_vs_contralateral_shoulder_R':corr(P[:,10],P[:,29]),
      'knee_R_vs_L':corr(P[:,6],P[:,13]),
    }
    active_ranges=[v['range_deg'] for v in qdiag.values()]
    max_steps=[v['max_step_deg'] for v in qdiag.values()]
    return {'ok':True,'q_diagnostics':qdiag,'correlations':corrs,
            'legs_unfrozen':bool(max(qdiag[_SKEL_Q_NAMES_130[i]]['range_deg'] for i in [3,4,5,6,7,10,11,12,13,14])>0.1),
            'arms_unfrozen':bool(max(qdiag[_SKEL_Q_NAMES_130[i]]['range_deg'] for i in [26,27,28,29,30,31,32,36,37,38,39,40,41,42])>0.1),
            'max_active_step_deg':float(max(max_steps)) if max_steps else None,
            'median_active_range_deg':float(np.median(active_ranges)) if active_ranges else None,
            'non_blocking':True}

def _segment_motion_audit_200(sequence):
    ok=[f for f in sequence if f.get('status')=='ok' and f.get('pose') is not None]
    ids=[3,4,5,6,7,10,11,12,13,14,26,27,28,29,30,31,32,36,37,38,39,40,41,42]
    if len(ok)<2: return {'ok':False,'reason':'secuencia insuficiente'}
    P=np.asarray([f['pose'] for f in ok],dtype=np.float64)
    ranges={_SKEL_Q_NAMES_130[i]:float(np.ptp(P[:,i])) for i in ids}
    arm_ids=[26,27,28,29,30,31,32,36,37,38,39,40,41,42]
    leg_ids=[3,4,5,6,7,10,11,12,13,14]
    return {'ok':True,'q_ranges_rad':ranges,'max_leg_range_rad':float(max(ranges[_SKEL_Q_NAMES_130[i]] for i in leg_ids)),
            'max_arm_range_rad':float(max(ranges[_SKEL_Q_NAMES_130[i]] for i in arm_ids)),
            'arms_unfrozen':bool(max(ranges[_SKEL_Q_NAMES_130[i]] for i in arm_ids)>1e-3),
            'legs_unfrozen':bool(max(ranges[_SKEL_Q_NAMES_130[i]] for i in leg_ids)>1e-3)}

def _fit_skel_sequence(torch, model, motion, seed_fit, start_index=0, iterations=16, progress_cb=None, empirical_profile=None, acquisition_mode="monocular_depth_safe"):
    frames=list((motion or {}).get('frames') or [])
    if not frames: raise RuntimeError('No hay frames V104/V107 para propagación temporal.')
    betas=torch.zeros((1,int(model.num_betas)),dtype=torch.float32); zero_trans=torch.zeros((1,3),dtype=torch.float32)
    scale=torch.tensor(float(seed_fit['scale']),dtype=torch.float32); axis_map=np.asarray(seed_fit.get('axis_map',np.eye(3)),dtype=np.float32)
    seed_pose=torch.tensor(seed_fit['pose'],dtype=torch.float32).reshape(1,-1); _clamp_direct_pose_130(torch,seed_pose)
    prev_rot=torch.tensor(seed_fit['rotvec'],dtype=torch.float32); prev_trans=torch.tensor(seed_fit['trans'],dtype=torch.float32)
    lower_profile=_lower_limb_target_profile_147(motion, acquisition_mode)

    # 1) POSTURA BASE CORREGIDA. El frame de referencia deja de heredar la mala alineación
    #    distal del fitting XYZ y se alinea por direcciones anatómicas de segmentos.
    reference_tdf=_target_df_for_frame(frames[int(start_index)],int(start_index))
    reference_pose_fixed,posture_alignment=_frame1_segment_posture_fit_200(
        torch,model,seed_pose,reference_tdf,betas,zero_trans,prev_rot,scale,prev_trans,axis_map,iterations=64)
    prev_pose=reference_pose_fixed.detach().clone()
    temporal_calib=_segment_temporal_calibration_200(torch,model,reference_pose_fixed,prev_rot,prev_trans,scale,betas,zero_trans,axis_map)
    ankle_calib=_ankle_calibration_203(torch,model,reference_pose_fixed,prev_rot,prev_trans,scale,betas,zero_trans,axis_map)

    sequence=[]
    for fi in range(int(start_index),len(frames)):
        tdf=_target_df_for_frame(frames[fi],fi); rows,_=_resolve_skel_correspondence(model,tdf)
        if len(rows)<8:
            sequence.append({'frame':fi+1,'status':'insufficient_landmarks','n_correspondences':len(rows)}); continue
        _,idx,tgt,_=_target_tensor_for_rows(torch,tdf,rows,None)
        pose=prev_pose.clone(); rotvec=prev_rot.clone(); trans=prev_trans.clone()
        seg_temporal={'accepted':False,'reason':'frame de referencia'}
        if fi==int(start_index):
            pose=reference_pose_fixed.detach().clone()
        else:
            # Mantener tronco/cabeza de la solución temporal, pero los miembros se escriben
            # explícitamente después desde el driver segmentario común.
            _direct_refine_130(torch,model,pose,rotvec,trans,scale,tdf,rows,betas,zero_trans,
                               iterations=max(20,int(iterations)),lr=0.006,reference_pose=prev_pose,temporal=True,
                               active_override=_active_for_target_133(tdf),axis_map=axis_map,freeze_global=True)
            before=pose.detach().clone()
            pose,seg_temporal=_segment_relative_fullbody_seed_200(
                torch,model,pose,tdf,reference_tdf,reference_pose_fixed,temporal_calib)
            pose,ankle_temporal=_apply_distal_ankle_driver_203(
                torch,pose,tdf,reference_tdf,reference_pose_fixed,ankle_calib)
            if isinstance(seg_temporal,dict):
                seg_temporal['ankle_distal_203']=ankle_temporal
                ids=[3,4,5,6,7,10,11,12,13,14,26,27,28,29,30,31,32,36,37,38,39,40,41,42]
                seg_temporal['q_before']={_SKEL_Q_NAMES_130[i]:float(before[0,i].cpu()) for i in ids}
                seg_temporal['q_after']={_SKEL_Q_NAMES_130[i]:float(pose[0,i].cpu()) for i in ids}
        with torch.no_grad():
            world,_,_=_direct_world_130(torch,model,pose,betas,zero_trans,rotvec,scale,trans,axis_map=axis_map)
            pred=world.index_select(0,idx); errors=torch.linalg.norm(pred-tgt,dim=1); rmse=float(torch.sqrt(torch.mean(errors*errors)).cpu())
            d=(pred-tgt).cpu().numpy(); rmse_xy=float(np.sqrt(np.mean(np.sum(d[:,:2]**2,axis=1)))); rmse_z=float(np.sqrt(np.mean(d[:,2]**2)))
            audit=_direct_anatomical_audit_130(model,world.cpu().numpy(),pose.cpu().numpy()[0],tdf)
            depth_gate=_acquisition_mode_gate_142(rmse_xy,rmse_z,audit,acquisition_mode,xy_limit=0.38)
            sequence.append({'frame':fi+1,'status':'ok','n_correspondences':len(rows),'rmse':rmse,'rmse_xy':rmse_xy,'rmse_z':rmse_z,
                'anatomical_audit':audit,'depth_safe_validation':depth_gate,'retargeting_mode':'fullbody_segment_frame_reference',
                'active_q_names':[_SKEL_Q_NAMES_130[i] for i in _active_for_target_133(tdf)],
                'chain_dof_map':{k:[_SKEL_Q_NAMES_130[i] for i in v if i in _active_for_target_133(tdf)] for k,v in _DIRECT_GROUPS_130.items()},
                'pose':pose.cpu().numpy()[0].astype(float).tolist(),'rotvec':rotvec.cpu().numpy().astype(float).tolist(),
                'trans':trans.cpu().numpy().astype(float).tolist(),'joints':world.cpu().numpy().astype(float).tolist(),
                'segment_temporal':seg_temporal,
                'lower_limb_q':{_SKEL_Q_NAMES_130[i]:float(pose[0,i].cpu()) for i in [3,4,5,6,7,10,11,12,13,14]},
                'upper_limb_q':{_SKEL_Q_NAMES_130[i]:float(pose[0,i].cpu()) for i in [26,27,28,29,30,31,32,36,37,38,39,40,41,42]}})
            prev_pose=pose.clone(); prev_rot=rotvec.clone(); prev_trans=trans.clone()
        if progress_cb: progress_cb(fi+1,len(frames),rmse)

    smooth=_refine_fullbody_q_sequence_202(torch,model,sequence,betas,zero_trans,axis_map,scale)
    result={'version':'110.3.22.13-segment-gentle-anatomical-layers','retargeting_mode':'fullbody_segment_frame_reference_skel_skeleton_layer','scale':float(scale.cpu()),
            'active_q_names':[_SKEL_Q_NAMES_130[i] for i in _DIRECT_ACTIVE_130], 'selected_q_indices':list(_DIRECT_ACTIVE_130),
            'joint_names':_skel_forward_joint_names_140(model),'acquisition_mode':acquisition_mode,
            'chain_dof_map':{k:[_SKEL_Q_NAMES_130[i] for i in v] for k,v in _DIRECT_GROUPS_130.items()},
            'axis_map':axis_map.astype(float).tolist(),'coordinate_system_calibration':_json_safe_134(seed_fit.get('coordinate_system_calibration',{})),
            'direct_model_metadata':_direct_model_metadata_130(model),'frames':sequence,
            'segment_posture_alignment_200':_json_safe_134(posture_alignment),
            'segment_temporal_calibration_200':_json_safe_134(temporal_calib),
            'ankle_calibration_203':_json_safe_134(ankle_calib),
            'fullbody_refinement_202':_json_safe_134(smooth),
            'segment_motion_audit_200':_json_safe_134(_segment_motion_audit_200(sequence)),
            'coordination_audit_202':_json_safe_134(_coordination_audit_202(sequence)),
            'distal_foot_driver_203':{'required_landmarks':['L/RKnee','L/RAnkle','L/RBigToe','L/RHeel preferred'],
                                      'q_indices':[7,14],
                                      'q_names':[_SKEL_Q_NAMES_130[7],_SKEL_Q_NAMES_130[14]],
                                      'semantics':'relative non-metric 3D angle; not absolute clinical dorsiflexion'}}
    result['lower_limb_target_profile']=_json_safe_134(lower_profile)
    try: result['pose_sequence_integrity']=_pose_sequence_integrity_audit_147(result,lower_profile)
    except Exception as exc: result['pose_sequence_integrity']={'ok':False,'error':f'{type(exc).__name__}: {exc}','non_blocking':True}
    try: result['lower_limb_motion_gate']=_lower_limb_motion_gate_147(result,lower_profile)
    except Exception as exc: result['lower_limb_motion_gate']={'ok':False,'error':f'{type(exc).__name__}: {exc}','non_blocking':True}
    result['lower_limb_motion_audit']=result['lower_limb_motion_gate']
    return result

def _plot_skel_sequence_animation(seq):
    ok=[f for f in seq.get("frames",[]) if f.get("status")=="ok" and f.get("joints")]
    if not ok:
        st.warning("No hay frames SKEL válidos para animar."); return
    try:
        import plotly.graph_objects as go
        names=[str(x) for x in seq.get("joint_names",[])]
        norm={_simple_name(n):i for i,n in enumerate(names)}
        # V110.3.13: conectividad REAL de los 24 joints que devuelve SKEL.forward().joints.
        edge_names=[
            ("pelvis","femur_r"),("femur_r","tibia_r"),("tibia_r","talus_r"),("talus_r","calcn_r"),("calcn_r","toes_r"),
            ("pelvis","femur_l"),("femur_l","tibia_l"),("tibia_l","talus_l"),("talus_l","calcn_l"),("calcn_l","toes_l"),
            ("pelvis","lumbar_body"),("lumbar_body","thorax"),("thorax","head"),
            ("thorax","scapula_r"),("scapula_r","humerus_r"),("humerus_r","ulna_r"),("ulna_r","radius_r"),("radius_r","hand_r"),
            ("thorax","scapula_l"),("scapula_l","humerus_l"),("humerus_l","ulna_l"),("ulna_l","radius_l"),("radius_l","hand_l"),
        ]
        edges=[]
        for a,b in edge_names:
            ia=norm.get(_simple_name(a)); ib=norm.get(_simple_name(b))
            if ia is not None and ib is not None: edges.append((ia,ib))
        all_xyz=np.concatenate([np.asarray(f["joints"],float) for f in ok],axis=0)
        mins=np.nanmin(all_xyz,axis=0); maxs=np.nanmax(all_xyz,axis=0); span=np.maximum(maxs-mins,1e-3); pad=0.08*span
        def traces(fr):
            J=np.asarray(fr["joints"],float)
            xs=[]; ys=[]; zs=[]
            for a,b in edges:
                xs += [J[a,0],J[b,0],None]; ys += [J[a,1],J[b,1],None]; zs += [J[a,2],J[b,2],None]
            return [
                go.Scatter3d(x=xs,y=ys,z=zs,mode="lines",name="SKEL",showlegend=False),
                go.Scatter3d(x=J[:,0],y=J[:,1],z=J[:,2],mode="markers",name="Joints SKEL"),
            ]
        anim_frames=[go.Frame(data=traces(f),name=str(f["frame"])) for f in ok]
        fig=go.Figure(data=traces(ok[0]),frames=anim_frames)
        steps=[dict(method="animate",args=[[str(f["frame"])],{"mode":"immediate","frame":{"duration":60,"redraw":True},"transition":{"duration":0}}],label=str(f["frame"])) for f in ok]
        fig.update_layout(
            height=680,margin=dict(l=0,r=0,t=45,b=0),title="V110.3.13 · SKEL · marcha propagada temporalmente",
            scene=dict(aspectmode="data",xaxis=dict(range=[mins[0]-pad[0],maxs[0]+pad[0]]),yaxis=dict(range=[mins[1]-pad[1],maxs[1]+pad[1]]),zaxis=dict(range=[mins[2]-pad[2],maxs[2]+pad[2]])),
            updatemenus=[dict(type="buttons",showactive=False,buttons=[dict(label="▶ Reproducir",method="animate",args=[None,{"fromcurrent":True,"frame":{"duration":60,"redraw":True},"transition":{"duration":0}}]),dict(label="⏸ Pausa",method="animate",args=[[None],{"mode":"immediate","frame":{"duration":0,"redraw":False}}])])],
            sliders=[dict(active=0,currentvalue={"prefix":"Frame "},steps=steps,pad={"t":35})]
        )
        st.plotly_chart(fig,use_container_width=True)
    except Exception as exc:
        st.warning(f"La secuencia se calculó, pero la animación 3D no pudo mostrarse: {type(exc).__name__}: {exc}")




def _plot_simplified_xyz_animation(motion):
    """Recupera el vídeo/animación 3D simplificado no calibrado V104/V107 como referencia visual."""
    frames=list((motion or {}).get('frames') or [])
    if not frames: return
    try:
        import plotly.graph_objects as go
        edge_names=[('Hip','LHip'),('Hip','RHip'),('LHip','LKnee'),('LKnee','LAnkle'),('RHip','RKnee'),('RKnee','RAnkle'),
                    ('Hip','Neck'),('Neck','Head'),('Neck','LShoulder'),('LShoulder','LElbow'),('LElbow','LWrist'),
                    ('Neck','RShoulder'),('RShoulder','RElbow'),('RElbow','RWrist')]
        payload=[]
        for i,f in enumerate(frames):
            P=_frame_points_from_frame(f)
            names=[n for n in JOINTS if n in P]
            pts=np.asarray([P[n] for n in names],float) if names else np.zeros((0,3))
            ex=[];ey=[];ez=[]
            for a,b in edge_names:
                if a in P and b in P:
                    ex += [P[a][0],P[b][0],None]; ey += [P[a][1],P[b][1],None]; ez += [P[a][2],P[b][2],None]
            traces=[go.Scatter3d(x=ex,y=ey,z=ez,mode='lines',line=dict(width=5),showlegend=False,hoverinfo='skip')]
            traces.append(go.Scatter3d(x=pts[:,0] if len(pts) else [],y=pts[:,1] if len(pts) else [],z=pts[:,2] if len(pts) else [],
                                       mode='markers+text',text=names,textposition='top center',marker=dict(size=5),showlegend=False))
            payload.append((i,traces))
        fig=go.Figure(data=payload[0][1],frames=[go.Frame(data=t,name=str(i+1)) for i,t in payload])
        fig.update_layout(height=600,margin=dict(l=0,r=0,t=45,b=0),title='V110.3.13 · Vídeo 3D simplificado V104/V107 · no calibrado',
            scene=dict(aspectmode='data'),updatemenus=[dict(type='buttons',showactive=False,buttons=[
                dict(label='▶ Reproducir',method='animate',args=[None,dict(frame=dict(duration=60,redraw=True),transition=dict(duration=0),fromcurrent=True)]),
                dict(label='⏸ Pausa',method='animate',args=[[None],dict(frame=dict(duration=0,redraw=False),mode='immediate')])])],
            sliders=[dict(active=0,currentvalue=dict(prefix='Frame '),steps=[dict(method='animate',label=str(i+1),args=[[str(i+1)],dict(mode='immediate',frame=dict(duration=0,redraw=True),transition=dict(duration=0))]) for i in range(len(payload))])])
        st.plotly_chart(fig,use_container_width=True)
    except Exception as exc:
        st.caption(f'Vídeo 3D simplificado no disponible: {exc}')

# V110.3.13 · SKEL MESH WALKER: malla corporal real `skin_verts` sobre la secuencia temporal validada.
def _skin_faces_numpy(model):
    """Recupera la topología fija de la malla corporal SKEL.

    La API oficial de SKEL usa `model.skin_f`; se conservan fallbacks tolerantes
    por si una revisión futura expone el mismo tensor con otro nombre.
    """
    for name in ("skin_f", "skin_faces", "faces_skin", "faces"):
        f=getattr(model,name,None)
        if f is None:
            continue
        try:
            if hasattr(f,"detach"):
                f=f.detach().cpu().numpy()
            else:
                f=np.asarray(f)
            f=np.asarray(f,dtype=np.int32)
            if f.ndim==3 and f.shape[0]==1:
                f=f[0]
            if f.ndim==2 and f.shape[1]>=3:
                return f[:,:3].copy(), name
        except Exception:
            pass
    raise RuntimeError("SKEL ha devuelto skin_verts, pero no se encontró la topología triangular de la piel (esperado: model.skin_f).")

# V110.3.20.0 · SKEL LOWER-LIMB MOTOR ISOLATION TEST
# Prueba deliberadamente independiente del retargeting V104/V107. Si esta secuencia
# mueve las piernas, quedan validados SKEL.forward(), skin_verts y el reproductor.
def _skel_lower_limb_motor_test_111(torch, model, seed_fit, gender='unknown', n_frames=32):
    n=max(16,int(n_frames))
    betas=torch.zeros((1,int(model.num_betas)),dtype=torch.float32,device='cpu')
    zero_trans=torch.zeros((1,3),dtype=torch.float32,device='cpu')
    scale=float(seed_fit.get('scale',1.0))
    axis_map=np.asarray(seed_fit.get('axis_map',np.eye(3)),dtype=np.float32)
    rot=np.asarray(seed_fit.get('rotvec',[0,0,0]),dtype=np.float32)
    trans=np.asarray(seed_fit.get('trans',[0,0,0]),dtype=np.float32)
    base=np.asarray(seed_fit.get('pose',np.zeros((int(model.num_q_params),),np.float32)),dtype=np.float32).reshape(-1)
    frames=[]
    # amplitudes conservadoras, suficientes para demostrar movimiento real de la cadena
    for k in range(n):
        ph=2.0*np.pi*float(k)/float(n)
        q=base.copy()
        # marcha sintética alternante: cadera, rodilla y tobillo de ambas piernas
        q[3] = float(base[3] + 0.38*np.sin(ph))
        q[6] = float(base[6] + 0.58*(0.5+0.5*np.sin(ph+0.55)))
        q[7] = float(base[7] - 0.16*np.sin(ph+0.30))
        q[10]= float(base[10]+0.38*np.sin(ph+np.pi))
        q[13]= float(base[13]+0.58*(0.5+0.5*np.sin(ph+np.pi+0.55)))
        q[14]= float(base[14]-0.16*np.sin(ph+np.pi+0.30))
        pose=torch.tensor(q,dtype=torch.float32,device='cpu').reshape(1,-1)
        _clamp_direct_pose_130(torch,pose)
        with torch.no_grad():
            world,_,_=_direct_world_130(torch,model,pose,betas,zero_trans,
                                         torch.tensor(rot,dtype=torch.float32),
                                         torch.tensor(scale,dtype=torch.float32),
                                         torch.tensor(trans,dtype=torch.float32),axis_map=axis_map)
        frames.append({'frame':k+1,'status':'ok','pose':pose[0].cpu().numpy().astype(float).tolist(),
                       'rotvec':rot.astype(float).tolist(),'trans':trans.astype(float).tolist(),
                       'joints':world.cpu().numpy().astype(float).tolist(),
                       'motor_test':True})
    seq={'version':'110.3.20.0-motor-test','retargeting_mode':'isolated_skel_lower_limb_motor_test',
         'scale':scale,'axis_map':axis_map.astype(float).tolist(),
         'joint_names':_skel_forward_joint_names_140(model),'skel_gender':str(gender),'frames':frames}
    mesh=_build_skel_skin_sequence(torch,model,seq)
    P=np.asarray(mesh.get('poses'),dtype=np.float32)
    J=np.asarray(mesh.get('joints'),dtype=np.float32) if mesh.get('joints') is not None else None
    names=[str(x) for x in mesh.get('joint_names',[])]
    nm={_simple_name(n):i for i,n in enumerate(names)}
    qids=[3,6,7,10,13,14]
    qr={_SKEL_Q_NAMES_130[i]:float(np.ptp(P[:,i])) for i in qids}
    jr={}
    if J is not None and J.ndim==3:
        for name in ('femur_r','tibia_r','talus_r','femur_l','tibia_l','talus_l'):
            ii=nm.get(_simple_name(name))
            if ii is not None:
                jr[name]=float(np.linalg.norm(np.ptp(J[:,ii,:],axis=0)))
    pass_q=bool(max(qr.values() or [0.0])>0.10)
    pass_joint=bool(max(jr.values() or [0.0])>0.01)
    audit={'ok':bool(pass_q and pass_joint),'q_ranges_rad':qr,'joint_ranges_world':jr,
           'interpretation':'PASS = SKEL.forward + skin_verts + reproductor pueden mover físicamente las piernas; el fallo restante está en V104/V107→q(t).' if pass_q and pass_joint else 'FAIL = revisar semántica q/SKEL.forward antes de retargeting.'}
    return seq,mesh,audit


def _target_knee_flexion_111(target_df, side):
    """Recupera el seed de bisagra que sí existía en V110.2.4: pi - ángulo cadera-rodilla-tobillo."""
    try:
        mp={str(r.joint):np.array([r.x,r.y,r.z],dtype=float) for r in target_df.itertuples()}
        h,k,a=(('RHip','RKnee','RAnkle') if str(side).upper().startswith('R') else ('LHip','LKnee','LAnkle'))
        if not all(x in mp and np.isfinite(mp[x]).all() for x in (h,k,a)):
            return None
        u=mp[h]-mp[k]; v=mp[a]-mp[k]
        den=max(float(np.linalg.norm(u)*np.linalg.norm(v)),1e-9)
        ang=float(np.arccos(np.clip(float(np.dot(u,v))/den,-1.0,1.0)))
        return float(np.clip(np.pi-ang,0.0,2.20))
    except Exception:
        return None


def _legacy_lower_hinge_fallback_111(torch, pose, target_df, before_pose=None):
    """Fallback mínimo y determinista tomado del principio de V110.2.4.
    Sólo escribe las bisagras de rodilla q6/q13 cuando el target las observa.
    Nunca toca estado de Streamlit ni persistencia.
    """
    p0=pose.detach().clone() if before_pose is None else before_pose.detach().clone()
    vals={}
    with torch.no_grad():
        for side,qi in [('R',6),('L',13)]:
            val=_target_knee_flexion_111(target_df,side)
            if val is not None and np.isfinite(val):
                pose[0,qi]=float(val); vals[side]=float(val)
        _clamp_direct_pose_130(torch,pose)
    ids=[3,4,5,6,7,10,11,12,13,14]
    dq={_SKEL_Q_NAMES_130[i]:float(pose[0,i].cpu()-p0[0,i].cpu()) for i in ids}
    return pose,{'applied':bool(vals),'knee_seed_rad':vals,'delta_q_from_before':dq,
                 'source':'V110.2.4 hinge seed: pi-angle(Hip,Knee,Ankle)'}

def _build_skel_skin_sequence(torch, model, seq, progress_cb=None):
    """Evalúa `skin_verts` para cada pose ya ajustada sin volver a optimizar la marcha.

    La identidad se mantiene fija (betas=0, igual que durante el fitting), y a cada
    malla se aplica exactamente la escala/orientación/traslación guardada en la secuencia V110.3.13 activa.
    """
    ok=[f for f in (seq or {}).get("frames",[]) if f.get("status")=="ok" and f.get("pose")]
    if not ok:
        raise RuntimeError("No hay poses SKEL válidas para generar la malla corporal.")
    faces,face_source=_skin_faces_numpy(model)
    betas=torch.zeros((1,int(model.num_betas)),dtype=torch.float32,device="cpu")
    zero_trans=torch.zeros((1,3),dtype=torch.float32,device="cpu")
    seq_scale=float(seq.get("scale",1.0))
    if not np.isfinite(seq_scale) or seq_scale<=0:
        raise RuntimeError(f"Escala de secuencia no válida: {seq_scale}")
    scale=torch.tensor(seq_scale,dtype=torch.float32,device="cpu")
    verts=[]; joints=[]; frame_ids=[]
    total=len(ok)
    with torch.no_grad():
        for k,fr in enumerate(ok,1):
            pose=torch.tensor(fr["pose"],dtype=torch.float32,device="cpu").reshape(1,-1)
            rot=torch.tensor(fr.get("rotvec",[0,0,0]),dtype=torch.float32,device="cpu")
            trans=torch.tensor(fr.get("trans",[0,0,0]),dtype=torch.float32,device="cpu")
            out=model(pose,betas,zero_trans,skelmesh=False)
            skin=getattr(out,"skin_verts",None)
            if skin is None:
                raise RuntimeError(f"Frame {fr.get('frame')}: SKEL no devolvió skin_verts.")
            V=skin[0]
            P=_axis_map_tensor_134(torch,seq.get("axis_map"),V.dtype,V.device)
            Vm=V@P.T
            Vt,_=_transform_joints_fixed(torch,Vm,rot,scale,trans)
            verts.append(Vt.detach().cpu().numpy().astype(np.float32))
            # Reusar joints temporales validados si existen; evita otro convenio geométrico.
            if fr.get("joints") is not None:
                joints.append(np.asarray(fr["joints"],dtype=np.float32))
            else:
                J=getattr(out,"joints",None)
                if J is not None:
                    Pj=_axis_map_tensor_134(torch,seq.get("axis_map"),J.dtype,J.device)
                    Jt,_=_transform_joints_fixed(torch,J[0]@Pj.T,rot,scale,trans)
                    joints.append(Jt.detach().cpu().numpy().astype(np.float32))
            frame_ids.append(int(fr.get("frame",k)))
            if progress_cb:
                progress_cb(k,total,int(V.shape[0]))
    V=np.stack(verts,axis=0)
    J=np.stack(joints,axis=0) if len(joints)==len(verts) else None
    poses=np.asarray([fr.get("pose") for fr in ok],dtype=np.float32) if all(fr.get("pose") is not None for fr in ok) else None
    return {
        "version":str(seq.get("version","110.3.20.8-segment-gentle")),
        "frame_ids":np.asarray(frame_ids,dtype=np.int16),
        "vertices":V,
        "faces":faces.astype(np.int32),
        "joints":J,
        "poses":poses,
        "joint_names":[str(x) for x in seq.get("joint_names",[])],
        "scale":float(seq.get("scale",1.0)),
        "betas":np.zeros((int(model.num_betas),),dtype=np.float32),
        "face_source":face_source,
        "source_sequence_version":str(seq.get("version","")),
        "source_sequence_scale":float(seq_scale),
        "axis_map":np.asarray(seq.get("axis_map",np.eye(3)),dtype=np.float32),
        "skel_gender":str(seq.get("skel_gender",getattr(model,"_physiosentinel_gender","unknown"))),
    }

def _audit_skel_animated_mesh(mesh_seq):
    """V110.3.13 · Control estructural de la malla animada sin reoptimizar la pose.

    Verifica que skin_verts sea finito y estable a lo largo de los frames, que la
    topología triangular sea válida y que no aparezcan explosiones geométricas
    entre frames. No modifica ningún vértice.
    """
    V=np.asarray((mesh_seq or {}).get("vertices"),dtype=np.float32)
    F=np.asarray((mesh_seq or {}).get("faces"),dtype=np.int32)
    if V.ndim!=3 or V.shape[-1]!=3 or V.shape[0]<1:
        return {"ok":False,"reasons":["vertices inválidos"],"frames":0}
    reasons=[]
    finite=bool(np.isfinite(V).all())
    if not finite: reasons.append("NaN/Inf en skin_verts")
    faces_ok=bool(F.ndim==2 and F.shape[1]>=3 and F.size>0 and F.min()>=0 and F.max()<V.shape[1])
    if not faces_ok: reasons.append("topología skin_f inválida")
    spans=np.nanmax(V,axis=1)-np.nanmin(V,axis=1)
    diag=np.linalg.norm(spans,axis=1)
    med=float(np.nanmedian(diag)) if diag.size else float('nan')
    ratio=float(np.nanmax(diag)/max(1e-8,med)) if diag.size and np.isfinite(med) else float('inf')
    if not np.isfinite(ratio) or ratio>2.5: reasons.append(f"explosión geométrica frame-a-frame (ratio={ratio:.2f})")
    if V.shape[0]>1:
        step=np.linalg.norm(np.diff(V,axis=0),axis=2)
        p99=float(np.nanpercentile(step,99))
        p50=float(np.nanpercentile(step,50))
        jump_ratio=float(p99/max(1e-8,p50))
    else:
        p99=p50=jump_ratio=0.0
    return {"ok":not reasons,"reasons":reasons,"frames":int(V.shape[0]),"vertices_per_frame":int(V.shape[1]),
            "triangles":int(F.shape[0]) if F.ndim==2 else 0,"finite":finite,"faces_ok":faces_ok,
            "bbox_diag_median":med,"bbox_diag_max_ratio":ratio,"vertex_step_p50":p50,"vertex_step_p99":p99,"vertex_step_ratio":jump_ratio}


def _mesh_sequence_npz_bytes(mesh_seq):
    buf=io.BytesIO()
    payload={
        "version":np.asarray([str(mesh_seq.get("version","110.3.13"))]),
        "q_names":np.asarray(_SKEL_Q_NAMES_130,dtype=str),
        "frame_ids":mesh_seq["frame_ids"],
        "vertices":mesh_seq["vertices"],
        "faces":mesh_seq["faces"],
        "scale":np.asarray([mesh_seq.get("scale",1.0)],dtype=np.float32),
        "betas":mesh_seq["betas"],
        "joint_names":np.asarray(mesh_seq.get("joint_names",[]),dtype=str),
        "source_sequence_version":np.asarray([str(mesh_seq.get("source_sequence_version",""))]),
        "source_sequence_scale":np.asarray([mesh_seq.get("source_sequence_scale",mesh_seq.get("scale",1.0))],dtype=np.float32),
        "axis_map":np.asarray(mesh_seq.get("axis_map",np.eye(3)),dtype=np.float32),
        "skel_gender":np.asarray([str(mesh_seq.get("skel_gender","unknown"))]),
        "skeleton_layer_version":np.asarray(["volumetric_anatomical_skel_v3"]),
        "skeleton_layer_semantics":np.asarray(["SKEL anatomical functional skeleton v2; segment-scaled bone representation driven by SKEL joints; not patient-specific CT/MRI bone geometry"]),
        "muscle_layer_version":np.asarray(["volumetric_muscle_v2"]),
        "muscle_layer_semantics":np.asarray(["Functional musculotendinous visualization derived from SKEL joint chains; no EMG, force, activation or patient-specific MRI muscle geometry inferred"]),
        "kinematics3d_version":np.asarray(["anatomical_kinematics3d_v2"]),
    }
    if mesh_seq.get("joints") is not None:
        payload["joints"]=mesh_seq["joints"]
    if mesh_seq.get("poses") is not None:
        payload["poses"]=np.asarray(mesh_seq["poses"],dtype=np.float32)
    if isinstance(mesh_seq.get("foot_landmark_scores"),dict):
        try:
            import json as _json
            payload["foot_landmark_scores_json"]=np.asarray([_json.dumps(mesh_seq["foot_landmark_scores"])])
        except Exception:
            pass
    payload["foot_qc_version"]=np.asarray(["157-foot-qc-v1"])
    np.savez_compressed(buf,**payload)
    return buf.getvalue()


def _skel_volumetric_anatomy_2010(joints, joint_names):
    """Versión 149 · Generates volumetric anatomically-shaped bone and muscle meshes.
    Geometry is registered to SKEL joints frame-by-frame and never modifies q(t).
    It is a visualization model, not patient-specific CT/MRI segmentation.
    """
    J=np.asarray(joints,dtype=np.float32)
    names=[str(x) for x in joint_names]
    norm={_simple_name(n):i for i,n in enumerate(names)}
    if J.ndim!=3 or J.shape[-1]!=3:
        return None

    # a,b,label,radius_a,radius_mid,radius_b. Radii are fractions of segment length.
    bone_specs=[
      ("pelvis","femur_r","Pelvis→cadera D",.13,.18,.10),("pelvis","femur_l","Pelvis→cadera I",.13,.18,.10),
      ("femur_r","tibia_r","Fémur D",.105,.075,.115),("femur_l","tibia_l","Fémur I",.105,.075,.115),
      ("tibia_r","talus_r","Tibia/peroné D",.105,.070,.085),("tibia_l","talus_l","Tibia/peroné I",.105,.070,.085),
      ("talus_r","calcn_r","Astrágalo/calcáneo D",.18,.22,.16),("talus_l","calcn_l","Astrágalo/calcáneo I",.18,.22,.16),
      ("calcn_r","toes_r","Pie D",.16,.115,.07),("calcn_l","toes_l","Pie I",.16,.115,.07),
      ("pelvis","lumbar_body","Sacro-lumbar",.16,.13,.11),("lumbar_body","thorax","Columna torácica",.12,.10,.13),
      ("thorax","head","Cervical/cráneo",.11,.09,.18),
      ("thorax","scapula_r","Clavícula/escápula D",.11,.085,.09),("thorax","scapula_l","Clavícula/escápula I",.11,.085,.09),
      ("scapula_r","humerus_r","Húmero D",.12,.075,.10),("scapula_l","humerus_l","Húmero I",.12,.075,.10),
      ("humerus_r","ulna_r","Cúbito D",.09,.055,.07),("humerus_l","ulna_l","Cúbito I",.09,.055,.07),
      ("ulna_r","radius_r","Radio D",.07,.05,.06),("ulna_l","radius_l","Radio I",.07,.05,.06),
      ("radius_r","hand_r","Mano D",.11,.09,.055),("radius_l","hand_l","Mano I",.11,.09,.055)]
    # Functional muscle bellies: thicker at middle, narrower near origin/insertion.
    muscle_specs=[
      ("pelvis","femur_r","Glúteo/iliopsoas D",.11,.23,.08),("pelvis","femur_l","Glúteo/iliopsoas I",.11,.23,.08),
      ("femur_r","tibia_r","Cuádriceps D",.07,.19,.055),("femur_l","tibia_l","Cuádriceps I",.07,.19,.055),
      ("pelvis","tibia_r","Isquiosurales D",.055,.12,.045),("pelvis","tibia_l","Isquiosurales I",.055,.12,.045),
      ("tibia_r","calcn_r","Gemelos/sóleo D",.06,.18,.045),("tibia_l","calcn_l","Gemelos/sóleo I",.06,.18,.045),
      ("tibia_r","toes_r","Tibial/peroneos D",.035,.075,.025),("tibia_l","toes_l","Tibial/peroneos I",.035,.075,.025),
      ("pelvis","thorax","Paravertebrales/abdominales",.09,.17,.10),
      ("thorax","humerus_r","Deltoides/pectoral D",.07,.18,.055),("thorax","humerus_l","Deltoides/pectoral I",.07,.18,.055),
      ("scapula_r","ulna_r","Bíceps/tríceps D",.045,.13,.035),("scapula_l","ulna_l","Bíceps/tríceps I",.045,.13,.035),
      ("humerus_r","radius_r","Antebrazo D",.035,.085,.025),("humerus_l","radius_l","Antebrazo I",.035,.085,.025)]

    def build(specs, sides=10, rings=5):
        allV=[]; faces=None; labels=[]
        valid=[s for s in specs if _simple_name(s[0]) in norm and _simple_name(s[1]) in norm]
        for t in range(J.shape[0]):
            vv=[]; ff=[]; off=0
            for a,b,label,r0,rm,r1 in valid:
                A=J[t,norm[_simple_name(a)]].astype(float); B=J[t,norm[_simple_name(b)]].astype(float)
                d=B-A; L=float(np.linalg.norm(d))
                if not np.isfinite(L) or L<1e-6: continue
                ez=d/L
                ref=np.array([0.,0.,1.]) if abs(ez[2])<.88 else np.array([0.,1.,0.])
                ex=np.cross(ez,ref); ex/=max(np.linalg.norm(ex),1e-9)
                ey=np.cross(ez,ex); ey/=max(np.linalg.norm(ey),1e-9)
                # anatomical spindle / tapered long-bone profile
                ts=np.linspace(0,1,rings)
                for ri,u in enumerate(ts):
                    # quadratic interpolation with expanded epiphysis / muscle belly
                    if u<=.5:
                        rr=(r0+(rm-r0)*(u/.5))*L
                    else:
                        rr=(rm+(r1-rm)*((u-.5)/.5))*L
                    c=A+u*d
                    for s in range(sides):
                        th=2*np.pi*s/sides
                        # mild elliptical cross-section
                        vv.append(c + rr*np.cos(th)*ex + (.82*rr)*np.sin(th)*ey)
                for ri in range(rings-1):
                    for s in range(sides):
                        s2=(s+1)%sides
                        a0=off+ri*sides+s; a1=off+ri*sides+s2
                        b0=off+(ri+1)*sides+s; b1=off+(ri+1)*sides+s2
                        ff.extend([[a0,b0,b1],[a0,b1,a1]])
                # end caps
                ca=len(vv); vv.append(A); cb=len(vv); vv.append(B)
                for s in range(sides):
                    s2=(s+1)%sides
                    ff.append([ca,off+s2,off+s])
                    e0=off+(rings-1)*sides+s; e1=off+(rings-1)*sides+s2
                    ff.append([cb,e0,e1])
                off=len(vv)
            allV.append(np.asarray(vv,dtype=np.float32))
            if faces is None: faces=np.asarray(ff,dtype=np.int32)
        return {"vertices":np.stack(allV) if allV and len({x.shape for x in allV})==1 else None,
                "faces":faces if faces is not None else np.zeros((0,3),dtype=np.int32),
                "labels":[s[2] for s in valid]}
    return {"bone":build(bone_specs),"muscle":build(muscle_specs),
            "version":"volumetric_anatomy_v1",
            "semantics":"Anatomically-shaped volumetric meshes registered to SKEL joints; not patient-specific CT/MRI segmentation."}




def _v2018_browser_mesh(part, max_faces=1800):
    """Create a browser-only lightweight copy of ONE anatomical mesh.
    Keeps real BodyParts3D geometry, but samples faces and compacts vertices.
    Does not alter the source atlas or offline MP4 renderer.
    """
    V0=np.asarray(part["V0"],dtype=np.float32)
    F=np.asarray(part["F"],dtype=np.int32)
    if V0.ndim!=2 or V0.shape[1]!=3 or F.ndim!=2 or F.shape[1]<3 or len(F)==0:
        return None
    step=max(1,int(np.ceil(len(F)/float(max_faces))))
    Fs=F[::step,:3]
    used=np.unique(Fs.reshape(-1))
    remap=np.full(len(V0),-1,dtype=np.int32)
    remap[used]=np.arange(len(used),dtype=np.int32)
    Fc=remap[Fs]
    return {
        "id":part["id"],
        "V0":V0[used],
        "F":Fc.astype(np.int32),
        "R":np.asarray(part["R"],dtype=np.float32),
        "S":np.asarray(part["S"],dtype=np.float32),
        "T":np.asarray(part["T"],dtype=np.float32),
        "source_vertices":int(len(V0)),
        "source_faces":int(len(F)),
        "browser_vertices":int(len(used)),
        "browser_faces":int(len(Fc)),
    }

def _v2018_prepare_browser_parts(parts, max_faces_each=1800):
    out=[]
    for p in parts:
        q=_v2018_browser_mesh(p,max_faces=max_faces_each)
        if q is not None:
            out.append(q)
    return out


def _v2019_rigid_skeleton_for_mesh(mesh_seq):
    """BodyParts3D skeleton in shared atlas coordinates, globally registered once, then rigidly articulated."""
    import os as _os
    J=np.asarray(mesh_seq.get("joints"),dtype=np.float32) if isinstance(mesh_seq,dict) and mesh_seq.get("joints") is not None else None
    names=[str(x) for x in (mesh_seq.get("joint_names",[]) if isinstance(mesh_seq,dict) else [])]
    if J is None or J.ndim!=3:
        return {"bones":[],"missing":["joints SKEL"],"method":"global_rigid_v2019"}
    _status=bodyparts3d_atlas_status()
    if _status.get("ready"):
        return load_bodyparts3d_rigid_skeleton(_status["root"],J,names)
    root=_os.path.join(_os.path.dirname(__file__),"assets","anatomical_atlas")
    return load_bodyparts3d_rigid_skeleton(root,J,names)

def _v2017_atlas_plan_for_mesh(mesh_seq, groups=("bones","muscles")):
    """Carga las mallas anatómicas REALES una sola vez y calcula sólo transformaciones por frame."""
    import os as _os
    J=np.asarray(mesh_seq.get("joints"),dtype=np.float32) if isinstance(mesh_seq,dict) and mesh_seq.get("joints") is not None else None
    names=[str(x) for x in (mesh_seq.get("joint_names",[]) if isinstance(mesh_seq,dict) else [])]
    if J is None or J.ndim!=3:
        return {"bones":[],"muscles":[],"missing":["joints SKEL"],"version":""}
    _status=bodyparts3d_atlas_status()
    if _status.get("ready"):
        return load_atlas_transform_plan(_status["root"],J,names,groups=groups)
    root=_os.path.join(_os.path.dirname(__file__),"assets","anatomical_atlas")
    return load_atlas_transform_plan(root,J,names,groups=groups)

def _v2013_render_video_mp4(mesh_seq, layer="skin", fps=25):
    """Render OFFLINE de los 75 frames ya calculados. No vuelve a ejecutar fitting ni SKEL.
    layer: skin | bones | muscles | all
    """
    import tempfile, os as _os, subprocess as _subprocess
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection as _Poly3DCollection
    import imageio_ffmpeg as _iioff

    if not isinstance(mesh_seq,dict):
        raise ValueError("No hay secuencia SKEL cargada.")
    V=np.asarray(mesh_seq.get("vertices"),dtype=np.float32)
    F=np.asarray(mesh_seq.get("faces"),dtype=np.int32)
    J=np.asarray(mesh_seq.get("joints"),dtype=np.float32) if mesh_seq.get("joints") is not None else None
    if V.ndim!=3 or F.ndim!=2 or V.shape[0]<1:
        raise ValueError("La secuencia SKEL no contiene vertices/faces válidos.")

    atlas={"bones":[],"muscles":[],"missing":[]}
    if layer in ("bones","muscles","all"):
        atlas=_v2017_atlas_plan_for_mesh(mesh_seq, groups=("bones","muscles"))
        if layer=="bones" and not atlas.get("bones"):
            raise ValueError("No hay mallas anatómicas reales de huesos cargadas en assets/anatomical_atlas/bones.")
        if layer=="muscles" and not atlas.get("muscles"):
            raise ValueError("No hay mallas anatómicas reales de músculos cargadas en assets/anatomical_atlas/muscles.")
        if layer=="all" and not (atlas.get("bones") or atlas.get("muscles")):
            raise ValueError("No hay mallas anatómicas reales de huesos/músculos cargadas.")

    Vviz=V.copy()
    bone_parts=atlas.get("bones",[])
    muscle_parts=atlas.get("muscles",[])
    # Centrado idéntico al visor: pelvis al origen cuando está disponible.
    if J is not None and J.ndim==3:
        names=[str(x) for x in mesh_seq.get("joint_names",[])]
        nm={re.sub(r'[^a-z0-9]','',n.lower()):i for i,n in enumerate(names)}
        pi=nm.get("pelvis")
        if pi is not None:
            C=J[:,pi,:].copy()
            Vviz-=C[:,None,:]
            # El atlas ya se registra sobre joints originales; aplicar el mismo centrado.
            for p in bone_parts:
                p["V"]=np.asarray(p["V"],dtype=np.float32)-C[:,None,:]
            for p in muscle_parts:
                p["V"]=np.asarray(p["V"],dtype=np.float32)-C[:,None,:]

    # Límites fijos para evitar zoom/bombeo entre frames.
    geom=[Vviz.reshape(-1,3)]
    for p in bone_parts: geom.append(np.asarray(p["V"],dtype=np.float32).reshape(-1,3))
    for p in muscle_parts: geom.append(np.asarray(p["V"],dtype=np.float32).reshape(-1,3))
    G=np.concatenate(geom,axis=0)
    mins=np.nanmin(G,axis=0); maxs=np.nanmax(G,axis=0)
    span=np.maximum(maxs-mins,1e-5); pad=np.maximum(.06*span,.02)
    mins-=pad; maxs+=pad
    center=(mins+maxs)/2; radius=max(float(np.max(maxs-mins))/2,1e-3)

    # Render moderado para Streamlit Cloud. La visualización interactiva conserva todas las caras.
    max_faces_skin=5000
    step_skin=max(1,int(np.ceil(len(F)/max_faces_skin)))
    Fs=F[::step_skin,:3]

    W,H=720,720
    tmpdir=tempfile.mkdtemp(prefix="physiosentinel_skel_video_")
    out=_os.path.join(tmpdir,f"SKEL_{layer}_{V.shape[0]}frames.mp4")
    writer=_iioff.write_frames(out,(W,H),fps=float(fps),codec="libx264",
                                quality=7,pix_fmt_in="rgb24",pix_fmt_out="yuv420p")
    writer.send(None)

    try:
        for t in range(Vviz.shape[0]):
            fig=_plt.figure(figsize=(W/100,H/100),dpi=100)
            ax=fig.add_subplot(111,projection="3d")
            ax.set_axis_off()
            ax.set_xlim(center[0]-radius,center[0]+radius)
            ax.set_ylim(center[1]-radius,center[1]+radius)
            ax.set_zlim(center[2]-radius,center[2]+radius)
            ax.set_box_aspect((1,1,1))
            ax.view_init(elev=8,azim=-78)

            if layer in ("skin","all"):
                tri=Vviz[t][Fs]
                pc=_Poly3DCollection(tri,linewidths=0.0,alpha=(0.28 if layer=="all" else 0.88))
                pc.set_facecolor((0.58,0.72,0.82,1.0))
                pc.set_edgecolor("none")
                ax.add_collection3d(pc)

            if layer in ("bones","all"):
                for p in bone_parts:
                    VV=apply_atlas_transform(p,t)
                    FF=np.asarray(p["F"],dtype=np.int32)
                    if VV.ndim!=2 or len(FF)==0: continue
                    step=max(1,int(np.ceil(len(FF)/2200)))
                    tri=VV[FF[::step,:3]]
                    pc=_Poly3DCollection(tri,linewidths=0.0,alpha=1.0)
                    pc.set_facecolor((0.92,0.88,0.72,1.0))
                    pc.set_edgecolor("none")
                    ax.add_collection3d(pc)

            if layer in ("muscles","all"):
                for p in muscle_parts:
                    VV=apply_atlas_transform(p,t)
                    FF=np.asarray(p["F"],dtype=np.int32)
                    if VV.ndim!=2 or len(FF)==0: continue
                    step=max(1,int(np.ceil(len(FF)/1800)))
                    tri=VV[FF[::step,:3]]
                    pc=_Poly3DCollection(tri,linewidths=0.0,alpha=(0.78 if layer=="muscles" else 0.62))
                    pc.set_facecolor((0.66,0.20,0.18,1.0))
                    pc.set_edgecolor("none")
                    ax.add_collection3d(pc)

            fig.canvas.draw()
            frame=np.asarray(fig.canvas.buffer_rgba(),dtype=np.uint8)[...,:3].copy()
            writer.send(frame.tobytes())
            _plt.close(fig)
    finally:
        try: writer.close()
        except Exception: pass

    data=Path(out).read_bytes()
    try:
        import shutil as _shutil
        _shutil.rmtree(tmpdir,ignore_errors=True)
    except Exception:
        pass
    return data

def _plot_skel_mesh_walker(mesh_seq):
    """Versión 151 · visor de atlas anatómico REAL sobre SKEL. Sin fallback geométrico."""
    try:
        import json as _json, os as _os
        import streamlit.components.v1 as components
        from plotly.offline import get_plotlyjs
        V=np.asarray(mesh_seq.get("vertices"),dtype=np.float32)
        F=np.asarray(mesh_seq.get("faces"),dtype=np.int32)
        J=np.asarray(mesh_seq.get("joints"),dtype=np.float32) if mesh_seq.get("joints") is not None else None
        ids=[int(x) for x in np.asarray(mesh_seq.get("frame_ids",np.arange(len(V))+1)).tolist()]
        names=[str(x) for x in mesh_seq.get("joint_names",[])]
        if V.ndim!=3 or F.ndim!=2 or J is None or J.ndim!=3:
            raise RuntimeError("La secuencia requiere skin, faces y joints SKEL.")
        norm={_simple_name(n):i for i,n in enumerate(names)}
        Vviz=V.copy(); Jviz=J.copy()
        _axis_map_v161=np.asarray(mesh_seq.get("axis_map",np.eye(3)),dtype=np.float32)
        try:
            from visible_human_atlas import neutralize_skel_foot_skin
            Vviz,_skin_foot_qc_v161=neutralize_skel_foot_skin(Vviz,Jviz,names,_axis_map_v161)
        except Exception as _skin_qc_error_v161:
            _skin_foot_qc_v161={"pass":False,"error":str(_skin_qc_error_v161)}
            st.error("V161 · Corrección neutra de la piel bloqueada por control anatómico: "+str(_skin_qc_error_v161))
        pi=norm.get(_simple_name("pelvis"))
        if pi is not None:
            C=Jviz[:,pi,:].copy(); Vviz-=C[:,None,:]; Jviz-=C[:,None,:]
        st.markdown("---")
        st.markdown("## 💪 Fuente muscular volumétrica · Visible Human")
        st.caption(
            "Arquitectura híbrida: SKEL aporta piel y movimiento; OpenSim/Hamner aporta el esqueleto; "
            "Visible Human Denver aporta la geometría muscular volumétrica de referencia del miembro inferior."
        )

        _vh1,_vh2=st.columns(2)
        _vh_subject=_vh1.selectbox("Atlas de referencia",["Female","Male"],key="v3220_subject")
        _vh_zip=_vh2.file_uploader(
            "ZIP · Final 3D STL Models (University of Denver)",
            type=["zip"],key="v3220_denver_zip",
            help="El ZIP final del dataset Visible Human de University of Denver (~139 MB)."
        )
        st.caption(
            "Licencia del atlas: CC BY 4.0. No se guarda en Supabase ni dentro del NPZ; "
            "se extrae en caché temporal del runtime."
        )

        if _vh_zip is not None:
            _vh_bytes=_vh_zip.getvalue()
            _vh_manifest=prepare_visible_human_zip(_vh_bytes,_vh_subject)
            _old_sha=(st.session_state.get("v3220_manifest") or {}).get("sha")
            st.session_state["v3220_manifest"]=_vh_manifest
            if _old_sha != _vh_manifest.get("sha"):
                st.session_state.pop("v32110_seq_key",None)
                st.session_state.pop("v3222_vh_muscle_vertices",None)
                st.session_state.pop("v3222_vh_muscle_faces",None)
                st.session_state.pop("v32110_bone_vertices",None)
                st.session_state.pop("v32110_bone_faces",None)
                st.session_state.pop("v3228_denver_real_bones_active",None)
                st.session_state.pop("v32212_denver_compact",None)
                st.session_state.pop("v149_denver_muscle_rig",None)
                st.session_state.pop("v152_denver_muscle_rig",None)
                st.session_state.pop("v152_musculoskeletal_bridge",None)
                st.session_state.pop("v152_bridge_rerun_key",None)
                st.session_state.pop("v32212_upper_vertices",None)
                st.session_state.pop("v32212_upper_faces",None)
        else:
            _vh_manifest=st.session_state.get("v3220_manifest")

        if _vh_manifest and _vh_manifest.get("ready"):
            _cnt=atlas_counts(_vh_manifest)
            st.success(
                f"Atlas Visible Human preparado · {_cnt['bones']} mallas óseas · "
                f"{_cnt['muscles']} mallas musculares detectadas."
            )
            st.caption(ATTRIBUTION)

            st.success("Atlas Denver leído. Se utilizará como atlas maestro muscular registrado una sola vez sobre OpenSim/Hamner y después retargeteado a los 75 frames SKEL.")
        else:
            st.info(
                "Para activar la anatomía volumétrica real, carga el ZIP «Final 3D STL Models» "
                "del dataset Visible Human Male/Female de University of Denver."
            )

        st.markdown("## 🧍🦴💪 Visor anatómico realista · Denver + rig OpenSim/SKEL")
        st.caption("Tal como lo dejes en pantalla con el ratón será la orientación usada para la descarga. Activa Multiplano solo si quieres que la cámara cambie durante la reproducción.")
        st.caption(
            "Arquitectura activa: Piel SKEL + Esqueleto OpenSim/Hamner + músculo Visible Human cuando el atlas está cargado. "
            "BodyParts3D y Rajagopal permanecen retirados del flujo activo."
        )

        _ham_status=hamner_status()
        if not _ham_status.get("ready"):
            st.info("Prepara OpenSim Hamner una sola vez para habilitar esqueleto y músculo.")
            if st.button("🧬 Preparar OpenSim Hamner",key="v32110_prepare_hamner",use_container_width=True):
                with st.spinner("Preparando geometrías OpenSim Hamner…"):
                    _rr=prepare_hamner(force=False)
                if _rr.get("ok"):
                    st.success(f"Hamner preparado · {_rr.get('mesh_count',0)} geometrías · {_rr.get('failed',0)} fallos.")
                    st.rerun()
                else:
                    st.error(f"No se pudo preparar Hamner: {_rr.get('error','error desconocido')}")
        else:
            _Jv=np.asarray(mesh_seq.get("joints"),dtype=np.float32)
            _JNv=[str(x) for x in mesh_seq.get("joint_names",[])]
            _Sv=np.asarray(Vviz,dtype=np.float32)
            _SF=np.asarray(F[:,:3],dtype=np.int32)
            _hb=load_hamner_bodies()

            _vh_cache_man=st.session_state.get("v3220_manifest") or {}
            _vh_cache_sha=str(_vh_cache_man.get("sha","NO_DENVER"))
            _cache_key=("v161",int(len(_Sv)),int(_Sv.shape[1]),float(np.nanmean(_Jv[0])),_vh_cache_sha,tuple(np.round(_axis_map_v161.reshape(-1),3)))
            if st.session_state.get("v32110_seq_key")!=_cache_key:
                with st.spinner("Preparando anatomía compacta Denver: una malla + transformaciones de 75 frames…"):
                    def _build_opt(_b,_j,_jn,frame=0,skin_vertices=None):
                        return register_hamner_frame_optimized(
                            _b,_j,_jn,skin_vertices,frame,build_hamner_fullbody_frame
                        )
                    _MSEQ,_MNAMES=muscle_paths_sequence(_Jv,_JNv,muscle_segments)
                    _vh_man=st.session_state.get("v3220_manifest")
                    # V155 · protección antirregresión anatómica. Un fallo temporal de construcción
                    # NO puede sustituir silenciosamente Denver por el esqueleto lineal/funcional antiguo.
                    _prev_DC=st.session_state.get("v32212_denver_compact")
                    _prev_MR=st.session_state.get("v152_denver_muscle_rig") or st.session_state.get("v149_denver_muscle_rig")
                    _prev_UBV=st.session_state.get("v32212_upper_vertices")
                    _prev_UBF=st.session_state.get("v32212_upper_faces")
                    _prev_sha=str(st.session_state.get("v155_last_valid_denver_sha", ""))
                    _DC=None; _MR=None; _UBV=None; _UBF=None; _BV=None; _BF=None
                    _denver_error=None

                    if _vh_man and _vh_man.get("ready"):
                        try:
                            from visible_human_atlas import build_visible_human_compact_payload
                            # Intento principal: misma anatomía validada de V153.
                            _DC=build_visible_human_compact_payload(
                                _vh_man,_Jv,_JNv,
                                max_bone_faces_each=700,
                                max_muscle_faces_each=300,
                                axis_map=_axis_map_v161,
                            )
                            if _DC is None:
                                raise RuntimeError("atlas compacto Denver vacío")
                        except Exception as _ve1:
                            # Segundo intento RAM-safe: mantiene mallas conectadas, sólo baja la densidad.
                            try:
                                _DC=build_visible_human_compact_payload(
                                    _vh_man,_Jv,_JNv,
                                    max_bone_faces_each=450,
                                    max_muscle_faces_each=220,
                                    axis_map=_axis_map_v161,
                                )
                                if _DC is None:
                                    raise RuntimeError("atlas compacto Denver vacío en segundo intento")
                                st.warning("V155: Denver se ha reconstruido en modo RAM-safe manteniendo mallas anatómicas conectadas.")
                            except Exception as _ve2:
                                _denver_error=f"{type(_ve2).__name__}: {_ve2}"
                                # Si el mismo atlas ya tenía un rig Denver válido en esta sesión, conservarlo.
                                if _prev_DC is not None and _prev_sha==_vh_cache_sha:
                                    _DC=_prev_DC; _MR=_prev_MR; _UBV=_prev_UBV; _UBF=_prev_UBF
                                    st.warning("V155: fallo temporal al reconstruir Denver; se conserva el último rig anatómico Denver válido de esta misma sesión.")
                                else:
                                    _DC=None

                        if _DC is not None and _MR is None:
                            # GeometryPath→retarget jerárquico→RMF→vientre Denver V11.
                            _poses_v149=np.asarray(mesh_seq.get("poses"),dtype=np.float32)
                            _osim_v149=Path(__file__).resolve().parent/"assets"/"Hamner2010_SimpleArms_Markers_v2_0.osim"
                            try:
                                if _poses_v149.ndim!=2 or len(_poses_v149)!=len(_Jv):
                                    raise RuntimeError("V155 requiere poses SKEL de 75 frames dentro del NPZ/secuencia")
                                _MR=build_all76_rig(
                                    _vh_man,_Jv,_JNv,_poses_v149,_osim_v149,max_faces_each=300
                                )
                            except Exception as _mre:
                                _MR=None
                                st.warning(
                                    "El rig muscular V155 no pudo construirse; los huesos Denver se conservan y "
                                    "la musculatura volumétrica Denver compacta permanece disponible como fallback. "
                                    f"{type(_mre).__name__}: {_mre}"
                                )

                        if _DC is not None and _UBV is None:
                            # Tren superior: fallback Hamner limitado, sin reemplazar jamás el Denver inferior.
                            _upper_ids={
                                "torso","humerus_r","humerus_l","ulna_r","ulna_l",
                                "radius_r","radius_l","hand_r","hand_l"
                            }
                            try:
                                _UBV,_UBF=build_hamner_sequence_filtered(
                                    _hb.get("bodies",{}),_Jv,_JNv,_Sv,
                                    _build_opt,_upper_ids,max_faces_each=80
                                )
                            except Exception as _ube:
                                _UBV=None; _UBF=None
                                st.warning(f"Tren superior Hamner no disponible: {type(_ube).__name__}: {_ube}")

                        if _DC is None:
                            # Antirregresión V155: con Denver cargado NO se muestra el esqueleto antiguo.
                            st.error(
                                "V155 · Denver está cargado pero no se pudo construir la anatomía volumétrica. "
                                "Se bloquea el fallback óseo antiguo para evitar mostrar una anatomía regresiva. "
                                + (f"Detalle: {_denver_error}" if _denver_error else "")
                            )
                    else:
                        # Solo cuando NO hay atlas Denver cargado se permite el fallback Hamner funcional.
                        _BV,_BF=build_hamner_sequence(
                            _hb.get("bodies",{}),_Jv,_JNv,_Sv,
                            _build_opt,max_faces_each=100
                        )

                st.session_state["v32212_denver_compact"]=_DC
                # V152: publicar explícitamente el rig para la Pestaña 13.
                # Se mantiene la clave histórica v149 por compatibilidad interna.
                st.session_state["v149_denver_muscle_rig"]=_MR
                st.session_state["v152_denver_muscle_rig"]=_MR
                st.session_state["v152_musculoskeletal_bridge"]={
                    "ready": bool(_MR is not None),
                    "rig_key": "v152_denver_muscle_rig",
                    "mesh_key": "v110_3_18_mesh_sequence",
                    "frames": int(len(_Jv)),
                    "atlas_sha": _vh_cache_sha,
                    "cache_key": repr(_cache_key),
                }
                st.session_state["v32212_upper_vertices"]=_UBV
                st.session_state["v32212_upper_faces"]=_UBF
                st.session_state["v32110_bone_vertices"]=_BV
                st.session_state["v32110_bone_faces"]=_BF
                st.session_state["v3228_denver_real_bones_active"]=bool(_DC is not None)
                if _DC is not None:
                    st.session_state["v155_last_valid_denver_sha"]=_vh_cache_sha
                st.session_state["v32110_muscle_seq"]=_MSEQ
                st.session_state["v32110_muscle_names"]=_MNAMES
                # Explicitly remove legacy 75-frame Denver arrays from RAM.
                st.session_state.pop("v3222_vh_muscle_vertices",None)
                st.session_state.pop("v3222_vh_muscle_faces",None)
                st.session_state["v32110_seq_key"]=_cache_key
                # V152: streamlit_app evalúa la Pestaña 13 antes que este bloque de la
                # Pestaña 11. Tras construir el rig por primera vez hacemos UN rerun
                # controlado para que Tab 13 lo vea inmediatamente en la misma sesión.
                _v152_rerun_key=repr(_cache_key)
                if _MR is not None and st.session_state.get("v152_bridge_rerun_key") != _v152_rerun_key:
                    st.session_state["v152_bridge_rerun_key"]=_v152_rerun_key
                    st.rerun()

            _DC=st.session_state.get("v32212_denver_compact")
            _MR=st.session_state.get("v152_denver_muscle_rig")
            if _MR is None:
                _MR=st.session_state.get("v149_denver_muscle_rig")
            _UBV=st.session_state.get("v32212_upper_vertices")
            _UBF=st.session_state.get("v32212_upper_faces")
            if _DC is None:
                _bv_state=st.session_state.get("v32110_bone_vertices")
                _bf_state=st.session_state.get("v32110_bone_faces")
                if _bv_state is not None and _bf_state is not None and not (_vh_manifest and _vh_manifest.get("ready")):
                    _BV=np.asarray(_bv_state,dtype=np.float32)
                    _BF=np.asarray(_bf_state,dtype=np.int32)
                else:
                    _BV=None; _BF=None
            else:
                _BV=None; _BF=None
            _MSEQ=st.session_state["v32110_muscle_seq"]
            _MNAMES=st.session_state["v32110_muscle_names"]
            _VMV=None
            _VMF=None
            if st.session_state.get("v3228_denver_real_bones_active"):
                if _MR is not None:
                    st.success(
                        "Versión 155 · anatomía compacta: huesos Denver conservan el registro landmark de V148 y "
                        "los 76 músculos usan GeometryPath Hamner + retarget jerárquico + RMF + volumen Denver V11. "
                        "No se almacenan 75 copias completas de cada malla."
                    )
                    st.caption(
                        "Mapping muscular: 68/76 directo, compuesto o compartido desde el .osim; 8/76 proxy explícito "
                        "(obturadores, plantaris y poplíteo bilaterales). No se presentan como PathPoints Hamner exactos."
                    )
                else:
                    st.info("Huesos Denver activos · rig muscular V150 no disponible en esta sesión; se usa fallback muscular anterior.")

            # QC del registro anatómico optimizado en el frame de referencia.
            try:
                _parts0=register_hamner_frame_optimized(
                    _hb.get("bodies",{}),_Jv,_JNv,_Sv[0],0,build_hamner_fullbody_frame
                )
                _qc=registration_quality(_parts0,_Sv[0],_Jv,_JNv,0)
                if _qc:
                    _qcdf=pd.DataFrame(_qc)
                    _mean_err=float(np.nanmean(_qcdf[["error_prox","error_dist"]].values))
                    st.metric("Error medio de ensamblaje óseo · frame 1",f"{_mean_err:.4f} u SKEL")
                    with st.expander("Control de calidad del registro anatómico",expanded=False):
                        st.dataframe(_qcdf.round(4),use_container_width=True,hide_index=True)
            except Exception:
                pass

            # Versión 151 — SKEL skin topology-preserving browser payload.
            # IMPORTANT: never subsample faces with F[::step] here. That creates
            # disconnected triangular islands (the visible “confetti” artefact).
            # The complete connected SKEL topology is small enough (~13.8k faces)
            # to send to the browser while retaining the same 75-frame vertex sequence.
            _SFc=np.asarray(_SF,dtype=np.int32)
            _SVc=np.asarray(_Sv,dtype=np.float32)

            _mus_frames=[]
            for _fr in _MSEQ:
                _mus_frames.append([np.round(np.asarray(_q,dtype=np.float32),5).tolist() for _q in _fr])

            import json as _json
            from plotly.offline import get_plotlyjs as _get_plotlyjs
            _mins=np.nanmin(_Sv.reshape(-1,3),axis=0); _maxs=np.nanmax(_Sv.reshape(-1,3),axis=0)
            _span=np.maximum(_maxs-_mins,1e-3); _pad=np.maximum(.08*_span,.03)
            _ranges=[[float(_mins[d]-_pad[d]),float(_maxs[d]+_pad[d])] for d in range(3)]
            _MRjson=None
            if _MR is not None:
                _MRjson={
                    "U":np.round(np.asarray(_MR["u"],dtype=np.float32),5).tolist(),
                    "T":np.round(np.asarray(_MR["transverse"],dtype=np.float32),5).tolist(),
                    "F":np.asarray(_MR["faces"],dtype=np.int32).tolist(),
                    "MI":np.asarray(_MR["muscle_index"],dtype=np.int16).tolist(),
                    "PS":np.round(np.asarray(_MR["path_samples"],dtype=np.float32),5).tolist(),
                    "PN":np.round(np.asarray(_MR["path_normals"],dtype=np.float32),5).tolist(),
                    "PB":np.round(np.asarray(_MR["path_binormals"],dtype=np.float32),5).tolist(),
                    "S":np.round(np.asarray(_MR["start_frac"],dtype=np.float32),5).tolist(),
                    "E":np.round(np.asarray(_MR["end_frac"],dtype=np.float32),5).tolist(),
                    "TS":np.round(np.asarray(_MR["trans_scale"],dtype=np.float32),5).tolist(),
                    "Q":list(_MR.get("mapping_quality",[])),
                    "N":list(_MR.get("muscle_names",[])),
                    "proxy_count":int(_MR.get("proxy_count",0)),
                }
            if _DC is not None:
                _aqc=_DC.get("anatomy_qc",{})
                if _aqc.get("pass"):
                    st.success(
                        "V161 · QC anatómico superado: sacro posterior, hallux bilateral medial y "
                        "neutro del tobillo validado antes del render."
                    )
                    with st.expander("V161 · Detalle del control anatómico",expanded=False):
                        st.json(_aqc)
                _DCjson={
                    "A":np.round(np.asarray(_DC["affines"],dtype=np.float32),6).tolist(),
                    "BV0":np.round(np.asarray(_DC["bone_vertices0"],dtype=np.float32),5).tolist(),
                    "BF":np.asarray(_DC["bone_faces"],dtype=np.int32).tolist(),
                    "BS":np.asarray(_DC["bone_seg"],dtype=np.int16).tolist(),
                    "MV0":np.round(np.asarray(_DC["muscle_vertices0"],dtype=np.float32),5).tolist(),
                    "MF":np.asarray(_DC["muscle_faces"],dtype=np.int32).tolist(),
                    "MS0":np.asarray(_DC["muscle_seg0"],dtype=np.int16).tolist(),
                    "MS1":np.asarray(_DC["muscle_seg1"],dtype=np.int16).tolist(),
                    "MH":np.asarray(_DC["muscle_host"],dtype=np.int16).tolist(),
                    "MWA":np.round(np.asarray(_DC["muscle_wa"],dtype=np.float32),4).tolist(),
                    "MWB":np.round(np.asarray(_DC["muscle_wb"],dtype=np.float32),4).tolist(),
                    "MWH":np.round(np.asarray(_DC["muscle_wh"],dtype=np.float32),4).tolist(),
                }
                _ubv=np.round(np.asarray(_UBV,dtype=np.float32),5).tolist() if _UBV is not None else None
                _ubf=np.asarray(_UBF,dtype=np.int32)
                _dbf=np.asarray(_DC["bone_faces"],dtype=np.int32)
                if _UBV is not None and _UBF is not None and len(_UBF):
                    _BFbrowser=np.vstack([_dbf,_ubf+len(_DC["bone_vertices0"])]).astype(np.int32)
                else:
                    _BFbrowser=_dbf
                _vmfbrowser=np.asarray(_MR["faces"],dtype=np.int32) if _MR is not None else np.asarray(_DC["muscle_faces"],dtype=np.int32)
                _bvjson=None
            else:
                _DCjson=None; _ubv=None
                if _BV is not None and _BF is not None:
                    _BFbrowser=np.asarray(_BF,dtype=np.int32)
                    _bvjson=np.round(_BV,5).tolist()
                else:
                    _BFbrowser=np.empty((0,3),dtype=np.int32)
                    _bvjson=[[] for _ in range(len(_SVc))]
                _vmfbrowser=None

            _payload=_json.dumps({
                "SV":np.round(_SVc,5).tolist(),"SF":_SFc.tolist(),
                "BV":_bvjson,"BF":_BFbrowser.tolist(),
                "DC":_DCjson,"MR":_MRjson,"UBV":_ubv,
                "M":_mus_frames,
                "VMV":None,
                "VMF":_vmfbrowser.tolist() if _vmfbrowser is not None else None,
                "ids":[int(x) for x in ids],"ranges":_ranges
            },separators=(",",":"))
            _js=_get_plotlyjs()

            _html=f"""<!doctype html>
<html>
<head>
<meta charset='utf-8'>
<style>
html,body{{margin:0;background:white;font-family:system-ui;color:#1f2937}}
#bar{{display:flex;gap:6px;align-items:center;padding:6px;flex-wrap:wrap;border-bottom:1px solid #e5e7eb}}
button{{padding:6px 10px;border:1px solid #cbd5e1;background:white;border-radius:5px;cursor:pointer}}
button:hover{{background:#f1f5f9}}
button.active{{background:#e2e8f0;font-weight:700}}
#counter{{font-weight:700;min-width:105px}}
#slider{{flex:1;min-width:260px}}
#plot{{height:730px;width:100%}}
#status{{font-size:12px;color:#475569;padding:4px 8px}}
</style>
<script>{_js}</script>
</head>
<body>
<div id='bar'>
  <button onclick='layer("skin")'>🧍 Piel</button>
  <button id='skinOpaque' onclick='skinMode(1)'>Opaca</button>
  <button id='skinTrans' onclick='skinMode(.24)' class='active'>Translúcida</button>
  <button onclick='layer("bone")'>🦴 Esqueleto</button>
  <button onclick='layer("muscle")'>💪 Músculo</button>
  <button onclick='layer("skinbone")'>🧍+🦴</button>
  <button onclick='layer("bone-muscle")'>🦴+💪</button>
  <button onclick='layer("all")' class='active'>Todas</button>
  <span style='width:8px'></span>
  <button onclick='play(0.5)'>▶0.5×</button>
  <button onclick='play(1)'>▶1×</button>
  <button onclick='play(2)'>▶2×</button>
  <button onclick='pause()'>⏸</button>
  <button id='multiBtn' onclick='toggleMulti()'>🎥 Ver en varios planos</button>
  <button id='recBtn' onclick='recordCurrentView()'>⬇ Descargar vídeo tal como se ve</button>
  <b id='counter'>Frame 1/{len(ids)}</b>
  <input id='slider' type='range' min='0' max='{max(len(ids)-1,0)}' value='0'>
</div>
<div id='status'>Coloca el modelo con el ratón como quieras. Pulsa reproducir y, cuando esté como deseas, descarga el vídeo exactamente con esa orientación.</div>
<div id='plot'></div>
<script>
const P={_payload};
const gd=document.getElementById('plot'),
      sl=document.getElementById('slider'),
      co=document.getElementById('counter'),
      st=document.getElementById('status'),
      multiBtn=document.getElementById('multiBtn'),
      recBtn=document.getElementById('recBtn');

let n=0,on=false,speed=1,last=0,raf=0,multiplano=false,currentLayer='all';
const sfi=P.SF.map(x=>x[0]),sfj=P.SF.map(x=>x[1]),sfk=P.SF.map(x=>x[2]);
const bfi=P.BF.map(x=>x[0]),bfj=P.BF.map(x=>x[1]),bfk=P.BF.map(x=>x[2]);
const vmfi=P.VMF?P.VMF.map(x=>x[0]):[], vmfj=P.VMF?P.VMF.map(x=>x[1]):[], vmfk=P.VMF?P.VMF.map(x=>x[2]):[];

function xyz(A,k){{const q=A[k];return[q.map(p=>p[0]),q.map(p=>p[1]),q.map(p=>p[2])]}}
function applyAffine(p,t){{
  return [
    t[0]*p[0]+t[1]*p[1]+t[2]*p[2]+t[9],
    t[3]*p[0]+t[4]*p[1]+t[5]*p[2]+t[10],
    t[6]*p[0]+t[7]*p[1]+t[8]*p[2]+t[11]
  ]
}}
function boneXYZ(k){{
  if(!P.DC) return xyz(P.BV,k);
  const A=P.DC.A[k], V=P.DC.BV0, S=P.DC.BS;
  let X=new Array(V.length),Y=new Array(V.length),Z=new Array(V.length);
  for(let i=0;i<V.length;i++){{const q=applyAffine(V[i],A[S[i]]);X[i]=q[0];Y[i]=q[1];Z[i]=q[2]}}
  if(P.UBV){{const u=P.UBV[k];for(const q of u){{X.push(q[0]);Y.push(q[1]);Z.push(q[2])}}}}
  return [X,Y,Z]
}}
function volMuscleXYZ(k){{
  if(P.MR){{
    const R=P.MR, Vn=R.U.length, K=R.PS[0][k].length;
    let X=new Array(Vn),Y=new Array(Vn),Z=new Array(Vn);
    for(let i=0;i<Vn;i++){{
      const mi=R.MI[i], u=R.U[i], q=R.S[mi][k]+u*(R.E[mi][k]-R.S[mi][k]);
      const a=Math.max(0,Math.min(K-2,Math.floor(q*(K-1)))), f=q*(K-1)-a;
      const p0=R.PS[mi][k][a],p1=R.PS[mi][k][a+1],n0=R.PN[mi][k][a],n1=R.PN[mi][k][a+1],b0=R.PB[mi][k][a],b1=R.PB[mi][k][a+1];
      const cx=p0[0]*(1-f)+p1[0]*f,cy=p0[1]*(1-f)+p1[1]*f,cz=p0[2]*(1-f)+p1[2]*f;
      let nx=n0[0]*(1-f)+n1[0]*f,ny=n0[1]*(1-f)+n1[1]*f,nz=n0[2]*(1-f)+n1[2]*f;
      let bx=b0[0]*(1-f)+b1[0]*f,by=b0[1]*(1-f)+b1[1]*f,bz=b0[2]*(1-f)+b1[2]*f;
      let nn=Math.hypot(nx,ny,nz)||1;nx/=nn;ny/=nn;nz/=nn;
      let bn=Math.hypot(bx,by,bz)||1;bx/=bn;by/=bn;bz/=bn;
      const sc=R.TS[mi][k], tx=R.T[i][0]*sc,ty=R.T[i][1]*sc;
      X[i]=cx+nx*tx+bx*ty;Y[i]=cy+ny*tx+by*ty;Z[i]=cz+nz*tx+bz*ty;
    }}
    return [X,Y,Z]
  }}
  if(!P.DC) return P.VMV?xyz(P.VMV,k):[[],[],[]];
  const D=P.DC,A=D.A[k],V=D.MV0;
  let X=new Array(V.length),Y=new Array(V.length),Z=new Array(V.length);
  for(let i=0;i<V.length;i++){{
    const ph=applyAffine(V[i],A[D.MH[i]]), pa=applyAffine(V[i],A[D.MS0[i]]), pb=applyAffine(V[i],A[D.MS1[i]]);
    const wh=D.MWH[i],wa=D.MWA[i],wb=D.MWB[i];
    X[i]=wh*ph[0]+wa*pa[0]+wb*pb[0];Y[i]=wh*ph[1]+wa*pa[1]+wb*pb[1];Z[i]=wh*ph[2]+wa*pa[2]+wb*pb[2];
  }}
  return [X,Y,Z]
}}
function mus(k){{
  let X=[],Y=[],Z=[];
  for(const q of P.M[k]){{
    for(const p of q){{X.push(p[0]);Y.push(p[1]);Z.push(p[2])}}
    X.push(null);Y.push(null);Z.push(null)
  }}
  return[X,Y,Z]
}}

const hasVolMus=!!P.MR || !!P.DC || !!P.VMV;
let s=xyz(P.SV,0),b=boneXYZ(0),m=mus(0),vm=volMuscleXYZ(0);
const data=[
  {{type:'mesh3d',x:s[0],y:s[1],z:s[2],i:sfi,j:sfj,k:sfk,name:'Piel SKEL',
    color:'rgb(61,184,218)',opacity:.24,visible:true,flatshading:false,showscale:false,
    lighting:{{ambient:.58,diffuse:.88,specular:.22,roughness:.72,fresnel:.08}},
    lightposition:{{x:100,y:-120,z:180}},hoverinfo:'skip'}},
  {{type:'mesh3d',x:b[0],y:b[1],z:b[2],i:bfi,j:bfj,k:bfk,name:'Esqueleto anatómico',
    opacity:1,visible:true,flatshading:false,showscale:false}},
  {{type:'mesh3d',x:vm[0],y:vm[1],z:vm[2],i:vmfi,j:vmfj,k:vmfk,name:'Músculo Denver volumétrico',
    opacity:.62,visible:hasVolMus,flatshading:false,showscale:false}},
  {{type:'scatter3d',mode:'lines',x:m[0],y:m[1],z:m[2],name:'Músculo OpenSim funcional',
    line:{{width:7}},visible:!hasVolMus}}
];

Plotly.newPlot(gd,data,{{
  title:'Piel + Esqueleto + Músculo',
  margin:{{l:0,r:0,t:42,b:0}},
  scene:{{
    aspectmode:'data',
    xaxis:{{range:P.ranges[0]}},
    yaxis:{{range:P.ranges[1]}},
    zaxis:{{range:P.ranges[2]}}
  }},
  showlegend:false
}},{{responsive:true,displaylogo:false,scrollZoom:true}});

window.layer=function(mode){{
  currentLayer=mode;
  Plotly.restyle(gd,{{visible:['skin','skinbone','all'].includes(mode)}},[0]);
  Plotly.restyle(gd,{{visible:['bone','skinbone','bone-muscle','all'].includes(mode)}},[1]);
  const musOn=['muscle','bone-muscle','all'].includes(mode);
  Plotly.restyle(gd,{{visible:musOn && hasVolMus}},[2]);
  Plotly.restyle(gd,{{visible:musOn && !hasVolMus}},[3]);
}};
window.skinMode=function(op){{
  Plotly.restyle(gd,{{opacity:op}},[0]);
  document.getElementById('skinOpaque').classList.toggle('active',op>=.99);
  document.getElementById('skinTrans').classList.toggle('active',op<.99);
}};

function frame(k){{
  n=(k+P.SV.length)%P.SV.length;
  s=xyz(P.SV,n);b=boneXYZ(n);m=mus(n);if(hasVolMus)vm=volMuscleXYZ(n);
  Plotly.restyle(gd,{{x:[s[0]],y:[s[1]],z:[s[2]]}},[0]);
  Plotly.restyle(gd,{{x:[b[0]],y:[b[1]],z:[b[2]]}},[1]);
  if(hasVolMus) Plotly.restyle(gd,{{x:[vm[0]],y:[vm[1]],z:[vm[2]]}},[2]);
  else Plotly.restyle(gd,{{x:[m[0]],y:[m[1]],z:[m[2]]}},[3]);
  sl.value=n;co.textContent='Frame '+P.ids[n]+'/'+P.ids[P.ids.length-1];
  if(multiplano) applyMultiviewCamera(n);
}}

sl.oninput=()=>{{on=false;frame(+sl.value)}};

function loop(t){{
  if(!on)return;
  if(!last)last=t;
  if(t-last>1000/(25*speed)){{
    frame(n+1);last=t
  }}
  raf=requestAnimationFrame(loop)
}}
window.play=function(v){{
  speed=v;on=true;last=0;cancelAnimationFrame(raf);raf=requestAnimationFrame(loop)
}};
window.pause=function(){{on=false;cancelAnimationFrame(raf)}};

function currentCamera(){{
  try{{return JSON.parse(JSON.stringify(gd._fullLayout.scene.camera))}}
  catch(e){{return null}}
}}

/* Multi-plane playback: camera changes by gait-cycle sector, while frame animation continues. */
const cams=[
 {{eye:{{x:0,y:-2.35,z:.15}},up:{{x:0,y:0,z:1}}}},    // frontal
 {{eye:{{x:2.35,y:0,z:.15}},up:{{x:0,y:0,z:1}}}},     // lateral right
 {{eye:{{x:0,y:2.35,z:.15}},up:{{x:0,y:0,z:1}}}},     // posterior
 {{eye:{{x:-2.35,y:0,z:.15}},up:{{x:0,y:0,z:1}}}},    // lateral left
 {{eye:{{x:0,y:0,z:2.55}},up:{{x:0,y:1,z:0}}}}        // superior
];
function applyMultiviewCamera(k){{
  const sector=Math.min(cams.length-1,Math.floor(k*cams.length/P.SV.length));
  Plotly.relayout(gd,{{'scene.camera':cams[sector]}});
}}
window.toggleMulti=function(){{
  multiplano=!multiplano;
  multiBtn.classList.toggle('active',multiplano);
  st.textContent=multiplano
    ? 'Modo multiplano activo: frontal → lateral → posterior → lateral → superior.'
    : 'Vista libre: mueve el modelo con el ratón. La descarga conserva esta orientación.';
  if(multiplano) applyMultiviewCamera(n);
}};

/* Exact-view recording in the browser.
   It records the Plotly canvas/DOM result while the same viewer advances frames.
   Chrome 153+ generally supports MP4 MediaRecorder; fallback is WebM if unavailable. */
async function renderFrameForExport(k,cam0,multi0){{
  n=(k+P.SV.length)%P.SV.length;
  s=xyz(P.SV,n); b=boneXYZ(n); m=mus(n); if(hasVolMus)vm=volMuscleXYZ(n);

  const jobs=[
    Plotly.restyle(gd,{{x:[s[0]],y:[s[1]],z:[s[2]]}},[0]),
    Plotly.restyle(gd,{{x:[b[0]],y:[b[1]],z:[b[2]]}},[1])
  ];
  if(hasVolMus) jobs.push(Plotly.restyle(gd,{{x:[vm[0]],y:[vm[1]],z:[vm[2]]}},[2]));
  else jobs.push(Plotly.restyle(gd,{{x:[m[0]],y:[m[1]],z:[m[2]]}},[3]));
  await Promise.all(jobs);

  if(multi0){{
    const sector=Math.min(cams.length-1,Math.floor(k*cams.length/P.SV.length));
    await Plotly.relayout(gd,{{'scene.camera':cams[sector]}});
  }} else if(cam0){{
    await Plotly.relayout(gd,{{'scene.camera':cam0}});
  }}

  /* Wait for Plotly/WebGL to finish the visual update before rasterizing. */
  await new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));
  sl.value=n; co.textContent='Frame '+P.ids[n]+'/'+P.ids[P.ids.length-1];
}}

async function loadMp4Muxer(){{
  /* Small browser-side muxer. It does not alter the model or rerun SKEL. */
  return await import('https://cdn.jsdelivr.net/npm/mp4-muxer@5.2.2/+esm');
}}

async function recordCurrentView(){{
  if(recBtn.disabled)return;
  recBtn.disabled=true;

  const wasOn=on;
  on=false;
  cancelAnimationFrame(raf);

  const startFrame=n;
  const cam0=currentCamera();
  const multi0=multiplano;
  const exportSpeed=(Number(speed)>0?Number(speed):1.0);

  st.textContent='Preparando exportación MP4 a velocidad fija…';

  try{{
    if(!('VideoEncoder' in window) || !('VideoFrame' in window)){{
      throw new Error('Este navegador no ofrece WebCodecs VideoEncoder. Usa Chrome/Edge actualizado.');
    }}

    const mod=await loadMp4Muxer();
    const Muxer=mod.Muxer;
    const ArrayBufferTarget=mod.ArrayBufferTarget;

    /* 720p square keeps good anatomical detail without making Plotly export excessively heavy. */
    const W=720,H=720;
    const cv=document.createElement('canvas');
    cv.width=W; cv.height=H;
    const ctx=cv.getContext('2d',{{alpha:false,desynchronized:false}});

    const target=new ArrayBufferTarget();
    const muxer=new Muxer({{
      target,
      video:{{codec:'avc',width:W,height:H}},
      fastStart:'in-memory',
      firstTimestampBehavior:'offset'
    }});

    let encoderError=null;
    const encoder=new VideoEncoder({{
      output:(chunk,meta)=>muxer.addVideoChunk(chunk,meta),
      error:e=>{{encoderError=e;}}
    }});

    const support=await VideoEncoder.isConfigSupported({{
      codec:'avc1.42001f',
      width:W,height:H,
      bitrate:6000000,
      framerate:25,
      avc:{{format:'avc'}}
    }});
    if(!support.supported) throw new Error('El navegador no admite codificación H.264/AVC mediante WebCodecs.');

    encoder.configure(support.config);

    /*
      IMPORTANT:
      Output timestamps are mathematical, not wall-clock capture times.
      Therefore Plotly may take 0.2 s or 2 s to render a frame and the final
      video still plays at the requested gait speed.
    */
    const baseFps=25.0;
    const frameDurationUs=Math.round(1000000.0/(baseFps*exportSpeed));

    for(let kk=0;kk<P.SV.length;kk++){{
      await renderFrameForExport(kk,cam0,multi0);

      const url=await Plotly.toImage(gd,{{format:'png',width:W,height:H,scale:1}});
      const im=new Image();
      await new Promise((resolve,reject)=>{{
        im.onload=resolve; im.onerror=reject; im.src=url;
      }});

      ctx.fillStyle='white';
      ctx.fillRect(0,0,W,H);
      ctx.drawImage(im,0,0,W,H);

      const vf=new VideoFrame(cv,{{
        timestamp:kk*frameDurationUs,
        duration:frameDurationUs
      }});
      encoder.encode(vf,{{keyFrame:(kk===0 || kk%25===0)}});
      vf.close();

      st.textContent='Codificando marcha '+(kk+1)+'/'+P.SV.length+
        ' · '+exportSpeed+'× · cámara '+(multi0?'multiplano':'actual');
    }}

    await encoder.flush();
    if(encoderError) throw encoderError;
    encoder.close();
    muxer.finalize();

    const blob=new Blob([target.buffer],{{type:'video/mp4'}});
    const a=document.createElement('a');
    a.href=URL.createObjectURL(blob);
    a.download='PhysioSentinel_'+currentLayer+'_'+
      (multi0?'multiplano':'vista_actual')+'_'+exportSpeed+'x.mp4';
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(()=>URL.revokeObjectURL(a.href),15000);

    const duration=(P.SV.length/(baseFps*exportSpeed)).toFixed(2);
    st.textContent='✓ MP4 descargado · '+P.SV.length+' frames · duración '+duration+
      ' s · orientación conservada · movimiento real.';
  }}catch(e){{
    console.error(e);
    st.textContent='No se pudo generar el MP4 de FPS fijo: '+e.message;
  }}finally{{
    await renderFrameForExport(startFrame,cam0,multi0);
    if(wasOn)play(exportSpeed);
    recBtn.disabled=false;
  }}
}}
</script>
</body></html>"""

            # Versión 149 — CRITICAL FLOW FIX:
            # the unified viewer HTML must actually be mounted before Tab 12.
            st.session_state["v3227_tab11_viewer_ready"]=True
            components.html(_html, height=805, scrolling=False)

    except Exception as exc:
        st.warning(f"No se pudo abrir el visor Anatomical Atlas Mesh: {exc}")

def _calibrate_skel_dofs(torch, model, delta=0.12, progress_cb=None):
    qn=int(model.num_q_params)
    betas=torch.zeros((1,int(model.num_betas)),dtype=torch.float32,device='cpu')
    zero_trans=torch.zeros((1,3),dtype=torch.float32,device='cpu')
    names=_skel_forward_joint_names_140(model)
    pose0=torch.zeros((1,qn),dtype=torch.float32,device='cpu')
    with torch.no_grad():
        base=model(pose0,betas,zero_trans,skelmesh=False).joints[0].detach().cpu().numpy().astype(np.float64)
    rows=[]; d=float(delta); jac=np.zeros((base.shape[0],3,qn),dtype=np.float64)
    for qi in range(qn):
        pp=pose0.clone(); pm=pose0.clone(); pp[0,qi]=d; pm[0,qi]=-d
        with torch.no_grad():
            jp=model(pp,betas,zero_trans,skelmesh=False).joints[0].detach().cpu().numpy().astype(np.float64)
            jm=model(pm,betas,zero_trans,skelmesh=False).joints[0].detach().cpu().numpy().astype(np.float64)
        deriv=(jp-jm)/(2.0*d); jac[:,:,qi]=deriv
        mag=np.linalg.norm(deriv,axis=1); order=np.argsort(-mag)
        mx=float(mag[order[0]]) if len(order) else 0.0
        moved=[int(i) for i in order if mag[i] >= max(mx*0.20,1e-5)][:8]
        top=[f"{names[i] if i < len(names) else i}:{mag[i]:.4f}" for i in order[:5]]
        if len(order):
            v=deriv[order[0]]; ax=int(np.argmax(np.abs(v))); axis='XYZ'[ax]; sign='+' if float(v[ax])>=0 else '-'
        else: axis='—'; sign='—'
        rows.append({'q_index':qi,'q_label':f'q{qi:02d}','sensibilidad_max_joint_u_por_rad':mx,
                     'joint_mas_sensible':names[order[0]] if len(order) and order[0] < len(names) else '—',
                     'eje_global_dominante_top_joint':axis,'signo_top_joint':sign,
                     'joints_afectados_20pct':', '.join(names[i] if i < len(names) else str(i) for i in moved),
                     'top5_joint_sensitivity':'; '.join(top),'efecto_detectable':bool(mx>1e-5)})
        if progress_cb: progress_cb(qi+1,qn,mx)
    profile={'jacobian':jac,'base_joints':base,'delta_rad':d,'joint_names':names}
    return pd.DataFrame(rows), profile

def _dof_calibration_json_bytes(df, profile, model):
    payload={'version':'110.3.13','method':'empirical central finite-difference around neutral pose',
             'delta_rad':float(profile.get('delta_rad',0.12)),'num_q_params':int(model.num_q_params),
             'joint_names':_skel_forward_joint_names_140(model),
             'warning':'Los nombres biomecanicos previos de q no se asumen. V110.3.13 usa el Jacobiano medido para seleccionar DOF observables.',
             'dofs':df.to_dict(orient='records'),
             'jacobian_joint_xyz_per_rad':np.asarray(profile.get('jacobian'),float).tolist()}
    return json.dumps(payload,ensure_ascii=False,indent=2).encode('utf-8')

def _skel_edges_from_names(names):
    norm={_simple_name(n):i for i,n in enumerate(names)}
    edge_names=[
        ('pelvis','femur_r'),('femur_r','tibia_r'),('tibia_r','talus_r'),('talus_r','calcn_r'),('calcn_r','toes_r'),
        ('pelvis','femur_l'),('femur_l','tibia_l'),('tibia_l','talus_l'),('talus_l','calcn_l'),('calcn_l','toes_l'),
        ('pelvis','lumbar_body'),('lumbar_body','thorax'),('thorax','head'),
        ('thorax','scapula_r'),('scapula_r','humerus_r'),('humerus_r','ulna_r'),('ulna_r','radius_r'),('radius_r','hand_r'),
        ('thorax','scapula_l'),('scapula_l','humerus_l'),('humerus_l','ulna_l'),('ulna_l','radius_l'),('radius_l','hand_l')
    ]
    return [(norm[_simple_name(a)],norm[_simple_name(b)]) for a,b in edge_names if _simple_name(a) in norm and _simple_name(b) in norm]

def _plot_xyz_static(points,title,ranges=None,camera=None):
    import plotly.graph_objects as go
    links=[('LShoulder','RShoulder'),('LShoulder','LElbow'),('LElbow','LWrist'),('RShoulder','RElbow'),('RElbow','RWrist'),
           ('LShoulder','LHip'),('RShoulder','RHip'),('LHip','RHip'),('LHip','LKnee'),('LKnee','LAnkle'),('RHip','RKnee'),
           ('RKnee','RAnkle'),('Neck','LShoulder'),('Neck','RShoulder'),('Neck','Head')]
    fig=go.Figure()
    for a,b in links:
        if a in points and b in points:
            A=np.asarray(points[a],float); B=np.asarray(points[b],float)
            fig.add_trace(go.Scatter3d(x=[A[0],B[0]],y=[A[1],B[1]],z=[A[2],B[2]],mode='lines',showlegend=False,hoverinfo='skip'))
    labs=[j for j in JOINTS if j in points]
    if labs:
        P=np.asarray([points[j] for j in labs],float)
        fig.add_trace(go.Scatter3d(x=P[:,0],y=P[:,1],z=P[:,2],mode='markers',text=labs,name='XYZ V104/V107'))
    scene=dict(aspectmode='data')
    if ranges is not None: scene.update(xaxis=dict(range=ranges[0]),yaxis=dict(range=ranges[1]),zaxis=dict(range=ranges[2]))
    if camera is not None: scene['camera']=camera
    fig.update_layout(height=500,margin=dict(l=0,r=0,t=42,b=0),title=title,scene=scene,showlegend=False)
    return fig

def _plot_joints_static(J,names,title,target_points=None,ranges=None,camera=None):
    import plotly.graph_objects as go
    J=np.asarray(J,float); edges=_skel_edges_from_names(names); xs=[];ys=[];zs=[]
    for a,b in edges: xs += [J[a,0],J[b,0],None]; ys += [J[a,1],J[b,1],None]; zs += [J[a,2],J[b,2],None]
    fig=go.Figure([go.Scatter3d(x=xs,y=ys,z=zs,mode='lines+markers',name='Joints SKEL')])
    if target_points:
        for target,ji in _SKEL24_TARGET_INDEX.items():
            if target not in target_points or int(ji)>=len(J): continue
            T=np.asarray(target_points[target],float); S=J[int(ji)]
            fig.add_trace(go.Scatter3d(x=[T[0],S[0]],y=[T[1],S[1]],z=[T[2],S[2]],mode='lines',showlegend=False,hoverinfo='skip'))
    scene=dict(aspectmode='data')
    if ranges is not None: scene.update(xaxis=dict(range=ranges[0]),yaxis=dict(range=ranges[1]),zaxis=dict(range=ranges[2]))
    if camera is not None: scene['camera']=camera
    fig.update_layout(height=500,margin=dict(l=0,r=0,t=42,b=0),title=title,scene=scene,showlegend=False)
    return fig

def _plot_mesh_static(V,F,title,J=None,names=None,ranges=None,camera=None):
    import plotly.graph_objects as go
    V=np.asarray(V,float); F=np.asarray(F,int)
    fig=go.Figure([go.Mesh3d(x=V[:,0],y=V[:,1],z=V[:,2],i=F[:,0],j=F[:,1],k=F[:,2],opacity=0.82,name='SKEL skin',flatshading=False)])
    if J is not None and names:
        J=np.asarray(J,float); xs=[];ys=[];zs=[]
        for a,b in _skel_edges_from_names(names): xs += [J[a,0],J[b,0],None]; ys += [J[a,1],J[b,1],None]; zs += [J[a,2],J[b,2],None]
        fig.add_trace(go.Scatter3d(x=xs,y=ys,z=zs,mode='lines+markers',name='Joints'))
    scene=dict(aspectmode='data')
    if ranges is not None: scene.update(xaxis=dict(range=ranges[0]),yaxis=dict(range=ranges[1]),zaxis=dict(range=ranges[2]))
    if camera is not None: scene['camera']=camera
    fig.update_layout(height=500,margin=dict(l=0,r=0,t=42,b=0),title=title,scene=scene,showlegend=False)
    return fig

def _side_by_side_validation(motion,seq,mesh_seq):
    ok=[x for x in (seq or {}).get('frames',[]) if x.get('status')=='ok' and x.get('joints') is not None]
    if not ok: return
    fmap={int(x.get('frame')):x for x in ok}
    mesh_ids=[int(x) for x in np.asarray(mesh_seq.get('frame_ids',[])).tolist()] if isinstance(mesh_seq,dict) else []
    common=[f for f in sorted(fmap) if (not mesh_ids or f in mesh_ids)]
    if not common: return
    fid=st.slider('Frame sincronizado para validación',min_value=min(common),max_value=max(common),value=common[0],step=1,key='v110_3_13_sync_frame')
    if fid not in fmap: fid=min(common,key=lambda x:abs(x-fid))
    fr=fmap[fid]; mframes=list((motion or {}).get('frames') or [])
    points=_frame_points_from_frame(mframes[max(0,min(len(mframes)-1,fid-1))]) if mframes else {}
    J=np.asarray(fr['joints'],float); names=[str(x) for x in seq.get('joint_names',[])]
    vals=[]
    if points: vals.extend(np.asarray(list(points.values()),float).reshape(-1,3).tolist())
    vals.extend(J.tolist()); V=F=None
    if isinstance(mesh_seq,dict) and fid in mesh_ids:
        mi=mesh_ids.index(fid); V=np.asarray(mesh_seq['vertices'][mi],float); F=np.asarray(mesh_seq['faces'],int)
        vals.extend(np.percentile(V,[2,98],axis=0).tolist())
    A=np.asarray(vals,float); mn=np.nanmin(A,axis=0); mx=np.nanmax(A,axis=0); sp=np.maximum(mx-mn,0.2); pad=0.08*sp
    ranges=[(float(mn[i]-pad[i]),float(mx[i]+pad[i])) for i in range(3)]
    cam=dict(eye=dict(x=1.45,y=1.45,z=1.05),up=dict(x=0,y=0,z=1))
    c1,c2,c3=st.columns(3)
    with c1: st.plotly_chart(_plot_xyz_static(points,f'Frame {fid} · XYZ simplificado',ranges,cam),use_container_width=True,key=f'v124_xyz_{fid}')
    with c2: st.plotly_chart(_plot_joints_static(J,names,f'Frame {fid} · SKEL joints',points,ranges,cam),use_container_width=True,key=f'v124_joints_{fid}')
    with c3:
        if V is not None: st.plotly_chart(_plot_mesh_static(V,F,f'Frame {fid} · SKEL mesh',J,names,ranges,cam),use_container_width=True,key=f'v124_mesh_{fid}')
        else: st.info('Genera primero la malla SKEL para completar la tercera columna.')
    rows=[]; norm={_simple_name(n):i for i,n in enumerate(names)}
    for target,cands in _SKEL_TARGET_CANDIDATES.items():
        if target not in points: continue
        ji=next((norm[_simple_name(c)] for c in cands if _simple_name(c) in norm),None)
        if ji is not None: rows.append({'landmark':target,'skel_joint':names[ji],'error_3D':float(np.linalg.norm(np.asarray(points[target],float)-J[ji]))})
    if rows:
        edf=pd.DataFrame(rows).sort_values('error_3D',ascending=False); a,b,c=st.columns(3)
        a.metric('Error medio frame',f"{edf['error_3D'].mean():.4f}"); b.metric('Error máximo frame',f"{edf['error_3D'].max():.4f}"); c.metric('Landmark peor',str(edf.iloc[0]['landmark']))
        with st.expander('Errores XYZ ↔ SKEL del frame sincronizado',expanded=False): st.dataframe(edf,use_container_width=True,hide_index=True)



def _patient_gender_145():
    """V110.3.19.3 · Sexo canónico de la ficha general, sin widget SKEL paralelo.

    `patient_sex` es la única fuente de verdad. El selector general está fuera del
    formulario y se sincroniza inmediatamente en cada rerun.
    """
    raw = st.session_state.get("patient_sex", "No especificado")
    sex = str(raw or "No especificado").strip().lower()
    if sex in {"mujer","female","femenino","femenina"}:
        st.session_state["patient_sex"] = "Mujer"
        return "female", "Mujer"
    if sex in {"hombre","male","masculino"}:
        st.session_state["patient_sex"] = "Hombre"
        return "male", "Hombre"
    return None, str(raw or "No especificado")


def _hardwired_lower_limb_temporal_update_146(torch,model,pose,rotvec,trans,scale,target_df,rows,betas,zero_trans,prev_pose,axis_map):
    """V110.3.18 · actualización temporal dura de miembros inferiores.

    No depende del optimizador global ni de un rollback que pueda dejar q de piernas
    idénticos frame a frame. Resuelve por Gauss-Newton amortiguado, directamente en XY,
    los q proximales/distales observables de cada pierna a partir de RHip/RKnee/RAnkle
    y LHip/LKnee/LAnkle. El resultado se escribe EN LA MISMA `pose` que después se
    persiste en la secuencia y alimenta skin_verts(t).
    """
    names=_skel_forward_joint_names_140(model); nm={_simple_name(n):i for i,n in enumerate(names)}
    tmap={r.joint:np.array([r.x,r.y,r.z],np.float32) for r in target_df.itertuples()}
    chains=[
        ('R',[3,4,5,6,7], [('RHip','femur_r'),('RKnee','tibia_r'),('RAnkle','talus_r')]),
        ('L',[10,11,12,13,14],[('LHip','femur_l'),('LKnee','tibia_l'),('LAnkle','talus_l')]),
    ]
    audit={'accepted':False,'strategy':'hard-wired damped Gauss-Newton XY','chains':{}}
    any_commit=False
    for side,ids,pairs0 in chains:
        pairs=[(lm,nm.get(_simple_name(jn))) for lm,jn in pairs0 if lm in tmap and nm.get(_simple_name(jn)) is not None]
        if len(pairs)<2:
            audit['chains'][side]={'accepted':False,'reason':'landmarks insuficientes'}; continue
        work=pose.detach().clone()
        def chain_xy(p):
            with torch.no_grad():
                world,_,_=_direct_world_130(torch,model,p,betas,zero_trans,rotvec,scale,trans,axis_map=axis_map)
            pred=np.stack([world[ji,:2].cpu().numpy() for _,ji in pairs],axis=0)
            tgt=np.stack([tmap[lm][:2] for lm,_ in pairs],axis=0)
            return pred,tgt,float(np.sqrt(np.mean(np.sum((pred-tgt)**2,axis=1))))
        pred0,tgt0,before=chain_xy(work)
        best=work.clone(); best_rmse=before; q0=work[0,ids].cpu().numpy().copy()
        eps=0.025
        for it in range(6):
            pred,tgt,cur=chain_xy(work)
            r=(tgt-pred).reshape(-1).astype(np.float64)
            J=np.zeros((r.size,len(ids)),np.float64)
            for k,qi in enumerate(ids):
                pp=work.clone(); pm=work.clone()
                pp[0,qi]+=eps; pm[0,qi]-=eps; _clamp_direct_pose_130(torch,pp); _clamp_direct_pose_130(torch,pm)
                yp,_,_=chain_xy(pp); ym,_,_=chain_xy(pm)
                J[:,k]=((yp-ym)/(2.0*eps)).reshape(-1)
            # DLS: actualización pequeña pero obligatoriamente conectada al target XY actual.
            lam=0.025 + 0.01*it
            A=J.T@J + lam*np.eye(len(ids)); b=J.T@r
            try: dq=np.linalg.solve(A,b)
            except Exception: dq=np.linalg.lstsq(A,b,rcond=1e-5)[0]
            dq=np.clip(dq,-0.22,0.22)
            # continuidad: limita salto frente a pose previa, pero no la inmoviliza.
            proposed=work.clone()
            with torch.no_grad():
                for k,qi in enumerate(ids): proposed[0,qi]+=float(dq[k])
                _clamp_direct_pose_130(torch,proposed)
                prevv=prev_pose[0,ids]
                dv=torch.clamp(proposed[0,ids]-prevv,-0.38,0.38)
                proposed[0,ids]=prevv+dv
                _clamp_direct_pose_130(torch,proposed)
            _,_,prmse=chain_xy(proposed)
            if prmse < best_rmse-1e-6:
                best=proposed.clone(); best_rmse=prmse; work=proposed
            else:
                # trust-region menor, sin revertir toda la cadena al valor previo.
                proposed=work.clone()
                with torch.no_grad():
                    for k,qi in enumerate(ids): proposed[0,qi]+=float(0.35*dq[k])
                    _clamp_direct_pose_130(torch,proposed)
                _,_,prmse2=chain_xy(proposed)
                if prmse2 < best_rmse-1e-6:
                    best=proposed.clone(); best_rmse=prmse2; work=proposed
                else:
                    break
        q1=best[0,ids].cpu().numpy().copy(); qdelta=float(np.linalg.norm(q1-q0))
        # commit si mejora o si genera una variación real minúscula con error no peor.
        accepted=bool(best_rmse <= before+1e-7 and qdelta>1e-7)
        if accepted:
            with torch.no_grad(): pose[0,ids]=best[0,ids]
            any_commit=True
        audit['chains'][side]={'accepted':accepted,'rmse_xy_before':before,'rmse_xy_after':best_rmse,
                               'improvement':float(before-best_rmse),'q_delta_norm':qdelta,
                               'q_indices':ids,'q_before':q0.astype(float).tolist(),'q_after':q1.astype(float).tolist()}
    audit['accepted']=any_commit
    return pose,audit


def _target_lower_limb_motion_146(motion):
    """Rango de movimiento de landmarks inferiores en la secuencia objetivo V104/V107."""
    frames=list((motion or {}).get('frames') or [])
    vals={k:[] for k in ['RKnee','RAnkle','LKnee','LAnkle']}
    for fi,f in enumerate(frames):
        try: df=_target_df_for_frame(f,fi)
        except Exception: continue
        mp={r.joint:np.array([r.x,r.y,r.z],float) for r in df.itertuples()}
        for k in vals:
            if k in mp and np.all(np.isfinite(mp[k])): vals[k].append(mp[k])
    out={}
    for k,arr in vals.items():
        if len(arr)>=2:
            A=np.asarray(arr,float); out[k]=float(np.linalg.norm(np.nanmax(A[:,:2],axis=0)-np.nanmin(A[:,:2],axis=0)))
    return out


def _lower_limb_motion_gate_146(seq, target_ranges=None):
    """Puerta temporal dura: si el target inferior se mueve, SKEL también debe moverse."""
    ok=[f for f in (seq or {}).get('frames',[]) if f.get('status')=='ok' and f.get('joints')]
    names=[str(x) for x in (seq or {}).get('joint_names',[])]
    if len(ok)<2 or not names: return {'ok':False,'reason':'secuencia insuficiente'}
    J=np.asarray([f['joints'] for f in ok],dtype=float); nm={_simple_name(n):i for i,n in enumerate(names)}
    wanted={'RKnee':'tibia_r','RAnkle':'talus_r','LKnee':'tibia_l','LAnkle':'talus_l'}
    ranges={}
    for lab,jn in wanted.items():
        i=nm.get(_simple_name(jn))
        if i is not None:
            A=J[:,i,:2]; ranges[lab]=float(np.linalg.norm(np.nanmax(A,axis=0)-np.nanmin(A,axis=0)))
    target_ranges=dict(target_ranges or {})
    threshold=0.004
    failures=[]
    for lab,val in ranges.items():
        targ=float(target_ranges.get(lab,0.0))
        # Si el objetivo prácticamente no cambia, no se exige movimiento artificial.
        if targ>0.01 and val<=threshold: failures.append(f'{lab}: target={targ:.4f}, SKEL={val:.4f}')
    return {'ok':not failures,'ranges_xy':ranges,'target_ranges_xy':target_ranges,'threshold':threshold,'failures':failures,
            'reason':'OK' if not failures else 'miembro inferior congelado pese a movimiento objetivo'}



def _lower_limb_target_profile_147(motion, acquisition_mode="monocular_depth_safe"):
    """Describe dónde se mueve realmente la cadena inferior objetivo.

    V110.3.16 resolvía únicamente XY. En registros biplanares la marcha sagital puede
    estar codificada sobre Z tras V104/V107, por lo que un solver XY puede concluir
    erróneamente que la pierna está quieta. Este perfil conserva el rango por eje y
    produce pesos de ajuste dependientes de la modalidad y del movimiento observado.
    """
    frames=list((motion or {}).get('frames') or [])
    wanted=['RHip','RKnee','RAnkle','LHip','LKnee','LAnkle']
    vals={k:[] for k in wanted}
    for fi,f in enumerate(frames):
        try: df=_target_df_for_frame(f,fi)
        except Exception: continue
        mp={r.joint:np.array([r.x,r.y,r.z],dtype=float) for r in df.itertuples()}
        for k in wanted:
            a=mp.get(k)
            if a is not None and np.all(np.isfinite(a)): vals[k].append(a)
    ranges={}; all_ranges=[]
    for k,arr in vals.items():
        if len(arr)>=2:
            A=np.asarray(arr,float)
            rr=np.nanmax(A,axis=0)-np.nanmin(A,axis=0)
            ranges[k]=rr.astype(float).tolist(); all_ranges.append(rr)
    if all_ranges:
        span=np.nanmedian(np.asarray(all_ranges,float),axis=0)
    else:
        span=np.zeros(3,float)
    mx=float(np.nanmax(span)) if np.any(np.isfinite(span)) else 0.0
    rel=(span/max(mx,1e-9)) if mx>1e-9 else np.zeros(3,float)
    mode=str(acquisition_mode or '')
    if mode=='calibrated_3d':
        weights=np.ones(3,float)
    elif mode=='biplanar_estimated':
        # Nunca ignorar el eje donde realmente se desplazan rodillas/tobillos.
        weights=np.clip(0.30+0.70*rel,0.30,1.0)
    else:
        # Monocular: XY manda; Z sólo aporta continuidad anatómica y nunca domina.
        weights=np.array([1.0,1.0,0.18],float)
        if rel[2]>0.65:
            weights[2]=0.28
    return {'ranges_xyz':ranges,'axis_span':span.astype(float).tolist(),
            'axis_weights':weights.astype(float).tolist(),'mode':mode,
            'motion_axis':int(np.argmax(span)) if mx>1e-9 else None}


def _pose_sequence_integrity_lower_update_147(torch,model,pose,rotvec,trans,scale,target_df,rows,betas,zero_trans,prev_pose,axis_map,profile,acquisition_mode):
    """Actualización temporal de piernas con persistencia verificable en q(t).

    Resuelve q3..q7 y q10..q14 contra el espacio observado ponderado. El vector que se
    devuelve es exactamente el que luego se guarda en `frames[].pose` y alimenta
    `SKEL.forward()` para skin_verts. No existe una segunda copia de pose que pueda
    sobrescribir el resultado después del commit.
    """
    names=_skel_forward_joint_names_140(model); nm={_simple_name(n):i for i,n in enumerate(names)}
    tmap={r.joint:np.array([r.x,r.y,r.z],np.float32) for r in target_df.itertuples()}
    w=np.asarray((profile or {}).get('axis_weights',[1.0,1.0,0.18]),dtype=np.float64).reshape(3)
    chains=[
        ('R',[3,4,5,6,7],[('RHip','femur_r'),('RKnee','tibia_r'),('RAnkle','talus_r')]),
        ('L',[10,11,12,13,14],[('LHip','femur_l'),('LKnee','tibia_l'),('LAnkle','talus_l')]),
    ]
    audit={'accepted':False,'strategy':'V110.3.18 weighted-XYZ DLS + pose-sequence commit','axis_weights':w.astype(float).tolist(),'chains':{}}
    any_commit=False
    for side,ids,pairs0 in chains:
        pairs=[(lm,nm.get(_simple_name(jn))) for lm,jn in pairs0 if lm in tmap and nm.get(_simple_name(jn)) is not None]
        if len(pairs)<2:
            audit['chains'][side]={'accepted':False,'reason':'landmarks insuficientes'}; continue
        base=pose.detach().clone()
        q_prev=prev_pose[0,ids].detach().cpu().numpy().astype(float)
        def eval_pose(p):
            with torch.no_grad():
                world,_,_=_direct_world_130(torch,model,p,betas,zero_trans,rotvec,scale,trans,axis_map=axis_map)
            pred=np.stack([world[ji,:].cpu().numpy() for _,ji in pairs],axis=0).astype(np.float64)
            tgt=np.stack([tmap[lm] for lm,_ in pairs],axis=0).astype(np.float64)
            d=(pred-tgt)*w[None,:]
            return pred,tgt,float(np.sqrt(np.mean(np.sum(d*d,axis=1))))
        _,_,before=eval_pose(base)
        work=base.clone(); best=base.clone(); best_rmse=before
        eps=0.02
        for it in range(10):
            pred,tgt,cur=eval_pose(work)
            r=((tgt-pred)*w[None,:]).reshape(-1)
            J=np.zeros((r.size,len(ids)),np.float64)
            for k,qi in enumerate(ids):
                pp=work.clone(); pm=work.clone()
                with torch.no_grad():
                    pp[0,qi]+=eps; pm[0,qi]-=eps
                    _clamp_direct_pose_130(torch,pp); _clamp_direct_pose_130(torch,pm)
                yp,_,_=eval_pose(pp); ym,_,_=eval_pose(pm)
                J[:,k]=(((yp-ym)/(2.0*eps))*w[None,:]).reshape(-1)
            lam=0.012+0.006*it
            A=J.T@J+lam*np.eye(len(ids)); b=J.T@r
            try: dq=np.linalg.solve(A,b)
            except Exception: dq=np.linalg.lstsq(A,b,rcond=1e-6)[0]
            dq=np.clip(dq,-0.28,0.28)
            candidates=[]
            for alpha in (1.0,0.5,0.25):
                cand=work.clone()
                with torch.no_grad():
                    for k,qi in enumerate(ids): cand[0,qi]+=float(alpha*dq[k])
                    _clamp_direct_pose_130(torch,cand)
                    # continuidad flexible: máximo 0.55 rad respecto al frame anterior.
                    dv=torch.clamp(cand[0,ids]-prev_pose[0,ids],-0.55,0.55)
                    cand[0,ids]=prev_pose[0,ids]+dv
                    _clamp_direct_pose_130(torch,cand)
                _,_,e=eval_pose(cand); candidates.append((e,cand))
            e,cand=min(candidates,key=lambda x:x[0])
            if e<best_rmse-1e-7:
                best_rmse=e; best=cand.clone(); work=cand.clone()
            else:
                break
        q_after=best[0,ids].detach().cpu().numpy().astype(float)
        q_delta=float(np.linalg.norm(q_after-q_prev))
        # Commit siempre que no empeore materialmente el ajuste y exista cambio real.
        accepted=bool(best_rmse<=before+1e-6 and q_delta>1e-8)
        if accepted:
            with torch.no_grad(): pose[0,ids].copy_(best[0,ids])
            any_commit=True
        audit['chains'][side]={'accepted':accepted,'rmse_weighted_before':float(before),'rmse_weighted_after':float(best_rmse),
                               'q_delta_from_prev':q_delta,'q_indices':ids,
                               'q_prev':q_prev.tolist(),'q_committed':pose[0,ids].detach().cpu().numpy().astype(float).tolist()}
    audit['accepted']=any_commit
    audit['committed_pose_leg_q']={_SKEL_Q_NAMES_130[i]:float(pose[0,i].detach().cpu()) for i in [3,4,5,6,7,10,11,12,13,14]}
    return pose,audit




def _lower_limb_vector_dls_196(torch,model,pose,rotvec,trans,scale,target_df,rows,betas,zero_trans,prev_pose,axis_map,profile,acquisition_mode):
    """V110.3.20.0 · propagación q(t) de miembros inferiores basada en VECTORES DE SEGMENTO.

    Corrige la causa del congelamiento residual: el solver anterior minimizaba posiciones
    absolutas de cadera/rodilla/tobillo mientras pelvis/transformación global permanecían
    congeladas. En marcha, una parte importante del cambio frame-a-frame puede ser una
    traslación corporal común; esa componente no debe impedir que los q de cadera/rodilla
    sigan la geometría relativa muslo→pierna.

    Este solver elimina la traslación común usando vectores relativos:
      - hip→knee
      - knee→ankle
      - hip→ankle
    y optimiza q3..q6 / q10..q13 con DLS ponderado XYZ. q7/q14 (tobillo) se
    conservan salvo que exista un landmark distal observable que les dé sensibilidad.
    El resultado se escribe en la MISMA `pose` persistida y usada por skin_verts(t).
    """
    names=_skel_forward_joint_names_140(model); nm={_simple_name(n):i for i,n in enumerate(names)}
    tmap={r.joint:np.array([r.x,r.y,r.z],np.float64) for r in target_df.itertuples()}
    w=np.asarray((profile or {}).get('axis_weights',[1.0,1.0,0.18]),dtype=np.float64).reshape(3)
    mode=str(acquisition_mode or '')
    chains=[
        ('R',[3,4,5,6],('RHip','RKnee','RAnkle'),('femur_r','tibia_r','talus_r')),
        ('L',[10,11,12,13],('LHip','LKnee','LAnkle'),('femur_l','tibia_l','talus_l')),
    ]
    audit={'accepted':False,'strategy':'V110.3.20.0 relative-segment weighted-XYZ DLS','axis_weights':w.astype(float).tolist(),'chains':{}}
    any_commit=False

    for side,ids,lms,jns in chains:
        if not all(k in tmap for k in lms):
            audit['chains'][side]={'accepted':False,'reason':'landmarks inferiores insuficientes'}; continue
        ji=[nm.get(_simple_name(x)) for x in jns]
        if any(x is None for x in ji):
            audit['chains'][side]={'accepted':False,'reason':'SKEL24 joint map incompleto'}; continue
        h,k,a=[tmap[x] for x in lms]
        tv=[k-h,a-k,a-h]
        # escalado robusto por longitud objetivo para que muslo/pierna aporten peso comparable.
        tl=[max(float(np.linalg.norm(v)),1e-6) for v in tv]

        def residual_and_error(p):
            with torch.no_grad():
                world,_,_=_direct_world_130(torch,model,p,betas,zero_trans,rotvec,scale,trans,axis_map=axis_map)
            W=world.detach().cpu().numpy().astype(np.float64)
            mh,mk,ma=W[ji[0]],W[ji[1]],W[ji[2]]
            mv=[mk-mh,ma-mk,ma-mh]
            rr=[]
            for m,t,L in zip(mv,tv,tl):
                # vector métrico relativo, ponderado por ejes y normalizado por longitud target.
                rr.extend((((t-m)*w)/L).tolist())
                # además dirección unitaria para impedir soluciones con orientación incorrecta.
                mn=max(float(np.linalg.norm(m)),1e-8); tn=max(float(np.linalg.norm(t)),1e-8)
                rr.extend((0.55*((t/tn)-(m/mn))*w).tolist())
            r=np.asarray(rr,np.float64)
            return r,float(np.sqrt(np.mean(r*r)))

        base=pose.detach().clone()
        r0,e0=residual_and_error(base)
        work=base.clone(); best=base.clone(); best_e=e0
        eps=0.018
        accepted_steps=0
        for it in range(12):
            r,cur=residual_and_error(work)
            J=np.zeros((r.size,len(ids)),np.float64)
            for kk,qi in enumerate(ids):
                pp=work.clone(); pm=work.clone()
                with torch.no_grad():
                    pp[0,qi]+=eps; pm[0,qi]-=eps
                    _clamp_direct_pose_130(torch,pp); _clamp_direct_pose_130(torch,pm)
                rp,_=residual_and_error(pp); rm,_=residual_and_error(pm)
                # r = target-model, por tanto dr/dq entra con este signo directamente.
                J[:,kk]=(rp-rm)/(2.0*eps)
            # Resolver J*dq ≈ -r para reducir el residual actual.
            lam=0.008+0.004*it
            A=J.T@J+lam*np.eye(len(ids)); b=-J.T@r
            try: dq=np.linalg.solve(A,b)
            except Exception: dq=np.linalg.lstsq(A,b,rcond=1e-7)[0]
            dq=np.clip(dq,-0.32,0.32)
            candidates=[]
            for alpha in (1.0,0.55,0.25):
                cand=work.clone()
                with torch.no_grad():
                    for kk,qi in enumerate(ids): cand[0,qi]+=float(alpha*dq[kk])
                    _clamp_direct_pose_130(torch,cand)
                    # continuidad temporal sin inmovilizar la cadena.
                    dv=torch.clamp(cand[0,ids]-prev_pose[0,ids],-0.60,0.60)
                    cand[0,ids]=prev_pose[0,ids]+dv
                    _clamp_direct_pose_130(torch,cand)
                _,ee=residual_and_error(cand); candidates.append((ee,cand))
            ee,cand=min(candidates,key=lambda x:x[0])
            if ee < best_e-1e-8:
                best_e=ee; best=cand.clone(); work=cand.clone(); accepted_steps+=1
            else:
                break

        q_prev=prev_pose[0,ids].detach().cpu().numpy().astype(float)
        q_best=best[0,ids].detach().cpu().numpy().astype(float)
        q_range=float(np.max(np.abs(q_best-q_prev))) if len(ids) else 0.0
        improved=float(e0-best_e)
        # El vector relativo es la referencia de commit. No se exige mejorar el error absoluto
        # global, que era precisamente la condición que congelaba la pierna.
        accepted=bool(best_e <= e0-1e-7 and q_range>1e-7)
        if accepted:
            with torch.no_grad(): pose[0,ids].copy_(best[0,ids])
            any_commit=True
        audit['chains'][side]={
            'accepted':accepted,'vector_rmse_before':float(e0),'vector_rmse_after':float(best_e),
            'improvement':improved,'accepted_steps':int(accepted_steps),'q_indices':list(ids),
            'q_delta_max_from_prev':q_range,'q_prev':q_prev.tolist(),
            'q_committed':pose[0,ids].detach().cpu().numpy().astype(float).tolist(),
            'target_segment_lengths':[float(x) for x in tl],
        }

    audit['accepted']=any_commit
    audit['committed_pose_leg_q']={_SKEL_Q_NAMES_130[i]:float(pose[0,i].detach().cpu()) for i in [3,4,5,6,7,10,11,12,13,14]}
    return pose,audit



def _lower_limb_delta_tracking_199_fast(torch,model,pose,rotvec,trans,scale,target_df,prev_target_df,betas,zero_trans,prev_pose,axis_map,profile,acquisition_mode):
    """V110.3.20.0 · solver inferior quirúrgico y rápido.

    Base funcional V110.3.19.7 intacta fuera de esta función. Sigue el delta frame-a-frame de
    los vectores muslo/pierna, pero limita el coste a dos pasos DLS por lado. El
    resultado se COMMITTEA directamente sobre `pose`, la misma pose que después
    entra en la secuencia y en skin_verts(). Nunca bloquea el pipeline: ante cualquier
    anomalía devuelve la pose previa/refinada con auditoría diagnóstica.
    """
    try:
        names=_skel_forward_joint_names_140(model); nm={_simple_name(n):i for i,n in enumerate(names)}
        cur={r.joint:np.array([r.x,r.y,r.z],np.float64) for r in target_df.itertuples()}
        prv={r.joint:np.array([r.x,r.y,r.z],np.float64) for r in prev_target_df.itertuples()}
        w=np.asarray((profile or {}).get('axis_weights',[1.0,1.0,0.18]),dtype=np.float64).reshape(3)
        chains=[('R',[3,4,5,6],('RHip','RKnee','RAnkle'),('femur_r','tibia_r','talus_r')),
                ('L',[10,11,12,13],('LHip','LKnee','LAnkle'),('femur_l','tibia_l','talus_l'))]
        audit={'accepted':False,'strategy':'V110.3.20.0 fast frame-delta DLS','axis_weights':w.tolist(),'chains':{}}
        with torch.no_grad():
            leg_ids=[3,4,5,6,7,10,11,12,13,14]
            pose[0,leg_ids].copy_(prev_pose[0,leg_ids]); _clamp_direct_pose_130(torch,pose)
        def world_np(p):
            with torch.no_grad():
                W,_,_=_direct_world_130(torch,model,p,betas,zero_trans,rotvec,scale,trans,axis_map=axis_map)
            return W.detach().cpu().numpy().astype(np.float64)
        Wprev=world_np(prev_pose)
        any_commit=False
        for side,ids,lms,jns in chains:
            try:
                if not all(k in cur and k in prv for k in lms):
                    audit['chains'][side]={'accepted':False,'reason':'landmarks inferiores insuficientes'}; continue
                ji=[nm.get(_simple_name(x)) for x in jns]
                if any(x is None for x in ji):
                    audit['chains'][side]={'accepted':False,'reason':'SKEL24 joint map incompleto'}; continue
                hp,kp,ap=[prv[x] for x in lms]; hc,kc,ac=[cur[x] for x in lms]
                dt=np.stack([(kc-hc)-(kp-hp),(ac-kc)-(ap-kp)],axis=0)
                target_motion=float(np.sqrt(np.mean((dt*w[None,:])**2)))
                mh,mk,ma=[Wprev[j] for j in ji]
                desired=np.stack([mk-mh,ma-mk],axis=0)+dt
                work=pose.detach().clone(); best=work.clone()
                def residual(p):
                    W=world_np(p); h,k,a=[W[j] for j in ji]
                    mv=np.stack([k-h,a-k],axis=0)
                    r=((desired-mv)*w[None,:]).reshape(-1)
                    return r,float(np.sqrt(np.mean(r*r)))
                r0,e0=residual(work); best_e=e0
                eps=0.02
                # Máximo DOS pasos: reduce drásticamente el tiempo frente a V110.3.19.8.
                for it in range(2):
                    r,cur_e=residual(work); J=np.zeros((r.size,len(ids)),np.float64)
                    for kk,qi in enumerate(ids):
                        pp=work.clone(); pm=work.clone()
                        with torch.no_grad():
                            pp[0,qi]+=eps; pm[0,qi]-=eps; _clamp_direct_pose_130(torch,pp); _clamp_direct_pose_130(torch,pm)
                        rp,_=residual(pp); rm,_=residual(pm); J[:,kk]=(rp-rm)/(2.0*eps)
                    try:
                        dq=np.linalg.solve(J.T@J+0.008*np.eye(len(ids)),-J.T@r)
                    except Exception:
                        dq=np.linalg.lstsq(J,-r,rcond=1e-6)[0]
                    dq=np.clip(dq,-0.18,0.18)
                    cand=work.clone()
                    with torch.no_grad():
                        for kk,qi in enumerate(ids): cand[0,qi]+=float(dq[kk])
                        dv=torch.clamp(cand[0,ids]-prev_pose[0,ids],-0.30,0.30)
                        cand[0,ids]=prev_pose[0,ids]+dv; _clamp_direct_pose_130(torch,cand)
                    _,ee=residual(cand)
                    if ee <= best_e + 1e-10:
                        best_e=ee; best=cand.clone(); work=cand
                    else:
                        break
                q_prev=prev_pose[0,ids].detach().cpu().numpy().astype(float)
                q_best=best[0,ids].detach().cpu().numpy().astype(float)
                q_delta=float(np.linalg.norm(q_best-q_prev))
                accepted=bool(q_delta>1e-8 and best_e<=e0+1e-8)
                if accepted:
                    with torch.no_grad(): pose[0,ids].copy_(best[0,ids])
                    any_commit=True
                audit['chains'][side]={'accepted':accepted,'target_delta_rms':target_motion,
                    'delta_rmse_before':float(e0),'delta_rmse_after':float(best_e),'q_delta_norm':q_delta,
                    'q_prev':q_prev.tolist(),'q_committed':pose[0,ids].detach().cpu().numpy().astype(float).tolist()}
            except Exception as exc:
                audit['chains'][side]={'accepted':False,'error':f'{type(exc).__name__}: {exc}','non_blocking':True}
        audit['accepted']=any_commit
        audit['committed_pose_leg_q']={_SKEL_Q_NAMES_130[i]:float(pose[0,i].detach().cpu()) for i in [3,4,5,6,7,10,11,12,13,14]}
        return pose,audit
    except Exception as exc:
        return pose,{'accepted':False,'strategy':'V110.3.20.0 fast frame-delta DLS','error':f'{type(exc).__name__}: {exc}','non_blocking':True}

def _pose_sequence_integrity_audit_147(seq, profile=None):
    frames=[f for f in (seq or {}).get('frames',[]) if f.get('status')=='ok' and f.get('pose') is not None]
    ids=[3,4,5,6,7,10,11,12,13,14]
    if len(frames)<2:
        return {'ok':False,'reason':'secuencia de poses insuficiente'}
    P=np.asarray([f['pose'] for f in frames],dtype=float)
    ranges={_SKEL_Q_NAMES_130[i]:float(np.nanmax(P[:,i])-np.nanmin(P[:,i])) for i in ids}
    right=max(ranges[_SKEL_Q_NAMES_130[i]] for i in [3,4,5,6,7])
    left=max(ranges[_SKEL_Q_NAMES_130[i]] for i in [10,11,12,13,14])
    span=np.asarray((profile or {}).get('axis_span',[0,0,0]),float)
    target_moves=bool(np.linalg.norm(span)>0.015)
    failures=[]
    if target_moves and right<1e-5: failures.append('q pierna derecha constante')
    if target_moves and left<1e-5: failures.append('q pierna izquierda constante')
    return {'ok':not failures,'q_ranges':ranges,'right_max_q_range':float(right),'left_max_q_range':float(left),
            'target_axis_span':span.astype(float).tolist(),'failures':failures,'reason':'OK' if not failures else '; '.join(failures)}


def _lower_limb_motion_gate_147(seq, profile=None):
    """Puerta dura usando el espacio de movimiento real, no sólo XY."""
    ok=[f for f in (seq or {}).get('frames',[]) if f.get('status')=='ok' and f.get('joints')]
    names=[str(x) for x in (seq or {}).get('joint_names',[])]
    pose_audit=(seq or {}).get('pose_sequence_integrity',{})
    if len(ok)<2 or not names:
        return {'ok':False,'reason':'secuencia insuficiente','pose_sequence_integrity':pose_audit}
    J=np.asarray([f['joints'] for f in ok],dtype=float); nm={_simple_name(n):i for i,n in enumerate(names)}
    w=np.asarray((profile or {}).get('axis_weights',[1,1,0.18]),float)
    wanted={'RKnee':'tibia_r','RAnkle':'talus_r','LKnee':'tibia_l','LAnkle':'talus_l'}
    ranges={}; target_ranges={}
    pr=(profile or {}).get('ranges_xyz',{})
    for lab,jn in wanted.items():
        i=nm.get(_simple_name(jn))
        if i is not None:
            A=J[:,i,:]; rr=np.nanmax(A,axis=0)-np.nanmin(A,axis=0)
            ranges[lab]=float(np.linalg.norm(rr*w))
        tr=np.asarray(pr.get(lab,[0,0,0]),float)
        target_ranges[lab]=float(np.linalg.norm(tr*w))
    failures=[]
    for lab,val in ranges.items():
        targ=float(target_ranges.get(lab,0.0))
        threshold=max(0.004,0.05*targ)
        if targ>0.012 and val<=threshold:
            failures.append(f'{lab}: target={targ:.4f}, SKEL={val:.4f}')
    if not pose_audit.get('ok',False):
        failures.extend(pose_audit.get('failures',[]) or ['pose sequence integrity FAIL'])
    return {'ok':not failures,'ranges_weighted':ranges,'target_ranges_weighted':target_ranges,
            'axis_weights':w.astype(float).tolist(),'failures':failures,
            'pose_sequence_integrity':pose_audit,
            'reason':'OK' if not failures else 'pose/q inferior no persiste o joint inferior congelado'}


def _load_mesh_sequence_npz_bytes_2012(raw):
    """Versión 151 · Reabre un NPZ SKEL ya calculado sin volver a generar los 75 frames."""
    try:
        z=np.load(io.BytesIO(raw),allow_pickle=False)
        req=["vertices","faces"]
        missing=[k for k in req if k not in z.files]
        if missing:
            raise ValueError("NPZ SKEL incompleto: faltan "+", ".join(missing))
        mesh={
            "version":str(z["version"][0]) if "version" in z.files and np.asarray(z["version"]).size else "imported_npz",
            "frame_ids":np.asarray(z["frame_ids"],dtype=np.int16) if "frame_ids" in z.files else np.arange(len(z["vertices"]),dtype=np.int16)+1,
            "vertices":np.asarray(z["vertices"],dtype=np.float32),
            "faces":np.asarray(z["faces"],dtype=np.int32),
            "joints":np.asarray(z["joints"],dtype=np.float32) if "joints" in z.files else None,
            "poses":np.asarray(z["poses"],dtype=np.float32) if "poses" in z.files else None,
            "joint_names":[str(x) for x in np.asarray(z["joint_names"]).tolist()] if "joint_names" in z.files else [],
            "scale":float(np.asarray(z["scale"]).ravel()[0]) if "scale" in z.files else 1.0,
            "betas":np.asarray(z["betas"],dtype=np.float32) if "betas" in z.files else np.zeros(10,dtype=np.float32),
            "source_sequence_version":str(np.asarray(z["source_sequence_version"]).ravel()[0]) if "source_sequence_version" in z.files else "imported_npz",
            "source_sequence_scale":float(np.asarray(z["source_sequence_scale"]).ravel()[0]) if "source_sequence_scale" in z.files else (float(np.asarray(z["scale"]).ravel()[0]) if "scale" in z.files else 1.0),
            "axis_map":np.asarray(z["axis_map"],dtype=np.float32) if "axis_map" in z.files else np.eye(3,dtype=np.float32),
            "skel_gender":str(np.asarray(z["skel_gender"]).ravel()[0]) if "skel_gender" in z.files else "unknown",
            "imported_from_npz":True,
        }
        # V157 · preserve optional detector-level distal scores in future NPZs.
        if "foot_landmark_scores_json" in z.files:
            try:
                import json as _json
                mesh["foot_landmark_scores"]=_json.loads(str(np.asarray(z["foot_landmark_scores_json"]).ravel()[0]))
            except Exception:
                pass
        V=mesh["vertices"]; F=mesh["faces"]
        if V.ndim!=3 or V.shape[-1]!=3 or V.shape[0]<1:
            raise ValueError(f"vertices inválidos: {V.shape}")
        if F.ndim!=2 or F.shape[1]<3 or F.size<3:
            raise ValueError(f"faces inválidas: {F.shape}")
        if mesh["joints"] is not None and len(mesh["joints"])!=len(V):
            raise ValueError("joints y vertices tienen distinto número de frames")
        if mesh["poses"] is not None and len(mesh["poses"])!=len(V):
            raise ValueError("poses y vertices tienen distinto número de frames")
        return mesh,None
    except Exception as exc:
        return None,f"{type(exc).__name__}: {exc}"


def _v2014_npz_standalone_results(mesh_seq):
    """Resultados SKEL que pueden reconstruirse EXCLUSIVAMENTE desde un NPZ ya calculado.
    No depende de haber vuelto a procesar los vídeos frontal/lateral.
    """
    if not isinstance(mesh_seq,dict):
        return None
    P=np.asarray(mesh_seq.get("poses"),dtype=float) if mesh_seq.get("poses") is not None else None
    J=np.asarray(mesh_seq.get("joints"),dtype=float) if mesh_seq.get("joints") is not None else None
    if P is None or P.ndim!=2 or P.shape[0]<3 or P.shape[1]<46:
        return None
    D=np.rad2deg(P)
    phase=np.linspace(0,100,len(D))
    curves={
        "Cadera D":D[:,3],"Cadera I":D[:,10],
        "Rodilla D":D[:,6],"Rodilla I":D[:,13],
        "Tobillo D":D[:,7],"Tobillo I":D[:,14],
        "Tronco flex/ext":D[:,18]+D[:,21],
        "Hombro D":D[:,29],"Hombro I":D[:,39],
        "Codo D":D[:,32],"Codo I":D[:,42],
    }
    vel={k:np.gradient(v) for k,v in curves.items()}
    acc={k:np.gradient(vel[k]) for k in curves}
    rows=[]
    for k,v in curves.items():
        rows.append({
            "Articulación":k,
            "ROM (°)":float(np.ptp(v)),
            "Mín (°)":float(np.min(v)),
            "Máx (°)":float(np.max(v)),
            "|vel| máx (°/frame)":float(np.max(np.abs(vel[k]))),
            "|acc| máx (°/frame²)":float(np.max(np.abs(acc[k]))),
            "Suavidad RMS acc":float(np.sqrt(np.mean(acc[k]**2))),
        })
    def _sym(a,b):
        a=np.asarray(a); b=np.asarray(b)
        den=max(1e-6,float(np.ptp(np.r_[a,b])))
        return float(max(0,100*(1-np.sqrt(np.mean((a-b)**2))/den)))
    symmetry={
        "Cadera":_sym(curves["Cadera D"],curves["Cadera I"]),
        "Rodilla":_sym(curves["Rodilla D"],curves["Rodilla I"]),
        "Tobillo":_sym(curves["Tobillo D"],curves["Tobillo I"]),
        "Hombro":_sym(curves["Hombro D"],curves["Hombro I"]),
        "Codo":_sym(curves["Codo D"],curves["Codo I"]),
    }
    def _corr(a,b):
        a=np.asarray(a); b=np.asarray(b)
        return float(np.corrcoef(a,b)[0,1]) if np.std(a)>1e-8 and np.std(b)>1e-8 else np.nan
    coordination={
        "Cadera D–Rodilla D":_corr(curves["Cadera D"],curves["Rodilla D"]),
        "Rodilla D–Tobillo D":_corr(curves["Rodilla D"],curves["Tobillo D"]),
        "Cadera I–Rodilla I":_corr(curves["Cadera I"],curves["Rodilla I"]),
        "Rodilla I–Tobillo I":_corr(curves["Rodilla I"],curves["Tobillo I"]),
        "Tronco–Cadera D":_corr(curves["Tronco flex/ext"],curves["Cadera D"]),
        "Tronco–Cadera I":_corr(curves["Tronco flex/ext"],curves["Cadera I"]),
        "Brazo D–Cadera I":_corr(curves["Hombro D"],curves["Cadera I"]),
        "Brazo I–Cadera D":_corr(curves["Hombro I"],curves["Cadera D"]),
    }
    traj=[]
    if J is not None and J.ndim==3:
        names=[str(x) for x in mesh_seq.get("joint_names",[])]
        nm={re.sub(r"[^a-z0-9]","",x.lower()):i for i,x in enumerate(names)}
        for label,key in [("Pelvis","pelvis"),("Pie D","toesr"),("Pie I","toesl"),
                          ("Mano D","handr"),("Mano I","handl"),("Tronco","thorax")]:
            i=nm.get(key)
            if i is not None:
                r=np.ptp(J[:,i,:],axis=0)
                traj.append({"Punto":label,"Rango X":float(r[0]),"Rango Y":float(r[1]),
                             "Rango Z":float(r[2]),
                             "Trayectoria 3D acumulada":float(np.linalg.norm(np.diff(J[:,i,:],axis=0),axis=1).sum())})
    return {"phase":phase,"curves":curves,"rows":rows,"symmetry":symmetry,
            "coordination":coordination,"trajectories":traj}

def _v2014_render_npz_standalone_results(mesh_seq):
    """Panel autocontenido para abrir un NPZ SKEL antes de analizar de nuevo los vídeos."""
    res=_v2014_npz_standalone_results(mesh_seq)
    if res is None:
        st.warning("El NPZ se ha abierto, pero no contiene poses/q(t) suficientes para reconstruir los resultados cinemáticos.")
        return
    st.markdown("### Resultados SKEL reconstruidos desde el NPZ")
    st.success("Modo NPZ autónomo activo · estos resultados NO requieren volver a procesar los vídeos frontal/lateral.")
    st.caption("Se reconstruyen únicamente los resultados que dependen de q(t), joints y geometría SKEL. Las métricas 2D específicas de los vídeos frontal/lateral no pueden recuperarse si no fueron guardadas dentro del registro.")
    try:
        import matplotlib.pyplot as plt
        ph=res["phase"]
        fig,ax=plt.subplots(figsize=(10,4.8))
        for k in ["Cadera D","Cadera I","Rodilla D","Rodilla I","Tobillo D","Tobillo I"]:
            ax.plot(ph,res["curves"][k],label=k)
        ax.set_xlabel("% ciclo normalizado"); ax.set_ylabel("Ángulo SKEL (°)")
        ax.set_title("Curvas articulares reconstruidas desde NPZ")
        ax.grid(alpha=.2); ax.legend(fontsize=8,ncol=3); fig.tight_layout()
        st.pyplot(fig,use_container_width=True); plt.close(fig)
    except Exception as _e:
        st.caption(f"No se pudo dibujar la gráfica: {_e}")
    st.dataframe(pd.DataFrame(res["rows"]).round(3),use_container_width=True,hide_index=True)
    c1,c2=st.columns(2)
    with c1:
        st.markdown("#### Simetría angular")
        st.dataframe(pd.DataFrame([{"Par":k,"Simetría (%)":v} for k,v in res["symmetry"].items()]).round(1),
                     use_container_width=True,hide_index=True)
    with c2:
        st.markdown("#### Coordinación intersegmentaria")
        st.dataframe(pd.DataFrame([{"Acoplamiento":k,"r":v} for k,v in res["coordination"].items()]).round(3),
                     use_container_width=True,hide_index=True)
    if res["trajectories"]:
        st.markdown("#### Trayectorias relativas 3D")
        st.dataframe(pd.DataFrame(res["trajectories"]).round(4),use_container_width=True,hide_index=True)
    try:
        from foot_qc_v158 import compute_foot_qc, qc_rows
        _fq=compute_foot_qc(mesh_seq)
        if isinstance(_fq,dict):
            st.markdown("#### QC específico de pie/tobillo · V157")
            st.dataframe(pd.DataFrame(qc_rows(_fq)).round(3),use_container_width=True,hide_index=True)
            if any(str(v.get('inversion_eversion_quality','')).startswith('Orientativo · baja') for v in _fq.get('sides',{}).values()):
                st.warning("Inversión/eversión: Orientativo · baja fiabilidad para este NPZ.")
            st.caption("El contacto 3D es auxiliar y no sustituye IC/TO 2D. La confianza geométrica Heel/Toe no es score del detector si el NPZ histórico no lo guardó.")
    except Exception:
        pass

def render_skel_poc_panel(motion, acquisition_mode="monocular_depth_safe", acquisition_label=None):
    st.markdown("### Reabrir un modelo SKEL ya calculado")
    st.caption("Versión 157 · Carga el NPZ descargado de este registro para abrir directamente sus 75 frames SKEL. No repite fitting, retargeting ni generación de los 75 frames.")
    _npz_up=st.file_uploader("📂 Cargar NPZ SKEL existente",type=["npz"],key="v110_3_20_12_import_npz",
                             help="El archivo permanece en la sesión actual. No se guarda automáticamente en Supabase.")
    if _npz_up is not None:
        _sig=(getattr(_npz_up,"name",""),getattr(_npz_up,"size",None))
        if st.session_state.get("v110_3_20_12_import_sig") != _sig:
            _mesh,_err=_load_mesh_sequence_npz_bytes_2012(_npz_up.getvalue())
            if _mesh is None:
                st.error("No se pudo abrir el NPZ SKEL: "+str(_err))
            else:
                st.session_state["v110_3_18_mesh_sequence"]=_mesh
                st.session_state["v110_3_20_12_import_sig"]=_sig
                st.session_state["v110_3_20_12_import_name"]=getattr(_npz_up,"name","modelo_skel.npz")
        _mesh=st.session_state.get("v110_3_18_mesh_sequence")
        if isinstance(_mesh,dict) and _mesh.get("imported_from_npz"):
            _n=int(np.asarray(_mesh.get("vertices")).shape[0])
            _j=np.asarray(_mesh.get("joints")) if _mesh.get("joints") is not None else None
            _p=np.asarray(_mesh.get("poses")) if _mesh.get("poses") is not None else None
            st.success(f"✅ NPZ SKEL reabierto · {_n} frames · sin recalcular los 75 frames")
            st.session_state["v110_npz_standalone_mode"]=True
            st.session_state["v110_npz_standalone_ready"]=True
            c1,c2,c3,c4=st.columns(4)
            c1.metric("Frames",_n)
            c2.metric("Vértices/frame",int(np.asarray(_mesh["vertices"]).shape[1]))
            c3.metric("Joints",int(_j.shape[1]) if _j is not None and _j.ndim==3 else 0)
            c4.metric("q(t)",int(_p.shape[1]) if _p is not None and _p.ndim==2 else 0)
            _plot_skel_mesh_walker(_mesh)
            _v2014_render_npz_standalone_results(_mesh)
            st.download_button("⬇️ Descargar copia NPZ SKEL",data=_npz_up.getvalue(),
                               file_name=getattr(_npz_up,"name","modelo_skel.npz"),
                               mime="application/octet-stream",use_container_width=True,
                               key="v110_3_20_12_redownload_npz")
            st.info("Modo NPZ autónomo: el visor y los resultados SKEL se reconstruyen aquí mismo. No necesitas haber calculado antes los resultados de los vídeos en esta sesión. Las métricas 2D frontal/lateral sólo estarán disponibles si fueron guardadas explícitamente con el registro.")
            st.divider()
            st.caption("Si sólo quieres consultar este registro, no necesitas continuar con la generación SKEL inferior.")
    frames=list((motion or {}).get("frames") or [])
    if not frames:
        if isinstance(st.session_state.get("v110_3_18_mesh_sequence"),dict):
            st.info("No hay vídeo/frames V104-V107 activos, pero el NPZ importado ya está disponible para el visor y la Cinemática Anatómica 3D.")
            return
        st.warning("No hay secuencia V104/V107 disponible. Puedes cargar arriba un NPZ SKEL existente o analizar un vídeo nuevo.")
        return

    best_i, pts, score = _select_best_frame(motion)
    # V110.3.13: conservar depth_weight/source_method del frame V104/V107.
    # Sin esto la puerta monocular se evaluaba erróneamente como 3D estricta.
    df = _target_df_for_frame(frames[best_i], best_i)
    raw_count = len(_frame_points_from_frame(frames[best_i])) if frames else 0

    st.success(f"Motor cinemático disponible: {len(frames)} frames. V110 usa el primer frame con cobertura articular suficiente como puerta de validación antes de animar SKEL.")
    st.caption("V110.3.20.0 · flujo directo mínimo: sin migración de NPZ y sin recuperación/persistencia de secuencias o mallas previas. Cada ejecución parte de los 75 frames V104/V107 activos y genera q(t) → skin_verts(t) → malla en memoria.")
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Frames V104/V107",len(frames))
    c2.metric("Landmarks objetivo",len(df))
    c3.metric("Frame seleccionado",f"{best_i+1}/{len(frames)}")
    c4.metric("SKEL","entrada XYZ")
    st.info(f"V110.3.13 · modalidad de adquisición explícita: {acquisition_label or acquisition_mode}. Esta etiqueta gobierna la puerta anatómica; no se infiere desde depth_weight ni desde la mera existencia de XYZ.")

    if len(df) == 0:
        st.error("V109.1 sigue sin encontrar landmarks compatibles dentro del payload V104/V107.")
        st.write("Claves presentes en el frame seleccionado:", list((frames[best_i].get("joints") or {}).keys())[:40])
        return
    elif len(df) < 10:
        st.warning(f"Sólo se han recuperado {len(df)} landmarks objetivo. El CSV es utilizable para diagnóstico, pero todavía no para un ajuste SKEL fiable.")
    else:
        st.info(f"Puente V104/V107 → SKEL recuperado: {len(df)} landmarks objetivo de {raw_count} puntos XYZ disponibles en el frame {best_i+1}.")

    st.download_button("⬇️ Frame objetivo V110 (CSV)",df.to_csv(index=False).encode("utf-8-sig"),
                       "V110_1_2_SKEL_target_frame.csv","text/csv",use_container_width=True)
    st.caption("Este CSV contiene las coordenadas XYZ que se usarán para el ajuste articular. Los centros derivados están identificados y no modifican ninguna métrica clínica.")

    _plot_frame(df)
    with st.expander("Ver coordenadas XYZ del frame objetivo", expanded=False):
        st.dataframe(df, use_container_width=True, hide_index=True)

    # Diagnóstico geométrico previo al fitting: no altera datos ni métricas clínicas.
    P={r.joint:np.array([r.x,r.y,r.z],dtype=float) for r in df.itertuples()}
    def dist(a,b):
        return float(np.linalg.norm(P[a]-P[b])) if a in P and b in P else float("nan")
    segs={
        "Pelvis L-R":dist("LHip","RHip"),
        "Fémur L":dist("LHip","LKnee"), "Fémur R":dist("RHip","RKnee"),
        "Tibia L":dist("LKnee","LAnkle"), "Tibia R":dist("RKnee","RAnkle"),
        "Húmero L":dist("LShoulder","LElbow"), "Húmero R":dist("RShoulder","RElbow"),
        "Antebrazo L":dist("LElbow","LWrist"), "Antebrazo R":dist("RElbow","RWrist"),
    }
    arr=df[["x","y","z"]].to_numpy(float)
    span=np.nanmax(arr,axis=0)-np.nanmin(arr,axis=0)
    st.markdown("**Control geométrico previo al fit**")
    q1,q2,q3=st.columns(3)
    q1.metric("Landmarks válidos",len(df))
    q2.metric("Extensión XYZ máx.",f"{float(np.max(span)):.3f}")
    finite=[v for v in segs.values() if np.isfinite(v) and v>0]
    q3.metric("Segmentos evaluables",len(finite))
    with st.expander("Longitudes del frame objetivo",expanded=False):
        st.dataframe(pd.DataFrame([{"segmento":k,"longitud_unidades_XYZ":v} for k,v in segs.items()]),use_container_width=True,hide_index=True)

    st.markdown("**Modelo SKEL privado · V110.3.20.0 · Lower-Limb q(t) Propagation Fix**")
    st.info("🔒 V110.3.9 queda congelada como primera versión SKEL anatómicamente válida. V110.3.13 sólo añade refinamiento periférico, auditoría XY/Z/3D y corrección de propiedad q→cadena; no reabre el mapa SKEL24 validado.")
    st.caption("V110.3.18 resuelve el modelo en este orden: CACHE HIT → B2 AUTH → S3 GetObject directo/Native → MANUAL FALLBACK. Una carga manual válida también queda en la caché `/tmp/physiosentinel_skel_b2_cache_v1_1` durante la vida de la instancia.")

    raw=None; skel_source=None
    with st.spinner("Buscando modelo SKEL privado automático…"):
        auto_raw,auto_source,auto_error=_load_private_skel_bundle_auto()
    if auto_raw is not None:
        raw=auto_raw; skel_source=auto_source
        st.success(f"✅ SKEL privado cargado automáticamente · {auto_source} · almacenamiento runtime: /tmp")
    else:
        if auto_error:
            st.info("Auto-Loader Backblaze B2 no activo: " + auto_error + " · puedes continuar con carga manual.")
        bundle=st.file_uploader("Fallback manual · ZIP oficial SKEL desde tu PC",type=["zip"],key="v110_1_skel_private_bundle",help="No se guarda en Supabase ni se incorpora a la exportación de PhysioSentinel.")
        if bundle is None:
            st.info("Entrada articular preparada: 15 landmarks. Configura Backblaze B2 en Streamlit Secrets o adjunta manualmente el ZIP SKEL para ejecutar el ajuste anatómico real.")
            return
        raw=bundle.getvalue(); skel_source='carga manual desde navegador'

    audit=_inspect_private_bundle(raw)
    if not audit.get("valid_zip"):
        st.error("El archivo aportado no es un ZIP SKEL válido."); return
    st.write({"skel_male.pkl":audit["has_male"],"skel_female.pkl":audit["has_female"]})
    if not (audit["has_male"] or audit["has_female"]):
        st.error("No encuentro skel_male.pkl ni skel_female.pkl en el ZIP. No se ejecuta ningún modelo."); return
    if skel_source == 'carga manual desde navegador':
        _pref=''
        try: _pref=str(st.secrets.get('B2_FILE','')).strip()
        except Exception: _pref=''
        _cached=_persist_manual_bundle_runtime_cache(raw,_pref)
        if _cached is not None:
            st.success(f"✅ MANUAL FALLBACK → CACHE GUARDADA · {_cached.name}. En los siguientes reruns de esta instancia no tendrás que seleccionar de nuevo el ZIP.")
    # V110.1.6: runtime SKEL CPU aislado + modelo privado extraído sólo a /tmp.
    # No se toca el venv administrado, NumPy, OpenSim ni el stack gráfico ModernGL.
    def _import_or_install_skel_cpu():
        try:
            import torch as _torch
            from skel.skel_model import SKEL as _SKEL  # type: ignore
            return _torch, _SKEL, None
        except Exception as first_exc:
            try:
                # Streamlit Cloud no permite escribir de forma fiable en el site-packages
                # del venv durante la ejecución. Instalamos SKEL en un target temporal
                # escribible, igual que el aislamiento probado de OpenCV V86.6.
                url = "git+https://github.com/MarilynKeller/SKEL.git@c32cf16581295bff19399379efe5b776d707cd95"
                target = Path(tempfile.gettempdir()) / "physiosentinel_skel_cpu_v110_1_7"
                marker = target / ".ready"
                target.mkdir(parents=True, exist_ok=True)
                if str(target) not in sys.path:
                    sys.path.insert(0, str(target))
                if not marker.exists():
                    cmd = [sys.executable, "-m", "pip", "install", "--no-deps",
                           "--disable-pip-version-check", "--no-cache-dir",
                           "--target", str(target), url]
                    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
                    if proc.returncode != 0:
                        tail = (proc.stderr or proc.stdout or "")[-4000:]
                        return None, None, f"Instalación SKEL aislada falló ({proc.returncode}):\n{tail}"
                    marker.write_text("ok", encoding="utf-8")
                importlib.invalidate_caches()
                # Evita conservar un import parcial fallido anterior al instalar.
                for name in list(sys.modules):
                    if name == "skel" or name.startswith("skel."):
                        sys.modules.pop(name, None)
                import torch as _torch
                from skel.skel_model import SKEL as _SKEL  # type: ignore
                return _torch, _SKEL, None
            except Exception as second_exc:
                return None, None, f"Import inicial: {type(first_exc).__name__}: {first_exc}\nInstalación/import CPU: {type(second_exc).__name__}: {second_exc}"

    with st.spinner("Preparando runtime SKEL CPU aislado (sin ModernGL)…"):
        torch, SKEL, runtime_error = _import_or_install_skel_cpu()
    runtime = torch is not None and SKEL is not None
    if not runtime:
        st.error("El bundle privado es válido, pero el runtime SKEL CPU no ha podido prepararse. V110.3.13 evita deliberadamente moderngl-window para mantener NumPy 2.x compatible con Pose2Sim/OpenSim.")
        st.code(runtime_error or "Error de importación no especificado")
    else:
        st.success(f"Runtime SKEL REAL detectado · PyTorch {torch.__version__} · modelo privado presente · modo CPU sin ModernGL.")

        # Extraer exclusivamente el PKL elegido a un directorio temporal escribible.
        genders=[]
        if audit.get("has_male"): genders.append("male")
        if audit.get("has_female"): genders.append("female")
        gender,patient_sex_label=_patient_gender_145()
        if gender is None:
            st.error("Selecciona Mujer u Hombre en ‘Datos del paciente / registro’. SKEL no elegirá un sexo por defecto.")
            return
        if gender not in genders:
            st.error(f"El ZIP privado no contiene skel_{gender}.pkl requerido por Sexo = {patient_sex_label}.")
            return
        model_name=f"skel_{gender}.pkl"
        st.success(f"Modelo SKEL: **{gender}** · determinado automáticamente por Sexo = **{patient_sex_label}**. No existe selector SKEL independiente.")
        model_root=Path(tempfile.gettempdir()) / "physiosentinel_skel_models_cache_v1_1"
        model_root.mkdir(parents=True,exist_ok=True)
        model_path=model_root / model_name
        try:
            if not model_path.exists() or model_path.stat().st_size < 1024:
                with zipfile.ZipFile(io.BytesIO(raw)) as z:
                    matches=[n for n in z.namelist() if n.lower().endswith(model_name)]
                    if not matches:
                        raise FileNotFoundError(model_name)
                    tmp_model=model_path.with_suffix(model_path.suffix+'.part')
                    with z.open(matches[0]) as src, open(tmp_model,"wb") as dst:
                        dst.write(src.read())
                    tmp_model.replace(model_path)
        except Exception as exc:
            st.error(f"No se pudo extraer {model_name} al runtime temporal: {type(exc).__name__}: {exc}")
            return

        # V110.3.13: auditar la estructura anatómica privada que gobierna el mapeo directo.
        private_meta={}
        try:
            import pickle
            with open(model_path,"rb") as pf:
                pdata=pickle.load(pf,encoding="latin1")
            req=["joints_name","pose_params_name","parameter_mapping","per_joint_rot","osim_kintree_table"]
            missing_meta=[k for k in req if k not in pdata]
            if missing_meta:
                raise KeyError("faltan claves anatómicas: "+", ".join(missing_meta))
            private_meta={
                "model_version":str(pdata.get("version","")),
                "joints_name_count":len(pdata.get("joints_name",[])),
                "pose_params_name_count":len(pdata.get("pose_params_name",[])),
                "parameter_mapping_shape":list(np.asarray(pdata.get("parameter_mapping")).shape),
                "per_joint_rot_shape":list(np.asarray(pdata.get("per_joint_rot")).shape),
                "osim_kintree_table_shape":list(np.asarray(pdata.get("osim_kintree_table")).shape),
            }
            st.success(f"✅ Metadatos anatómicos SKEL verificados · modelo {private_meta['model_version']} · joints={private_meta['joints_name_count']} · per_joint_rot={private_meta['per_joint_rot_shape']}")
        except Exception as meta_exc:
            st.error(f"El PKL SKEL existe, pero V110.3.13 no puede verificar su mapa anatómico interno: {type(meta_exc).__name__}: {meta_exc}")
            return

        def _build_skel_model():
            # El commit fijado y versiones posteriores han usado ambas convenciones:
            # model_path explícito o SKEL_MODEL_PATH/directorio por defecto. Probamos
            # de forma controlada sin escribir fuera de /tmp.
            errs=[]
            try:
                # En el commit c32cf165..., `model_path` debe ser el DIRECTORIO;
                # SKEL concatena internamente skel_<gender>.pkl.
                _m=SKEL(gender=gender, model_path=str(model_root)); _m._physiosentinel_gender=gender; return _m, "model_path=/tmp/.../", None
            except Exception as exc:
                errs.append(f"model_path=dir: {type(exc).__name__}: {exc}")
                return None, None, "\n".join(errs)

        with st.spinner("Instanciando SKEL y ejecutando el primer forward CPU (1 frame)…"):
            model, init_mode, init_error = _build_skel_model()
            if model is None:
                st.error("SKEL se importa correctamente, pero no ha podido abrir el modelo privado desde /tmp.")
                st.code(init_error or "Error de inicialización no especificado")
                return
            try:
                model=model.to("cpu")
                model._physiosentinel_private_meta=private_meta
                pose=torch.zeros((1,int(model.num_q_params)),dtype=torch.float32,device="cpu")
                betas=torch.zeros((1,int(model.num_betas)),dtype=torch.float32,device="cpu")
                trans=torch.zeros((1,3),dtype=torch.float32,device="cpu")
                with torch.no_grad():
                    out=model(pose,betas,trans)
                skin=getattr(out,"skin_verts",None)
                skelv=getattr(out,"skel_verts",None)
                joints=getattr(out,"joints",None)
                if joints is None:
                    joints=getattr(out,"joints_ori",None)
                st.success("✅ Primer forward SKEL REAL completado en CPU para 1 frame.")
                f1,f2,f3,f4=st.columns(4)
                f1.metric("q / pose",int(model.num_q_params))
                f2.metric("betas",int(model.num_betas))
                f3.metric("skin verts",int(skin.shape[-2]) if skin is not None else "—")
                f4.metric("skel verts",int(skelv.shape[-2]) if skelv is not None else "—")
                st.caption(f"Inicialización: {init_mode} · modelo privado: {model_name} · almacenamiento temporal: /tmp · batch=1 · CPU")
                if joints is not None:
                    st.caption(f"Salida articular SKEL: shape {tuple(joints.shape)}")
                st.info("Puerta de runtime superada. V110.3.13 usa el mapa anatómico oficial q de SKEL. Ya no ejecuta calibración empírica ni selección global de DOF: el ajuste se limita a las articulaciones anatómicas correspondientes a cada cadena.")

                st.markdown("### V110.3.13 · Forward-Kinematics Ground Truth SKEL")
                st.caption("Se abandona la selección empírica q0…q45. La versión usa la semántica oficial de los 46 parámetros SKEL (cadera, rodilla, columna, escápula, hombro y codo), registro corporal pelvis–caderas–cuello y ajuste por vectores óseos dentro de cada cadena anatómica.")
                qmeta=_direct_model_metadata_130(model)
                qrows=[]
                for g,names_g in qmeta['q_groups'].items():
                    for name in names_g: qrows.append({'cadena':g,'parámetro SKEL':name,'q':_SKEL_Q_NAMES_130.index(name)})
                with st.expander("Mapa anatómico q oficial utilizado por V110.3.13",expanded=False):
                    st.dataframe(pd.DataFrame(qrows),use_container_width=True,hide_index=True)
                profile=None

                st.markdown("### V110.3.13 · Frame semilla · Unified Frame → Official SKEL24 → Correct FK → Dependency Graph → Monotonic Hierarchical IK → Automata")
                st.caption("Primero se prueban exhaustivamente 48 convenciones de ejes/signos/handedness sobre la pose neutra y se registra pelvis–caderas–cuello. Después se ajustan únicamente los parámetros anatómicos SKEL oficiales de tronco, piernas y brazos contra posiciones y vectores óseos. Betas permanecen neutras y q0–q2 no duplican la rotación global.")
                try:
                    with st.spinner("Ajustando pose SKEL al frame XYZ (CPU, una sola vez)…"):
                        fit=_fit_one_skel_frame(torch,model,df,max_iter=120,empirical_profile=profile,acquisition_mode=acquisition_mode)
                    nfit=len(fit["rows"])
                    _coord=fit.get("coordinate_system_calibration",{}) or {}
                    st.markdown("#### V110.3.13 · Calibración automática del sistema de coordenadas")
                    _det=float(_coord.get("det",1.0)) if _coord.get("det") is not None else 1.0
                    _perm=_coord.get("perm",[0,1,2]); _signs=_coord.get("signs",[1,1,1])
                    _hand="reflexión/handedness corregido" if _det < 0 else "orientación propia"
                    c1,c2,c3,c4=st.columns(4)
                    c1.metric("Convenciones probadas","48/48")
                    c2.metric("Determinante",f"{_det:+.0f}")
                    c3.metric("Rotación residual",f"{float(_coord.get('residual_rotation_deg',float('nan'))):.1f}°" if np.isfinite(_coord.get('residual_rotation_deg',np.nan)) else "—")
                    c4.metric("RMSE XY neutro · 15 pts",f"{float(_coord.get('rmse_xy_all_correspondences',float('nan'))):.4f}" if np.isfinite(_coord.get('rmse_xy_all_correspondences',np.nan)) else "—")
                    st.success(f"✅ Convención espacial seleccionada antes de CMA-ES · permutación {_perm} · signos {_signs} · {_hand}.")
                    _ug=fit.get('unified_module_gate',{}) or {}
                    _ui=fit.get('unified_frame_invariance',{}) or {}
                    st.markdown("#### V110.3.13 · Unified Coordinate Frame · test de invariancia")
                    u1,u2,u3,u4=st.columns(4)
                    u1.metric("RMSE Coordinate · 15 pts",f"{float(_ug.get('coordinate_rmse_xy_all15',float('nan'))):.6f}" if np.isfinite(_ug.get('coordinate_rmse_xy_all15',np.nan)) else "—")
                    u2.metric("RMSE LocalDoF entrada",f"{float(_ug.get('local_dof_rmse_xy_all15',float('nan'))):.6f}" if np.isfinite(_ug.get('local_dof_rmse_xy_all15',np.nan)) else "—")
                    u3.metric("Δ RMSE entre módulos",f"{float(_ug.get('delta_rmse_xy_modules',float('nan'))):.8f}" if np.isfinite(_ug.get('delta_rmse_xy_modules',np.nan)) else "—")
                    u4.metric("Máx Δ coordenada",f"{float(_ui.get('max_coordinate_delta',float('nan'))):.2e}" if np.isfinite(_ui.get('max_coordinate_delta',np.nan)) else "—")
                    if _ug.get('ok') and _ui.get('ok'):
                        st.success("✅ T_SKEL→V104 congelada y coherente: Coordinate Calibration, Local DoF, CMA-ES, refinamiento y malla parten del mismo marco.")
                    else:
                        st.error("⛔ Unified Coordinate Frame inconsistente. Se bloquea el ajuste anatómico antes de CMA-ES.")
                    _dof=fit.get('dof_local_calibration',{}) or {}
                    _drows=_dof.get('rows',[]) or []
                    _safe=_dof.get('safe_q',[]) or []
                    _lin=fit.get('linear_dof_seed',{}) or {}
                    st.markdown("#### V110.3.13 · Official SKEL24 Joint Map · Correct FK Ground Truth · 46 DoF")
                    d1,d2,d3,d4=st.columns(4)
                    d1.metric("DoF calibrados",f"{len(_drows)}/46")
                    d2.metric("DoF locales seguros",len(_safe))
                    d3.metric("RMSE XY neutro",f"{float(_lin.get('rmse_xy_neutral',float('nan'))):.4f}" if np.isfinite(_lin.get('rmse_xy_neutral',np.nan)) else "—")
                    d4.metric("RMSE XY semilla Jacobiana",f"{float(_lin.get('rmse_xy_linear_seed',float('nan'))):.4f}" if np.isfinite(_lin.get('rmse_xy_linear_seed',np.nan)) else "—")
                    st.caption("Cada q se perturba +Δ/−Δ sobre SKEL neutro registrado. Se mide su cadena real, sensibilidad XY/Z, joints afectados y eje angular efectivo. CMA-ES parte después de esta semilla medida, no de q=0.")
                    with st.expander("Auditoría completa de los 46 DoF locales",expanded=False):
                        if _drows:
                            _ddf=pd.DataFrame(_drows)
                            if 'top_moved_joints' in _ddf.columns: _ddf['top_moved_joints']=_ddf['top_moved_joints'].apply(lambda x:', '.join(x) if isinstance(x,list) else x)
                            st.dataframe(_ddf,use_container_width=True,hide_index=True)
                        st.write({"linear_seed":_json_safe_134(_lin),"safe_q":_safe})
                    with st.expander("Auditoría de las mejores convenciones de ejes",expanded=False):
                        _tops=_coord.get("top_candidates",[])
                        if _tops: st.dataframe(pd.DataFrame(_tops),use_container_width=True,hide_index=True)
                        st.write({"axis_map":fit.get("axis_map"),"calibracion":_json_safe_134(_coord)})
                    a1,a2,a3,a4=st.columns(4)
                    a1.metric("Correspondencias",f"{nfit}")
                    a2.metric("RMSE antes",f"{fit['rmse_before']:.4f}")
                    a3.metric("RMSE XY (prioritario)",f"{fit.get('rmse_xy',fit['rmse_after']):.4f}")
                    improve=(1.0-fit['rmse_after']/max(fit['rmse_before'],1e-12))*100.0
                    a4.metric("Mejora",f"{improve:.1f} %")
                    _ba=fit.get('bone_vector_audit',{})
                    b1,b2=st.columns(2)
                    b1.metric('Error angular óseo medio', f"{_ba.get('bone_angle_mean_deg',float('nan')):.1f}°" if np.isfinite(_ba.get('bone_angle_mean_deg',np.nan)) else '—')
                    b2.metric('Error angular óseo máximo', f"{_ba.get('bone_angle_max_deg',float('nan')):.1f}°" if np.isfinite(_ba.get('bone_angle_max_deg',np.nan)) else '—')
                    selected_labels=fit.get("active_q_names",[])
                    _fk=fit.get('fk_ground_truth',{}) or {}
                    _s24=fit.get('skel24_joint_map_sanity',{}) or {}
                    if _s24:
                        if _s24.get('ok'):
                            st.success('✅ Official SKEL24 joint map validado contra forward().joints: 24/24 · lateralidad FK coherente.')
                        else:
                            st.error('⛔ Official SKEL24 joint map sanity FAIL. Se bloquea el retargeting.')
                        with st.expander('Mapa oficial SKEL24 y test de lateralidad',expanded=False):
                            st.json(_s24)
                    if _fk:
                        _frows=_fk.get('rows',[]) or []
                        _qbg=_fk.get('q_by_group',{}) or {}
                        c1,c2,c3=st.columns(3)
                        c1.metric('DoF medidos por forward',f"{len(_frows)}/46")
                        c2.metric('DoF utilizables',len(_fk.get('usable_q',[]) or []))
                        c3.metric('Dependencias q→joint',sum(int(r.get('moved_joint_count',0) or 0) for r in _frows))
                        st.caption('La dependencia de cada q se infiere de los joints SKEL que realmente se desplazan con +Δ/−Δ; no del nombre biomecánico supuesto.')
                        with st.expander('Dependency Graph · q → joints realmente afectados',expanded=False):
                            if _frows: st.dataframe(pd.DataFrame(_frows),use_container_width=True,hide_index=True)
                            st.write({'q_by_group':_qbg})
                    _hik=fit.get('hierarchical_ik',{})
                    if _hik:
                        st.markdown('#### V110.3.13 · Monotonic Hierarchical IK · commit/rollback')
                        h1,h2=st.columns(2)
                        h1.metric('RMSE XY tras IK jerárquica',f"{float(_hik.get('rmse_xy_final',float('nan'))):.4f}" if np.isfinite(_hik.get('rmse_xy_final',np.nan)) else '—')
                        h2.metric('Etapas resueltas',len(_hik.get('history',[])))
                        if _hik.get('history'): st.dataframe(pd.DataFrame(_hik['history']),use_container_width=True,hide_index=True)
                    _pref=fit.get('peripheral_refinement',{}) or {}
                    if _pref:
                        st.markdown('#### V110.3.13 · Peripheral Landmark Anatomical Refinement · commit/rollback')
                        if _pref.get('history'): st.dataframe(pd.DataFrame(_pref['history']),use_container_width=True,hide_index=True)
                    _lea=fit.get('landmark_error_audit',[]) or []
                    if _lea:
                        st.markdown('#### V110.3.13 · Error por landmark · XY / Z / 3D')
                        _edf=pd.DataFrame(_lea)
                        st.dataframe(_edf,use_container_width=True,hide_index=True)
                        _worst=_lea[0]
                        st.caption(f"Peor landmark 3D: {_worst.get('landmark')} · error XY {_worst.get('error_xy',float('nan')):.4f} · |Z| {_worst.get('error_z_abs',float('nan')):.4f} · 3D {_worst.get('error_3d',float('nan')):.4f}")
                    _auto=fit.get('automata',{})
                    st.caption(f"SKELAnatomicalAutomata: CMA-ES {int(_auto.get('generations_run',0))} generaciones × población {int(_auto.get('population',0))} · Top-K={len(_auto.get('top_k',[]))} · coste final {float(_auto.get('score',float('nan'))):.4f}")
                    st.info(f"Parámetros anatómicos calibrados V110.3.13: {len(selected_labels)} q con semántica SKEL oficial: " + ", ".join(selected_labels))
                    cmap=fit.get("chain_mapping",[])
                    if cmap:
                        st.dataframe(pd.DataFrame(cmap),use_container_width=True,hide_index=True)
                    _fgate=fit.get('frame1_gate',{}) or {}
                    if _fgate.get('ok'):
                        if _fgate.get('mode')=='monocular_depth_safe':
                            st.success(f"✅ Frame 1 VALIDADO · Depth-Safe monocular · RMSE XY {float(_fgate.get('rmse_xy',np.nan)):.4f} < {float(_fgate.get('xy_limit',0.38)):.2f}. La Z inferida ({float(_fgate.get('rmse_z',np.nan)):.4f}) es informativa y no bloquea.")
                        elif _fgate.get('mode')=='biplanar_estimated':
                            st.success(f"✅ Frame 1 VALIDADO · biplanar 3D estimado · RMSE XY {float(_fgate.get('rmse_xy',np.nan)):.4f}. La Z estimada/no métrica ({float(_fgate.get('rmse_z',np.nan)):.4f}) es informativa.")
                        else:
                            st.success("✅ Frame 1 VALIDADO · puerta 3D estricta calibrada superada (XY/Z + auditoría anatómica).")
                    else:
                        _why='; '.join(_fgate.get('reasons',[]) or []) or 'criterios anatómicos no superados'
                        st.warning(f"⛔ Frame 1 NO VALIDADO · {_why}")
                    _plot_skel_fit(df,fit["rows"],fit["before"],fit["after"],fit.get("all_after"),fit.get("joint_names"))
                    erows=[]
                    for i,(target_name,jidx,skel_name) in enumerate(fit["rows"]):
                        erows.append({"XYZ objetivo":target_name,"SKEL joint":skel_name,"índice SKEL":jidx,"error antes":float(fit['errors_before'][i]),"error después":float(fit['errors_after'][i])})
                    with st.expander("Auditoría de correspondencias y error articular",expanded=False):
                        st.dataframe(pd.DataFrame(erows),use_container_width=True,hide_index=True)
                        st.write({"escala_global":fit["scale"],"traslacion_global":fit["trans"].tolist(),"rotacion_axis_angle":fit["rotvec"].tolist()})
                    export={
                        "version":"110.3.20.0","frame":int(best_i)+1,"gender":gender,"rmse_before":fit["rmse_before"],"rmse_after":fit["rmse_after"],
                        "scale":fit["scale"],"translation":fit["trans"].tolist(),"global_rotation_axis_angle":fit["rotvec"].tolist(),
                        "pose_46":fit["pose"].tolist(),"rmse_xy":fit.get("rmse_xy"),"rmse_z":fit.get("rmse_z"),"automata":fit.get("automata",{}),"retargeting_mode":fit.get("retargeting_mode"),"bone_vector_audit":fit.get("bone_vector_audit",{}),"coordinate_frame_seed_rotvec":fit.get("coordinate_frame_seed_rotvec",[]),"axis_map":fit.get("axis_map",[]),"coordinate_system_calibration":fit.get("coordinate_system_calibration",{}),"unified_coordinate_transform":fit.get("unified_coordinate_transform",{}),"unified_frame_invariance":fit.get("unified_frame_invariance",{}),"unified_module_gate":fit.get("unified_module_gate",{}),"dof_local_calibration":fit.get("dof_local_calibration",{}),"fk_ground_truth":fit.get("fk_ground_truth",{}),"skel24_joint_map_sanity":fit.get("skel24_joint_map_sanity",{}),"linear_dof_seed":fit.get("linear_dof_seed",{}),"hierarchical_ik":fit.get("hierarchical_ik",{}),"peripheral_refinement":fit.get("peripheral_refinement",{}),"landmark_error_audit":fit.get("landmark_error_audit",[]),"frame1_gate":fit.get("frame1_gate",{}),"acquisition_mode":acquisition_mode,"acquisition_label":acquisition_label,"depth_weight":fit.get("depth_weight"),"selected_q_indices":fit.get("selected_q",[]),"selected_q_labels":fit.get("active_q_names",[]),"chain_dof_map":fit.get("chain_dof_map",{}),"chain_mapping":fit.get("chain_mapping",[]),
                        "correspondences":[{"target":r[0],"skel_index":int(r[1]),"skel_joint":r[2]} for r in fit["rows"]]
                    }
                    st.download_button("⬇️ Descargar ajuste SKEL frame 1 (JSON)",json.dumps(_json_safe_134(export),ensure_ascii=False,indent=2).encode("utf-8"),"V110_3_13_SKEL_fit_frame1.json","application/json",use_container_width=True)

                    st.markdown("### 🧪 V110.3.20.6 · SKEL Lower Limb Motor Test")
                    st.caption("Prueba aislada del motor SKEL: NO usa la marcha V104/V107 ni el solver temporal. Impone una flexo-extensión sintética directamente en q3/q6/q7 y q10/q13/q14, ejecuta el forward real, genera skin_verts y reproduce la malla. Sirve para separar motor SKEL de retargeting.")
                    if st.button("🧪 Ejecutar prueba motora aislada de piernas",use_container_width=True,key="v110_3_19_11_motor_test"):
                        try:
                            _mseq,_mmesh,_maudit=_skel_lower_limb_motor_test_111(torch,model,fit,gender=gender,n_frames=32)
                            st.session_state['v110_3_19_11_motor_seq']=_mseq
                            st.session_state['v110_3_19_11_motor_mesh']=_mmesh
                            st.session_state['v110_3_19_11_motor_audit']=_maudit
                        except Exception as _mt_exc:
                            st.session_state.pop('v110_3_19_11_motor_seq',None)
                            st.session_state.pop('v110_3_19_11_motor_mesh',None)
                            st.session_state.pop('v110_3_19_11_motor_audit',None)
                            st.error("La prueba motora aislada SKEL ha fallado.")
                            st.code(f"{type(_mt_exc).__name__}: {_mt_exc}")
                    _mmesh=st.session_state.get('v110_3_19_11_motor_mesh')
                    _maudit=st.session_state.get('v110_3_19_11_motor_audit') or {}
                    if isinstance(_mmesh,dict) and isinstance(_mmesh.get('vertices'),np.ndarray):
                        if _maudit.get('ok'):
                            st.success("✅ MOTOR SKEL DE MIEMBROS INFERIORES: PASS. Las piernas se mueven al modificar directamente sus q; cualquier congelación de la marcha real está antes de skin_verts, en V104/V107→q(t).")
                        else:
                            st.error("⛔ MOTOR SKEL DE MIEMBROS INFERIORES: FAIL. No continuar con el retargeting hasta revisar q/forward.")
                        _qa=_maudit.get('q_ranges_rad',{}) or {}; _ja=_maudit.get('joint_ranges_world',{}) or {}
                        c1,c2,c3,c4=st.columns(4)
                        c1.metric('Rango q3 cadera D',f"{float(_qa.get('hip_flexion_r',0)):.3f} rad")
                        c2.metric('Rango q6 rodilla D',f"{float(_qa.get('knee_angle_r',0)):.3f} rad")
                        c3.metric('Rango q10 cadera I',f"{float(_qa.get('hip_flexion_l',0)):.3f} rad")
                        c4.metric('Rango q13 rodilla I',f"{float(_qa.get('knee_angle_l',0)):.3f} rad")
                        st.caption('Desplazamiento de joints en el test: '+ ' · '.join(f"{k}={v:.4f}" for k,v in _ja.items()))
                        _plot_skel_mesh_walker(_mmesh)
                        st.download_button("⬇️ Descargar Motor Test SKEL (.NPZ)",_mesh_sequence_npz_bytes(_mmesh),"V110_3_20_3_SKEL_lower_limb_motor_test.npz","application/octet-stream",use_container_width=True)

                    st.markdown("### V110.3.20.8 SEGMENT · arquitectura segmentaria estabilizada suavemente → SKEL · Piel + Esqueleto")
                    st.caption("Driver: 3D estimado directo/no métrico. La escala y betas quedan congeladas; la marcha se transfiere por direcciones segmentarias y ángulos relativos. Con L/RBigToe disponibles, q7/q14 del tobillo también reciben un driver observable. Las coordenadas normalizadas no se interpretan como cm/mm.")
                    _fgate=fit.get("frame1_gate",{}) or {}
                    frame_gate_ok=bool(_fgate.get("ok",False))
                    g1,g2,g3,g4=st.columns(4)
                    g1.metric("Modo de validación",str(_fgate.get("mode_label") or acquisition_label or acquisition_mode))
                    g2.metric("RMSE XY",f"{float(_fgate.get('rmse_xy',np.nan)):.4f}" if np.isfinite(_fgate.get('rmse_xy',np.nan)) else "—")
                    g3.metric("RMSE Z",f"{float(_fgate.get('rmse_z',np.nan)):.4f}" if np.isfinite(_fgate.get('rmse_z',np.nan)) else "—")
                    g4.metric("Puerta Frame 1","PASS ✅" if frame_gate_ok else "FAIL ⛔")
                    if frame_gate_ok:
                        if _fgate.get('mode')=='monocular_depth_safe':
                            st.success("✅ Puerta anatómica Depth-Safe superada. En frontal/posterior monocular se valida XY + límites articulares + SKEL24/Unified Frame; la Z inferida no causa FAIL por sí sola.")
                        elif _fgate.get('mode')=='biplanar_estimated':
                            st.success("✅ Puerta biplanar estimada superada. Se valida XY + coherencia anatómica; la Z estimada/no métrica se informa pero no bloquea por sí sola.")
                        else:
                            st.success("✅ Puerta anatómica 3D estricta calibrada superada. Se valida XY, Z y auditoría ósea.")
                    else:
                        st.warning("⛔ Puerta anatómica frame 1 NO superada: " + ('; '.join(_fgate.get('reasons',[]) or []) or 'criterios no superados'))
                    _already_75f = isinstance(st.session_state.get("v110_3_18_sequence"),dict)
                    if _already_75f:
                        st.success("✅ Los 75 frames ya están calculados en esta sesión. No se repetirán mientras la sesión siga activa.")
                    if (not _already_75f) and st.button("🧍▶️ Procesar 75 frames + generar malla anatómica SKEL",type="primary",use_container_width=True,key="v110_3_18_run_sequence",disabled=not frame_gate_ok):
                        bar=st.progress(0,text="1/2 · Preparando propagación temporal SKEL…")
                        status=st.empty()
                        def _pcb(done,total,rmse):
                            frac=0.68*min(1.0,max(0.0,float(done)/max(1,total)))
                            txt=f"1/2 · Pose SKEL frame {done}/{total}" + (f" · RMSE {rmse:.4f}" if rmse is not None else "")
                            bar.progress(frac,text=txt); status.caption(txt)
                        seq=None
                        try:
                            # Flujo ESTRICTO V110.3.13: nada de persistencia, manifiestos ni reruns
                            # entre el fitting 75F y la construcción de skin_verts.
                            st.session_state.pop("v110_3_18_mesh_sequence",None)
                            st.session_state.pop("v110_3_18_mesh_audit",None)
                            seq=_fit_skel_sequence(torch,model,motion,fit,start_index=int(best_i),iterations=24,progress_cb=_pcb,empirical_profile=profile,acquisition_mode=acquisition_mode)
                            seq['skel_gender']=gender
                            st.session_state["v110_3_18_sequence"]=seq

                            oks_auto=[x for x in seq.get("frames",[]) if x.get("status")=="ok"]
                            expected=max(1,len(frames)-int(best_i))
                            _ll_gate=seq.get('lower_limb_motion_gate',{}) or seq.get('lower_limb_motion_audit',{}) or {}
                            if len(oks_auto)>=max(2,int(0.9*expected)):
                                if _ll_gate and not bool(_ll_gate.get('ok',True)):
                                    st.warning("⚠️ Auditoría temporal de miembros inferiores: REVISAR. No bloquea skin_verts(t).")
                                def _auto_mesh_cb(done,total,nverts):
                                    frac=0.68+0.32*min(1.0,max(0.0,float(done)/max(1,total)))
                                    txt=f"2/2 · skin_verts frame {done}/{total} · {nverts} vértices"
                                    bar.progress(frac,text=txt); status.caption(txt)
                                try:
                                    # Esta llamada sucede INMEDIATAMENTE tras 75F, igual que V110.3.13.
                                    mesh_seq=_build_skel_skin_sequence(torch,model,seq,progress_cb=_auto_mesh_cb)
                                    if abs(float(mesh_seq.get("scale",0))-float(seq.get("scale",0)))>1e-7:
                                        raise RuntimeError("La escala de la malla no coincide con la secuencia temporal activa.")
                                    mesh_audit=_audit_skel_animated_mesh(mesh_seq)
                                    st.session_state["v110_3_18_mesh_sequence"]=mesh_seq
                                    st.session_state["v110_3_18_mesh_audit"]=mesh_audit
                                    bar.progress(1.0,text="Marcha anatómica SKEL completada")
                                    st.success("✅ 75/75 → skin_verts → malla SKEL completados en la misma ejecución.")

                                    # V110.3.20.6: fin del camino crítico. No se persiste ni recarga nada aquí.
                                except Exception as mesh_auto_exc:
                                    # Igual que V110.3.13: conservar q(t) aunque skin_verts falle.
                                    st.session_state.pop("v110_3_18_mesh_sequence",None)
                                    st.session_state.pop("v110_3_18_mesh_audit",None)
                                    bar.progress(1.0,text="Poses completadas; malla automática pendiente")
                                    st.warning("La secuencia q(t) se ha conservado, pero la generación automática de skin_verts falló. Puede regenerarse sin repetir el fitting.")
                                    st.code(f"{type(mesh_auto_exc).__name__}: {mesh_auto_exc}")
                            else:
                                bar.progress(1.0,text="Secuencia parcial: malla automática omitida")
                                st.warning("No hay suficientes poses válidas (mínimo 90 %) para construir una malla temporal completa.")
                            status.empty()
                        except Exception as seq_exc:
                            # No borrar un resultado ya obtenido. Si seq existe, se conserva para
                            # diagnóstico/recuperación; sólo se informa del fallo.
                            if isinstance(seq,dict) and seq.get("frames"):
                                st.session_state["v110_3_18_sequence"]=seq
                                st.warning("Los frames calculados se han conservado pese a un error posterior.")
                            st.error("No se pudo completar la propagación temporal SKEL.")
                            st.code(f"{type(seq_exc).__name__}: {seq_exc}")
                    seq=st.session_state.get("v110_3_18_sequence")
                    if isinstance(seq,dict) and seq.get("frames"):
                        # V110.3.20.0 · sólo memoria de sesión. Si q(t) existe y la malla no,
                        # se reconstruye directamente desde ESA MISMA secuencia, sin disco/NPZ/cache.
                        _mesh_now=st.session_state.get("v110_3_18_mesh_sequence")
                        if not (isinstance(_mesh_now,dict) and isinstance(_mesh_now.get("vertices"),np.ndarray)):
                            _ll_auto=seq.get('lower_limb_motion_gate',{}) or seq.get('lower_limb_motion_audit',{}) or {}
                            _ok_auto=[x for x in seq.get('frames',[]) if x.get('status')=='ok']
                            if len(_ok_auto)>=max(2,int(0.9*max(1,len(frames)-int(best_i)))):
                                if _ll_auto and not bool(_ll_auto.get('ok',True)):
                                    st.warning("⚠️ Auditoría de miembros inferiores: REVISAR. No bloquea la construcción de malla.")
                                st.info("Secuencia q(t) presente en memoria. Generando skin_verts y malla SKEL directamente, sin persistencia ni NPZ intermedio.")
                                _auto_bar=st.progress(0,text="Generando malla desde q(t) en memoria…")
                                def _recovery_mesh_cb(done,total,nverts):
                                    _auto_bar.progress(min(1.0,float(done)/max(1,total)),text=f"Malla {done}/{total} · {nverts} vértices")
                                try:
                                    _mesh_now=_build_skel_skin_sequence(torch,model,seq,progress_cb=_recovery_mesh_cb)
                                    _mesh_audit_now=_audit_skel_animated_mesh(_mesh_now)
                                    st.session_state["v110_3_18_mesh_sequence"]=_mesh_now
                                    st.session_state["v110_3_18_mesh_audit"]=_mesh_audit_now
                                    _auto_bar.progress(1.0,text="Malla SKEL lista")
                                    st.success("✅ q(t) → skin_verts → malla completado íntegramente en memoria.")
                                except Exception as _rec_exc:
                                    st.error("La secuencia q(t) está en memoria, pero la construcción de malla falló.")
                                    st.code(f"{type(_rec_exc).__name__}: {_rec_exc}")
                        oks=[x for x in seq["frames"] if x.get("status")=="ok"]
                        rmses=np.array([x.get("rmse",np.nan) for x in oks],float) if oks else np.array([],float)
                        m1,m2,m3,m4=st.columns(4)
                        m1.metric("Frames resueltos",f"{len(oks)}/{len(frames)-int(best_i)}")
                        m2.metric("RMSE medio",f"{np.nanmean(rmses):.4f}" if rmses.size else "—")
                        m3.metric("RMSE máximo",f"{np.nanmax(rmses):.4f}" if rmses.size else "—")
                        m4.metric("Escala fija",f"{seq.get('scale',float('nan')):.4f}")
                        _ll=seq.get('lower_limb_motion_audit',{}) or {}
                        if _ll.get('ok'):
                            st.success("✅ Movimiento temporal de miembros inferiores detectado: " + " · ".join(f"{k} {v:.3f}" for k,v in (_ll.get('ranges_xy') or _ll.get('ranges') or {}).items()))
                        else:
                            st.error("⛔ Auditoría temporal de piernas: cadena inferior congelada o insuficiente. No considerar la marcha válida hasta corregirla. " + str(_ll.get('ranges_xy') or _ll.get('ranges') or _ll.get('failures') or _ll.get('reason','')))
                        _vgates=[x.get("depth_safe_validation",{}) for x in oks]
                        unsafe=[g for g in _vgates if not g.get("ok",True)]
                        if unsafe:
                            st.warning(f"Control anatómico coherente con la modalidad: {len(unsafe)} frame(s) no superan la puerta correspondiente. Revísalos antes de considerar la secuencia plenamente validada.")
                        else:
                            _mode=next((g.get('mode_label') for g in _vgates if g.get('mode_label')), acquisition_label or acquisition_mode)
                            st.success(f"✅ Control anatómico {_mode} superado en todos los frames resueltos.")
                        if len(oks)>=max(2,int(0.9*(len(frames)-int(best_i)))):
                            st.success("✅ Propagación temporal completada. SKEL dispone ya de una pose continua para la secuencia V104/V107 y puede reproducirse como marcha articulada.")
                        else:
                            st.warning("La secuencia se ha generado parcialmente. Revisa los frames con landmarks insuficientes antes de considerarla validada.")
                        st.markdown('### Referencia cinemática · vídeo 3D simplificado no calibrado')
                        st.caption('Referencia V104/V107 recuperada para comparar la cinemática objetivo con SKEL. No representa 3D métrico calibrado.')
                        _plot_simplified_xyz_animation(motion)
                        _plot_skel_sequence_animation(seq)
                        export_seq={k:v for k,v in seq.items()}
                        st.download_button("⬇️ Descargar secuencia SKEL V110.3.13 (JSON)",json.dumps(_json_safe_134(export_seq),ensure_ascii=False).encode("utf-8"),"V110_3_19_11_SKEL_sequence.json","application/json",use_container_width=True)
                        with st.expander("Auditoría temporal por frame",expanded=False):
                            audit_rows=[{"frame":x.get("frame"),"estado":x.get("status"),"correspondencias":x.get("n_correspondences"),"RMSE 3D":x.get("rmse"),"RMSE XY":x.get("rmse_xy"),"RMSE Z":x.get("rmse_z"),"puerta":"PASS" if x.get("depth_safe_validation",{}).get("ok",False) else "REVISAR","modo":x.get("depth_safe_validation",{}).get("mode","—"),"alertas":"; ".join(x.get("depth_safe_validation",{}).get("reasons",[]) or [])} for x in seq["frames"]]
                            st.dataframe(pd.DataFrame(audit_rows),use_container_width=True,hide_index=True)

                        st.markdown("### V110.3.20.0 · SKEL ANIMATED ANATOMICAL MESH")
                        st.caption("Cada pose q(t) ya validada se evalúa una sola vez con el forward real de SKEL para obtener `skin_verts(t)`. La malla NO se reoptimiza: sólo representa la marcha temporal aceptada. Topología y betas permanecen constantes.")
                        if st.button("🔁 Regenerar malla anatómica SKEL",use_container_width=True,key="v110_3_13_build_skin_mesh"):
                            mb=st.progress(0,text="Preparando malla corporal SKEL…")
                            ms=st.empty()
                            def _mpcb(done,total,nverts):
                                frac=min(1.0,max(0.0,float(done)/max(1,total)))
                                txt=f"Malla frame {done}/{total} · {nverts} vértices"
                                mb.progress(frac,text=txt); ms.caption(txt)
                            try:
                                unsafe_frames=[x for x in seq.get("frames",[]) if x.get("status")=="ok" and not x.get("anatomical_audit",{}).get("ok",True)]
                                if unsafe_frames:
                                    st.warning(f"Se generará la malla con {len(unsafe_frames)} alerta(s) anatómica(s) para inspección; la versión las conserva en la auditoría.")
                                mesh_seq=_build_skel_skin_sequence(torch,model,seq,progress_cb=_mpcb)
                                if abs(float(mesh_seq.get("scale",0))-float(seq.get("scale",0)))>1e-7:
                                    raise RuntimeError("La escala de la malla no coincide con la secuencia temporal activa.")
                                st.session_state["v110_3_18_mesh_sequence"]=mesh_seq
                                st.session_state["v110_3_18_mesh_audit"]=_audit_skel_animated_mesh(mesh_seq)
                                mb.progress(1.0,text="Malla SKEL 75 frames completada")
                                ms.empty()
                            except Exception as mesh_exc:
                                st.session_state.pop("v110_3_18_mesh_sequence",None)
                                st.session_state.pop("v110_3_18_mesh_audit",None)
                                st.error("La marcha articular está validada, pero no se pudo generar la malla corporal SKEL.")
                                st.code(f"{type(mesh_exc).__name__}: {mesh_exc}")
                        mesh_seq=st.session_state.get("v110_3_18_mesh_sequence")
                        if isinstance(mesh_seq,dict) and isinstance(mesh_seq.get("vertices"),np.ndarray):
                            VV=mesh_seq["vertices"]; FF=mesh_seq["faces"]
                            z1,z2,z3,z4=st.columns(4)
                            z1.metric("Frames malla",int(VV.shape[0]))
                            z2.metric("Vértices/frame",int(VV.shape[1]))
                            z3.metric("Triángulos",int(FF.shape[0]))
                            z4.metric("Topología",str(mesh_seq.get("face_source","skin_f")))
                            if VV.shape[0]==len(oks):
                                st.success("✅ SKEL ANIMATED ANATOMICAL MESH generado. Los 6890 vértices de la misma identidad corporal siguen la pose temporal validada frame a frame.")
                            _ma=st.session_state.get("v110_3_18_mesh_audit") or _audit_skel_animated_mesh(mesh_seq)
                            a1,a2,a3,a4=st.columns(4)
                            a1.metric("Integridad malla","PASS ✅" if _ma.get("ok") else "REVISAR ⚠️")
                            a2.metric("Frames",int(_ma.get("frames",0)))
                            a3.metric("Vértices/frame",int(_ma.get("vertices_per_frame",0)))
                            a4.metric("Triángulos",int(_ma.get("triangles",0)))
                            if not _ma.get("ok"):
                                st.warning("Auditoría de malla: " + "; ".join(_ma.get("reasons",[]) or ["revisión necesaria"]))
                            _plot_skel_mesh_walker(mesh_seq)
                            _npz_bytes=_mesh_sequence_npz_bytes(mesh_seq)
                            st.download_button("⬇️ DESCARGAR MARCHA SKEL 75F (.NPZ)",_npz_bytes,"V110_3_19_11_SKEL_mesh_sequence.npz","application/octet-stream",use_container_width=True,type="primary")
                            st.caption("NPZ científico generado al vuelo para descarga: no se guarda ni se recarga automáticamente dentro de la app.")
                            st.markdown('### V110.3.20.0 · Validación sincronizada Side-by-Side')
                            st.caption('El mismo frame se representa como XYZ V104/V107, joints SKEL y malla. La validación prioriza coherencia anatómica de tronco y extremidades además del RMSE.')
                            _side_by_side_validation(motion,seq,mesh_seq)
                        else:
                            st.info("La secuencia articular está disponible pero la malla todavía no. Usa ‘Regenerar malla anatómica SKEL’ para reconstruir `skin_verts` sin repetir el fitting.")
                    else:
                        if frame_gate_ok:
                            st.info("✅ El frame semilla está validado. Pulsa el botón anterior para resolver frames 2→75; después V110.3.20.0 generará automáticamente skin_verts(t) y la malla corporal animada en la misma ejecución.")
                        else:
                            st.info("El frame semilla no está validado; la propagación permanece bloqueada hasta superar la única puerta de validación mostrada arriba.")
                except Exception as fit_exc:
                    st.error("SKEL funciona, pero V110.3.13 no ha podido completar el ajuste del frame semilla.")
                    st.code(f"{type(fit_exc).__name__}: {fit_exc}")
            except Exception as exc:
                st.error("El modelo SKEL se ha instanciado, pero el primer forward CPU ha fallado.")
                st.code(f"{type(exc).__name__}: {exc}")
                return
