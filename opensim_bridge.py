from __future__ import annotations
from pathlib import Path
import json, re, xml.etree.ElementTree as ET
import numpy as np
import requests

RAJAGOPAL_MODEL_URL = "https://raw.githubusercontent.com/opensim-org/opensim-models/master/Models/Rajagopal/Rajagopal2016.osim"
RAJAGOPAL_GEOM_BASES = [
    "https://raw.githubusercontent.com/opensim-org/opensim-models/master/Models/Rajagopal/Geometry",
    "https://raw.githubusercontent.com/opensim-org/opensim-models/master/Geometry",
    "https://raw.githubusercontent.com/opensim-org/opensim-models/master/Models/Rajagopal_OpenSense/Geometry",
]
CACHE_ROOT = Path("/tmp/physiosentinel_opensim_rajagopal_v3211")

# Lower-body geometries are the first target because they overlap strongly
# with the current SKEL gait use case.
BODY_TO_SKEL = {
    "pelvis": ("pelvis","femur_r"),   # special global/pelvic body
    "femur_r": ("femur_r","tibia_r"),
    "femur_l": ("femur_l","tibia_l"),
    "tibia_r": ("tibia_r","talus_r"),
    "tibia_l": ("tibia_l","talus_l"),
    "talus_r": ("talus_r","calcn_r"),
    "talus_l": ("talus_l","calcn_l"),
    "calcn_r": ("calcn_r","toes_r"),
    "calcn_l": ("calcn_l","toes_l"),
    "toes_r": ("calcn_r","toes_r"),
    "toes_l": ("calcn_l","toes_l"),
}

def _get(url, dst, timeout=45):
    dst=Path(dst); dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists() and dst.stat().st_size>32:
        return "cached"
    r=requests.get(url,timeout=timeout)
    r.raise_for_status()
    dst.write_bytes(r.content)
    return "downloaded"


def _get_geometry_with_fallback(filename, dst, force=False):
    dst=Path(dst)
    if force and dst.exists():
        dst.unlink()
    if dst.exists() and dst.stat().st_size>32:
        return "cached", None
    errs=[]
    for base in RAJAGOPAL_GEOM_BASES:
        try:
            st=_get(f"{base}/{filename}",dst)
            return st, base
        except Exception as e:
            errs.append(f"{base}: {type(e).__name__}: {e}")
            if dst.exists():
                try: dst.unlink()
                except Exception: pass
    raise RuntimeError(" | ".join(errs))

def _strip_ns(root):
    for el in root.iter():
        if "}" in el.tag:
            el.tag=el.tag.split("}",1)[1]
    return root

def _parse_model(model_path):
    root=_strip_ns(ET.parse(model_path).getroot())
    out={"model_name":"","credits":"","publications":"","bodies":{},"mesh_files":[]}
    model=root.find(".//Model")
    if model is not None:
        out["model_name"]=model.attrib.get("name","Rajagopal2016")
        c=model.find("credits"); p=model.find("publications")
        out["credits"]=(c.text or "").strip() if c is not None else ""
        out["publications"]=(p.text or "").strip() if p is not None else ""
    for body in root.findall(".//BodySet/objects/Body"):
        bn=body.attrib.get("name","")
        meshes=[]
        for mesh in body.findall("./attached_geometry/Mesh"):
            mf=mesh.find("mesh_file")
            sf=mesh.find("scale_factors")
            if mf is None or not (mf.text or "").strip():
                continue
            fn=(mf.text or "").strip()
            scale=[1.,1.,1.]
            if sf is not None and (sf.text or "").strip():
                try: scale=[float(x) for x in sf.text.split()[:3]]
                except Exception: pass
            meshes.append({"file":fn,"scale":scale})
            out["mesh_files"].append(fn)
        out["bodies"][bn]={"meshes":meshes}
    out["mesh_files"]=sorted(set(out["mesh_files"]))
    return out

def prepare_rajagopal(force=False):
    root=CACHE_ROOT
    root.mkdir(parents=True,exist_ok=True)
    model_path=root/"Rajagopal2016.osim"
    if force and model_path.exists(): model_path.unlink()
    states=[]
    try:
        states.append(_get(RAJAGOPAL_MODEL_URL,model_path))
        meta=_parse_model(model_path)
    except Exception as e:
        return {"ok":False,"root":str(root),"error":f"{type(e).__name__}: {e}"}

    # Download only geometry actually referenced by the model.
    downloaded=cached=failed=0; errors=[]
    gdir=root/"Geometry"; gdir.mkdir(exist_ok=True)
    resolved_bases={}
    for fn in meta["mesh_files"]:
        dst=gdir/fn
        try:
            st,base_used=_get_geometry_with_fallback(fn,dst,force=force)
            downloaded += st=="downloaded"; cached += st=="cached"
            if base_used:
                resolved_bases[fn]=base_used
        except Exception as e:
            failed+=1; errors.append(f"{fn}: {type(e).__name__}: {e}")
    manifest={"source":"OpenSim Rajagopal2016","model_url":RAJAGOPAL_MODEL_URL,
              "geometry_bases":RAJAGOPAL_GEOM_BASES,"resolved_bases":resolved_bases,"meta":meta,
              "downloaded":downloaded,"cached":cached,"failed":failed,"errors":errors}
    (root/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    return {"ok":True,"root":str(root),"model":meta.get("model_name","Rajagopal2016"),
            "mesh_count":len(meta["mesh_files"]),"downloaded":downloaded,"cached":cached,
            "failed":failed,"errors":errors}

def rajagopal_status():
    root=CACHE_ROOT
    mp=root/"manifest.json"
    if not mp.exists():
        return {"ready":False,"root":str(root)}
    try:
        m=json.loads(mp.read_text(encoding="utf-8"))
        n=len(m.get("meta",{}).get("mesh_files",[]))
        present=sum((root/"Geometry"/fn).exists() for fn in m.get("meta",{}).get("mesh_files",[]))
        return {"ready":present>0,"root":str(root),"mesh_count":n,"present":present,
                "failed":m.get("failed",0),"model":m.get("meta",{}).get("model_name","Rajagopal2016")}
    except Exception:
        return {"ready":False,"root":str(root)}

def _parse_vtp_ascii(path):
    """Parse ASCII VTK PolyData (.vtp). Returns V,F.
    OpenSim geometries may vary; unsupported binary/compressed VTP returns an empty mesh
    rather than inventing geometry.
    """
    path=Path(path)
    try:
        root=_strip_ns(ET.parse(path).getroot())
    except Exception:
        return np.empty((0,3),np.float32),np.empty((0,3),np.int32)
    piece=root.find(".//Piece")
    if piece is None:
        return np.empty((0,3),np.float32),np.empty((0,3),np.int32)
    pts=piece.find("./Points/DataArray")
    if pts is None or pts.attrib.get("format","ascii").lower()!="ascii":
        return np.empty((0,3),np.float32),np.empty((0,3),np.int32)
    vals=np.fromstring(pts.text or "",sep=" ",dtype=np.float32)
    if len(vals)%3:
        return np.empty((0,3),np.float32),np.empty((0,3),np.int32)
    V=vals.reshape(-1,3)
    polys=piece.find("./Polys")
    if polys is None:
        return V,np.empty((0,3),np.int32)
    arrays=polys.findall("./DataArray")
    conn=offs=None
    for da in arrays:
        nm=(da.attrib.get("Name","") or "").lower()
        arr=np.fromstring(da.text or "",sep=" ",dtype=np.int64)
        if "connect" in nm: conn=arr
        elif "offset" in nm: offs=arr
    if conn is None or offs is None:
        return V,np.empty((0,3),np.int32)
    faces=[]; st=0
    for en in offs:
        poly=conn[st:int(en)]; st=int(en)
        if len(poly)==3:
            faces.append(poly.tolist())
        elif len(poly)>3:
            # fan triangulation, preserving mesh surface
            for k in range(1,len(poly)-1):
                faces.append([int(poly[0]),int(poly[k]),int(poly[k+1])])
    return V,np.asarray(faces,dtype=np.int32)

def load_rajagopal_lower_body():
    """Load only lower-body real OpenSim display geometry.
    This is not CT/MRI-grade surface anatomy; it is the geometry distributed with
    the musculoskeletal model for biomechanical visualization.
    """
    st=rajagopal_status()
    if not st.get("ready"):
        return {"parts":[],"unsupported":[]}
    root=Path(st["root"])
    man=json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    bodies=man.get("meta",{}).get("bodies",{})
    parts=[]; unsupported=[]
    for body,seg in BODY_TO_SKEL.items():
        b=bodies.get(body,{})
        for mm in b.get("meshes",[]):
            f=root/"Geometry"/mm["file"]
            V,F=_parse_vtp_ascii(f)
            if V.size==0 or F.size==0:
                unsupported.append(mm["file"]); continue
            sc=np.asarray(mm.get("scale",[1,1,1]),dtype=np.float32)
            V=(V*sc).astype(np.float32)
            parts.append({"id":f"{body}:{mm['file']}","body":body,
                          "proximal":seg[0],"distal":seg[1],"V":V,"F":F})
    return {"parts":parts,"unsupported":unsupported}

def build_rajagopal_skel_frame(parts, joints, joint_names, frame=0):
    """Attach OpenSim visual geometry to SKEL for inspection.

    V110.3.21.1 fixes the principal error of .21.0:
    OpenSim long bones extend mainly from y≈0 proximally toward negative y distally.
    We therefore anchor MAX-Y at the proximal SKEL joint and map -Y to the segment axis.
    Scale is computed once from frame 0 and reused; no frame-by-frame stretching.

    This remains a bridge/visual validation, not OpenSim inverse kinematics.
    """
    J=np.asarray(joints,np.float32)
    names=[str(n) for n in joint_names]
    nm={n:i for i,n in enumerate(names)}
    out=[]

    hipr=nm.get("femur_r"); hipl=nm.get("femur_l"); pel=nm.get("pelvis")
    if hipr is not None and hipl is not None:
        lr0=np.asarray(J[0,hipr]-J[0,hipl],float)
        lr0=lr0/max(np.linalg.norm(lr0),1e-9)
        hip_width=float(np.linalg.norm(J[0,hipr]-J[0,hipl]))
    else:
        lr0=np.array([1.,0.,0.]); hip_width=0.20

    for p in parts:
        V=np.asarray(p["V"],np.float32)
        F=np.asarray(p["F"],np.int32)
        body=p["body"]

        # Pelvis is not a long bone. Keep its three OpenSim pieces together and
        # scale them from hip width rather than from pelvis->one hip length.
        if body=="pelvis":
            if pel is None:
                continue
            xmin,xmax=float(V[:,0].min()),float(V[:,0].max())
            xspan=max(xmax-xmin,1e-6)
            # Each pelvis visual piece may be unilateral; avoid explosive scaling.
            static_scale=min(max((hip_width*1.35)/xspan,0.35),3.0)
            X=V*static_scale
            center=np.mean(X,axis=0)
            X=X-center
            # OpenSim: x roughly anterior/posterior, z roughly left/right depending file;
            # use PCA only for whole pelvis orientation would be unstable across pieces.
            # Keep native orientation and translate all pieces to pelvis center.
            W=X + J[frame,pel]
            out.append({"id":p["id"],"body":body,"V":W.astype(np.float32),"F":F,
                        "scale":float(static_scale),"mapping":"pelvis_static"})
            continue

        aidx=nm.get(p["proximal"]); bidx=nm.get(p["distal"])
        if aidx is None or bidx is None:
            continue

        a0=J[0,aidx]; b0=J[0,bidx]
        at=J[frame,aidx]; bt=J[frame,bidx]
        target_len=float(np.linalg.norm(b0-a0))
        if target_len<1e-6:
            continue

        # Native OpenSim long-bone direction is predominantly -Y from proximal to distal.
        ymin=float(V[:,1].min()); ymax=float(V[:,1].max())
        yspan=max(ymax-ymin,1e-6)
        scale=target_len/yspan

        # Static scaled local coordinates: transverse dimensions preserved.
        X=V*scale
        ymax_s=float(X[:,1].max())
        longitudinal=(ymax_s-X[:,1])  # proximal max-Y -> 0; distal min-Y -> +L
        tx=X[:,0]-np.median(X[:,0])
        tz=X[:,2]-np.median(X[:,2])

        # Target segment frame.
        z=np.asarray(bt-at,float); z=z/max(np.linalg.norm(z),1e-9)
        x=lr0-np.dot(lr0,z)*z
        if np.linalg.norm(x)<1e-6:
            h=np.array([0.,0.,1.]); x=h-np.dot(h,z)*z
        x=x/max(np.linalg.norm(x),1e-9)
        y=np.cross(z,x); y=y/max(np.linalg.norm(y),1e-9)

        W=(at[None,:]
           + longitudinal[:,None]*z[None,:]
           + tx[:,None]*x[None,:]
           + tz[:,None]*y[None,:])

        out.append({"id":p["id"],"body":body,"V":W.astype(np.float32),"F":F,
                    "scale":float(scale),"mapping":"proximal_maxY_to_SKEL"})

    return out
