
from __future__ import annotations
import io, tempfile
from pathlib import Path
import numpy as np

def compact_mesh(V,F,max_faces=320):
    V=np.asarray(V,np.float32); F=np.asarray(F,np.int32)
    if V.ndim!=2 or F.ndim!=2 or len(F)==0:
        return np.empty((0,3),np.float32),np.empty((0,3),np.int32)
    step=max(1,int(np.ceil(len(F)/float(max_faces))))
    Fs=F[::step,:3]
    used=np.unique(Fs.reshape(-1))
    remap=np.full(len(V),-1,np.int32)
    remap[used]=np.arange(len(used),dtype=np.int32)
    return V[used],remap[Fs]

def merge_parts(parts,max_faces_each=260):
    verts=[]; faces=[]; off=0
    for p in parts:
        V,F=compact_mesh(p["V"],p["F"],max_faces=max_faces_each)
        if not len(V) or not len(F):
            continue
        verts.append(V); faces.append(F+off); off+=len(V)
    if not verts:
        return np.empty((0,3),np.float32),np.empty((0,3),np.int32)
    return np.vstack(verts).astype(np.float32),np.vstack(faces).astype(np.int32)

def build_hamner_sequence(bodies,joints,joint_names,skin_seq,build_frame_fn,max_faces_each=220):
    J=np.asarray(joints,np.float32)
    S=np.asarray(skin_seq,np.float32)
    Vframes=[]; Fref=None
    # Same source topology per frame, after deterministic per-body face sampling.
    for t in range(len(J)):
        parts=build_frame_fn(
            bodies,J,joint_names,frame=t,
            skin_vertices=S[t] if S.ndim==3 and t<len(S) else None
        )
        V,F=merge_parts(parts,max_faces_each=max_faces_each)
        if Fref is None:
            Fref=F
        # Topology should stay deterministic; if not, still use each frame vertices with first topology.
        Vframes.append(V)
    if Fref is None:
        Fref=np.empty((0,3),np.int32)
    # normalize vertex count defensively
    n=min([len(v) for v in Vframes]) if Vframes else 0
    if n:
        Vframes=[v[:n] for v in Vframes]
        Fref=Fref[np.all(Fref<n,axis=1)]
    return np.asarray(Vframes,np.float32),np.asarray(Fref,np.int32)

def muscle_paths_sequence(joints,joint_names,muscle_fn):
    J=np.asarray(joints,np.float32)
    seq=[]
    names=[]
    for t in range(len(J)):
        mm=muscle_fn(J[t],joint_names)
        if t==0:
            names=[m["id"] for m in mm]
        frame=[]
        for m in mm:
            frame.append(np.asarray(m["points"],np.float32))
        seq.append(frame)
    return seq,names

def render_combined_mp4(
    skin_seq,skin_faces,bone_seq,bone_faces,muscle_seq,
    fps=25,dpi=105,max_skin_faces=2300,max_bone_faces=3000
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import imageio.v2 as imageio

    S=np.asarray(skin_seq,np.float32)
    SF=np.asarray(skin_faces,np.int32)[:,:3]
    B=np.asarray(bone_seq,np.float32)
    BF=np.asarray(bone_faces,np.int32)[:,:3]
    T=len(S)

    sf_step=max(1,int(np.ceil(len(SF)/float(max_skin_faces))))
    SFd=SF[::sf_step]
    bf_step=max(1,int(np.ceil(len(BF)/float(max_bone_faces)))) if len(BF) else 1
    BFd=BF[::bf_step] if len(BF) else BF

    pts=S.reshape(-1,3)
    lo=np.nanpercentile(pts,1,axis=0); hi=np.nanpercentile(pts,99,axis=0)
    cen=(lo+hi)/2
    span=max(float(np.max(hi-lo)),1e-3)
    half=0.58*span

    tmp=Path(tempfile.mkdtemp(prefix="physiosentinel_all_layers_"))
    out=tmp/"PhysioSentinel_piel_esqueleto_musculo.mp4"

    with imageio.get_writer(str(out),fps=int(fps),codec="libx264",quality=7,macro_block_size=None) as wr:
        for t in range(T):
            fig=plt.figure(figsize=(6.8,7.4),dpi=dpi)
            ax=fig.add_subplot(111,projection="3d")

            if len(SFd):
                pc=Poly3DCollection(S[t][SFd],alpha=0.18,linewidths=0.0)
                pc.set_edgecolor("none")
                ax.add_collection3d(pc)

            if t<len(B) and len(BFd):
                pc2=Poly3DCollection(B[t][BFd],alpha=0.98,linewidths=0.0)
                pc2.set_edgecolor("none")
                ax.add_collection3d(pc2)

            if t<len(muscle_seq):
                for ptsm in muscle_seq[t]:
                    q=np.asarray(ptsm)
                    if len(q):
                        ax.plot(q[:,0],q[:,1],q[:,2],linewidth=3.0)

            ax.set_xlim(cen[0]-half,cen[0]+half)
            ax.set_ylim(cen[1]-half,cen[1]+half)
            ax.set_zlim(cen[2]-half,cen[2]+half)
            ax.set_box_aspect((1,1,1))
            ax.view_init(elev=8,azim=-78)
            ax.set_title(f"Piel + Esqueleto + Músculo · frame {t+1}/{T}")
            ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
            fig.tight_layout()
            fig.canvas.draw()
            rgba=np.asarray(fig.canvas.buffer_rgba())
            wr.append_data(rgba[:,:,:3])
            plt.close(fig)

    return str(out)

CAMERA_PRESETS = {
    "Frontal": {"elev": 8, "azim": -90, "eye": {"x": 0.0, "y": -2.35, "z": 0.15}},
    "Posterior": {"elev": 8, "azim": 90, "eye": {"x": 0.0, "y": 2.35, "z": 0.15}},
    "Lateral derecha": {"elev": 8, "azim": 0, "eye": {"x": 2.35, "y": 0.0, "z": 0.15}},
    "Lateral izquierda": {"elev": 8, "azim": 180, "eye": {"x": -2.35, "y": 0.0, "z": 0.15}},
    "Oblicua clínica": {"elev": 10, "azim": -55, "eye": {"x": 1.65, "y": -1.65, "z": 0.35}},
}

def camera_preset(name):
    return CAMERA_PRESETS.get(name, CAMERA_PRESETS["Oblicua clínica"])

def _add_layer_matplotlib(ax, layer, t, skin_seq, skin_faces, bone_seq, bone_faces, muscle_seq):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    S=np.asarray(skin_seq,np.float32)
    SF=np.asarray(skin_faces,np.int32)[:,:3]
    B=np.asarray(bone_seq,np.float32)
    BF=np.asarray(bone_faces,np.int32)[:,:3]

    want_skin = layer in ("Piel","Piel + Esqueleto","Todas")
    want_bone = layer in ("Esqueleto","Piel + Esqueleto","Esqueleto + Músculo","Todas")
    want_muscle = layer in ("Músculo funcional","Esqueleto + Músculo","Todas")

    if want_skin and len(SF):
        step=max(1,int(np.ceil(len(SF)/2300.0)))
        pc=Poly3DCollection(S[t][SF[::step]],alpha=0.18,linewidths=0.0)
        pc.set_edgecolor("none")
        ax.add_collection3d(pc)

    if want_bone and t < len(B) and len(BF):
        step=max(1,int(np.ceil(len(BF)/3200.0)))
        pc=Poly3DCollection(B[t][BF[::step]],alpha=0.98,linewidths=0.0)
        pc.set_edgecolor("none")
        ax.add_collection3d(pc)

    if want_muscle and t < len(muscle_seq):
        for ptsm in muscle_seq[t]:
            q=np.asarray(ptsm)
            if len(q):
                ax.plot(q[:,0],q[:,1],q[:,2],linewidth=3.0)

def render_layer_mp4(
    layer, orientation, skin_seq, skin_faces, bone_seq, bone_faces, muscle_seq,
    fps=25, dpi=105
):
    """Robust offline MP4 for the selected layer/combo and selected camera preset.
    Uses the same camera preset as the interactive viewer.
    """
    import tempfile
    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio_ffmpeg

    S=np.asarray(skin_seq,np.float32)
    T=len(S)
    pts=S.reshape(-1,3)
    lo=np.nanpercentile(pts,1,axis=0); hi=np.nanpercentile(pts,99,axis=0)
    cen=(lo+hi)/2
    span=max(float(np.max(hi-lo)),1e-3)
    half=0.58*span

    cam=camera_preset(orientation)
    tmp=Path(tempfile.mkdtemp(prefix="physiosentinel_layer_video_"))
    out=tmp/f"PhysioSentinel_{layer.replace(' ','_').replace('+','mas')}.mp4"

    w,h=760,820
    writer=imageio_ffmpeg.write_frames(
        str(out),(w,h),fps=int(fps),codec="libx264",
        quality=7,pix_fmt_in="rgb24",pix_fmt_out="yuv420p",
        macro_block_size=2
    )
    writer.send(None)
    try:
        for t in range(T):
            fig=plt.figure(figsize=(w/dpi,h/dpi),dpi=dpi)
            ax=fig.add_subplot(111,projection="3d")
            _add_layer_matplotlib(ax,layer,t,S,skin_faces,bone_seq,bone_faces,muscle_seq)
            ax.set_xlim(cen[0]-half,cen[0]+half)
            ax.set_ylim(cen[1]-half,cen[1]+half)
            ax.set_zlim(cen[2]-half,cen[2]+half)
            ax.set_box_aspect((1,1,1))
            ax.view_init(elev=float(cam["elev"]),azim=float(cam["azim"]))
            ax.set_title(f"{layer} · {orientation} · frame {t+1}/{T}")
            ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
            fig.tight_layout()
            fig.canvas.draw()
            rgb=np.asarray(fig.canvas.buffer_rgba())[:,:,:3]
            if rgb.shape[1] != w or rgb.shape[0] != h:
                rgb=rgb[:h,:w,:]
            writer.send(rgb.tobytes())
            plt.close(fig)
    finally:
        writer.close()
    return str(out)

def camera_from_angles(azim_deg, elev_deg, roll_deg=0.0, distance=2.35):
    az=np.deg2rad(float(azim_deg)); el=np.deg2rad(float(elev_deg))
    eye={
        "x": float(distance*np.cos(el)*np.cos(az)),
        "y": float(distance*np.cos(el)*np.sin(az)),
        "z": float(distance*np.sin(el)),
    }
    # default up then roll around view axis approximately
    r=np.deg2rad(float(roll_deg))
    up={"x": float(np.sin(r)), "y": float(np.cos(r)*0.0), "z": float(np.cos(r))}
    return {"eye":eye,"up":up}

def _robust_center_and_halfspan(points, zoom=1.0):
    P=np.asarray(points,float).reshape(-1,3)
    lo=np.nanpercentile(P,1,axis=0); hi=np.nanpercentile(P,99,axis=0)
    cen=0.5*(lo+hi)
    span=max(float(np.max(hi-lo)),1e-3)
    half=0.58*span/max(float(zoom),1e-3)
    return cen,half

def render_layer_mp4_manual_camera(
    layer, skin_seq, skin_faces, bone_seq, bone_faces, muscle_seq,
    azim=-55.0, elev=10.0, roll=0.0, zoom=1.0,
    center_offset=(0.0,0.0,0.0), fps=25, dpi=105
):
    """Export using explicit camera controls shared with the interactive viewer.
    Uses orthographic projection, fixed body-centred limits, and no per-frame recentering.
    """
    import tempfile
    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio_ffmpeg

    S=np.asarray(skin_seq,np.float32)
    cen,half=_robust_center_and_halfspan(S,zoom=zoom)
    cen=cen+np.asarray(center_offset,float)
    T=len(S)

    tmp=Path(tempfile.mkdtemp(prefix="physiosentinel_manual_cam_"))
    out=tmp/f"PhysioSentinel_{layer.replace(' ','_').replace('+','mas')}_manual.mp4"

    w,h=760,820
    writer=imageio_ffmpeg.write_frames(
        str(out),(w,h),fps=int(fps),codec="libx264",
        quality=7,pix_fmt_in="rgb24",pix_fmt_out="yuv420p",macro_block_size=2
    )
    writer.send(None)
    try:
        for t in range(T):
            fig=plt.figure(figsize=(w/dpi,h/dpi),dpi=dpi)
            ax=fig.add_subplot(111,projection="3d",proj_type="ortho")
            _add_layer_matplotlib(ax,layer,t,S,skin_faces,bone_seq,bone_faces,muscle_seq)
            ax.set_xlim(cen[0]-half,cen[0]+half)
            ax.set_ylim(cen[1]-half,cen[1]+half)
            ax.set_zlim(cen[2]-half,cen[2]+half)
            ax.set_box_aspect((1,1,1))
            try:
                ax.view_init(elev=float(elev),azim=float(azim),roll=float(roll))
            except TypeError:
                ax.view_init(elev=float(elev),azim=float(azim))
            ax.set_title(f"{layer} · cámara manual · frame {t+1}/{T}")
            ax.set_axis_off()
            fig.subplots_adjust(left=0,right=1,bottom=0,top=0.94)
            fig.canvas.draw()
            rgb=np.asarray(fig.canvas.buffer_rgba())[:,:,:3]
            writer.send(rgb.tobytes())
            plt.close(fig)
    finally:
        writer.close()
    return str(out)

def render_layer_mp4_multiview(
    layer, skin_seq, skin_faces, bone_seq, bone_faces, muscle_seq,
    fps=25, seconds_per_view=1.6, dpi=105, zoom=1.0
):
    """Sequential clinical orbit:
    frontal -> right lateral -> posterior -> left lateral -> superior -> oblique.
    Each view plays the full gait cycle before moving to the next view.
    """
    import tempfile
    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio_ffmpeg

    views=[
        ("Frontal",-90,5,0),
        ("Lateral derecha",0,5,0),
        ("Posterior",90,5,0),
        ("Lateral izquierda",180,5,0),
        ("Superior",-90,88,0),
        ("Oblicua",-55,12,0),
    ]
    S=np.asarray(skin_seq,np.float32)
    cen,half=_robust_center_and_halfspan(S,zoom=zoom)
    T=len(S)

    tmp=Path(tempfile.mkdtemp(prefix="physiosentinel_multiview_"))
    out=tmp/f"PhysioSentinel_{layer.replace(' ','_').replace('+','mas')}_multivista.mp4"
    w,h=760,820
    writer=imageio_ffmpeg.write_frames(
        str(out),(w,h),fps=int(fps),codec="libx264",
        quality=7,pix_fmt_in="rgb24",pix_fmt_out="yuv420p",macro_block_size=2
    )
    writer.send(None)
    try:
        for name,az,elev,roll in views:
            # one gait cycle per view
            for t in range(T):
                fig=plt.figure(figsize=(w/dpi,h/dpi),dpi=dpi)
                ax=fig.add_subplot(111,projection="3d",proj_type="ortho")
                _add_layer_matplotlib(ax,layer,t,S,skin_faces,bone_seq,bone_faces,muscle_seq)
                ax.set_xlim(cen[0]-half,cen[0]+half)
                ax.set_ylim(cen[1]-half,cen[1]+half)
                ax.set_zlim(cen[2]-half,cen[2]+half)
                ax.set_box_aspect((1,1,1))
                try:
                    ax.view_init(elev=elev,azim=az,roll=roll)
                except TypeError:
                    ax.view_init(elev=elev,azim=az)
                ax.set_title(f"{layer} · {name} · frame {t+1}/{T}")
                ax.set_axis_off()
                fig.subplots_adjust(left=0,right=1,bottom=0,top=0.94)
                fig.canvas.draw()
                rgb=np.asarray(fig.canvas.buffer_rgba())[:,:,:3]
                writer.send(rgb.tobytes())
                plt.close(fig)
    finally:
        writer.close()
    return str(out)


def build_hamner_sequence_filtered(bodies,joints,joint_names,skin_seq,build_frame_fn,
                                   include_ids,max_faces_each=160):
    """Build only selected Hamner bodies. Used as upper-body fallback when Denver supplies lower limbs."""
    J=np.asarray(joints,np.float32)
    S=np.asarray(skin_seq,np.float32)
    include=set(include_ids)
    Vframes=[]; Fref=None
    for t in range(len(J)):
        parts=build_frame_fn(
            bodies,J,joint_names,frame=t,
            skin_vertices=S[t] if S.ndim==3 and t<len(S) else None
        )
        parts=[p for p in parts if p.get("id") in include]
        V,F=merge_parts(parts,max_faces_each=max_faces_each)
        if Fref is None:
            Fref=F
        Vframes.append(V)
    if Fref is None:
        Fref=np.empty((0,3),np.int32)
    n=min([len(v) for v in Vframes]) if Vframes else 0
    if n:
        Vframes=[v[:n] for v in Vframes]
        Fref=Fref[np.all(Fref<n,axis=1)]
    return np.asarray(Vframes,np.float32),np.asarray(Fref,np.int32)

def merge_mesh_sequences(seq_a,faces_a,seq_b,faces_b):
    """Merge two same-length animated meshes with one shared topology."""
    A=np.asarray(seq_a,np.float32); B=np.asarray(seq_b,np.float32)
    FA=np.asarray(faces_a,np.int32); FB=np.asarray(faces_b,np.int32)
    if A.ndim!=3 or len(A)==0:
        return B,FB
    if B.ndim!=3 or len(B)==0:
        return A,FA
    T=min(len(A),len(B))
    na=A.shape[1]
    return np.concatenate([A[:T],B[:T]],axis=1),np.vstack([FA,FB+na]).astype(np.int32)
