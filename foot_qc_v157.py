import numpy as np
import re


def _norm(s):
    return re.sub(r'[^a-z0-9]', '', str(s).lower())


def _unit(v, fallback=(0.0, 1.0, 0.0)):
    v=np.asarray(v, float)
    n=float(np.linalg.norm(v))
    if not np.isfinite(n) or n < 1e-12:
        return np.asarray(fallback, float)
    return v/n


def _angle_deg(a,b):
    a=_unit(a); b=_unit(b)
    c=float(np.clip(np.dot(a,b),-1.0,1.0))
    return float(np.degrees(np.arccos(c)))


def _robust_threshold(x, floor):
    x=np.asarray(x,float)
    x=x[np.isfinite(x)]
    if x.size==0:
        return float(floor)
    med=float(np.median(x)); mad=float(np.median(np.abs(x-med)))
    return float(max(floor, med + 3.0*1.4826*mad))


def compute_foot_qc(mesh_seq):
    """V157 foot/ankle quality control from a SKEL sequence.

    This is a QC layer, not a diagnostic model. It does not alter q(t), joints,
    skin vertices, contact events or clinical results. When detector-level
    Heel/Toe confidences are absent (legacy NPZ), geometric confidence is
    reported separately and never presented as detector confidence.
    """
    if not isinstance(mesh_seq,dict):
        return None
    J=mesh_seq.get('joints'); P=mesh_seq.get('poses')
    if J is None:
        return None
    J=np.asarray(J,float)
    if J.ndim!=3 or J.shape[-1]!=3 or len(J)<3:
        return None
    names=[str(x) for x in mesh_seq.get('joint_names',[])]
    nm={_norm(n):i for i,n in enumerate(names)}
    req=['pelvis','thorax','talusr','calcnr','toesr','talusl','calcnl','toesl']
    if not all(k in nm for k in req):
        return None

    # Stable anatomical up estimate from pelvis -> thorax, independent of raw XYZ axis labels.
    up=_unit(np.nanmedian(J[:,nm['thorax'],:]-J[:,nm['pelvis'],:],axis=0),(0,1,0))
    P=np.asarray(P,float) if P is not None else None
    raw_scores=mesh_seq.get('foot_landmark_scores')
    detector_scores_available=isinstance(raw_scores,dict)

    out={'version':'157-foot-qc-v1','frames':int(len(J)),'detector_scores_available':bool(detector_scores_available),'sides':{}}
    contact_series={}
    for side,lab,aq,sq,mq in [('r','Derecho',7,8,9),('l','Izquierdo',14,15,16)]:
        ia=nm[f'talus{side}']; ih=nm[f'calcn{side}']; it=nm[f'toes{side}']
        ankle=J[:,ia,:]; heel=J[:,ih,:]; toe=J[:,it,:]
        ht=toe-heel; ah=heel-ankle; at=toe-ankle
        lengths=np.linalg.norm(ht,axis=1)
        finite=np.isfinite(np.c_[ankle,heel,toe]).all(axis=1) & np.isfinite(lengths) & (lengths>1e-8)
        med_len=float(np.nanmedian(lengths[finite])) if finite.any() else np.nan
        cv=float(np.nanstd(lengths[finite])/max(abs(np.nanmean(lengths[finite])),1e-12)*100.0) if finite.any() else np.nan

        # Directional continuity of Heel->Toe; orientation jumps are more informative than length alone.
        direction=np.full_like(ht,np.nan)
        oklen=np.linalg.norm(ht,axis=1)>1e-8
        direction[oklen]=ht[oklen]/np.linalg.norm(ht[oklen],axis=1)[:,None]
        dtheta=[]
        for i in range(1,len(direction)):
            if np.isfinite(direction[i-1]).all() and np.isfinite(direction[i]).all():
                dtheta.append(_angle_deg(direction[i-1],direction[i]))
            else:
                dtheta.append(np.nan)
        dtheta=np.asarray(dtheta,float)
        jump_thr=_robust_threshold(dtheta,15.0)
        orient_jumps=int(np.sum(np.isfinite(dtheta)&(dtheta>jump_thr)))
        max_orient_step=float(np.nanmax(dtheta)) if np.isfinite(dtheta).any() else np.nan

        # Talus-calcaneus-toes coherence: included angle and temporal stability.
        chain_ang=np.full(len(J),np.nan,float)
        for i in range(len(J)):
            v1=ankle[i]-heel[i]; v2=toe[i]-heel[i]
            if np.isfinite(v1).all() and np.isfinite(v2).all() and np.linalg.norm(v1)>1e-8 and np.linalg.norm(v2)>1e-8:
                chain_ang[i]=_angle_deg(v1,v2)
        chain_sd=float(np.nanstd(chain_ang)) if np.isfinite(chain_ang).any() else np.nan
        chain_step=float(np.nanmax(np.abs(np.diff(chain_ang)))) if np.isfinite(chain_ang).sum()>1 else np.nan

        # q(t) continuity. No smoothing is applied here.
        ankle_q=np.rad2deg(P[:,aq]) if P is not None and P.ndim==2 and P.shape[1]>aq else np.full(len(J),np.nan)
        dq=np.diff(ankle_q); ddq=np.diff(ankle_q,n=2)
        dq_thr=_robust_threshold(np.abs(dq),6.0)
        q_jump_count=int(np.sum(np.isfinite(dq)&(np.abs(dq)>dq_thr)))
        q_rom=float(np.nanmax(ankle_q)-np.nanmin(ankle_q)) if np.isfinite(ankle_q).any() else np.nan
        q_max_step=float(np.nanmax(np.abs(dq))) if np.isfinite(dq).any() else np.nan
        q_curv_rms=float(np.sqrt(np.nanmean(ddq**2))) if np.isfinite(ddq).any() else np.nan

        # Geometric Heel/Toe confidence. Deliberately separate from model/detector probability.
        finite_pct=100.0*float(np.mean(finite))
        cv_score=float(np.clip(1.0-(cv/3.0 if np.isfinite(cv) else 1.0),0,1))
        cont_score=float(np.clip(1.0-(np.nanmedian(dtheta)/12.0 if np.isfinite(dtheta).any() else 1.0),0,1))
        geom_conf=100.0*(0.45*(finite_pct/100.0)+0.30*cv_score+0.25*cont_score)
        det_conf=np.nan
        if detector_scores_available:
            try:
                vals=[]
                for key in (f'{side}_heel',f'{side}_toe',f'{side}_ankle'):
                    if key in raw_scores:
                        a=np.asarray(raw_scores[key],float); vals.extend(a[np.isfinite(a)].tolist())
                if vals: det_conf=100.0*float(np.clip(np.nanmean(vals),0,1))
            except Exception:
                det_conf=np.nan

        # Auxiliary 3D contact hint: low distal height + low vertical speed.
        # It is never called IC/TO and never replaces the 2D event detector.
        distal=(heel+toe)/2.0
        h=distal@up
        vh=np.gradient(h)
        h0=float(np.nanpercentile(h,10)); hr=max(float(np.nanpercentile(h,90)-h0),1e-9)
        hthr=h0+0.22*hr
        vabs=np.abs(vh); vthr=_robust_threshold(vabs,0.0)
        # robust threshold can be too permissive; cap at p60 to retain low-velocity meaning
        if np.isfinite(vabs).any(): vthr=min(vthr,float(np.nanpercentile(vabs,60)))
        contact=np.isfinite(h)&np.isfinite(vh)&(h<=hthr)&(vabs<=vthr)
        contact_series[side]=contact
        contact_pct=100.0*float(np.mean(contact))

        subt=np.rad2deg(P[:,sq]) if P is not None and P.ndim==2 and P.shape[1]>sq else np.full(len(J),np.nan)
        mtp=np.rad2deg(P[:,mq]) if P is not None and P.ndim==2 and P.shape[1]>mq else np.full(len(J),np.nan)
        subt_rom=float(np.nanmax(subt)-np.nanmin(subt)) if np.isfinite(subt).any() else np.nan
        mtp_rom=float(np.nanmax(mtp)-np.nanmin(mtp)) if np.isfinite(mtp).any() else np.nan
        if (not np.isfinite(subt_rom)) or subt_rom < 1.0:
            inv_quality='Orientativo · baja fiabilidad'
            inv_reason='La señal subtalar no muestra ROM dinámico suficiente para inferir inversión/eversión con confianza.'
        elif geom_conf < 70 or max_orient_step>25:
            inv_quality='Orientativo · confianza limitada'
            inv_reason='La geometría distal presenta confianza/continuidad insuficiente para equiparar inversión/eversión a cadera o rodilla.'
        else:
            inv_quality='Orientativo · confianza moderada'
            inv_reason='Estimación SKEL 3D; no equivale a una medición clínica instrumentada del retropié.'

        geom_quality='Alta' if geom_conf>=85 else ('Moderada' if geom_conf>=70 else 'Baja')
        temporal_quality='Alta' if q_jump_count==0 and (not np.isfinite(q_max_step) or q_max_step<=8) else ('Moderada' if q_jump_count<=2 else 'Baja')
        chain_quality='Alta' if np.isfinite(chain_step) and chain_step<=3 else ('Moderada' if np.isfinite(chain_step) and chain_step<=8 else 'Baja')
        out['sides'][side]={
            'label':lab,'finite_pct':finite_pct,'heel_toe_length_median':med_len,'heel_toe_length_cv_pct':cv,
            'heel_toe_geometric_confidence_pct':float(geom_conf),'detector_confidence_pct':float(det_conf) if np.isfinite(det_conf) else None,
            'confidence_type':'detector+geométrica' if np.isfinite(det_conf) else 'geométrica (NPZ sin score detector)',
            'orientation_max_step_deg':max_orient_step,'orientation_jump_threshold_deg':float(jump_thr),'orientation_jump_count':orient_jumps,
            'ankle_rom_deg':q_rom,'ankle_max_step_deg_per_frame':q_max_step,'ankle_jump_threshold_deg_per_frame':float(dq_thr),
            'ankle_jump_count':q_jump_count,'ankle_curvature_rms_deg_per_frame2':q_curv_rms,
            'talus_calcaneus_toes_angle_sd_deg':chain_sd,'talus_calcaneus_toes_max_step_deg':chain_step,
            'contact_hint_pct':contact_pct,'contact_hint_semantics':'auxiliar 3D; no IC/TO ni GRF',
            'subtalar_rom_deg':subt_rom,'mtp_rom_deg':mtp_rom,'inversion_eversion_quality':inv_quality,'inversion_eversion_reason':inv_reason,
            'geometry_quality':geom_quality,'temporal_quality':temporal_quality,'chain_quality':chain_quality,
        }
    out['contact_hint_overlap_pct']=100.0*float(np.mean(contact_series['r']&contact_series['l'])) if 'r' in contact_series and 'l' in contact_series else np.nan
    out['contact_note']='La detección de contacto 3D es solo un control auxiliar. PhysioSentinel conserva IC/TO 2D como fuente principal cuando están disponibles y fiables.'
    return out


def qc_rows(qc):
    if not isinstance(qc,dict): return []
    rows=[]
    for side in ('r','l'):
        d=(qc.get('sides') or {}).get(side)
        if not isinstance(d,dict): continue
        rows.append({
            'Lado':d.get('label'),
            'Confianza talón/punta geométrica (%)':d.get('heel_toe_geometric_confidence_pct'),
            'Confianza detector (%)':d.get('detector_confidence_pct'),
            'CV longitud Heel→Toe (%)':d.get('heel_toe_length_cv_pct'),
            'Salto máx orientación pie (°/frame)':d.get('orientation_max_step_deg'),
            'Saltos orientación detectados':d.get('orientation_jump_count'),
            'ROM tobillo (°)':d.get('ankle_rom_deg'),
            'Salto máx tobillo (°/frame)':d.get('ankle_max_step_deg_per_frame'),
            'Saltos tobillo detectados':d.get('ankle_jump_count'),
            'Coherencia talus–calcáneo–toes (máx Δ°)':d.get('talus_calcaneus_toes_max_step_deg'),
            'Contacto 3D auxiliar (%)':d.get('contact_hint_pct'),
            'ROM subtalar (°)':d.get('subtalar_rom_deg'),
            'ROM MTP (°)':d.get('mtp_rom_deg'),
            'Fiabilidad inversión/eversión':d.get('inversion_eversion_quality'),
        })
    return rows
