from __future__ import annotations
from pathlib import Path
import re, math, xml.etree.ElementTree as ET
import numpy as np
import trimesh
from scipy.interpolate import CubicSpline
from visible_human_atlas import _simplify_connected_mesh

POSE_NAMES=['pelvis_tilt','pelvis_list','pelvis_rotation','hip_flexion_r','hip_adduction_r','hip_rotation_r','knee_angle_r','ankle_angle_r','subtalar_angle_r','mtp_angle_r','hip_flexion_l','hip_adduction_l','hip_rotation_l','knee_angle_l','ankle_angle_l','subtalar_angle_l','mtp_angle_l','lumbar_bending','lumbar_extension','lumbar_twist','thorax_bending','thorax_extension','thorax_twist','head_bending','head_extension','head_twist','scapula_abduction_r','scapula_elevation_r','scapula_upward_rot_r','shoulder_r_x','shoulder_r_y','shoulder_r_z','elbow_flexion_r','pro_sup_r','wrist_flexion_r','wrist_deviation_r','scapula_abduction_l','scapula_elevation_l','scapula_upward_rot_l','shoulder_l_x','shoulder_l_y','shoulder_l_z','elbow_flexion_l','pro_sup_l','wrist_flexion_l','wrist_deviation_l']
PI={n:i for i,n in enumerate(POSE_NAMES)}
MAP={
'adductor brevis':['add_brev'],'adductor longus':['add_long'],'adductor magnus':['add_mag1','add_mag2','add_mag3'],
'biceps femoris long':['bifemlh'],'biceps femoris short':['bifemsh'],'extensor digitorum longus':['ext_dig'],'extensor hallucis longus':['ext_hal'],'flexor digitorum longus':['flex_dig'],'flexor hallucis longus':['flex_hal'],'gastrocnemius lateral':['lat_gas'],'gastrocnemius medial':['med_gas'],'gluteus maximus':['glut_max1','glut_max2','glut_max3'],'gluteus medius':['glut_med1','glut_med2','glut_med3'],'gluteus minimus':['glut_min1','glut_min2','glut_min3'],'gracilis':['grac'],'iliacus':['iliacus'],'inferior gemellus':['gem'],'pectineus':['pect'],'peroneus longus':['per_long'],'fibularis longus':['per_long'],'piriformis':['peri'],'psoas major':['psoas'],'quadratus femoris':['quad_fem'],'rectus femoris':['rect_fem'],'sartorius':['sar'],'semimembranosus':['semimem'],'semitendinosus':['semiten'],'soleus':['soleus'],'superior gemellus':['gem'],'tensor fasciae latae':['tfl'],'tibialis anterior':['tib_ant'],'tibialis posterior':['tib_post'],'vastus intermedius':['vas_int'],'vastus lateralis':['vas_lat'],'vastus medialis':['vas_med']}
PROXY={'obturator externus':['quad_fem'],'obturator internus':['gem'],'plantaris':['lat_gas'],'popliteus':['bifemsh']}
SAMPLES=24

def _strip(t): return t.split('}')[-1]
def _fvec(x): return np.array([float(v) for v in (x or '').split()],float)
def _unit(v,fb=(1,0,0)):
    v=np.asarray(v,float); n=np.linalg.norm(v); return v/n if n>1e-12 else np.asarray(fb,float)
def _canon(s):
    s=str(s or '').lower().replace('quadratis femoris','quadratus femoris').replace('semitendonosus','semitendinosus').replace('illiacus','iliacus')
    return re.sub(r'\s+',' ',s).strip()

def _load_mesh(p,max_faces=300):
    m=trimesh.load_mesh(p,process=True)
    if isinstance(m,trimesh.Scene): m=trimesh.util.concatenate(tuple(m.geometry.values()))
    V,F=_simplify_connected_mesh(np.asarray(m.vertices,float),np.asarray(m.faces,int),int(max_faces))
    return trimesh.Trimesh(vertices=V,faces=F,process=True)

def _fun(container):
    for e in container.iter():
        tg=_strip(e.tag)
        if tg in ('NaturalCubicSpline','SimmSpline'):
            x=y=None
            for c in list(e):
                if _strip(c.tag)=='x':x=_fvec(c.text)
                elif _strip(c.tag)=='y':y=_fvec(c.text)
            if x is not None and y is not None:
                xx=[];yy=[]
                for a,b in zip(x,y):
                    a=float(a);b=float(b)
                    if xx and abs(a-xx[-1])<1e-12:yy[-1]=b
                    else:xx.append(a);yy.append(b)
                cs=CubicSpline(np.asarray(xx),np.asarray(yy),bc_type='natural',extrapolate=True)
                return lambda q,cs=cs,a=xx[0],b=xx[-1]:float(cs(np.clip(q,a,b)))
        if tg=='LinearFunction':
            for c in list(e):
                if _strip(c.tag)=='coefficients':
                    co=_fvec(c.text);return lambda q,a=float(co[0]),b=float(co[1]):a*q+b
        if tg=='Constant':
            val=0.0
            for c in list(e):
                if _strip(c.tag)=='value' and (c.text or '').strip():val=float(c.text)
            return lambda q,val=val:val
    return lambda q:0.0

def _basis(a,b,lat):
    y=_unit(np.asarray(b)-np.asarray(a),(0,-1,0));z=np.asarray(lat,float);z=z-y*np.dot(z,y);z=_unit(z,(0,0,1));x=_unit(np.cross(y,z),(1,0,0));z=_unit(np.cross(x,y),z);return np.column_stack([x,y,z])
def _simseg(sa,sb,sl,ta,tb,tl):
    Bs=_basis(sa,sb,sl);Bt=_basis(ta,tb,tl);sc=np.linalg.norm(np.asarray(tb)-ta)/max(np.linalg.norm(np.asarray(sb)-sa),1e-12);R=Bt@Bs.T;tr=np.asarray(ta)-sc*(R@np.asarray(sa));return sc,R,tr
def _apply(T,p):
    s,R,tr=T;return s*(R@np.asarray(p,float))+tr

def _resample(P,n=SAMPLES):
    P=np.asarray(P,float)
    if len(P)<2:return None
    seg=np.diff(P,axis=0);L=np.linalg.norm(seg,axis=1);cum=np.r_[0,np.cumsum(L)];tot=float(cum[-1])
    if tot<1e-12:return None
    sq=np.linspace(0,tot,n);idx=np.clip(np.searchsorted(cum,sq,side='right')-1,0,len(L)-1);f=(sq-cum[idx])/np.maximum(L[idx],1e-12)
    return P[idx]+seg[idx]*f[:,None]

def _rmf(P,initial):
    P=np.asarray(P,float);T=np.zeros_like(P);T[0]=_unit(P[1]-P[0]);T[-1]=_unit(P[-1]-P[-2])
    for i in range(1,len(P)-1):T[i]=_unit(_unit(P[i]-P[i-1])+_unit(P[i+1]-P[i]))
    N=np.zeros_like(P);B=np.zeros_like(P);n=np.asarray(initial,float);n=n-T[0]*np.dot(n,T[0]);N[0]=_unit(n,(1,0,0));B[0]=_unit(np.cross(T[0],N[0]),(0,0,1));N[0]=_unit(np.cross(B[0],T[0]),N[0])
    for i in range(len(P)-1):
        a,b=T[i],T[i+1];v=np.cross(a,b);sv=np.linalg.norm(v);c=np.clip(np.dot(a,b),-1,1);n=N[i]
        if sv>1e-10:
            ax=v/sv;ang=math.atan2(sv,c);n=n*math.cos(ang)+np.cross(ax,n)*math.sin(ang)+ax*np.dot(ax,n)*(1-math.cos(ang))
        n=n-b*np.dot(n,b);N[i+1]=_unit(n,N[i]);B[i+1]=_unit(np.cross(b,N[i+1]),B[i]);N[i+1]=_unit(np.cross(B[i+1],b),N[i+1])
    return N,B

def build_all76_rig(man,joints,joint_names,poses,osim_path,max_faces_each=300):
    J=np.asarray(joints,float);Q=np.asarray(poses,float);names=[str(x) for x in joint_names];JI={n:i for i,n in enumerate(names)}
    if J.ndim!=3 or Q.ndim!=2 or len(J)!=len(Q):raise ValueError('joints/poses SKEL no válidos para V149')
    root=ET.parse(osim_path).getroot();elems={}
    for e in root.iter():
        nm=e.attrib.get('name')
        if nm and any(_strip(c.tag)=='GeometryPath' for c in list(e)):elems[nm]=e
    paths={}
    for nm,e in elems.items():
        if not nm.endswith(('_r','_l')):continue
        pts=[]
        for pp in e.iter():
            typ=_strip(pp.tag)
            if typ not in ('PathPoint','ConditionalPathPoint','MovingPathPoint'):continue
            d={'type':typ,'body':'','location':None,'coordinate':None,'range':None,'moving':{}}
            for c in list(pp):
                tg=_strip(c.tag);txt=(c.text or '').strip()
                if tg=='body':d['body']=txt
                elif tg=='location' and txt:d['location']=_fvec(txt)
                elif tg=='coordinate':d['coordinate']=txt
                elif tg=='range' and txt:d['range']=_fvec(txt)
                elif tg in ('x_location','y_location','z_location'):d['moving'][tg[0]]=_fun(c)
                elif tg in ('x_coordinate','y_coordinate','z_coordinate'):d['moving'][tg[0]+'coord']=txt
            pts.append(d)
        paths[nm]=pts
    kt={}
    for side in ('r','l'):
        je=next(e for e in root.iter() if e.attrib.get('name')==f'knee_{side}' and _strip(e.tag)=='CustomJoint');st=next(c for c in list(je) if _strip(c.tag)=='SpatialTransform');arr=[]
        for ta in st.iter():
            if _strip(ta.tag)!='TransformAxis' or ta.attrib.get('name','') not in ('translation1','translation2','translation3'):continue
            axis=np.zeros(3);coords=[];fn=lambda q:0.0
            for c in list(ta):
                if _strip(c.tag)=='axis':axis=_fvec(c.text)
                elif _strip(c.tag)=='coordinates':coords=(c.text or '').split()
                elif _strip(c.tag)=='function':fn=_fun(c)
            arr.append((axis,coords,fn))
        kt[side]=arr
    def qcoord(n,t):
        if not n or n not in PI or PI[n]>=Q.shape[1]:return 0.0
        v=float(Q[t,PI[n]]);return -v if n.startswith('knee_angle_') else v
    def ksrc(side,t):
        q=qcoord(f'knee_angle_{side}',t);p=np.zeros(3)
        for a,c,f in kt[side]:p+=a*f(q if c else 0.0)
        return p
    hip={'r':np.array([-0.0707,-0.0661,0.0835]),'l':np.array([-0.0707,-0.0661,-0.0835])};back=np.array([-0.1007,0.0815,0.0]);ank=np.array([0,-0.43,0.0]);sub={'r':np.array([-0.04877,-0.04195,0.00792]),'l':np.array([-0.04877,-0.04195,-0.00792])};mtp={'r':np.array([0.1788,-0.002,0.00108]),'l':np.array([0.1788,-0.002,-0.00108])}
    def pelvisT(t):
        src=np.stack([hip['l'],hip['r'],back]);tgt=np.stack([J[t,JI['femur_l']],J[t,JI['femur_r']],J[t,JI['lumbar_body']]]);ms=src.mean(0);mt=tgt.mean(0);X=src-ms;Y=tgt-mt;U,D,Vt=np.linalg.svd((Y.T@X)/len(src));S=np.eye(3)
        if np.linalg.det(U@Vt)<0:S[-1,-1]=-1
        R=U@S@Vt;sc=np.trace(np.diag(D)@S)/max((X*X).sum()/len(src),1e-12);return sc,R,mt-sc*(R@ms)
    def Ts(t,side):
        h=J[t,JI[f'femur_{side}']];k=J[t,JI[f'tibia_{side}']];a=J[t,JI[f'talus_{side}']];he=J[t,JI[f'calcn_{side}']];to=J[t,JI[f'toes_{side}']];lat=_unit(J[t,JI['femur_r']]-J[t,JI['femur_l']],(0,0,1));tl=lat if side=='r' else -lat;sl=np.array([0,0,1 if side=='r' else -1.0]);Tf=_simseg(np.zeros(3),ksrc(side,t),sl,h,k,tl);Tt=_simseg(np.zeros(3),ank,sl,k,a,tl);Ttal=_simseg(np.zeros(3),sub[side],sl,a,he,tl);Tc=_simseg(np.zeros(3),mtp[side],sl,he,to,tl);sc,R,_=Tc
        return {'pelvis':pelvisT(t),f'femur_{side}':Tf,f'tibia_{side}':Tt,f'talus_{side}':Ttal,f'calcn_{side}':Tc,f'toes_{side}':(sc,R,to.copy())}
    def evlocal(pp,t):
        loc=np.array(pp['location'],float) if pp['location'] is not None else np.zeros(3)
        if pp['type']=='ConditionalPathPoint':
            q=qcoord(pp['coordinate'],t);lo,hi=pp['range'];return loc,bool(lo<=q<=hi)
        if pp['type']=='MovingPathPoint':
            out=loc.copy()
            for k,i in zip('xyz',range(3)):
                fn=pp['moving'].get(k);co=pp['moving'].get(k+'coord')
                if fn is not None:out[i]=fn(qcoord(co,t))
            return out,True
        return loc,True
    def exact(comp,side,t):
        nm=f'{comp}_{side}';tt=Ts(t,side);P=[]
        for pp in paths.get(nm,[]):
            loc,on=evlocal(pp,t)
            if on and pp['body'] in tt:P.append(_apply(tt[pp['body']],loc))
        return np.asarray(P,float),tt
    parts=sorted([p for p in man.get('parts',[]) if p.get('kind')=='muscle'],key=lambda p:(str(p.get('side')),str(p.get('name')),str(p.get('path'))))
    if len(parts)!=76:raise ValueError(f'Se esperaban 76 músculos Denver y se detectaron {len(parts)}')
    # atlas anatomical transverse axes from femur centroids/long axes approximated by Denver shared coordinates
    fem={}
    for side in ('l','r'):
        pp=next((p for p in man.get('parts',[]) if p.get('kind')=='bone' and p.get('side')==side and p.get('name')=='femur'),None)
        if pp:fem[side]=_load_mesh(pp['path'],500)
    if len(fem)==2:
        lr=_unit(np.asarray(fem['r'].vertices).mean(0)-np.asarray(fem['l'].vertices).mean(0),(1,0,0))
        longs=[]
        for m in (fem['l'],fem['r']):
            X=np.asarray(m.vertices)-np.asarray(m.vertices).mean(0);longs.append(_unit(np.linalg.svd(X,full_matrices=False)[2][0],(0,1,0)))
        sup=_unit(longs[0]+longs[1],(0,1,0));sup=_unit(sup-lr*np.dot(sup,lr),(0,1,0));ant=_unit(np.cross(lr,sup),(0,0,1))
    else:lr=np.array([1.,0,0]);sup=np.array([0,1.,0]);ant=np.array([0,0,1.])
    # unit inference: Denver STL usually mm
    all_extent=[]
    for p in parts[:8]:
        try:
            m=_load_mesh(p['path'],80);all_extent.append(np.ptp(np.asarray(m.vertices),axis=0).max())
        except Exception:pass
    us=0.001 if all_extent and np.nanmedian(all_extent)>10 else 1.0
    Fout=[];Uout=[];Tout=[];MI=[];off=0;mus_names=[];mus_sides=[];qualities=[];maps=[];belly=[]
    for mi,p in enumerate(parts):
        side=p.get('side') or 'r';name=_canon(p.get('name'));m=_load_mesh(p['path'],max_faces_each);V=np.asarray(m.vertices,float)*us;F=np.asarray(m.faces,int);C=V.mean(0);X=V-C;vh=np.linalg.svd(X,full_matrices=False)[2];axis=_unit(vh[0]);
        if np.dot(axis,sup)>0:axis=-axis
        t1=lr-axis*np.dot(lr,axis)
        if np.linalg.norm(t1)<0.2:t1=ant-axis*np.dot(ant,axis)
        t1=_unit(t1,(1,0,0));t2=_unit(np.cross(axis,t1),(0,0,1));t1=_unit(np.cross(t2,axis),t1);qv=X@axis;q0,q1=np.quantile(qv,[.01,.99]);u=np.clip((qv-q0)/max(q1-q0,1e-12),0,1);tr=np.column_stack([X@t1,X@t2])
        Fout.append(F+off);Uout.append(u.astype(np.float32));Tout.append(tr.astype(np.float32));MI.extend([mi]*len(V));off+=len(V);mus_names.append(name);mus_sides.append(side);comps=MAP.get(name,PROXY.get(name,[]));qual='proxy' if name in PROXY else ('composite' if len(comps)>1 else ('shared' if name in ('inferior gemellus','superior gemellus') else 'exact'));qualities.append(qual);maps.append('+'.join(comps));belly.append(float(q1-q0))
    nmus=len(parts);nf=len(J);PS=np.full((nmus,nf,SAMPLES,3),np.nan,np.float32);PN=np.full_like(PS,np.nan);PB=np.full_like(PS,np.nan);ST=np.zeros((nmus,nf),np.float32);EN=np.zeros_like(ST);TR=np.ones_like(ST)
    hipspan_src=np.linalg.norm(hip['r']-hip['l']);back_src=np.linalg.norm(back-(hip['r']+hip['l'])/2)
    for mi,(name,side) in enumerate(zip(mus_names,mus_sides)):
        comps=MAP.get(name,PROXY.get(name,[]))
        for t in range(nf):
            curves=[];tt=None
            for c in comps:
                P,tt=exact(c,side,t);R=_resample(P)
                if R is not None:curves.append(R)
            if not curves:continue
            P=np.mean(np.stack(curves),axis=0) if len(curves)>1 else curves[0];PS[mi,t]=P
            lat=_unit(J[t,JI['femur_r']]-J[t,JI['femur_l']],(0,0,1));lat=lat if side=='r' else -lat;N,B=_rmf(P,lat);PN[mi,t]=N;PB[mi,t]=B;L=float(np.linalg.norm(np.diff(P,axis=0),axis=1).sum())
            # longitudinal V9
            sfem=tt[f'femur_{side}'][0];stib=tt[f'tibia_{side}'][0];sL=float(np.clip(.5*(sfem+stib),0.75,2.5));
            if name.startswith('gastrocnemius'):maxocc,stf=.68,.08
            elif name=='soleus':maxocc,stf=.62,.16
            elif name=='rectus femoris':maxocc,stf=.66,.10
            elif name.startswith('vastus'):maxocc,stf=.66,.06
            else:maxocc,stf=.66,.08
            desired=min(max(belly[mi]*sL,.30*L),maxocc*L);sta=min(stf*L,max(0,L-desired));ena=min(L,sta+desired);ST[mi,t]=sta/max(L,1e-12);EN[mi,t]=ena/max(L,1e-12)
            # V11 anatomical transverse scale: geometric mean of real segment scales traversed by source paths
            vals=[]
            for comp in comps:
                for pp in paths.get(f'{comp}_{side}',[]):
                    body=pp.get('body');
                    if body in tt and body not in ('pelvis',):vals.append(float(tt[body][0]))
                    elif body=='pelvis':
                        hs=np.linalg.norm(J[t,JI['femur_r']]-J[t,JI['femur_l']]);bt=np.linalg.norm(J[t,JI['lumbar_body']]-(J[t,JI['femur_r']]+J[t,JI['femur_l']])/2);vals.append(float(math.sqrt((hs/max(hipspan_src,1e-12))*(bt/max(back_src,1e-12)))))
            vals=[v for v in vals if np.isfinite(v) and v>0];TR[mi,t]=float(np.exp(np.mean(np.log(vals)))) if vals else sL
    return {'u':np.concatenate(Uout).astype(np.float32),'transverse':np.vstack(Tout).astype(np.float32),'faces':np.vstack(Fout).astype(np.int32),'muscle_index':np.asarray(MI,np.int16),'path_samples':PS,'path_normals':PN,'path_binormals':PB,'start_frac':ST,'end_frac':EN,'trans_scale':TR,'muscle_names':mus_names,'muscle_sides':mus_sides,'mapping_quality':qualities,'hamner_mapping':maps,'proxy_count':sum(q=='proxy' for q in qualities),'version':'151'}
