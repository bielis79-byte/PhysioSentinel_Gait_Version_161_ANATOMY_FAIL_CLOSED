from pathlib import Path
import base64, json


def rigged_model_html(motion, safe_gpu=True):
    asset=Path(__file__).with_name('PhysioSentinel_Skeleton_Rigged_v1.glb')
    if not asset.exists():
        return "<div style='padding:20px;color:#ff8c8c'>Falta PhysioSentinel_Skeleton_Rigged_v1.glb en el despliegue.</div>"
    b64=base64.b64encode(asset.read_bytes()).decode('ascii')
    motion_json=json.dumps(motion,ensure_ascii=False,separators=(',',':'))
    safe_js='true' if safe_gpu else 'false'
    tpl=r'''<!doctype html><html><head><meta charset="utf-8"><style>
html,body{margin:0;background:#07111d;color:#dce8f5;font:13px system-ui;height:100%;overflow:hidden}.wrap{display:grid;grid-template-columns:360px 1fr;height:720px}.hud{padding:14px;background:#0a1420;border-right:1px solid #1d3347;overflow:auto}.view{position:relative;min-width:0}.view canvas{position:absolute;inset:0;width:100%;height:100%;display:block}.ok{color:#62e6a7;font-weight:800}.safe{color:#ffcf6e;font-weight:700}.stage{margin:6px 0;padding:6px 8px;border-radius:6px;background:#0c1b2a}.stage.ok2{color:#6ee7a8}.stage.err{color:#ff8c8c}.frame{font-weight:800;margin:8px 0}.controls{display:flex;flex-wrap:wrap;gap:6px;margin:9px 0}button{padding:7px 10px;border:0;border-radius:7px;cursor:pointer}button:disabled{opacity:.38}.active{outline:2px solid #63d9ff}.diag{white-space:pre-wrap;color:#9fc4e8;line-height:1.35}.warn{color:#ffca74;margin-top:10px}.dl{display:inline-block;padding:7px 10px;border-radius:7px;background:#dce8f5;color:#07111d;text-decoration:none;font-weight:700;margin-top:6px}input{width:100%}</style></head><body><div class="wrap"><div class="hud">
<div class="ok">V107.4 · VALIDACIÓN ARTICULAR CUANTITATIVA · GLB LOCAL</div><div class="safe">🛡️ GPU SAFE · pivotes conocidos, sin clasificación espacial</div>
<div id="rigStage" class="stage">Rig: iniciando…</div><div id="engineStage" class="stage">Three.js: pendiente</div><div id="loaderStage" class="stage">GLTFLoader: pendiente</div><div id="modelStage" class="stage">Modelo V107: pendiente</div><div id="nodesStage" class="stage">Nodos anatómicos: pendiente</div><div id="renderStage" class="stage">Primer render: pendiente</div>
<div class="frame" id="frame">Frame —</div><div class="controls"><button id="play">▶ Reproducir</button><button id="prev">−1</button><button id="next">+1</button><button id="reset">Reset</button><button id="fit">Reencuadrar</button><button id="record">⏺ Grabar ciclo</button></div>
<div class="controls"><button id="rig">Rig</button><button id="bones" disabled>Esqueleto</button><button id="both" disabled class="active">Rig + Esqueleto</button></div><div id="download"></div><input id="scrub" type="range" min="0" max="0" value="0"><div id="metrics" class="stage">Validación: pendiente</div><div id="diag" class="diag">Preparando secuencia…</div><div class="warn">V107.4 no cambia la cinemática de V107.3. Añade validación cuantitativa frame a frame entre los pivotes reales del GLB y los landmarks V104, con líneas de error y error normalizado por altura corporal.</div></div>
<div class="view"><canvas id="gl"></canvas></div></div>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.180.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.180.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from 'three'; import {OrbitControls} from 'three/addons/controls/OrbitControls.js'; import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
const SAFE=__SAFE__, motion=__MOTION__, F=motion.frames||[], MODEL_B64='__MODEL__'; const $=s=>document.querySelector(s); const canvas=$('#gl');
const setStage=(id,text,kind='')=>{const e=$(id);e.textContent=text;e.className='stage '+(kind==='ok'?'ok2':kind==='err'?'err':'')};
setStage('#rigStage','Rig: OK · '+F.length+' frames','ok'); setStage('#engineStage','Three.js: OK','ok'); setStage('#loaderStage','GLTFLoader + OrbitControls: OK','ok');
const renderer=new THREE.WebGLRenderer({canvas,antialias:false,powerPreference:'low-power',preserveDrawingBuffer:false}); renderer.setPixelRatio(SAFE?1:Math.min(devicePixelRatio,1.5)); renderer.setClearColor(0x07111d,1); renderer.outputColorSpace=THREE.SRGBColorSpace;
const scene=new THREE.Scene(); scene.background=new THREE.Color(0x07111d); const camera=new THREE.PerspectiveCamera(32,1,.01,100); const controls=new OrbitControls(camera,canvas); controls.enableDamping=false;
scene.add(new THREE.AmbientLight(0xffffff,1.4)); scene.add(new THREE.HemisphereLight(0xffffff,0x25364a,1.8)); const key=new THREE.DirectionalLight(0xffffff,1.8); key.position.set(3,5,5); scene.add(key); const grid=new THREE.GridHelper(4,16,0x2a4a67,0x173047); scene.add(grid);
const rigGroup=new THREE.Group(); scene.add(rigGroup); const rigMat=new THREE.LineBasicMaterial({color:0x00e5ff}); const ptMat=new THREE.PointsMaterial({color:0xffd166,size:.035,sizeAttenuation:true});
const modelJointGroup=new THREE.Group(); scene.add(modelJointGroup); const modelPtMat=new THREE.PointsMaterial({color:0xff4fd8,size:.055,sizeAttenuation:true});
const errorGroup=new THREE.Group(); scene.add(errorGroup); const errorMat=new THREE.LineBasicMaterial({color:0xff355e});
const JOINTS=['Head','Neck','LShoulder','RShoulder','LElbow','RElbow','LWrist','RWrist','Hip','LHip','RHip','LKnee','RKnee','LAnkle','RAnkle','LHeel','RHeel','LBigToe','RBigToe'];
const LINKS=[['Neck','LShoulder'],['LShoulder','LElbow'],['LElbow','LWrist'],['Neck','RShoulder'],['RShoulder','RElbow'],['RElbow','RWrist'],['Neck','Hip'],['LHip','RHip'],['Hip','LHip'],['Hip','RHip'],['LHip','LKnee'],['LKnee','LAnkle'],['LAnkle','LBigToe'],['RHip','RKnee'],['RKnee','RAnkle'],['RAnkle','RBigToe']];
function rawP(f,n){const p=f?.joints?.[n]||f?.xyz?.[n]||f?.points?.[n]; if(!p)return null; const a=Array.isArray(p)?p:[p.x,p.y,p.z]; return a.length>=3&&a.every(Number.isFinite)?new THREE.Vector3(+a[0],+a[1],+a[2]):null}
function avg(...vs){const a=vs.filter(Boolean); if(!a.length)return null; return a.reduce((s,v)=>s.add(v),new THREE.Vector3()).multiplyScalar(1/a.length)}
const f0=F[0]||{}; const hip0=avg(rawP(f0,'LHip'),rawP(f0,'RHip'),rawP(f0,'Hip'))||new THREE.Vector3(); let ys=[]; for(const n of JOINTS){const p=rawP(f0,n);if(p)ys.push(p.y)} const bodyRange=Math.max(.5,(ys.length?(Math.max(...ys)-Math.min(...ys)):1)); const MAP_SCALE=2.65/bodyRange;
function MAPP(p){return p?p.clone().sub(hip0).multiplyScalar(MAP_SCALE):null} function P(f,n){return MAPP(rawP(f,n))}
let model=null,nodes={},current=0,playing=false,last=0,mode='both',recorder=null,recording=false,chunks=[];const FRAME_MS=1000/24;
const BODY_H=2.65;
const VALIDATE={LHip:'Femur_L',RHip:'Femur_R',LKnee:'Tibia_L',RKnee:'Tibia_R',LAnkle:'Foot_L',RAnkle:'Foot_R',LShoulder:'Humerus_L',RShoulder:'Humerus_R',LElbow:'Forearm_L',RElbow:'Forearm_R',LWrist:'Hand_L',RWrist:'Hand_R'};
const required=['Pelvis','Spine','Thorax','Neck','Head','Femur_L','Tibia_L','Foot_L','Femur_R','Tibia_R','Foot_R','Humerus_L','Forearm_L','Hand_L','Humerus_R','Forearm_R','Hand_R'];
const AXIS_YN=new THREE.Vector3(0,-1,0),AXIS_YP=new THREE.Vector3(0,1,0),AXIS_ZP=new THREE.Vector3(0,0,1);
const BASE_LEN={Spine:.55,Neck:.18,Femur_L:.72,Femur_R:.72,Tibia_L:.67,Tibia_R:.67,Foot_L:.30,Foot_R:.30,Humerus_L:.52,Humerus_R:.52,Forearm_L:.47,Forearm_R:.47};
function basisQuat(xv,yv){const x=xv.clone().normalize(),y=yv.clone().normalize();let z=new THREE.Vector3().crossVectors(x,y);if(z.lengthSq()<1e-8)z.set(0,0,1);z.normalize();y.crossVectors(z,x).normalize();const M=new THREE.Matrix4().makeBasis(x,y,z);return new THREE.Quaternion().setFromRotationMatrix(M)}
function segPose(a,b,axis,baseLen){if(!a||!b)return null;const d=b.clone().sub(a),L=d.length();if(L<1e-7)return null;const q=new THREE.Quaternion().setFromUnitVectors(axis,d.clone().normalize());let sc=new THREE.Vector3(1,1,1),r=L/baseLen;if(Math.abs(axis.y)>.5)sc.y=r;else sc.z=r;return {p:a,q,sc,end:b,L}}
function setPose(name,pose){const o=nodes[name];if(!o||!pose)return;o.matrixAutoUpdate=true;o.position.copy(pose.p);o.quaternion.copy(pose.q);o.scale.copy(pose.sc);o.updateMatrix();o.updateMatrixWorld(true)}
function updateRig(f){while(rigGroup.children.length){const o=rigGroup.children[0];rigGroup.remove(o);o.geometry?.dispose?.()}const lp=[];for(const [a,b] of LINKS){const A=P(f,a),B=P(f,b);if(A&&B)lp.push(...A.toArray(),...B.toArray())}const lg=new THREE.BufferGeometry();lg.setAttribute('position',new THREE.Float32BufferAttribute(lp,3));rigGroup.add(new THREE.LineSegments(lg,rigMat));const pp=[];for(const n of JOINTS){const q=P(f,n);if(q)pp.push(...q.toArray())}const pg=new THREE.BufferGeometry();pg.setAttribute('position',new THREE.Float32BufferAttribute(pp,3));rigGroup.add(new THREE.Points(pg,ptMat))}
function clearGroup(g){while(g.children.length){const o=g.children[0];g.remove(o);o.geometry?.dispose?.()}}
function updateValidation(f){
  clearGroup(modelJointGroup); clearGroup(errorGroup); if(!model)return;
  model.updateMatrixWorld(true);
  const modelPts=[], linePts=[], rows=[]; let sum2=0,maxE=0,n=0;
  for(const [joint,nodeName] of Object.entries(VALIDATE)){
    const target=P(f,joint), node=nodes[nodeName]; if(!target||!node)continue;
    const actual=node.getWorldPosition(new THREE.Vector3()); const e=actual.distanceTo(target); const pct=100*e/BODY_H;
    modelPts.push(...actual.toArray()); linePts.push(...actual.toArray(),...target.toArray()); rows.push([joint,e,pct]); sum2+=e*e;maxE=Math.max(maxE,e);n++;
  }
  if(modelPts.length){const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(modelPts,3));modelJointGroup.add(new THREE.Points(g,modelPtMat))}
  if(linePts.length){const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(linePts,3));errorGroup.add(new THREE.LineSegments(g,errorMat))}
  const rms=n?Math.sqrt(sum2/n):NaN, rmsPct=n?100*rms/BODY_H:NaN, maxPct=n?100*maxE/BODY_H:NaN;
  const top=rows.sort((a,b)=>b[1]-a[1]).slice(0,6).map(r=>`${r[0]} ${r[1].toFixed(4)} u · ${r[2].toFixed(3)}% H`).join('\n');
  $('#metrics').textContent = n ? `VALIDACIÓN FRAME ${current+1}/${F.length} · ${n} articulaciones\nRMS: ${rms.toFixed(4)} u (${rmsPct.toFixed(3)}% altura) · Máx: ${maxE.toFixed(4)} u (${maxPct.toFixed(3)}% altura)\n${top}` : 'Validación: sin pares disponibles';
  $('#metrics').className='stage '+(n&&rmsPct<0.5?'ok2':n&&rmsPct<1.5?'':'err');
}
function updateModel(f){if(!model)return;const LH=P(f,'LHip'),RH=P(f,'RHip'),H=P(f,'Hip')||avg(LH,RH),LS=P(f,'LShoulder'),RS=P(f,'RShoulder'),SM=avg(LS,RS),N=P(f,'Neck')||SM,HD=P(f,'Head')||P(f,'Nose')||N;
  // pelvis: rigid basis from hip line + trunk axis; centered exactly on hip midpoint
  if(H&&LH&&RH&&SM){const q=basisQuat(RH.clone().sub(LH),SM.clone().sub(H));const hw=LH.distanceTo(RH);setPose('Pelvis',{p:H,q,sc:new THREE.Vector3(Math.max(.35,hw/.58),1,1)})}
  setPose('Spine',segPose(H,SM,AXIS_YP,BASE_LEN.Spine));
  if(SM&&LS&&RS&&H){const q=basisQuat(RS.clone().sub(LS),SM.clone().sub(H));const sw=LS.distanceTo(RS);const tr=H.distanceTo(SM);setPose('Thorax',{p:SM,q,sc:new THREE.Vector3(Math.max(.35,sw/.72),Math.max(.35,tr/.55),1)})}
  setPose('Neck',segPose(N,HD,AXIS_YP,BASE_LEN.Neck));
  if(N&&HD){const v=HD.clone().sub(N),q=new THREE.Quaternion().setFromUnitVectors(AXIS_YP,v.lengthSq()>1e-8?v.clone().normalize():AXIS_YP);setPose('Head',{p:N,q,sc:new THREE.Vector3(1,Math.max(.5,v.length()/.18),1)})}
  for(const side of ['L','R']){
    const hp=P(f,side+'Hip'),kn=P(f,side+'Knee'),an=P(f,side+'Ankle'),to=P(f,side+'BigToe')||P(f,side+'Heel'),sh=P(f,side+'Shoulder'),el=P(f,side+'Elbow'),wr=P(f,side+'Wrist');
    setPose('Femur_'+side,segPose(hp,kn,AXIS_YN,BASE_LEN['Femur_'+side]));setPose('Tibia_'+side,segPose(kn,an,AXIS_YN,BASE_LEN['Tibia_'+side]));setPose('Foot_'+side,segPose(an,to,AXIS_ZP,BASE_LEN['Foot_'+side]));setPose('Humerus_'+side,segPose(sh,el,AXIS_YN,BASE_LEN['Humerus_'+side]));setPose('Forearm_'+side,segPose(el,wr,AXIS_YN,BASE_LEN['Forearm_'+side]));
    if(wr&&el){const q=new THREE.Quaternion().setFromUnitVectors(AXIS_YN,wr.clone().sub(el).normalize());setPose('Hand_'+side,{p:wr,q,sc:new THREE.Vector3(1,1,1)})}
  }
  model.updateMatrixWorld(true);updateValidation(f)
}
function apply(n){if(!F.length)return;current=(n+F.length)%F.length;const f=F[current];updateRig(f);updateModel(f);$('#frame').textContent=(playing?'ANIMACIÓN ACTIVA':'DETENIDA')+' · Frame '+(current+1)+' / '+F.length;$('#scrub').value=current;render()}
function setMode(m){mode=m;$('#rig').classList.toggle('active',m==='rig');$('#bones').classList.toggle('active',m==='bones');$('#both').classList.toggle('active',m==='both');rigGroup.visible=m!=='bones';modelJointGroup.visible=m==='both';errorGroup.visible=m==='both';if(model)model.visible=m!=='rig';render()}
function resize(){const r=canvas.getBoundingClientRect(),w=Math.max(2,r.width|0),h=Math.max(2,r.height|0);if(canvas.width!==w||canvas.height!==h)renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix()}
function fit(){const box=new THREE.Box3();if(rigGroup.visible)box.expandByObject(rigGroup);if(model?.visible)box.expandByObject(model);if(box.isEmpty())return;const sz=box.getSize(new THREE.Vector3()),cc=box.getCenter(new THREE.Vector3()),rad=Math.max(sz.x,sz.y,sz.z)/2||1,d=rad/Math.tan(THREE.MathUtils.degToRad(camera.fov)/2)*1.55;camera.position.set(cc.x,cc.y+sz.y*.02,cc.z+d);camera.near=Math.max(.01,d/1000);camera.far=Math.max(50,d*20);camera.updateProjectionMatrix();controls.target.copy(cc);controls.update();grid.position.y=cc.y-sz.y*.55;render()}
function render(){resize();renderer.render(scene,camera)}
async function loadModel(){try{setStage('#modelStage','Modelo V107.4: decodificando GLB local…');const bin=Uint8Array.from(atob(MODEL_B64),c=>c.charCodeAt(0));const url=URL.createObjectURL(new Blob([bin],{type:'model/gltf-binary'}));const gltf=await new GLTFLoader().loadAsync(url);URL.revokeObjectURL(url);model=gltf.scene;scene.add(model);for(const n of required)nodes[n]=model.getObjectByName(n);const found=required.filter(n=>nodes[n]).length;setStage('#modelStage','Modelo V107.4: GLB local OK','ok');setStage('#nodesStage','Nodos anatómicos: '+found+' / '+required.length+(found===required.length?' OK':''),found===required.length?'ok':'err');if(found!==required.length)throw new Error('faltan nodos del modelo');
    // V107.3: los controladores conocidos pasan a depender directamente de la raíz del GLB.
    // Se hace una sola vez; no hay clasificación espacial ni reparenting por frame.
    model.updateMatrixWorld(true);for(const n of required){const o=nodes[n];if(o.parent!==model)model.attach(o)}
    model.traverse(o=>{if(o.isMesh){o.frustumCulled=false;o.material=new THREE.MeshBasicMaterial({color:0xe8dfcf,side:THREE.DoubleSide})}});
    $('#bones').disabled=false;$('#both').disabled=false;apply(0);setMode('both');fit();setStage('#renderStage','Primer render WebGL: OK · pivotes coincidentes','ok');$('#diag').textContent='V107.4 · Validación cuantitativa\nFrames: '+F.length+'\nEscala V104→modelo: '+MAP_SCALE.toFixed(4)+'\nCian = rig V104 · magenta = pivotes reales GLB · rojo = vector de error.\nRMS y máximo se normalizan por altura corporal mapeada (2.65 u).\nLa cinemática es exactamente la de V107.3; esta versión solo añade instrumentación de validación.'}catch(e){setStage('#modelStage','Modelo V107.4: ERROR · '+e.message,'err');$('#diag').textContent='ERROR V107.4: '+e.stack}}
$('#scrub').max=Math.max(0,F.length-1);$('#play').onclick=e=>{playing=!playing;e.target.textContent=playing?'⏸ Pausa':'▶ Reproducir'};$('#prev').onclick=()=>{playing=false;apply(current-1)};$('#next').onclick=()=>{playing=false;apply(current+1)};$('#reset').onclick=()=>{playing=false;apply(0)};$('#fit').onclick=fit;$('#rig').onclick=()=>setMode('rig');$('#bones').onclick=()=>setMode('bones');$('#both').onclick=()=>setMode('both');$('#scrub').oninput=e=>{playing=false;apply(+e.target.value)};
function mime(){for(const m of ['video/mp4;codecs=h264','video/mp4','video/webm;codecs=vp9','video/webm'])if(window.MediaRecorder&&MediaRecorder.isTypeSupported(m))return m;return ''}
$('#record').onclick=()=>{if(!canvas.captureStream||!window.MediaRecorder)return;const m=mime();try{recorder=new MediaRecorder(canvas.captureStream(24),m?{mimeType:m}:undefined)}catch(e){$('#diag').textContent+='\nGrabación ERROR: '+e;return}chunks=[];recording=true;$('#download').innerHTML='';recorder.ondataavailable=e=>{if(e.data?.size)chunks.push(e.data)};recorder.onstop=()=>{recording=false;const type=recorder.mimeType||m||'video/webm',ext=type.includes('mp4')?'mp4':'webm',blob=new Blob(chunks,{type}),url=URL.createObjectURL(blob),a=document.createElement('a');a.className='dl';a.href=url;a.download='PhysioSentinel_V107_4_'+mode+'.'+ext;a.textContent='⬇ Descargar vídeo '+mode+' ('+ext.toUpperCase()+')';$('#download').appendChild(a)};apply(0);recorder.start(250);playing=true};
function loop(t){if(playing&&t-last>FRAME_MS){last=t;const old=current;apply(current+1);if(recording&&old===F.length-1&&current===0){playing=false;try{recorder.stop()}catch(e){}}}requestAnimationFrame(loop)}
window.addEventListener('resize',render);window.addEventListener('error',e=>setStage('#renderStage','JavaScript ERROR · '+e.message,'err'));window.addEventListener('unhandledrejection',e=>setStage('#renderStage','Promise ERROR · '+String(e.reason),'err'));
setStage('#renderStage','WebGL inicializado','ok');apply(0);loadModel();requestAnimationFrame(loop);
</script></body></html>'''
    return tpl.replace('__SAFE__',safe_js).replace('__MOTION__',motion_json).replace('__MODEL__',b64)
