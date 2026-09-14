import json

ATLAS_PRIMARY = "https://raw.githubusercontent.com/JohanBellander/BodyExplorer/main/public/skeleton.glb"
ATLAS_FALLBACK = "https://cdn.jsdelivr.net/gh/JohanBellander/BodyExplorer@main/public/skeleton.glb"

def anatomical_atlas_html(motion, safe_gpu=True):
    motion_json = json.dumps(motion, ensure_ascii=False, separators=(",", ":"))
    safe_js = "true" if safe_gpu else "false"
    tpl = r'''<!doctype html><html><head><meta charset="utf-8"><style>
html,body{margin:0;background:#07111d;color:#dce8f5;font:13px system-ui;height:100%;overflow:hidden}.wrap{display:grid;grid-template-columns:370px 1fr;height:720px}.hud{padding:14px;background:#0a1420;border-right:1px solid #1d3347;overflow:auto}.view{position:relative;min-width:0}.view canvas{position:absolute;inset:0;width:100%;height:100%;display:block}.ok{color:#62e6a7;font-weight:800}.safe{color:#ffcf6e;font-weight:700}.stage{margin:6px 0;padding:6px 8px;border-radius:6px;background:#0c1b2a}.stage.ok2{color:#6ee7a8}.stage.err{color:#ff8c8c}.frame{font-weight:800;margin:8px 0}.controls{display:flex;flex-wrap:wrap;gap:6px;margin:9px 0}button{padding:7px 10px;border:0;border-radius:7px;cursor:pointer}button:disabled{opacity:.38}.active{outline:2px solid #63d9ff}.diag{white-space:pre-wrap;color:#9fc4e8;line-height:1.35}.warn{color:#ffca74;margin-top:10px}.dl{display:inline-block;padding:7px 10px;border-radius:7px;background:#dce8f5;color:#07111d;text-decoration:none;font-weight:700;margin-top:6px}input{width:100%}</style></head><body><div class="wrap"><div class="hud">
<div class="ok">V108.3 · ATLAS ÓSEO REAL · MAPA ANATÓMICO POR IDENTIDAD · 201 MALLAS</div><div class="safe">🛡️ GPU SAFE · binding anatómico una sola vez</div>
<div id="rigStage" class="stage">Rig: iniciando…</div><div id="engineStage" class="stage">Three.js: pendiente</div><div id="loaderStage" class="stage">GLTFLoader: pendiente</div><div id="atlasStage" class="stage">skeleton.glb: pendiente</div><div id="bindStage" class="stage">Binding anatómico: pendiente</div><div id="renderStage" class="stage">Primer render: pendiente</div>
<div class="frame" id="frame">Frame —</div><div class="controls"><button id="play">▶ Reproducir</button><button id="prev">−1</button><button id="next">+1</button><button id="reset">Reset</button><button id="fit">Reencuadrar</button><button id="record">⏺ Grabar ciclo</button></div>
<div class="controls"><button id="rig">Rig</button><button id="bones" disabled>Atlas real</button><button id="both" disabled class="active">Rig + Atlas</button><button id="map" disabled>Mapa binding</button></div><div id="download"></div><input id="scrub" type="range" min="0" max="0" value="0"><div id="metrics" class="stage">Atlas: pendiente</div><div id="diag" class="diag">Preparando V108.3…</div><div class="warn">V108.3 mantiene el registro global V108.3 y sustituye la clasificación puramente espacial por un mapa anatómico por identidad: nombre/metadata de cada malla cuando está disponible, con fallback espacial conservador. El mapa se congela una sola vez durante los 75 frames.</div></div>
<div class="view"><canvas id="gl"></canvas></div></div>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.180.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.180.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from 'three'; import {OrbitControls} from 'three/addons/controls/OrbitControls.js'; import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
const SAFE=__SAFE__, motion=__MOTION__, F=motion.frames||[], PRIMARY='__PRIMARY__', FALLBACK='__FALLBACK__'; const $=s=>document.querySelector(s),canvas=$('#gl');
const setStage=(id,text,kind='')=>{const e=$(id);e.textContent=text;e.className='stage '+(kind==='ok'?'ok2':kind==='err'?'err':'')};
setStage('#rigStage','Rig V104/V107: OK · '+F.length+' frames','ok');setStage('#engineStage','Three.js: OK','ok');setStage('#loaderStage','GLTFLoader + OrbitControls: OK','ok');
const renderer=new THREE.WebGLRenderer({canvas,antialias:false,powerPreference:'low-power',preserveDrawingBuffer:false});renderer.setPixelRatio(SAFE?1:Math.min(devicePixelRatio,1.5));renderer.setClearColor(0x07111d,1);renderer.outputColorSpace=THREE.SRGBColorSpace;
const scene=new THREE.Scene();scene.background=new THREE.Color(0x07111d);const camera=new THREE.PerspectiveCamera(32,1,.01,100);const controls=new OrbitControls(camera,canvas);controls.enableDamping=false;scene.add(new THREE.AmbientLight(0xffffff,1.6));const dl=new THREE.DirectionalLight(0xffffff,1.4);dl.position.set(3,5,4);scene.add(dl);const grid=new THREE.GridHelper(4,16,0x2a4a67,0x173047);scene.add(grid);
const rigGroup=new THREE.Group();scene.add(rigGroup);const rigMat=new THREE.LineBasicMaterial({color:0x00e5ff});const ptMat=new THREE.PointsMaterial({color:0xffd166,size:.035,sizeAttenuation:true});
const JOINTS=['Head','Neck','LShoulder','RShoulder','LElbow','RElbow','LWrist','RWrist','Hip','LHip','RHip','LKnee','RKnee','LAnkle','RAnkle','LHeel','RHeel','LBigToe','RBigToe'];
const LINKS=[['Neck','LShoulder'],['LShoulder','LElbow'],['LElbow','LWrist'],['Neck','RShoulder'],['RShoulder','RElbow'],['RElbow','RWrist'],['Neck','Hip'],['LHip','RHip'],['Hip','LHip'],['Hip','RHip'],['LHip','LKnee'],['LKnee','LAnkle'],['LAnkle','LBigToe'],['RHip','RKnee'],['RKnee','RAnkle'],['RAnkle','RBigToe']];
function rawP(f,n){const p=f?.joints?.[n]||f?.xyz?.[n]||f?.points?.[n];if(!p)return null;const a=Array.isArray(p)?p:[p.x,p.y,p.z];return a.length>=3&&a.every(Number.isFinite)?new THREE.Vector3(+a[0],+a[1],+a[2]):null}
function avg(...vs){const a=vs.filter(Boolean);if(!a.length)return null;return a.reduce((s,v)=>s.add(v),new THREE.Vector3()).multiplyScalar(1/a.length)}
const f0=F[0]||{},hip0=avg(rawP(f0,'LHip'),rawP(f0,'RHip'),rawP(f0,'Hip'))||new THREE.Vector3();let ys=[];for(const n of JOINTS){const p=rawP(f0,n);if(p)ys.push(p.y)}const bodyRange=Math.max(.5,(ys.length?(Math.max(...ys)-Math.min(...ys)):1)),MAP_SCALE=2.65/bodyRange;function MAPP(p){return p?p.clone().sub(hip0).multiplyScalar(MAP_SCALE):null}function P(f,n){return MAPP(rawP(f,n))}
let atlas=null,current=0,playing=false,last=0,mode='both',recorder=null,recording=false,chunks=[],bindings=[];const FRAME_MS=1000/24;
function updateRig(f){while(rigGroup.children.length){const o=rigGroup.children[0];rigGroup.remove(o);o.geometry?.dispose?.()}const lp=[];for(const [a,b] of LINKS){const A=P(f,a),B=P(f,b);if(A&&B)lp.push(...A.toArray(),...B.toArray())}const lg=new THREE.BufferGeometry();lg.setAttribute('position',new THREE.Float32BufferAttribute(lp,3));rigGroup.add(new THREE.LineSegments(lg,rigMat));const pp=[];for(const n of JOINTS){const q=P(f,n);if(q)pp.push(...q.toArray())}const pg=new THREE.BufferGeometry();pg.setAttribute('position',new THREE.Float32BufferAttribute(pp,3));rigGroup.add(new THREE.Points(pg,ptMat))}
function basisMatrix(origin,xv,yv){const x=xv.clone().normalize(),y=yv.clone().normalize();let z=new THREE.Vector3().crossVectors(x,y);if(z.lengthSq()<1e-8)z.set(0,0,1);z.normalize();y.crossVectors(z,x).normalize();const m=new THREE.Matrix4().makeBasis(x,y,z);m.setPosition(origin);return m}
function segMatrix(a,b,up=new THREE.Vector3(0,-1,0)){if(!a||!b)return new THREE.Matrix4();const d=b.clone().sub(a);if(d.lengthSq()<1e-9)return new THREE.Matrix4().makeTranslation(a.x,a.y,a.z);const q=new THREE.Quaternion().setFromUnitVectors(up,d.clone().normalize());return new THREE.Matrix4().compose(a,q,new THREE.Vector3(1,1,1))}
function segmentFrames(f){const LH=P(f,'LHip'),RH=P(f,'RHip'),H=P(f,'Hip')||avg(LH,RH),LS=P(f,'LShoulder'),RS=P(f,'RShoulder'),SM=avg(LS,RS),N=P(f,'Neck')||SM,HD=P(f,'Head')||P(f,'Nose')||N;const out={};if(H&&LH&&RH&&SM)out.Pelvis=basisMatrix(H,RH.clone().sub(LH),SM.clone().sub(H));if(H&&SM)out.Trunk=segMatrix(H,SM,new THREE.Vector3(0,1,0));if(SM&&LS&&RS&&H)out.Thorax=basisMatrix(SM,RS.clone().sub(LS),SM.clone().sub(H));if(N&&HD)out.Head=segMatrix(N,HD,new THREE.Vector3(0,1,0));for(const s of ['L','R']){const hp=P(f,s+'Hip'),kn=P(f,s+'Knee'),an=P(f,s+'Ankle'),to=P(f,s+'BigToe')||P(f,s+'Heel'),sh=P(f,s+'Shoulder'),el=P(f,s+'Elbow'),wr=P(f,s+'Wrist');out['Femur_'+s]=segMatrix(hp,kn);out['Tibia_'+s]=segMatrix(kn,an);out['Foot_'+s]=segMatrix(an,to,new THREE.Vector3(0,0,1));out['Humerus_'+s]=segMatrix(sh,el);out['Forearm_'+s]=segMatrix(el,wr);out['Hand_'+s]=segMatrix(wr,wr&&el?wr.clone().add(wr.clone().sub(el).normalize().multiplyScalar(.18)):wr)}return out}
const REST=segmentFrames(f0);
function pointSegDist(p,a,b){const ab=b.clone().sub(a),d=ab.lengthSq();if(d<1e-9)return p.distanceTo(a);const t=THREE.MathUtils.clamp(p.clone().sub(a).dot(ab)/d,0,1);return p.distanceTo(a.clone().addScaledVector(ab,t))}
function segDefs(f){const defs=[];function add(name,a,b){a=P(f,a);b=P(f,b);if(a&&b)defs.push({name,a,b})}add('Femur_L','LHip','LKnee');add('Tibia_L','LKnee','LAnkle');add('Foot_L','LAnkle','LBigToe');add('Femur_R','RHip','RKnee');add('Tibia_R','RKnee','RAnkle');add('Foot_R','RAnkle','RBigToe');add('Humerus_L','LShoulder','LElbow');add('Forearm_L','LElbow','LWrist');add('Humerus_R','RShoulder','RElbow');add('Forearm_R','RElbow','RWrist');add('Trunk','Hip','Neck');add('Head','Neck','Head');return defs}
function nearestSeg(p,defs){let best=null,bd=1e9;for(const d of defs){let v=pointSegDist(p,d.a,d.b);if(v<bd){bd=v;best=d.name}}return best||'Trunk'}
function rigRestBox(){const b=new THREE.Box3();for(const n of JOINTS){const p=P(f0,n);if(p)b.expandByPoint(p)}return b}
function fitAtlasToRig(obj){
  // V108.3: registro global determinista por envolvente corporal completa.
  // Mantiene escala uniforme (sin deformar la anatomía) y alinea centro X/Z + suelo Y.
  obj.updateMatrixWorld(true);
  const ab0=new THREE.Box3().setFromObject(obj), as0=ab0.getSize(new THREE.Vector3());
  const rb=rigRestBox(), rs=rb.getSize(new THREE.Vector3()), rc=rb.getCenter(new THREE.Vector3());
  const scale=Math.max(.0001,rs.y)/Math.max(as0.y,1e-6);
  obj.scale.setScalar(scale);obj.updateMatrixWorld(true);
  const ab=new THREE.Box3().setFromObject(obj), ac=ab.getCenter(new THREE.Vector3());
  // suelo con suelo; centro mediolateral y profundidad con centro del rig.
  obj.position.x += rc.x-ac.x;
  obj.position.z += rc.z-ac.z;
  obj.position.y += rb.min.y-ab.min.y;
  obj.updateMatrixWorld(true);
  const out=new THREE.Box3().setFromObject(obj), os=out.getSize(new THREE.Vector3());
  return {rigBox:rb,atlasBox:out,scale,rigSize:rs,atlasSize:os};
}
const META_PRIMARY='https://raw.githubusercontent.com/JohanBellander/BodyExplorer/main/public/mesh_mapping.json';
const META_FALLBACK='https://cdn.jsdelivr.net/gh/JohanBellander/BodyExplorer@main/public/mesh_mapping.json';
let meshMeta=null,metaSource='sin metadata';
function normText(v){return String(v||'').toLowerCase().normalize('NFD').replace(/[\u0300-\u036f]/g,'').replace(/[_\-.]+/g,' ')}
function sideFromText(t){
  t=' '+normText(t)+' ';
  if(/\b(left|izq|sinister|sinistra| l )\b/.test(t)||/(^|[ _.-])l($|[ _.-])/.test(String(t)))return 'L';
  if(/\b(right|der|dexter|dextra| r )\b/.test(t)||/(^|[ _.-])r($|[ _.-])/.test(String(t)))return 'R';
  return null;
}
function flattenMeta(obj,prefix='',out={}){
  if(!obj||typeof obj!=='object')return out;
  if(Array.isArray(obj)){obj.forEach((v,i)=>flattenMeta(v,prefix?prefix+':'+i:String(i),out));return out}
  for(const [k,v] of Object.entries(obj)){
    const key=String(k), path=prefix?prefix+':'+key:key;
    if(v&&typeof v==='object'){
      const blob=JSON.stringify(v);
      out[key]=blob; out[path]=blob;
      const ids=['mesh','mesh_name','name','id','bp3d_id','object','object_name','source_id'];
      for(const id of ids){if(v[id]!=null)out[String(v[id])]=blob}
      flattenMeta(v,path,out);
    }
  }
  return out;
}
async function loadMeshMetadata(){
  for(const u of [META_PRIMARY,META_FALLBACK]){
    try{const r=await fetch(u,{cache:'force-cache'});if(!r.ok)throw new Error('HTTP '+r.status);const j=await r.json();meshMeta=flattenMeta(j);metaSource=u;return true}catch(e){console.warn('metadata',u,e)}
  }
  meshMeta={};return false;
}
function metadataText(mesh){
  const keys=[mesh.name,mesh.parent&&mesh.parent.name,mesh.userData&&mesh.userData.name,mesh.userData&&mesh.userData.id].filter(Boolean);
  let out=keys.join(' ');
  if(meshMeta){for(const k of keys){if(meshMeta[k])out+=' '+meshMeta[k]}}
  try{out+=' '+JSON.stringify(mesh.userData||{})}catch(e){}
  return normText(out);
}
function identitySegment(mesh,atlasBox){
  const t=metadataText(mesh);
  const mb=new THREE.Box3().setFromObject(mesh), c=mb.getCenter(new THREE.Vector3());
  const ctr=atlasBox.getCenter(new THREE.Vector3());
  const LH=P(f0,'LHip'), RH=P(f0,'RHip');
  const lowXIsLeft=(LH&&RH)?(LH.x<RH.x):true;
  const spatialSide=((c.x<ctr.x)===lowXIsLeft)?'L':'R';
  const side=sideFromText(t)||spatialSide;
  // Identidades anatómicas fuertes. La cintura escapular queda axial/tronco.
  if(/\b(femur|femoral)\b/.test(t)) return 'Femur_'+side;
  if(/\b(patella|patellar)\b/.test(t)) return 'Femur_'+side;
  if(/\b(tibia|fibula|fibular|tibial)\b/.test(t)) return 'Tibia_'+side;
  if(/\b(talus|calcaneus|calcaneum|navicular|cuboid|cuneiform|metatars|phalange.*toe|toe phalan|foot)\b/.test(t)) return 'Foot_'+side;
  if(/\b(humerus|humeral)\b/.test(t)) return 'Humerus_'+side;
  if(/\b(radius|radial|ulna|ulnar)\b/.test(t)) return 'Forearm_'+side;
  if(/\b(carpal|metacarp|phalange.*hand|finger|hand)\b/.test(t)) return 'Hand_'+side;
  if(/\b(ilium|iliac|ischium|ischial|pubis|pubic|innominate|coxal|sacrum|sacral|coccyx|coccygeal|pelvis)\b/.test(t)) return 'Pelvis';
  if(/\b(skull|cranium|cranial|mandible|mandibular|maxilla|zygomatic|nasal|frontal bone|parietal|temporal bone|occipital|sphenoid|ethmoid|hyoid|head)\b/.test(t)) return 'Head';
  if(/\b(vertebra|vertebral|spine|spinal|rib|costal|sternum|sternal|clavicle|clavicular|scapula|scapular|thorax|trunk)\b/.test(t)) return 'Trunk';
  return null;
}
function spatialFallback(mesh,atlasBox){
  const mb=new THREE.Box3().setFromObject(mesh), c=mb.getCenter(new THREE.Vector3());
  const s=atlasBox.getSize(new THREE.Vector3()), ctr=atlasBox.getCenter(new THREE.Vector3());
  const fy=THREE.MathUtils.clamp((c.y-atlasBox.min.y)/Math.max(s.y,1e-6),0,1);
  const ax=Math.abs(c.x-ctr.x)/Math.max(s.x,1e-6);
  const LH=P(f0,'LHip'), RH=P(f0,'RHip'); const lowXIsLeft=(LH&&RH)?(LH.x<RH.x):true;
  const side=(((c.x<ctr.x)===lowXIsLeft)?'L':'R');
  if(fy>=.84) return 'Head';
  if(fy<.49){if(fy<.12)return 'Foot_'+side;if(fy<.31)return 'Tibia_'+side;return 'Femur_'+side}
  if(fy<.59&&ax<.23)return 'Pelvis';
  // Fallback superior deliberadamente conservador: sólo zonas realmente laterales se asignan a brazo.
  if(ax>.32){if(fy<.50)return 'Hand_'+side;if(fy<.66)return 'Forearm_'+side;if(fy<.82)return 'Humerus_'+side}
  return 'Trunk';
}
function atlasRegionClassifier(mesh,atlasBox,rigBox){
  const id=identitySegment(mesh,atlasBox);
  return {seg:id||spatialFallback(mesh,atlasBox),method:id?'IDENTIDAD':'ESPACIAL'};
}
const SEG_COLORS={Pelvis:0xf59e0b,Trunk:0x60a5fa,Head:0xe879f9,Femur_L:0x34d399,Femur_R:0x10b981,Tibia_L:0x22d3ee,Tibia_R:0x06b6d4,Foot_L:0xa78bfa,Foot_R:0x8b5cf6,Humerus_L:0xfb7185,Humerus_R:0xf43f5e,Forearm_L:0xfda4af,Forearm_R:0xe11d48,Hand_L:0xfbbf24,Hand_R:0xf59e0b};
let registrationInfo=null;
function bindAtlas(){
  bindings=[];const counts={},methodCounts={IDENTIDAD:0,ESPACIAL:0};atlas.updateMatrixWorld(true);
  const meshes=[];atlas.traverse(o=>{if(o.isMesh)meshes.push(o)});
  const atlasBox=new THREE.Box3().setFromObject(atlas),rigBox=rigRestBox();
  for(const mesh of meshes){
    const cls=atlasRegionClassifier(mesh,atlasBox,rigBox),seg=cls.seg;counts[seg]=(counts[seg]||0)+1;methodCounts[cls.method]=(methodCounts[cls.method]||0)+1;
    const world=mesh.matrixWorld.clone();scene.attach(mesh);mesh.matrixAutoUpdate=false;mesh.matrix.copy(world);mesh.matrixWorld.copy(world);
    const mat=new THREE.MeshLambertMaterial({color:0xe8dfcf,side:THREE.DoubleSide});
    mesh.material=mat;mesh.frustumCulled=false;
    bindings.push({mesh,seg,restWorld:world.clone(),restSeg:(REST[seg]||new THREE.Matrix4()).clone(),baseMaterial:mat,mapColor:SEG_COLORS[seg]||0xcbd5e1});
  }
  atlas.visible=false;
  const sorted=Object.entries(counts).sort((a,b)=>a[0].localeCompare(b[0]));
  const maxEntry=sorted.reduce((m,e)=>e[1]>m[1]?e:m,['',0]);
  const sanity=maxEntry[1]<=70?'OK':'REVISAR';
  $('#metrics').textContent='Atlas real: '+meshes.length+' mallas · mapa V108.3 congelado\n'+sorted.map(([k,v])=>k+': '+v).join(' · ')+'\nIdentidad/metadata: '+methodCounts.IDENTIDAD+' · fallback espacial: '+methodCounts.ESPACIAL+'\nControl distribución: '+sanity+' · grupo máx. '+maxEntry[0]+'='+maxEntry[1];
  setStage('#bindStage','Binding anatómico V108.3: OK · '+bindings.length+' mallas · '+sanity,'ok');
}
function updateAtlas(f){if(!bindings.length)return;const cur=segmentFrames(f);for(const b of bindings){const C=cur[b.seg]||new THREE.Matrix4(),R=b.restSeg.clone();const delta=C.clone().multiply(R.invert());b.mesh.matrix.copy(delta.multiply(b.restWorld));b.mesh.matrixWorldNeedsUpdate=true}}
function apply(n){if(!F.length)return;current=(n+F.length)%F.length;const f=F[current];updateRig(f);updateAtlas(f);$('#frame').textContent=(playing?'ANIMACIÓN ACTIVA':'DETENIDA')+' · Frame '+(current+1)+' / '+F.length;$('#scrub').value=current;render()}
function setMode(m){mode=m;for(const id of ['rig','bones','both','map'])$('#'+id).classList.toggle('active',m===id);rigGroup.visible=(m==='rig'||m==='both'||m==='map');for(const b of bindings){b.mesh.visible=m!=='rig';if(b.mesh.material){if(m==='map'){b.mesh.material.color.setHex(b.mapColor);b.mesh.material.transparent=true;b.mesh.material.opacity=.82}else{b.mesh.material.color.setHex(0xe8dfcf);b.mesh.material.transparent=(m==='both');b.mesh.material.opacity=(m==='both'?.72:1)}}}render()}
function resize(){const r=canvas.getBoundingClientRect(),w=Math.max(2,r.width|0),h=Math.max(2,r.height|0);if(canvas.width!==w||canvas.height!==h)renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix()}function render(){resize();renderer.render(scene,camera)}
function fit(){const box=new THREE.Box3();if(rigGroup.visible)box.expandByObject(rigGroup);for(const b of bindings)if(b.mesh.visible)box.expandByObject(b.mesh);if(box.isEmpty())return;const sz=box.getSize(new THREE.Vector3()),cc=box.getCenter(new THREE.Vector3()),rad=Math.max(sz.x,sz.y,sz.z)/2||1,d=rad/Math.tan(THREE.MathUtils.degToRad(camera.fov)/2)*1.55;camera.position.set(cc.x,cc.y,cc.z+d);camera.near=Math.max(.01,d/1000);camera.far=Math.max(50,d*20);camera.updateProjectionMatrix();controls.target.copy(cc);controls.update();grid.position.y=box.min.y-.03;render()}
async function loadURL(url,timeout=25000){return await Promise.race([new GLTFLoader().loadAsync(url),new Promise((_,rej)=>setTimeout(()=>rej(new Error('timeout '+url)),timeout))])}
async function loadAtlas(){let gltf=null,used='';for(const u of [PRIMARY,FALLBACK]){try{setStage('#atlasStage','skeleton.glb: cargando '+(u===PRIMARY?'GitHub Raw':'jsDelivr')+'…');gltf=await loadURL(u);used=u;break}catch(e){$('#diag').textContent+='\nFallo '+u+': '+e.message}}if(!gltf){setStage('#atlasStage','skeleton.glb: ERROR en ambas fuentes','err');return}atlas=gltf.scene;scene.add(atlas);registrationInfo=fitAtlasToRig(atlas);await loadMeshMetadata();let mc=0;atlas.traverse(o=>{if(o.isMesh)mc++});setStage('#atlasStage','skeleton.glb real: OK · '+mc+' mallas','ok');bindAtlas();$('#bones').disabled=false;$('#both').disabled=false;$('#map').disabled=false;apply(0);setMode('both');fit();setStage('#renderStage','Primer render WebGL: OK · atlas V108.3 registrado','ok');const ri=registrationInfo;$('#diag').textContent='V108.3 · BodyExplorer/Z-Anatomy skeleton.glb\nFuente: '+used+'\nMallas detectadas: '+mc+'\nFrames: '+F.length+'\nEscala V104→visor: '+MAP_SCALE.toFixed(4)+'\nRegistro atlas→rig (uniforme): '+(ri?ri.scale.toFixed(5):'—')+'\nAltura rig: '+(ri?ri.rigSize.y.toFixed(3):'—')+' · altura atlas: '+(ri?ri.atlasSize.y.toFixed(3):'—')+'\nMetadata: '+metaSource+'\nBinding: identidad/nombre/metadata anatómica; fallback espacial conservador. Mapa congelado una sola vez.\nCian/amarillo = rig validado. Marfil = atlas. Mapa binding = color por controlador.'}
$('#scrub').max=Math.max(0,F.length-1);$('#play').onclick=e=>{playing=!playing;e.target.textContent=playing?'⏸ Pausa':'▶ Reproducir'};$('#prev').onclick=()=>{playing=false;apply(current-1)};$('#next').onclick=()=>{playing=false;apply(current+1)};$('#reset').onclick=()=>{playing=false;apply(0)};$('#fit').onclick=fit;$('#rig').onclick=()=>setMode('rig');$('#bones').onclick=()=>setMode('bones');$('#both').onclick=()=>setMode('both');$('#map').onclick=()=>setMode('map');$('#scrub').oninput=e=>{playing=false;apply(+e.target.value)};
function mime(){for(const m of ['video/mp4;codecs=h264','video/mp4','video/webm;codecs=vp9','video/webm'])if(window.MediaRecorder&&MediaRecorder.isTypeSupported(m))return m;return ''}
$('#record').onclick=()=>{if(!canvas.captureStream||!window.MediaRecorder)return;const m=mime();try{recorder=new MediaRecorder(canvas.captureStream(24),m?{mimeType:m}:undefined)}catch(e){return}chunks=[];recording=true;$('#download').innerHTML='';recorder.ondataavailable=e=>{if(e.data?.size)chunks.push(e.data)};recorder.onstop=()=>{recording=false;const type=recorder.mimeType||m||'video/webm',ext=type.includes('mp4')?'mp4':'webm',blob=new Blob(chunks,{type}),url=URL.createObjectURL(blob),a=document.createElement('a');a.className='dl';a.href=url;a.download='PhysioSentinel_V108_2_'+mode+'.'+ext;a.textContent='⬇ Descargar vídeo '+mode+' ('+ext.toUpperCase()+')';$('#download').appendChild(a)};apply(0);recorder.start(250);playing=true};
function loop(t){if(playing&&t-last>FRAME_MS){last=t;const old=current;apply(current+1);if(recording&&old===F.length-1&&current===0){playing=false;try{recorder.stop()}catch(e){}}}requestAnimationFrame(loop)}window.addEventListener('resize',render);window.addEventListener('error',e=>setStage('#renderStage','JavaScript ERROR · '+e.message,'err'));window.addEventListener('unhandledrejection',e=>setStage('#renderStage','Promise ERROR · '+String(e.reason),'err'));setStage('#renderStage','WebGL inicializado','ok');apply(0);loadAtlas();requestAnimationFrame(loop);
</script></body></html>'''
    return (tpl.replace('__SAFE__',safe_js).replace('__MOTION__',motion_json)
            .replace('__PRIMARY__',ATLAS_PRIMARY).replace('__FALLBACK__',ATLAS_FALLBACK))
