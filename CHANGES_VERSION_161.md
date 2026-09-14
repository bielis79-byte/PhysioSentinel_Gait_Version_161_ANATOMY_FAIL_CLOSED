# PhysioSentinel Gait · Version 161

## Registro anatómico adaptativo y seguro

- La orientación de la pelvis deja de corregirse con un giro fijo.
- El control combina la paridad de `axis_map` con la posición del sacro/cóccix.
- El sacro debe quedar posterior antes de permitir el render anatómico.
- Se prueban las asignaciones Denver D/I y I/D para los pies.
- Sólo se acepta la asignación que deja el proxy del hallux bilateralmente medial.
- El offset plantar se calcula por lado y se compone como calibración fija con
  los 75 frames, preservando la variación temporal de tobillo.
- La neutralización se aplica también a la piel SKEL únicamente en el visor,
  con transición suave y sin modificar el fitting, `poses`, `q(t)` ni el NPZ.
- Si el offset supera 80 grados, difiere más de 25 grados entre lados, falta el
  sacro o no existe una asignación bilateral con hallux medial, el montaje se
  bloquea y muestra el motivo.

## Validación cruzada

- NPZ histórico de control (`det(axis_map)=+1`): pelvis no invertida; pies D/D
  e I/I; QC superado.
- `alejandro.npz` (`det(axis_map)=-1`): pelvis AP corregida; pies Denver
  intercambiados; QC superado.
- Fémur, rótula, tibia y peroné mantienen su cadena cinemática y no reciben la
  corrección local de pelvis/pie.

Véase `VALIDACION_ANATOMICA_V161.json`.

