# PhysioSentinel Gait · Version 159

## Denver anatomical registration hardening

- Corrige un fallo critico del payload compacto: frame 0 aplicaba una segunda rotacion al mezclar `_semantic_target_frame()` con `_segment_frame_from_joints()`. V159 usa el mismo marco semantico en origen y destino; frame 0 queda identidad numerica.
- Pelvis Denver registrada sobre el punto medio de los centros de cadera SKEL, no sobre el joint abstracto `pelvis`, preservando acetabulo-cabeza femoral.
- Marcos pelvis ortonormales y diestros; escalado uniforme (sin aplanamiento anisotropico).
- Tibia/perone: Denver usa el centro real fibula-tibia para definir lateralidad; SKEL usa la direccion ipsilateral alejandose de la cadera contralateral.
- Femur/rotula: patella Denver define anterior y se alinea con anterior anatomico estimado.
- Pie: cuneiforme medial/lateral Denver define medialidad; target medial apunta a linea media.
- Sacro/coccix se mantienen posteriores.
- Se mantienen QC distal, contacto/swing y banderas de fiabilidad de V158.

## Validacion previa al empaquetado

75/75 frames del NPZ real. PASS: marcos diestros, frame0 identidad, rotulas anteriores bilateralmente, perones laterales bilateralmente, cuneiformes mediales, sacro posterior, centros de cabeza femoral anclados a cadera, continuidad temporal <10 grados/frame. Distancia cabeza femoral-acetabulo durante 75 frames: max ~0.023 D y ~0.028 I en unidades SKEL no metricas. Todos los anchors articulares evaluados permanecen dentro del bounding box de la piel SKEL.
