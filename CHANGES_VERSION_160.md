# PhysioSentinel Gait · Version 160

## Denver anatomical registration V160

- Replaced side-dependent medial/lateral frames with proper anatomical frames using side-independent RIGHT/ANTERIOR/UP semantics (determinant +1).
- Feet now use a no-reflection 3-landmark similarity at frame 0: Denver ankle + calcaneus + toe -> SKEL talus + calcn + toes.
- This removes the previous 180-degree / apparent R-L foot inversion failure mode while preserving Denver right and left anatomy.
- Greater trochanter QC added from the proximal femur geometry: trochanter proxy must remain lateral to the femoral head/shaft for both sides.
- Fibula QC is evaluated in the local shank frame: fibula must remain lateral to tibia through all 75 frames.
- Patella anterior and medial-foot (medial cuneiform proxy for hallux side) checks retained.
- All segment frames are proper rotations with no reflection and temporal continuity is checked over all 75 frames.
- Denver bone cache bumped to v160_sideproper to prevent reuse of V159 geometry.

## Validation on the historical 75-frame NPZ

Source: V110_3_19_11_SKEL_mesh_sequence (18).npz

- 75/75-frame anatomical QC: PASS.
- Patella anterior: bilateral PASS.
- Greater trochanter lateral: bilateral PASS.
- Fibula lateral in local shank frame: bilateral PASS.
- Medial foot/hallux-side proxy inward: bilateral PASS.
- Foot similarity transform determinant: +1 bilateral (no reflection).
- Foot landmark frame-0 RMSE: ~0.0235 right / ~0.0238 left in SKEL non-metric units.
- Maximum frame-to-frame segment rotation: 6.31 degrees (left foot), with no 180-degree jumps.

See `VALIDACION_ANATOMICA_75_FRAMES_V160.json`.
