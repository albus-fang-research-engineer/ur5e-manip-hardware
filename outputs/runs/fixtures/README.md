# Symbol fixtures for `check_symbols.py` (milestone 1)

Hand-authored `frames.json` files used to validate the checker itself before
it judges grounding output. Two files per object, committed once authored so
the acceptance run is reproducible:

    mug_frames_hand.json     positive control: correct symbols
    mug_frames_wrong.json    negative control: same file with handle_center
                             displaced +3 cm (must fail at that symbol, and
                             only there)

## Authoring rules

- Coordinates are in the **body frame of the tracked mesh** -- the file
  `summary["objects"][<key>]["pose_on_final"]["mesh"]` points to (Any6D's
  `final_mesh_<obj>.obj`), under the identity transform. Not the canonical
  GLB, not a sim asset: one mesh, one body frame. TRELLIS canonical frames
  are NOT axis-aligned (the mug's is tilted ~25 deg), so never author from
  an assumed axis; load the mesh and read coordinates off it:

      python3 -c "import trimesh; m = trimesh.load('final_mesh_mug.obj'); \
                  print(m.bounds, m.centroid)"   # then pick in a viewer

- Schema is the sim's `frames.json` (`points/axes/quantities`, each entry
  `{xyz|value, status, comment}`) plus a per-point `depth_check` field:

      "surface"    on the mesh surface (rim points)        [default]
      "interior"   centerline/interior point (handle_center: a correct one
                   reads ~ +half the local thickness on both residuals)
      "skip"       free-space symbol (opening_center: the ray hits the far
                   inner wall ~8 cm behind -- residuals are meaningless;
                   drawn and in-mask-gated only)

- Put the **ruler numbers in `comment`** (real handle reach / body diameter /
  height): the fixture then carries the mesh-fidelity ground truth alongside
  the mesh-frame coordinates, which is what disambiguates a `diff` failure
  (fidelity/registration) from a `vs_mesh` failure (grounding) at the bench.

## Example shape

    {
     "object": "mug",
     "coordinates": "body frame of final_mesh_mug.obj (identity transform)",
     "points": {
      "handle_center": {"xyz": [x, y, z], "depth_check": "interior",
                        "status": "hand", "comment": "ruler: handle reach NN mm"},
      "opening_center": {"xyz": [x, y, z], "depth_check": "skip", "status": "hand"},
      "rim_front":      {"xyz": [x, y, z], "status": "hand"}
     },
     "axes":      {"up_axis": {"xyz": [x, y, z], "status": "hand"}},
     "quantities": {"rim_radius": {"value": r, "status": "hand",
                                   "comment": "ruler: NN mm"}}
    }
