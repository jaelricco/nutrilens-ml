# Pour Data Collection Protocol

One protocol for every session. Deviations are OK if documented; undocumented deviations are the #1 cause of garbage-in/garbage-out bugs downstream.

## Rig

- Phone on a tripod, camera at fixed height (measured, logged).
- Plain, matte, non-reflective background.
- Mass scale under the receiving container. Tared before each pour.
- Lighting: diffuse, no direct shadows across the receiving container.
- Camera: 1080p, 30 fps, locked exposure, locked white balance.

## Per-Session Metadata (stored in `session.json` alongside the video)

```json
{
  "session_id": "2026-05-12-olive-oil-01",
  "operator": "initials",
  "liquid": "olive_oil",
  "source_container": {"brand": "...", "nozzle": "standard|flip-cap|dropper"},
  "receiving_container": {"shape": "glass|bowl|measuring-cup", "diameter_cm": 8.2},
  "ambient_temp_c": 21.0,
  "camera_distance_cm": 35.0,
  "scale_zero_g": 0.0
}
```

## Per-Pour Labels (stored in `pour.json`)

```json
{
  "pour_id": "2026-05-12-olive-oil-01-p03",
  "video_path": "pours/2026-05-12-olive-oil-01-p03.mp4",
  "liquid": "olive_oil",
  "start_frame": 12,
  "end_frame": 74,
  "total_ml": 14.3,
  "method": "scale",    // "scale" | "graduated_cylinder" | "syringe"
  "per_frame_cumulative_ml": [0, 0.1, 0.3, 0.7, ...]   // optional, from syncd scale log
}
```

- `total_ml` is the ground truth. Convert from grams using the liquid's published density at ambient temperature.
- `per_frame_cumulative_ml` is optional. Include it only when scale samples are time-synced to video frames; otherwise leave it out — noisy auxiliary targets hurt more than they help.

## Volume Ranges Per Liquid

Collect across a realistic range; avoid fixating on "pretty" pours.

| Liquid | Range per pour | Target pours | Notes |
|--------|----------------|--------------|-------|
| olive_oil | 2–40 ml | 60 | include dribble and slug variants |
| cream | 5–80 ml | 50 | full-fat only at v0 |
| milk | 10–250 ml | 60 | cartons + glass bottles |
| soy_sauce | 1–20 ml | 50 | small-drop accuracy matters |
| liquid_chocolate | 5–50 ml | 40 | warm to realistic cooking temp |
| vinegar | 2–30 ml | 40 | |

## Augmentation

Post-hoc augmentation that's OK:
- Horizontal flips
- Mild color jitter (±10%)
- Background replacement with rendered kitchen scenes

Augmentation that's NOT OK — it changes the physics:
- Speed up / slow down
- Per-frame cropping that changes the container size in frame
- Liquid colour swaps

## Quarantine Reasons

Sessions are quarantined (not deleted) when:
- Scale zero drifts >2 g between pours
- Camera moves during the session (check the first and last frame visually)
- Mass measurement is inconsistent with a syringe / graduated-cylinder check by >5%
- Audio or video dropped frames (rare, but happens on cheap phones)

Every quarantined session gets a `reason.txt` alongside its data. We review them quarterly — some are recoverable with a re-label pass.
