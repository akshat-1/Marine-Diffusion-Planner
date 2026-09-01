# preparedataset.py Analysis

This document provides an in-depth analysis of `preparedataset.py`, assessing:
- How it constructs training tensors.
- Specific mathematical calculations embedded.
- Handling of input coastline data compatibility.
- Identification of any garbage issues and corrections.

---
## Core Functionality

`AISScenarioDataset` builds a data pipeline, deriving tensors for `torch.utils.data.Dataset`. Input components:

- **Observation Frames** (historical): 20 frames of `ego` and `agents_history` velocity and positional data.
- **Target Frames** (future prediction): 20 frames (default) representing target trajectory coordinates.
- **Coastline (`StaticObstacle`) Tensor** (`map_lines`): Relevant segments clipped from scenarios.

### Key Attributes:
| Name           | Shape                      | Description                                 |
|----------------|----------------------------|---------------------------------------------|
| `ego_history`  | `(obs_frames, 6)`          | [`x`, `y`, `velocity_x`, `velocity_y`, `heading`, `yaw_rate`] rows. |
| `agents_hist`  | `(max_agents, obs, 6)`     | Same as `ego_history` but for other ships. |
| `map_lines`    | `(max_polylines, max_pts)` | Stores coastline x-y after egotransform. |

Tensors validate compatibility:
- Torch boolean masks (`agent_mask`/`map_mask`) set values `True` to mark padding for absent entities.
- **Sliding Window Index Map** aligns obs/pred windows over trajectories.

---
## Mathematical Foundations

### Egocentric Transform Formula:

For each feature vector `(x, y, vx, vy, theta, yaw_rate)` of any agent/polyline point, the transform to ego-centric coordinates at time `t_current` is:

**Translation** (origin shift to ego position):
```
dx = x - ego_x
dy = y - ego_y
```

**Rotation** (align with ego heading at `t_current`, i.e., `ego_theta`):
```
cos_t = cos(-ego_theta)
sin_t = sin(-ego_theta)

rel_x = cos_t * dx - sin_t * dy
rel_y = sin_t * dx + cos_t * dy
```

**Velocity rotation** (same rotation matrix):
```
rel_vx = cos_t * vx - sin_t * vy
rel_vy = sin_t * vx + cos_t * vy
```

**Heading normalization** (wrap to [-π, π]):
```
rel_theta = (theta - ego_theta + π) % (2π) - π
```

This ensures ego vessel at `t_current` is exactly at `(0, 0)` with heading `0`.

### State Feature Extraction:

```python
def _extract_state_features(self, state):
    return np.array([
        state.position[0],   # x (meters, local aeqd projection)
        state.position[1],   # y
        state.velocity,      # velocity_x  (surge_u)
        state.velocity_y,    # velocity_y  (sway_v)
        state.orientation,   # heading (radians)
        state.yaw_rate       # yaw_rate (rad/s)
    ])
```

**Note:** CommonOcean states provide `velocity` (`surge_u`) and `velocity_y` (`sway_v`) as separate components already in vessel body frame.

---
## Data Flow Pipeline

```
1. Index Map Building
   ├─ Iterate all XMLs in scenario_dir
   ├─ Filter ANCHOREDVESSEL
   ├─ For each moving ship:
   │    traj_length = len(prediction.trajectory.state_list) + 1 (initial_state)
   │    valid_windows = traj_length - (obs_frames + pred_frames) + 1
   │    For each start_frame: add {file_path, ego_id, start_frame} to index_map

2. __getitem__(idx)
   ├─ Read scenario XML
   ├─ Get ego vessel by ego_id
   ├─ Anchor at t_current = start_frame + obs_frames - 1
   ├─ Build ego_history [t_start ... t_current] (20 frames)
   ├─ Build ego_target  [t_current+1 ... t_current+pred_frames] (20 frames)
   ├─ Build agents_history for other vessels in same time window
   ├─ Extract static obstacles (coastlines) as polylines
   │    - vertices from StaticObstacle.obstacle_shape.vertices (Polygon)
   │    - Transform each vertex to egocentric
   │    - Truncate/pad to 20 points per polyline, 20 polylines max
   └─ Return dict of tensors
```

---
## Coastline Data Compatibility

### Scenario Generation (aisdatageneratoroffline.py):
- Uses Natural Earth / GSHHG shapefiles via `geopandas`
- Buffers coastline lines by 20m, simplifies to 5m tolerance
- Indexed via STRtree for fast spatial queries
- Added as `StaticObstacle(obstacle_type=ObstacleType.LAND, obstacle_shape=CRPolygon(...))`

### Dataset Loading (preparedataset.py):
- Reads `scenario.static_obstacles`
- Extracts vertices via multiple fallbacks:
  ```python
  if hasattr(shape, 'vertices') and callable(shape.vertices):
      vertices = shape.vertices()
  elif hasattr(shape, 'vertices'):
      vertices = shape.vertices
  elif hasattr(shape, 'get_vertices'):
      vertices = shape.get_vertices()
  else:
      # shapely fallback via _polygon
  ```

### Verification on `/run/media/akshat/Akshat_USB/generated_scenarios20/scenario_0000.xml`:
- **7 StaticObstacles** found (all `ObstacleType.LAND`)
- Vertices are `numpy.ndarray` with shape `(N, 2)` ✓
- Example vertex counts: 192, 748, 160, 73, 9, 13, 806
- **Transformation works correctly** — `map_lines` tensor produced with valid coordinates

**Compatibility Result: ✓ COMPATIBLE**

The offline coastline data from `aisdatageneratoroffline.py` is correctly embedded in the XML and successfully consumed by `preparedataset.py`.

---
## Garbage Detection & Issues

### 1. CRITICAL BUG: Wrong Velocity Feature Mapping

**Location:** `_extract_state_features` (lines 64-73)

```python
def _extract_state_features(self, state):
    return np.array([
        state.position[0],
        state.position[1],
        state.velocity,      # ← This is surge_u (body-frame forward velocity)
        state.velocity_y,    # ← This is sway_v (body-frame lateral velocity)
        state.orientation,
        state.yaw_rate
    ])
```

**Problem:** The function assumes `state.velocity` = `velocity_x` (world-frame x-component) and `state.velocity_y` = `velocity_y` (world-frame y-component). But CommonOcean's `InitialState` / `TFState` store:
- `velocity` = surge velocity (body-frame longitudinal)
- `velocity_y` = sway velocity (body-frame lateral)

**Impact:** The `_transform_to_egocentric` function then applies a rotation matrix treating these as world-frame velocities:
```python
rel_vx = cos_t * vx - sin_t * vy  # WRONG: vx,vy already in body frame!
rel_vy = sin_t * vx + cos_t * vy
```

This double-rotates the velocities — first from world to body (done by AIS generator), then body to ego (done by dataset). The resulting velocity features are **garbage**.

**Correct approach:** Either:
- Store raw surge/sway in dataset and let model learn the transform, OR
- Convert surge/sway to world-frame BEFORE egocentric transform:
  ```python
  # In _extract_state_features:
  heading = state.orientation
  vx_world = state.velocity * cos(heading) - state.velocity_y * sin(heading)
  vy_world = state.velocity * sin(heading) + state.velocity_y * cos(heading)
  ```

### 2. Inconsistent First-Frame Ego Positions

Tested on multiple indices:
```
idx 0:     ego first obs pos: (-1.615,  219.310)  # y=219m off-center?
idx 5000:  ego first obs pos: (-301.339,  5.079)  # x=-301m off-center?
idx 50000: ego first obs pos: (-375.672, 33.023)  # x=-375m off-center?
```

**Expected:** At `t_start`, ego should be within scene radius (typically < 100m) from its position at `t_current`.

**Root Cause:** The sliding window index map includes windows where the ego ship has incomplete trajectory data near edges, or the `initial_state` position differs significantly from `state_list[0]` due to AIS gaps/resampling.

**Mitigation:** Add validation that `|ego_history[0, :2]| < MAX_SCENE_RADIUS_M` or filter windows where ego moves too far.

### 3. Coastline Point Sampling Bias

Current code takes first 20 vertices of each polygon:
```python
num_pts = min(len(vertices), 20)
for i in range(num_pts):
    raw_feats = np.array([vertices[i][0], vertices[i][1], 0, 0, 0, 0])
```

**Issue:** Coastline polygons can have hundreds of vertices. Taking only the first 20 biases toward an arbitrary polygon corner, potentially missing coastline near the ego vessel.

**Fix:** Resample/interpolate polyline to uniform spacing, or select vertices nearest to ego.

### 4. Map Lines Feature Dimensionality

Current: `map_lines` shape `(max_polylines, 20, 2)` — only x,y stored.

But `HighCapacityVectorSceneEncoder` expects 5D map features (line 237-241):
```python
delta = map_flat[:, 1:] - map_flat[:, :-1]
dist = torch.norm(map_flat, dim=-1, keepdim=True)
map_feats_5d = torch.cat([map_flat, delta, dist], dim=-1)  # (x, y, dx, dy, dist)
```

The dataset provides only `(x, y)`, but encoder expects to compute `delta` and `dist` internally. This **works** because encoder computes it, but means `map_lines` wastes capacity (could precompute).

### 5. XML Corruption Tolerance

2 of 3157 files corrupted:
- `scenario_2941.xml`: "XML or text declaration not at start of entity"
- `scenario_3104.xml`: Same error

Likely caused by interrupted writes during scenario generation. The dataset gracefully skips these.

---
## Summary of Fixes Required

| Priority | Issue | Fix |
|----------|-------|-----|
| **CRITICAL** | Velocity double-rotation | Fix `_extract_state_features` to output world-frame velocities |
| HIGH | First-frame ego position outliers | Add window validation / filter |
| MEDIUM | Coastline vertex sampling bias | Resample polylines uniformly or nearest-ego |
| LOW | Precompute map deltas/dist | Modify dataset to output 5D map features |
| LOW | Corrupted XML files | Fix generator write atomicity |

---
## Testing Verification

### Quick Test:
```bash
# With venv activated
python -c "
from preparedataset import AISScenarioDataset
ds = AISScenarioDataset('/run/media/akshat/Akshat_USB/generated_scenarios20')
print(f'Samples: {len(ds)}')
s = ds[0]
for k, v in s.items():
    print(f'{k}: {v.shape} {v.dtype} range=[{v.min():.1f}, {v.max():.1f}] nan={v.isnan().any()} inf={v.isinf().any()}')
"
```

### Expected Clean Output (after velocity fix):
- `ego_history[:, :2]` range ~ [-100, 100] meters (within scene radius)
- `ego_history[:, 2:4]` range ~ [-10, 10] m/s (reasonable vessel speeds)
- `map_lines` range ~ [-8000, 8000] meters (coastlines within `MAX_SCENE_RADIUS_M`)

---
## Conclusion

The `preparedataset.py` script **is compatible** with the offline coastline scenarios generated by `aisdatageneratoroffline.py`. The coastline data flows correctly through the pipeline as `StaticObstacle` → `map_lines` tensor.

**However**, the dataset produces **garbage velocity features** due to a fundamental misunderstanding of CommonOcean's state representation. This must be fixed before training.

Once the velocity extraction is corrected, the dataset should produce valid, physics-consistent tensors suitable for the DP-VLA diffusion transformer training.