import os
import torch
import numpy as np
from torch.utils.data import Dataset
# Assuming you use commonocean/commonroad XML readers, or a lightweight custom XML parser
from commonocean.common.file_reader import CommonOceanFileReader

def compute_external_forces(u, v, r, du_dt, dv_dt, dr_dt, heading, cog, length=150.0, width=25.0, draft=5.0):
    """
    Computes 4D hydrodynamic external force residual vector:
    [Fx_ext/m, Fy_ext/m, N_ext/(m*L), beta_drift]
    """
    rho_w = 1025.0  # Seawater density kg/m^3
    Cb = 0.70       # Block coefficient
    length = max(length, 10.0)
    width = max(width, 2.0)
    draft = max(draft, 1.0)

    # 1. Mass & Added Mass
    m = rho_w * Cb * length * width * draft
    mx = 0.05 * m
    my = rho_w * (np.pi / 2.0) * (draft ** 2) * length * (1.0 + 0.4 * (width / length))
    Iz = m * (0.25 * length) ** 2
    Jz = (1.0 / 12.0) * my * (length ** 2)  # Hydrodynamic strip theory added yaw inertia (Jz = 1/12 * my * L^2)

    # 2. Baseline Hull Damping & Cross-Flow Resistance
    Fx_hull = -0.5 * rho_w * 0.03 * (width * draft) * u * abs(u)
    Fy_hull = -0.5 * rho_w * 0.80 * (length * draft) * v * abs(v)
    N_hull = -(1.0 / 16.0) * rho_w * 0.80 * (length ** 2) * draft * r * abs(r)

    # 3. Inverse Dynamics Residuals
    Fx_ext = (m + mx) * du_dt - (m + my) * v * r - Fx_hull
    Fy_ext = (m + my) * dv_dt + (m + mx) * u * r - Fy_hull
    N_ext = (Iz + Jz) * dr_dt - N_hull

    # 4. Drift Angle
    beta_drift = (cog - heading + np.pi) % (2.0 * np.pi) - np.pi

    # Acceleration scale for numerical stability: (m/s^2, m/s^2, rad/s^2, rad)
    return np.array([Fx_ext / m, Fy_ext / m, N_ext / (m * length), beta_drift], dtype=np.float32)


class AISScenarioDataset(Dataset):
    def __init__(self, scenario_dir, obs_frames=20, pred_frames=20, max_agents=10, max_polylines=20):
        super().__init__()
        self.scenario_dir = scenario_dir
        self.obs_frames = obs_frames
        self.pred_frames = pred_frames
        self.window_size = obs_frames + pred_frames
        self.max_agents = max_agents
        self.max_polylines = max_polylines
        
        self.scenario_files = [os.path.join(scenario_dir, f) for f in os.listdir(scenario_dir) if f.endswith('.xml')]
        
        # Build the Index Map: A flat list of every valid sliding window
        self.index_map = []
        self._build_index_map()

    def _build_index_map(self):
        """
        Scans all XMLs, finds moving ships, and calculates valid sliding windows.
        """
        print("Building sliding window index map...")
        valid_file_count = 0
        for file_path in self.scenario_files:
            try:
                scenario, _ = CommonOceanFileReader(file_path).open()
            except Exception as e:
                print(f"Skipping corrupted XML file {os.path.basename(file_path)}: {e}")
                continue
            valid_file_count += 1
            
            # Find all ships that are NOT anchored
            moving_ships = [
                obs for obs in scenario.dynamic_obstacles 
                if obs.obstacle_type.name != "ANCHOREDVESSEL"
            ]
            
            for ship in moving_ships:
                # Get the length of this specific ship's trajectory
                # Add 1 because initial_state is the first frame, prediction holds the rest
                ship_states = [ship.initial_state] + ship.prediction.trajectory.state_list
                traj_length = len(ship_states)
                
                # Calculate how many full sliding windows fit in this trajectory
                valid_windows = traj_length - self.window_size + 1
                
                for start_frame in range(valid_windows):
                    t_curr = start_frame + self.obs_frames - 1
                    p0 = ship_states[start_frame].position
                    p_curr = ship_states[t_curr].position
                    disp = np.hypot(p0[0] - p_curr[0], p0[1] - p_curr[1])
                    if disp > 8000.0:  # Skip windows with excessive position jump > 8km
                        continue
                    fname = os.path.basename(file_path).lower()
                    if 'opensea' in fname:
                        category = 'opensea'
                    elif 'congested' in fname:
                        category = 'congested'
                    else:
                        category = 'coastal'

                    self.index_map.append({
                        'file_path': file_path,
                        'ego_id': ship.obstacle_id,
                        'start_frame': start_frame,
                        'category': category,
                    })
                    
        print(f"Dataset ready: {len(self.index_map)} total tensors generated from {valid_file_count} valid scenarios (out of {len(self.scenario_files)} files).")

    def __len__(self):
        return len(self.index_map)

    def _extract_state_features(self, state, prev_state=None, dt=10.0, length=150.0, width=25.0, draft=5.0):
        # Extracts 10D feature vector:
        # [x, y, vx_world, vy_world, heading, yaw_rate, Fx_ext, Fy_ext, N_ext, beta_drift]
        u = state.velocity
        v = state.velocity_y
        theta = state.orientation
        r = state.yaw_rate
        vx_world = u * np.cos(theta) - v * np.sin(theta)
        vy_world = u * np.sin(theta) + v * np.cos(theta)
        cog = np.arctan2(vy_world, vx_world)

        if prev_state is not None:
            u_prev = prev_state.velocity
            v_prev = prev_state.velocity_y
            r_prev = prev_state.yaw_rate
            du_dt = (u - u_prev) / dt
            dv_dt = (v - v_prev) / dt
            dr_dt = (r - r_prev) / dt
        else:
            du_dt, dv_dt, dr_dt = 0.0, 0.0, 0.0

        fx, fy, n_ext, beta = compute_external_forces(
            u, v, r, du_dt, dv_dt, dr_dt, theta, cog, length, width, draft
        )

        return np.array([
            state.position[0], 
            state.position[1], 
            vx_world, 
            vy_world, 
            theta, 
            r,
            fx,
            fy,
            n_ext,
            beta
        ], dtype=np.float32)

    def _resample_vertices(self, vertices, n_points=20):
        """Resample polygon vertices to uniform arc-length spacing."""
        if len(vertices) <= n_points:
            return vertices
        deltas = np.sqrt(np.sum(np.diff(vertices, axis=0) ** 2, axis=1))
        cum_len = np.insert(np.cumsum(deltas), 0, 0)
        total = cum_len[-1]
        if total < 1e-6:
            return vertices[:n_points]
        target_dists = np.linspace(0, total, n_points)
        resampled = np.zeros((n_points, 2), dtype=np.float64)
        for i, d in enumerate(target_dists):
            idx = min(np.searchsorted(cum_len, d, side='right') - 1, len(deltas) - 1)
            t = (d - cum_len[idx]) / max(deltas[idx], 1e-9)
            resampled[i] = vertices[idx] + t * (vertices[idx + 1] - vertices[idx])
        return resampled

    def _transform_to_egocentric(self, features, ego_x, ego_y, ego_theta):
        """Applies translation and rotation to center the scene on the ego vessel."""
        if len(features) >= 10:
            x, y, vx, vy, theta, yaw_rate, fx, fy, n_ext, beta = features[:10]
        else:
            x, y, vx, vy, theta, yaw_rate = features[:6]
            fx, fy, n_ext, beta = 0.0, 0.0, 0.0, 0.0
        
        # Translate
        dx = x - ego_x
        dy = y - ego_y
        
        # Rotate positions
        cos_t = np.cos(-ego_theta)
        sin_t = np.sin(-ego_theta)
        rel_x = cos_t * dx - sin_t * dy
        rel_y = sin_t * dx + cos_t * dy
        
        # Rotate velocities
        rel_vx = cos_t * vx - sin_t * vy
        rel_vy = sin_t * vx + cos_t * vy
        
        # Rotate external force vector into Ego body frame
        rel_fx = cos_t * fx - sin_t * fy
        rel_fy = sin_t * fx + cos_t * fy
        
        # Relative heading
        rel_theta = (theta - ego_theta + np.pi) % (2 * np.pi) - np.pi
        
        return np.array([rel_x, rel_y, rel_vx, rel_vy, rel_theta, yaw_rate, rel_fx, rel_fy, n_ext, beta], dtype=np.float32)

    def __getitem__(self, idx):
        mapping = self.index_map[idx]
        try:
            scenario, _ = CommonOceanFileReader(mapping['file_path']).open()
        except Exception as e:
            # Fallback to next sample if XML reading fails during DataLoader worker execution
            return self.__getitem__((idx + 1) % len(self.index_map))
        
        ego_id = mapping['ego_id']
        t_start = mapping['start_frame']
        t_current = t_start + self.obs_frames - 1 # The exact moment T=0 for prediction
        
        # Extract Ego Vehicle
        ego_vessel = scenario.obstacle_by_id(ego_id)
        ego_states = [ego_vessel.initial_state] + ego_vessel.prediction.trajectory.state_list
        ego_len = float(getattr(ego_vessel.obstacle_shape, 'length', 150.0))
        ego_wid = float(getattr(ego_vessel.obstacle_shape, 'width', 25.0))
        ego_draft = float(getattr(ego_vessel, 'depth', 5.0) or 5.0)

        # Get the anchor state at T=current for coordinate transformation
        anchor_state = ego_states[t_current]
        ego_x, ego_y = anchor_state.position
        ego_theta = anchor_state.orientation
        
        # 1. Build Ego History Tensor (10D features)
        ego_history = np.zeros((self.obs_frames, 10), dtype=np.float32)
        for i, t in enumerate(range(t_start, t_start + self.obs_frames)):
            prev_st = ego_states[t - 1] if t > 0 else None
            raw_feats = self._extract_state_features(ego_states[t], prev_st, dt=scenario.dt, length=ego_len, width=ego_wid, draft=ego_draft)
            ego_history[i] = self._transform_to_egocentric(raw_feats, ego_x, ego_y, ego_theta)
            
        # 2. Build Ego Target Tensor (10D features)
        ego_target = np.zeros((self.pred_frames, 10), dtype=np.float32)
        for i, t in enumerate(range(t_current + 1, t_current + 1 + self.pred_frames)):
            prev_st = ego_states[t - 1]
            raw_feats = self._extract_state_features(ego_states[t], prev_st, dt=scenario.dt, length=ego_len, width=ego_wid, draft=ego_draft)
            ego_target[i] = self._transform_to_egocentric(raw_feats, ego_x, ego_y, ego_theta)
            
        # 3. Build Agent History Tensor (10D features)
        agents_history = np.zeros((self.max_agents, self.obs_frames, 10), dtype=np.float32)
        agent_mask = np.ones((self.max_agents,), dtype=bool) # True means pad/ignore
        
        agent_idx = 0
        for obs in scenario.dynamic_obstacles:
            if obs.obstacle_id == ego_id:
                continue
                
            obs_states = [obs.initial_state] + obs.prediction.trajectory.state_list
            if len(obs_states) <= t_current:
                continue
                
            ag_len = float(getattr(obs.obstacle_shape, 'length', 150.0))
            ag_wid = float(getattr(obs.obstacle_shape, 'width', 25.0))
            ag_draft = float(getattr(obs, 'depth', 5.0) or 5.0)

            agent_mask[agent_idx] = False
            for i, t in enumerate(range(t_start, t_start + self.obs_frames)):
                prev_st = obs_states[t - 1] if t > 0 else None
                raw_feats = self._extract_state_features(obs_states[t], prev_st, dt=scenario.dt, length=ag_len, width=ag_wid, draft=ag_draft)
                agents_history[agent_idx, i] = self._transform_to_egocentric(raw_feats, ego_x, ego_y, ego_theta)
                
            agent_idx += 1
            if agent_idx >= self.max_agents:
                break
                
        # 4. Build Map/Coastline Tensor
        map_lines = np.zeros((self.max_polylines, 20, 2), dtype=np.float32) # Assuming 20 points per line max
        map_mask = np.ones((self.max_polylines,), dtype=bool)
        
        line_idx = 0
        for obs in scenario.static_obstacles:
            # Note: You may need to extract the raw vertices depending on how CommonOcean stores StaticObstacle polygons
            # Robust vertex extraction for different CommonOcean/CommonRoad versions
            if hasattr(obs.obstacle_shape, 'vertices') and callable(obs.obstacle_shape.vertices):
                # Method-based access (newer versions)
                vertices = obs.obstacle_shape.vertices()
            elif hasattr(obs.obstacle_shape, 'vertices'):
                # Property-based access (older versions)
                vertices = obs.obstacle_shape.vertices
            elif hasattr(obs.obstacle_shape, 'get_vertices'):
                # Alternative method name
                vertices = obs.obstacle_shape.get_vertices()
            else:
                # Fallback: try to get from shape's internal polygon
                try:
                    import shapely.geometry as geom
                    if hasattr(obs.obstacle_shape, '_polygon'):
                        poly = obs.obstacle_shape._polygon
                        if hasattr(poly, 'exterior'):
                            vertices = np.array(poly.exterior.coords)
                        else:
                            vertices = np.array(poly.coords)
                    else:
                        vertices = []
                except Exception:
                    vertices = []
            
            # Ensure we have a numpy array of vertices
            if len(vertices) == 0:
                continue
                
            if not isinstance(vertices, np.ndarray):
                vertices = np.array(vertices)
            
            map_mask[line_idx] = False
            # Resample vertices uniformly to fixed length (20 points)
            vertices = self._resample_vertices(vertices, n_points=20)
            num_pts = len(vertices)
            for i in range(num_pts):
                raw_feats = np.array([vertices[i][0], vertices[i][1], 0, 0, 0, 0])
                rel_feats = self._transform_to_egocentric(raw_feats, ego_x, ego_y, ego_theta)
                map_lines[line_idx, i] = rel_feats[:2] # Only keep x, y
                
            line_idx += 1
            if line_idx >= self.max_polylines:
                break

        return {
            'ego_history': torch.tensor(ego_history),
            'agents_history': torch.tensor(agents_history),
            'map_lines': torch.tensor(map_lines),
            'agent_mask': torch.tensor(agent_mask),
            'map_mask': torch.tensor(map_mask),
            'ego_target': torch.tensor(ego_target)
        }