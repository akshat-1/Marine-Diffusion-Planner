import os
import torch
import numpy as np
from torch.utils.data import Dataset
# Assuming you use commonocean/commonroad XML readers, or a lightweight custom XML parser
from commonocean.common.file_reader import CommonOceanFileReader

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
                traj_length = len(ship.prediction.trajectory.state_list) + 1
                
                # Calculate how many full sliding windows fit in this trajectory
                valid_windows = traj_length - self.window_size + 1
                
                for start_frame in range(valid_windows):
                    self.index_map.append({
                        'file_path': file_path,
                        'ego_id': ship.obstacle_id,
                        'start_frame': start_frame
                    })
                    
        print(f"Dataset ready: {len(self.index_map)} total tensors generated from {valid_file_count} valid scenarios (out of {len(self.scenario_files)} files).")

    def __len__(self):
        return len(self.index_map)

    def _extract_state_features(self, state):
        # Extracts standard features: [x, y, velocity_x, velocity_y, heading, yaw_rate]
        return np.array([
            state.position[0], 
            state.position[1], 
            state.velocity, 
            state.velocity_y, 
            state.orientation, 
            state.yaw_rate
        ])

    def _transform_to_egocentric(self, features, ego_x, ego_y, ego_theta):
        """Applies translation and rotation to center the scene on the ego vessel."""
        x, y, vx, vy, theta, yaw_rate = features
        
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
        
        # Relative heading
        rel_theta = (theta - ego_theta + np.pi) % (2 * np.pi) - np.pi
        
        return np.array([rel_x, rel_y, rel_vx, rel_vy, rel_theta, yaw_rate], dtype=np.float32)

    def __getitem__(self, idx):
        mapping = self.index_map[idx]
        scenario, _ = CommonOceanFileReader(mapping['file_path']).open()
        
        ego_id = mapping['ego_id']
        t_start = mapping['start_frame']
        t_current = t_start + self.obs_frames - 1 # The exact moment T=0 for prediction
        
        # Extract Ego Vehicle
        ego_vessel = scenario.obstacle_by_id(ego_id)
        ego_states = [ego_vessel.initial_state] + ego_vessel.prediction.trajectory.state_list
        
        # Get the anchor state at T=current for coordinate transformation
        anchor_state = ego_states[t_current]
        ego_x, ego_y = anchor_state.position
        ego_theta = anchor_state.orientation
        
        # 1. Build Ego History Tensor
        ego_history = np.zeros((self.obs_frames, 6), dtype=np.float32)
        for i, t in enumerate(range(t_start, t_start + self.obs_frames)):
            raw_feats = self._extract_state_features(ego_states[t])
            ego_history[i] = self._transform_to_egocentric(raw_feats, ego_x, ego_y, ego_theta)
            
        # 2. Build Ego Target Tensor (What the DiT needs to predict)
        ego_target = np.zeros((self.pred_frames, 6), dtype=np.float32)
        for i, t in enumerate(range(t_current + 1, t_current + 1 + self.pred_frames)):
            raw_feats = self._extract_state_features(ego_states[t])
            ego_target[i] = self._transform_to_egocentric(raw_feats, ego_x, ego_y, ego_theta)
            
        # 3. Build Agent History Tensor
        agents_history = np.zeros((self.max_agents, self.obs_frames, 6), dtype=np.float32)
        agent_mask = np.ones((self.max_agents,), dtype=bool) # True means pad/ignore
        
        agent_idx = 0
        for obs in scenario.dynamic_obstacles:
            if obs.obstacle_id == ego_id:
                continue
                
            obs_states = [obs.initial_state] + obs.prediction.trajectory.state_list
            # Ensure agent existed during this specific time window
            if len(obs_states) <= t_current:
                continue
                
            agent_mask[agent_idx] = False
            for i, t in enumerate(range(t_start, t_start + self.obs_frames)):
                raw_feats = self._extract_state_features(obs_states[t])
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
            # Truncate or pad vertices to fixed length (e.g., 20)
            num_pts = min(len(vertices), 20)
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