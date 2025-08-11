import numpy as np
import config
from physics import PhysicsEngine

class ParticleEnv:
    """A wrapper for the PhysicsEngine to make it a standard RL environment."""
    def __init__(self):
        self.physics_engine = PhysicsEngine(config.PARTICLE_COUNT)
        self.agent_idx = config.AGENT_PARTICLE_INDEX
        self.k_nearest = 5 # Number of nearest particles to observe

    def reset(self):
        """Resets the environment to an initial state."""
        self.physics_engine = PhysicsEngine(config.PARTICLE_COUNT)
        return self._get_state()

    def step(self, action):
        """
        Takes an action, updates the environment, and returns the next state,
        reward, and a done flag.
        """
        # 1. Apply agent's action
        self.physics_engine.apply_agent_action(action)

        # 2. Update the physics simulation
        dt = 1 / config.FPS
        self.physics_engine.update(dt)

        # 3. Check for collisions (the terminal condition)
        done, collision_penalty = self._check_collisions()

        # 4. Calculate reward
        # Reward is a small positive value for survival + a large negative for collision
        reward = 0.1 + collision_penalty

        # 5. Get the next state
        next_state = self._get_state()

        return next_state, reward, done

    def _get_state(self):
        """
        Constructs the state vector for the agent.
        State: [own_vx, own_vy, rel_px1, rel_py1, rel_vx1, rel_vy1, ...] for k nearest
        """
        agent_particle = self.physics_engine.particles[self.agent_idx]
        other_particles = np.delete(self.physics_engine.particles, self.agent_idx, axis=0)

        # Agent's own velocity
        agent_vel = agent_particle[2:4]

        # Find k-nearest neighbors
        displacements = other_particles[:, :2] - agent_particle[:2]
        distances_sq = np.sum(displacements**2, axis=1)
        
        # Get indices of the k nearest particles
        num_neighbors = min(self.k_nearest, len(other_particles))
        nearest_indices = np.argsort(distances_sq)[:num_neighbors]
        
        nearest_features = []
        for i in nearest_indices:
            neighbor = other_particles[i]
            # Relative position
            rel_pos = neighbor[:2] - agent_particle[:2]
            # Relative velocity
            rel_vel = neighbor[2:4] - agent_particle[2:4]
            nearest_features.extend(rel_pos / config.SCREEN_WIDTH) # Normalize
            nearest_features.extend(rel_vel)

        # Pad with zeros if there are fewer than k neighbors
        padding_size = (self.k_nearest * 4) - len(nearest_features)
        if padding_size > 0:
            nearest_features.extend([0] * padding_size)

        # Combine agent velocity and neighbor features into the final state vector
        state = np.concatenate((agent_vel, np.array(nearest_features))).astype(np.float32)
        
        # Ensure state is the correct size, just in case
        return state[:config.STATE_SIZE]

    def _check_collisions(self):
        """Checks if the agent particle has collided with any other particle."""
        agent_p = self.physics_engine.particles[self.agent_idx]
        other_ps = np.delete(self.physics_engine.particles, self.agent_idx, axis=0)
        
        for other_p in other_ps:
            dist_sq = np.sum((agent_p[:2] - other_p[:2])**2)
            radii_sum_sq = (agent_p[5] + other_p[5])**2
            if dist_sq < radii_sum_sq:
                return True, -100.0  # Collision occurred, return done=True and a penalty
        
        return False, 0.0 # No collision