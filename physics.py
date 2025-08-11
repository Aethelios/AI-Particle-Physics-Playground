import numpy as np
import config

class PhysicsEngine:
    """
    Manages the state and physics of all particles in the simulation.
    """
    def __init__(self, count):
        self.particle_count = count
        self.particles = self._initialize_particles()
        self.previous_particles = None # For Verlet integration
        self.first_step = True

    def _initialize_particles(self):
        """
        Creates the main NumPy array to hold particle data.
        Array structure: [x, y, vx, vy, mass, radius, charge]
        """
        particles = np.zeros((self.particle_count, 7), dtype=np.float32)
        
        # Random positions within the screen bounds
        particles[:, 0] = np.random.rand(self.particle_count) * config.SCREEN_WIDTH
        particles[:, 1] = np.random.rand(self.particle_count) * config.SCREEN_HEIGHT
        
        # Random initial velocities
        particles[:, 2] = (np.random.rand(self.particle_count) - 0.5) * 2 # -1 to 1
        particles[:, 3] = (np.random.rand(self.particle_count) - 0.5) * 2 # -1 to 1
        
        # Random mass and radius
        particles[:, 4] = np.random.uniform(config.MIN_MASS, config.MAX_MASS, self.particle_count)
        particles[:, 5] = np.random.uniform(config.MIN_RADIUS, config.MAX_RADIUS, self.particle_count)
        
        # Random charge for electromagnetic forces
        particles[:, 6] = np.random.uniform(config.MIN_CHARGE, config.MAX_CHARGE, self.particle_count)
        
        return particles
    
    def _calculate_gravitational_forces(self):
        """
        Calculates the net gravitational force on each particle.
        This is a vectorized operation for performance.
        """
        # Extract position and mass data for easy access
        positions = self.particles[:, :2] # Shape: (N, 2)
        masses = self.particles[:, 4]    # Shape: (N,)

        # Calculate pairwise displacement vectors
        # Broadcasting positions (N, 1, 2) - (1, N, 2) -> (N, N, 2)
        displacements = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        
        # Calculate pairwise distances squared
        # np.sum(displacements**2, axis=-1) gives a (N, N) matrix of dist^2
        distances_sq = np.sum(displacements**2, axis=-1)
        
        # Avoid division by zero for a particle with itself
        # We add a small epsilon and set the diagonal to infinity
        distances_sq[distances_sq < 1e-6] = 1e-6 # Avoid near-zero distance instability
        np.fill_diagonal(distances_sq, np.inf)

        # Calculate gravitational force magnitude: F = G * (m1 * m2) / r^2
        # Broadcasting masses (N, 1) * (1, N) -> (N, N)
        mass_products = masses[:, np.newaxis] * masses[np.newaxis, :]
        force_magnitudes = config.G * mass_products / distances_sq
        
        # Calculate force vectors
        # Force vector = magnitude * unit_direction_vector
        # unit_direction_vector = displacements / distances
        distances = np.sqrt(distances_sq)
        force_vectors = force_magnitudes[:, :, np.newaxis] * (displacements / distances[:, :, np.newaxis])
        
        # Sum all forces for each particle to get the net force
        # np.sum(..., axis=1) sums forces from all other particles
        net_forces = np.sum(force_vectors, axis=1)
        return net_forces
    
    def _calculate_electromagnetic_forces(self):
        """
        Calculates the net electromagnetic force on each particle.
        F = k * (q1 * q2) / r^2, repulsive for like charges, attractive for opposite charges.
        """
        # Extract position and charge data
        positions = self.particles[:, :2] # Shape: (N, 2)
        charges = self.particles[:, 6]    # Shape: (N,)

        # Calculate pairwise displacement vectors
        displacements = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        
        # Calculate pairwise distances squared
        distances_sq = np.sum(displacements**2, axis=-1)
        
        # Avoid division by zero
        distances_sq[distances_sq < 1e-6] = 1e-6
        np.fill_diagonal(distances_sq, np.inf)

        # Calculate electromagnetic force magnitude: F = k * (q1 * q2) / r^2
        charge_products = charges[:, np.newaxis] * charges[np.newaxis, :]
        force_magnitudes = config.K_E * charge_products / distances_sq
        
        # Calculate force vectors (repulsive for same sign charges, attractive for opposite)
        distances = np.sqrt(distances_sq)
        # Note: The sign of charge_products determines attraction/repulsion
        force_vectors = force_magnitudes[:, :, np.newaxis] * (displacements / distances[:, :, np.newaxis])
        
        # Sum all forces for each particle
        net_forces = np.sum(force_vectors, axis=1)
        return net_forces
    
    def _calculate_all_forces(self):
        """
        Calculates the total net force on each particle (gravitational + electromagnetic).
        """
        gravitational_forces = self._calculate_gravitational_forces()
        electromagnetic_forces = self._calculate_electromagnetic_forces()
        return gravitational_forces + electromagnetic_forces
    
    def _detect_collisions(self):
        """
        Detects all particle-particle collisions.
        Returns arrays of collision pairs and collision information.
        """
        positions = self.particles[:, :2]
        radii = self.particles[:, 5]
        
        # Calculate pairwise distances
        displacements = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        distances = np.sqrt(np.sum(displacements**2, axis=-1))
        
        # Calculate sum of radii for each pair
        radii_sums = radii[:, np.newaxis] + radii[np.newaxis, :]
        
        # Find colliding pairs (distance < sum of radii)
        # Use upper triangle to avoid duplicate pairs and self-collisions
        collision_mask = (distances < radii_sums) & (distances > 0)
        collision_mask = np.triu(collision_mask, k=1)  # Upper triangle, excluding diagonal
        
        # Get indices of colliding pairs
        i_indices, j_indices = np.where(collision_mask)
        
        return i_indices, j_indices, distances[collision_mask], radii_sums[collision_mask]
    
    def _resolve_collision(self, i, j, distance, radii_sum):
        """
        Resolves collision between particles i and j using elastic collision physics.
        """
        # Get particle data
        pos_i = self.particles[i, :2]
        pos_j = self.particles[j, :2]
        vel_i = self.particles[i, 2:4]
        vel_j = self.particles[j, 2:4]
        mass_i = self.particles[i, 4]
        mass_j = self.particles[j, 4]
        
        # Calculate collision normal (unit vector from i to j)
        displacement = pos_j - pos_i
        if distance < 1e-6:  # Avoid division by zero
            # Use random direction if particles are exactly on top of each other
            angle = np.random.uniform(0, 2 * np.pi)
            normal = np.array([np.cos(angle), np.sin(angle)])
        else:
            normal = displacement / distance
        
        # Separate overlapping particles
        overlap = radii_sum - distance
        separation = (overlap + config.MIN_COLLISION_DISTANCE) * 0.5
        self.particles[i, :2] -= normal * separation * mass_j / (mass_i + mass_j)
        self.particles[j, :2] += normal * separation * mass_i / (mass_i + mass_j)
        
        # Calculate relative velocity
        relative_velocity = vel_i - vel_j
        
        # Calculate relative velocity along collision normal
        velocity_along_normal = np.dot(relative_velocity, normal)
        
        # Don't resolve if velocities are separating
        if velocity_along_normal > 0:
            return
        
        # Calculate impulse scalar
        impulse = (1 + config.RESTITUTION_COEFFICIENT) * velocity_along_normal
        impulse /= (1/mass_i + 1/mass_j)
        
        # Apply impulse to velocities
        impulse_vector = impulse * normal
        self.particles[i, 2:4] -= impulse_vector / mass_i
        self.particles[j, 2:4] += impulse_vector / mass_j
    
    def _handle_collisions(self):
        """
        Detects and resolves all collisions in the current frame.
        """
        i_indices, j_indices, distances, radii_sums = self._detect_collisions()
        
        # Resolve each collision
        for k in range(len(i_indices)):
            i, j = i_indices[k], j_indices[k]
            distance = distances[k]
            radii_sum = radii_sums[k]
            self._resolve_collision(i, j, distance, radii_sum)
    
    def update_euler(self, dt):
        """
        Updates the state of all particles using Euler integration.
        """
        forces = self._calculate_all_forces()
        
        # F = ma -> a = F/m
        # We need to reshape masses for broadcasting: (N,) -> (N, 1)
        accelerations = forces / self.particles[:, 4:5]
        
        # Euler Integration
        # Update velocities: v_new = v_old + a * dt
        self.particles[:, 2:4] += accelerations * dt
        
        # Update positions: p_new = p_old + v * dt
        self.particles[:, 0:2] += self.particles[:, 2:4] * dt
        
        # Handle screen boundaries (wrap-around)
        self.particles[:, 0] %= config.SCREEN_WIDTH
        self.particles[:, 1] %= config.SCREEN_HEIGHT
        
        # Handle collisions
        self._handle_collisions()
    
    def update_verlet(self, dt):
        """
        Updates the state of all particles using Verlet integration.
        More stable for physics simulations, especially with strong forces.
        """
        forces = self._calculate_all_forces()
        accelerations = forces / self.particles[:, 4:5]
        
        if self.first_step or self.previous_particles is None:
            # For the first step, use Euler method to get previous positions
            self.previous_particles = self.particles.copy()
            
            # Update velocities and positions using Euler
            self.particles[:, 2:4] += accelerations * dt
            self.particles[:, 0:2] += self.particles[:, 2:4] * dt
            
            self.first_step = False
        else:
            # Verlet integration: x(t+dt) = 2*x(t) - x(t-dt) + a(t)*dt^2
            current_positions = self.particles[:, 0:2].copy()
            
            # Calculate new positions
            self.particles[:, 0:2] = (2 * current_positions - 
                                     self.previous_particles[:, 0:2] + 
                                     accelerations * dt * dt)
            
            # Update velocities: v = (x(t+dt) - x(t-dt)) / (2*dt)
            self.particles[:, 2:4] = (self.particles[:, 0:2] - self.previous_particles[:, 0:2]) / (2 * dt)
            
            # Store previous positions
            self.previous_particles[:, 0:2] = current_positions
        
        # Handle screen boundaries (wrap-around)
        self.particles[:, 0] %= config.SCREEN_WIDTH
        self.particles[:, 1] %= config.SCREEN_HEIGHT
        
        # Handle collisions
        self._handle_collisions()
    
    def update(self, dt):
        """
        Updates the state of all particles for a given time step 'dt'.
        Uses either Euler or Verlet integration based on configuration.
        """
        if config.USE_VERLET:
            self.update_verlet(dt)
        else:
            self.update_euler(dt)

    def apply_agent_action(self, action_index):
        """
        Applies a force to the agent particle based on the action index.
        """
        if action_index is None:
            return

        force_vector = np.zeros(2)
        magnitude = config.AGENT_FORCE_MAGNITUDE

        if action_index == 0:   # Up
            force_vector[1] = -magnitude
        elif action_index == 1: # Down
            force_vector[1] = magnitude
        elif action_index == 2: # Left
            force_vector[0] = -magnitude
        elif action_index == 3: # Right
            force_vector[0] = magnitude
        # if action_index == 4 (None), force_vector remains [0, 0]
        
        # Apply this force directly to the particle's velocity/position
        # We'll treat this as an impulse for simplicity, directly adding to velocity
        # A more realistic way would be to add to a 'forces' array before integration
        agent_mass = self.particles[config.AGENT_PARTICLE_INDEX, 4]
        acceleration = force_vector / agent_mass
        
        # Assuming dt=1 for the impulse. This is a simplification.
        self.particles[config.AGENT_PARTICLE_INDEX, 2:4] += acceleration