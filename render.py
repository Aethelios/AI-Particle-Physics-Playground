import pygame
import numpy as np
import config
import math

def lerp_color(color1, color2, t):
    """Linearly interpolates between two colors."""
    t = np.clip(t, 0, 1)
    return tuple(int(c1 + (c2 - c1) * t) for c1, c2 in zip(color1, color2))

def hsv_to_rgb(h, s, v):
    """Convert HSV to RGB color space."""
    h = h % 360
    c = v * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c
    
    if 0 <= h < 60:
        rgb = (c, x, 0)
    elif 60 <= h < 120:
        rgb = (x, c, 0)
    elif 120 <= h < 180:
        rgb = (0, c, x)
    elif 180 <= h < 240:
        rgb = (0, x, c)
    elif 240 <= h < 300:
        rgb = (x, 0, c)
    else:
        rgb = (c, 0, x)
    
    return tuple(int((comp + m) * 255) for comp in rgb)

class Renderer:
    def __init__(self, surface):
        self.surface = surface
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_medium = pygame.font.SysFont("monospace", config.UI_FONT_SIZE)
        self.font_large = pygame.font.SysFont("monospace", 32)
        
        # Visual enhancement settings
        self.show_velocity_vectors = False
        self.show_particle_info = False
        self.show_grid = False
        self.show_center_of_mass = False
        self.show_energy_info = False
        self.particle_glow = True
        
        # Performance optimization
        self.detail_level = 1.0  # Can be reduced for better performance
        
    def draw(self, particles, fps, show_trails=False, show_forces=False, show_agent=False):
        """
        Enhanced drawing method with multiple visual options.
        """
        # Clear background
        self.surface.fill(config.BACKGROUND_COLOR)
        
        # Draw grid if enabled
        if self.show_grid:
            self._draw_grid()
        
        # Draw center of mass if enabled
        if self.show_center_of_mass:
            self._draw_center_of_mass(particles)
        
        # Draw particles with various visual enhancements
        self._draw_particles(particles, show_agent)
        
        # Draw velocity vectors if enabled
        if self.show_velocity_vectors:
            self._draw_velocity_vectors(particles)
        
        # Draw particle information if enabled
        if self.show_particle_info:
            self._draw_particle_info(particles)
        
        # Draw energy information
        if self.show_energy_info:
            self._draw_energy_info(particles)
        
        # Draw main UI information
        self._draw_main_ui(fps, len(particles))
    
    def _draw_grid(self):
        """Draw a background grid for reference."""
        grid_size = 50
        grid_color = (30, 30, 50)
        
        # Vertical lines
        for x in range(0, config.SCREEN_WIDTH, grid_size):
            pygame.draw.line(self.surface, grid_color, 
                           (x, 0), (x, config.SCREEN_HEIGHT), 1)
        
        # Horizontal lines
        for y in range(0, config.SCREEN_HEIGHT, grid_size):
            pygame.draw.line(self.surface, grid_color, 
                           (0, y), (config.SCREEN_WIDTH, y), 1)
    
    def _draw_center_of_mass(self, particles):
        """Draw the center of mass of the system."""
        if len(particles) == 0:
            return
        
        # Calculate center of mass
        total_mass = np.sum(particles[:, 4])
        if total_mass == 0:
            return
        
        com_x = np.sum(particles[:, 0] * particles[:, 4]) / total_mass
        com_y = np.sum(particles[:, 1] * particles[:, 4]) / total_mass
        
        com_pos = (int(com_x), int(com_y))
        
        # Draw center of mass marker
        pygame.draw.circle(self.surface, (255, 255, 0), com_pos, 8, 2)
        pygame.draw.circle(self.surface, (255, 255, 0), com_pos, 4, 2)
        
        # Draw label
        text = self.font_small.render("COM", True, (255, 255, 0))
        self.surface.blit(text, (com_pos[0] + 10, com_pos[1] - 10))
    
    def _draw_particles(self, particles, show_agent=False):
        """Enhanced particle rendering with multiple visual modes."""
        positions = particles[:, 0:2]
        masses = particles[:, 4]
        radii = particles[:, 5]
        charges = particles[:, 6]
        velocities = particles[:, 2:4]
        
        # Normalize values for color mapping
        min_m, max_m = config.MIN_MASS, config.MAX_MASS
        min_c, max_c = config.MIN_CHARGE, config.MAX_CHARGE
        
        # Calculate velocity magnitudes for color coding
        vel_magnitudes = np.sqrt(np.sum(velocities**2, axis=1))
        max_vel = np.max(vel_magnitudes) if len(vel_magnitudes) > 0 else 1.0
        
        for i in range(len(particles)):
            pos = (int(positions[i, 0]), int(positions[i, 1]))
            radius = max(2, int(radii[i] * self.detail_level))
            
            # Determine particle color based on multiple factors
            if self.particle_glow:
                color = self._get_enhanced_particle_color(
                    masses[i], charges[i], vel_magnitudes[i], max_vel, min_m, max_m, min_c, max_c
                )
                
                # Draw glow effect
                if radius > 3:
                    glow_radius = radius + 3
                    glow_color = tuple(c // 3 for c in color)
                    pygame.draw.circle(self.surface, glow_color, pos, glow_radius)
                    
                    glow_radius = radius + 1
                    glow_color = tuple(c // 2 for c in color)
                    pygame.draw.circle(self.surface, glow_color, pos, glow_radius)
            else:
                # Simple color based on mass
                norm_mass = (masses[i] - min_m) / (max_m - min_m) if max_m > min_m else 0.5
                color = lerp_color(
                    config.PARTICLE_COLOR_LOW_MASS, 
                    config.PARTICLE_COLOR_HIGH_MASS, 
                    norm_mass
                )
            
            # Draw main particle
            pygame.draw.circle(self.surface, color, pos, radius)
            
            # Draw particle outline
            outline_color = tuple(min(255, c + 50) for c in color)
            pygame.draw.circle(self.surface, outline_color, pos, radius, 1)
            
            # Highlight agent particle
            if show_agent and i == config.AGENT_PARTICLE_INDEX:
                pygame.draw.circle(self.surface, (0, 255, 0), pos, radius + 4, 3)
                pygame.draw.circle(self.surface, (0, 255, 0, 128), pos, radius + 7, 1)
    
    def _get_enhanced_particle_color(self, mass, charge, velocity, max_vel, min_m, max_m, min_c, max_c):
        """Calculate enhanced particle color based on multiple properties."""
        # Base hue from charge (blue for negative, red for positive)
        if max_c > min_c:
            charge_norm = (charge - min_c) / (max_c - min_c)
        else:
            charge_norm = 0.5
        
        # Hue: 0-240 (red to blue spectrum)
        hue = 240 * (1 - charge_norm)  # Negative charges are blue, positive are red
        
        # Saturation based on mass
        if max_m > min_m:
            mass_norm = (mass - min_m) / (max_m - min_m)
        else:
            mass_norm = 0.5
        saturation = 0.3 + 0.7 * mass_norm  # Higher mass = more saturated
        
        # Brightness based on velocity
        if max_vel > 0:
            vel_norm = velocity / max_vel
        else:
            vel_norm = 0
        brightness = 0.4 + 0.6 * vel_norm  # Higher velocity = brighter
        
        return hsv_to_rgb(hue, saturation, brightness)
    
    def _draw_velocity_vectors(self, particles):
        """Draw velocity vectors for particles."""
        positions = particles[:, 0:2]
        velocities = particles[:, 2:4]
        
        vector_scale = 10  # Scale factor for vector visibility
        
        for i in range(len(particles)):
            pos = (int(positions[i, 0]), int(positions[i, 1]))
            vel = velocities[i] * vector_scale
            
            if np.linalg.norm(vel) > 1:  # Only draw significant velocities
                end_pos = (int(pos[0] + vel[0]), int(pos[1] + vel[1]))
                
                # Ensure end position is within screen bounds
                end_pos = (
                    max(0, min(config.SCREEN_WIDTH - 1, end_pos[0])),
                    max(0, min(config.SCREEN_HEIGHT - 1, end_pos[1]))
                )
                
                # Color based on velocity magnitude
                vel_mag = np.linalg.norm(velocities[i])
                color_intensity = min(255, int(vel_mag * 50))
                vector_color = (color_intensity, 255 - color_intensity, 100)
                
                pygame.draw.line(self.surface, vector_color, pos, end_pos, 2)
                
                # Draw arrowhead
                self._draw_arrow_head(pos, end_pos, vector_color)
    
    def _draw_arrow_head(self, start, end, color):
        """Draw an arrowhead at the end of a vector."""
        if start == end:
            return
        
        # Calculate direction vector
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.sqrt(dx*dx + dy*dy)
        
        if length < 5:
            return
        
        # Normalize direction
        dx /= length
        dy /= length
        
        # Create arrowhead points
        arrow_length = 8
        arrow_angle = 0.5
        
        # Left point
        left_x = end[0] - arrow_length * (dx * math.cos(arrow_angle) + dy * math.sin(arrow_angle))
        left_y = end[1] - arrow_length * (dy * math.cos(arrow_angle) - dx * math.sin(arrow_angle))
        
        # Right point
        right_x = end[0] - arrow_length * (dx * math.cos(arrow_angle) - dy * math.sin(arrow_angle))
        right_y = end[1] - arrow_length * (dy * math.cos(arrow_angle) + dx * math.sin(arrow_angle))
        
        # Draw arrowhead
        pygame.draw.polygon(self.surface, color, [
            end, 
            (int(left_x), int(left_y)), 
            (int(right_x), int(right_y))
        ])
    
    def _draw_particle_info(self, particles):
        """Draw detailed information for particles."""
        # Only show info for particles near the mouse cursor or a few selected ones
        mouse_pos = pygame.mouse.get_pos()
        
        for i in range(min(5, len(particles))):  # Limit to first 5 particles to avoid clutter
            particle = particles[i]
            pos = (int(particle[0]), int(particle[1]))
            
            # Check if mouse is near this particle
            distance = math.sqrt((pos[0] - mouse_pos[0])**2 + (pos[1] - mouse_pos[1])**2)
            
            if distance < 50:  # Show info if mouse is within 50 pixels
                info_text = [
                    f"P{i}",
                    f"M:{particle[4]:.2f}",
                    f"C:{particle[6]:.2f}",
                    f"V:{np.linalg.norm(particle[2:4]):.1f}"
                ]
                
                # Draw info box
                for j, text in enumerate(info_text):
                    text_surface = self.font_small.render(text, True, (255, 255, 255))
                    bg_rect = text_surface.get_rect()
                    bg_rect.topleft = (pos[0] + 15, pos[1] + j * 18 - 10)
                    
                    # Draw background
                    pygame.draw.rect(self.surface, (0, 0, 0, 180), bg_rect.inflate(4, 2))
                    self.surface.blit(text_surface, bg_rect.topleft)
    
    def _draw_energy_info(self, particles):
        """Draw system energy information."""
        if len(particles) == 0:
            return
        
        # Calculate kinetic energy
        velocities = particles[:, 2:4]
        masses = particles[:, 4]
        kinetic_energy = 0.5 * np.sum(masses * np.sum(velocities**2, axis=1))
        
        # Calculate potential energy (simplified)
        potential_energy = 0.0
        positions = particles[:, 0:2]
        
        for i in range(len(particles)):
            for j in range(i + 1, len(particles)):
                r = np.linalg.norm(positions[i] - positions[j])
                if r > 0:
                    # Gravitational potential energy
                    potential_energy -= config.G * masses[i] * masses[j] / r
                    
                    # Electromagnetic potential energy
                    charges = particles[:, 6]
                    potential_energy += config.K_E * charges[i] * charges[j] / r
        
        total_energy = kinetic_energy + potential_energy
        
        # Draw energy information
        energy_info = [
            f"Kinetic: {kinetic_energy:.1f}",
            f"Potential: {potential_energy:.1f}",
            f"Total: {total_energy:.1f}"
        ]
        
        y_offset = config.SCREEN_HEIGHT - 80
        for i, info in enumerate(energy_info):
            text = self.font_small.render(info, True, config.UI_FONT_COLOR)
            bg_rect = text.get_rect()
            bg_rect.topleft = (10, y_offset + i * 20)
            
            # Draw semi-transparent background
            pygame.draw.rect(self.surface, (0, 0, 0, 128), bg_rect.inflate(6, 4))
            self.surface.blit(text, bg_rect.topleft)
    
    def _draw_main_ui(self, fps, particle_count):
        """Draw main UI information."""
        # FPS and particle count
        fps_text = self.font_medium.render(f"FPS: {int(fps)}", True, config.UI_FONT_COLOR)
        count_text = self.font_medium.render(f"Particles: {particle_count}", True, config.UI_FONT_COLOR)
        
        # Controls help
        controls = [
            "[P] Pause  [R] Reset  [T] Trails",
            "[F] Forces  [A] Agent  [Space] Train",
            "[G] Grid  [V] Vectors  [I] Info"
        ]
        
        # Draw main info
        self.surface.blit(fps_text, (10, 10))
        self.surface.blit(count_text, (10, 35))
        
        # Draw controls help
        for i, control in enumerate(controls):
            text = self.font_small.render(control, True, (150, 150, 170))
            self.surface.blit(text, (10, 70 + i * 18))
    
    def toggle_velocity_vectors(self):
        """Toggle velocity vector display."""
        self.show_velocity_vectors = not self.show_velocity_vectors
    
    def toggle_particle_info(self):
        """Toggle particle information display."""
        self.show_particle_info = not self.show_particle_info
    
    def toggle_grid(self):
        """Toggle background grid display."""
        self.show_grid = not self.show_grid
    
    def toggle_center_of_mass(self):
        """Toggle center of mass display."""
        self.show_center_of_mass = not self.show_center_of_mass
    
    def toggle_energy_info(self):
        """Toggle energy information display."""
        self.show_energy_info = not self.show_energy_info
    
    def toggle_particle_glow(self):
        """Toggle particle glow effect."""
        self.particle_glow = not self.particle_glow
    
    def set_detail_level(self, level):
        """Set rendering detail level (0.1 to 1.0)."""
        self.detail_level = max(0.1, min(1.0, level))