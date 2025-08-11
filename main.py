import sys
import pygame
import pygame_gui
import config
from physics import PhysicsEngine
from render import Renderer
from environment import ParticleEnv
from rl_agent import Agent
import numpy as np
import json
import os

class PhysicsPlayground:
    def __init__(self):
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((config.SCREEN_WIDTH + 300, config.SCREEN_HEIGHT))
        pygame.display.set_caption("AI Particle Physics Playground")
        self.clock = pygame.time.Clock()
        
        # Create simulation surface (left side)
        self.sim_surface = pygame.Surface((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
        
        # Initialize GUI manager for the control panel
        self.manager = pygame_gui.UIManager((config.SCREEN_WIDTH + 300, config.SCREEN_HEIGHT))
        
        # Initialize core components
        self.engine = PhysicsEngine(config.PARTICLE_COUNT)
        self.renderer = Renderer(self.sim_surface)
        self.env = ParticleEnv()
        self.agent = Agent(config.STATE_SIZE, config.ACTION_SIZE)
        
        # State variables
        self.running = True
        self.paused = False
        self.show_trails = False
        self.trail_surface = pygame.Surface((config.SCREEN_WIDTH, config.SCREEN_HEIGHT), pygame.SRCALPHA)
        self.show_forces = False
        self.show_agent = False
        self.training_mode = False
        self.episode_count = 0
        self.step_count = 0
        self.total_reward = 0.0
        self.eps = 1.0
        
        # Performance tracking
        self.fps_history = []
        self.particle_count_history = []
        
        self.setup_gui()
        self.load_presets()
        
    def setup_gui(self):
        """Setup the GUI control panel"""
        panel_x = config.SCREEN_WIDTH + 10
        y_offset = 10
        
        # Title
        self.title_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(panel_x, y_offset, 280, 30),
            text='Physics Control Panel',
            manager=self.manager
        )
        y_offset += 40
        
        # Simulation Controls
        self.sim_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(panel_x, y_offset, 280, 25),
            text='Simulation Controls',
            manager=self.manager
        )
        y_offset += 30
        
        self.pause_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(panel_x, y_offset, 90, 30),
            text='Pause',
            manager=self.manager
        )
        
        self.reset_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(panel_x + 95, y_offset, 90, 30),
            text='Reset',
            manager=self.manager
        )
        
        self.step_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(panel_x + 190, y_offset, 90, 30),
            text='Step',
            manager=self.manager
        )
        y_offset += 40
        
        # Physics Parameters
        self.physics_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(panel_x, y_offset, 280, 25),
            text='Physics Parameters',
            manager=self.manager
        )
        y_offset += 30
        
        # Gravity slider
        self.gravity_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(panel_x, y_offset, 140, 25),
            text=f'Gravity: {config.G:.3f}',
            manager=self.manager
        )
        
        self.gravity_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect(panel_x + 145, y_offset, 135, 25),
            start_value=config.G,
            value_range=(0.0, 0.2),
            manager=self.manager
        )
        y_offset += 35
        
        # Electromagnetic constant slider
        self.em_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(panel_x, y_offset, 140, 25),
            text=f'EM Force: {config.K_E:.3f}',
            manager=self.manager
        )
        
        self.em_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect(panel_x + 145, y_offset, 135, 25),
            start_value=config.K_E,
            value_range=(0.0, 0.02),
            manager=self.manager
        )
        y_offset += 35
        
        # Particle count slider
        self.count_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(panel_x, y_offset, 140, 25),
            text=f'Particles: {config.PARTICLE_COUNT}',
            manager=self.manager
        )
        
        self.count_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect(panel_x + 145, y_offset, 135, 25),
            start_value=config.PARTICLE_COUNT,
            value_range=(10, 500),
            manager=self.manager
        )
        y_offset += 35
        
        # Restitution coefficient
        self.restitution_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(panel_x, y_offset, 140, 25),
            text=f'Bounciness: {config.RESTITUTION_COEFFICIENT:.2f}',
            manager=self.manager
        )
        
        self.restitution_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect(panel_x + 145, y_offset, 135, 25),
            start_value=config.RESTITUTION_COEFFICIENT,
            value_range=(0.0, 1.0),
            manager=self.manager
        )
        y_offset += 45
        
        # Visual Options
        self.visual_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(panel_x, y_offset, 280, 25),
            text='Visual Options',
            manager=self.manager
        )
        y_offset += 30
        
        self.trails_checkbox = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(panel_x, y_offset, 90, 30),
            text='Trails: OFF',
            manager=self.manager
        )
        
        self.forces_checkbox = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(panel_x + 95, y_offset, 90, 30),
            text='Forces: OFF',
            manager=self.manager
        )
        y_offset += 40
        
        # AI Controls
        self.ai_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(panel_x, y_offset, 280, 25),
            text='AI Controls',
            manager=self.manager
        )
        y_offset += 30
        
        self.agent_checkbox = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(panel_x, y_offset, 90, 30),
            text='Agent: OFF',
            manager=self.manager
        )
        
        self.train_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(panel_x + 95, y_offset, 90, 30),
            text='Train',
            manager=self.manager
        )
        
        self.save_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(panel_x + 190, y_offset, 90, 30),
            text='Save AI',
            manager=self.manager
        )
        y_offset += 40
        
        # AI Info
        self.episode_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(panel_x, y_offset, 280, 25),
            text=f'Episode: {self.episode_count}',
            manager=self.manager
        )
        y_offset += 25
        
        self.reward_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(panel_x, y_offset, 280, 25),
            text=f'Reward: {self.total_reward:.2f}',
            manager=self.manager
        )
        y_offset += 25
        
        self.epsilon_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(panel_x, y_offset, 280, 25),
            text=f'Exploration: {self.eps:.3f}',
            manager=self.manager
        )
        y_offset += 35
        
        # Presets
        self.preset_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(panel_x, y_offset, 280, 25),
            text='Presets',
            manager=self.manager
        )
        y_offset += 30
        
        self.preset_dropdown = pygame_gui.elements.UIDropDownMenu(
            relative_rect=pygame.Rect(panel_x, y_offset, 180, 30),
            options_list=['Default', 'High Gravity', 'Electromagnetic', 'Chaos Mode'],
            starting_option='Default',
            manager=self.manager
        )
        
        self.save_preset_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(panel_x + 185, y_offset, 95, 30),
            text='Save Preset',
            manager=self.manager
        )
        y_offset += 40
        
        # Performance Info
        self.performance_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(panel_x, y_offset, 280, 25),
            text='Performance',
            manager=self.manager
        )
        y_offset += 30
        
        self.fps_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(panel_x, y_offset, 280, 25),
            text='FPS: 0',
            manager=self.manager
        )
        y_offset += 25
        
        self.performance_info_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(panel_x, y_offset, 280, 50),
            text='Avg FPS: 0\nCollisions: 0',
            manager=self.manager
        )
    
    def load_presets(self):
        """Load presets from file if it exists"""
        self.presets = {
            'Default': {
                'G': 6.674e-2,
                'K_E': 8.99e-3,
                'PARTICLE_COUNT': 100,
                'RESTITUTION_COEFFICIENT': 0.8
            },
            'High Gravity': {
                'G': 0.15,
                'K_E': 0.001,
                'PARTICLE_COUNT': 80,
                'RESTITUTION_COEFFICIENT': 0.6
            },
            'Electromagnetic': {
                'G': 0.01,
                'K_E': 0.02,
                'PARTICLE_COUNT': 150,
                'RESTITUTION_COEFFICIENT': 0.9
            },
            'Chaos Mode': {
                'G': 0.1,
                'K_E': 0.015,
                'PARTICLE_COUNT': 200,
                'RESTITUTION_COEFFICIENT': 1.0
            }
        }
        
        if os.path.exists('presets.json'):
            try:
                with open('presets.json', 'r') as f:
                    saved_presets = json.load(f)
                    self.presets.update(saved_presets)
            except:
                pass
    
    def apply_preset(self, preset_name):
        """Apply a preset configuration"""
        if preset_name in self.presets:
            preset = self.presets[preset_name]
            config.G = preset['G']
            config.K_E = preset['K_E']
            config.PARTICLE_COUNT = preset['PARTICLE_COUNT']
            config.RESTITUTION_COEFFICIENT = preset['RESTITUTION_COEFFICIENT']
            
            # Update sliders
            self.gravity_slider.set_current_value(config.G)
            self.em_slider.set_current_value(config.K_E)
            self.count_slider.set_current_value(config.PARTICLE_COUNT)
            self.restitution_slider.set_current_value(config.RESTITUTION_COEFFICIENT)
            
            # Reset simulation with new parameters
            self.engine = PhysicsEngine(config.PARTICLE_COUNT)
            self.env = ParticleEnv()
    
    def handle_gui_events(self, event):
        """Handle GUI events"""
        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == self.pause_button:
                self.paused = not self.paused
                self.pause_button.set_text('Resume' if self.paused else 'Pause')
            
            elif event.ui_element == self.reset_button:
                self.engine = PhysicsEngine(config.PARTICLE_COUNT)
                self.env = ParticleEnv()
                self.trail_surface.fill((0, 0, 0, 0))
                self.episode_count = 0
                self.step_count = 0
                self.total_reward = 0.0
                self.eps = 1.0
                self.training_mode = False
            
            elif event.ui_element == self.step_button:
                if self.paused:
                    dt = 1 / config.FPS
                    self.engine.update(dt)
            
            elif event.ui_element == self.trails_checkbox:
                self.show_trails = not self.show_trails
                self.trails_checkbox.set_text(f'Trails: {"ON" if self.show_trails else "OFF"}')
                if not self.show_trails:
                    self.trail_surface.fill((0, 0, 0, 0))
            
            elif event.ui_element == self.forces_checkbox:
                self.show_forces = not self.show_forces
                self.forces_checkbox.set_text(f'Forces: {"ON" if self.show_forces else "OFF"}')
            
            elif event.ui_element == self.agent_checkbox:
                self.show_agent = not self.show_agent
                self.agent_checkbox.set_text(f'Agent: {"ON" if self.show_agent else "OFF"}')
            
            elif event.ui_element == self.train_button:
                self.training_mode = not self.training_mode
                self.train_button.set_text('Stop Train' if self.training_mode else 'Train')
                if self.training_mode:
                    self.show_agent = True
                    self.agent_checkbox.set_text('Agent: ON')
            
            elif event.ui_element == self.save_button:
                self.agent.save('trained_model.pth')
                print("Model saved as 'trained_model.pth'")
            
            elif event.ui_element == self.save_preset_button:
                self.save_current_preset()
        
        elif event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
            if event.ui_element == self.gravity_slider:
                config.G = event.value
                self.gravity_label.set_text(f'Gravity: {config.G:.3f}')
            
            elif event.ui_element == self.em_slider:
                config.K_E = event.value
                self.em_label.set_text(f'EM Force: {config.K_E:.3f}')
            
            elif event.ui_element == self.count_slider:
                new_count = int(event.value)
                if new_count != config.PARTICLE_COUNT:
                    config.PARTICLE_COUNT = new_count
                    self.count_label.set_text(f'Particles: {config.PARTICLE_COUNT}')
                    self.engine = PhysicsEngine(config.PARTICLE_COUNT)
                    self.env = ParticleEnv()
            
            elif event.ui_element == self.restitution_slider:
                config.RESTITUTION_COEFFICIENT = event.value
                self.restitution_label.set_text(f'Bounciness: {config.RESTITUTION_COEFFICIENT:.2f}')
        
        elif event.type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED:
            if event.ui_element == self.preset_dropdown:
                self.apply_preset(event.text)
    
    def save_current_preset(self):
        """Save current configuration as a preset"""
        preset_name = f"Custom_{len(self.presets)}"
        self.presets[preset_name] = {
            'G': config.G,
            'K_E': config.K_E,
            'PARTICLE_COUNT': config.PARTICLE_COUNT,
            'RESTITUTION_COEFFICIENT': config.RESTITUTION_COEFFICIENT
        }
        
        with open('presets.json', 'w') as f:
            json.dump(self.presets, f, indent=2)
        
        print(f"Preset saved as '{preset_name}'")
    
    def update_ai_training(self):
        """Handle AI training loop"""
        if not self.training_mode:
            return
        
        if self.step_count == 0:
            self.current_state = self.env.reset()
            self.episode_reward = 0.0
        
        # Agent selects action
        action = self.agent.act(self.current_state, self.eps)
        
        # Environment step
        next_state, reward, done = self.env.step(action)
        
        # Store experience
        self.agent.step(self.current_state, action, reward, next_state, done)
        
        self.current_state = next_state
        self.episode_reward += reward
        self.step_count += 1
        
        if done or self.step_count > 1000:
            self.episode_count += 1
            self.total_reward = self.episode_reward
            self.step_count = 0
            self.eps = max(0.01, self.eps * 0.995)  # Decay epsilon
    
    def update_gui_info(self):
        """Update GUI information displays"""
        current_fps = self.clock.get_fps()
        self.fps_history.append(current_fps)
        if len(self.fps_history) > 60:  # Keep last 60 frames
            self.fps_history.pop(0)
        
        avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
        
        # Count collisions (simplified)
        collision_count = 0
        particles = self.engine.particles
        for i in range(len(particles)):
            for j in range(i + 1, len(particles)):
                dist_sq = np.sum((particles[i][:2] - particles[j][:2])**2)
                radii_sum_sq = (particles[i][5] + particles[j][5])**2
                if dist_sq < radii_sum_sq:
                    collision_count += 1
        
        # Update labels
        self.fps_label.set_text(f'FPS: {int(current_fps)}')
        self.performance_info_label.set_text(f'Avg FPS: {int(avg_fps)}\nCollisions: {collision_count}')
        self.episode_label.set_text(f'Episode: {self.episode_count}')
        self.reward_label.set_text(f'Reward: {self.total_reward:.2f}')
        self.epsilon_label.set_text(f'Exploration: {self.eps:.3f}')
    
    def draw_forces(self):
        """Draw force vectors on particles"""
        if not self.show_forces:
            return
        
        # This is a simplified force visualization
        particles = self.engine.particles
        for i, particle in enumerate(particles):
            pos = (int(particle[0]), int(particle[1]))
            
            # Calculate net force (simplified)
            net_force = np.array([0.0, 0.0])
            for j, other in enumerate(particles):
                if i == j:
                    continue
                
                displacement = other[:2] - particle[:2]
                distance_sq = np.sum(displacement**2)
                if distance_sq < 1e-6:
                    continue
                
                distance = np.sqrt(distance_sq)
                direction = displacement / distance
                
                # Gravitational force
                grav_force = config.G * particle[4] * other[4] / distance_sq
                net_force += grav_force * direction
                
                # Electromagnetic force
                em_force = config.K_E * particle[6] * other[6] / distance_sq
                net_force += em_force * direction
            
            # Draw force vector
            if np.linalg.norm(net_force) > 0.1:  # Only draw significant forces
                force_end = pos + net_force * 1000  # Scale for visibility
                force_end = (int(force_end[0]), int(force_end[1]))
                
                if 0 <= force_end[0] < config.SCREEN_WIDTH and 0 <= force_end[1] < config.SCREEN_HEIGHT:
                    pygame.draw.line(self.sim_surface, (255, 255, 0), pos, force_end, 1)
    
    def draw_agent_highlight(self):
        """Highlight the agent particle"""
        if not self.show_agent:
            return
        
        agent_particle = self.engine.particles[config.AGENT_PARTICLE_INDEX]
        pos = (int(agent_particle[0]), int(agent_particle[1]))
        radius = int(agent_particle[5]) + 5
        
        # Draw a glowing ring around the agent
        pygame.draw.circle(self.sim_surface, (0, 255, 0), pos, radius, 2)
        pygame.draw.circle(self.sim_surface, (0, 255, 0, 100), pos, radius + 2, 1)
    
    def update_trails(self):
        """Update particle trails"""
        if not self.show_trails:
            return
        
        # Fade existing trails
        fade_surface = pygame.Surface((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
        fade_surface.set_alpha(5)  # Fade amount
        fade_surface.fill((0, 0, 0))
        self.trail_surface.blit(fade_surface, (0, 0))
        
        # Add new trail points
        particles = self.engine.particles
        for particle in particles:
            pos = (int(particle[0]), int(particle[1]))
            pygame.draw.circle(self.trail_surface, (100, 100, 255, 50), pos, 1)
    
    def run(self):
        """Main game loop"""
        while self.running:
            time_delta = self.clock.tick(config.FPS) / 1000.0
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                
                # Handle keyboard shortcuts
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
                        self.paused = not self.paused
                        self.pause_button.set_text('Resume' if self.paused else 'Pause')
                    elif event.key == pygame.K_r:
                        self.engine = PhysicsEngine(config.PARTICLE_COUNT)
                        self.env = ParticleEnv()
                        self.trail_surface.fill((0, 0, 0, 0))
                    elif event.key == pygame.K_t:
                        self.show_trails = not self.show_trails
                        self.trails_checkbox.set_text(f'Trails: {"ON" if self.show_trails else "OFF"}')
                    elif event.key == pygame.K_f:
                        self.show_forces = not self.show_forces
                        self.forces_checkbox.set_text(f'Forces: {"ON" if self.show_forces else "OFF"}')
                    elif event.key == pygame.K_a:
                        self.show_agent = not self.show_agent
                        self.agent_checkbox.set_text(f'Agent: {"ON" if self.show_agent else "OFF"}')
                    elif event.key == pygame.K_SPACE:
                        self.training_mode = not self.training_mode
                        self.train_button.set_text('Stop Train' if self.training_mode else 'Train')
                
                # Handle GUI events
                self.handle_gui_events(event)
                self.manager.process_events(event)
            
            # Update simulation
            if not self.paused:
                dt = 1 / config.FPS
                self.engine.update(dt)
                
                # Update AI training
                self.update_ai_training()
            
            # Update trails
            self.update_trails()
            
            # Update GUI info
            self.update_gui_info()
            self.manager.update(time_delta)
            
            # Draw everything
            self.screen.fill((30, 30, 30))  # Background for control panel
            
            # Draw simulation
            self.renderer.draw(self.engine.particles, self.clock.get_fps())
            
            # Draw additional visual elements
            if self.show_trails:
                self.sim_surface.blit(self.trail_surface, (0, 0))
            
            self.draw_forces()
            self.draw_agent_highlight()
            
            # Blit simulation surface to main screen
            self.screen.blit(self.sim_surface, (0, 0))
            
            # Draw GUI
            self.manager.draw_ui(self.screen)
            
            pygame.display.flip()
        
        pygame.quit()
        sys.exit()

if __name__ == '__main__':
    app = PhysicsPlayground()
    app.run()