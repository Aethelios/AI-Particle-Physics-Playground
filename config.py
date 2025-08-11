# Enhanced Configuration System with validation and dynamic updates
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple
import numpy as np

@dataclass
class PhysicsConfig:
    """Physics simulation parameters."""
    G: float = 6.674e-2  # Gravitational constant
    K_E: float = 8.99e-3  # Coulomb's constant
    MIN_MASS: float = 1.0
    MAX_MASS: float = 5.0
    MIN_RADIUS: float = 2
    MAX_RADIUS: float = 6
    MIN_CHARGE: float = -2.0
    MAX_CHARGE: float = 2.0
    RESTITUTION_COEFFICIENT: float = 0.8
    MIN_COLLISION_DISTANCE: float = 1e-3
    USE_VERLET: bool = False
    PARTICLE_COUNT: int = 100
    
    def validate(self):
        """Validate physics parameters."""
        errors = []
        
        if self.G < 0:
            errors.append("Gravitational constant cannot be negative")
        if self.MIN_MASS <= 0 or self.MAX_MASS <= 0:
            errors.append("Mass values must be positive")
        if self.MIN_MASS >= self.MAX_MASS:
            errors.append("MIN_MASS must be less than MAX_MASS")
        if self.MIN_RADIUS <= 0 or self.MAX_RADIUS <= 0:
            errors.append("Radius values must be positive")
        if self.MIN_RADIUS >= self.MAX_RADIUS:
            errors.append("MIN_RADIUS must be less than MAX_RADIUS")
        if not 0 <= self.RESTITUTION_COEFFICIENT <= 1:
            errors.append("Restitution coefficient must be between 0 and 1")
        if self.PARTICLE_COUNT <= 0:
            errors.append("Particle count must be positive")
        if self.PARTICLE_COUNT > 1000:
            errors.append("Particle count too high (max 1000)")
            
        return errors

@dataclass
class DisplayConfig:
    """Display and rendering parameters."""
    SCREEN_WIDTH: int = 1280
    SCREEN_HEIGHT: int = 720
    FPS: int = 60
    BACKGROUND_COLOR: Tuple[int, int, int] = (10, 10, 25)
    PARTICLE_COLOR_LOW_MASS: Tuple[int, int, int] = (255, 255, 255)
    PARTICLE_COLOR_HIGH_MASS: Tuple[int, int, int] = (255, 100, 100)
    UI_FONT_COLOR: Tuple[int, int, int] = (200, 200, 220)
    UI_FONT_SIZE: int = 24
    
    def validate(self):
        """Validate display parameters."""
        errors = []
        
        if self.SCREEN_WIDTH < 640 or self.SCREEN_HEIGHT < 480:
            errors.append("Screen resolution too small (minimum 640x480)")
        if self.SCREEN_WIDTH > 3840 or self.SCREEN_HEIGHT > 2160:
            errors.append("Screen resolution too large (maximum 3840x2160)")
        if not 30 <= self.FPS <= 120:
            errors.append("FPS must be between 30 and 120")
        if self.UI_FONT_SIZE < 12 or self.UI_FONT_SIZE > 48:
            errors.append("UI font size must be between 12 and 48")
            
        return errors

@dataclass
class AIConfig:
    """AI and reinforcement learning parameters."""
    AGENT_PARTICLE_INDEX: int = 0
    STATE_SIZE: int = 22
    ACTION_SIZE: int = 5
    AGENT_FORCE_MAGNITUDE: float = 500.0
    
    # DQN Hyperparameters
    BUFFER_SIZE: int = 100000
    BATCH_SIZE: int = 64
    GAMMA: float = 0.99
    LR: float = 5e-4
    TAU: float = 1e-3
    UPDATE_EVERY: int = 4
    
    # Training parameters
    MAX_EPISODES: int = 2000
    MAX_STEPS_PER_EPISODE: int = 1000
    EPS_START: float = 1.0
    EPS_END: float = 0.01
    EPS_DECAY: float = 0.995
    TARGET_SCORE: float = 200.0
    
    def validate(self):
        """Validate AI parameters."""
        errors = []
        
        if self.AGENT_PARTICLE_INDEX < 0:
            errors.append("Agent particle index cannot be negative")
        if self.STATE_SIZE <= 0 or self.ACTION_SIZE <= 0:
            errors.append("State and action sizes must be positive")
        if not 0 < self.GAMMA < 1:
            errors.append("Gamma must be between 0 and 1")
        if self.LR <= 0 or self.LR > 1:
            errors.append("Learning rate must be between 0 and 1")
        if not 0 <= self.EPS_END <= self.EPS_START <= 1:
            errors.append("Invalid epsilon values")
        if self.BATCH_SIZE <= 0 or self.BATCH_SIZE > self.BUFFER_SIZE:
            errors.append("Invalid batch size")
            
        return errors

class ConfigManager:
    """Manages all configuration settings."""
    
    def __init__(self, config_file='config.json'):
        self.config_file = config_file
        self.physics = PhysicsConfig()
        self.display = DisplayConfig()
        self.ai = AIConfig()
        self.presets = {}
        
        self.load_config()
        self.load_presets()
    
    def load_config(self):
        """Load configuration from file."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                
                # Update configurations
                if 'physics' in data:
                    for key, value in data['physics'].items():
                        if hasattr(self.physics, key):
                            setattr(self.physics, key, value)
                
                if 'display' in data:
                    for key, value in data['display'].items():
                        if hasattr(self.display, key):
                            setattr(self.display, key, value)
                
                if 'ai' in data:
                    for key, value in data['ai'].items():
                        if hasattr(self.ai, key):
                            setattr(self.ai, key, value)
                            
            except Exception as e:
                print(f"Error loading config: {e}")
    
    def save_config(self):
        """Save current configuration to file."""
        try:
            config_data = {
                'physics': asdict(self.physics),
                'display': asdict(self.display),
                'ai': asdict(self.ai)
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
                
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def load_presets(self):
        """Load configuration presets."""
        self.presets = {
            'Default': {
                'physics': {
                    'G': 6.674e-2,
                    'K_E': 8.99e-3,
                    'PARTICLE_COUNT': 100,
                    'RESTITUTION_COEFFICIENT': 0.8,
                    'USE_VERLET': False
                }
            },
            'High Gravity': {
                'physics': {
                    'G': 0.15,
                    'K_E': 0.001,
                    'PARTICLE_COUNT': 80,
                    'RESTITUTION_COEFFICIENT': 0.6,
                    'USE_VERLET': True
                }
            },
            'Electromagnetic Chaos': {
                'physics': {
                    'G': 0.01,
                    'K_E': 0.02,
                    'PARTICLE_COUNT': 150,
                    'RESTITUTION_COEFFICIENT': 0.9,
                    'MIN_CHARGE': -5.0,
                    'MAX_CHARGE': 5.0
                }
            },
            'Orbital Mechanics': {
                'physics': {
                    'G': 0.1,
                    'K_E': 0.0,
                    'PARTICLE_COUNT': 50,
                    'RESTITUTION_COEFFICIENT': 1.0,
                    'MIN_MASS': 0.5,
                    'MAX_MASS': 10.0,
                    'USE_VERLET': True
                }
            },
            'Quantum Simulation': {
                'physics': {
                    'G': 0.001,
                    'K_E': 0.05,
                    'PARTICLE_COUNT': 200,
                    'RESTITUTION_COEFFICIENT': 1.0,
                    'MIN_RADIUS': 1,
                    'MAX_RADIUS': 3
                }
            },
            'AI Training Optimized': {
                'physics': {
                    'G': 0.05,
                    'K_E': 0.01,
                    'PARTICLE_COUNT': 75,
                    'RESTITUTION_COEFFICIENT': 0.7
                },
                'ai': {
                    'AGENT_FORCE_MAGNITUDE': 750.0,
                    'BATCH_SIZE': 32,
                    'LR': 1e-3
                }
            }
        }
        
        # Load additional presets from file if exists
        preset_file = 'presets.json'
        if os.path.exists(preset_file):
            try:
                with open(preset_file, 'r') as f:
                    user_presets = json.load(f)
                    self.presets.update(user_presets)
            except Exception as e:
                print(f"Error loading presets: {e}")
    
    def save_presets(self):
        """Save presets to file."""
        try:
            with open('presets.json', 'w') as f:
                json.dump(self.presets, f, indent=2)
        except Exception as e:
            print(f"Error saving presets: {e}")
    
    def apply_preset(self, preset_name):
        """Apply a configuration preset."""
        if preset_name not in self.presets:
            return False
        
        preset = self.presets[preset_name]
        
        # Apply physics settings
        if 'physics' in preset:
            for key, value in preset['physics'].items():
                if hasattr(self.physics, key):
                    setattr(self.physics, key, value)
        
        # Apply display settings
        if 'display' in preset:
            for key, value in preset['display'].items():
                if hasattr(self.display, key):
                    setattr(self.display, key, value)
        
        # Apply AI settings
        if 'ai' in preset:
            for key, value in preset['ai'].items():
                if hasattr(self.ai, key):
                    setattr(self.ai, key, value)
        
        return True
    
    def create_preset(self, name):
        """Create a new preset from current settings."""
        self.presets[name] = {
            'physics': asdict(self.physics),
            'display': asdict(self.display),
            'ai': asdict(self.ai)
        }
        self.save_presets()
    
    def validate_all(self):
        """Validate all configurations."""
        all_errors = []
        
        physics_errors = self.physics.validate()
        if physics_errors:
            all_errors.extend([f"Physics: {error}" for error in physics_errors])
        
        display_errors = self.display.validate()
        if display_errors:
            all_errors.extend([f"Display: {error}" for error in display_errors])
        
        ai_errors = self.ai.validate()
        if ai_errors:
            all_errors.extend([f"AI: {error}" for error in ai_errors])
        
        return all_errors
    
    def get_physics_summary(self):
        """Get a summary of current physics settings."""
        return {
            'Gravity': f"{self.physics.G:.4f}",
            'EM Force': f"{self.physics.K_E:.4f}",
            'Particles': f"{self.physics.PARTICLE_COUNT}",
            'Mass Range': f"{self.physics.MIN_MASS:.1f}-{self.physics.MAX_MASS:.1f}",
            'Charge Range': f"{self.physics.MIN_CHARGE:.1f}-{self.physics.MAX_CHARGE:.1f}",
            'Bounciness': f"{self.physics.RESTITUTION_COEFFICIENT:.2f}",
            'Integration': "Verlet" if self.physics.USE_VERLET else "Euler"
        }
    
    def get_performance_settings(self):
        """Get recommended settings for different performance levels."""
        return {
            'Low Performance': {
                'PARTICLE_COUNT': 50,
                'FPS': 30,
                'USE_VERLET': False,
                'detail_level': 0.5
            },
            'Medium Performance': {
                'PARTICLE_COUNT': 100,
                'FPS': 60,
                'USE_VERLET': False,
                'detail_level': 0.8
            },
            'High Performance': {
                'PARTICLE_COUNT': 200,
                'FPS': 60,
                'USE_VERLET': True,
                'detail_level': 1.0
            },
            'Ultra Performance': {
                'PARTICLE_COUNT': 500,
                'FPS': 120,
                'USE_VERLET': True,
                'detail_level': 1.0
            }
        }
    
    def auto_tune_performance(self, target_fps=60):
        """Automatically adjust settings for target FPS."""
        current_fps = target_fps  # This would come from the actual measurement
        
        if current_fps < target_fps * 0.8:  # If performance is poor
            # Reduce particle count
            self.physics.PARTICLE_COUNT = max(25, int(self.physics.PARTICLE_COUNT * 0.8))
            # Use simpler integration
            self.physics.USE_VERLET = False
            # Reduce visual quality
            self.display.FPS = max(30, int(self.display.FPS * 0.9))
            
        elif current_fps > target_fps * 1.2:  # If performance is good
            # Increase particle count
            self.physics.PARTICLE_COUNT = min(500, int(self.physics.PARTICLE_COUNT * 1.1))
            # Use better integration
            self.physics.USE_VERLET = True
            # Increase visual quality
            self.display.FPS = min(120, int(self.display.FPS * 1.05))
    
    def reset_to_defaults(self):
        """Reset all settings to default values."""
        self.physics = PhysicsConfig()
        self.display = DisplayConfig()
        self.ai = AIConfig()

# Create global config manager instance
config_manager = ConfigManager()

# Export commonly used values for backward compatibility
SCREEN_WIDTH = config_manager.display.SCREEN_WIDTH
SCREEN_HEIGHT = config_manager.display.SCREEN_HEIGHT
FPS = config_manager.display.FPS
BACKGROUND_COLOR = config_manager.display.BACKGROUND_COLOR
PARTICLE_COLOR_LOW_MASS = config_manager.display.PARTICLE_COLOR_LOW_MASS
PARTICLE_COLOR_HIGH_MASS = config_manager.display.PARTICLE_COLOR_HIGH_MASS
UI_FONT_COLOR = config_manager.display.UI_FONT_COLOR
UI_FONT_SIZE = config_manager.display.UI_FONT_SIZE

PARTICLE_COUNT = config_manager.physics.PARTICLE_COUNT
MIN_MASS = config_manager.physics.MIN_MASS
MAX_MASS = config_manager.physics.MAX_MASS
MIN_RADIUS = config_manager.physics.MIN_RADIUS
MAX_RADIUS = config_manager.physics.MAX_RADIUS
MIN_CHARGE = config_manager.physics.MIN_CHARGE
MAX_CHARGE = config_manager.physics.MAX_CHARGE
G = config_manager.physics.G
K_E = config_manager.physics.K_E
RESTITUTION_COEFFICIENT = config_manager.physics.RESTITUTION_COEFFICIENT
MIN_COLLISION_DISTANCE = config_manager.physics.MIN_COLLISION_DISTANCE
USE_VERLET = config_manager.physics.USE_VERLET

AGENT_PARTICLE_INDEX = config_manager.ai.AGENT_PARTICLE_INDEX
STATE_SIZE = config_manager.ai.STATE_SIZE
ACTION_SIZE = config_manager.ai.ACTION_SIZE
AGENT_FORCE_MAGNITUDE = config_manager.ai.AGENT_FORCE_MAGNITUDE
BUFFER_SIZE = config_manager.ai.BUFFER_SIZE
BATCH_SIZE = config_manager.ai.BATCH_SIZE
GAMMA = config_manager.ai.GAMMA
LR = config_manager.ai.LR
TAU = config_manager.ai.TAU
UPDATE_EVERY = config_manager.ai.UPDATE_EVERY

def update_config_values():
    """Update global variables from config manager (call after config changes)."""
    global SCREEN_WIDTH, SCREEN_HEIGHT, FPS, BACKGROUND_COLOR
    global PARTICLE_COLOR_LOW_MASS, PARTICLE_COLOR_HIGH_MASS, UI_FONT_COLOR, UI_FONT_SIZE
    global PARTICLE_COUNT, MIN_MASS, MAX_MASS, MIN_RADIUS, MAX_RADIUS
    global MIN_CHARGE, MAX_CHARGE, G, K_E, RESTITUTION_COEFFICIENT
    global MIN_COLLISION_DISTANCE, USE_VERLET
    global AGENT_PARTICLE_INDEX, STATE_SIZE, ACTION_SIZE, AGENT_FORCE_MAGNITUDE
    global BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY
    
    # Display settings
    SCREEN_WIDTH = config_manager.display.SCREEN_WIDTH
    SCREEN_HEIGHT = config_manager.display.SCREEN_HEIGHT
    FPS = config_manager.display.FPS
    BACKGROUND_COLOR = config_manager.display.BACKGROUND_COLOR
    PARTICLE_COLOR_LOW_MASS = config_manager.display.PARTICLE_COLOR_LOW_MASS
    PARTICLE_COLOR_HIGH_MASS = config_manager.display.PARTICLE_COLOR_HIGH_MASS
    UI_FONT_COLOR = config_manager.display.UI_FONT_COLOR
    UI_FONT_SIZE = config_manager.display.UI_FONT_SIZE
    
    # Physics settings
    PARTICLE_COUNT = config_manager.physics.PARTICLE_COUNT
    MIN_MASS = config_manager.physics.MIN_MASS
    MAX_MASS = config_manager.physics.MAX_MASS
    MIN_RADIUS = config_manager.physics.MIN_RADIUS
    MAX_RADIUS = config_manager.physics.MAX_RADIUS
    MIN_CHARGE = config_manager.physics.MIN_CHARGE
    MAX_CHARGE = config_manager.physics.MAX_CHARGE
    G = config_manager.physics.G
    K_E = config_manager.physics.K_E
    RESTITUTION_COEFFICIENT = config_manager.physics.RESTITUTION_COEFFICIENT
    MIN_COLLISION_DISTANCE = config_manager.physics.MIN_COLLISION_DISTANCE
    USE_VERLET = config_manager.physics.USE_VERLET
    
    # AI settings
    AGENT_PARTICLE_INDEX = config_manager.ai.AGENT_PARTICLE_INDEX
    STATE_SIZE = config_manager.ai.STATE_SIZE
    ACTION_SIZE = config_manager.ai.ACTION_SIZE
    AGENT_FORCE_MAGNITUDE = config_manager.ai.AGENT_FORCE_MAGNITUDE
    BUFFER_SIZE = config_manager.ai.BUFFER_SIZE
    BATCH_SIZE = config_manager.ai.BATCH_SIZE
    GAMMA = config_manager.ai.GAMMA
    LR = config_manager.ai.LR
    TAU = config_manager.ai.TAU
    UPDATE_EVERY = config_manager.ai.UPDATE_EVERY