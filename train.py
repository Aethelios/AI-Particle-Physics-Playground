import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import json
import time
import os
from datetime import datetime
import threading
import queue

from config import config_manager, update_config_values
from environment import ParticleEnv
from rl_agent import Agent

class TrainingMonitor:
    """Monitors and logs training progress."""
    
    def __init__(self, log_dir='training_logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.episode_scores = []
        self.episode_lengths = []
        self.loss_history = []
        self.epsilon_history = []
        self.collision_history = []
        self.training_start_time = None
        
        # Performance metrics
        self.best_score = float('-inf')
        self.best_avg_score = float('-inf')
        self.episodes_without_improvement = 0
        
        # Real-time plotting
        self.plot_queue = queue.Queue()
        self.plot_thread = None
        self.plotting_enabled = False
    
    def start_episode(self, episode_num, epsilon):
        """Called at the start of each episode."""
        if self.training_start_time is None:
            self.training_start_time = time.time()
        
        self.current_episode = episode_num
        self.current_epsilon = epsilon
        self.episode_start_time = time.time()
    
    def end_episode(self, score, steps, collisions=0):
        """Called at the end of each episode."""
        episode_time = time.time() - self.episode_start_time
        
        self.episode_scores.append(score)
        self.episode_lengths.append(steps)
        self.epsilon_history.append(self.current_epsilon)
        self.collision_history.append(collisions)
        
        # Update best scores
        if score > self.best_score:
            self.best_score = score
            self.episodes_without_improvement = 0
        else:
            self.episodes_without_improvement += 1
        
        # Calculate rolling average
        window_size = min(100, len(self.episode_scores))
        if window_size > 0:
            avg_score = np.mean(self.episode_scores[-window_size:])
            if avg_score > self.best_avg_score:
                self.best_avg_score = avg_score
        
        # Log progress
        if self.current_episode % 10 == 0:
            self.log_progress()
        
        # Update real-time plot
        if self.plotting_enabled:
            self.plot_queue.put({
                'episode': self.current_episode,
                'score': score,
                'avg_score': avg_score if window_size > 0 else score,
                'epsilon': self.current_epsilon
            })
    
    def log_progress(self):
        """Log current training progress."""
        if not self.episode_scores:
            return
        
        window_size = min(100, len(self.episode_scores))
        avg_score = np.mean(self.episode_scores[-window_size:])
        avg_length = np.mean(self.episode_lengths[-window_size:])
        
        elapsed_time = time.time() - self.training_start_time
        
        progress_msg = (
            f"Episode {self.current_episode:4d} | "
            f"Score: {self.episode_scores[-1]:7.2f} | "
            f"Avg Score: {avg_score:7.2f} | "
            f"Best: {self.best_score:7.2f} | "
            f"Avg Length: {avg_length:5.1f} | "
            f"ε: {self.current_epsilon:.3f} | "
            f"Time: {elapsed_time/60:.1f}m"
        )
        
        print(progress_msg)
    
    def save_training_data(self, filename=None):
        """Save training data to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.log_dir}/training_data_{timestamp}.json"
        
        data = {
            'episode_scores': self.episode_scores,
            'episode_lengths': self.episode_lengths,
            'epsilon_history': self.epsilon_history,
            'collision_history': self.collision_history,
            'best_score': self.best_score,
            'best_avg_score': self.best_avg_score,
            'training_duration': time.time() - self.training_start_time if self.training_start_time else 0,
            'config': {
                'physics': config_manager.physics.__dict__,
                'ai': config_manager.ai.__dict__
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Training data saved to {filename}")
    
    def create_plots(self, save_path=None):
        """Create comprehensive training plots."""
        if not self.episode_scores:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Progress Analysis', fontsize=16)
        
        episodes = range(1, len(self.episode_scores) + 1)
        
        # Score progression
        axes[0, 0].plot(episodes, self.episode_scores, alpha=0.7, label='Episode Score')
        if len(self.episode_scores) >= 100:
            rolling_avg = [np.mean(self.episode_scores[max(0, i-99):i+1]) 
                          for i in range(len(self.episode_scores))]
            axes[0, 0].plot(episodes, rolling_avg, 'r-', linewidth=2, label='100-Episode Average')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Score Progression')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Episode length progression
        axes[0, 1].plot(episodes, self.episode_lengths, alpha=0.7, color='green')
        if len(self.episode_lengths) >= 50:
            rolling_avg_length = [np.mean(self.episode_lengths[max(0, i-49):i+1]) 
                                 for i in range(len(self.episode_lengths))]
            axes[0, 1].plot(episodes, rolling_avg_length, 'orange', linewidth=2)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].set_title('Episode Length Progression')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Epsilon decay
        axes[1, 0].plot(episodes, self.epsilon_history, color='purple', linewidth=2)
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Epsilon')
        axes[1, 0].set_title('Exploration Rate (Epsilon)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Collision analysis
        if self.collision_history:
            axes[1, 1].plot(episodes, self.collision_history, alpha=0.7, color='red')
            if len(self.collision_history) >= 50:
                rolling_avg_collisions = [np.mean(self.collision_history[max(0, i-49):i+1]) 
                                         for i in range(len(self.collision_history))]
                axes[1, 1].plot(episodes, rolling_avg_collisions, 'darkred', linewidth=2)
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Collisions per Episode')
            axes[1, 1].set_title('Collision Frequency')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(f"{self.log_dir}/training_plots_{timestamp}.png", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_training_summary(self):
        """Get a summary of training performance."""
        if not self.episode_scores:
            return {}
        
        window_size = min(100, len(self.episode_scores))
        recent_avg = np.mean(self.episode_scores[-window_size:])
        
        return {
            'total_episodes': len(self.episode_scores),
            'best_score': self.best_score,
            'best_average_score': self.best_avg_score,
            'recent_average_score': recent_avg,
            'current_epsilon': self.epsilon_history[-1] if self.epsilon_history else 0,
            'average_episode_length': np.mean(self.episode_lengths),
            'total_training_time': time.time() - self.training_start_time if self.training_start_time else 0,
            'episodes_without_improvement': self.episodes_without_improvement
        }

class AdaptiveTrainer:
    """Advanced trainer with adaptive hyperparameters."""
    
    def __init__(self, env, agent, monitor):
        self.env = env
        self.agent = agent
        self.monitor = monitor
        self.early_stopping_patience = 200
        self.lr_scheduler_patience = 50
        self.min_lr = 1e-6
        
    def should_stop_early(self):
        """Check if training should stop early."""
        return self.monitor.episodes_without_improvement >= self.early_stopping_patience
    
    def adjust_learning_rate(self):
        """Adjust learning rate based on performance."""
        if (self.monitor.episodes_without_improvement > 0 and 
            self.monitor.episodes_without_improvement % self.lr_scheduler_patience == 0):
            
            current_lr = self.agent.optimizer.param_groups[0]['lr']
            new_lr = max(self.min_lr, current_lr * 0.5)
            
            if new_lr != current_lr:
                for param_group in self.agent.optimizer.param_groups:
                    param_group['lr'] = new_lr
                print(f"Reduced learning rate to {new_lr:.6f}")
    
    def adaptive_epsilon_decay(self, episode, performance_trend):
        """Adapt epsilon decay based on performance."""
        base_eps = config_manager.ai.EPS_END + (
            config_manager.ai.EPS_START - config_manager.ai.EPS_END
        ) * (config_manager.ai.EPS_DECAY ** episode)
        
        # If performance is declining, increase exploration
        if performance_trend < -0.1:
            return min(1.0, base_eps * 1.5)
        # If performance is improving, reduce exploration faster
        elif performance_trend > 0.1:
            return max(config_manager.ai.EPS_END, base_eps * 0.8)
        
        return base_eps

def train_agent(config_name='Default', save_checkpoints=True, plot_realtime=False):
    """
    Enhanced training function with comprehensive monitoring.
    
    Args:
        config_name: Name of configuration preset to use
        save_checkpoints: Whether to save model checkpoints
        plot_realtime: Whether to show real-time plotting (experimental)
    """
    
    # Load configuration
    if config_name != 'Default':
        config_manager.apply_preset(config_name)
        update_config_values()
    
    # Validate configuration
    errors = config_manager.validate_all()
    if errors:
        print("Configuration errors found:")
        for error in errors:
            print(f"  - {error}")
        return None
    
    # Initialize components
    env = ParticleEnv()
    agent = Agent(config_manager.ai.STATE_SIZE, config_manager.ai.ACTION_SIZE)
    monitor = TrainingMonitor()
    trainer = AdaptiveTrainer(env, agent, monitor)
    
    # Training parameters
    n_episodes = config_manager.ai.MAX_EPISODES
    max_t = config_manager.ai.MAX_STEPS_PER_EPISODE
    eps = config_manager.ai.EPS_START
    
    print("=" * 60)
    print(f"Starting Enhanced AI Training")
    print(f"Configuration: {config_name}")
    print(f"Episodes: {n_episodes}, Max Steps: {max_t}")
    print(f"Physics: G={config_manager.physics.G:.4f}, K_E={config_manager.physics.K_E:.4f}")
    print(f"Particles: {config_manager.physics.PARTICLE_COUNT}")
    print("=" * 60)
    
    try:
        for i_episode in range(1, n_episodes + 1):
            monitor.start_episode(i_episode, eps)
            
            state = env.reset()
            score = 0
            collisions = 0
            
            for t in range(max_t):
                # Agent selects action
                action = agent.act(state, eps)
                
                # Environment step
                next_state, reward, done = env.step(action)
                
                # Count collisions
                if reward < -50:  # Collision penalty threshold
                    collisions += 1
                
                # Agent learns
                agent.step(state, action, reward, next_state, done)
                
                state = next_state
                score += reward
                
                if done:
                    break
            
            # End episode monitoring
            monitor.end_episode(score, t + 1, collisions)
            
            # Adaptive adjustments
            trainer.adjust_learning_rate()
            
            # Calculate performance trend for adaptive epsilon
            if len(monitor.episode_scores) >= 20:
                recent_scores = monitor.episode_scores[-20:]
                performance_trend = (np.mean(recent_scores[-10:]) - 
                                   np.mean(recent_scores[:10])) / np.mean(recent_scores[:10])
                eps = trainer.adaptive_epsilon_decay(i_episode, performance_trend)
            else:
                eps = max(config_manager.ai.EPS_END, 
                         config_manager.ai.EPS_DECAY * eps)
            
            # Save checkpoints
            if save_checkpoints:
                if i_episode % 100 == 0:
                    agent.save(f'checkpoint_episode_{i_episode}.pth')
                
                # Save best model
                if monitor.episode_scores[-1] == monitor.best_score:
                    agent.save('best_model.pth')
            
            # Check for early stopping
            if trainer.should_stop_early():
                print(f"\nEarly stopping at episode {i_episode}")
                print(f"No improvement for {trainer.early_stopping_patience} episodes")
                break
            
            # Check if solved
            window_size = min(100, len(monitor.episode_scores))
            if (window_size >= 100 and 
                np.mean(monitor.episode_scores[-window_size:]) >= config_manager.ai.TARGET_SCORE):
                print(f'\nEnvironment solved in {i_episode-100:d} episodes!')
                print(f'Average Score: {np.mean(monitor.episode_scores[-window_size:]):.2f}')
                agent.save('solved_model.pth')
                break
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    except Exception as e:
        print(f"\nTraining error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Save final results
        agent.save('final_model.pth')
        monitor.save_training_data()
        monitor.create_plots()
        
        # Print summary
        summary = monitor.get_training_summary()
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        for key, value in summary.items():
            if 'time' in key.lower():
                print(f"{key.replace('_', ' ').title()}: {value/60:.1f} minutes")
            elif 'score' in key.lower() or 'epsilon' in key.lower():
                print(f"{key.replace('_', ' ').title()}: {value:.3f}")
            else:
                print(f"{key.replace('_', ' ').title()}: {value}")
        print("=" * 60)
    
    return agent, monitor

def compare_configs(config_names, episodes_per_config=500):
    """Compare training performance across different configurations."""
    results = {}
    
    for config_name in config_names:
        print(f"\nTraining with configuration: {config_name}")
        
        # Backup current config
        original_config = {
            'physics': config_manager.physics.__dict__.copy(),
            'ai': config_manager.ai.__dict__.copy()
        }
        
        try:
            # Apply config and train
            config_manager.apply_preset(config_name)
            config_manager.ai.MAX_EPISODES = episodes_per_config
            update_config_values()
            
            agent, monitor = train_agent(config_name, save_checkpoints=False)
            
            if monitor:
                results[config_name] = monitor.get_training_summary()
        
        except Exception as e:
            print(f"Error training with {config_name}: {e}")
            results[config_name] = {'error': str(e)}
        
        finally:
            # Restore original config
            for key, value in original_config['physics'].items():
                setattr(config_manager.physics, key, value)
            for key, value in original_config['ai'].items():
                setattr(config_manager.ai, key, value)
            update_config_values()
    
    # Create comparison plot
    plt.figure(figsize=(12, 8))
    config_names = list(results.keys())
    best_scores = [results[name].get('best_score', 0) for name in config_names if 'error' not in results[name]]
    avg_scores = [results[name].get('recent_average_score', 0) for name in config_names if 'error' not in results[name]]
    
    x = np.arange(len(config_names))
    width = 0.35
    
    plt.bar(x - width/2, best_scores, width, label='Best Score', alpha=0.8)
    plt.bar(x + width/2, avg_scores, width, label='Average Score', alpha=0.8)
    
    plt.xlabel('Configuration')
    plt.ylabel('Score')
    plt.title('Training Performance Comparison')
    plt.xticks(x, config_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'training_logs/config_comparison_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results

if __name__ == '__main__':
    # Example usage
    print("Enhanced Training System")
    print("1. Single training run")
    print("2. Configuration comparison")
    print("3. Quick test run")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == '1':
        config_name = input("Enter config name (or 'Default'): ").strip() or 'Default'
        train_agent(config_name, save_checkpoints=True, plot_realtime=False)
    
    elif choice == '2':
        configs = ['Default', 'High Gravity', 'Electromagnetic Chaos', 'AI Training Optimized']
        episodes = int(input("Episodes per config (default 200): ").strip() or "200")
        compare_configs(configs, episodes)
    
    elif choice == '3':
        config_manager.ai.MAX_EPISODES = 100
        config_manager.ai.MAX_STEPS_PER_EPISODE = 200
        config_manager.physics.PARTICLE_COUNT = 50
        update_config_values()
        train_agent('Default', save_checkpoints=False)
    
    else:
        print("Invalid choice")