import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import json
import os
from datetime import datetime
from collections import deque
import time

# Set device for GPU acceleration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class RandomAI:
    def __init__(self, n_agents=2):
        self.n_agents = n_agents
        self.action_probability = 0.3  # Probability of taking action vs staying still

    def act(self, obs):
        # Return random actions for each agent with some probability of no-op
        actions = []
        for i in range(self.n_agents):
            if np.random.random() < self.action_probability:
                # Take a random action (0-3 for movement, 4 for no-op)
                action = np.random.randint(0, 5)
            else:
                # Stay still
                action = 4
            actions.append(action)
        return actions

class SimpleChaseAI:
    def __init__(self, n_agents=2):
        self.n_agents = n_agents

    def act(self, obs):
        # obs = [p1x, p1y, p1vx, p1vy, p2x, p2y, p2vx, p2vy, ballx, bally, ballvx, ballvy]
        actions = []
        ball_x, ball_y = obs[-4], obs[-3]
        for i in range(self.n_agents):
            px, py = obs[i*4], obs[i*4+1]
            # Move toward the ball
            if abs(ball_x - px) > abs(ball_y - py):
                action = 3 if ball_x > px else 2
            else:
                action = 1 if ball_y > py else 0
            actions.append(action)
        return actions

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        shared_out = self.shared(state)
        action_logits = self.actor(shared_out)
        value = self.critic(shared_out)
        return action_logits, value
    
    def get_action(self, state):
        action_logits, value = self.forward(state)
        dist = Categorical(logits=action_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value

class ExperienceBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done, log_prob, value):
        self.buffer.append((state, action, reward, next_state, done, log_prob, value))
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        return zip(*batch)
    
    def __len__(self):
        return len(self.buffer)

class GameLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.game_logs = []
        self.current_game = []
    
    def log_step(self, obs, actions, rewards, done):
        """Log a single step of the game"""
        step_data = {
            'timestamp': time.time(),
            'obs': obs.tolist(),
            'actions': actions,
            'rewards': rewards,
            'done': done
        }
        self.current_game.append(step_data)
    
    def log_game_end(self, final_score, episode):
        """Log the end of a game"""
        game_data = {
            'episode': episode,
            'final_score': final_score,
            'steps': len(self.current_game),
            'game_data': self.current_game
        }
        self.game_logs.append(game_data)
        
        # Save individual game log
        game_file = os.path.join(self.log_dir, f"game_{episode}_{self.timestamp}.json")
        with open(game_file, 'w') as f:
            json.dump(game_data, f, indent=2)
        
        self.current_game = []
    
    def save_summary(self, training_stats):
        """Save training summary"""
        summary_file = os.path.join(self.log_dir, f"training_summary_{self.timestamp}.json")
        with open(summary_file, 'w') as f:
            json.dump(training_stats, f, indent=2)

class MAPPOAgent:
    def __init__(self, state_dim=12, action_dim=5, lr=3e-4, gamma=0.99, 
                 eps_clip=0.2, k_epochs=4, hidden_dim=256, device=device):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.device = device
        
        # Neural networks
        self.actor_critic = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        
        # Experience buffer
        self.buffer = ExperienceBuffer()
        
        # Training stats
        self.training_stats = {
            'episodes': 0,
            'total_rewards': [],
            'win_rate': 0.0,
            'avg_episode_length': 0.0
        }
    
    def act(self, obs):
        """Get action from current policy"""
        state = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob, value = self.actor_critic.get_action(state)
        return action, log_prob, value
    
    def store_experience(self, state, action, reward, next_state, done, log_prob, value):
        """Store experience in buffer"""
        self.buffer.push(state, action, reward, next_state, done, log_prob, value)
    
    def update_policy(self, other_agent_buffer=None):
        """Update policy using PPO"""
        if len(self.buffer) < 32:  # Need minimum batch size
            return
        
        # Sample experiences
        states, actions, rewards, next_states, dones, old_log_probs, old_values = self.buffer.sample(32)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        old_values = torch.FloatTensor(old_values).to(self.device)
        
        # Compute returns
        returns = self.compute_returns(rewards, dones)
        advantages = returns - old_values
        
        # Normalize advantages
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO updates
        for _ in range(self.k_epochs):
            # Get current policy outputs
            action_logits, values = self.actor_critic(states)
            dist = Categorical(logits=action_logits)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy()
            
            # Compute ratios
            ratios = torch.exp(new_log_probs - old_log_probs)
            
            # Compute surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values.squeeze(), returns)
            
            # Total loss
            total_loss = actor_loss + 0.5 * value_loss - 0.01 * entropy.mean()
            
            # Update
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
            self.optimizer.step()
    
    def compute_returns(self, rewards, dones):
        """Compute discounted returns"""
        returns = torch.zeros_like(rewards)
        running_return = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                running_return = 0
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return
        
        return returns
    
    def save_model(self, filepath):
        """Save the model"""
        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats
        }, filepath)
    
    def load_model(self, filepath):
        """Load the model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_stats = checkpoint['training_stats']

class ParallelSimulationManager:
    def __init__(self, num_simulations=4, log_dir="logs"):
        self.num_simulations = num_simulations
        self.log_dir = log_dir
        self.simulations = []
        self.loggers = []
        self.results = []
        
    def create_simulation(self, simulation_id):
        """Create a single simulation environment"""
        from game import FootballEnv
        
        env = FootballEnv()
        env.screen = None  # Disable rendering for training
        logger = GameLogger(os.path.join(self.log_dir, f"sim_{simulation_id}"))
        
        # Create MAPPO agents
        agent1 = MAPPOAgent(device=device)
        agent2 = MAPPOAgent(device=device)
        
        return env, logger, agent1, agent2
    
    def run_simulation(self, simulation_id, num_episodes=100):
        """Run a single simulation"""
        env, logger, agent1, agent2 = self.create_simulation(simulation_id)
        
        print(f"Starting simulation {simulation_id} with {num_episodes} episodes...")
        
        for episode in range(num_episodes):
            obs = env.reset()
            episode_reward = [0, 0]
            step_count = 0
            max_steps = 1000
            
            while not env.done and step_count < max_steps:
                # Get actions from both agents
                action1, log_prob1, value1 = agent1.act(obs)
                action2, log_prob2, value2 = agent2.act(obs)
                
                # Log the step
                logger.log_step(obs, [action1, action2], [0, 0], env.done)
                
                # Step environment
                next_obs, rewards, done, info = env.step([action1, action2])
                
                # Store experiences
                agent1.store_experience(obs, action1, rewards[0], next_obs, done, log_prob1, value1)
                agent2.store_experience(obs, action2, rewards[1], next_obs, done, log_prob2, value2)
                
                episode_reward[0] += rewards[0]
                episode_reward[1] += rewards[1]
                
                obs = next_obs
                step_count += 1
            
            # Log game end
            logger.log_game_end(env.score, episode)
            
            # Update policies every few episodes to allow for experience accumulation
            if episode % 10 == 0 and episode > 0:
                agent1.update_policy()
                agent2.update_policy()
                print(f"Sim {simulation_id}, Episode {episode}: Updated policies")
            
            # Update training stats
            agent1.training_stats['episodes'] += 1
            agent2.training_stats['episodes'] += 1
            
            # Print progress with percentage
            progress = (episode + 1) / num_episodes * 100
            if episode % 10 == 0 or episode < 10:
                print(f"Sim {simulation_id} [{progress:.1f}%] Episode {episode}: Score {env.score['left']}-{env.score['right']}, Steps {step_count}")
            
            # Store results directly
            self.results.append({
                'simulation_id': simulation_id,
                'episode': episode,
                'rewards': episode_reward,
                'score': env.score.copy(),
                'steps': step_count
            })
        
        # Final policy update
        agent1.update_policy()
        agent2.update_policy()
        
        # Save models
        agent1.save_model(os.path.join(self.log_dir, f"sim_{simulation_id}_agent1.pt"))
        agent2.save_model(os.path.join(self.log_dir, f"sim_{simulation_id}_agent2.pt"))
    
    def run_parallel_simulations(self, num_episodes=100):
        """Run multiple simulations sequentially (pygame doesn't work well with threading)"""
        print(f"Running {self.num_simulations} simulations sequentially to avoid pygame threading issues...")
        
        for sim_id in range(self.num_simulations):
            print(f"Starting simulation {sim_id + 1}/{self.num_simulations}")
            self.run_simulation(sim_id, num_episodes)
            print(f"Completed simulation {sim_id + 1}/{self.num_simulations}")
        
        return self.collect_results()
    
    def collect_results(self):
        """Collect results from all simulations"""
        return self.results

