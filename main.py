# main.py
import pygame
import argparse
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from game import FootballEnv
from football_ai import RandomAI, SimpleChaseAI, MAPPOAgent, ParallelSimulationManager, GameLogger, SkilledHeuristicAI

def run_demo_game(fps=60):
    """Run a demo game with a skilled heuristic agent vs a simple chase AI"""
    env = FootballEnv(num_simulations=1)

    # Create heuristic / baseline agents
    skilled = SkilledHeuristicAI()
    opponent = SimpleChaseAI()

    print("Demo: Skilled heuristic agent vs SimpleChaseAI")
    max_steps = 1000
    step_count = 0

    obs = env.reset()
    while not env.done and step_count < max_steps:
        # pass agent index so heuristics act for the correct player
        a1, _, _ = skilled.act(obs, 0)
        a2, _, _ = opponent.act(obs, 1)

        # Step environment with AI actions
        next_obs, rewards, done, info = env.step([int(a1), int(a2)])

        obs = next_obs
        env.render()
        if env.screen is not None and fps > 0:
            env.clock.tick(fps)
        step_count += 1

    print("Game finished!")
    print(f"Final Score: Left {env.game.score['left']} - Right {env.game.score['right']}")
    pygame.quit()

def train_mappo_agents(num_episodes=1000, num_simulations=4, save_models=True, log_dir="logs"):
    """Train MAPPO agents with parallel simulations"""
    print(f"Starting MAPPO training with {num_simulations} parallel simulations...")
    print(f"Training for {num_episodes} episodes per simulation")
    print(f"Logs will be saved to: {log_dir}")
    
    # Create parallel simulation manager
    sim_manager = ParallelSimulationManager(
        num_simulations=num_simulations,
        log_dir=log_dir
    )
    
    # Run parallel simulations
    start_time = time.time()
    results = sim_manager.run_parallel_simulations(num_episodes)
    end_time = time.time()
    
    print(f"Training completed in {end_time - start_time:.2f} seconds")
    print(f"Total episodes completed: {len(results)}")
    
    # Analyze results
    analyze_training_results(results, log_dir)
    
    return results

def analyze_training_results(results, log_dir):
    """Analyze and visualize training results"""
    if not results:
        print("No results to analyze")
        return
    
    # Extract metrics
    episodes = [r['episode'] for r in results]
    rewards = [r['rewards'] for r in results]
    steps = [r['steps'] for r in results]
    
    # Calculate win rates
    agent1_wins = sum(1 for r in results if r['score']['left'] > r['score']['right'])
    agent2_wins = sum(1 for r in results if r['score']['right'] > r['score']['left'])
    draws = len(results) - agent1_wins - agent2_wins
    
    print(f"\nTraining Analysis:")
    print(f"Agent 1 wins: {agent1_wins} ({agent1_wins/len(results)*100:.1f}%)")
    print(f"Agent 2 wins: {agent2_wins} ({agent2_wins/len(results)*100:.1f}%)")
    print(f"Draws: {draws} ({draws/len(results)*100:.1f}%)")
    print(f"Average episode length: {np.mean(steps):.1f} steps")
    
    # Plot training curves (commented out to avoid automatic graphs)
    # plot_training_curves(results, log_dir)

def plot_training_curves(results, log_dir):
    """Create training visualization plots"""
    if not results:
        return
    
    # Extract data
    episodes = [r['episode'] for r in results]
    agent1_rewards = [r['rewards'][0] for r in results]
    agent2_rewards = [r['rewards'][1] for r in results]
    steps = [r['steps'] for r in results]
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Reward curves
    axes[0, 0].plot(episodes, agent1_rewards, label='Agent 1', alpha=0.7)
    axes[0, 0].plot(episodes, agent2_rewards, label='Agent 2', alpha=0.7)
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Moving average rewards
    window = 50
    if len(agent1_rewards) >= window:
        ma1 = np.convolve(agent1_rewards, np.ones(window)/window, mode='valid')
        ma2 = np.convolve(agent2_rewards, np.ones(window)/window, mode='valid')
        axes[0, 1].plot(episodes[window-1:], ma1, label='Agent 1 (MA)', linewidth=2)
        axes[0, 1].plot(episodes[window-1:], ma2, label='Agent 2 (MA)', linewidth=2)
        axes[0, 1].set_title(f'Moving Average Rewards (window={window})')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Reward')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # Episode length
    axes[1, 0].plot(episodes, steps, alpha=0.7)
    axes[1, 0].set_title('Episode Length')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Steps')
    axes[1, 0].grid(True)
    
    # Win rate over time
    window = 100
    if len(results) >= window:
        win_rates = []
        for i in range(window, len(results) + 1):
            recent_results = results[i-window:i]
            agent1_wins = sum(1 for r in recent_results if r['score']['left'] > r['score']['right'])
            win_rate = agent1_wins / window
            win_rates.append(win_rate)
        
        axes[1, 1].plot(episodes[window-1:], win_rates, linewidth=2)
        axes[1, 1].set_title(f'Agent 1 Win Rate (window={window})')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Win Rate')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Training curves saved to {os.path.join(log_dir, 'training_curves.png')}")

def evaluate_trained_agents(model_path1, model_path2, num_games=10, render=True, fps=60):
    """Evaluate trained agents against each other"""
    print(f"Evaluating trained agents for {num_games} games...")
    
    # Load trained agents
    agent1 = MAPPOAgent()
    agent2 = MAPPOAgent()
    
    if os.path.exists(model_path1):
        agent1.load_model(model_path1)
        print(f"Loaded Agent 1 from {model_path1}")
    else:
        print(f"Model file {model_path1} not found, using random initialization")
    
    if os.path.exists(model_path2):
        agent2.load_model(model_path2)
        print(f"Loaded Agent 2 from {model_path2}")
    else:
        print(f"Model file {model_path2} not found, using random initialization")
    
    # Run evaluation games
    env = FootballEnv()
    results = []
    
    for game in range(num_games):
        obs = env.reset()
        step_count = 0
        max_steps = 1000
        
        while not env.done and step_count < max_steps:
            # Get actions from trained agents
            action1, _, _ = agent1.act(obs)
            action2, _, _ = agent2.act(obs)
            
            # Step environment
            obs, rewards, done, info = env.step([action1, action2])
            
            if render:
                env.render()
                if fps > 0:
                    env.clock.tick(fps)
            # If not rendering, run at full speed
            
            step_count += 1
        
        results.append({
            'game': game,
            'score': env.score.copy(),
            'steps': step_count
        })
        
        print(f"Game {game+1}: Left {env.score['left']} - Right {env.score['right']} ({step_count} steps)")
    
    # Analyze evaluation results
    agent1_wins = sum(1 for r in results if r['score']['left'] > r['score']['right'])
    agent2_wins = sum(1 for r in results if r['score']['right'] > r['score']['left'])
    draws = len(results) - agent1_wins - agent2_wins
    
    print(f"\nEvaluation Results:")
    print(f"Agent 1 wins: {agent1_wins} ({agent1_wins/len(results)*100:.1f}%)")
    print(f"Agent 2 wins: {agent2_wins} ({agent2_wins/len(results)*100:.1f}%)")
    print(f"Draws: {draws} ({draws/len(results)*100:.1f}%)")
    
    if render:
        pygame.quit()
    
    return results

def create_visualization_from_logs(log_dir, game_id=0, output_dir="visualizations"):
    """Create visualizations from logged game data"""
    import json
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find game log file
    game_files = [f for f in os.listdir(log_dir) if f.startswith(f"game_{game_id}_")]
    if not game_files:
        print(f"No game log found for game {game_id}")
        return
    
    game_file = os.path.join(log_dir, game_files[0])
    
    with open(game_file, 'r') as f:
        game_data = json.load(f)
    
    # Extract positions
    steps = game_data['game_data']
    player1_positions = []
    player2_positions = []
    ball_positions = []
    
    for step in steps:
        obs = step['obs']
        # obs = [p1x, p1y, p1vx, p1vy, p2x, p2y, p2vx, p2vy, ballx, bally, ballvx, ballvy]
        player1_positions.append([obs[0], obs[1]])
        player2_positions.append([obs[4], obs[5]])
        ball_positions.append([obs[8], obs[9]])
    
    # Create trajectory plot
    plt.figure(figsize=(12, 8))
    
    # Plot field
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axhline(y=600, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)
    plt.axvline(x=800, color='k', linewidth=0.5)
    
    # Plot trajectories
    p1_pos = np.array(player1_positions)
    p2_pos = np.array(player2_positions)
    ball_pos = np.array(ball_positions)
    
    plt.plot(p1_pos[:, 0], p1_pos[:, 1], 'r-', linewidth=2, label='Player 1', alpha=0.7)
    plt.plot(p2_pos[:, 0], p2_pos[:, 1], 'b-', linewidth=2, label='Player 2', alpha=0.7)
    plt.plot(ball_pos[:, 0], ball_pos[:, 1], 'w-', linewidth=1, label='Ball', alpha=0.5)
    
    # Mark start and end positions
    plt.scatter(p1_pos[0, 0], p1_pos[0, 1], color='red', s=100, marker='o', label='P1 Start')
    plt.scatter(p2_pos[0, 0], p2_pos[0, 1], color='blue', s=100, marker='o', label='P2 Start')
    plt.scatter(ball_pos[0, 0], ball_pos[0, 1], color='white', s=50, marker='o', label='Ball Start')
    
    plt.xlim(-50, 850)
    plt.ylim(-50, 650)
    plt.title(f'Game {game_id} Trajectories')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f'game_{game_id}_trajectories.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Visualization saved to {os.path.join(output_dir, f'game_{game_id}_trajectories.png')}")

def compute_reward_shaping(obs, prev_obs, player_id):
    """Compute shaped rewards to help agents learn"""
    # Extract positions from observation
    # obs = [p1x, p1y, p1vx, p1vy, p2x, p2y, p2vx, p2vy, ballx, bally, ballvx, ballvy]
    player_x, player_y = obs[player_id*4], obs[player_id*4+1]
    ball_x, ball_y = obs[8], obs[9]
    
    # Distance to ball
    dist_to_ball = np.sqrt((player_x - ball_x)**2 + (player_y - ball_y)**2)
    
    # Reward for getting closer to ball
    if prev_obs is not None:
        prev_player_x, prev_player_y = prev_obs[player_id*4], prev_obs[player_id*4+1]
        prev_dist_to_ball = np.sqrt((prev_player_x - ball_x)**2 + (prev_player_y - ball_y)**2)
        ball_proximity_reward = (prev_dist_to_ball - dist_to_ball) * 0.1
    else:
        ball_proximity_reward = 0
    
    # Reward for ball possession (being close to ball)
    possession_reward = max(0, 0.1 - dist_to_ball * 0.01) if dist_to_ball < 50 else 0
    
    # Reward for moving (encourage exploration)
    if prev_obs is not None:
        prev_player_x, prev_player_y = prev_obs[player_id*4], prev_obs[player_id*4+1]
        movement = np.sqrt((player_x - prev_player_x)**2 + (player_y - prev_player_y)**2)
        movement_reward = min(0.01, movement * 0.001)
    else:
        movement_reward = 0
    
    return ball_proximity_reward + possession_reward + movement_reward

def run_learning_test(num_episodes=50):
    """Run a quick learning test to verify MAPPO agents are learning"""
    print(f"Running learning test for {num_episodes} episodes...")
    print("This will show if the agents are actually learning from rewards")
    
    from football_ai import MAPPOAgent
    from game import FootballEnv
    
    # Create environment without rendering for testing
    env = FootballEnv()
    env.screen = None  # Disable rendering
    agent1 = MAPPOAgent()
    agent2 = MAPPOAgent()
    
    scores = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        prev_obs = None
        step_count = 0
        max_steps = 500
        
        while not env.done and step_count < max_steps:
            action1, log_prob1, value1 = agent1.act(obs)
            action2, log_prob2, value2 = agent2.act(obs)
            
            next_obs, rewards, done, info = env.step([action1, action2])
            
            # Add shaped rewards
            shaped_reward1 = compute_reward_shaping(obs, prev_obs, 0)
            shaped_reward2 = compute_reward_shaping(obs, prev_obs, 1)
            
            total_reward1 = rewards[0] + shaped_reward1
            total_reward2 = rewards[1] + shaped_reward2
            
            # Store experiences with shaped rewards
            agent1.store_experience(obs, action1, total_reward1, next_obs, done, log_prob1, value1)
            agent2.store_experience(obs, action2, total_reward2, next_obs, done, log_prob2, value2)
            
            prev_obs = obs
            obs = next_obs
            step_count += 1
        
        # Update policies every 5 episodes
        if episode % 5 == 0 and episode > 0:
            agent1.update_policy()
            agent2.update_policy()
        
        scores.append(env.score.copy())
        
        if episode % 10 == 0:
            print(f"Episode {episode}: Score {env.score['left']}-{env.score['right']}, Steps {step_count}")
    
    # Analyze learning progress
    early_scores = scores[:10]
    late_scores = scores[-10:]
    
    early_goals = sum(s['left'] + s['right'] for s in early_scores)
    late_goals = sum(s['left'] + s['right'] for s in late_scores)
    
    print(f"\nLearning Analysis:")
    print(f"Early episodes (0-9): {early_goals} total goals")
    print(f"Late episodes ({num_episodes-10}-{num_episodes-1}): {late_goals} total goals")
    
    if late_goals > early_goals:
        print("✓ Agents are learning! More goals scored in later episodes.")
    else:
        print("⚠ Agents may not be learning effectively. Consider adjusting learning parameters.")
    
    return scores

def main():
    parser = argparse.ArgumentParser(description='AI Sim Football - MAPPO Training and Evaluation')
    parser.add_argument('--mode', choices=['demo', 'train', 'eval', 'visualize', 'test'], default='demo',
                       help='Mode to run: demo, train, eval, visualize, or test')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of episodes for training')
    parser.add_argument('--simulations', type=int, default=4,
                       help='Number of parallel simulations')
    parser.add_argument('--log-dir', type=str, default='logs',
                       help='Directory to save logs and models')
    parser.add_argument('--model1', type=str, default='logs/sim_0_agent1.pt',
                       help='Path to Agent 1 model for evaluation')
    parser.add_argument('--model2', type=str, default='logs/sim_0_agent2.pt',
                       help='Path to Agent 2 model for evaluation')
    parser.add_argument('--games', type=int, default=10,
                       help='Number of games for evaluation')
    parser.add_argument('--render', action='store_true',
                       help='Render games during evaluation')
    parser.add_argument('--game-id', type=int, default=0,
                       help='Game ID for visualization')
    parser.add_argument('--fps', type=int, default=60,
                       help='FPS limit for rendering (0 = unlimited)')
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        print("Running demo game with MAPPO agents...")
        run_demo_game(fps=args.fps)
    
    elif args.mode == 'test':
        print("Running learning test...")
        run_learning_test(args.episodes)
    
    elif args.mode == 'train':
        print("Starting MAPPO training...")
        train_mappo_agents(
            num_episodes=args.episodes,
            num_simulations=args.simulations,
            log_dir=args.log_dir
        )
    
    elif args.mode == 'eval':
        print("Evaluating trained agents...")
        evaluate_trained_agents(
            model_path1=args.model1,
            model_path2=args.model2,
            num_games=args.games,
            render=args.render,
            fps=args.fps
        )
    
    elif args.mode == 'visualize':
        print("Creating visualizations from logs...")
        create_visualization_from_logs(
            log_dir=args.log_dir,
            game_id=args.game_id
        )

if __name__ == "__main__":
    main()

