# main.py
import pygame
from game import FootballEnv
from football_ai import RandomAI, SimpleChaseAI

def run_game():
    env = FootballEnv()
    
    # Choose AIs for each player
    ai_agents = [
        SimpleChaseAI(n_agents=1),  # Player 1
        RandomAI(n_agents=1)        # Player 2
    ]

    max_steps = 1000  # limit game length
    step_count = 0

    while not env.done and step_count < max_steps:
        # Get current observation
        obs = env.get_obs()
        
        # Generate actions from AI
        actions = [ai.act(obs) for ai in ai_agents]
        # Flatten actions (each AI returns list)
        actions = [a[0] for a in actions]

        # Step environment with AI actions
        obs, reward, done, info = env.step(actions)

        # Render
        env.render()

        env.clock.tick(60)
        step_count += 1

    print("Game finished!")
    print(f"Final Score: Left {env.score['left']} - Right {env.score['right']}")
    pygame.quit()


if __name__ == "__main__":
    run_game()

