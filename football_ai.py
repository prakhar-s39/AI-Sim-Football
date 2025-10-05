import numpy as np

class RandomAI:
    def __init__(self, n_agents=2):
        self.n_agents = n_agents

    def act(self, obs):
        # Return random actions for each agent
        return np.random.randint(0, 5, size=self.n_agents).tolist()

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

