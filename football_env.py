import math
import pygame
import numpy as np
from game_core import Game, SCREEN_PADDING, BLACK

class FootballEnv:
    """
    Wrapper for one-or-many Game instances.
    Maintains backward-compatible attributes for single-env mode (env.game, env.score).
    """
    def __init__(self, num_simulations=1, cols=None, fps=30):
        pygame.init(); pygame.font.init()
        self.num_simulations = max(1, num_simulations); self.fps = fps
        self.font = pygame.font.SysFont(None, 28)
        self.games = [Game() for _ in range(self.num_simulations)]
        if self.num_simulations == 1:
            self.game = self.games[0]
            self.screen = pygame.display.set_mode((self.game.field.screen_width, self.game.field.screen_height))
            pygame.display.set_caption("Football Env")
            self.clock = pygame.time.Clock()
        else:
            if cols is None: cols = int(math.ceil(math.sqrt(self.num_simulations)))
            self.cols = max(1, cols); self.rows = int(math.ceil(self.num_simulations / self.cols))
            sub_w = self.games[0].field.screen_width; sub_h = self.games[0].field.screen_height
            win_w = self.cols * sub_w + (self.cols + 1) * SCREEN_PADDING // 2
            win_h = self.rows * sub_h + (self.rows + 1) * SCREEN_PADDING // 2
            self.screen = pygame.display.set_mode((win_w, win_h))
            pygame.display.set_caption(f"FootballEnv - {self.num_simulations} sims")
            self.clock = pygame.time.Clock()
    def reset(self, idx=None):
        if self.num_simulations == 1 or idx is not None:
            if idx is None: idx = 0
            return self.games[idx].reset()
        else:
            return [g.reset() for g in self.games]
    def step(self, actions=None):
        if self.num_simulations == 1:
            return self.games[0].step(actions)
        else:
            results = []
            keys = pygame.key.get_pressed()
            for i, g in enumerate(self.games):
                act = None
                if isinstance(actions, list) and i < len(actions): act = actions[i]
                res = g.step(act, keys if act is None else None)
                results.append(res)
            return results
    @property
    def done(self):
        if self.num_simulations == 1: return self.games[0].done
        return all(g.done for g in self.games)
    @property
    def score(self):
        if self.num_simulations == 1: return self.games[0].score
        return [g.score for g in self.games]
    def render(self):
        # Ensure pygame/display initialized and a window exists even if external code cleared self.screen
        if not pygame.get_init():
            pygame.init(); pygame.font.init()
        if self.screen is None:
            # recreate appropriate window for single or multi simulations
            if self.num_simulations == 1:
                size = (self.games[0].field.screen_width, self.games[0].field.screen_height)
            else:
                sub_w = self.games[0].field.screen_width
                sub_h = self.games[0].field.screen_height
                win_w = self.cols * sub_w + (self.cols + 1) * SCREEN_PADDING // 2
                win_h = self.rows * sub_h + (self.rows + 1) * SCREEN_PADDING // 2
                size = (win_w, win_h)
            self.screen = pygame.display.set_mode(size)
            caption = "Football Env" if self.num_simulations == 1 else f"FootballEnv - {self.num_simulations} sims"
            pygame.display.set_caption(caption)
            self.clock = pygame.time.Clock()

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                for g in self.games: g.done = True
                return

        if self.num_simulations == 1:
            surf = self.games[0].render_to_surface(self.font)
            self.screen.blit(surf, (0,0)); pygame.display.flip(); self.clock.tick(self.fps)
        else:
            sub_w = self.games[0].field.screen_width; sub_h = self.games[0].field.screen_height
            self.screen.fill(BLACK)
            for idx, g in enumerate(self.games):
                r = idx // self.cols; c = idx % self.cols
                x = SCREEN_PADDING // 4 + c * (sub_w + SCREEN_PADDING // 4)
                y = SCREEN_PADDING // 4 + r * (sub_h + SCREEN_PADDING // 4)
                surf = g.render_to_surface(self.font)
                self.screen.blit(surf, (x,y))
            pygame.display.flip(); self.clock.tick(self.fps)

if __name__ == "__main__":
    env = FootballEnv(num_simulations=4)
    running_steps = 500
    for _ in range(running_steps):
        acts = []
        for _ in range(env.num_simulations):
            acts.append([np.random.randint(0,5), np.random.randint(0,5)])
        env.step(acts); env.render()
        if env.done: break
    pygame.quit()