import pygame
import numpy as np

# --- Sizes ---
FIELD_WIDTH, FIELD_HEIGHT = 800, 600
GOAL_WIDTH, GOAL_HEIGHT = 100, 200
SCREEN_PADDING = 50  # extra space around the play area

PLAYER_RADIUS = 15
BALL_RADIUS = 10

WHITE = (255, 255, 255)
RED   = (200, 50, 50)
BLUE  = (50, 50, 200)
GREEN = (50, 200, 50)
BLACK = (0, 0, 0)

FRICTION = 0.98

# --- Classes ---
class Field:
    def __init__(self, width=FIELD_WIDTH, height=FIELD_HEIGHT, goal_width=GOAL_WIDTH, goal_height=GOAL_HEIGHT):
        self.width = width
        self.height = height
        self.goal_width = goal_width
        self.goal_height = goal_height

        self.screen_width = width + 2 * goal_width + 2 * SCREEN_PADDING
        self.screen_height = max(height, goal_height) + 2 * SCREEN_PADDING

        # Offset to draw field in the middle
        self.offset_x = SCREEN_PADDING + goal_width
        self.offset_y = SCREEN_PADDING

        # Goals
        self.goals = [Goal("left", self), Goal("right", self)]

    def get_playable_bounds(self):
        # Unified bounds for players and ball (allows entering goals)
        left = self.goals[0].rect.left + BALL_RADIUS
        right = self.goals[1].rect.right - BALL_RADIUS
        top = self.offset_y + BALL_RADIUS
        bottom = self.offset_y + self.height - BALL_RADIUS
        return left, right, top, bottom

    def draw(self, screen):
        # Draw field
        pygame.draw.rect(screen, (50, 150, 50),
                         pygame.Rect(self.offset_x, self.offset_y, self.width, self.height))
        pygame.draw.rect(screen, WHITE,
                         pygame.Rect(self.offset_x, self.offset_y, self.width, self.height), 2)

        # Draw goals
        for goal in self.goals:
            goal.draw(screen)

    def check_goal(self, ball):
        for goal in self.goals:
            if goal.check_score(ball):
                return goal.side
        return None


class Ball:
    def __init__(self, pos, mass=1.0):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.zeros(2, dtype=float)
        self.mass = mass

    def update(self, field):
        self.pos += self.vel
        self.vel *= FRICTION
        if np.linalg.norm(self.vel) < 0.01:
            self.vel[:] = 0

        # Check for goal
        goal_side = field.check_goal(self)
        if goal_side:
            return goal_side

        left, right, top, bottom = field.get_playable_bounds()

        # Bounce off top/bottom walls
        if self.pos[1] - BALL_RADIUS < top:
            self.vel[1] *= -1
            self.pos[1] = top + BALL_RADIUS
        if self.pos[1] + BALL_RADIUS > bottom:
            self.vel[1] *= -1
            self.pos[1] = bottom - BALL_RADIUS

        # Bounce left/right only outside goal height
        left_goal = field.goals[0].rect
        right_goal = field.goals[1].rect
        if self.pos[0] - BALL_RADIUS < left_goal.right and not (left_goal.top <= self.pos[1] <= left_goal.bottom):
            self.vel[0] *= -1
            self.pos[0] = left_goal.right + BALL_RADIUS
        if self.pos[0] + BALL_RADIUS > right_goal.left and not (right_goal.top <= self.pos[1] <= right_goal.bottom):
            self.vel[0] *= -1
            self.pos[0] = right_goal.left - BALL_RADIUS

        # Clip inside unified playable area
        self.pos[0] = np.clip(self.pos[0], left, right)
        self.pos[1] = np.clip(self.pos[1], top, bottom)

        return None

    def draw(self, screen):
        pygame.draw.circle(screen, WHITE, (int(self.pos[0]), int(self.pos[1])), BALL_RADIUS)


class Player:
    def __init__(self, pos, color, controls, mass=3.0, speed=4.0):
        self.pos = np.array(pos, dtype=float)
        self.color = color
        self.mass = mass
        self.speed = speed
        self.controls = controls
        self.vel = np.zeros(2, dtype=float)

    def handle_input(self, keys, field):
        move = np.zeros(2, dtype=float)
        if keys[self.controls["up"]]:
            move[1] -= 1
        if keys[self.controls["down"]]:
            move[1] += 1
        if keys[self.controls["left"]]:
            move[0] -= 1
        if keys[self.controls["right"]]:
            move[0] += 1

        if np.linalg.norm(move) > 0:
            move = move / np.linalg.norm(move) * self.speed
        self.vel = move
        self.pos += self.vel

        if np.all(move == 0):
            self.vel[:] = 0

        left_goal = field.goals[0].rect
        right_goal = field.goals[1].rect
        field_left = field.offset_x
        field_right = field.offset_x + field.width
        field_top = field.offset_y
        field_bottom = field.offset_y + field.height

        # --- Horizontal confinement ---
        # Only allow X inside goal width if Y is within goal height
        if left_goal.top <= self.pos[1] <= left_goal.bottom:
            self.pos[0] = np.clip(self.pos[0], left_goal.left + PLAYER_RADIUS, field_right - PLAYER_RADIUS)
        else:
            self.pos[0] = max(self.pos[0], field_left + PLAYER_RADIUS)

        if right_goal.top <= self.pos[1] <= right_goal.bottom:
            self.pos[0] = np.clip(self.pos[0], field_left + PLAYER_RADIUS, right_goal.right - PLAYER_RADIUS)
        else:
            self.pos[0] = min(self.pos[0], field_right - PLAYER_RADIUS)

        # --- Vertical confinement ---
        if field_left <= self.pos[0] <= field_right:
            # Inside main field horizontally
            self.pos[1] = np.clip(self.pos[1], field_top + PLAYER_RADIUS, field_bottom - PLAYER_RADIUS)
        elif left_goal.left <= self.pos[0] <= left_goal.right:
            # Inside left goal
            self.pos[1] = np.clip(self.pos[1], left_goal.top + PLAYER_RADIUS, left_goal.bottom - PLAYER_RADIUS)
        elif right_goal.left <= self.pos[0] <= right_goal.right:
            # Inside right goal
            self.pos[1] = np.clip(self.pos[1], right_goal.top + PLAYER_RADIUS, right_goal.bottom - PLAYER_RADIUS)

    def interact_with_ball(self, ball):
        diff = ball.pos - self.pos
        dist = np.linalg.norm(diff)
        if dist < PLAYER_RADIUS + BALL_RADIUS:
            if dist == 0:
                direction = np.array([1.0, 0.0])
            else:
                direction = diff / dist

            relative_vel = direction * self.speed
            total_mass = self.mass + ball.mass
            ball.vel += (2 * self.mass / total_mass) * relative_vel

            overlap = (PLAYER_RADIUS + BALL_RADIUS) - dist
            ball.pos += direction * overlap

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.pos[0]), int(self.pos[1])), PLAYER_RADIUS)


class Goal:
    def __init__(self, side, field):
        self.side = side
        self.field = field
        if side == "left":
            self.rect = pygame.Rect(field.offset_x - field.goal_width,
                                    field.offset_y + (field.height - field.goal_height)//2,
                                    field.goal_width, field.goal_height)
        else:
            self.rect = pygame.Rect(field.offset_x + field.width,
                                    field.offset_y + (field.height - field.goal_height)//2,
                                    field.goal_width, field.goal_height)

    def check_score(self, ball):
        return self.rect.collidepoint(ball.pos[0], ball.pos[1])

    def draw(self, screen):
        pygame.draw.rect(screen, GREEN, self.rect)
        pygame.draw.rect(screen, WHITE, self.rect, 2)  # outline


class FootballEnv:
    def __init__(self):
        pygame.init()
        self.field = Field()
        self.screen = pygame.display.set_mode((self.field.screen_width, self.field.screen_height))
        pygame.display.set_caption("Football Env")
        self.font = pygame.font.SysFont(None, 32)

        self.clock = pygame.time.Clock()
        self.done = False

        # Entities
        self.ball = Ball([self.field.offset_x + self.field.width//2,
                          self.field.offset_y + self.field.height//2])
        self.players = [
            Player([self.field.offset_x + 100, self.field.offset_y + self.field.height//2], RED,
                   {"up": pygame.K_w, "down": pygame.K_s, "left": pygame.K_a, "right": pygame.K_d}),
            Player([self.field.offset_x + self.field.width - 100, self.field.offset_y + self.field.height//2], BLUE,
                   {"up": pygame.K_UP, "down": pygame.K_DOWN, "left": pygame.K_LEFT, "right": pygame.K_RIGHT})
        ]
        self.score = {"left": 0, "right": 0}

    def reset_ball(self):
        self.ball.pos[:] = [self.field.offset_x + self.field.width//2,
                            self.field.offset_y + self.field.height//2]
        self.ball.vel[:] = 0

    def reset(self):
        for i, player in enumerate(self.players):
            if i == 0:
                player.pos[:] = [self.field.offset_x + 100, self.field.offset_y + self.field.height//2]
            else:
                player.pos[:] = [self.field.offset_x + self.field.width - 100, self.field.offset_y + self.field.height//2]
            player.vel[:] = 0
        self.reset_ball()
        self.score = {"left": 0, "right": 0}

        return self.get_obs()

    def get_obs(self):
        obs = []
        for player in self.players:
            obs.extend([player.pos[0], player.pos[1], player.vel[0], player.vel[1]])
        obs.extend([self.ball.pos[0], self.ball.pos[1], self.ball.vel[0], self.ball.vel[1]])
        return np.array(obs, dtype=np.float32)

    def apply_action(self, player, action):
        move = np.zeros(2)
        # action: 0=up,1=down,2=left,3=right,4=no-op
        if action == 0: move[1] -= 1
        elif action == 1: move[1] += 1
        elif action == 2: move[0] -= 1
        elif action == 3: move[0] += 1
        if np.linalg.norm(move) > 0:
            move = move / np.linalg.norm(move) * player.speed
        player.vel = move
        player.pos += move

        left_goal = self.field.goals[0].rect
        right_goal = self.field.goals[1].rect
        field_left = self.field.offset_x
        field_right = self.field.offset_x + self.field.width
        field_top = self.field.offset_y
        field_bottom = self.field.offset_y + self.field.height

        # Horizontal confinement with goal checks
        if left_goal.top <= player.pos[1] <= left_goal.bottom:
            player.pos[0] = np.clip(player.pos[0], left_goal.left + PLAYER_RADIUS, field_right - PLAYER_RADIUS)
        else:
            player.pos[0] = max(player.pos[0], field_left + PLAYER_RADIUS)

        if right_goal.top <= player.pos[1] <= right_goal.bottom:
            player.pos[0] = np.clip(player.pos[0], field_left + PLAYER_RADIUS, right_goal.right - PLAYER_RADIUS)
        else:
            player.pos[0] = min(player.pos[0], field_right - PLAYER_RADIUS)

        # Vertical confinement
        if field_left <= player.pos[0] <= field_right:
            player.pos[1] = np.clip(player.pos[1], field_top + PLAYER_RADIUS, field_bottom - PLAYER_RADIUS)
        elif left_goal.left <= player.pos[0] <= left_goal.right:
            player.pos[1] = np.clip(player.pos[1], left_goal.top + PLAYER_RADIUS, left_goal.bottom - PLAYER_RADIUS)
        elif right_goal.left <= player.pos[0] <= right_goal.right:
            player.pos[1] = np.clip(player.pos[1], right_goal.top + PLAYER_RADIUS, right_goal.bottom - PLAYER_RADIUS)

    def step(self, actions=None):
        # If no actions, fall back to human input
        if actions is None:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_q]:
                self.done = True
            for player in self.players:
                player.handle_input(keys, self.field)
        else:
            for i, player in enumerate(self.players):
                self.apply_action(player, actions[i])

        # Ball interaction
        for player in self.players:
            player.interact_with_ball(self.ball)

        self.resolve_player_collisions()

        goal_side = self.ball.update(self.field)
        reward = [0]*len(self.players)
        if goal_side == "left":
            reward[1] = 1
            reward[0] = -1
            self.score["right"] += 1
            self.reset_ball()
        elif goal_side == "right":
            reward[0] = 1
            reward[1] = -1
            self.score["left"] += 1
            self.reset_ball()

        obs = self.get_obs()
        return obs, reward, self.done, {}

    def resolve_player_collisions(self):
        # Separate overlapping players and apply high friction damping on collision
        num_players = len(self.players)
        if num_players < 2:
            return
        for i in range(num_players):
            for j in range(i + 1, num_players):
                p1 = self.players[i]
                p2 = self.players[j]
                diff = p2.pos - p1.pos
                dist = np.linalg.norm(diff)
                min_dist = 2 * PLAYER_RADIUS
                if dist == 0:
                    # Avoid division by zero; choose an arbitrary separation direction
                    direction = np.array([1.0, 0.0])
                    dist = 1e-6
                else:
                    direction = diff / dist
                overlap = min_dist - dist
                if overlap > 0:
                    # Push players apart equally
                    correction = direction * (overlap / 2.0)
                    p1.pos -= correction
                    p2.pos += correction
                    # High friction on collision: heavily damp velocities
                    p1.vel *= 0.2
                    p2.vel *= 0.2

    def render(self):
        # Basic rendering to support main.py
        self.screen.fill(BLACK)
        self.field.draw(self.screen)
        for player in self.players:
            player.draw(self.screen)
        self.ball.draw(self.screen)
        # Draw score at the top
        score_text = f"Left {self.score['left']} - Right {self.score['right']}"
        text_surface = self.font.render(score_text, True, WHITE)
        text_rect = text_surface.get_rect()
        text_rect.centerx = self.field.offset_x + self.field.width // 2
        text_rect.top = 10
        self.screen.blit(text_surface, text_rect)
        pygame.display.flip()

if __name__ == "__main__":
    env = FootballEnv()
    env.run()

