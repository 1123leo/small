#!/usr/bin/env python3
"""
Flappy Bird: Human vs AI - Visual Edition with Pygame
Left: Player (controlled by you)
Right: AI Agent (automatically playing)
Controls: Press SPACE or any key to jump
"""

import gymnasium as gym
import flappy_bird_gymnasium
import numpy as np
import pygame
import time
import sys
import warnings
import os

warnings.filterwarnings('ignore')

# Try to import PPO 
try:
    from stable_baselines3 import PPO
    PPO_AVAILABLE = True
except ImportError:
    PPO_AVAILABLE = False

# Constants
GAME_WIDTH = 384
GAME_HEIGHT = 512
WINDOW_WIDTH = GAME_WIDTH * 2 + 40
WINDOW_HEIGHT = GAME_HEIGHT + 180
FPS = 30

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (220, 220, 220)
BLUE = (30, 100, 200)
RED = (220, 0, 0)
GREEN = (0, 150, 0)


class GameStats:
    def __init__(self):
        self.player_wins = 0
        self.ai_wins = 0
        self.total_games = 0
        self.player_high = 0
        self.ai_high = 0


class GamePlayer:
    def __init__(self, name, is_ai=False):
        self.name = name
        self.is_ai = is_ai
        self.model = None
        self.use_ppo = False
        
        try:
            self.env = gym.make("FlappyBird-v0", render_mode="rgb_array", use_lidar=True)
            self.obs, _ = self.env.reset()
        except:
            self.env = None
            self.obs = None
        
        self.score = 0
        self.done = False
        self.frame_buffer = None
        
        # Try to load PPO model if AI
        if self.is_ai:
            self._load_ppo_model()
    
    def _load_ppo_model(self):
        """Try to load the best available PPO model"""
        if not PPO_AVAILABLE:
            print("[AI] PPO not available, using heuristic")
            return
            
        model_path = "models/ppo_flappybird_optimized_1000000_steps.zip"
        
        try:
            if os.path.exists(model_path):
                self.model = PPO.load(model_path)
                self.use_ppo = True
                print("[AI] Loaded optimized PPO model (1M steps)")
            else:
                print(f"[AI] Model not found at {model_path}, using heuristic")
        except Exception as e:
            print(f"[AI] Failed to load PPO model: {e}, using heuristic")
    
    def get_action(self, player_input=None):
        if self.is_ai:
            # Try PPO first
            if self.use_ppo and self.model is not None and self.obs is not None:
                try:
                    action, _ = self.model.predict(self.obs, deterministic=True)
                    return int(action)
                except:
                    pass
            
            # Fallback to heuristic
            if self.obs is not None and len(self.obs) >= 5:
                bird_y = self.obs[0]
                pipe_top = self.obs[3]
                pipe_bottom = self.obs[4]
                pipe_center = (pipe_top + pipe_bottom) / 2
                
                if bird_y > pipe_center + 0.08:
                    return 1
                if bird_y < pipe_center - 0.12:
                    return 0
                return 1 if np.random.random() < 0.18 else 0
            return 0
        else:
            return player_input if player_input is not None else 0
    
    def step(self, action):
        if self.env is None:
            self.done = True
            return True
        try:
            self.obs, _, done, truncated, info = self.env.step(action)
            self.score = info.get("score", 0)
            self.done = done or truncated
            return self.done
        except:
            self.done = True
            return True
    
    def render(self):
        if self.env is None:
            return self.frame_buffer
        try:
            frame = self.env.render()
            if frame is not None:
                self.frame_buffer = frame
            return self.frame_buffer
        except:
            return self.frame_buffer
    
    def reset(self):
        if self.env is not None:
            try:
                self.obs, _ = self.env.reset()
                self.score = 0
                self.done = False
            except:
                pass
    
    def close(self):
        if self.env is not None:
            try:
                self.env.close()
            except:
                pass
            self.env = None


class GameDisplay:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Flappy Bird: Human vs AI")
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 40)
        self.font_medium = pygame.font.Font(None, 30)
        self.font_small = pygame.font.Font(None, 20)
        self.running = True
        self.player_action = 0
    
    def handle_events(self):
        self.player_action = 0
        try:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_SPACE, pygame.K_UP, pygame.K_w):
                        self.player_action = 1
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.player_action = 1
        except pygame.error:
            # Re-initialize pygame if video system is lost
            try:
                pygame.quit()
                pygame.init()
                self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
                pygame.display.set_caption("Flappy Bird: Human vs AI")
            except:
                self.running = False
        return self.player_action
    
    def array_to_surface(self, array):
        if array is None:
            return None
        try:
            array = np.transpose(array, (1, 0, 2))
            surface = pygame.surfarray.make_surface(array)
            return pygame.transform.scale(surface, (GAME_WIDTH, GAME_HEIGHT))
        except:
            return None
    
    def draw(self, player, ai, stats, game_num, frame_num):
        try:
            self.screen.fill(GRAY)
            
            # Draw game frames
            player_surface = self.array_to_surface(player.render())
            ai_surface = self.array_to_surface(ai.render())
            
            # Draw backgrounds and frames
            pygame.draw.rect(self.screen, WHITE, (10, 10, GAME_WIDTH, GAME_HEIGHT))
            pygame.draw.rect(self.screen, BLUE, (10, 10, GAME_WIDTH, GAME_HEIGHT), 3)
            
            pygame.draw.rect(self.screen, WHITE, (GAME_WIDTH + 30, 10, GAME_WIDTH, GAME_HEIGHT))
            pygame.draw.rect(self.screen, BLUE, (GAME_WIDTH + 30, 10, GAME_WIDTH, GAME_HEIGHT), 3)
            
            # Blit game frames
            if player_surface:
                self.screen.blit(player_surface, (10, 10))
            if ai_surface:
                self.screen.blit(ai_surface, (GAME_WIDTH + 30, 10))
            
            # Draw titles and scores
            player_text = self.font_medium.render(f"PLAYER: {player.score}", True, BLACK)
            ai_text = self.font_medium.render(f"AI: {ai.score}", True, BLACK)
            
            self.screen.blit(player_text, (20, GAME_HEIGHT + 20))
            self.screen.blit(ai_text, (GAME_WIDTH + 40, GAME_HEIGHT + 20))
            
            # Draw status
            if player.done:
                status = self.font_small.render("CRASH", True, RED)
                self.screen.blit(status, (100, GAME_HEIGHT + 50))
            if ai.done:
                status = self.font_small.render("CRASH", True, RED)
                self.screen.blit(status, (GAME_WIDTH + 100, GAME_HEIGHT + 50))
            
            # Draw stats
            stats_text = self.font_small.render(
                f"Game {game_num} | Frame {frame_num} | Wins: P{stats.player_wins} A{stats.ai_wins} | High: P{stats.player_high} A{stats.ai_high}",
                True, BLACK
            )
            self.screen.blit(stats_text, (10, GAME_HEIGHT + 80))
            
            # Draw hint
            hint = self.font_small.render("Press SPACE or CLICK to JUMP", True, BLUE)
            self.screen.blit(hint, (WINDOW_WIDTH//2 - 140, WINDOW_HEIGHT - 30))
            
            pygame.display.flip()
        except pygame.error as e:
            print(f"Display error: {e}")
            # Try to recover
            try:
                pygame.quit()
                pygame.init()
                self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
            except:
                self.running = False
    
    def tick(self):
        self.clock.tick(FPS)
    
    def close(self):
        pygame.quit()


def main():
    pygame.init()
    display = GameDisplay()
    stats = GameStats()
    
    # Create players ONCE
    player = GamePlayer("Player", is_ai=False)
    ai = GamePlayer("AI", is_ai=True)
    
    game_num = 0
    
    try:
        while display.running and game_num < 100:
            game_num += 1
            
            print(f"Starting game {game_num}...")
            
            # Reset players instead of recreating
            player.reset()
            ai.reset()
            
            frame_num = 0
            frame_pause = 0
            
            # Single game loop
            while display.running:
                display.handle_events()
                
                if not display.running:
                    break
                
                frame_num += 1
                
                # Game steps
                if not player.done:
                    player.step(display.player_action)
                
                if not ai.done:
                    ai_action = ai.get_action()
                    ai.step(ai_action)
                
                # Collision detection
                if player.done or ai.done:
                    frame_pause += 1
                    if frame_pause > 90:  # 3 second pause
                        # Update stats
                        if player.score >= ai.score:
                            stats.player_wins += 1
                        else:
                            stats.ai_wins += 1
                        stats.total_games += 1
                        stats.player_high = max(stats.player_high, player.score)
                        stats.ai_high = max(stats.ai_high, ai.score)
                        print(f"Game {game_num} Over - Player: {player.score}, AI: {ai.score}")
                        break
                
                # Draw
                display.draw(player, ai, stats, game_num, frame_num)
                display.tick()
        
        print(f"\nFinal Stats - Player: {stats.player_wins} wins, AI: {stats.ai_wins} wins")
        print(f"Player High: {stats.player_high}, AI High: {stats.ai_high}")
        
        # Keep window open for 3 seconds
        for _ in range(90):
            display.handle_events()
            if not display.running:
                break
            display.tick()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if player is not None:
            player.close()
        if ai is not None:
            ai.close()
        display.close()


if __name__ == "__main__":
    main()
