# -*- coding: utf-8 -*-
import gymnasium as gym
import flappy_bird_gymnasium
import pygame
import numpy as np
from stable_baselines3 import PPO
import traceback

# ========================
# Config
CONFIG = {
    "MODEL_PATH": "ppo_flappybird_final",
    "FPS": 30,
    "SCREEN_WIDTH": 1400,
    "SCREEN_HEIGHT": 800,
    "FRAME_WIDTH": 450,
    "FRAME_HEIGHT": 600,
    "FRAME_GAP": 60,
    "HEADER_HEIGHT": 120,
    "SCOREBAR_H": 28,
    "SCOREBAR_PAD": 8,
    "PANEL_HEIGHT": 100,
    "PANEL_MARGIN_X": 40,
    "PANEL_MARGIN_Y": 20,
    "TITLE_Y": 40,
    "PLAYER_X": 30,
    "AI_X": 770,
    "FRAME_Y": 120,
    "BG_COLOR": (20, 20, 40),
    "TEXT_COLOR": (255, 255, 255),
    "ACCENT_COLOR": (0, 200, 255),
    "TITLE_COLOR": (255, 215, 0),
    "FONT_SIZE": 36,
    "SMALL_FONT_SIZE": 24,
    "TINY_FONT_SIZE": 18,
}


def apply_layout(config):
    frame_gap = config["FRAME_GAP"]
    frame_w = config["FRAME_WIDTH"]
    screen_w = config["SCREEN_WIDTH"]
    side_margin = max(20, (screen_w - (2 * frame_w) - frame_gap) // 2)
    config["PLAYER_X"] = side_margin
    config["AI_X"] = side_margin + frame_w + frame_gap
    config["FRAME_Y"] = config["HEADER_HEIGHT"]
    config["SCOREBAR_Y"] = config["FRAME_Y"] - config["SCOREBAR_H"] - config["SCOREBAR_PAD"]
    config["PANEL_Y"] = config["SCREEN_HEIGHT"] - config["PANEL_HEIGHT"] - config["PANEL_MARGIN_Y"]


apply_layout(CONFIG)


class GameStats:
    
    def __init__(self):
        self.player_high_score = 0
        self.ai_high_score = 0
        self.player_wins = 0
        self.ai_wins = 0
        self.total_games = 0
    
    def update(self, player_score, ai_score):
        if player_score >= ai_score:
            self.player_wins += 1
        else:
            self.ai_wins += 1
        
        self.player_high_score = max(self.player_high_score, player_score)
        self.ai_high_score = max(self.ai_high_score, ai_score)
        self.total_games += 1
    
    def get_stats(self):
        return {
            "player_high_score": self.player_high_score,
            "ai_high_score": self.ai_high_score,
            "player_wins": self.player_wins,
            "ai_wins": self.ai_wins,
            "total_games": self.total_games,
        }


class GamePlayer:
    
    def __init__(self, name, is_ai=False, model=None):
        self.name = name
        self.is_ai = is_ai
        self.model = model
        self.env = gym.make("FlappyBird-v0", render_mode="rgb_array", use_lidar=True)
        self.obs, _ = self.env.reset()
        self.score = 0
    
    def get_action(self):
        if self.is_ai:
            action, _ = self.model.predict(self.obs, deterministic=True)
            return int(action)
        return 0
    
    def step(self, action):
        self.obs, _, done, truncated, info = self.env.step(action)
        self.score = info.get("score", 0)
        return done or truncated

    def reset(self):
        self.obs, _ = self.env.reset()
        self.score = 0

    def reset_if_done(self, done):
        if done:
            self.obs, _ = self.env.reset()
    
    def render(self):
        return self.env.render()
    
    def close(self):
        self.env.close()


class GameDisplay:
    
    def __init__(self, config):
        self.config = config
        
        print(f"Pygame init before display: {pygame.get_init()}")
        if not pygame.get_init():
            print("Initializing pygame in GameDisplay...")
            pygame.init()
        
        print("Creating pygame display...")
        try:
            self.screen = pygame.display.set_mode(
                (config["SCREEN_WIDTH"], config["SCREEN_HEIGHT"])
            )
            print(f"Display created: {self.screen}")
        except Exception as e:
            print(f"Failed to create display: {e}")
            raise
        
        pygame.display.set_caption("Flappy Bird: Human vs AI Exhibition")
        self.clock = pygame.time.Clock()
        
        font_path = None
        try:
            font_path = "C:\\Windows\\Fonts\\msjh.ttc"
            self.font_large = pygame.font.Font(font_path, config["FONT_SIZE"])
            self.font = pygame.font.Font(font_path, config["SMALL_FONT_SIZE"])
            self.font_small = pygame.font.Font(font_path, config["TINY_FONT_SIZE"])
            print(f"Fonts loaded from {font_path}")
        except:
            try:
                font_path = "C:\\Windows\\Fonts\\msyh.ttc"
                self.font_large = pygame.font.Font(font_path, config["FONT_SIZE"])
                self.font = pygame.font.Font(font_path, config["SMALL_FONT_SIZE"])
                self.font_small = pygame.font.Font(font_path, config["TINY_FONT_SIZE"])
                print(f"Fonts loaded from {font_path}")
            except:
                print("Using default system fonts")
                self.font_large = pygame.font.SysFont("Arial", config["FONT_SIZE"], bold=True)
                self.font = pygame.font.SysFont("Arial", config["SMALL_FONT_SIZE"])
                self.font_small = pygame.font.SysFont("Arial", config["TINY_FONT_SIZE"])
    
    def render_frame(self, frame, target_width, target_height):
        surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        return pygame.transform.scale(surface, (target_width, target_height))
    
    def draw(self, player_surface, ai_surface, player_score, ai_score, 
             player_done, ai_done, stats, game_over_time=0):
        self.screen.fill(self.config["BG_COLOR"])
        
        title = self.font_large.render("Flappy Bird: Human vs AI", True, self.config["TITLE_COLOR"])
        title_rect = title.get_rect(center=(self.config["SCREEN_WIDTH"] // 2, self.config["TITLE_Y"]))
        self.screen.blit(title, title_rect)

        scorebar_y = self.config["SCOREBAR_Y"]
        scorebar_h = self.config["SCOREBAR_H"]
        panel_x = self.config["PANEL_MARGIN_X"]
        panel_y = self.config["PANEL_Y"]
        panel_w = self.config["SCREEN_WIDTH"] - (panel_x * 2)
        panel_h = self.config["PANEL_HEIGHT"]
        
        border_color = self.config["ACCENT_COLOR"]
        
        pygame.draw.rect(self.screen, border_color,
                        (self.config["PLAYER_X"] - 3, self.config["FRAME_Y"] - 3,
                         self.config["FRAME_WIDTH"] + 6, self.config["FRAME_HEIGHT"] + 6), 3)
        
        pygame.draw.rect(self.screen, border_color,
                        (self.config["AI_X"] - 3, self.config["FRAME_Y"] - 3,
                         self.config["FRAME_WIDTH"] + 6, self.config["FRAME_HEIGHT"] + 6), 3)
        
        self.screen.blit(player_surface, (self.config["PLAYER_X"], self.config["FRAME_Y"]))
        self.screen.blit(ai_surface, (self.config["AI_X"], self.config["FRAME_Y"]))
        
        player_bg = (100, 200, 100) if (not player_done and player_score >= ai_score) else (100, 150, 255)
        ai_bg = (100, 200, 100) if (not ai_done and ai_score > player_score) else (255, 100, 100)
        
        pygame.draw.rect(self.screen, player_bg,
                        (self.config["PLAYER_X"] - 3, scorebar_y, self.config["FRAME_WIDTH"] + 6, scorebar_h), 0)
        pygame.draw.rect(self.screen, (255, 255, 255),
                        (self.config["PLAYER_X"] - 3, scorebar_y, self.config["FRAME_WIDTH"] + 6, scorebar_h), 2)
        
        pygame.draw.rect(self.screen, ai_bg,
                        (self.config["AI_X"] - 3, scorebar_y, self.config["FRAME_WIDTH"] + 6, scorebar_h), 0)
        pygame.draw.rect(self.screen, (255, 255, 255),
                        (self.config["AI_X"] - 3, scorebar_y, self.config["FRAME_WIDTH"] + 6, scorebar_h), 2)
        
        player_text = self.font.render(f"Player - Score: {player_score}", True, (0, 0, 0))
        ai_text = self.font.render(f"AI Agent - Score: {ai_score}", True, (0, 0, 0))
        
        self.screen.blit(player_text, (self.config["PLAYER_X"] + 10, scorebar_y + 2))
        self.screen.blit(ai_text, (self.config["AI_X"] + 10, scorebar_y + 2))
        
        stats_data = stats.get_stats()
        pygame.draw.rect(self.screen, (50, 50, 80), (panel_x, panel_y, panel_w, panel_h))
        pygame.draw.rect(self.screen, self.config["ACCENT_COLOR"], 
                        (panel_x, panel_y, panel_w, panel_h), 2)
        
        stats_line1 = f"High Score - Player: {stats_data['player_high_score']}  |  AI: {stats_data['ai_high_score']}"
        stats_line2 = f"Record - Player Wins: {stats_data['player_wins']}  |  AI Wins: {stats_data['ai_wins']}  |  Total: {stats_data['total_games']}"
        
        text1 = self.font_small.render(stats_line1, True, (255, 255, 255))
        text2 = self.font_small.render(stats_line2, True, (255, 255, 255))
        
        self.screen.blit(text1, (panel_x + 20, panel_y + 15))
        self.screen.blit(text2, (panel_x + 20, panel_y + 55))
        
        if game_over_time > 0 and game_over_time < 120:
            result_text = self.font_large.render("GAME OVER", True, (255, 215, 0))
            result_rect = result_text.get_rect(center=(
                self.config["SCREEN_WIDTH"] // 2, self.config["SCREEN_HEIGHT"] // 2
            ))
            self.screen.blit(result_text, result_rect)
            
            if player_score > ai_score:
                winner_text = self.font.render("Player Wins!", True, (0, 255, 0))
            elif ai_score > player_score:
                winner_text = self.font.render("AI Wins!", True, (255, 100, 100))
            else:
                winner_text = self.font.render("Draw!", True, (255, 255, 0))
            
            winner_rect = winner_text.get_rect(center=(
                self.config["SCREEN_WIDTH"] // 2, self.config["SCREEN_HEIGHT"] // 2 + 80
            ))
            self.screen.blit(winner_text, winner_rect)
        
        pygame.display.flip()
    
    def draw_start_screen(self):
        self.screen.fill(self.config["BG_COLOR"])
        
        title = self.font_large.render("Flappy Bird Exhibition", True, self.config["TITLE_COLOR"])
        title_rect = title.get_rect(center=(self.config["SCREEN_WIDTH"] // 2, 100))
        self.screen.blit(title, title_rect)
        
        desc1 = self.font.render("Human vs AI Competition", True, self.config["ACCENT_COLOR"])
        desc1_rect = desc1.get_rect(center=(self.config["SCREEN_WIDTH"] // 2, 250))
        self.screen.blit(desc1, desc1_rect)
        
        desc2 = self.font.render("Powered by Reinforcement Learning", True, self.config["ACCENT_COLOR"])
        desc2_rect = desc2.get_rect(center=(self.config["SCREEN_WIDTH"] // 2, 320))
        self.screen.blit(desc2, desc2_rect)
        
        start_text = self.font_large.render("Press any key to start", True, (0, 255, 0))
        start_rect = start_text.get_rect(center=(self.config["SCREEN_WIDTH"] // 2, 500))
        self.screen.blit(start_text, start_rect)
        
        pygame.display.flip()
    
    def tick(self):
        self.clock.tick(self.config["FPS"])
    
    def close(self):
        pygame.quit()


def main():
    try:
        print("Initializing pygame...")
        pygame.init()
        print(f"Pygame initialized: {pygame.get_init()}")
        
        print("Creating display...")
        display = GameDisplay(CONFIG)
        print("Display created successfully!")
        
        stats = GameStats()
        
        running = True
        game_started = False
        
        print("Starting start screen loop...")
        while running and not game_started:
            print("Drawing start screen...")
            display.draw_start_screen()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                elif event.type in [pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN]:
                    game_started = True
            
            display.tick()
        
        if not running:
            display.close()
            return
        
        print("Loading AI model...")
        model = PPO.load(CONFIG["MODEL_PATH"])
        print("AI model loaded successfully!")

        player = GamePlayer("Player", is_ai=False)
        ai = GamePlayer("AI", is_ai=True, model=model)
        
        while running:
            player.reset()
            ai.reset()
            player_action = 0
            game_over_time = 0
            
            while running:
                player_action = 0
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        break
                    elif event.type == pygame.KEYDOWN:
                        if game_over_time == 0:
                            player_action = 1
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        if game_over_time == 0:
                            player_action = 1
                
                if not running:
                    break
                
                if game_over_time == 0:
                    
                    ai_action = ai.get_action()
                    
                    player_done = player.step(player_action)
                    ai_done = ai.step(ai_action)
                    
                    if player_done or ai_done:
                        game_over_time = 1
                        stats.update(player.score, ai.score)
                else:
                    game_over_time += 1
                    player_done = True
                    ai_done = True
                    
                    if game_over_time > 180:
                        break
                
                player_frame = player.render()
                ai_frame = ai.render()
                
                player_surface = display.render_frame(
                    player_frame, CONFIG["FRAME_WIDTH"], CONFIG["FRAME_HEIGHT"]
                )
                ai_surface = display.render_frame(
                    ai_frame, CONFIG["FRAME_WIDTH"], CONFIG["FRAME_HEIGHT"]
                )
                
                display.draw(player_surface, ai_surface, player.score, ai.score,
                            player_done, ai_done, stats, game_over_time)
                display.tick()
            
        player.close()
        ai.close()
        display.close()
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()



