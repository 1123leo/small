import gymnasium as gym
import flappy_bird_gymnasium
import pygame
import numpy as np
from stable_baselines3 import PPO
import traceback

# ========================
# 設定常數
# ========================
CONFIG = {
    "MODEL_PATH": "ppo_flappybird_final",
    "FPS": 30,
    "SCREEN_WIDTH": 1400,
    "SCREEN_HEIGHT": 800,
    "FRAME_WIDTH": 450,
    "FRAME_HEIGHT": 600,
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


class GameStats:
    """遊戲統計追蹤"""
    
    def __init__(self):
        self.player_high_score = 0
        self.ai_high_score = 0
        self.player_wins = 0
        self.ai_wins = 0
        self.total_games = 0
    
    def update(self, player_score, ai_score):
        """更新統計"""
        if player_score >= ai_score:
            self.player_wins += 1
        else:
            self.ai_wins += 1
        
        self.player_high_score = max(self.player_high_score, player_score)
        self.ai_high_score = max(self.ai_high_score, ai_score)
        self.total_games += 1
    
    def get_stats(self):
        """返回統計資料"""
        return {
            "player_high_score": self.player_high_score,
            "ai_high_score": self.ai_high_score,
            "player_wins": self.player_wins,
            "ai_wins": self.ai_wins,
            "total_games": self.total_games,
        }


class GamePlayer:
    """遊戲玩家"""
    
    def __init__(self, name, is_ai=False, model=None):
        self.name = name
        self.is_ai = is_ai
        self.model = model
        self.env = gym.make("FlappyBird-v0", render_mode="rgb_array", use_lidar=True)
        self.obs, _ = self.env.reset()
        self.score = 0
    
    def get_action(self):
        """獲取下一個動作"""
        if self.is_ai:
            action, _ = self.model.predict(self.obs, deterministic=True)
            return int(action)
        return 0
    
    def step(self, action):
        """執行一步"""
        self.obs, _, done, truncated, info = self.env.step(action)
        self.score = info.get("score", 0)
        return done or truncated
    
    def reset_if_done(self, done):
        """如果遊戲結束則重新開始"""
        if done:
            self.obs, _ = self.env.reset()
    
    def render(self):
        """獲取畫面"""
        return self.env.render()
    
    def close(self):
        """關閉環境"""
        self.env.close()


class GameDisplay:
    """遊戲顯示管理"""
    
    def __init__(self, config):
        self.config = config
        
        # 確保 pygame 已初始化
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
        
        # 設置字體 - 優先使用中文字體
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
                # 使用預設字體
                print("Using default system fonts")
                self.font_large = pygame.font.SysFont("Arial", config["FONT_SIZE"], bold=True)
                self.font = pygame.font.SysFont("Arial", config["SMALL_FONT_SIZE"])
                self.font_small = pygame.font.SysFont("Arial", config["TINY_FONT_SIZE"])
    
    def render_frame(self, frame, target_width, target_height):
        """將 numpy array 轉換為 pygame surface"""
        surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        return pygame.transform.scale(surface, (target_width, target_height))
    
    def draw(self, player_surface, ai_surface, player_score, ai_score, 
             player_done, ai_done, stats, game_over_time=0):
        """繪製遊戲畫面"""
        # 清空畫面
        self.screen.fill(self.config["BG_COLOR"])
        
        # 標題
        title = self.font_large.render("Flappy Bird: Human vs AI", True, self.config["TITLE_COLOR"])
        title_rect = title.get_rect(center=(self.config["SCREEN_WIDTH"] // 2, 40))
        self.screen.blit(title, title_rect)
        
        # 邊框和遊戲畫面
        border_color = self.config["ACCENT_COLOR"]
        
        pygame.draw.rect(self.screen, border_color,
                        (self.config["PLAYER_X"] - 3, self.config["FRAME_Y"] - 3,
                         self.config["FRAME_WIDTH"] + 6, self.config["FRAME_HEIGHT"] + 6), 3)
        
        pygame.draw.rect(self.screen, border_color,
                        (self.config["AI_X"] - 3, self.config["FRAME_Y"] - 3,
                         self.config["FRAME_WIDTH"] + 6, self.config["FRAME_HEIGHT"] + 6), 3)
        
        # 遊戲畫面
        self.screen.blit(player_surface, (self.config["PLAYER_X"], self.config["FRAME_Y"]))
        self.screen.blit(ai_surface, (self.config["AI_X"], self.config["FRAME_Y"]))
        
        # 玩家標籤
        player_bg = (100, 200, 100) if (not player_done and player_score >= ai_score) else (100, 150, 255)
        ai_bg = (100, 200, 100) if (not ai_done and ai_score > player_score) else (255, 100, 100)
        
        pygame.draw.rect(self.screen, player_bg,
                        (self.config["PLAYER_X"] - 3, 105, self.config["FRAME_WIDTH"] + 6, 25), 0)
        pygame.draw.rect(self.screen, (255, 255, 255),
                        (self.config["PLAYER_X"] - 3, 105, self.config["FRAME_WIDTH"] + 6, 25), 2)
        
        pygame.draw.rect(self.screen, ai_bg,
                        (self.config["AI_X"] - 3, 105, self.config["FRAME_WIDTH"] + 6, 25), 0)
        pygame.draw.rect(self.screen, (255, 255, 255),
                        (self.config["AI_X"] - 3, 105, self.config["FRAME_WIDTH"] + 6, 25), 2)
        
        # 分數文字
        player_text = self.font.render(f"Player - Score: {player_score}", True, (0, 0, 0))
        ai_text = self.font.render(f"AI Agent - Score: {ai_score}", True, (0, 0, 0))
        
        self.screen.blit(player_text, (self.config["PLAYER_X"] + 10, 107))
        self.screen.blit(ai_text, (self.config["AI_X"] + 10, 107))
        
        # 統計面板
        stats_data = stats.get_stats()
        panel_y = self.config["SCREEN_HEIGHT"] - 110
        
        pygame.draw.rect(self.screen, (50, 50, 80), (30, panel_y, self.config["SCREEN_WIDTH"] - 60, 100))
        pygame.draw.rect(self.screen, self.config["ACCENT_COLOR"], 
                        (30, panel_y, self.config["SCREEN_WIDTH"] - 60, 100), 2)
        
        stats_line1 = f"High Score - Player: {stats_data['player_high_score']}  |  AI: {stats_data['ai_high_score']}"
        stats_line2 = f"Record - Player Wins: {stats_data['player_wins']}  |  AI Wins: {stats_data['ai_wins']}  |  Total: {stats_data['total_games']}"
        
        text1 = self.font_small.render(stats_line1, True, (255, 255, 255))
        text2 = self.font_small.render(stats_line2, True, (255, 255, 255))
        
        self.screen.blit(text1, (50, panel_y + 15))
        self.screen.blit(text2, (50, panel_y + 55))
        
        # 結束提示
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
        """繪製開始畫面"""
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
        """更新計時"""
        self.clock.tick(self.config["FPS"])
    
    def close(self):
        """關閉"""
        pygame.quit()


def main():
    """主程式"""
    try:
        # 初始化 pygame
        print("Initializing pygame...")
        pygame.init()
        print(f"Pygame initialized: {pygame.get_init()}")
        
        # 初始化顯示
        print("Creating display...")
        display = GameDisplay(CONFIG)
        print("Display created successfully!")
        
        stats = GameStats()
        
        # 顯示開始畫面
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
        
        # 載入 AI 模型
        print("Loading AI model...")
        model = PPO.load(CONFIG["MODEL_PATH"])
        print("AI model loaded successfully!")
        
        # 主遊戲迴圈
        player_action = 0
        
        while running:
            player = GamePlayer("Player", is_ai=False)
            ai = GamePlayer("AI", is_ai=True, model=model)
            
            game_over_time = 0
            
            # 單場遊戲迴圈
            while running:
                # 處理輸入
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
                
                # 處理遊戲邏輯
                if game_over_time == 0:
                    # 重置每幀的動作
                    player_action = 0
                    
                    # 獲取輸入
                    keys = pygame.key.get_pressed()
                    if keys[pygame.K_SPACE]:
                        player_action = 1
                    
                    # 獲取 AI 動作
                    ai_action = ai.get_action()
                    
                    # 執行遊戲步驟
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
                
                # 渲染
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
