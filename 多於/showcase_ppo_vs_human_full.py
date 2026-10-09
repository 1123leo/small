import gymnasium as gym
import flappy_bird_gymnasium
import pygame
import numpy as np
from stable_baselines3 import PPO

# ========================
# 設定常數
# ========================
CONFIG = {
    "MODEL_PATH": "ppo_flappybird_final",
    "FPS": 30,
    "AI_SPEED_MULTIPLIER": 2,
    "SCREEN_WIDTH": 1400,
    "SCREEN_HEIGHT": 800,
    "FRAME_WIDTH": 450,
    "FRAME_HEIGHT": 600,
    "PLAYER_X": 30,
    "AI_X": 770,
    "FRAME_Y": 120,
    "BG_COLOR": (20, 20, 40),  # 深藍色背景，更專業
    "TEXT_COLOR": (255, 255, 255),
    "ACCENT_COLOR": (0, 200, 255),  # 亮藍色強調
    "TITLE_COLOR": (255, 215, 0),  # 金色標題
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
    """遊戲玩家（可以是人類或 AI）"""
    
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

    def reset(self):
        self.obs, _ = self.env.reset()
        self.score = 0
    
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
        if not pygame.get_init():
            pygame.init()
        pygame.key.set_repeat(0)
        
        self.screen = pygame.display.set_mode(
            (config["SCREEN_WIDTH"], config["SCREEN_HEIGHT"])
        )
        pygame.display.set_caption("Flappy Bird: Human vs AI")
        self.clock = pygame.time.Clock()
        
        # 使用 Windows 系統中文字體
        try:
            # 嘗試使用微軟正黑體或其他中文字體
            self.font_large = pygame.font.Font("C:\\Windows\\Fonts\\msjh.ttc", config["FONT_SIZE"])
            self.font = pygame.font.Font("C:\\Windows\\Fonts\\msjh.ttc", config["SMALL_FONT_SIZE"])
            self.font_small = pygame.font.Font("C:\\Windows\\Fonts\\msjh.ttc", config["TINY_FONT_SIZE"])
        except:
            try:
                # 備選：微軟雅黑
                self.font_large = pygame.font.Font("C:\\Windows\\Fonts\\msyh.ttc", config["FONT_SIZE"])
                self.font = pygame.font.Font("C:\\Windows\\Fonts\\msyh.ttc", config["SMALL_FONT_SIZE"])
                self.font_small = pygame.font.Font("C:\\Windows\\Fonts\\msyh.ttc", config["TINY_FONT_SIZE"])
            except:
                # 如果都失敗，使用系統預設（會轉為英文字體）
                self.font_large = pygame.font.SysFont("Arial", config["FONT_SIZE"], bold=True)
                self.font = pygame.font.SysFont("Arial", config["SMALL_FONT_SIZE"])
                self.font_small = pygame.font.SysFont("Arial", config["TINY_FONT_SIZE"])
    
    def render_frame(self, frame, target_width, target_height):
        """將 numpy array 轉換為 pygame surface 並縮放"""
        surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        return pygame.transform.scale(surface, (target_width, target_height))
    
    def draw_title(self):
        """繪製標題區域"""
        # 標題背景
        pygame.draw.rect(self.screen, self.config["ACCENT_COLOR"], 
                        (0, 0, self.config["SCREEN_WIDTH"], 90))
        pygame.draw.line(self.screen, self.config["TITLE_COLOR"],
                        (0, 90), (self.config["SCREEN_WIDTH"], 90), 3)
        
        # 主標題
        title_text = self.font_large.render(
            "Flappy Bird: Human vs AI", True, self.config["TITLE_COLOR"]
        )
        title_rect = title_text.get_rect(center=(self.config["SCREEN_WIDTH"] // 2, 50))
        self.screen.blit(title_text, title_rect)
    
    def draw_player_labels(self, player_done, ai_done, player_score, ai_score):
        """繪製玩家標籤"""
        # 玩家標籤背景
        player_bg_color = (100, 200, 100) if (not player_done and player_score > ai_score) else (100, 150, 255)
        ai_bg_color = (100, 200, 100) if (not ai_done and ai_score > player_score) else (255, 100, 100)
        
        # 玩家邊框和背景
        pygame.draw.rect(self.screen, player_bg_color,
                        (self.config["PLAYER_X"] - 5, 105, 
                         self.config["FRAME_WIDTH"] + 10, 25), 0)
        pygame.draw.rect(self.screen, (255, 255, 255),
                        (self.config["PLAYER_X"] - 5, 105,
                         self.config["FRAME_WIDTH"] + 10, 25), 2)
        
        # AI 邊框和背景
        pygame.draw.rect(self.screen, ai_bg_color,
                        (self.config["AI_X"] - 5, 105,
                         self.config["FRAME_WIDTH"] + 10, 25), 0)
        pygame.draw.rect(self.screen, (255, 255, 255),
                        (self.config["AI_X"] - 5, 105,
                         self.config["FRAME_WIDTH"] + 10, 25), 2)
        
        # 玩家名稱和分數
        player_text = self.font.render(
            f"Player (You) - Score: {player_score}", True, (0, 0, 0)
        )
        ai_text = self.font.render(
            f"AI Agent - Score: {ai_score}", True, (0, 0, 0)
        )
        
        self.screen.blit(player_text, (self.config["PLAYER_X"], 107))
        self.screen.blit(ai_text, (self.config["AI_X"], 107))
        
        # 遊戲結束提示
        if player_done:
            status_text = self.font_small.render("GAME OVER", True, (255, 100, 100))
            self.screen.blit(status_text, (self.config["PLAYER_X"] + 140, self.config["FRAME_Y"] + 280))
        
        if ai_done:
            status_text = self.font_small.render("GAME OVER", True, (255, 100, 100))
            self.screen.blit(status_text, (self.config["AI_X"] + 140, self.config["FRAME_Y"] + 280))
    
    def draw_stats_panel(self, stats):
        """繪製統計面板"""
        panel_x = 30
        panel_y = self.config["SCREEN_HEIGHT"] - 110
        panel_width = self.config["SCREEN_WIDTH"] - 60
        
        # 面板背景
        pygame.draw.rect(self.screen, (50, 50, 80), (panel_x, panel_y, panel_width, 100))
        pygame.draw.rect(self.screen, self.config["ACCENT_COLOR"], 
                        (panel_x, panel_y, panel_width, 100), 2)
        
        # 統計信息
        stats_data = stats.get_stats()
        
        line1 = f"High Score - Player: {stats_data['player_high_score']}  |  AI: {stats_data['ai_high_score']}"
        line2 = f"Record - Player Wins: {stats_data['player_wins']}  |  AI Wins: {stats_data['ai_wins']}  |  Total Games: {stats_data['total_games']}"
        
        text1 = self.font_small.render(line1, True, self.config["TEXT_COLOR"])
        text2 = self.font_small.render(line2, True, self.config["TEXT_COLOR"])
        
        self.screen.blit(text1, (panel_x + 15, panel_y + 15))
        self.screen.blit(text2, (panel_x + 15, panel_y + 55))
    
    def draw_start_screen(self):
        """繪製開始畫面"""
        self.screen.fill(self.config["BG_COLOR"])
        self.draw_title()
        
        # 歡迎訊息
        welcome_text = self.font_large.render(
            "Welcome to Flappy Bird Exhibition", True, self.config["TITLE_COLOR"]
        )
        welcome_rect = welcome_text.get_rect(center=(
            self.config["SCREEN_WIDTH"] // 2,
            self.config["SCREEN_HEIGHT"] // 2 - 100
        ))
        self.screen.blit(welcome_text, welcome_rect)
        
        # 說明文字
        desc1 = self.font.render(
            "Experience Human vs AI Flappy Bird Competition", True, self.config["ACCENT_COLOR"]
        )
        desc1_rect = desc1.get_rect(center=(
            self.config["SCREEN_WIDTH"] // 2,
            self.config["SCREEN_HEIGHT"] // 2
        ))
        self.screen.blit(desc1, desc1_rect)
        
        desc2 = self.font.render(
            "AI trained with Reinforcement Learning", True, self.config["ACCENT_COLOR"]
        )
        desc2_rect = desc2.get_rect(center=(
            self.config["SCREEN_WIDTH"] // 2,
            self.config["SCREEN_HEIGHT"] // 2 + 50
        ))
        self.screen.blit(desc2, desc2_rect)
        
        # 開始按鍵提示
        start_text = self.font_large.render(
            "Press any key or click to start", True, (0, 255, 0)
        )
        start_rect = start_text.get_rect(center=(
            self.config["SCREEN_WIDTH"] // 2,
            self.config["SCREEN_HEIGHT"] // 2 + 150
        ))
        self.screen.blit(start_text, start_rect)
        
        pygame.display.flip()
    
    def draw(self, player_surface, ai_surface, player_score, ai_score, 
             player_done, ai_done, stats, game_over_time=0):
        """繪製整個遊戲畫面"""
        # 清空畫面
        self.screen.fill(self.config["BG_COLOR"])
        
        # 繪製標題
        self.draw_title()
        
        # 繪製遊戲區域邊框
        border_color = self.config["ACCENT_COLOR"]
        border_width = 3
        
        pygame.draw.rect(self.screen, border_color,
                        (self.config["PLAYER_X"] - border_width,
                         self.config["FRAME_Y"] - border_width,
                         self.config["FRAME_WIDTH"] + 2 * border_width,
                         self.config["FRAME_HEIGHT"] + 2 * border_width),
                        border_width)
        
        pygame.draw.rect(self.screen, border_color,
                        (self.config["AI_X"] - border_width,
                         self.config["FRAME_Y"] - border_width,
                         self.config["FRAME_WIDTH"] + 2 * border_width,
                         self.config["FRAME_HEIGHT"] + 2 * border_width),
                        border_width)
        
        # 繪製遊戲畫面
        self.screen.blit(player_surface, (self.config["PLAYER_X"], self.config["FRAME_Y"]))
        self.screen.blit(ai_surface, (self.config["AI_X"], self.config["FRAME_Y"]))
        
        # 繪製玩家標籤
        self.draw_player_labels(player_done, ai_done, player_score, ai_score)
        
        # 繪製統計面板
        self.draw_stats_panel(stats)
        
        # 遊戲結束時的暫停提示
        if game_over_time > 0:
            if game_over_time < 120:  # 暫停 4 秒（30 FPS * 4 = 120 frames）
                result_text = self.font_large.render("GAME OVER", True, (255, 215, 0))
                result_rect = result_text.get_rect(center=(
                    self.config["SCREEN_WIDTH"] // 2,
                    self.config["SCREEN_HEIGHT"] // 2
                ))
                self.screen.blit(result_text, result_rect)
                
                # 顯示獲勝者
                if player_score > ai_score:
                    winner_text = self.font.render("Player Wins!", True, (0, 255, 0))
                elif ai_score > player_score:
                    winner_text = self.font.render("AI Wins!", True, (255, 100, 100))
                else:
                    winner_text = self.font.render("Draw!", True, (255, 255, 0))
                
                winner_rect = winner_text.get_rect(center=(
                    self.config["SCREEN_WIDTH"] // 2,
                    self.config["SCREEN_HEIGHT"] // 2 + 80
                ))
                self.screen.blit(winner_text, winner_rect)
        
        pygame.display.flip()
    
    def tick(self):
        """更新畫面計時"""
        self.clock.tick(self.config["FPS"])
    
    def close(self):
        """關閉 pygame"""
        pygame.quit()


def handle_input():
    """處理輸入事件，返回 (should_continue, action, should_start)"""
    action = 0
    should_continue = True
    should_start = False
    
    try:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False, 0, False
            elif event.type == pygame.KEYDOWN:
                should_start = True
                if event.key == pygame.K_SPACE and not getattr(event, "repeat", False):
                    action = 1
            elif event.type == pygame.MOUSEBUTTONDOWN:
                action = 1
                should_start = True
    except:
        # pygame 未初始化時忽略錯誤
        pass
    
    return should_continue, action, should_start


def main():
    """主程式"""
    # 初始化
    display = GameDisplay(CONFIG)
    stats = GameStats()
    
    # 顯示開始畫面
    running = True
    game_started = False
    
    while running and not game_started:
        display.draw_start_screen()
        
        should_continue, action, should_start = handle_input()
        if not should_continue:
            running = False
            break
        
        if should_start:
            game_started = True
        
        display.tick()
    
    # 載入模型
    model = PPO.load(CONFIG["MODEL_PATH"])
    player = GamePlayer("Player", is_ai=False)
    ai = GamePlayer("AI", is_ai=True, model=model)
    
    ai_speed_multiplier = max(1, int(CONFIG.get("AI_SPEED_MULTIPLIER", 1)))

    # 主遊戲迴圈
    while running:
        # 每輪遊戲
        player.reset()
        ai.reset()
        
        game_over_time = 0
        
        # 單場遊戲迴圈
        while running:
            # 處理輸入
            should_continue, player_action, _ = handle_input()
            if not should_continue:
                running = False
                break
            
            # 如果遊戲已結束，等待後開始新遊戲
            if game_over_time > 0:
                game_over_time += 1
                if game_over_time > 180:  # 暫停 6 秒後自動開始新遊戲
                    break
            
            # 只有遊戲未結束才執行遊戲邏輯
            if game_over_time == 0:
                # 人類維持原速：每幀一步
                player_done = player.step(player_action)
                # AI 加速：每幀多步
                ai_done = False
                for _ in range(ai_speed_multiplier):
                    ai_action = ai.get_action()
                    ai_done = ai.step(ai_action)
                    if ai_done:
                        break
                
                # 檢查是否遊戲結束
                if player_done or ai_done:
                    game_over_time = 1
                    stats.update(player.score, ai.score)
            else:
                player_done = True
                ai_done = True
            
            # 獲取畫面並縮放
            player_frame = player.render()
            ai_frame = ai.render()
            
            player_surface = display.render_frame(
                player_frame, CONFIG["FRAME_WIDTH"], CONFIG["FRAME_HEIGHT"]
            )
            ai_surface = display.render_frame(
                ai_frame, CONFIG["FRAME_WIDTH"], CONFIG["FRAME_HEIGHT"]
            )
            
            # 繪製畫面
            display.draw(player_surface, ai_surface, player.score, ai.score,
                        player_done, ai_done, stats, game_over_time)
            display.tick()
        
    # 清理資源
    player.close()
    ai.close()
    display.close()


if __name__ == "__main__":
    main()
