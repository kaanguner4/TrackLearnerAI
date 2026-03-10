import pygame
import sys
from track_env import TrackEnvironment
from agent import Agent

# Renkler
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)

def main():
    pygame.init()
    
    # --- 1. EKRAN VE ÖLÇEKLEME AYARLARI ---
    infoObject = pygame.display.Info()
    monitor_w, monitor_h = infoObject.current_w, infoObject.current_h
    
    track_path = "assets/tracks/track1.png"
    env = TrackEnvironment(track_path)
    
    max_window_w = int(monitor_w * 0.85)
    max_window_h = int(monitor_h * 0.85)
    
    aspect_ratio = env.width / env.height
    if env.width > max_window_w or env.height > max_window_h:
        if (max_window_w / aspect_ratio) <= max_window_h:
            win_w = max_window_w
            win_h = int(max_window_w / aspect_ratio)
        else:
            win_h = max_window_h
            win_w = int(max_window_h * aspect_ratio)
    else:
        win_w = env.width
        win_h = env.height
        
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("TrackLearnerAI - Çizgi Kesişimi Sürüşü")
    
    virtual_surface = pygame.Surface((env.width, env.height))
    clock = pygame.time.Clock()
    
    # --- 2. OYUN NESNELERİNİ OLUŞTUR ---
    start_position= (env.finish_line['center'][0], env.finish_line['center'][1])
    agent = Agent(start_position, env)
    track_image = pygame.image.load(track_path).convert()

    hud_font = pygame.font.SysFont(None, 36)
    crash_font = pygame.font.SysFont(None, 72)
    cp_font = pygame.font.SysFont(None, 20) 
    
    running = True
    lap_count = 0

    while running:
        clock.tick(60)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- MANUEL KONTROLLER ---
        if agent.is_alive:
            keys = pygame.key.get_pressed()
            
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                agent.speed += agent.acceleration
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
                agent.speed -= agent.acceleration
            else:
                if agent.speed > 0:
                    agent.speed -= 0.1
                elif agent.speed < 0:
                    agent.speed += 0.1
                    
            agent.speed = max(-agent.max_speed / 2, min(agent.speed, agent.max_speed))
            
            if abs(agent.speed) > 0.1:
                if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                    agent.angle -= agent.turn_speed
                if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                    agent.angle += agent.turn_speed

        # Ajanı Güncelle 
        agent.update()

        # --- TUR (FINISH) SAYACI MANTIĞI ---
        # Eski finish_line crossed mantığı yerine çizgi kesişimi ile tur ölçümü yapıyoruz
        if env.is_finish_line_crossed(agent.prev_x, agent.prev_y, agent.x, agent.y):
            if agent.current_checkpoint == int(len(env.checkpoints)):
                lap_count += 1
                agent.current_checkpoint = 0 # Yeni tur!

        # --- ÇİZİM AŞAMASI (Sanal Yüzeye Yapılır) ---
        virtual_surface.blit(track_image, (0, 0))
        
        for i, cp in enumerate(env.checkpoints):
            # Checkpoint çizgilerini ("Kapıları") yeşil çizgi olarak çiz
            pygame.draw.line(virtual_surface, GREEN, cp['p1'], cp['p2'], 2)
            
            # Numaraları tam ortaya (merkeze) yaz
            num_text = cp_font.render(str(i), True, BLACK)
            center_x, center_y = cp['center']
            virtual_surface.blit(num_text, (center_x + 6, center_y - 8))
            
        agent.draw(virtual_surface)

        # --- HUD (BİLGİ EKRANI) ÇİZİMİ ---
        hud_bg = pygame.Surface((380, 380)) 
        hud_bg.set_alpha(180) 
        hud_bg.fill(BLACK)
        virtual_surface.blit(hud_bg, (10, 10))
        
        hud_texts = [
            f"FPS: {int(clock.get_fps())}",
            f"Geçilen Checkpoint: {agent.current_checkpoint} / {len(env.checkpoints)}",
            f"Tamamlanan Tur: {lap_count}",
            f"Hız: {agent.speed:.2f} px/frame"
        ]
        
        for i, text in enumerate(hud_texts):
            rendered_text = hud_font.render(text, True, WHITE)
            virtual_surface.blit(rendered_text, (20, 20 + (i * 35))) 

        # Kaza Durumu
        if not agent.is_alive:
            text1 = crash_font.render("KAZA YAPTIN!", True, RED)
            text2 = hud_font.render("(Yeniden baslatmak icin dosyayi tekrar calistir)", True, BLACK)
            virtual_surface.blit(text1, (env.width//2 - text1.get_width()//2, env.height//2 - 50))
            virtual_surface.blit(text2, (env.width//2 - text2.get_width()//2, env.height//2 + 20))

        # --- EKRANA YANSITMA (Ölçekleme) ---
        scaled_surface = pygame.transform.smoothscale(virtual_surface, (win_w, win_h))
        screen.blit(scaled_surface, (0, 0))

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()