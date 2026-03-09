import pygame
import sys
import math
from track_env import TrackEnvironment
from agent import Agent

# Renkler
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

def main():
    pygame.init()
    
    # --- 1. EKRAN VE ÖLÇEKLEME AYARLARI ---
    # Monitörünüzün çözünürlüğünü alıyoruz
    infoObject = pygame.display.Info()
    monitor_w, monitor_h = infoObject.current_w, infoObject.current_h
    
    # Çevreyi Yükle
    track_path = "assets/tracks/track1.png"
    env = TrackEnvironment(track_path)
    
    # Pencerenin monitörünüzden taşmaması için maksimum limit (Monitörün %85'i)
    max_window_w = int(monitor_w * 0.85)
    max_window_h = int(monitor_h * 0.85)
    
    # Görüntü oranını (aspect ratio) bozmadan yeni pencere boyutunu hesapla
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
        
    # Pygame Ekranını Ayarla (Ölçeklenmiş yeni boyutlarla)
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("TrackLearnerAI - Manuel Test Sürüşü")
    
    # Tüm çizimleri yapacağımız, orijinal pist boyutundaki "Sanal Yüzey"
    virtual_surface = pygame.Surface((env.width, env.height))
    
    clock = pygame.time.Clock()
    
    # --- 2. OYUN NESNELERİNİ OLUŞTUR ---
    agent = Agent((800, 400), env)
    track_image = pygame.image.load(track_path).convert()

    # Yazı tipleri
    hud_font = pygame.font.SysFont(None, 36)
    crash_font = pygame.font.SysFont(None, 72)
    
    running = True
    
    # Tur (Lap) sayacı değişkenleri
    lap_count = 0
    was_on_finish_line = False

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

        # --- CHECKPOINT KONTROL MANTIĞI ---
        if agent.current_checkpoint < len(env.checkpoints):
            # Hedefimiz olan sıradaki checkpointin (x, y) koordinatlarını al
            target_cp = env.checkpoints[agent.current_checkpoint]
            
            # Ajanın merkezi ile checkpoint arasındaki mesafeyi ölç
            dist = math.hypot(agent.x - target_cp[0], agent.y - target_cp[1])
            
            # Eğer ajan checkpoint'e belli bir piksel kadar yaklaştıysa (Örn: 60 piksel)
            # Bu değeri pistinizin genişliğine göre 40, 50, 80 gibi ayarlayabilirsiniz.
            if dist < 60: 
                agent.current_checkpoint += 1
        

        # --- TUR (FINISH) SAYACI MANTIĞI ---
        is_on_finish_line = env.is_finish_line_crossed(agent.x, agent.y)
        # Sadece bir önceki karede çizgide değilsen ve şimdi çizgideysen turu artır
        if is_on_finish_line and not was_on_finish_line:
            lap_count += 1
            agent.current_checkpoint = 0 # Checkpoint sıfırla, yeni tura başla
        was_on_finish_line = is_on_finish_line

        # --- ÇİZİM AŞAMASI (Sanal Yüzeye Yapılır) ---
        virtual_surface.blit(track_image, (0, 0))
        
        for cp in env.checkpoints:
            pygame.draw.circle(virtual_surface, RED, cp, 4)
            
        agent.draw(virtual_surface)

        # --- HUD (BİLGİ EKRANI) ÇİZİMİ ---
        # Okunabilirlik için yarı saydam siyah bir arka plan oluştur
        hud_bg = pygame.Surface((380, 120))
        hud_bg.set_alpha(180) # Şeffaflık seviyesi (0-255)
        hud_bg.fill(BLACK)
        virtual_surface.blit(hud_bg, (10, 10))
        
        # Yazıları oluştur
        hud_texts = [
            f"FPS: {int(clock.get_fps())}",
            f"Geçilen Checkpoint: {agent.current_checkpoint} / {len(env.checkpoints)}",
            f"Tamamlanan Tur: {lap_count}"
        ]
        
        for i, text in enumerate(hud_texts):
            rendered_text = hud_font.render(text, True, WHITE)
            virtual_surface.blit(rendered_text, (20, 20 + (i * 35))) # Yukarıdan aşağı hizala

        # Kaza Durumu
        if not agent.is_alive:
            text1 = crash_font.render("KAZA YAPTIN!", True, RED)
            text2 = hud_font.render("(Yeniden baslatmak icin dosyayi tekrar calistir)", True, BLACK)
            virtual_surface.blit(text1, (env.width//2 - text1.get_width()//2, env.height//2 - 50))
            virtual_surface.blit(text2, (env.width//2 - text2.get_width()//2, env.height//2 + 20))

        # --- EKRANA YANSITMA (Ölçekleme) ---
        # Sanal yüzeyi, başta hesapladığımız monitöre sığan boyutlara (win_w, win_h) küçült/büyüt
        scaled_surface = pygame.transform.smoothscale(virtual_surface, (win_w, win_h))
        screen.blit(scaled_surface, (0, 0))

        # Ekranı Yenile
        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()