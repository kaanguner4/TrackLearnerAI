import cv2
import numpy as np
import math

class TrackEnviroment:
    def __init__(self, image_path):
        # Görseli OpenCV ile okuyoruz
        self.track_image = cv2.imread(image_path)
        self.height, self.width, _ = self.track_image.shape
    
        # Yapay zekanın kullanacağı matrisler (maskeler)
        self.border_mask = None
        self.checkpoint_mask = None
        self.finish_line_mask = None
        self.checkpoints = [] # Sıralı checkpoint koordinatları

        self._process_track()
    
    def _process_track(self):
        # Renkleri daha net ayırmak için BGR'dan HSV renk uzayına geçiyoruz
        hsv_img = cv2.cvtColor(self.track_image, cv2.COLOR_BGR2HSV)

        # 1. SİYAJ RENK: Sınırları belirlemek için
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 50])
        self.border_mask = cv2.inRange(hsv_img, lower_black, upper_black)

        # 2. MAVİ RENK: Checkpoint'leri belirlemek için
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
        self.checkpoint_mask = cv2.inRange(hsv_img, lower_blue, upper_blue)

        # Mavi çizgilerin her birinin merkez koordinatını hesaplama
        contours, _ = cv2.findContours(self.checkpoint_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        raw_checkpoints = []
        for contour in contours:
            if cv2.contourArea(contour) > 10: # Küçük gürültüleri filtrelemek için alan sınırı
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    contour_x = int(M["m10"] / M["m00"])
                    contour_y = int(M["m01"] / M["m00"])
                    raw_checkpoints.append((contour_x, contour_y))

        # 3. Kırmızı Çizgiyi (Başlangıç/Bitiş) Bulma
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])
        self.finish_line_mask = cv2.bitwise_or(cv2.inRange(hsv_img, lower_red1, upper_red1), cv2.inRange(hsv_img, lower_red2, upper_red2))

        # Kırmızı çizginin merkez koordinatını bulma
        contours, _ = cv2.findContours(self.finish_line_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                self.finish_line = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # 4. Checkpoint'leri sıralama (finish line'a olan uzaklığa göre) / (Saat yönünde dizmek için merkeze göre açı hesaplıyoruz)
        center_x, center_y = self.width // 2, self.height // 2

        def angle_from_center(point):
            x, y = point
            # atan2 fonksiyonu, merkeze göre açıyı radyan cinsinden verir (-pi ile pi arası)
            return math.atan2(y - center_y, x - center_x)

        # Checkpoint'leri açısal olarak sıralama
        self.checkpoints = sorted(raw_checkpoints, key=angle_from_center, reverse=False)

    def check_collision(self, x , y):
        # Ajan resim sınırları içinde mi? Değilse direkt çarpmış say.
        if 0 <= int(x) < self.width and 0 <= int(y) < self.height:
            return self.border_mask[int(y), int(x)] == 255 # Sınır maskesinde beyaz (255) ise çarpışma var demektir
        return True # Resim sınırları dışında ise çarpışma var
    
    def is_checkpoint_passed(self, agent_x, agent_y, target_checkpoint_index):
        # Araba hedef checkpoint'in etki alanına (örn: 25 piksel yakınına) girdi mi?
        if target_checkpoint_index < len(self.checkpoints):
            checkpoint_x, checkpoint_y = self.checkpoints[target_checkpoint_index]
            distance = math.hypot(agent_x - checkpoint_x, agent_y - checkpoint_y)
            if distance < 25: # Checkpoint'e 25 piksel yaklaşıldığında geçilmiş sayılır
                return True
        return False
    
    def is_finish_line_crossed(self, agent_x, agent_y):
        # Arava bitiş çizgisini geçti mi? (Çizgiye 25 piksel yaklaşıldığında geçilmiş sayılır)
        if self.finish_line is not None:
            distance = math.hypot(agent_x - self.finish_line[0], agent_y - self.finish_line[1])
            if distance < 25:
                return True
        return False
    
    
print("TrackEnviroment sınıfı başarıyla tanımlandı.")
print("checkpoint koordinatları:", TrackEnviroment("assets/tracks/track1.png").checkpoints)