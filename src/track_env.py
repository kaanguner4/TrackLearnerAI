import cv2
import numpy as np
import math
import os

class TrackEnvironment:
    def __init__(self, image_path):
        # 1. Dosya yolunun fiziksel olarak var olup olmadığını kontrol et
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Hata: Belirtilen dosya yolu bulunamadı -> '{image_path}'")
            
        self.track_image = cv2.imread(image_path)
        
        # 2. Dosya mevcut olsa bile OpenCV tarafından geçerli bir resim olarak okunup okunamadığını kontrol et
        if self.track_image is None:
            raise ValueError(f"Hata: Görüntü okunamadı veya geçersiz/bozuk bir format -> '{image_path}'")
            
        self.height, self.width, _ = self.track_image.shape
    
        self.border_mask = None
        self.checkpoint_mask = None
        self.finish_line_mask = None
        # Checkpointler artık sadece merkez değil, çizginin iki ucunu da tutan sözlükler olacak
        self.checkpoints = []
        self.finish_line = None
        self.start_angle = 180  # Varsayilan, _process_track sonrasi hesaplanacak

        self._process_track()
        self._calculate_start_angle()
    
    def _process_track(self):
        hsv_img = cv2.cvtColor(self.track_image, cv2.COLOR_BGR2HSV)

        # 1. SİYAH RENK: Sınırlar
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 50])
        self.border_mask = cv2.inRange(hsv_img, lower_black, upper_black)

        # 2. MAVİ RENK: Checkpoint Çizgileri
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
        self.checkpoint_mask = cv2.inRange(hsv_img, lower_blue, upper_blue)

        contours, _ = cv2.findContours(self.checkpoint_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        raw_checkpoints = []
        for contour in contours:
            if cv2.contourArea(contour) > 10: 
                # Çizginin başlangıç ve bitiş noktalarını (en uzak iki nokta) bulalım
                hull = cv2.convexHull(contour)
                max_d = -1
                p1, p2 = (0,0), (0,0)
                for i in range(len(hull)):
                    for j in range(i+1, len(hull)):
                        d = math.hypot(hull[i][0][0] - hull[j][0][0], hull[i][0][1] - hull[j][0][1])
                        if d > max_d:
                            max_d = d
                            p1 = tuple(hull[i][0])
                            p2 = tuple(hull[j][0])

                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    raw_checkpoints.append({'center': (cx, cy), 'p1': p1, 'p2': p2})

        # 3. Kırmızı Çizgi (Bitiş)
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])
        self.finish_line_mask = cv2.bitwise_or(cv2.inRange(hsv_img, lower_red1, upper_red1), cv2.inRange(hsv_img, lower_red2, upper_red2))

        contours, _ = cv2.findContours(self.finish_line_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(largest_contour)
            max_d = -1
            p1, p2 = (0,0), (0,0)
            for i in range(len(hull)):
                for j in range(i+1, len(hull)):
                    d = math.hypot(hull[i][0][0] - hull[j][0][0], hull[i][0][1] - hull[j][0][1])
                    if d > max_d:
                        max_d = d
                        p1 = tuple(hull[i][0])
                        p2 = tuple(hull[j][0])

            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                self.finish_line = {'center': (cx, cy), 'p1': p1, 'p2': p2}

        # 4. Sıralama
        if raw_checkpoints and self.finish_line is not None:
            self.checkpoints = self._sort_checkpoints_by_path(raw_checkpoints, self.finish_line)
        else:
            self.checkpoints = raw_checkpoints

    def _sort_checkpoints_by_path(self, raw_checkpoints, start_point):
        remaining = list(raw_checkpoints)
        sorted_cps = []
        current_center = start_point['center']

        while remaining:
            nearest_idx = 0
            nearest_dist = math.hypot(remaining[0]['center'][0] - current_center[0], remaining[0]['center'][1] - current_center[1])
            for i in range(1, len(remaining)):
                d = math.hypot(remaining[i]['center'][0] - current_center[0], remaining[i]['center'][1] - current_center[1])
                if d < nearest_dist:
                    nearest_dist = d
                    nearest_idx = i

            nearest_cp = remaining.pop(nearest_idx)
            sorted_cps.append(nearest_cp)
            current_center = nearest_cp['center']

        signed_area = 0
        n = len(sorted_cps)
        for i in range(n):
            x1, y1 = sorted_cps[i]['center']
            x2, y2 = sorted_cps[(i + 1) % n]['center']
            signed_area += (x2 - x1) * (y2 + y1)

        if signed_area > 0:
            sorted_cps.reverse()

        return sorted_cps

    def _calculate_start_angle(self):
        """Finish line'dan ilk checkpoint'e dogru olan aciyi hesaplar."""
        if self.finish_line is None or not self.checkpoints:
            return
        sx, sy = self.finish_line['center']
        tx, ty = self.checkpoints[0]['center']
        self.start_angle = math.degrees(math.atan2(ty - sy, tx - sx))

    def check_collision(self, x , y):
        if 0 <= int(x) < self.width and 0 <= int(y) < self.height:
            return self.border_mask[int(y), int(x)] == 255 
        return True 
    
    # İki çizgi parçasının kesişip kesişmediğini bulan matematiksel vektör algoritması
    @staticmethod
    def _ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    @staticmethod
    def _segments_intersect(A, B, C, D):
        return TrackEnvironment._ccw(A, C, D) != TrackEnvironment._ccw(B, C, D) and \
               TrackEnvironment._ccw(A, B, C) != TrackEnvironment._ccw(A, B, D)

    # Artık çember (yarıçap) yok. Arabanın eski ve yeni konumu çizgiyi kesiyor mu diye bakıyoruz
    def is_checkpoint_passed(self, prev_x, prev_y, curr_x, curr_y, target_index):
        if target_index < len(self.checkpoints):
            cp = self.checkpoints[target_index]
            A = (prev_x, prev_y) # Arabanın bir önceki karedeki yeri
            B = (curr_x, curr_y) # Arabanın şu anki yeri
            C = cp['p1']         # Checkpoint çizgisinin 1. ucu
            D = cp['p2']         # Checkpoint çizgisinin 2. ucu
            return self._segments_intersect(A, B, C, D)
        return False
    
    def is_finish_line_crossed(self, prev_x, prev_y, curr_x, curr_y):
        if self.finish_line is not None:
            A = (prev_x, prev_y)
            B = (curr_x, curr_y)
            C = self.finish_line['p1']
            D = self.finish_line['p2']
            return self._segments_intersect(A, B, C, D)
        return False