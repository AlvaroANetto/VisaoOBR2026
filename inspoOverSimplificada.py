#!/usr/bin/env python3
"""
Visão OBR 2026 - Rescue Line
RPi5 + OpenCV + Serial → Arduino Mega
Foco: Estabilidade em curva, recuperação de linha e comunicação robusta
"""

import cv2
import numpy as np
import serial
import time
import math
from collections import deque
from datetime import datetime
import json, os




def load_calibration():
    cfg_path = os.path.join(os.path.dirname(__file__), "calibration_black.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            data = json.load(f)
        return np.array(data["HSV_BLACK_MIN"]), np.array(data["HSV_BLACK_MAX"])
    print("⚠ calibration_black.json não encontrado. Usando valores padrão.")
    return np.array([0, 0, 0]), np.array([180, 255, 80])


# ==================== CONFIGURAÇÃO ====================
class Config:
    # Câmera
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_FPS = 30
    ROI_Y_START = 200  # Ignora parte inferior (chão próximo)
    ROI_Y_END = 400    # Foca no horizonte útil
    
    HSV_BLACK_MIN, HSV_BLACK_MAX = load_calibration()
    
    # Filtros
    ANGLE_FILTER_ALPHA = 0.75  # Filtro exponencial: 0.9=muito suave, 0.5=responsivo
    ANGLE_HISTORY_SIZE = 5     # Média móvel adicional
    
    # Controle
    DEADBAND = 0.08            # Ignora desvios < 8% (evita micro-oscilações)
    MAX_ANGLE_SEND = 1.0       # Normalizado [-1, 1]
    
    # Blind spot
    MAX_BLIND_FRAMES = 8       # ~260ms a 30 FPS antes de entrar em modo busca
    BLIND_RECOVERY_ANGLE = 0.35 # Ângulo suave de busca quando perde linha
    
    # Serial
    SERIAL_PORT = '/dev/ttyAMA0'
    SERIAL_BAUD = 115200
    SERIAL_TIMEOUT = 0.01
    
    # Debug
    DEBUG_FPS = True
    DEBUG_VISUALIZE = True  # Mostra janela OpenCV (desative em produção)

# ==================== FILTROS ====================
class ExponentialFilter:
    """Filtro passa-baixa exponencial para suavizar ângulo"""
    def __init__(self, alpha=0.75, init_value=0.0):
        self.alpha = alpha
        self.value = init_value
        self.initialized = False
    
    def update(self, new_value):
        if not self.initialized:
            self.value = new_value
            self.initialized = True
        else:
            self.value = self.alpha * self.value + (1 - self.alpha) * new_value
        return self.value

class MovingAverage:
    """Média móvel com deque para histórico fixo"""
    def __init__(self, window_size=5):
        self.window = deque(maxlen=window_size)
    
    def update(self, value):
        self.window.append(value)
        return sum(self.window) / len(self.window)
    
    def is_ready(self):
        return len(self.window) == self.window.maxlen

# ==================== VISÃO ====================
class LineDetector:
    def __init__(self, config):
        self.cfg = config
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        
    def preprocess(self, frame):
        """Prepara frame: ROI + CLAHE + conversão HSV"""
        # Crop ROI vertical
        roi = frame[self.cfg.ROI_Y_START:self.cfg.ROI_Y_END, :]
        
        # Melhora contraste em iluminação variável
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = self.clahe.apply(l)
        lab_eq = cv2.merge([l_eq, a, b])
        roi_proc = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
        
        # Converte para HSV e aplica threshold
        hsv = cv2.cvtColor(roi_proc, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.cfg.HSV_BLACK_MIN, self.cfg.HSV_BLACK_MAX)
        
        # Morfologia para limpar ruído
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask, roi.shape[:2]
    
    def detect_angle(self, mask, shape):
        """
        Calcula ângulo normalizado [-1, 1] usando momentos de contorno.
        Retorna: (angle_normalized, line_found, debug_info)
        """
        h, w = shape
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0, False, None
        
        # Pega maior contorno (presume-se que seja a linha)
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        
        if area < 50:  # Threshold mínimo de área
            return 0.0, False, None
        
        # Método 1: Momentos (mais estável para curvas)
        M = cv2.moments(largest)
        if M["m00"] < 1:
            return 0.0, False, None
            
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # Normaliza desvio horizontal: -1 (esq) a +1 (dir)
        deviation_norm = (cx - w//2) / (w//2)
        
        # Opcional: usar inclinação se tiver pontos suficientes
        if len(largest) >= 5:
            [vx, vy, _, _] = cv2.fitLine(largest, cv2.DIST_L2, 0, 0.01, 0.01)
            angle_rad = math.atan2(vy, vx)
            angle_deg = math.degrees(angle_rad)
            # Converte para normalizado [-1, 1] considerando -90° a +90°
            angle_norm = np.clip(angle_deg / 90.0, -1.0, 1.0)
            # Combina levemente com desvio para melhor resposta
            final_angle = 0.7 * angle_norm + 0.3 * deviation_norm
        else:
            final_angle = deviation_norm
        
        debug = {'cx': cx, 'cy': cy, 'area': area, 'contour': largest}
        return np.clip(final_angle, -1.0, 1.0), True, debug

# ==================== COMUNICAÇÃO ====================
class SerialProtocol:
    """Protocolo simples com header + checksum"""
    HEADER = 'A'  # Identifica pacote de ângulo
    
    @staticmethod
    def encode(angle):
        payload = f"{SerialProtocol.HEADER}{angle:.2f}"
        checksum = sum(ord(c) for c in payload) & 0xFF
        return f"{payload}*{checksum:02X}\n".encode()
    
    @staticmethod
    def validate(line_bytes):
        try:
            line = line_bytes.decode('ascii').strip()
            if '*' not in line:
                return False, None
            payload_part, checksum_part = line.split('*')
            received_checksum = int(checksum_part, 16)
            calc_checksum = sum(ord(c) for c in payload_part) & 0xFF
            if received_checksum != calc_checksum:
                return False, None
            if payload_part[0] != SerialProtocol.HEADER:
                return False, None
            angle = float(payload_part[1:])
            return True, np.clip(angle, -1.0, 1.0)
        except:
            return False, None

# ==================== CONTROLE PRINCIPAL ====================
class RescueBot:
    def __init__(self):
        self.cfg = Config()
        self.detector = LineDetector(self.cfg)
        self.angle_filter = ExponentialFilter(self.cfg.ANGLE_FILTER_ALPHA)
        self.angle_history = MovingAverage(self.cfg.ANGLE_HISTORY_SIZE)
        
        # Estado blind spot
        self.blind_counter = 0
        self.last_valid_angle = 0.0
        
        # Serial
        self.ser = None
        self._connect_serial()
        
        # Timing
        self.frame_count = 0
        self.start_time = time.perf_counter()
        
    def _connect_serial(self):
        for attempt in range(3):
            try:
                self.ser = serial.Serial(
                    self.cfg.SERIAL_PORT, 
                    self.cfg.SERIAL_BAUD,
                    timeout=self.cfg.SERIAL_TIMEOUT,
                    write_timeout=self.cfg.SERIAL_TIMEOUT
                )
                time.sleep(0.1)  # Aguarda handshake
                print(f"✓ Serial conectado em {self.cfg.SERIAL_PORT}")
                return
            except Exception as e:
                print(f"⚠ Tentativa {attempt+1} falhou: {e}")
                time.sleep(0.5)
        print("✗ Falha ao conectar serial. Continuando sem comunicação.")
    
    def send_angle(self, angle):
        """Envia ângulo com protocolo validado"""
        if not self.ser or not self.ser.is_open:
            self._connect_serial()
            return
        try:
            packet = SerialProtocol.encode(angle)
            self.ser.write(packet)
        except Exception as e:
            print(f"✗ Erro ao enviar: {e}")
    
    def handle_blind_spot(self, line_found):
        if line_found:
            self.blind_counter = 0
            return False, self.last_valid_angle
        
        self.blind_counter += 1
        
        if self.blind_counter < self.cfg.MAX_BLIND_FRAMES:
            # Mantém último ângulo válido com decaimento suave
            recovered_angle = self.last_valid_angle * 0.92
            return False, np.clip(recovered_angle, -1, 1)
        else:
            # Modo busca: gira suave na direção do último movimento
            search_angle = self.cfg.BLIND_RECOVERY_ANGLE * (1 if self.last_valid_angle >= 0 else -1)
            return True, search_angle
    
    def process_frame(self, frame):
       
        # 1. Detecção
        mask, shape = self.detector.preprocess(frame)
        angle_raw, line_found, debug = self.detector.detect_angle(mask, shape)
        
        # 2. Blind spot handling
        is_blind, angle_decision = self.handle_blind_spot(line_found)
        
        if line_found:
            # Atualiza histórico apenas quando vê linha
            self.last_valid_angle = angle_raw
            angle_filtered = self.angle_filter.update(angle_raw)
            angle_final = self.angle_history.update(angle_filtered)
        else:
            angle_final = angle_decision
        
        # 3. Deadband: ignora pequenos desvios
        if abs(angle_final) < self.cfg.DEADBAND:
            angle_final = 0.0
        
        # 4. Envio serial
        self.send_angle(angle_final)
        
        # 5. Debug visual (opcional)
        if self.cfg.DEBUG_VISUALIZE and debug and line_found:
            display = frame.copy()
            roi_h = self.cfg.ROI_Y_END - self.cfg.ROI_Y_START
            mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mask_vis = cv2.resize(mask_vis, (display.shape[1], roi_h))
            display[self.cfg.ROI_Y_START:self.cfg.ROI_Y_END, :] = \
                cv2.addWeighted(display[self.cfg.ROI_Y_START:self.cfg.ROI_Y_END, :], 0.7, 
                               mask_vis, 0.3, 0)
            
            # Desenha centro e contorno
            cv2.circle(display, (debug['cx'], debug['cy'] + self.cfg.ROI_Y_START), 
                      8, (0, 255, 0), -1)
            cv2.drawContours(display, [debug['contour']], -1, (255, 0, 0), 2)
            
            # Info overlay
            status = "BUSCA" if is_blind else "OK"
            color = (0, 0, 255) if is_blind else (0, 255, 0)
            cv2.putText(display, f"{status} | Ang: {angle_final:.2f} | Blind: {self.blind_counter}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.imshow('RescueBot', display)
            cv2.waitKey(1)
        
        return angle_final, line_found, is_blind
    
    def run(self):
        """Loop principal com timestep fixo"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, self.cfg.CAMERA_FPS)
        
        if not cap.isOpened():
            print("✗ Falha ao abrir câmera")
            return
        
        target_frame_time = 1.0 / self.cfg.CAMERA_FPS
        print(f"▶ Iniciando loop a {self.cfg.CAMERA_FPS} FPS...")
        
        try:
            while True:
                loop_start = time.perf_counter()
                
                # Leitura não-bloqueante
                ret, frame = cap.read()
                if not ret:
                    print("⚠ Frame não capturado")
                    continue
                
                # Processamento
                angle, found, blind = self.process_frame(frame)
                
                # Timing fixo
                elapsed = time.perf_counter() - loop_start
                wait_time = max(0, target_frame_time - elapsed)
                time.sleep(wait_time)
                
                # Stats
                self.frame_count += 1
                if self.cfg.DEBUG_FPS and self.frame_count % 30 == 0:
                    elapsed_total = time.perf_counter() - self.start_time
                    fps = self.frame_count / elapsed_total
                    print(f"📊 FPS médio: {fps:.1f} | Ângulo: {angle:.2f} | Blind: {blind}")
                    
        except KeyboardInterrupt:
            print("\n⏹ Parando...")
        finally:
            cap.release()
            if self.ser and self.ser.is_open:
                self.ser.close()
            cv2.destroyAllWindows()

# ==================== ENTRY POINT ====================
if __name__ == "__main__":
    bot = RescueBot()
    bot.run()
