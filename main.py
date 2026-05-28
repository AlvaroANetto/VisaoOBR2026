#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import math
import time
import json
import numpy as np
import cv2
import serial # <-- Importação da Serial adicionada

# Controle geral
_terminate = False
_line_status = 0
_turn_dir = 0
_line_angle = 0.0

# Calibração de cores
_calibrate_color_status = 0
_calibration_color = ""

# Marcadores detectados
_green_marker = 0
_red_marker = 0

class Timer:
    def __init__(self):
        self._start_time = None
        self._elapsed = 0.0

    def start(self):
        self._start_time = time.perf_counter()

    def stop(self):
        if self._start_time is not None:
            self._elapsed = time.perf_counter() - self._start_time
            self._start_time = None
        return self._elapsed

    def elapsed(self):
        return self._elapsed

    def reset(self):
        self._start_time = None
        self._elapsed = 0.0

_time_values = []
_TIME_ARR_MAX_SIZE = 30

def add_time_value(value):
    global _time_values
    _time_values.append(value)
    if len(_time_values) > _TIME_ARR_MAX_SIZE:
        _time_values.pop(0)

def get_time_average():
    if len(_time_values) == 0:
        return 0
    return sum(_time_values) / len(_time_values)

# ===========================================================================
# CONFIG MANAGER
# ===========================================================================
class config_manager:
    CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "calibration_config.json")

    @staticmethod
    def _load():
        if os.path.exists(config_manager.CONFIG_FILE):
            try:
                with open(config_manager.CONFIG_FILE, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    @staticmethod
    def _save(data):
        with open(config_manager.CONFIG_FILE, "w") as f:
            json.dump(data, f, indent=4)

    @staticmethod
    def read_variable(key, default=None):
        data = config_manager._load()
        return data.get(key, default)

    @staticmethod
    def write_variable(key, value):
        data = config_manager._load()
        data[key] = value
        config_manager._save(data)

debug_mode = True
CAPTURE_WIDTH = 320 # Reduzido para melhorar a performance no RPi
CAPTURE_HEIGHT = 240

# Valores Padrão HSV
DEFAULT_BLACK_LOWER = np.array([0, 0, 0])
DEFAULT_BLACK_UPPER = np.array([180, 255, 60])

DEFAULT_GREEN_LOWER = np.array([35, 80, 80])
DEFAULT_GREEN_UPPER = np.array([85, 255, 255])

DEFAULT_RED_LOWER_1 = np.array([0, 80, 80])
DEFAULT_RED_UPPER_1 = np.array([10, 255, 255])
DEFAULT_RED_LOWER_2 = np.array([170, 80, 80])
DEFAULT_RED_UPPER_2 = np.array([180, 255, 255])

def load_calibration_values():
    cal = {}
    cal["black_lower"] = np.array(config_manager.read_variable("black_lower", DEFAULT_BLACK_LOWER.tolist()))
    cal["black_upper"] = np.array(config_manager.read_variable("black_upper", DEFAULT_BLACK_UPPER.tolist()))

    cal["green_lower"] = np.array(config_manager.read_variable("green_lower", DEFAULT_GREEN_LOWER.tolist()))
    cal["green_upper"] = np.array(config_manager.read_variable("green_upper", DEFAULT_GREEN_UPPER.tolist()))

    cal["red_lower_1"] = np.array(config_manager.read_variable("red_lower_1", DEFAULT_RED_LOWER_1.tolist()))
    cal["red_upper_1"] = np.array(config_manager.read_variable("red_upper_1", DEFAULT_RED_UPPER_1.tolist()))
    cal["red_lower_2"] = np.array(config_manager.read_variable("red_lower_2", DEFAULT_RED_LOWER_2.tolist()))
    cal["red_upper_2"] = np.array(config_manager.read_variable("red_upper_2", DEFAULT_RED_UPPER_2.tolist()))

    return cal

def preprocess_frame(frame):
    return cv2.GaussianBlur(frame, (5, 5), 0)

def detect_black_line(frame, cal, debug_frame=None):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, w = frame.shape[:2]

    # ROI ajustada para olhar apenas os 20% inferiores (resolve a antecipação de curva)
    roi_top = int(h * 0.8)
    roi = hsv[roi_top:, :]

    mask_black = cv2.inRange(roi, cal["black_lower"], cal["black_upper"])

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_CLOSE, kernel)
    mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    angle = 0.0
    line_detected = False
    cx, cy = w // 2, h // 2

    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        min_area = (w * h) * 0.005

        if area > min_area:
            line_detected = True

            M = cv2.moments(largest)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"]) + roi_top
            else:
                rect = cv2.boundingRect(largest)
                cx = rect[0] + rect[2] // 2
                cy = rect[1] + rect[3] // 2 + roi_top

            center_x = w // 2
            deviation = cx - center_x
            max_deviation = w // 2
            angle = (deviation / max_deviation) * 90.0

            if len(largest) >= 5:
                try:
                    [vx, vy, x0, y0] = cv2.fitLine(largest, cv2.DIST_L2, 0, 0.01, 0.01)
                    line_angle_rad = math.atan2(vy, vx)
                    fitted_angle = math.degrees(line_angle_rad)
                    angle = fitted_angle + (deviation / max_deviation) * 45.0
                except Exception:
                    pass

            # Trava o ângulo em no máximo 90 graus
            angle = max(-90.0, min(90.0, angle))
            # NORMALIZA PARA A ESCALA DO ARDUINO (-1.0 a 1.0)
            angle = angle / 90.0

            if debug_frame is not None:
                cv2.drawContours(debug_frame[roi_top:, :], [largest], -1, (0, 255, 255), 2)
                cv2.circle(debug_frame, (cx, cy), 8, (0, 0, 255), -1)
                cv2.line(debug_frame, (center_x, roi_top), (center_x, h), (255, 0, 0), 1)
                cv2.putText(debug_frame, f"Angulo Arduino: {angle:.2f}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return angle, line_detected, cx, cy

def detect_green_marker(frame, cal, debug_frame=None):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, w = frame.shape[:2]

    mask_green = cv2.inRange(hsv, cal["green_lower"], cal["green_upper"])

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected = False
    best_contour = None
    center = (0, 0)
    min_area = (w * h) * 0.003

    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area > min_area:
            detected = True
            best_contour = largest
            M = cv2.moments(largest)
            if M["m00"] > 0:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            if debug_frame is not None:
                cv2.drawContours(debug_frame, [largest], -1, (0, 255, 0), 3)
                cv2.putText(debug_frame, "VERDE", (center[0] - 30, center[1] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return detected, best_contour, center

def detect_red_marker(frame, cal, debug_frame=None):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, w = frame.shape[:2]

    mask_red_1 = cv2.inRange(hsv, cal["red_lower_1"], cal["red_upper_1"])
    mask_red_2 = cv2.inRange(hsv, cal["red_lower_2"], cal["red_upper_2"])
    mask_red = cv2.bitwise_or(mask_red_1, mask_red_2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected = False
    best_contour = None
    center = (0, 0)
    min_area = (w * h) * 0.003

    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area > min_area:
            detected = True
            best_contour = largest
            M = cv2.moments(largest)
            if M["m00"] > 0:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            if debug_frame is not None:
                cv2.drawContours(debug_frame, [largest], -1, (0, 0, 255), 3)
                cv2.putText(debug_frame, "VERMELHO", (center[0] - 40, center[1] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return detected, best_contour, center

_calibration_active = False
_calibration_color_name = ""
_calibration_samples = []
_CALIBRATION_SQUARE_SIZE = 60

def start_calibration(color_name):
    global _calibration_active, _calibration_color_name, _calibration_samples
    global _calibrate_color_status, _calibration_color
    _calibration_active = True
    _calibration_color_name = color_name
    _calibration_samples = []
    _calibrate_color_status = 1
    _calibration_color = color_name
    print(f"\n[CALIBRACAO] Iniciada para cor: {color_name}")

def _calibration_mouse_callback(event, x, y, flags, param):
    global _calibration_samples
    if event == cv2.EVENT_LBUTTONDOWN and _calibration_active:
        frame_hsv = param
        if frame_hsv is not None:
            half = _CALIBRATION_SQUARE_SIZE // 2
            h, w = frame_hsv.shape[:2]
            y1 = max(0, y - half)
            y2 = min(h, y + half)
            x1 = max(0, x - half)
            x2 = min(w, x + half)

            roi = frame_hsv[y1:y2, x1:x2]
            mean_hsv = cv2.mean(roi)[:3]
            _calibration_samples.append(mean_hsv)
            print(f"  Amostra {len(_calibration_samples)}: H={mean_hsv[0]:.0f} S={mean_hsv[1]:.0f} V={mean_hsv[2]:.0f}")

def process_calibration(frame, debug_frame=None):
    if not _calibration_active:
        return

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, w = frame.shape[:2]

    cv2.setMouseCallback("Vision PC - Debug", _calibration_mouse_callback, hsv)

    if debug_frame is not None:
        cv2.putText(debug_frame, f"CALIBRANDO: {_calibration_color_name}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(debug_frame, f"Amostras: {len(_calibration_samples)}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

def finish_calibration(save=True):
    global _calibration_active, _calibration_color_name, _calibration_samples
    global _calibrate_color_status, _calibration_color

    if not _calibration_active:
        return

    if save and len(_calibration_samples) > 0:
        samples = np.array(_calibration_samples)
        mean_h = np.mean(samples[:, 0])
        mean_s = np.mean(samples[:, 1])
        mean_v = np.mean(samples[:, 2])
        std_h = np.std(samples[:, 0])
        std_s = np.std(samples[:, 1])
        std_v = np.std(samples[:, 2])

        margin_h = max(15, std_h * 2)
        margin_s = max(30, std_s * 2)
        margin_v = max(30, std_v * 2)

        lower = [max(0, int(mean_h - margin_h)), max(0, int(mean_s - margin_s)), max(0, int(mean_v - margin_v))]
        upper = [min(180, int(mean_h + margin_h)), min(255, int(mean_s + margin_s)), min(255, int(mean_v + margin_v))]

        color = _calibration_color_name.lower()

        if color == "red":
            if lower[0] < 10 or upper[0] > 170:
                config_manager.write_variable("red_lower_1", [0, lower[1], lower[2]])
                config_manager.write_variable("red_upper_1", [10, upper[1], upper[2]])
                config_manager.write_variable("red_lower_2", [170, lower[1], lower[2]])
                config_manager.write_variable("red_upper_2", [180, upper[1], upper[2]])
            else:
                config_manager.write_variable("red_lower_1", lower)
                config_manager.write_variable("red_upper_1", upper)
        else:
            config_manager.write_variable(f"{color}_lower", lower)
            config_manager.write_variable(f"{color}_upper", upper)

        print(f"\n[CALIBRACAO] Salva para '{color}'")
    
    _calibration_active = False
    _calibration_color_name = ""
    _calibration_samples = []
    _calibrate_color_status = 0
    _calibration_color = ""

def main():
    global _terminate, _line_angle, _line_status, _turn_dir
    global _green_marker, _red_marker

    print("============================================================")
    print("  VISAO COMPUTACIONAL E CONTROLE DE TRACAO - RASPBERRY PI   ")
    print("============================================================")

    # --- Inicializa a Serial (Comunicação com Arduino) ---
    print("[INFO] Conectando ao Arduino via Serial...")
    try:
        # Tente usar /dev/ttyAMA0 ou /dev/serial0
        ser = serial.Serial('/dev/serial0', 115200, timeout=1) 
        print("[SUCESSO] Arduino conectado na porta Serial!")
        time.sleep(2) # Aguarda reset do Arduino
    except Exception as e:
        print(f"[AVISO] Falha ao conectar Serial: {e}")
        ser = None

    # --- Inicializa a câmera ---
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2) # V4L2 é geralmente o melhor backend pro RPi
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)

    cal = load_calibration_values()
    print("[INFO] Valores de calibracao de cores carregados.")

    frame_timer = Timer()
    fps_timer = Timer()
    fps_timer.start()

    if debug_mode:
        cv2.namedWindow("Vision PC - Debug", cv2.WINDOW_NORMAL)

    show_debug = debug_mode
    ultimo_angulo_valido = 0.0 # Memória do ponto cego

    try:
        while not _terminate:
            frame_timer.start()

            ret, raw_frame = cap.read()
            if not ret:
                continue

            frame = raw_frame.copy()
            debug_frame = frame.copy() if show_debug else None

            processed = preprocess_frame(frame)

            if _calibration_active:
                process_calibration(frame, debug_frame)
            else:
                # ==========================================
                # DETECÇÕES
                # ==========================================
                angle, line_found, cx, cy = detect_black_line(processed, cal, debug_frame)
                
                green_found, _, _ = detect_green_marker(processed, cal, debug_frame)
                _green_marker = 1 if green_found else 0

                red_found, _, _ = detect_red_marker(processed, cal, debug_frame)
                _red_marker = 1 if red_found else 0

                # ==========================================
                # LÓGICA DE NAVEGAÇÃO E PONTO CEGO
                # ==========================================
                if line_found:
                    ultimo_angulo_valido = angle
                    _line_angle = angle
                else:
                    # PONTO CEGO: A linha sumiu (curva de 90 graus brusca)
                    # Força giro máximo na direção que a linha estava indo
                    if ultimo_angulo_valido < 0:
                        _line_angle = -1.0 # Vira tudo pra esquerda
                    else:
                        _line_angle = 1.0  # Vira tudo pra direita
                    
                    if debug_mode and debug_frame is not None:
                        cv2.putText(debug_frame, "PONTO CEGO! GIRANDO!", (10, CAPTURE_HEIGHT - 80), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                _line_status = 1 if line_found else 0

                # Define a direção puramente para interface de log
                if abs(_line_angle) < 0.1:
                    _turn_dir = 0
                elif _line_angle < 0:
                    _turn_dir = -1
                else:
                    _turn_dir = 1

                # ==========================================
                # COMUNICAÇÃO SERIAL COM O ARDUINO
                # ==========================================
                if ser is not None:
                    # Envia o ângulo processado via Serial (ex: "0.45\n", "-1.00\n")
                    mensagem = f"{_line_angle:.2f}\n"
                    ser.write(mensagem.encode('utf-8'))

            # ==========================================
            # INTERFACE E DEBUG
            # ==========================================
            frame_time = frame_timer.stop()
            add_time_value(frame_time)
            avg_time = get_time_average()
            fps = 1.0 / avg_time if avg_time > 0 else 0

            if show_debug and debug_frame is not None:
                cv2.putText(debug_frame, f"FPS: {fps:.1f}", (10, CAPTURE_HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                status_y = 60
                statuses = [
                    (f"Linha: {'SIM' if _line_status else 'NAO'}", (0, 255, 0) if _line_status else (0, 0, 255)),
                    (f"Verde: {'SIM' if _green_marker else 'NAO'}", (0, 255, 0) if _green_marker else (128, 128, 128)),
                    (f"Vermelho: {'SIM' if _red_marker else 'NAO'}", (0, 0, 255) if _red_marker else (128, 128, 128))
                ]
                for text, color in statuses:
                    cv2.putText(debug_frame, text, (CAPTURE_WIDTH - 150, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    status_y += 20

                cv2.imshow("Vision PC - Debug", debug_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('d'):
                show_debug = not show_debug
                if not show_debug:
                    cv2.destroyAllWindows()
                else:
                    cv2.namedWindow("Vision PC - Debug", cv2.WINDOW_NORMAL)
            elif key == ord('c') and not _calibration_active:
                print("\n[CALIBRACAO] Escolha a cor: 1-Preto, 2-Verde, 3-Vermelho, 0-Cancelar")
                while True:
                    k = cv2.waitKey(0) & 0xFF
                    if k == ord('1'): start_calibration("black"); break
                    elif k == ord('2'): start_calibration("green"); break
                    elif k == ord('3'): start_calibration("red"); break
                    elif k == ord('0') or k == 27: break
            elif key == 13 and _calibration_active:  # ENTER
                finish_calibration(save=True)
                cal = load_calibration_values()
            elif key == 27 and _calibration_active:  # ESC
                finish_calibration(save=False)

    except KeyboardInterrupt:
        print("\n[INFO] Interrompido pelo usuario (Ctrl+C).")

    finally:
        if ser is not None:
            # Para os motores antes de desligar
            ser.write("0.00\n".encode('utf-8'))
            ser.close()
            print("[INFO] Conexao Serial encerrada.")
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Programa encerrado.")

if __name__ == "__main__":
    main()
