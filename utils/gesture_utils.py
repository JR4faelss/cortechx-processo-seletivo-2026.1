import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import os
import urllib.request

class HandDetector:
    """[MAPEAR] Encapsula o MediaPipe para detecção de landmarks em tempo real."""
    def __init__(self, model_path='models/hand_landmarker.task'):
        self._check_model(model_path)
        # Configurações do MediaPipe: modo VIDEO para processamento em stream de baixa latência
        # e thresholds de confiança de 0.7 para mitigar falsos positivos durante o tracking
        opts = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(opts)

    def _check_model(self, path):
        """Garante que o arquivo de modelo .task exista localmente."""
        if not os.path.exists(path):
            print("Baixando modelo MediaPipe Hand Landmarker...")
            url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
            urllib.request.urlretrieve(url, path)

    def get_landmarks(self, frame_rgb):
        """Retorna os landmarks e a lateralidade da mão detectada."""
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        # O processamento no modo VIDEO exige um timestamp contínuo em milissegundos para o rastreamento
        res = self.detector.detect_for_video(mp_img, int(time.time() * 1000))
        if res.hand_landmarks:
            # Retorna as 21 coordenadas espaciais (x,y,z) da primeira mão e sua lateralidade ('Left'/'Right')
            return res.hand_landmarks[0], res.handedness[0][0].category_name
        return None, None

class ScreenMapper:
    """[TRANSLADAR] Converte coordenadas da câmera para a resolução da tela."""
    def __init__(self, suavizacao=0.45, margem=100):
        self.suavizacao = suavizacao
        self.margem = margem
        self.screen_w, self.screen_h = pyautogui.size()
        self.prev_x, self.prev_y = 0, 0

    def map_to_screen(self, lm, frame_w, frame_h):
        """Aplica interpolação linear e filtro de suavização exponencial."""
        # Utiliza interpolação para mapear o movimento. A 'margem' compensa os limites físicos da câmera,
        # garantindo que o usuário consiga mover o cursor para as bordas extremas do monitor.
        xt = np.interp(lm.x * frame_w, (self.margem, frame_w - self.margem), (0, self.screen_w))
        yt = np.interp(lm.y * frame_h, (self.margem, frame_h - self.margem), (0, self.screen_h))
        
        # Filtro de suavização (Média Móvel Exponencial): ameniza variações bruscas e tremores das mãos (jitter)
        cx = self.prev_x + (xt - self.prev_x) * self.suavizacao
        cy = self.prev_y + (yt - self.prev_y) * self.suavizacao
        self.prev_x, self.prev_y = cx, cy
        return cx, cy

class GestureController:
    """[INTERAGIR] Gerencia o estado dos cliques e ações do sistema."""
    def __init__(self):
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0
        self.drag_start = 0
        self.is_dragging = False
        self.right_clicked = False
        self.prev_scroll_y = None

    def execute(self, fingers, coords, y_raw):
        """Executa comandos com base na combinação de dedos levantados."""
        pol, ind, med, ane, min_ = fingers
        cx, cy = coords

        # Logica Principal: Apenas Indicador estendido movimenta o mouse.
        # Indicador + Polegar prepara para interações de clique.
        if ind and not any([med, ane, min_]):
            pyautogui.moveTo(cx, cy)
            if pol: # Formato de "L" com os dedos: ativador de clique
                if self.drag_start == 0: self.drag_start = time.time()
                # Threshold de 0.5s: manter o gesto pressiona o botão esquerdo para início de um arraste (Drag)
                if time.time() - self.drag_start > 0.5 and not self.is_dragging:
                    pyautogui.mouseDown()
                    self.is_dragging = True
            else:
                # Ao recolher o polegar, dispara-se um clique rápido ou encerra-se o arraste em andamento
                self._reset_drag()
            self.right_clicked = False
            self.prev_scroll_y = None

        # Gesto de Rock (Indicador + Mindinho) = Clique Direito
        elif ind and min_ and not any([pol, med, ane]):
            if not self.right_clicked:
                pyautogui.click(button='right')
                self.right_clicked = True
            self._reset_drag()
            self.prev_scroll_y = None

        # Gesto de Paz (Indicador + Médio): utilizado para navegação vertical contínua
        elif ind and med and not any([pol, ane, min_]):
            if self.prev_scroll_y is not None:
                # Calcula a diferença (delta) da posição real 'y' para inferir a velocidade e direção da rolagem
                diff = self.prev_scroll_y - y_raw
                if abs(diff) > 8:
                    pyautogui.scroll(int(diff * 2))
            self.prev_scroll_y = y_raw
            self._reset_drag()
            self.right_clicked = False
        
        else:
            self._reset_drag()
            self.right_clicked = False
            self.prev_scroll_y = None

    def _reset_drag(self):
        """Finaliza ações de clique ou arraste pendentes."""
        if self.drag_start != 0:
            if self.is_dragging:
                pyautogui.mouseUp()
                self.is_dragging = False
            else:
                pyautogui.click()
            self.drag_start = 0
