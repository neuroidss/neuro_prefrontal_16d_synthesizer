#!/usr/bin/env python3
"""
🤖 NEUROCANVAS: INDEPENDENT PSYTRANCE LSL AGENT
- 100% Инкапсуляция. Основной движок ничего не знает о внутренностях агента.
- Генерирует 4 LSL потока по 16 каналов (250 Hz).
- Строгий биологический Entrainment (4.66 Hz / 140 BPM).
"""

import time
import math
import numpy as np
import multiprocessing as mp
from pylsl import StreamInfo, StreamOutlet

# Координаты электродов
COORDS_X = np.array([10.14, 7.43, 2.75, 2.72, -2.72, -2.75, -7.42, -10.14, -10.14, -7.43, -2.75, -2.72, 2.72, 2.75, 7.43, 10.14], dtype=np.float32)
COORDS_Y = np.array([-2.72, -7.43, -4.77, -10.15, -10.14, -4.77, -7.42, -2.73, 2.72, 7.43, 4.76, 10.14, 10.15, 4.77, 7.42, 2.71], dtype=np.float32)
IS_CORE = np.hypot(COORDS_X, COORDS_Y) < 8.0
TWO_PI = 2.0 * math.pi

# Логика квестов живет ИСКЛЮЧИТЕЛЬНО в агенте
QUESTS = [
    {"name": "1. GROOVE", "desc": "Lock Torus & Syntax: 4/4 Rhythm", "f3": [0.95, 0.5, 0.0, 0.0], "f4": [0.9, 0.8, 0.5, 0.0], "afz": [0.0, 0.0, -1.0, -1.0], "fpz": [0.3, 0.5, 0.2, -1.0]},
    {"name": "2. THE VOID", "desc": "Mute Kick, Grounding ry < 0", "f3": [-1.0, -1.0, 0.0, -0.6], "f4": [0.2, 0.3, 0.8, 0.9], "afz": [0.3, 0.5, -1.0, -1.0], "fpz": [0.0, 0.0, 0.3, -1.0]},
    {"name": "3. BRANCHING", "desc": "Syncopation ry > 0.5, Fpz builds", "f3": [0.95, 0.4, 0.2, 0.7], "f4": [0.95, 0.9, 0.6, 0.0], "afz": [0.5, 1.0, 0.9, 0.9], "fpz": [0.5, 1.0, 0.85, 0.0]},
    {"name": "4. THE DROP", "desc": "Phase Reset -> Catharsis", "f3": [0.95, 0.5, 0.0, 0.0], "f4": [0.95, 0.95, 0.7, 0.0], "afz": [0.5, 1.0, -1.0, -1.0], "fpz": [0.5, 1.0, 0.2, 1.0]}
]

class MusicalAgentProcess(mp.Process):
    def __init__(self, shm):
        super().__init__()
        self.shm = shm
        self.daemon = True

    def run(self):
        outlets = []
        for i in range(4):
            info = StreamInfo(f'FreeEEG_Node{i}', 'EEG', 16, 250.0, 'float32', f'sim_music_node_{i}')
            outlets.append(StreamOutlet(info))

        print("🤖 [MUSIC AGENT] Запущен честный генератор LSL потоков (250 Hz)...")

        q_idx = 0
        timer = 0.0
        cur_cmd = np.zeros((4, 4), dtype=np.float32)
        tgt_cmd = np.zeros((4, 4), dtype=np.float32)
        
        start_time = time.time()
        chunk_size = 10
        agent_phase = 0.0 

        while self.shm['is_running'].value:
            dt = chunk_size / 250.0
            t_now = time.time() - start_time
            t_vec = np.linspace(t_now, t_now + dt, chunk_size, endpoint=False)

            timer += dt
            q = QUESTS[q_idx]

            match = min(1.0, timer / (14.0 if q_idx in (0, 1) else (7.0 if q_idx == 2 else 2.0)))

            tgt_cmd[0] = q["f3"]
            tgt_cmd[1] = q["f4"]
            tgt_cmd[2] = q["afz"]
            tgt_cmd[3] = q["fpz"]

            if q_idx == 2:
                tgt_cmd[2, 2] = match * 2.0 - 1.0
                tgt_cmd[2, 3] = match
                tgt_cmd[3, 2] = match

            dur = 14.0 if q_idx in (0, 1) else (7.0 if q_idx == 2 else 2.0)
            if timer >= dur:
                timer = 0.0
                q_idx = (q_idx + 1) % len(QUESTS)
                agent_phase = 0.0 # Phase Reset

            cur_cmd += (tgt_cmd - cur_cmd) * (dt / 0.25)
            
            # Пишем статус в разделяемую память для GUI
            self.shm['q_idx'].value = q_idx
            self.shm['timer'].value = max(0.0, dur - timer)

            live_freqs = 4.6667 + 0.01 * np.sin(TWO_PI * 0.1 * t_vec)
            phase_incs = TWO_PI * live_freqs / 250.0
            theta_phases = agent_phase + np.cumsum(phase_incs)
            agent_phase = theta_phases[-1] % TWO_PI

            gamma_phase = 2.0 * math.pi * 55.0 * t_vec
            env_nucleus = (np.clip(np.cos(theta_phases), 0, 1) ** 2)
            noise = np.random.normal(0, 0.005, (16, len(t_vec))) 

            for node_i in range(4):
                cmd_lx, cmd_ly, cmd_rx, cmd_ry = cur_cmd[node_i]

                spatial_phase = (COORDS_X * (cmd_lx * 0.35) + COORDS_Y * (cmd_ly * 0.35))[:, None]
                curl_mod = np.where(IS_CORE, -cmd_rx * 0.7, cmd_rx * 0.7)[:, None]
                pwr_mod = (1.0 + cmd_ry * 0.4)

                raw_sig = 10.0 * np.sin(theta_phases + curl_mod * 0.15) + \
                          pwr_mod * 4.5 * env_nucleus * np.sin(gamma_phase + spatial_phase + curl_mod) + noise

                outlets[node_i].push_chunk(raw_sig.T.tolist())

            time.sleep(dt)

class SyntheticMusicalAgent:
    def __init__(self):
        self.shm = {
            'is_running': mp.Value('b', True),
            'q_idx': mp.Value('i', 0),
            'timer': mp.Value('d', 0.0)
        }
        self.process = MusicalAgentProcess(self.shm)
        self.process.start()

    def get_hud_text(self):
        """Интерфейс для движка. Агент сам формирует текст о своем состоянии."""
        idx = self.shm['q_idx'].value
        timer = self.shm['timer'].value
        q = QUESTS[idx]
        return f"AGENT STATE: [{q['name']}] (Rem: {timer:.1f}s) -> {q['desc']}"

    def stop(self):
        self.shm['is_running'].value = False
        self.process.join()
