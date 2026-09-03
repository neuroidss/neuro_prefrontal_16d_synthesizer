#!/usr/bin/env python3
"""
🧠 NEUROCANVAS LIVE v138.2: PURE C++ DSP & LSL HETERARCHY CORE
- 100% СЛЕПОЙ ДВИЖОК: Движок не знает ни о каких "квестах" агента.
- Ожидает только LSL-поток. Если подключен скриптовый агент, просит у него готовый текст для HUD.
- Идеальный студийный реалтайм без фильтров и без "прыжков" BPM.
"""

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import sys
import math
import time
import threading
import numpy as np
from collections import deque
import torch
import sounddevice as sd
import queue
import pygame
import argparse

from neuro_heterarchy_core import HeterarchicalBrainEngine, NUM_MAX_DEVICES, DEVICE

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

SAMPLE_RATE = 44100
BLOCK_SIZE = 1024
TWO_PI = 2.0 * math.pi
RING_BUFFER_SIZE = 64  

WIDTH = 1600
HEIGHT = 960

ROOT_FREQS_HZ = [73.42, 110.00, 82.41, 123.47, 92.50, 69.30, 103.83, 77.78, 116.54, 87.31, 65.41, 98.00]
ROOT_NAMES    = ["D",  "A",    "E",   "B",    "F#",  "Db",  "Ab",   "Eb",  "Bb",   "F",   "C",   "G"]
JUST_RATIOS = [1.0, 16.0/15.0, 9.0/8.0, 6.0/5.0, 5.0/4.0, 4.0/3.0, 45.0/32.0, 3.0/2.0, 8.0/5.0, 5.0/3.0, 9.0/5.0, 15.0/8.0]

MODAL_PATTERNS = [
    [0, 1, 4, 5, 7, 8, 10],  # Phrygian Dom
    [0, 2, 3, 5, 7, 8, 10],  # Aeolian Minor
    [0, 2, 3, 5, 7, 9, 10],  # Dorian
    [0, 1, 3, 5, 6, 8, 10]   # Locrian
]
MODAL_NAMES = ["Phrygian Dom", "Aeolian Minor", "Dorian", "Locrian"]
OP_MODES = ["0. LIVE EEG", "1. ACTIVE AGENT", "2. PSYTRANCE DEMO", "3. AMBIENT DEMO"]

# ==============================================================================
# 1. C++ СКОМПИЛИРОВАННОЕ ЯДРО (TORCHSCRIPT)
# ==============================================================================
class TorchScriptDSP(torch.nn.Module):
    def __init__(self, sample_rate: float, block_size: int):
        super().__init__()
        self.sr = sample_rate
        self.bs = block_size
        self.two_pi = 2.0 * math.pi
        
        self.register_buffer("p_step", torch.zeros(1))
        self.register_buffer("p_kick", torch.zeros(1))
        self.register_buffer("p_bass", torch.zeros(1))
        self.register_buffer("p_acid", torch.zeros(1))
        self.register_buffer("p_sh", torch.zeros(1))
        self.register_buffer("p_pad", torch.zeros(1))
        self.register_buffer("crash_amp", torch.zeros(1))
        
        self.register_buffer("s_f3", torch.tensor([0.9, 0.4, 0.0, 0.0]))
        self.register_buffer("s_f4", torch.tensor([0.9, 0.8, 0.0, 0.0]))
        self.register_buffer("s_afz", torch.tensor([0.0, 0.0, -1.0, -1.0]))
        self.register_buffer("s_fpz", torch.tensor([0.0, 0.0, 0.0, -1.0]))
        self.register_buffer("meaning_m", torch.zeros(1))

        self.register_buffer("m2_f3", torch.tensor([0.95, 0.50, 0.0, 0.0]))
        self.register_buffer("m2_f4", torch.tensor([0.90, 0.85, 0.5, 0.0]))
        self.register_buffer("m2_afz", torch.tensor([0.0, 0.0, -1.0, -1.0]))
        self.register_buffer("m2_fpz", torch.tensor([0.5, 0.5, 0.2, -1.0]))
        
        self.register_buffer("m3_f3", torch.tensor([-1.0, -1.0, 0.0, 0.0]))
        self.register_buffer("m3_f4", torch.tensor([0.20, 0.30, 0.8, 0.9]))
        self.register_buffer("m3_afz", torch.tensor([0.33, 0.5, -1.0, -1.0]))
        self.register_buffer("m3_fpz", torch.tensor([0.8, 0.2, 0.6, -1.0]))

        self.register_buffer("t_indices", torch.arange(block_size, dtype=torch.float32))
        self.register_buffer("roots_hz", torch.tensor(ROOT_FREQS_HZ, dtype=torch.float32))
        self.register_buffer("just_ratios", torch.tensor(JUST_RATIOS, dtype=torch.float32))
        self.register_buffer("modal_patterns", torch.tensor(MODAL_PATTERNS, dtype=torch.long))
        self.register_buffer("seq_pat", torch.tensor([0, 12, 1, 12, 3, 1, 0, 7, 12, 10, 8, 7, 1, 3, 7, 0], dtype=torch.long))
        
        c_max = int(sample_rate * 2.0)
        self.register_buffer("cathedral_buf", torch.zeros((2, c_max), dtype=torch.float32))
        self.register_buffer("cathedral_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("c_max_t", torch.tensor(c_max, dtype=torch.long))

    @torch.jit.export
    def render(self, target_16d: torch.Tensor, dt: float, theta_hz: float, theta_sync: float, 
               drop_trigger: float, op_mode: int, manual_kick_mute: float, 
               out_audio: torch.Tensor, out_telem: torch.Tensor):
        
        if op_mode == 2:
            t_f3, t_f4, t_afz, t_fpz = self.m2_f3, self.m2_f4, self.m2_afz, self.m2_fpz
            theta_hz, theta_sync = 4.6667, 0.85
        elif op_mode == 3:
            t_f3, t_f4, t_afz, t_fpz = self.m3_f3, self.m3_f4, self.m3_afz, self.m3_fpz
            theta_hz, theta_sync = 4.2, 0.60
        else:
            t_f3, t_f4, t_afz, t_fpz = target_16d[0], target_16d[1], target_16d[2], target_16d[3]

        self.s_f3.add_((t_f3 - self.s_f3) * 0.08)
        self.s_f4.add_((t_f4 - self.s_f4) * 0.08)
        self.s_afz.add_((t_afz - self.s_afz) * 0.08)
        self.s_fpz.add_((t_fpz - self.s_fpz) * 0.08)

        target_m = max(0.0, min(1.0, float(theta_sync * 1.3 - 0.2)))
        self.meaning_m.add_((target_m - self.meaning_m) * 0.04)
        M = self.meaning_m[0]

        clamped_theta = max(3.6, min(8.5, float(theta_hz)))
        step_freq = clamped_theta * 2.0
        step_inc = step_freq / self.sr

        # ИДЕАЛЬНЫЙ АКУСТИЧЕСКИЙ ЧАСОВОЙ МЕХАНИЗМ
        step_phases = self.p_step[0] + self.t_indices * step_inc
        self.p_step[0] = (step_phases[-1] + step_inc) % 64.0

        step_idx = step_phases.long() % 16
        step_frac = step_phases % 1.0

        att = torch.clamp(step_frac * 30.0, min=0.0, max=1.0)
        rel = torch.clamp((1.0 - step_frac) * 20.0, min=0.0, max=1.0)
        smooth_gate = att * rel

        ry_val = self.s_f3[3]
        is_kick_step = torch.where(
            ry_val > 0.2,
            ((step_idx == 0) | (step_idx == 3) | (step_idx == 6) | (step_idx == 9) | (step_idx == 12)).float(),
            torch.where(
                ry_val < -0.2,
                ((step_idx == 0) | (step_idx == 8)).float(),
                ((step_idx % 4) == 0).float()
            )
        )
        is_bass_step = (1.0 - is_kick_step) * (step_idx % 2 != 0).float()
        is_hat_open  = ((step_idx % 4 == 2)).float()

        th_afz = (torch.atan2(self.s_afz[1], self.s_afz[0]) + self.two_pi) % self.two_pi
        root_idx = int((th_afz / self.two_pi) * 12.0) % 12
        active_root_hz = self.roots_hz[root_idx]

        rad_afz = torch.hypot(self.s_afz[0], self.s_afz[1])
        scale_idx = int(torch.clamp(rad_afz * 3.99, min=0.0, max=3.0).item())
        active_scale = self.modal_patterns[scale_idx]

        th_fpz = (torch.atan2(self.s_fpz[1], self.s_fpz[0]) + self.two_pi) % self.two_pi
        shadow_root_idx = int((th_fpz / self.two_pi) * 12.0) % 12
        shadow_root_hz = self.roots_hz[shadow_root_idx]
        shadow_scale_idx = (scale_idx + 1) % 4
        shadow_scale = self.modal_patterns[shadow_scale_idx]

        afz_wash = torch.clamp(self.s_afz[3], min=0.0, max=1.0) if op_mode not in [2, 3] else self.s_afz[3] * 0.0
        washout_gain = torch.clamp(1.0 - afz_wash, min=0.0, max=1.0)

        k_pitch = 48.0 + 180.0 * torch.exp(-step_frac * 9.0)
        k_inc = (self.two_pi / self.sr) * k_pitch * is_kick_step
        kick_phases = self.p_kick[0] + torch.cumsum(k_inc, dim=0)
        self.p_kick[0] = kick_phases[-1] % self.two_pi
        
        kick_env = torch.exp(-step_frac * 4.0) * smooth_gate * is_kick_step
        kick_gain = torch.clamp(self.s_f3[0] * 1.2, min=0.0, max=1.2) * washout_gain
        if manual_kick_mute > 0.5: kick_gain = kick_gain * 0.0
        kick = torch.tanh(torch.sin(kick_phases) * 2.5) * kick_env * kick_gain * 1.2

        b_octave = (step_idx % 4 == 3).float() + 1.0
        b_freq = active_root_hz * b_octave
        b_inc = (self.two_pi / self.sr) * b_freq * is_bass_step
        bass_phases = self.p_bass[0] + torch.cumsum(b_inc, dim=0)
        self.p_bass[0] = bass_phases[-1] % self.two_pi
        
        b_saw = 2.0 * ((bass_phases / self.two_pi) % 1.0) - 1.0
        bass_decay = torch.clamp(12.0 - self.s_f3[1] * 4.0, min=6.0, max=18.0)
        bass_env = torch.exp(-step_frac * bass_decay) * smooth_gate * is_bass_step
        bass_gain = torch.clamp(self.s_f3[0] * 1.2, min=0.0, max=1.0) * washout_gain
        bass = torch.sin(b_saw * 1.5) * bass_env * bass_gain * 1.4

        cur_step_val = int(step_idx[0].item())
        pat_idx = int(self.seq_pat[cur_step_val].item()) % 7 
        semitone = active_scale[pat_idx]
        
        f_12tet = (active_root_hz * 2.0) * (2.0 ** (semitone.float() / 12.0))
        f_just = (active_root_hz * 2.0) * self.just_ratios[int(semitone.item()) % 12] * (2.0 ** (semitone // 12).float())
        
        beating_wobble = (1.0 - M) * 0.007 * torch.sin(self.p_step[0] * 0.5)
        a_target_freq = ((1.0 - M) * f_12tet * (1.0 + beating_wobble)) + (M * f_just)

        a_inc = (self.two_pi / self.sr) * a_target_freq
        acid_phases = self.p_acid[0] + self.t_indices * a_inc
        self.p_acid[0] = (acid_phases[-1] + a_inc) % self.two_pi

        a_saw = 2.0 * ((acid_phases / self.two_pi) % 1.0) - 1.0
        fold_depth = torch.clamp(1.2 + self.s_f4[0] * 3.5, min=0.5, max=5.0)
        a_folded = torch.sin(a_saw * fold_depth)

        acid_gate = (step_idx % 3 != 0).float()
        acid_gain = torch.clamp((self.s_f4[1] + 0.5) * 0.8, min=0.1, max=1.2)
        acid_env = torch.exp(-step_frac * 6.0) * smooth_gate * acid_gate * acid_gain
        acid = a_folded * acid_env * 0.85

        shadow_readiness = torch.clamp(self.s_fpz[2] * 0.6 + self.s_fpz[3] * 0.4, min=0.0, max=1.0)
        shadow_presence = torch.clamp(M * 0.7 + shadow_readiness * 0.5, min=0.0, max=1.0)

        sh_pat_idx = int(self.seq_pat[(cur_step_val + 3) % 16].item()) % 7
        sh_semi = shadow_scale[sh_pat_idx]
        sh_target_freq = (shadow_root_hz * 2.0) * self.just_ratios[int(sh_semi.item()) % 12] * (2.0 ** (sh_semi // 12).float())

        sh_inc = (self.two_pi / self.sr) * sh_target_freq
        sh_phases = self.p_sh[0] + self.t_indices * sh_inc
        self.p_sh[0] = (sh_phases[-1] + sh_inc) % self.two_pi

        sh_saw = 2.0 * ((sh_phases / self.two_pi) % 1.0) - 1.0
        sh_env = torch.exp(-step_frac * 8.0) * smooth_gate * (step_idx % 3 == 0).float()
        sh_fold = 1.0 + torch.clamp(self.s_fpz[3], min=0.0, max=1.0) * 3.0
        acid_shadow = torch.sin(sh_saw * sh_fold) * sh_env * (0.75 * shadow_presence)

        hat_metal = torch.sin(step_frac * 18.0 * self.two_pi) * torch.sin(step_frac * 18.0 * self.two_pi * 1.414)
        hat_env = torch.exp(-step_frac * 30.0) * smooth_gate * is_hat_open * 0.45 * bass_gain
        hats = hat_metal * hat_env

        pad_inc = (self.two_pi / self.sr) * (active_root_hz * 2.0)
        pad_phases = self.p_pad[0] + self.t_indices * pad_inc
        self.p_pad[0] = (pad_phases[-1] + pad_inc) % self.two_pi
        pad = (torch.sin(pad_phases) + torch.sin(pad_phases * 1.5)) * 0.25 * torch.clamp(1.2 - self.s_f3[0], min=0.1, max=1.0)

        afz_tension = torch.clamp(self.s_afz[2], min=0.0, max=1.0) if op_mode not in [2, 3] else self.s_afz[2] * 0.0
        snare = self.t_indices * 0.0
        if afz_tension.item() > 0.15:
            snare_rate = 1.0 + afz_tension * 12.0
            snare_frac = (step_frac * snare_rate) % 1.0
            snare = hat_metal * torch.exp(-snare_frac * 14.0) * (afz_tension * 0.5)

        mono = kick + bass + acid + hats + pad + snare

        # PHASE RESET
        if drop_trigger > 0.5:
            self.crash_amp[0] = 1.0
            self.p_step[0] = 0.0

        if self.crash_amp[0].item() > 0.01:
            mono += hat_metal * self.crash_amp[0] * 0.8
            self.crash_amp[0] *= 0.9992

        left_ch = mono + acid_shadow * 0.45
        right_ch = mono - acid_shadow * 0.45 + acid_shadow * 0.85

        if M.item() > 0.15:
            cathedral_gain = (M - 0.15) * 0.40
            c_max_int = int(self.c_max_t.item())
            delay_samp = int(self.sr * 0.125)
            ptr_val = int(self.cathedral_ptr[0].item())
            
            r_ptr = (ptr_val - delay_samp) % c_max_int
            idx_v = (self.t_indices.long() + r_ptr) % c_max_int
            
            c_left = self.cathedral_buf[0, idx_v]
            c_right = self.cathedral_buf[1, idx_v]

            left_ch += c_right * cathedral_gain
            right_ch += c_left * cathedral_gain

            w_idx = (self.t_indices.long() + ptr_val) % c_max_int
            self.cathedral_buf[0, w_idx] = torch.tanh(left_ch * 0.7 + c_right * 0.35)
            self.cathedral_buf[1, w_idx] = torch.tanh(right_ch * 0.7 + c_left * 0.35)
            self.cathedral_ptr[0] = (ptr_val + self.bs) % c_max_int

        out_audio[:, 0] = torch.tanh(left_ch * 1.1) * 0.88
        out_audio[:, 1] = torch.tanh(right_ch * 1.1) * 0.88

        out_telem[0] = torch.sqrt(torch.mean(out_audio[:, 0]**2)) 
        out_telem[1] = float(root_idx)
        out_telem[2] = float(scale_idx)
        out_telem[3] = float(shadow_root_idx)
        out_telem[4] = float(shadow_scale_idx)
        out_telem[5] = float(clamped_theta * 30.0) 
        out_telem[6] = M
        out_telem[7] = shadow_presence

# ==============================================================================
# ПАМЯТЬ И ПОТОКИ 
# ==============================================================================
CONTROL_TENSOR = torch.zeros(21, device=DEVICE, dtype=torch.float32)
TELEMETRY_TENSOR = torch.zeros(8, device=DEVICE, dtype=torch.float32)

PINNED_AUDIO = torch.zeros((RING_BUFFER_SIZE, BLOCK_SIZE, 2), dtype=torch.float32, pin_memory=True)
audio_write_idx = 0
audio_read_idx = 0
buffer_lock = threading.Lock()
buffer_cond = threading.Condition(buffer_lock)

def audio_synthesis_worker(stop_event, compiled_engine, audio_feat_q):
    global audio_write_idx, audio_read_idx
    dt = BLOCK_SIZE / SAMPLE_RATE
    out_audio_gpu = torch.zeros((BLOCK_SIZE, 2), device=DEVICE, dtype=torch.float32)

    while not stop_event.is_set():
        with buffer_lock:
            in_queue = (audio_write_idx - audio_read_idx) % RING_BUFFER_SIZE
        
        if in_queue < 48:
            t_16d = CONTROL_TENSOR[0:16].view(4, 4)
            th_hz = CONTROL_TENSOR[16].item()
            th_sy = CONTROL_TENSOR[17].item()
            drop  = CONTROL_TENSOR[18].item()
            op_m  = int(CONTROL_TENSOR[19].item())
            k_mt  = CONTROL_TENSOR[20].item()
            
            if drop > 0.5: CONTROL_TENSOR[18] = 0.0

            compiled_engine.render(t_16d, dt, th_hz, th_sy, drop, op_m, k_mt, out_audio_gpu, TELEMETRY_TENSOR)
            
            PINNED_AUDIO[audio_write_idx].copy_(out_audio_gpu, non_blocking=True)
            
            with buffer_lock:
                audio_write_idx = (audio_write_idx + 1) % RING_BUFFER_SIZE
                buffer_cond.notify()
                
            if audio_feat_q.qsize() < 2:
                stereo_cpu = out_audio_gpu.cpu().numpy()
                spec = np.abs(np.fft.rfft(stereo_cpu[:, 0]))[:270]
                tel = TELEMETRY_TENSOR.cpu().numpy()
                try: audio_feat_q.put_nowait((spec, tel))
                except queue.Full: pass
        else:
            time.sleep(0.005)

def audio_playback_loop(stop_event):
    global audio_write_idx, audio_read_idx
    def cb(outdata, frames, time_info, status):
        global audio_read_idx
        with buffer_lock:
            if audio_read_idx == audio_write_idx:
                outdata.fill(0) 
            else:
                outdata[:] = PINNED_AUDIO[audio_read_idx].numpy()
                audio_read_idx = (audio_read_idx + 1) % RING_BUFFER_SIZE

    print("🔈 [Audio] Заполнение Pinned-буфера...", flush=True)
    with buffer_lock:
        while (audio_write_idx - audio_read_idx) % RING_BUFFER_SIZE < 16 and not stop_event.is_set():
            buffer_cond.wait(0.1)
            
    print("🔈 [Audio] Запуск ALSA потока.", flush=True)
    with sd.OutputStream(samplerate=SAMPLE_RATE, channels=2, callback=cb, blocksize=BLOCK_SIZE, dtype='float32'):
        while not stop_event.is_set(): time.sleep(0.1)

# ==============================================================================
# СТАТИЧНЫЙ ТОР С КОМЕТНЫМ ШЛЕЙФОМ
# ==============================================================================
def draw_static_torus_and_tail(screen, gx, gy, c, node_idx, state_16d, tails):
    lx, ly, rx, ry = state_16d[node_idx]
    pitch = 0.75
    R_main, r_tube = 55.0, 18.0

    for th in np.linspace(0, TWO_PI, 12, endpoint=False):
        pts = []
        for ph in np.linspace(0, TWO_PI, 16):
            x = (R_main + r_tube * math.cos(ph)) * math.cos(th)
            y = (R_main + r_tube * math.cos(ph)) * math.sin(th)
            z = r_tube * math.sin(ph)
            y_proj = y * math.cos(pitch) - z * math.sin(pitch)
            pts.append((int(gx + x), int(gy - 20 - y_proj)))
        pygame.draw.lines(screen, (30, 40, 50), True, pts, 1)

    th_macro = (math.atan2(ly, lx) + TWO_PI) % TWO_PI
    ph_macro = math.hypot(lx, ly) * math.pi
    
    x_m = (R_main + r_tube * math.cos(ph_macro)) * math.cos(th_macro)
    y_m = (R_main + r_tube * math.cos(ph_macro)) * math.sin(th_macro)
    z_m = r_tube * math.sin(ph_macro)
    y_proj = y_m * math.cos(pitch) - z_m * math.sin(pitch)
    
    px, py = int(gx + x_m), int(gy - 20 - y_proj)
    tails[node_idx].append((px, py))
    
    t_list = list(tails[node_idx])
    for k in range(len(t_list) - 1):
        p1 = t_list[k]
        p2 = t_list[k+1]
        prog = k / len(t_list)
        if math.hypot(p1[0]-p2[0], p1[1]-p2[1]) < 60:
            col = (int(c[0]*prog), int(c[1]*prog), int(c[2]*prog))
            pygame.draw.line(screen, col, p1, p2, max(1, int(prog * 4)))
            
    if len(t_list) > 0:
        head = t_list[-1]
        pygame.draw.circle(screen, c, head, 6)
        pygame.draw.circle(screen, (255, 255, 255), head, 2)

# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim', action='store_true', help="Запуск агента LSL")
    parser.add_argument('--headless', action='store_true', help="Режим без графики (только звук)")
    args = parser.parse_args()

    print("⚡ Компиляция DSP-ядра в TorchScript (C++)...", flush=True)
    raw_engine = TorchScriptDSP(SAMPLE_RATE, BLOCK_SIZE).to(DEVICE)
    compiled_engine = torch.jit.script(raw_engine)
    print("✅ Компиляция успешно завершена.", flush=True)

    brain_core = HeterarchicalBrainEngine()
    brain_core.start()

    agent = None
    if args.sim:
        try:
            # Агент загружается динамически, если запрошен.
            # Движок ничего не знает о его внутренней структуре.
            from synthetic_psytrance_agent import SyntheticMusicalAgent
            agent = SyntheticMusicalAgent()
            CONTROL_TENSOR[19] = 1.0 
        except ImportError:
            print("⚠️ Ошибка: файл synthetic_psytrance_agent.py не найден. Агент не запущен.")
            CONTROL_TENSOR[19] = 0.0
    else:
        CONTROL_TENSOR[19] = 0.0 

    audio_feat_q = queue.Queue(maxsize=4)
    stop_event = threading.Event()
    t_synth = threading.Thread(target=audio_synthesis_worker, args=(stop_event, compiled_engine, audio_feat_q), daemon=True)
    t_play = threading.Thread(target=audio_playback_loop, args=(stop_event,), daemon=True)
    t_synth.start()
    t_play.start()

    if args.headless:
        print("\n🥷 Запущен режим --headless. Графика отключена. Нажмите Ctrl+C для выхода.\n", flush=True)
        try:
            while True:
                frame = brain_core.get_frame()
                inputs_16d_np = np.zeros((4, 4), dtype=np.float32)
                for i in range(min(4, len(frame.nodes))):
                    k = frame.nodes[i].kinematics
                    inputs_16d_np[i] = [k.lx, k.ly, k.rx, k.ry]
                
                CONTROL_TENSOR[0:16] = torch.from_numpy(inputs_16d_np.flatten())
                
                if CONTROL_TENSOR[19] in [0.0, 1.0]:
                    CONTROL_TENSOR[16] = frame.theta_freq
                    CONTROL_TENSOR[17] = frame.theta_sync
                
                if inputs_16d_np[3, 3] > 0.8:
                    CONTROL_TENSOR[18] = 1.0

                tel = TELEMETRY_TENSOR.cpu().numpy()
                rms, r_idx, s_idx, sr_idx, ss_idx, bpm, m_val, sh_pres = tel
                
                if agent and hasattr(agent, 'get_hud_text'):
                    print(f"[AGENT] {agent.get_hud_text()}")
                
                print(f"[Core Status] BPM: {bpm:.1f} | Key: {ROOT_NAMES[int(r_idx)]} {MODAL_NAMES[int(s_idx)]} | Meaning M: {m_val*100:.1f}% | Shadow: {sh_pres*100:.0f}%", flush=True)
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
        finally:
            stop_event.set()
            if agent: agent.stop()
            brain_core.stop()
            sys.exit(0)

    import pygame
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("NeuroCanvas: Toroidal Working Memory Manifolds")
    clock = pygame.time.Clock()

    font_huge = pygame.font.SysFont("consolas", 18, bold=True)
    font_b = pygame.font.SysFont("consolas", 13, bold=True)
    font_s = pygame.font.SysFont("consolas", 11)

    gyro_centers = [(220, 230), (220, HEIGHT - 200), (WIDTH - 220, 230), (WIDTH - 220, HEIGHT - 200)]
    roles = ["F3: SYNTAX (Rhythm/Kick)", "F4: SEMANTICS (Timbre/Acid)", "AFz: TONAL TORUS (Janata 𝕋²)", "Fpz: BRANCHING (Drop)"]
    colors = [(0, 220, 255), (255, 100, 220), (255, 200, 50), (160, 80, 255)]
    labels = [
        ["lx: Kick", "ly: Bass Dec", "rx: Swing", "ry: Temporal Bias"],
        ["lx: Wavefold", "ly: Cutoff", "rx: Width", "ry: Overdrive"],
        ["lx: Key Θ", "ly: Mode Φ", "rx: Snare", "ry: Washout"],
        ["lx: Shadow X", "ly: Shadow Y", "rx: Ready", "ry: DROP"]
    ]

    tails = [deque(maxlen=30) for _ in range(4)]
    cx, cy = WIDTH // 2, HEIGHT // 2
    flash = 0.0
    spec_data = np.zeros(270, dtype=np.float32)
    drop_triggered = False

    try:
        while True:
            clock.tick(60)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT: raise KeyboardInterrupt
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE: CONTROL_TENSOR[19] = (CONTROL_TENSOR[19] + 1) % len(OP_MODES)

            frame = brain_core.get_frame()
            inputs_16d_np = np.zeros((4, 4), dtype=np.float32)
            for i in range(min(4, len(frame.nodes))):
                k = frame.nodes[i].kinematics
                inputs_16d_np[i] = [k.lx, k.ly, k.rx, k.ry]

            CONTROL_TENSOR[0:16] = torch.from_numpy(inputs_16d_np.flatten())
            
            if CONTROL_TENSOR[19] in [0.0, 1.0]:
                CONTROL_TENSOR[16] = frame.theta_freq
                CONTROL_TENSOR[17] = frame.theta_sync
            
            # ЗАЩИТА ОТ ДВОЙНОГО ДРОПА (Rising Edge Trigger)
            current_fpz_ry = inputs_16d_np[3, 3]
            if current_fpz_ry > 0.8 and not drop_triggered:
                CONTROL_TENSOR[18] = 1.0
                flash = 255.0
                drop_triggered = True
            elif current_fpz_ry <= 0.8:
                drop_triggered = False

            while not audio_feat_q.empty():
                spec_data, tel = audio_feat_q.get_nowait()
            else:
                tel = TELEMETRY_TENSOR.cpu().numpy()

            rms, r_idx, s_idx, sr_idx, ss_idx, bpm, m_val, sh_pres = tel
            state_16d = CONTROL_TENSOR[0:16].view(4, 4).cpu().numpy()

            screen.fill((7, 9, 13))

            if flash > 0:
                s_flash = pygame.Surface((WIDTH, HEIGHT))
                s_flash.set_alpha(int(flash))
                s_flash.fill((255, 240, 150))
                screen.blit(s_flash, (0, 0))
                flash = max(0.0, flash - 10.0)

            pulse_rad = int(90 + rms * 120 + m_val * 30)
            base_color = (int(255 * (1.0 - m_val)), int(180 + 75 * m_val), int(50 + 205 * m_val))
            pygame.draw.circle(screen, base_color, (cx, cy), pulse_rad, 2)

            max_s = max(0.01, float(np.max(spec_data)))
            for i in range(270):
                angle = (i / 270) * TWO_PI
                h = (float(spec_data[i]) / max_s) * (110 + m_val * 50)
                x1, y1 = int(cx + math.cos(angle) * pulse_rad), int(cy + math.sin(angle) * pulse_rad)
                x2, y2 = int(cx + math.cos(angle) * (pulse_rad + h)), int(cy + math.sin(angle) * (pulse_rad + h))
                pygame.draw.line(screen, base_color, (x1, y1), (x2, y2), 2)

            if sh_pres > 0.08:
                num_filaments = int(4 + sh_pres * 16)
                for f_i in range(num_filaments):
                    ang_f = (f_i / float(num_filaments)) * TWO_PI
                    fx1, fy1 = int(cx + math.cos(ang_f) * 70), int(cy + math.sin(ang_f) * 70)
                    fx2, fy2 = int(cx + 260 + math.cos(ang_f) * (30 + sh_pres * 40)), int(cy - 20 + math.sin(ang_f) * (30 + sh_pres * 40))
                pygame.draw.circle(screen, (160, 80, 255), (cx + 260, cy - 20), int(25 + sh_pres * 20), 2)
                screen.blit(font_s.render(f"SHADOW: {ROOT_NAMES[int(sr_idx)]} {MODAL_NAMES[int(ss_idx)]}", True, (200, 140, 255)), (cx + 195, cy + 30))

            pygame.draw.rect(screen, (13, 17, 24), (20, 12, WIDTH - 40, 125), border_radius=8)
            pygame.draw.rect(screen, (0, 255, 200), (20, 12, WIDTH - 40, 125), 1, border_radius=8)
            
            nodes_status = f"Nodes: {frame.num_live}/4"
            screen.blit(font_huge.render(f"MODE: [{OP_MODES[int(CONTROL_TENSOR[19])]}] | {nodes_status} | BPM: {bpm:.1f}", True, (0, 255, 255)), (35, 18))
            
            m_label = "12-TET QUANTIZED (ROUNDED)" if m_val < 0.4 else ("MICROTONAL CONTINUUM" if m_val < 0.75 else "PURE JUST INTONATION")
            screen.blit(font_s.render(f"MEANING EXPANSION M: {m_val*100:4.1f}% | {m_label}", True, base_color), (35, 46))
            pygame.draw.rect(screen, (30, 40, 52), (35, 62, 420, 12), border_radius=4)
            pygame.draw.rect(screen, base_color, (35, 62, int(420 * m_val), 12), border_radius=4)

            # Выводим статус агента, если он подключен и мы в режиме симуляции (1)
            if CONTROL_TENSOR[19] == 1.0:
                if agent and hasattr(agent, 'get_hud_text'):
                    agent_text = agent.get_hud_text()
                    screen.blit(font_b.render(agent_text, True, (255, 220, 50)), (35, 82))
                else:
                    screen.blit(font_b.render("AGENT QUEST: [NOT CONNECTED] Run with --sim", True, (255, 100, 100)), (35, 82))
            else:
                screen.blit(font_s.render(f"TONAL TORUS: {ROOT_NAMES[int(r_idx)]} {MODAL_NAMES[int(s_idx)]} | Plasm Tension: {sh_pres*100:.0f}%", True, (255, 220, 100)), (35, 82))

            for i in range(4):
                gx, gy = gyro_centers[i]
                c = colors[i]
                pygame.draw.rect(screen, (12, 16, 22), (gx - 175, gy - 155, 350, 330), border_radius=10)
                pygame.draw.rect(screen, c, (gx - 175, gy - 155, 350, 330), 1, border_radius=10)
                screen.blit(font_b.render(roles[i], True, c), (gx - 160, gy - 140))
                
                draw_static_torus_and_tail(screen, gx, gy, c, i, state_16d, tails)
                
                for j in range(4):
                    yy = gy + 65 + j * 24
                    val = float(np.clip((state_16d[i, j] + 1.0) * 0.5, 0.0, 1.0))
                    screen.blit(font_s.render(labels[i][j], True, (180, 180, 180)), (gx - 160, yy))
                    pygame.draw.rect(screen, (30, 40, 50), (gx - 10, yy, 140, 10), border_radius=3)
                    pygame.draw.rect(screen, c, (gx - 10, yy, int(140 * val), 10), border_radius=3)

            pygame.display.flip()

    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        if agent: agent.stop()
        brain_core.stop()
        pygame.quit()

if __name__ == '__main__':
    main()
