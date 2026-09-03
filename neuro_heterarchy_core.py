#!/usr/bin/env python3
"""
🧠 NEURO-HETERARCHY CORE v80.0 (UNIVERSAL N-DEVICE HARDWARE & DSP ENGINE)
- 100% Аппаратно- и анатомически-независимый HAL.
- Авто-обнаружение любых 16-канальных LSL потоков (без привязки к именам электродов).
- Параллельный CUDA DSP: 32 PAC Gamma слота, 120-edge directed iPLV.
- Встроенный расчет 4D-кинематики (lx, ly, rx, ry) для каждого слота на GPU.
"""

import os
import time
import math
import ctypes
import numpy as np
import multiprocessing as mp
from dataclasses import dataclass
import torch
from pylsl import StreamInlet, resolve_streams

try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

FS = 250.0
BUF_SIZE = 256
NUM_CHANNELS = 16
NUM_MAX_DEVICES = 4
NUM_FREQS = 32
NUM_PAIRS = 120

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Геометрия 26-мм сенсора FreeEEG16-alpha2 (12 внешних + 4 внутренних пина)
COORDS_X = np.array([10.14, 7.43, 2.75, 2.72, -2.72, -2.75, -7.42, -10.14, -10.14, -7.43, -2.75, -2.72, 2.72, 2.75, 7.43, 10.14], dtype=np.float32)
COORDS_Y = np.array([-2.72, -7.43, -4.77, -10.15, -10.14, -4.77, -7.42, -2.73, 2.72, 7.43, 4.76, 10.14, 10.15, 4.77, 7.42, 2.71], dtype=np.float32)
I_IDX, J_IDX = np.triu_indices(NUM_CHANNELS, k=1)

DX_PAIRS = (COORDS_X[J_IDX] - COORDS_X[I_IDX]).astype(np.float32)
DY_PAIRS = (COORDS_Y[J_IDX] - COORDS_Y[I_IDX]).astype(np.float32)

DX_GPU = torch.from_numpy(DX_PAIRS).to(DEVICE)
DY_GPU = torch.from_numpy(DY_PAIRS).to(DEVICE)

@dataclass
class Kinematics4D:
    lx: float
    ly: float
    rx: float
    ry: float

@dataclass
class NodeState:
    device_id: int
    name: str
    source_id: str
    is_connected: bool
    phase_theta: float
    kinematics: Kinematics4D
    iplv_32: np.ndarray # [32, 120]

@dataclass
class UniversalFrame:
    nodes: list[NodeState]
    theta_freq: float
    theta_sync: float
    theta_phase: float
    is_real: bool
    num_live: int

class GPU_Daemon_Process(mp.Process):
    def __init__(self, shared_mem):
        super().__init__()
        self.daemon = True
        self.shm = shared_mem

    def run(self):
        freqs = torch.fft.fftfreq(BUF_SIZE, d=1.0/FS).to(DEVICE)
        notch = torch.ones_like(freqs)
        notch[(torch.abs(freqs) >= 48.0) & (torch.abs(freqs) <= 52.0)] = 0.0
        notch[(torch.abs(freqs) >= 98.0) & (torch.abs(freqs) <= 102.0)] = 0.0
        notch = notch.view(1, 1, BUF_SIZE)

        f_theta = (torch.exp(-0.5 * ((freqs - 6.0) / 1.5)**2) * 2.0).view(1, 1, BUF_SIZE)
        f_theta[:, :, freqs < 0] = 0.0

        gamma_centers = torch.linspace(30.0, 85.0, NUM_FREQS, device=DEVICE).view(1, NUM_FREQS, 1, 1)
        freqs_4d = freqs.view(1, 1, 1, BUF_SIZE)
        gamma_filters = torch.exp(-0.5 * ((freqs_4d - gamma_centers) / 4.5)**2) * 2.0
        gamma_filters[:, :, :, freqs < 0] = 0.0

        slot_angles = (-math.pi + (2.0 * math.pi / NUM_FREQS) * (torch.arange(NUM_FREQS, device=DEVICE) + 0.5)).view(1, NUM_FREQS, 1, 1)

        I_GPU = torch.from_numpy(I_IDX).to(DEVICE, dtype=torch.long)
        J_GPU = torch.from_numpy(J_IDX).to(DEVICE, dtype=torch.long)

        inlets = [None] * NUM_MAX_DEVICES
        stream_names = ["Empty"] * NUM_MAX_DEVICES
        stream_uids_list = [""] * NUM_MAX_DEVICES
        connected_uids = set()

        raw_buffers = np.zeros((NUM_MAX_DEVICES, NUM_CHANNELS, BUF_SIZE), dtype=np.float32)
        raw_buf_gpu = torch.zeros((NUM_MAX_DEVICES, NUM_CHANNELS, BUF_SIZE), device=DEVICE, dtype=torch.float32)

        sh_dev_phase = np.frombuffer(self.shm['dev_phase'].get_obj(), dtype=np.float64)
        sh_kinematics = np.frombuffer(self.shm['kinematics'].get_obj(), dtype=np.float64).reshape(NUM_MAX_DEVICES, 4)
        sh_iplv = np.frombuffer(self.shm['iplv'].get_obj(), dtype=np.float64).reshape(NUM_MAX_DEVICES, NUM_FREQS, NUM_PAIRS)

        last_resolve_time = 0.0

        while self.shm['is_running'].value:
            now = time.time()
            
            # Агностическое авто-обнаружение любых 16-канальных EEG потоков
            if (None in inlets) and (now - last_resolve_time > 1.5):
                last_resolve_time = now
                try:
                    streams = resolve_streams(wait_time=0.2)
                    # СОРТИРУЕМ ПОТОКИ ПО ИМЕНИ, ЧТОБЫ Node0 -> Slot0, Node1 -> Slot1
                    streams = sorted(streams, key=lambda s: s.name())
                    for s in streams:
                        s_uid = s.uid()
                        if s_uid not in connected_uids and s.channel_count() == NUM_CHANNELS:
                            for slot_i in range(NUM_MAX_DEVICES):
                                if inlets[slot_i] is None:
                                    try:
                                        inlets[slot_i] = StreamInlet(s, max_buflen=1, max_chunklen=BUF_SIZE, recover=True)
                                        connected_uids.add(s_uid)
                                        stream_names[slot_i] = s.name()
                                        stream_uids_list[slot_i] = s.source_id()
                                        print(f"✅ [CORE HAL] Привязан Slot [{slot_i}] <- '{s.name()}'")
                                        break
                                    except Exception:
                                        pass
                except Exception:
                    pass

            num_live = sum(1 for inl in inlets if inl is not None)
            is_real = (num_live > 0)
            self.shm['is_real'].value = is_real
            self.shm['num_live'].value = num_live

            if is_real:
                for i in range(NUM_MAX_DEVICES):
                    if inlets[i] is not None:
                        try:
                            chunk, _ = inlets[i].pull_chunk(timeout=0.0, max_samples=BUF_SIZE)
                            if chunk:
                                arr = np.array(chunk, dtype=np.float32).T
                                n = arr.shape[1]
                                if n >= BUF_SIZE: raw_buffers[i] = arr[:NUM_CHANNELS, -BUF_SIZE:]
                                else:
                                    raw_buffers[i] = np.roll(raw_buffers[i], -n, axis=1)
                                    raw_buffers[i][:, -n:] = arr[:NUM_CHANNELS, :]
                        except Exception:
                            # Очистка при отключении
                            if stream_uids_list[i] in connected_uids:
                                connected_uids.remove(stream_uids_list[i])
                            inlets[i] = None
                            stream_names[i] = "Empty"
                            stream_uids_list[i] = ""
                raw_buf_gpu.copy_(torch.from_numpy(raw_buffers))
            else:
                time.sleep(0.001)
                continue

            with torch.inference_mode():
                centered = raw_buf_gpu - torch.mean(raw_buf_gpu, dim=2, keepdim=True)
                fft_clean = torch.fft.fft(centered, dim=-1) * notch

                # Ведущий Тета-ритм (по первому активному датчику)
                first_active = 0
                for idx in range(NUM_MAX_DEVICES):
                    if inlets[idx] is not None:
                        first_active = idx
                        break

                Z_theta = torch.fft.ifft(fft_clean * f_theta, dim=-1)
                P_theta = Z_theta / (torch.abs(Z_theta) + 1e-12)
                mean_th_phasors = torch.mean(P_theta, dim=1)
                phi_theta_all = torch.angle(mean_th_phasors)

                self.shm['theta_phase'].value = float(phi_theta_all[first_active, -1].item())
                self.shm['theta_sync'].value = float(torch.mean(torch.abs(mean_th_phasors[first_active])).item())

                th_vec = phi_theta_all[first_active]
                d_phi = (th_vec[1:] - th_vec[:-1] + math.pi) % (2.0 * math.pi) - math.pi
                self.shm['theta_freq'].value = float(np.clip((torch.mean(d_phi) / (2.0 * math.pi) * FS).item(), 3.5, 9.0))

                dev_phases = phi_theta_all[:, -1].cpu().numpy()

                # 32 Гамма-слота PAC по всем подключенным девайсам
                fft_exp = fft_clean.unsqueeze(1)
                Z_gamma = torch.fft.ifft(fft_exp * gamma_filters, dim=-1)
                P_gamma = Z_gamma / (torch.abs(Z_gamma) + 1e-12)

                p_diff = phi_theta_all[first_active:first_active+1].view(1, 1, 1, BUF_SIZE) - slot_angles
                w = torch.exp(3.2 * torch.cos(p_diff))
                w = w / (torch.sum(w, dim=-1, keepdim=True) + 1e-6)

                cg_gamma = P_gamma[:, :, I_GPU, :] * torch.conj(P_gamma[:, :, J_GPU, :])
                psi_field = torch.sum(cg_gamma * w, dim=-1)

                past_anchor = psi_field[:, 0:1, :]
                gamma_120 = torch.imag(psi_field * torch.conj(past_anchor)) # [4, 32, 120]

                # Векторизованный расчет 4D кинематики (lx, ly, rx, ry) для всех 4 слотов на GPU
                sum_pwr = torch.sum(torch.abs(gamma_120), dim=-1, keepdim=True) + 1e-6
                raw_x = -torch.sum(gamma_120 * DX_GPU.view(1, 1, NUM_PAIRS), dim=-1) / sum_pwr.squeeze(-1)
                raw_y = -torch.sum(gamma_120 * DY_GPU.view(1, 1, NUM_PAIRS), dim=-1) / sum_pwr.squeeze(-1)

                traj_x = torch.clamp(raw_x / 6.0, -1.0, 1.0)
                traj_y = torch.clamp(raw_y / 6.0, -1.0, 1.0)

                lx = traj_x[:, 31] - traj_x[:, 0]
                ly = traj_y[:, 31] - traj_y[:, 0]

                L_len = torch.hypot(lx, ly) + 1e-5
                mid_x = torch.mean(traj_x[:, 11:22], dim=-1)
                mid_y = torch.mean(traj_y[:, 11:22], dim=-1)
                chord_mid_x = (traj_x[:, 0] + traj_x[:, 31]) * 0.5
                chord_mid_y = (traj_y[:, 0] + traj_y[:, 31]) * 0.5

                rx = ((mid_x - chord_mid_x) * (-ly) + (mid_y - chord_mid_y) * lx) / L_len
                rx = torch.clamp(rx * 2.5, -1.0, 1.0)

                pwr_past = torch.sum(torch.abs(gamma_120[:, :11, :]), dim=(1, 2))
                pwr_fut  = torch.sum(torch.abs(gamma_120[:, 21:, :]), dim=(1, 2))
                ry = torch.clamp((pwr_fut - pwr_past) / (pwr_fut + pwr_past + 1e-6) * 2.0, -1.0, 1.0)

                kinematics_gpu = torch.stack([lx, ly, rx, ry], dim=-1) # [4, 4]

                np.copyto(sh_dev_phase, dev_phases)
                np.copyto(sh_kinematics, kinematics_gpu.cpu().numpy())
                np.copyto(sh_iplv, gamma_120.cpu().numpy())

class HeterarchicalBrainEngine:
    def __init__(self):
        self.shm = {
            'is_running': mp.Value(ctypes.c_bool, True),
            'is_real': mp.Value(ctypes.c_bool, False),
            'num_live': mp.Value('i', 0),
            'theta_sync': mp.Value('d', 0.0),
            'theta_freq': mp.Value('d', 6.0),
            'theta_phase': mp.Value('d', 0.0),
            'dev_phase': mp.Array('d', NUM_MAX_DEVICES),
            'kinematics': mp.Array('d', NUM_MAX_DEVICES * 4),
            'iplv': mp.Array('d', NUM_MAX_DEVICES * NUM_FREQS * NUM_PAIRS)
        }
        self._dev_phase = np.frombuffer(self.shm['dev_phase'].get_obj(), dtype=np.float64)
        self._kinematics = np.frombuffer(self.shm['kinematics'].get_obj(), dtype=np.float64).reshape(NUM_MAX_DEVICES, 4)
        self._iplv = np.frombuffer(self.shm['iplv'].get_obj(), dtype=np.float64).reshape(NUM_MAX_DEVICES, NUM_FREQS, NUM_PAIRS)
        
        self.process = GPU_Daemon_Process(self.shm)

    def start(self): 
        self.process.start()

    def stop(self):
        self.shm['is_running'].value = False
        self.process.join(timeout=1.0)

    def get_frame(self) -> UniversalFrame:
        nodes = []
        for i in range(NUM_MAX_DEVICES):
            k = self._kinematics[i]
            nodes.append(NodeState(
                device_id=i,
                name=f"Device_{i}",
                source_id=f"Slot_{i}",
                is_connected=bool(self._dev_phase[i] != 0.0),
                phase_theta=float(self._dev_phase[i]),
                kinematics=Kinematics4D(lx=float(k[0]), ly=float(k[1]), rx=float(k[2]), ry=float(k[3])),
                iplv_32=self._iplv[i].copy()
            ))
        return UniversalFrame(
            nodes=nodes,
            theta_freq=self.shm['theta_freq'].value,
            theta_sync=self.shm['theta_sync'].value,
            theta_phase=self.shm['theta_phase'].value,
            is_real=self.shm['is_real'].value,
            num_live=self.shm['num_live'].value
        )
