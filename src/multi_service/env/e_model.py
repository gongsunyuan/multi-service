import numpy as np
import math

# ==============================================================================
# 1. VoIP Model (G.107)
# ==============================================================================
class FullG107Calculator:
    def __init__(self):
      self.R0 = 93.2 
      self.Is = 0.0
      self.A = 0.0 
      self.Ie_base = 0.0
      self.Bpl = 4.3

    def calculate_mos(self, delay_ms, loss_pct, jitter_ms):
        Ta = delay_ms + (2 * jitter_ms) + 10.0 
        if Ta <= 100:
            Id = 0.0
        else:
            X = np.log(Ta / 100.0) / np.log(2.0)
            Id_base = 25.0 * ( (1 + X**6)**(1/6) - 3 * (1 + (X/3)**6)**(1/6) + 2 )

            if Ta > 150:
                Id_penalty = (Ta - 150) * 0.2
                Id = Id_base + Id_penalty
            else:
                Id = Id_base

        Ppl = loss_pct
        Ie_eff = self.Ie_base + (95 - self.Ie_base) * (Ppl / (Ppl + self.Bpl))
        
        R = self.R0 - self.Is - Id - Ie_eff + self.A
        
        if R < 0: return 1.0
        if R > 100: return 4.5
        return 1 + 0.035 * R + 7e-6 * R * (R - 60) * (100 - R)

# ==============================================================================
# 2. Video Model (Pure Math - High Performance)
# ==============================================================================
class RigorousVideoEvaluator:
    def __init__(self, target_bitrate_kbps=5000, max_buffer_sec=10.0):
        self.target_bitrate_kbps = target_bitrate_kbps
        self.max_buffer_sec = max_buffer_sec
        self.current_buffer_sec = 6.0 

        # P.1203 Coefficients (Calibrated for 5Mbps = Excellent)
        self.q1 = 4.66
        self.q2 = 0.70
        self.q3 = 500.0  # Calibrated inflection point
        self.q4 = 2.5

    def reset(self):
        self.current_buffer_sec = 6.0

    def _get_tcp_throughput_mathis(self, rtt_ms, loss_pct):
        mss_bits = 1460 * 8  
        rtt_sec = max(0.001, rtt_ms / 1000.0)
        p = loss_pct / 100.0
        if p <= 1e-6: return 99999999.0
        c = 1.22
        bw_bps = (mss_bits / rtt_sec) * (c / math.sqrt(p))
        return bw_bps / 1000.0 

    def _fallback_math_mos(self, bitrate, stall, duration):
        # O.22 Video Quality
        if bitrate <= 10: mos_v = 1.0
        else:
            term = self.q4 * (math.log10(self.q3) - math.log10(bitrate))
            mos_v = self.q2 + (self.q1 - self.q2) / (1 + math.exp(term))
        
        # O.23 Stalling Penalty
        if stall > 0.001:
            ratio = stall / max(duration, 1.0)
            penalty = math.exp(-15.0 * ratio)
            final_mos = 1.0 + (mos_v - 1.0) * penalty
        else:
            final_mos = mos_v
        
        return max(1.0, min(5.0, final_mos))

    def calculate_mos(self, loss_pct, rtt_ms, physical_bw_kbps, duration_sec, stateless_mode=True):
        if stateless_mode: self.current_buffer_sec = 0.5

        mathis_bw = self._get_tcp_throughput_mathis(rtt_ms, loss_pct)
        effective_bw = min(physical_bw_kbps, mathis_bw)

        downloaded_content_sec = (effective_bw * duration_sec) / self.target_bitrate_kbps
        prev_buffer = self.current_buffer_sec
        new_buffer = prev_buffer + downloaded_content_sec - duration_sec
        
        stall_duration = 0.0
        if new_buffer < 0:
            stall_duration = abs(new_buffer)
            self.current_buffer_sec = 0.0
        else:
            self.current_buffer_sec = min(new_buffer, self.max_buffer_sec)

        viewing_bitrate = min(effective_bw, self.target_bitrate_kbps)

        # 直接调用数学公式，不再尝试调用官方库
        return self._fallback_math_mos(viewing_bitrate, stall_duration, duration_sec)

# ==============================================================================
# 3. Cloud Gaming Model (G.1072)
# ==============================================================================
class RigorousCloudGamingEvaluator:
    def __init__(self, target_bitrate_kbps=8000):
        self.base_mos = 4.5
        self.target_bitrate_kbps = target_bitrate_kbps 

    def calculate_mos(self, delay_ms, loss_pct, physical_bw_kbps):
        effective_bw = min(physical_bw_kbps, self.target_bitrate_kbps)
        if effective_bw <= 100: I_coding = 3.5 
        else:
            ratio = effective_bw / self.target_bitrate_kbps
            I_coding = 1.8 * np.exp(-3.0 * ratio) 

        total_lag = delay_ms + 30.0 
        if total_lag <= 30: I_delay = 0.0
        else:
            if total_lag < 100: I_delay = (total_lag - 30) * 0.015
            else: I_delay = (100 - 30) * 0.015 + (total_lag - 100) * 0.06

        I_loss = 4.0 * (loss_pct / 100.0) * 10.0 
        mos = self.base_mos - I_coding - I_delay - I_loss
        return np.clip(mos, 1.0, 4.5)

# ==============================================================================
# 4. Legacy FPS Model (CSa)
# ==============================================================================
class RigorousLegacyFPSEvaluator:
    def __init__(self):
        self.base_mos = 4.5
        self.delay_threshold = 20.0 

    def calculate_mos(self, delay_ms, loss_pct, jitter_ms):
        if delay_ms <= self.delay_threshold: I_delay = 0.0
        elif delay_ms <= 60: I_delay = (delay_ms - self.delay_threshold) * 0.02 
        else: I_delay = 0.8 + (delay_ms - 60) * 0.08

        if jitter_ms <= 5: I_jitter = 0.0
        else: I_jitter = (jitter_ms - 5.0) * 0.15

        # Aggressive Loss Penalty
        I_loss = 80.0 * (loss_pct / 100.0) 
        
        mos = self.base_mos - I_delay - I_jitter - I_loss
        return np.clip(mos, 1.0, 4.5)
        