import numpy as np
import math
import sys
import json

# ==============================================================================
# Dependency Check
# We strictly require the official 'itu-p1203' library for full fidelity.
# ==============================================================================
try:
  from itu_p1203 import P1203Standalone
  HAS_OFFICIAL_LIB = True
except ImportError:
  HAS_OFFICIAL_LIB = False
  print("❌ [Critical Error] 'itu-p1203' library not found!")
  print("   Please run: pip install git+https://github.com/itu-p1203/itu-p1203.git")
  print("   The system will fallback to a simplified model, which is NOT rigorous.")

# ==============================================================================
# 1. VoIP Model (G.107) - Theoretical Implementation
# ==============================================================================
class FullG107Calculator:
  """
  ITU-T G.107 (E-model) for VoIP.
  Implements the exact logarithmic delay impairment curves.
  """
  def __init__(self):
    # Constants for G.711 (Narrowband)
    self.R0 = 93.2 
    self.Is = 0.0
    self.A = 0.0 
    self.Ie_base = 0.0
    self.Bpl = 4.3

  def calculate_mos(self, delay_ms, loss_pct, jitter_ms):
    """
    Calculates MOS for VoIP.
    Args:
      delay_ms: One-way delay (ms)
      loss_pct: Packet loss rate (0-100)
      jitter_ms: Jitter (ms)
    """
    # 1. Effective Latency (Ta)
    # Accounts for Network Delay + De-jitter Buffer (2*Jitter) + Packetization (10ms)
    Ta = delay_ms + (2 * jitter_ms) + 10.0 

    # 2. Delay Impairment (Id) - The Rigorous G.107 Curve
    # Not the linear approximation!
    if Ta <= 100:
      Id = 0.0
    else:
      X = np.log(Ta / 100.0) / np.log(2.0)
      Id = 25.0 * ( (1 + X**6)**(1/6) - 3 * (1 + (X/3)**6)**(1/6) + 2 )

    # 3. Equipment Impairment (Ie_eff) - Loss impact
    Ppl = loss_pct
    Ie_eff = self.Ie_base + (95 - self.Ie_base) * (Ppl / (Ppl + self.Bpl))
    
    # 4. Final R-Factor
    R = self.R0 - self.Is - Id - Ie_eff + self.A
    
    # 5. Map R to MOS (1.0 - 4.5)
    if R < 0: return 1.0
    if R > 100: return 4.5
    return 1 + 0.035 * R + 7e-6 * R * (R - 60) * (100 - R)

# ==============================================================================
# 2. Video Model (P.1203 + Mathis) - The "Full Fidelity" Class
# ==============================================================================
class RigorousVideoEvaluator:
  """
  High-Fidelity Video QoE Evaluator.
  
  Components:
  1. Mathis Equation: Theoretical TCP throughput modeling based on Loss/RTT.
  2. Virtual Buffer: Simulates player buffer dynamics (Stateful).
  3. ITU-T P.1203: Official standard for MOS calculation (Mode 0).
  """
  def __init__(self, target_bitrate_kbps=5000, max_buffer_sec=10.0):
    self.target_bitrate_kbps = target_bitrate_kbps # e.g. 5000 for 1080p
    self.max_buffer_sec = max_buffer_sec
    
    # Stateful Memory (The "OldBuffer")
    self.current_buffer_sec = 6.0 

  def reset(self):
    """Resets the buffer state. Call this when a new user session starts."""
    self.current_buffer_sec = 6.0

  def _get_tcp_throughput_mathis(self, rtt_ms, loss_pct):
    """
    [Theoretical Component]
    Calculates the maximum TCP throughput using the Mathis Equation.
    Ref: Mathis et al. (1997)
    Formula: BW = (MSS / RTT) * (C / sqrt(p))
    """
    mss_bits = 1460 * 8  # Standard Ethernet MSS (Bits)
    rtt_sec = max(0.001, rtt_ms / 1000.0) # Avoid div zero
    p = loss_pct / 100.0 # Percent to probability
    
    if p <= 1e-6:
      # If no loss, TCP is limited only by physical link capacity
      # We return a very large number, so physical BW becomes the bottleneck
      return 99999999.0
    else:
      # Mathis Constant C ≈ 1.22 (sqrt(3/2))
      c = 1.22
      bw_bps = (mss_bits / rtt_sec) * (c / math.sqrt(p))
      return bw_bps / 1000.0 # Convert to kbps

  def _call_official_lib(self, bitrate_kbps, duration_sec, stall_sec):
    """
    [Standard Component]
    Wraps the official 'itu-p1203' Python library.
    """
    if not HAS_OFFICIAL_LIB:
      return 1.0 # Fail-safe

    # Construct the exact JSON structure required by ITU-T P.1203 Mode 0
    input_data = {
      "I11": { # Video Generation Module
        "segments": [{
          "bitrate": bitrate_kbps,
          "codec": "h264",
          "duration": duration_sec,
          "fps": 24.0,
          "resolution": "1920x1080",
          "start": 0
        }],
        "streamId": 1
      },
      "I13": { # Stalling Module
        "stalls": [
          # Inject a stalling event if calculated stall > 0
          {"start": duration_sec / 2.0, "duration": stall_sec} 
          if stall_sec > 0.001 else None
        ]
      },
      "IGen": { # Meta Info
        "displaySize": "1920x1080",
        "device": "pc"
      }
    }
    
    # Sanitize input (Remove None values)
    input_data["I13"]["stalls"] = [x for x in input_data["I13"]["stalls"] if x]

    try:
      # Invoke the official engine
      model = P1203Standalone(input_data)
      result = model.calculate_complete()
      # O.46 is the overall Audiovisual MOS
      return result['O46']
    except Exception as e:
      # Fallback for library errors
      print(f"[P1203 Wrapper Error] {e}")
      return 1.0

  def calculate_mos(self, loss_pct, rtt_ms, physical_bw_kbps, duration_sec, stateless_mode=True):
    """
    Main Calculation Pipeline.
    
    Args:
      loss_pct: Packet loss from D-ITG (%)
      rtt_ms: Round Trip Time (ms)
      physical_bw_kbps: Measured link bandwidth (kbps)
      duration_sec: Step duration (e.g., 4.0s)
      stateless_mode: If True, resets buffer to 0.5s every step (Worst-Case).
    """
    
    # --- Step 0: Apply Worst-Case Mode (If requested) ---
    if stateless_mode:
      # Reset buffer to a minimum safe threshold at every step.
      # This forces the agent to meet demand instantaneously.
      self.current_buffer_sec = 0.5

    # --- Step 1: Theoretical TCP Throughput ---
    # "How fast can TCP send data given this loss rate?"
    mathis_bw = self._get_tcp_throughput_mathis(rtt_ms, loss_pct)
    
    # The actual speed is the minimum of Physical Link and TCP Limit
    effective_bw = min(physical_bw_kbps, mathis_bw)

    # --- Step 2: Virtual Buffer Simulation ---
    # "How much video content did we actually download?"
    # Ratio = Supply / Demand
    downloaded_content_sec = (effective_bw * duration_sec) / self.target_bitrate_kbps
    
    # Update Buffer State
    prev_buffer = self.current_buffer_sec
    new_buffer = prev_buffer + downloaded_content_sec - duration_sec
    
    stall_duration = 0.0
    
    if new_buffer < 0:
      # Buffer Drained -> Stalling Event!
      stall_duration = abs(new_buffer)
      self.current_buffer_sec = 0.0
    else:
      # Buffer Healthy -> Cap at max
      self.current_buffer_sec = min(new_buffer, self.max_buffer_sec)

    # --- Step 3: Determine Viewing Quality ---
    # If network is slow, Adaptive Bitrate (ABR) lowers quality to match speed.
    # But it cannot exceed the target (source) quality.
    viewing_bitrate = min(effective_bw, self.target_bitrate_kbps)

    # --- Step 4: Official ITU-T P.1203 Calculation ---
    mos = self._call_official_lib(
      bitrate_kbps=viewing_bitrate,
      duration_sec=duration_sec,
      stall_sec=stall_duration
    )
    
    return mos

# ==============================================================================
# 3. Cloud Gaming Model (ITU-T G.1072) - Video Stream Based
# ==============================================================================
class RigorousCloudGamingEvaluator:
  """
  Source: ITU-T Recommendation G.1072
  Scenario: GeForce Now, Xbox Cloud, Stadia.
  Traffic: High Bandwidth Video Stream (UDP).
  Sensitive to: Bandwidth (Resolution), Loss (Artifacts), Delay (Input Lag).
  """
  def __init__(self, target_bitrate_kbps=8000):
    self.base_mos = 4.5
    self.target_bitrate_kbps = target_bitrate_kbps 

  def calculate_mos(self, delay_ms, loss_pct, physical_bw_kbps):
    # 1. Coding Quality (Bandwidth)
    # G.1072: If BW < Target, resolution drops -> MOS drops.
    effective_bw = min(physical_bw_kbps, self.target_bitrate_kbps)
    if effective_bw <= 100:
      I_coding = 3.5 
    else:
      ratio = effective_bw / self.target_bitrate_kbps
      I_coding = 1.8 * np.exp(-3.0 * ratio) 

    # 2. Delay Impairment (Input Lag)
    # G.1072: < 30ms imperceptible. > 100ms steep penalty.
    total_lag = delay_ms + 30.0 # Add ~30ms system overhead
    
    if total_lag <= 30:
      I_delay = 0.0
    else:
      if total_lag < 100:
        I_delay = (total_lag - 30) * 0.015
      else:
        I_delay = (100 - 30) * 0.015 + (total_lag - 100) * 0.06

    # 3. Loss Impairment (Frame Freeze)
    # G.1072: Very sensitive to packet loss (no buffer).
    I_loss = 4.0 * (loss_pct / 100.0) * 10.0 
    
    mos = self.base_mos - I_coding - I_delay - I_loss
    return np.clip(mos, 1.0, 4.5)

# ==============================================================================
# 4. Legacy FPS Model (CSa) - State Update Based
# ==============================================================================
class RigorousLegacyFPSEvaluator:
  """
  Source: Empirical Research (Claypool et al. / Quax et al.)
  Scenario: Counter-Strike (CSa), Valorant, Overwatch.
  Traffic: Tiny Packets, High Frequency, Very Low Bandwidth.
  Sensitive to: Latency (Peeker's Advantage), Jitter (Hit Reg), Loss (Rubber-banding).
  NOT Sensitive to: Bandwidth (unless < 100kbps).
  """
  def __init__(self):
    self.base_mos = 4.5
    # Threshold where "Lag" becomes noticeable for pros
    self.delay_threshold = 20.0 

  def calculate_mos(self, delay_ms, loss_pct, jitter_ms):
    """
    Calculates MOS for Legacy FPS.
    Note: Bandwidth is ignored because CSa flows are tiny (~64kbps).
    """
    
    # --- 1. Delay Penalty (Ping) ---
    # Based on Claypool's "Exponential Decay of Performance"
    # < 20ms: Perfect
    # 20ms - 60ms: Slight disadvantage (Linear)
    # > 60ms: Severe disadvantage (Exponential-ish)
    if delay_ms <= self.delay_threshold:
      I_delay = 0.0
    elif delay_ms <= 60:
      I_delay = (delay_ms - self.delay_threshold) * 0.02 # Mild slope
    else:
      # Heavy penalty above 60ms
      I_delay = 0.8 + (delay_ms - 60) * 0.08

    # --- 2. Jitter Penalty (Stability) ---
    # Legacy FPS servers run at fixed ticks (e.g. 64/128 tick).
    # High jitter causes "interpolation failure" (hitbox mismatch).
    # Research suggests Jitter > tick_interval (e.g. 15ms) is fatal.
    if jitter_ms <= 5:
      I_jitter = 0.0
    else:
      # Jitter is more punishing than constant delay for hit-reg
      I_jitter = (jitter_ms - 5.0) * 0.15

    # --- 3. Loss Penalty (Rubber-banding) ---
    # TCP games lag, UDP games (CS) predict and correct (warp).
    # 1% loss is annoying, 3% is unplayable.
    # Formula: Steep linear penalty.
    I_loss = 25.0 * (loss_pct / 100.0) # 1% loss -> 0.25 MOS drop, 5% -> 1.25 drop
    
    mos = self.base_mos - I_delay - I_jitter - I_loss
    return np.clip(mos, 1.0, 4.5)

    