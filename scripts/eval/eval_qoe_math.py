import sys
import os
import torch

# Add project root to path
sys.path.append(os.getcwd())

from src.env.E_model import (
  FullG107Calculator, 
  RigorousVideoEvaluator, 
  RigorousLegacyFPSEvaluator
)

def test_math_logic():
  print("====== [Unit Test] Verifying QoE Math Models ======")
  
  # 1. Initialize Calculators
  voip = FullG107Calculator()
  video = RigorousVideoEvaluator(target_bitrate_kbps=5000) # 1080p
  game = RigorousLegacyFPSEvaluator() # CSa

  # --- Scenario A: Perfect Network ---
  # Delay=5ms, Loss=0%, BW=100Mbps
  print("\n[Scenario A] Perfect Network (5ms, 0% Loss, 100Mbps)")
  mos_v = voip.calculate_mos(delay_ms=5, loss_pct=0, jitter_ms=1)
  mos_s = video.calculate_mos(loss_pct=0, rtt_ms=10, physical_bw_kbps=100000, duration_sec=6.0, stateless_mode=True)
  mos_g = game.calculate_mos(delay_ms=5, loss_pct=0, jitter_ms=1)
  
  print(f"  VoIP MOS: {mos_v:.2f} (Expected: > 4.4)")
  print(f"  Video MOS: {mos_s:.2f} (Expected: > 4.5)")
  print(f"  Game MOS: {mos_g:.2f} (Expected: 4.5)")

  # --- Scenario B: High Latency (Satellite Link) ---
  # Delay=200ms, Loss=0%, BW=100Mbps
  print("\n[Scenario B] High Latency (200ms, 0% Loss)")
  mos_v = voip.calculate_mos(delay_ms=200, loss_pct=0, jitter_ms=5)
  mos_g = game.calculate_mos(delay_ms=200, loss_pct=0, jitter_ms=5)
  
  print(f"  VoIP MOS: {mos_v:.2f} (Expected: ~3.5-4.0, VoIP handles latency ok)")
  print(f"  Game MOS: {mos_g:.2f} (Expected: < 2.0, FPS hates latency!)")

  # --- Scenario C: Low Bandwidth (Congestion) ---
  # Delay=20ms, Loss=0%, BW=2Mbps (Target is 5Mbps)
  print("\n[Scenario C] Low Bandwidth (2Mbps, Video needs 5Mbps)")
  mos_s = video.calculate_mos(loss_pct=0, rtt_ms=40, physical_bw_kbps=2000, duration_sec=4.0, stateless_mode=True)
  print(f"  Video MOS: {mos_s:.2f} (Expected: < 3.0, Stalling/Low Res)")

  # --- Scenario D: Packet Loss ---
  # Delay=20ms, Loss=5%
  print("\n[Scenario D] High Loss (5%)")
  mos_v = voip.calculate_mos(delay_ms=20, loss_pct=5.0, jitter_ms=5)
  mos_g = game.calculate_mos(delay_ms=20, loss_pct=5.0, jitter_ms=5)
  print(f"  VoIP MOS: {mos_v:.2f} (Expected: Low, sound breaks)")
  print(f"  Game MOS: {mos_g:.2f} (Expected: 1.0, unplayable)")

if __name__ == "__main__":
  test_math_logic()