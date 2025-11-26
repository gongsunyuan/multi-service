import numpy as np

class FullG107Calculator:
  def __init__(self):
    # 1. 基本信噪比 (Basic Signal-to-Noise Ratio)
    # G.107 推荐值: 93.2 (窄带) 或 94.76 (宽带，暂取窄带标准)
    self.R0 = 93.2
    
    # 2. 同步损伤 (Simultaneous Impairment)
    # 对于现代数字网络，量化失真极小，取默认值
    self.Is = 0.0
    
    # 3. 优势因子 (Advantage Factor)
    # 0: 有线网络 (Mininet default)
    # 5: 移动通信
    # 10: 卫星通信
    self.A = 0.0 

    # 4. 编解码器参数 (G.711 Codec Profile)
    self.Ie_base = 0.0   # G.711 基础损伤
    self.Bpl = 4.3       # G.711 丢包鲁棒性因子

  def calculate_r_factor(self, delay_ms, loss_pct, jitter_ms):
    """
    Strict implementation of ITU-T G.107
    Args:
      delay_ms: One-way delay (d)
      loss_pct: Packet loss probability (0-100)
      jitter_ms: Jitter
    """
    # --- Step 1: Effective Latency Calculation ---
    # G.107 规定: 有效延迟 Ta = OneWayDelay + Pdd (Packetization Delay)
    # G.711 打包延迟通常为 10ms 或 20ms，我们取 10ms
    # Jitter Buffer 带来的额外延迟通常设为 2 * Jitter
    Ta = delay_ms + (2 * jitter_ms) + 10.0 

    # --- Step 2: Id (Delay Impairment) Calculation ---
    # 严格公式: Id = Idte + Idle + Idd
    # 在纯网络传输中，重点是 Idd (Absolute Delay Impairment)
    # 公式来源: G.107 Eq. (4) - (6)
    
    if Ta <= 100:
      Id = 0.0  # 100ms 以内几乎无感知
    else:
      X = np.log(Ta / 100.0) / np.log(2.0) # log2(Ta/100)
      Id = 25.0 * ( (1 + X**6)**(1/6) - 3 * (1 + (X/3)**6)**(1/6) + 2 )

    # --- Step 3: Ie,eff (Effective Equipment Impairment) ---
    # 考虑丢包的影响。公式来源: G.107 Eq. (3-28)
    # Ppl: Packet Loss Probability (0.0 to 1.0) -> input is percent
    Ppl = loss_pct # G.107 uses percent directly in some versions, check scaling.
    # 标准公式: Ie,eff = Ie + (95 - Ie) * Ppl / (Ppl + Bpl)
    # 注意: 如果 loss_pct 是 0-100
    
    Ie_eff = self.Ie_base + (95 - self.Ie_base) * (Ppl / (Ppl + self.Bpl))
    
    # --- Step 4: Final R Calculation ---
    R = self.R0 - self.Is - Id - Ie_eff + self.A
    return R

  def r_to_mos(self, R):
      """
      ITU-T G.107 Eq. (30): Conversion R -> MOS
      """
      if R < 0: return 1.0
      if R > 100: return 4.5
      
      mos = 1 + 0.035 * R + 7e-6 * R * (R - 60) * (100 - R)
      return np.clip(mos, 1.0, 4.5)

class FullP1203Calculator:
  def __init__(self):
    # 预设设备显示参数 (假设 1080p 屏幕)
    self.device_res_x = 1920
    self.device_res_y = 1080
    
    # P.1203 参数系数 (简化提取版，完整版有几百个系数)
    self.o1 = 4.69 # Max score
    self.o2 = 4.0  # Quantization parameter
  
  def calculate_stall(received_bw_kbps, packet_loss_rate, duration_sec):
    target_bitrate = 5000  # 假设我们要看 5Mbps 的 1080p 视频 (Demand)
    stall_time = 0.0

    # --- 1. Bandwidth Stall (下载太慢) ---
    # 我们下载这些数据实际花了多久？
    # Time_Needed = Data_Size / Speed
    # Data_Size = Target_Bitrate * Duration
    if received_bw_kbps < 10: # 防止除以0
      download_time = 999 # 无限卡顿
    else:
      # 如果带宽只有目标的一半，下载就要花 2倍的时间
      download_time = (target_bitrate * duration_sec) / received_bw_kbps
    
    # 额外的耗时就是卡顿时间
    if download_time > duration_sec:
      stall_time += (download_time - duration_sec)

    # --- 2. Loss Stall (丢包重传) ---
    # 经验公式：丢包率 > 2% 开始导致明显卡顿
    # 假设每 1% 的丢包会导致 0.5秒 的等待 (简单拟合)
    if packet_loss_rate > 2.0:
      stall_time += (packet_loss_rate - 2.0) * 0.5

    # 卡顿时间不能超过物理时间的限制 (逻辑边界)
    # 但在 P.1203 中，stall 可以很长，这里我们设个上限防止数值爆炸
    return min(stall_time, duration_sec * 5)