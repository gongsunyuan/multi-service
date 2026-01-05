import logging
import sys
import torch
import numpy as np
import os
import networkx as nx
from datetime import datetime
from threading import Lock
from pathlib import Path

class VerboseLogger:
	"""
	统一日志管理器 (单例模式)
	封装了控制台输出 (tqdm 安全)、文件记录以及针对 QoS、路径和流量指纹的专用格式。
	"""
	_instance = None
	_lock = Lock()

	def __new__(cls):
		if not cls._instance:
			with cls._lock:
				if not cls._instance:
					cls._instance = super(VerboseLogger, cls).__new__(cls)
					cls._instance._initialized = False
		return cls._instance

	def __init__(self):
		if self._initialized: return
		self.enabled = True
		self.log_to_console = True
		self.debug_mode = False  # [New] Debug Mode
		self.tag_width = 12
		self.pbar = None
		self._file_logger = None
		self._initialized = True

	def configure(self, log_file: str | None = None, verbose: bool = True, log_to_console: bool = True, debug_mode: bool = False) -> None:
		"""[初始化入口] 在程序启动时配置一次。"""
		self.enabled = verbose
		self.log_to_console = log_to_console
		self.debug_mode = debug_mode
		if log_file:
			logger = logging.getLogger("SDN_File_Log")
			logger.setLevel(logging.INFO)
			if logger.hasHandlers(): logger.handlers.clear()
			try:
				os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
				fh = logging.FileHandler(log_file, encoding='utf-8')
				fh.setFormatter(logging.Formatter('%(message)s'))
				logger.addHandler(fh)
				self._file_logger = logger
			except Exception as e:
				print(f"[Logger Err] Failed to setup log file: {e}")

	def set_pbar(self, pbar):
		"""绑定 tqdm 进度条。"""
		self.pbar = pbar

	def log(self, message: str, tag: str | None = None, timestamp: datetime | None = None, log_to_console: bool | None = None, **kwargs):
		"""核心打印函数 (替代 vprint)。"""
		# [New] Filter Debug logs if debug_mode is False
		if tag and tag.lower() == "debug" and not self.debug_mode:
			return

		# [Fix] Correct logic for log_to_console override
		if log_to_console is None: 
			log_to_console = self.log_to_console
			
		if not self.enabled: return
		now = timestamp if timestamp else datetime.now()
		time_str = now.strftime("%H:%M:%S.%f")[:-3]
		tag_prefix = f"[{str(tag)[:self.tag_width]:^{self.tag_width}}] " if tag else ""
		full_msg = f"[{time_str}] {tag_prefix}{message}"
		if self._file_logger: self._file_logger.info(full_msg)
		if log_to_console:
			if self.pbar is not None: self.pbar.write(full_msg)
			else: print(full_msg, **kwargs)

	# =========================================================================
	# 业务专用方法
	# =========================================================================

	def log_matrix(self, tensor: torch.Tensor, filename: str, flow_id: int) -> None:
		"""
		[整合] 将流量指纹矩阵写入文件。
		[cite_start]将 [1, N, M] 形状的 Tensor 写入文件，保持 N x M 结构 [cite: 204]。
		"""
		if not self.enabled: return
		
		# [cite_start]1. 转换数据 (N, M) [cite: 204]
		data_np = tensor.squeeze(0).cpu().numpy()
		Path(filename).parent.mkdir(parents=True, exist_ok=True)
		
		# [cite_start]2. 写入文件 [cite: 205]
		try:
			with open(filename, 'a') as f:
				f.write(f"\n{'='*68}\n")
				f.write(f"Flow ID: {flow_id} | Shape: {data_np.shape}\n") 
				f.write("Features: [Size/1600, IAT/0.1]\n")
				f.write(f"{'-'*68}\n")
				np.savetxt(f, data_np, fmt='%.6f', delimiter=', ')
			self.log(f"Fingerprint matrix saved to {filename}", tag="Storage")
		except IOError as e:
			self.log(f"Failed to write matrix: {e}", tag="Error")

	def log_qos(self, service_type: str, delay: float, jitter: float, bw: float, loss: float, tag: str = "QoS Report"):
		"""QoS 报告格式化 (简洁对齐版)。"""
		if not self.enabled: return
		locked_now = datetime.now()
		self.log(f"Report for {service_type}:", tag=tag, timestamp=locked_now)
		indent = " " * (self.tag_width + 3)
		def _p(label: str, val: float, unit: str):
			self.log(f"{indent}> {label:<10} {val:>8.2f} {unit}", tag=None, timestamp=locked_now)
		_p("Delay:", delay, "ms")
		_p("Jitter:", jitter, "ms")
		_p("Bandwidth:", bw, "Mbps")
		_p("Loss Rate:", loss, "%")

	def log_path(self, G: nx.Graph, path: list):
		"""路径详情分析。"""
		if not self.enabled: return
		self.log("-" * 120)
		self.log(f"Path Detail: {path}")
		header = (f" {'Hop (u->v)':<12} | {'Cap':<6} | {'Util%':<7} | {'Avail':<6} | "
							f"{'Delay':<8} | {'Loss%':<6} || {'[Node u] Buffer%':<16} | "
							f"{'[Node u] ProcD':<14} | {'Status'}")
		self.log(header)
		self.log("-" * 120)
		total_delay = 0.0
		for u, v in zip(path[:-1], path[1:]):
			e_data = G[u][v]
			cap = e_data.get('bandwidth', 100.0)
			util = e_data.get('utilization', 0.0)
			delay = e_data.get('delay', 1.0)
			loss = e_data.get('loss', 0.0)
			buff = G.nodes[u].get('buffer_occupancy', 0.0)
			proc = G.nodes[u].get('proc_delay', 0.0)
			total_delay += delay + proc
			status = "OK"
			if loss > 0.001: status = "PKT_DROP"
			elif util > 0.95: status = "CONGEST"
			elif buff > 0.80: status = "BUF_FULL"
			line = (f" {u:<2} -> {v:<2}     | {cap:<6.0f} | {util:<7.2%} | {cap*(1-util):<6.1f} | "
							f"{delay:<6.2f}ms | {loss:<6.2%} || {buff:<16.2%} | {proc:<14.4f} | {status}")
			self.log(line)
		self.log("-" * 120)
		self.log(f" >>> Path Summary: Total Delay ~ {total_delay:.2f} ms")
	
	def log_network_status(self, G: nx.Graph):
		"""
		打印全网节点与链路状态。
		节点按介数中心性排序，边按利用率排序。
		"""
		if not self.enabled: return
		
		# --- 1. 节点状态 (按 Betweenness 降序排序) ---
		self.log("", tag="Net Status")
		self.log("=" * 60)
		self.log(f"[Nodes] Sorted by Betweenness Centrality")
		node_header = f" {'Node':<4} | {'Centrality':<10} | {'Buffer%':<10} | {'ProcDelay'}"
		self.log(node_header)
		self.log("-" * 60)
		
		# 使用 lambda 获取 'betweenness' 进行排序
		sorted_nodes = sorted(
			G.nodes(data=True), 
			key=lambda x: x[1].get('buffer_occupancy', 0.0), 
			reverse=True)
		
		for n, data in sorted_nodes:
			buff = data.get('buffer_occupancy', 0.0)
			proc = data.get('proc_delay', 0.0)
			betw = data.get('betweenness', 0.0)
			
			marker = " [HUB]" if betw > G.graph.get('avg_betw', 0.5) else "" # 假设有一个平均参考值
			self.log(f" {n:<4} | {betw:<10.4f} | {buff:<10.2%} | {proc:<10.4f}{marker}")
		
		self.log("-" * 60)

		# --- 2. 链路状态 (按 Utilization 降序排序) ---
		self.log(f"[Edges] Sorted by Link Utilization")
		edge_header = f" {'Link':<10} | {'Util%':<10} | {'Cap(Mbps)':<10} | {'Delay(ms)':<10} | {'Loss%'}"
		self.log(edge_header)
		self.log("-" * 60)
		
		# 使用 lambda 获取 data 中的 'utilization' 进行排序
		sorted_edges = sorted(G.edges(data=True), 
													key=lambda x: x[2].get('utilization', 0.0), 
													reverse=True)
		
		for u, v, data in sorted_edges:
			cap = data.get('capacity', 100.0)
			util = data.get('utilization', 0.0)
			delay = data.get('delay', 0.0)
			loss = data.get('loss', 0.0)
			
			status = ""
			if util > 0.90: status = " [CONGESTED]"
			if loss > 0.01: status += " [DROP]"
			
			self.log(f" {u:<2}->{v:<2}     | {util:<10.2%} | {cap:<10.1f} | {delay:<10.2f} | {loss:.2%}{status}")
			
		self.log("=" * 80)

# 全局单例导出 
logger = VerboseLogger()

# 兼容性别名 
vprint = logger.log
vprint_qos = logger.log_qos
vprint_path_status = logger.log_path
vprint_matrix = logger.log_matrix
vprint_network_status = logger.log_network_status
# =========================================================================
# 测试逻辑
# =========================================================================

if __name__ == "__main__":
	import time
	from tqdm import tqdm
	
	logger.configure(log_file="./test_log.txt", verbose=True)
	logger.log("System starting...", tag="System")
	
	# 测试矩阵写入
	mock_tensor = torch.randn(1, 5, 2)
	logger.log_matrix(mock_tensor, "./fingerprint_test.txt", flow_id=101)
	
	# 测试 Tqdm
	pbar = tqdm(total=5)
	logger.set_pbar(pbar)
	for i in range(5):
		time.sleep(0.2)
		logger.log(f"Training Step {i} - Loss: 0.01", tag="Train")
		pbar.update(1)
	
	logger.set_pbar(None)
	print("Done. Check ./test_log.txt and ./fingerprint_test.txt")
	time.sleep(0.1)