import argparse
import sys
from argparse import ArgumentParser

class SdnParaser(ArgumentParser):
  def __init__(self):
    ArgumentParser.__init__(
      self,
      description="调整构建Mininet网络拓扑时的参数")
    self.add_argument(
      "-rp", "--remote_port",
      type=int,
      default=None,
      help="远程流表控制器的端口"
    )
    self.add_argument(
      "-dv","--device",
      type=str,
      default="",
      help="指定使用gpu-'0,1,2,3'"
    )
    self.add_argument(
      '--yaml', 
      type=str, 
      default='config.yaml', 
      help='Path to the YAML configuration file (default: ./config.yaml)'
    )
    self.add_argument(
      '--checkpoint',
      type=str,
      default=None,
      help='Path to the training checkpoint file'
    )
    self.add_argument(
      '--verbose',
      action='store_true',
      help='Enable verbose output for debugging'
    )
    self.add_argument(
      '--load_flow',
      type=int,
      default=0,
      help='Load flow rate in Mbps for evaluation (default: 0)'
    )
    


