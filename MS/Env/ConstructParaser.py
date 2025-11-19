import argparse
import sys
from argparse import ArgumentParser

class TopoParaser(ArgumentParser):
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

    


