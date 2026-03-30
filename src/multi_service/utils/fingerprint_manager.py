from omegaconf import DictConfig
import torch
import random

from loguru import logger


class BankTrafficManager:
    """
    使用预存指纹库的流量管理器
    """

    def __init__(self, src_nodes: list, dst_nodes: list,
                 fgpt_path: str) -> None:
        self.src_nodes = src_nodes
        self.dst_nodes = dst_nodes
        self.fgpt_path = fgpt_path
        # 加载指纹库
        raw_bank = torch.load(self.fgpt_path, map_location='cpu')

        # 将所有 Tensor 统一为 (1, 30, 2) 维度
        self.bank = {}
        for k, v_list in raw_bank.items():
            processed_list = []
            for t in v_list:
                if t.dim() == 2:
                    t = t.unsqueeze(0)
                processed_list.append(t)
            self.bank[k.lower()] = processed_list

        logger.debug(
            f"Fingerprint Bank loaded and pre-processed from {self.fgpt_path}")

    def generate_batch(self,
                       batch_size: int,
                       src_nodes: list | None = None,
                       dst_nodes: list | None = None) -> list:
        """
        均衡生成各类型的流，从库中随机采样指纹
        """
        # 避免循环引用，在此处导入
        from ..env.flow_generator import FlowType

        flows = []
        all_types = list(FlowType)
        num_types = len(all_types)

        # 1. 规划每种类型的数量 (Balanced Sampling)
        base_count = batch_size // num_types
        remainder = batch_size % num_types

        target_types = []
        for t in all_types:
            target_types.extend([t] * base_count)

        # 余数随机分配
        if remainder > 0:
            target_types.extend(random.sample(all_types, remainder))

        # 打乱顺序
        random.shuffle(target_types)

        # 确定源宿节点池
        src_pool = src_nodes if src_nodes is not None else self.src_nodes
        dst_pool = dst_nodes if dst_nodes is not None else self.dst_nodes

        for f_type in target_types:
            # 2. 随机选择源宿节点
            s = random.choice(src_pool)
            d = random.choice(dst_pool)

            # 3. 从预处理过的库中采样
            type_key = f_type.name.lower()
            available_fingerprints = self.bank.get(type_key, [])

            if not available_fingerprints:
                logger.error(f"No fingerprint data for type: {f_type.name}!")
                # 兜底：生成全 0 的张量
                fingerprint = torch.zeros((1, 30, 2))
            else:
                # 随机抽取并移动到目标设备
                fingerprint = random.choice(available_fingerprints)

            # 构建 Flow 对象
            flow_obj = type(
                'Flow', (), {
                    'src': s,
                    'dst': d,
                    'label': f_type.value - 1,
                    'flow_type': f_type,
                    'fingerprint': fingerprint
                })
            flows.append(flow_obj)

        return flows
