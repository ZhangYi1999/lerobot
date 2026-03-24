#!/usr/bin/env python
"""Print the structure of a PEFT model from a checkpoint."""

import torch
from peft import PeftModel, PeftConfig

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy

PRETRAINED_PATH = "continuallearning/dit_fft_pretraining_v2_lerobot30_seed1000"
ADAPTER_PATH = "outputs/lora_to_clare/clare_converted_v2_merged/adapter"
DATASET_REPO_ID = "continuallearning/real_0_put_bowl_filtered"

# Load base policy
policy_cfg = PreTrainedConfig.from_pretrained(PRETRAINED_PATH)
policy_cfg.pretrained_path = PRETRAINED_PATH
ds_meta = LeRobotDatasetMetadata(DATASET_REPO_ID)
policy = make_policy(cfg=policy_cfg, ds_meta=ds_meta)
policy.eval()

# Load PEFT model
peft_policy = PeftModel.from_pretrained(policy, ADAPTER_PATH, is_trainable=False, autocast_adapter_dtype=False)

print(peft_policy)
