# 数据层 (data)

- **数据集加载**（被 training / 脚本引用）  
  - `profile_dataset.py`：Profile jsonl、`build_initial_prompt`  
  - `rl_dataset.py`：`load_rl_dataset`（profile / 偏好对）  
  - `sft_dataset.py`：`load_sft_dataset`（SFT 对话格式）
- **已废弃**：`virtual_rlhf_dataset.py`（无引用，仅保留作参考）

数据构建与测试脚本已移至 **scripts/data/**（如 `build_emo_senti_dataset*.py`、`build_emo_test_en.py`、`test_sst2.py`、`test_PN.py`）。

详见 **docs/src_overview.md**。
