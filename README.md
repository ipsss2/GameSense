
# Cultivating Game Sense for Yourself: Making VLMs Gaming Experts


Official implementation of "**Cultivating Game Sense for Yourself: Making VLMs Gaming Experts**" accepted to **ACL** 2025.  
This repository contains the complete codebase for experiments described in the paper.

---

## 📄 Paper Information
- **Title**: Cultivating Game Sense for Yourself: Making VLMs Gaming Experts
- **Authors**: Wenxuan lu, Jiangyang He, Zhanqiu Zhang, Steven Y. Guo, Tianning Zang
- **Conference**: ACL 2025  
- **PDF**: https://aclanthology.org/2025.acl-long.643/  

---


## 📁 Project Architecture Overview

### Core Components
```
├── action_manager/            # Action execution & keyboard mapping
│   ├── model/                 # Action models
│   │   └── wukong_trained/    # Pretrained ResNet models
│   └── New_action_mamager.py  # Controller implementation

├── agent/                     # AI agent modules
│   ├── fast_module_trainner_agent/ # RL training framework
│   └── player_agent/          # Cognitive agent components
│       ├── Self_Reflection.py # Experience learning
│       ├── task_planner.py    # Goal-oriented planning
│       └── state_inference.py # Game state analysis

├── utils/                     # Utility functions
│   ├── bar_detector.py        # Health/Mana/Boss status detection
│   └── video_capture.py       # Screen capture module

├── boss_env.py                # Reinforcement Learning environment for boss battles
├── fight_with_boss.py         # RL training entry point
└── new_agent_with_map&history.py # Main agent loop entry point
```

---

## 🚀 Key Implementation Details

### 1. Agent Execution Flow
- **Main Entry**: [new_agent_with_map&history.py](new_agent_with_map&history.py) implements the complete agent loop:
  ```bash
  python new_agent_with_map&history.py
  ```
  - Integrates map analysis, task planning, and action execution
  - Uses RAG for memory-based decision making
  - Captures and stores experience for reflection


### 2. Boss Battle RL Training (Standalone & Integrated)
- **Training Entry**: [fight_with_boss.py](fight_with_boss.py) offers **dual usage modes**:
  ```bash
  python fight_with_boss.py  # Standalone RL training mode
  ```

#### 🔍 Standalone Mode
- **Use Case**: Dedicated boss battle training
- **Key Features**:
  - Implements Double DQN algorithm in `models/new_model.py`
  - Uses ResNet model from `action_manager/model/wukong_trained/`
- **Training Workflow**:
  1. Start cheat engine (`FLiNG Trainer`) for teleportation
  2. Launch game and navigate to boss arena
  3. Run training: `python fight_with_boss.py`
  4. Automatically detects boss health using [bar_detector.py](utils/bar_detector.py)

#### 🧩 Integrated Mode
- **Embedded Usage**: Can be imported into main agent loop for autonomous gameplay
- **Use Case**: The integration utilizes the Controller class in [New_action_mamager.py](action_manager/New_action_mamager.py#42) (in line 42) to seamlessly launch training scripts from the main agent loop.

---

## 🎮 Game Environment Setup

### Screen Capture Configuration
Adjust screen capture parameters in `utils/bar_detector.py` according to your display:
```python
# Screen capture parameters (modify according to your display)
screen_width, screen_height = pyautogui.size()
capture_width = 1600  # Desired capture width
capture_height = 900  # Desired capture height
left = (screen_width - capture_width) // 2  # X-coordinate for capture
top = (screen_height - capture_height) // 2  # Y-coordinate for capture
```

### Cheat Engine Requirements
For stable boss battle training, use **FLiNG Trainer** cheat engine:
- **Key Bindings**:
  - `K`: Mark current position
  - `L`: Teleport to marked position
- Required for environment reset and position control during RL training

---

## 📎 Citation
```bibtex
@inproceedings{lu2025cultivating,
  title={Cultivating Gaming Sense for Yourself: Making VLMs Gaming Experts},
  author={Lu, Wenxuan and He, Jiangyang and Zhang, Zhanqiu and Guo, Steven Y and Zang, Tianning},
  booktitle={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={13132--13152},
  year={2025}
}
```

