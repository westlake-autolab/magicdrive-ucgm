# MagicDrive-V2 Custom Modifications

This repository contains custom modifications to the MagicDrive-V2 codebase for improved training stability and debugging capabilities, tested on H800.

## 🔧 Main Modifications

### 1. UCGM Scheduler Fixes (`magicdrivedit/schedulers/ucgm.py`)
- **Fixed tensor size mismatch errors** in validation CFG logic
- **Added tensor broadcasting fixes** for multi-dimensional operations
- **Improved model output parsing** to handle different return formats
- **CFG drop condition fix** to align with magicdrivedit

### 2. Training Utils (`magicdrivedit/utils/train_utils.py`)
- **Fixed CFG compatibility** for UCGM scheduler in validation
- **Prevented double batch size** by skipping `add_null_condition` for UCGM
- **Maintained separate CFG paths** for different scheduler types

### 3. Video Saving (`magicdrivedit/datasets/utils.py`) (This change would not be needed if using a higher version of MoviePy)
- **Fixed MoviePy compatibility** by removing deprecated `verbose` parameter
- **Maintained video quality** with proper bitrate and logger settings

### 4. VAE Debugging (`magicdrivedit/models/vae/vae_cogvideox.py`)
- **Added logging** for VAE encode/decode operations
- **Enhanced shape tracking** for debugging tensor dimensions

### 5. Attention Debugging (`magicdrivedit/models/layers/blocks.py`)
- **Debug Flash Attention detection** and logging

### 6. Training and Inference Configurations
- **Compatiblilty** with adjusted ucgm scheduler
- **Debugging flags** for variance monitoring

## 📁 File Structure

```
├── magicdrivedit/
│   ├── schedulers/
│   │   └── ucgm.py                    # Fixed CFG logic and sampling
│   ├── utils/
│   │   └── train_utils.py             # Fixed validation CFG compatibility
│   ├── datasets/
│   │   └── utils.py                   # Fixed MoviePy compatibility (only if used with specific moviepy version)
│   ├── models/
│   │   ├── vae/
│   │   │   └── vae_cogvideox.py       # Enhanced VAE debugging
│   │   └── layers/
│   │       └── blocks.py              # Enhanced attention debugging
├── finetune.py                        # Training script with 10k step limit
├── training_config*.py                # Optimized training configurations
└── README.md                          # This file
```

## 🚀 Usage

These modifications are designed to be drop-in replacements for the original MagicDrive-V2 files.

## 🔍 Debug Tools

- **GPU Memory Monitoring**: Tracking of memory usage
- **Layer-wise Monitoring variance patch**: Block-by-block execution tracking of output variance across the tensor


