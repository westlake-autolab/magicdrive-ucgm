# MagicDrive-V2 Custom Modifications

This repository contains custom modifications to the MagicDrive-V2 codebase for improved training stability and debugging capabilities.

## 🔧 Main Modifications

### 1. UCGM Scheduler Fixes (`magicdrivedit/schedulers/ucgm.py`)
- **Fixed tensor size mismatch errors** in validation CFG logic
- **Restored complete sampling loop** with proper time step handling
- **Added tensor broadcasting fixes** for multi-dimensional operations
- **Improved model output parsing** to handle different return formats

### 2. Training Utils (`magicdrivedit/utils/train_utils.py`)
- **Fixed CFG compatibility** for UCGM scheduler in validation
- **Prevented double batch size** by skipping `add_null_condition` for UCGM
- **Maintained separate CFG paths** for different scheduler types

### 3. Video Saving (`magicdrivedit/datasets/utils.py`)
- **Fixed MoviePy compatibility** by removing deprecated `verbose` parameter
- **Maintained video quality** with proper bitrate and logger settings

### 4. VAE Debugging (`magicdrivedit/models/vae/vae_cogvideox.py`)
- **Added comprehensive logging** for VAE encode/decode operations
- **Enhanced shape tracking** for debugging tensor dimensions
- **Improved error diagnostics** for VAE-related issues

### 5. Attention Debugging (`magicdrivedit/models/layers/blocks.py`)
- **Added layer-wise monitoring** for attention mechanisms
- **Enhanced Flash Attention detection** and logging
- **Improved debugging for sequence parallelism**

### 6. Training Configurations
- **Optimized memory usage** with adjusted batch sizes and sequence parallelism
- **Enhanced debugging flags** for comprehensive monitoring
- **Improved bucket configurations** for variable-length training

## 🎯 Key Improvements

1. **Stability**: Fixed critical tensor size mismatches that caused training crashes
2. **Debugging**: Added comprehensive logging for GPU memory, VAE operations, and attention mechanisms  
3. **Compatibility**: Resolved MoviePy API changes and scheduler CFG conflicts
4. **Performance**: Optimized configurations for H800 GPU training

## 📁 File Structure

```
├── magicdrivedit/
│   ├── schedulers/
│   │   └── ucgm.py                    # Fixed CFG logic and sampling
│   ├── utils/
│   │   └── train_utils.py             # Fixed validation CFG compatibility
│   ├── datasets/
│   │   └── utils.py                   # Fixed MoviePy compatibility
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

These modifications are designed to be drop-in replacements for the original MagicDrive-V2 files. Simply replace the corresponding files in your MagicDrive-V2 installation.

## 🔍 Debug Features

- **GPU Memory Monitoring**: Real-time tracking of memory usage
- **VAE Operation Logging**: Detailed input/output shape analysis  
- **Flash Attention Detection**: Automatic detection and reporting
- **Layer-wise Monitoring**: Block-by-block execution tracking

## ⚡ Performance Optimizations

- **Tensor Broadcasting**: Proper handling of multi-dimensional operations
- **Memory Management**: Optimized configurations for large-scale training
- **CFG Optimization**: Separate conditional/unconditional forward passes

---
*Created for enhanced MagicDrive-V2 training stability and debugging*

