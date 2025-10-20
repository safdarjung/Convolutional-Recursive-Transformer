#!/bin/bash
# Clean setup using python -m pip

# 1. Remove old environment
conda deactivate 2>/dev/null || true
conda env remove -n crt_v2 -y

# 2. Create fresh Python 3.12 environment
conda create -n crt_v2 python=3.12 -y

# 3. Activate environment
conda activate crt_v2

# 4. Verify environment
echo "============================================"
echo "Environment Check:"
which python
python --version
echo "============================================"

# 5. Upgrade pip first
python -m pip install --upgrade pip

# 6. Install PyTorch 2.8.0 with CUDA 12.6
echo "Installing PyTorch 2.8.0..."
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu126

# 7. Verify PyTorch & CUDA
echo "Verifying PyTorch..."
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# 8. Install Transformers & ML libraries
echo "Installing ML libraries..."
python -m pip install transformers==4.55.4 \
  datasets==4.0.0 \
  accelerate==1.10.1 \
  peft==0.17.1 \
  sentencepiece \
  tokenizers \
  huggingface-hub==0.34.4 \
  safetensors

# 9. Install scientific computing libraries
echo "Installing scientific libraries..."
python -m pip install numpy \
  scipy==1.14.1 \
  matplotlib \
  seaborn \
  tqdm \
  pandas \
  jupyter \
  ipywidgets \
  psutil

# 10. Install optional tools
echo "Installing optional tools..."
python -m pip install wandb==0.21.1 \
  tensorboard==2.19.0

# 11. Final verification
echo ""
echo "============================================"
echo "Final Verification:"
echo "============================================"
python << 'EOF'
import sys
import torch
import transformers
import datasets
import accelerate
import numpy as np
import matplotlib
import json
from datetime import datetime

print(f"\n✅ Python: {sys.version.split()[0]}")
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ Transformers: {transformers.__version__}")
print(f"✅ Datasets: {datasets.__version__}")
print(f"✅ Accelerate: {accelerate.__version__}")
print(f"✅ NumPy: {np.__version__}")

if torch.cuda.is_available():
    print(f"\n🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")
    print(f"🔧 CUDA: {torch.version.cuda}")
    
    # GPU test
    x = torch.randn(1000, 1000, device='cuda')
    y = x @ x.T
    print(f"✅ GPU computation working!")
    
    print(f"\n💡 RTX 4050 6GB Config:")
    print(f"   Batch size: 4")
    print(f"   Grad accumulation: 4 steps")
    print(f"   Effective batch: 16")
    print(f"   Sequence length: 512")
    print(f"   Mixed precision: bfloat16")
    print(f"   Expected time: ~2-2.5 hours")
else:
    print("\n❌ CUDA not available!")

# Save specs
specs = {
    "python": sys.version.split()[0],
    "pytorch": torch.__version__,
    "transformers": transformers.__version__,
    "datasets": datasets.__version__,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
    "vram_gb": torch.cuda.get_device_properties(0).total_memory/1024**3 if torch.cuda.is_available() else 0,
    "created": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

with open('environment_specs.json', 'w') as f:
    json.dump(specs, f, indent=2)

print("\n📄 Specs saved to environment_specs.json")
EOF

echo ""
echo "============================================"
echo "🎉 SETUP COMPLETE!"
echo "============================================"
echo ""
echo "To start training:"
echo "  conda activate crt_v2"
echo "  python train_crt_v2.py"
echo ""
