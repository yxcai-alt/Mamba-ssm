# Mamba-ssm
This repository is based on [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d)  and [mamba](https://github.com/state-spaces/mamba).

#  Install pre-built wheels

requirements:

```shell
Python  3.10
PyTorch 2.1.1
CUDA    11.8 
```

[download pre-built wheels](https://github.com/yxcai-alt/Mamba-ssm/releases).

```shell
pip install causal_conv1d-1.1.1-cp310-cp310-linux_x86_64.whl
pip install mamba_ssm-1.1.2-cp310-cp310-linux_x86_64.whl
```

# Build and install

```
git clone https://github.com/yxcai-alt/Mamba-ssm.git
pip install setuptools wheel 

cd ./Mamba-ssm/causal-conv1d-1.1.1
CAUSAL_CONV1D_FORCE_BUILD=TRUE python setup.py bdist_wheel
pip install ./dist/causal_conv1d-1.1.1-cp310-cp310-linux_x86_64.whl

cd Mamba-ssm/mamba-1.1.2
MAMBA_FORCE_BUILD=TRUE python setup.py bdist_wheel
pip install ./dist/mamba_ssm-1.1.2-cp310-cp310-linux_x86_64.whl
```

#  Use

```python
from mamba_ssm import Mamba

# use_causal_conv1d: enable causal convolution. If False, use conv1d.
model_causal_conv1d = Mamba(d_model=self.dim_embedding, d_state=64, d_conv=4, expand=2, use_causal_conv1d=True)
model_conv1d = Mamba(d_model=self.dim_embedding, d_state=64, d_conv=4, expand=2, use_causal_conv1d=False)
```



