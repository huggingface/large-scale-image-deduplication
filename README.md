# Data Deduplication

## Environment Setup
```bash
uv init --bare --python 3.10
uv sync --python 3.10
source .venv/bin/activate
uv pip install -r ./requirements.txt
```

## [SSCD](https://github.com/facebookresearch/sscd-copy-detection)

Generates a descriptor for the data with the SSCD Model from FAIR