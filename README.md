# FIPL-DA: Federated Implicit Prototype Learning with Domain-Aware Alignment

A privacy-preserving federated learning algorithm targeting **domain skew**. 

## Installation

```bash
# Python 3.8+, PyTorch 1.13+
pip install -r requirements.txt
```

## Data preparation

Datasets must be organized by domain directory; each domain directory contains `class/` subdirectories in ImageFolder layout. The default data root is `data/` under the project, e.g.:

```
data/
├── PACS/
│   ├── photo/            # domain dir, containing class subdirectories
│   ├── art_painting/
│   ├── cartoon/
│   └── sketch/
├── digits/
│   ├── mnist/  usps/  svhn/  synth/  mnist-m/
└── ...
```

Supported datasets (`--dataset`): `fl_pacs`, `fl_digits`.

## Run

```bash
# non-DP baseline
python run_test.py --model fipl_da --dataset fl_pacs --backbone lightresnet \
    --communication_epoch 50 --local_epoch 1 --online_ratio 1.0 \
    --eps 0.0 --device_id 0

# DP mode (eps=5.0, clipping threshold C=3.0)
python run_test.py --model fipl_da --dataset fl_pacs --backbone lightresnet \
    --communication_epoch 50 --local_epoch 1 --online_ratio 1.0 \
    --eps 5.0 --max_grad_norm 3.0 --device_id 0
```

Common arguments:

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset` | Dataset | `fl_pacs` |
| `--backbone` | Backbone network (only `lightresnet`) | `lightresnet` |
| `--communication_epoch` | Number of communication rounds | `50` |
| `--local_epoch` | Local training epochs | `1` |
| `--online_ratio` | Online client ratio | `1` |
| `--eps` | Privacy budget ε (<=0 disables DP) | `0` |
| `--max_grad_norm` | Gradient clipping threshold C | `1.0` |
| `--num_clients` | Total number of clients (0 = dataset default) | `0` |
| `--mu` | Implicit prototype loss weight (default from best_args) | — |
| `--mu_decay` | Per-round decay factor for `mu` (`mu *= mu_decay`) | `0.9` |
| `--scale` | Learnable initial temperature of the cosine classifier | `10.0` |
| `--etf_lambda` | Server prototype separation strength (0 = plain FedAvg classifier averaging) | `1.0` |
| `--proto_margin` | Prototype separation margin (only penalizes pairs with cosine > margin) | `-0.1` |
| `--etf_lr` | Server prototype separation GD step size | `0.1` |
| `--etf_init` | Initialize classifier weights with simplex ETF (`--no_etf_init` to disable) | on |

Results are saved in `result/{dataset}/fipl_da/para{N}/`, including `acc.csv`, `summary.json`, etc.

## Project structure

```
├── main.py                     # Entry point: argument parsing + training launch
├── run_test.py                 # Command-line launcher
├── models/
│   ├── fipl_da.py              # FIPL-DA core implementation
│   └── utils/
│       ├── federated_model.py  # Parameter-pool architecture base class + DP infra
│       └── prototype_viz.py    # Prototype t-SNE visualization
├── datasets/                   # Dataset loading and domain-skew partitioning
├── backbone/                   # LightResNet backbone network
├── utils/                      # DP / training loop / logging
└── client_data/                # Client data index cache (generated at runtime)
```

## Citation

<!-- Paper citation info (to be added) -->
