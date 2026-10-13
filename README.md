# Noise: Speedrun for pretraining without backpropagation
![Noise baselines](val_loss_animation.gif)

Noise is a pretraining speedrun for learning algorithms that never run a backward pass. We think replacing
backprop with a much better credit assignment algorithm is inevitable. Deep learning's architectures,
optimizers and hardware have co-evolved with backprop, but as compute grows, the
[bitter lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) suggests that more general,
brute-force learning algorithms based on search will win. [Dust](https://qlabs.sh/research/dust) is a big step
forward: it rivals backprop under SGD and is 1,000 to 10,000x more efficient than EGGROLL, the state-of-the-art
weight-space ES, but still trails backprop under Adam. These methods have also had a small fraction of the
research that went into backprop. We made Noise to close the gap: better estimators, and optimizers co-designed
for them instead of for backprop gradients.

The model, the data and the score are fixed: the 37.7M-parameter transformer of Dust, its 10M-token FineWeb
training set, and the cross-entropy on its validation set. What changes is the learning algorithm, which may
only run the model forward. The architecture is fixed for now, but we will relax this constraint later.

## Tracks

1. **Validation loss.** The lowest validation loss after one epoch on one 8xH100 node, with no time limit.
2. **Speedrun.** The shortest training time on one 8xH100 node to reach validation loss **4.2**.

## Rules

- **No backpropagation.** No autograd (reverse or forward mode) and no hand-written backward pass. Learning
  signal comes from evaluating the model forward. Entries are judged by the spirit of this rule.
- **Fixed model.** Final weights must load into [`utils/model.py`](utils/model.py) with a strict state-dict
  match. How training computes the same function, and how it initializes it, is free.
- **Fixed data.** Train only on `data/train.pt` (4,880 sequences of 2,048 predictions), for at most one
  epoch: each training sequence feeds at most one update. Batch size and order are free.
- **Score.** [`utils/eval.py`](utils/eval.py) on the final weights: mean next-token cross-entropy over the
  1,114,112 predictions of `data/val.pt`. No checkpoint selection on validation loss.
- **Timing** (speedrun track; reported on the other). From the start of the first update to the end of the
  last, excluding any validation. An untimed warmup on random tokens may compile kernels, as long as training
  then starts from a fresh state (the baselines rebuild everything from the seed). Runs use seed 0.

## Setup

Requires Linux, an NVIDIA GPU with bfloat16 support and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run utils/prepare_data.py
```

`utils/prepare_data.py` streams FineWeb `sample-10BT`, tokenizes it with the fixed BPE-4096 tokenizer in
[`tokenizer/`](tokenizer/tokenizer.json), and writes `data/train.pt` and `data/val.pt`, the `train_10M.pt`
and `val.pt` of the [Dust repo](https://github.com/qlabs-eng/dust). Both are checked against pinned hashes.

## Running the records

```bash
uv run torchrun --standalone --nproc_per_node=8 train/val/run.py         # track 1
uv run torchrun --standalone --nproc_per_node=8 train/val/anvil.py
uv run torchrun --standalone --nproc_per_node=8 train/speedrun/run.py    # track 2
uv run torchrun --standalone --nproc_per_node=8 train/val/eggroll.py
```

Each run writes `config.json`, `metrics.jsonl`, `final.pt` and `result.json` (validation loss and training time)
to `logs/`. `uv run utils/eval.py logs/val/final.pt` rescores a checkpoint.

## Records

### Track 1: validation loss

| # | Val loss | Description | Time | Script |
| - | - | - | - | - |
| 1 | 5.954 | EGGROLL w/ Adam, 128k antithetic pairs, lr cooldown | 5.33 h | [`train/val/eggroll.py`](train/val/eggroll.py) |
| 2 | 5.058 | Dust w/ SGD, 16k population | 1.32 h | [`dust.py`](https://github.com/qlabs-eng/dust/blob/main/dust.py) |
| 3 | 4.167 | Dust w/ Adam, 24k population, lr cooldown | 1.76 h | [`train/val/run.py`](train/val/run.py) |
| 4 | 4.088 | Dust w/ ANVIL II on the block matrices and Adam elsewhere, 24k population, lr cooldown | 1.78 h | [`train/val/anvil.py`](train/val/anvil.py) |

### Track 2: speedrun to 4.2

| # | Val loss | Description | Time | Script |
| - | - | - | - | - |
| 1 | 4.167 | Dust w/ Adam, 24k population, lr cooldown | 1.76 h | [`train/speedrun/run.py`](train/speedrun/run.py) |

## What backprop gets

The same model, data and epoch trained with backprop, scored the same way:

| Optimizer | Val loss |
| - | - |
| SGD | 5.007 |
| Adam w/ lr cooldown | 3.769 |

The goal is to beat backprop by a large margin.

## Submitting

Open a PR that adds the new single-file training script to its track's folder in [`train/`](train/), with its
log. A record must beat the current one in its track under the rules above; the training script is the record.

## Citation

```bibtex
@misc{noise_2026,
  title  = {Noise: Speedrun for pretraining without backpropagation},
  author = {G{\"u}lbahar, Serdar and Mandal, Bishwas and Vegesna, Akshay and Dahal, Samip},
  year   = {2026},
  url    = {https://github.com/qlabs-eng/noise}
}
```

**Core contributors:** Serdar Gülbahar and Bishwas Mandal.

## License

Released under the [MIT License](LICENSE).
