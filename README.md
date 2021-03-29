# Reinforcement Learning with Augmented Data (RAD): ProcGen

Official codebase for [Reinforcement Learning with Augmented Data](https://mishalaskin.github.io/rad) on [Procgen Benchmark](https://github.com/openai/procgen). This codebase was originally forked from [Procgen](https://github.com/openai/train-procgen). Official codebases for DM control and OpenAI Gym are available at [RAD: DM control](https://github.com/MishaLaskin/rad) [RAD: OpenAI Gym](https://github.com/pokaxpoka/rad_openaigym).

## BibTex

```
@article{laskin2020reinforcement,
  title={Reinforcement learning with augmented data},
  author={Laskin, Michael and Lee, Kimin and Stooke, Adam and Pinto, Lerrel and Abbeel, Pieter and Srinivas, Aravind},
  journal={arXiv preprint arXiv:2004.14990},
  year={2020}
}
```


## Install

You can get miniconda from https://docs.conda.io/en/latest/miniconda.html if you don't have it, or install the dependencies from [`environment.yml`](environment.yml) manually.

```
git clone https://github.com/openai/train-procgen.git
conda env update --name train-procgen --file train-procgen/environment.yml
conda activate train-procgen
```
got to Procgen_Envs

```
pip install -e .
```
comback to Procgen
```
pip install https://github.com/openai/baselines/archive/9ee399f5b20cd70ac0a871927a6cf043b478193f.zip
pip install -e train-procgen
pip uninstall tensorflow
conda install -n train-procgen tensorflow-gpu=1.15 cudatoolkit=10.0
pip install torch matplotlib scikit-image
```

change `NeurIPS2020_Procgen_Envs/procgen/src/game.h` for random crop

## Try it out

Pixel PPO on the environment StarPilot:

```
./scripts/train_normal.sh starpilot
```

PPO + RAD (crop) on the environment StarPilot:
```
./scripts/train.sh starpilot crop
```

PPO + RAD (flip) on the environment StarPilot:
```
./scripts/train.sh starpilot flip
```

PPO + RAD (color_jitter) on the environment StarPilot:
```
./scripts/train.sh starpilot color_jitter
```

PPO + RAD (rotate) on the environment StarPilot:
```
./scripts/train.sh starpilot rotate
```

PPO + RAD (cutout_color) on the environment StarPilot:
```
./scripts/train.sh starpilot cutout_color
```

PPO + RAD (cutout) on the environment StarPilot:
```
./scripts/train.sh starpilot cutout
```

PPO + RAD (gray) on the environment StarPilot:
```
./scripts/train.sh starpilot gray
```

PPO + RAD (random conv) on the environment StarPilot:
```
./scripts/train_random.sh starpilot
```
