# RL4design

This git repo contains the code used
for [Reinforcement learning for freeform robot design](https://arxiv.org/abs/2310.05670).

![summary.png](images%2Fsummary.png)

| Generation process               | Result                           |
|----------------------------------|----------------------------------|
| ![robot.gif](images%2Frobot.gif) | ![robot.png](images%2Frobot.png) |

## Bibtex

```Bibtex
@article{li2023reinforcement,
  title={Reinforcement learning for freeform robot design},
  author={Muhan Li and David Matthews and Sam Kriegman},
  journal={arXiv preprint arXiv:2310.05670},
  year={2023}
}
```

## Installation

First make sure that you have installed [voxcraft-viz](https://github.com/voxcraft/voxcraft-viz), this program is
required
if you want to visualize simulated robots in voxcraft environment using the navigator function, otherwise you may skip
this step since it doesn't affect training or other visualization functions.

And make sure that you have installed [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
or [miniconda](https://docs.conda.io/projects/miniconda/en/latest/),
then update `CONDA_PATH` in `install.sh`

Finally, just run `install.sh`, or copy commands to your terminal and execute them.

## Start training

To customize configurations for training, go to `config.py` under each experiment directory, then
start training with following commands:

```bash
export PYTHONPATH=`pwd`

# To optimize robots for moving further in the voxcraft environment 
python main.py -r experiments/vec_patch_voxcraft/optimize_rl.py

# To optimize robots for achieving various shape requirements (eg: bigger volume)
python main.py -r experiments/vec_patch_shape/optimize_rl.py
```

Before running the experiment, you will be prompted to add a comment to the experiment
using your preferred editor, which will appear later in the navigator:

```
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
Options:
0) Draw metrics for multiple trials
1) Draw metrics for single trial
Choice:1
Trials
0) /home/mlw0504/RL4design_results/CustomPPO_2023-10-12_18-22-02/CustomPPO_VoxcraftSingleRewardVectorizedPatchEnvironment_26445_00000_0_2023-10-12_18-22-02
    comment: <The comment added by you>
    reward: 6.665
```

A snapshot of the code directory will be saved along with experiment data.
eg: `/home/<user name>/RL4design_results/CustomPPO_2023-10-12_18-22-02/CustomPPO_VoxcraftSingleRewardVectorizedPatchEnvironment_26445_00000_0_2023-10-12_18-22-02/code`

## Visualization

To run visualizations for robot metrics, rewards, draw the process of robot generation, etc, you may use the
following commands. Navigator is a versatile built-in program for such functionalities.

```bash
export PYTHONPATH=`pwd`

python main.py -n <you may optionally specify path to saved results>
```

Example command line interface:

```
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
Options:
0) Draw metrics for multiple trials
1) Draw metrics for single trial
Choice:0
Options:
0) Draw aggregated reward curve for multiple trials
1) Draw separate reward curves for each trial
2) Draw aggregated robot metric curves for multiple trials
3) Draw volume task result (For reproducing figures in paper)
4) Draw voxcraft task result (For reproducing figures in paper)
Choice:
```
