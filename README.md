# Virtual rodent training in Julia

An implementation of [Proximal Policy Optimization](https://en.wikipedia.org/wiki/Proximal_policy_optimization) to train a MuJoCo model of a rat body to imitate the movements of a real rat, using the same biomechanical model and data as in [Aldarondo et al. (2024)](https://www.nature.com/articles/s41586-024-07633-4).

## Installation
Clone the repository, pull motion capture data from the [dvc](https://dvc.org/) repository (reach out to request access, or use your own data), and instantiate the julia environment:
```
git clone https://github.com/emiwar/julia-virtual-rodent.git
dvc pull
cd julia-virtual rodent
julia
]instantiate .
```
To log the progress of training, the code uses [Weights and Biases](https://wandb.ai/) which requires registering an account.

## Getting started
The project has a tentative config system using TOML files. Call `src/main.jl` with an appropriate config file.
```
$: julia --project=. --threads=16 src/main.jl configs/imitation_sota.toml\
-rollout.n_envs 512 -rollout.n_epochs 20000 -rollout.use_mpi false
```

### Old launch system (OBSOLETE)
The files in the `experiments` folder are meant to be entry points for different training scenarios. For example,
```
julia --project=. --threads=16 experiments/variational_imitation_local.jl
```
will start training an imitation network with a variational (stochastic) bottleneck, using only local multithreading (no MPI) with 16 threads.

## Dependencies
The biomechanical model is implemented in the physics engine [MuJoCo](https://mujoco.org), the neural networks in [Flux.jl](https://fluxml.ai/), and multi-machine training using [MPI.jl](https://juliaparallel.org/MPI.jl).

## Aim
In [previous work](https://www.nature.com/articles/s41586-024-07633-4), the models were trained used multi-objective [V-MPO](https://arxiv.org/abs/1909.12238). This was run using 4,000 actors for simulating the physics of the model and collecting samples, and a TPUv2 chip for training the networks.

The aim of this repository is to (1) increase the computational efficiency so that fewer CPU cores are required to train imitation in a reasonable timeframe, (2) simplify using more complex network architectures efficiently and (3) make it easy to create and modify RL environment logic. The aim is not to implement a fully-fledged RL library.

## Alternative approaches (or, why julia and not python?)
Training the rodent to do imitation well requires billions of samples, from several years of simulated time. To collect this in any reasonable time, some form of parallellism is required.

### Multiprocessing in Python
Python does not support _multithreading_ due to the Global Interpreter Lock, so traditional RL libraries in python must rely on _multiprocessing_ to simulate multiple physics instances in parallell. However, RL requires a closed loop between the actor and the physics environment, which means that the physics simulation and the actor neural network must communicate on every timestep. Inter-process communication has substatinal overhead (sometimes tens of ms), so the most common approach is for each worker process to have its own copy of the actor network. This approach has a few drawbacks:
1) It increases complexity of the software, since the actor networks regularly have to be serialized and synchronized to all workers.
2) The actor networks have to run on CPUs, and the weights must be copied for each worker. This is fine for small networks typically used in continuous control, but nontheless limits the complexity and size of the networks that can be efficiently trained.
3) It makes on-policy RL more challenging/expensive since the synchronization step is expensive.

In julia, on the other hand, multithreading is straight-forward, so that each worker can efficiently simulate 32 or 64 parallell physics instances. Additionally, MPI communication is built for low-latency/low-overhead communicaton between machines. Combined, this allows for a much simpler setup with only a single copy of the actor network, which can be placed on a GPU. Samples are collected synchroniously, meaning there is no need for cachers or for serializing and transferring any networks.

### Multithreading in C/C++
One workaround is to reimplement the bindings between MuJoCo (written in C) and python, to allow for C-based multithreading. This is the approach taken by [EnvPool](https://github.com/sail-sg/envpool). The downside is that is requires all environment logic (including reward terms, translations between allocentric and egocentric coordinate systems, motion capture representations, etc) to be written in C or C++, and that any change to the environment requires recompilation of the python package.

### JAX
The most promising alternative approach is to use the JAX-based implementation of MuJoCo, [MJX](https://mujoco.readthedocs.io/en/stable/mjx.html). This is very appealing because it allows physics to be run on GPU(s) alongside the networks, resulting in potentially very high performance. One limitation in JAX is however that collisions get very computationally costly, so MJX suffers in scences with lots of potential collision pairs. This is not a huge problem for pure imitation learning where contacts between the feet and floor might be sufficient, and self-collisions can be omitted. However, down the line in tasks where the virtual rodent could interact with equipment or objects in its environment it might matter more. Nevertheless, once MJX evolves past these limitations, it may be beneficial to fully switch to MJX.

