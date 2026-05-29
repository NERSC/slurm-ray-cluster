# slurm-ray-cluster

Slurm batch scripts for launching multi-node [Ray](https://docs.ray.io/) clusters on NERSC Perlmutter GPU nodes with `ray symmetric-run`.

## Symmetric Run

This repository uses `ray symmetric-run` because it matches Slurm's single `srun` launch model while preserving Ray's head/worker runtime model. All allocated nodes run the same launch command, Ray starts the head and worker processes, and the entrypoint runs only on the head node after the requested number of nodes has joined.

On NERSC, `ray symmetric-run` requires Ray >= 2.53.0 to work correctly. After the entrypoint finishes, Ray stops the cluster automatically.

## Scripts

| Script | Runtime | Container notes |
|---|---|---|
| `submit-ray-symmetric.sbatch_shifter` | Shifter | Uses `nersc/pytorch:26.01.01` with `gpu,nccl-cu13-plugin` |
| `submit-ray-symmetric.sbatch_podmanhpc` | podman-hpc | Uses `nersc/pytorch:26.01.01` with `--gpu --nccl-cu13` |

## Usage

Copy the script that matches your environment, adjust the Slurm options and entrypoint command, then submit:

```bash
sbatch submit-ray-symmetric.sbatch_shifter
# or
sbatch submit-ray-symmetric.sbatch_podmanhpc
```

The default entrypoint is the MNIST Ray Tune example:

```bash
python examples/mnist_pytorch_trainable.py --cuda
```

To run a different workload, replace the command after the `--` separator in the batch script. Example workloads are in `examples/`.

## Notes

- The scripts request one Slurm task per node and pass the allocated CPUs and GPUs to Ray with `--num-cpus` and `--num-gpus`. Keep those values consistent with the `#SBATCH` resource requests when adapting a script.
- The example scripts assume Perlmutter GPU nodes with `-C gpu`, 4 GPUs per node, and a CUDA 13 NERSC PyTorch image.
- If you change the container CUDA version, update the NCCL option to match it. For CUDA 13, use Shifter `nccl-cu13-plugin` or podman-hpc `--nccl-cu13`; for CUDA 12, use the CUDA 12 NCCL option documented by NERSC.
- Shifter images need to be pulled before use; see the Shifter docs for more information. podman-hpc can pull images on the fly.

## References

- [Ray: Deploying on Slurm](https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html)
- [NERSC: Hyperparameter optimization / RayTune](https://docs.nersc.gov/machinelearning/hpo/)
- [NERSC: Shifter](https://docs.nersc.gov/development/containers/shifter/how-to-use/)
- [NERSC: podman-hpc](https://docs.nersc.gov/development/containers/podman-hpc/overview/)
