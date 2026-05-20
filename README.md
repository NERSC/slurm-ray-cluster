# slurm-ray-cluster

Slurm batch scripts for launching multi-node [Ray](https://docs.ray.io/) clusters on NERSC Perlmutter GPU nodes.

## Approaches

### Traditional head/worker (`submit-ray-cluster.sbatch`)

Starts the Ray head node and workers as separate `srun` steps using `start-head.sh` and `start-worker.sh`. Suitable for older Ray versions.

### Symmetric run (`submit-ray-symmetric.sbatch_*`)

Uses `ray symmetric-run` (requires Ray ≥ 2.53.0) to bring up the full cluster in a single `srun` call — all nodes join before the entrypoint runs on the head. Two container variants:

| Variant | Runtime |
|---|---|
| `_shifter` | Shifter (`nersc/pytorch:26.01.01`) |
| `_podmanhpc` | podman-hpc (`nersc/pytorch:26.01.01`) |

## Usage

Copy the script that matches your environment, adjust `--nodes`, account (`-A`), and the entrypoint command, then submit:

```bash
sbatch submit-ray-symmetric.sbatch_shifter
# or
sbatch submit-ray-symmetric.sbatch_podmanhpc
```

Example workloads are in `examples/`.
