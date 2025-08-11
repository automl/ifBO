# Run HPO experiments (Section 5.2)

**Important**: Ensure benchmarks are downloaded in `../pfns_hpo/data/` directory.

Customize SLURM configs, number of seeds, benchmarks, or algorithms in `{taskset,lcbench,pd1}.sh`.

### Taskset
```bash
sbatch --partition {SLURM_PARTITION} taskset.sh
```

### LCBench (TODO)
### PD1 (TODO)

# Generate figures (regret and rank plots)
Adapt `plot.sh` with your specifications, then run:
```bash
bash plot.sh
```