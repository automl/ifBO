# List available receipes
@list:
  just --list
# Run local experiment:
@run algorithm="random_search" benchmark="mfh3_bad" experiment_group="debug" seed="200" n_workers="1" :
  HYDRA_FULL_ERROR=1 python -m pfns_hpo.run \
    algorithm={{algorithm}} \
    benchmark={{benchmark}} \
    experiment_group={{experiment_group}} \
    seed={{seed}} \
    n_workers={{n_workers}} \
    hydra/job_logging=full

# Download surrogate
@download_pfn version="0.0.1" path="./src/PFNS4HPO/final_models/":
  time python -m pfns_hpo.download \
    --version {{version}} \
    --path {{path}}

# Submit job
@submit algorithms benchmarks seeds="range(1)" experiment_group="test" job_name="default" partition="TODO:default-partition-name" max_tasks="1000" time="0-23:59" memory="0" n_worker="1" gpus="0":
  python -m pfns_hpo.submit \
    --experiment_group {{experiment_group}} \
    --max_tasks {{max_tasks}} \
    --time {{time}} \
    --job_name {{job_name}} \
    --partition {{partition}} \
    --memory {{memory}} \
    --n_worker {{n_worker}} \
    --gpus {{gpus}} \
    --arguments algorithm={{algorithms}} benchmark={{benchmarks}} n_workers={{n_worker}} seed="{{seeds}}" hydra/job_logging=full \

# Plot job
@plot experiment_group benchmarks algorithms filename ext="pdf" base_path=justfile_directory() :
  python -m pfns_hpo.plot \
    --experiment_group {{experiment_group}} \
    --benchmarks {{benchmarks}} \
    --algorithm {{algorithms}} \
    --filename {{filename}} \
    --base_path {{base_path}} \
    --ext {{ext}} \
    --x_range 0 20 \
    --plot_default \
    --plot_optimum \
    --parallel

# Table job
@table experiment_group benchmarks algorithms filename budget base_path=justfile_directory() :
  python -m pfns_hpo.final_table \
    --experiment_group {{experiment_group}} \
    --benchmark {{benchmarks}} \
    --algorithm {{algorithms}} \
    --filename {{filename}} \
    --base_path {{base_path}} \
    --budget {{budget}}

# List all available benchmarks
@benchmarks:
    ls -1 ./src/pfns_hpo/pfns_hpo/configs/benchmark | grep ".yaml" | sed -e "s/\.yaml$//"

# Generate all available benchmarks
@generate_benchmarks:
    python "./src/pfns_hpo/pfns_hpo/configs/benchmark/generate.py"

# List all available algorithms
@algorithms:
    ls -1 ./src/pfns_hpo/pfns_hpo/configs/algorithm | grep ".yaml" | sed -e "s/\.yaml$//"

@download:
    python -m ..mf_prior_bench.mfpbench.download --data-dir "/work/dlclarge1/mallik-pfns4hpo/"
