""" Submits to slurm using a hydra API.

"""
import argparse
import itertools
import os
from pathlib import Path


def parse_argument_string(args):
    def get_argument_settings(argument):
        """Transforms 'a=1,2,3' to ('a=1', 'a=2', 'a=3')."""
        name, values = argument.split("=")
        if values.startswith("range("):
            range_params = values[len("range(") : -1]
            range_params = (int(param) for param in range_params.split(", "))
            return [f"{name}={value}" for value in range(*range_params)]
        else:
            return [f"{name}={value}" for value in values.split(",")]

    def get_all_argument_settings(arguments):
        """
        Transforms ['a=1,2,3', 'b=1'] to [('a=1', 'b=1'), ('a=2', 'b=1'), ('a=3', 'b=1')]
        """
        return itertools.product(
            *(get_argument_settings(argument) for argument in arguments)
        )

    def get_all_argument_strings(argument_settings):
        """Transforms [('a=1', 'b=1')] to ('a=1 b=1',)"""
        return (" ".join(argument_setting) for argument_setting in argument_settings)

    argument_settings = get_all_argument_settings(args.arguments)
    argument_strings = list(get_all_argument_strings(argument_settings))
    argument_string = "\n".join(argument_strings)
    argument_final_string = f"ARGS=(\n{argument_string}\n)"

    return argument_final_string, len(argument_strings)


def is_gpu_partition(args):
    return True if "gpu" in args.partition else False


def get_gpu_srun_command(cmd, nworkers=1):
    script = list()
    script.append(f"for i in $(seq 1 {nworkers}); do")
    script.append(f"    {cmd} &")  # the & is important
    script.append("done")
    script.append("wait")  # important to wait for all jobs to finish
    return script


def construct_script(args, cluster_oe_dir):
    argument_string, num_tasks = parse_argument_string(args)
    cmd = (
        f"python -m pfns_hpo.run experiment_group={args.experiment_group} "
        f"${{ARGS[@]:{len(args.arguments)}*$SLURM_ARRAY_TASK_ID:{len(args.arguments)}}}"
    )

    script = list()
    script.append("#!/bin/bash")
    script.append(f"#SBATCH --time {args.time}")
    script.append(f"#SBATCH --job-name {args.job_name}")
    script.append(f"#SBATCH --partition {args.partition}")
    script.append(f"#SBATCH --array 0-{num_tasks - 1}%{args.max_tasks}")
    script.append(f"#SBATCH --error {cluster_oe_dir}/%N_%A_%x_%a.oe")
    script.append(f"#SBATCH --output {cluster_oe_dir}/%N_%A_%x_%a.oe")

    if args.exclude:
        script.append(f"#SBATCH --exclude {args.exclude}")

    if args.gpus and not is_gpu_partition(args):
        raise ValueError("Cannot request GPUs on non-gpu partition!")

    # TODO: remove limitation where this check restricts each job to be on one node
    min_cpus_per_gpu = 2
    c = min(20, args.gpus * args.n_worker * min_cpus_per_gpu)  # 20 is the maximum number of CPUs per node
    if args.n_worker == 1:
        # Explicitly state the cpu and gpus and it's total memory
        if args.gpus > 0:
            script.append(f"#SBATCH --gres=gpu:{args.gpus}")
            script.append(f"#SBATCH -c {min_cpus_per_gpu}")
        else:
            script.append(f"#SBATCH -c 1")
        script.append(f"#SBATCH --mem {args.memory}")
    else:
        # Each worker will need this much memory as they each
        # load the benchmark individually
        if args.gpus > 0:
            script.append(f"#SBATCH --gres=gpu:{args.n_worker * args.gpus}")
            script.append(f"#SBATCH -c {c}")  # requests the maximal CPU for all workers
        else:
            script.append(f"#SBATCH -c {args.n_worker}")
        script.append(f"#SBATCH --mem-per-cpu {args.memory}")

        # Prepend the cmd with srun to enable it to run multiple times
        cpus_per_task = c // args.n_worker if args.gpus > 0 else 1
        gpus_per_task = args.gpus
        cmd = f"srun --ntasks 1 --cpus-per-task {cpus_per_task} --gres=gpu:{gpus_per_task} --exclusive {cmd}"  # --gpu-bind=closest

        if args.n_worker > 1 and args.gpus > 0:
            # adjusts script to launch all the srun jobs sequentially as background processes
            new_cmd = get_gpu_srun_command(cmd, args.n_worker)

    # TODO: verify 1 worker 1 GPU run and runs where each worker uses multiple GPUs
    script.append("")
    script.append(argument_string)
    script.append("")
    if args.n_worker > 1 and args.gpus > 0:
        script.extend(new_cmd)
    else:
        script.append(cmd)

    return "\n".join(script) + "\n"  # type: ignore[assignment]


def construct_moab_script(args, cluster_oe_dir):
    argument_string, num_tasks = parse_argument_string(args)
    cmd = (
        f"python3.10 -m pfns_hpo.run experiment_group={args.experiment_group} "
        f"${{ARGS[@]:{len(args.arguments)}*$MOAB_JOBARRAYINDEX:{len(args.arguments)}}}"
    )

    time = args.time.split("-")[1].split(":")[0] + ":00:00"

    script = list()
    script.append("#!/bin/bash")
    script.append(f"#MSUB -l walltime={time}")
    script.append(f"#MSUB -N {args.job_name}")
    script.append(f"#MSUB -t 0-{num_tasks - 1}")
    script.append(f"#MSUB -j oe")
    script.append(f"#MSUB -o {cluster_oe_dir}/%I_%J.oe")
    script.append("#MSUB -m n")  # No email notifications


    memory = args.memory / 1000
    ppn = int(memory // 6)
    memory = str(min(memory, 6)) + "gb"

    script.append(f"#MSUB -l pmem={memory}:ppn={ppn}")

    script.append("")
    script.append("cd ./lcpfn-hpo-0")
    script.append("")
    script.append("source setup_script.sh")
    script.append("")
    script.append("cd pfns_hpo")
    script.append("")
    script.append(argument_string)
    script.append("")

    if args.n_worker > 1 and args.gpus > 0:
        script.extend(new_cmd)
    else:
        script.append(cmd)

    return "\n".join(script) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_group", default="test")
    parser.add_argument("--server", default="kislurm")
    parser.add_argument("--max_tasks", default=100, type=int)
    parser.add_argument("--time", default="0-23:59")
    parser.add_argument("--job_name", default="test")
    parser.add_argument("--memory", default=0, type=int)
    parser.add_argument("--n_worker", default=1, type=int)
    parser.add_argument("--partition", required=True)
    parser.add_argument("--arguments", nargs="+")
    parser.add_argument(
        "--exclude", default=None, type=str, help="example: kisexe24,kisexe34"
    )
    parser.add_argument(
        "--gpus", 
        default=0, 
        type=int, 
        help="if GPUs to be used by worker (0 if CPU only and assumes single node)"
    )
    args = parser.parse_args()

    experiment_group_dir = Path("results", args.experiment_group)
    cluster_oe_dir = Path(experiment_group_dir, ".cluster_oe")
    scripts_dir = Path(experiment_group_dir, ".submit")

    experiment_group_dir.mkdir(parents=True, exist_ok=True)
    cluster_oe_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir.mkdir(parents=True, exist_ok=True)

    num_scripts = len(list(scripts_dir.glob("*.sh")))
    script_path = Path(scripts_dir, f"{num_scripts}.sh")

    if args.server == "kislurm":
        submission_commmand = f"sbatch {script_path}"
        script = construct_script(args, cluster_oe_dir)
    elif args.server == "nemo":
        submission_commmand = f"msub {script_path}"
        script = construct_moab_script(args, cluster_oe_dir)
    else:
        raise ValueError("Unknown server type")

    print(f"Running {submission_commmand} with script:\n\n{script}")
    if input("Ok? [Y|n] -- ").lower() in {"y", ""}:
        script_path.write_text(script, encoding="utf-8")  # type: ignore[arg-type]
        os.system(submission_commmand)
    else:
        print("Not submitting.")