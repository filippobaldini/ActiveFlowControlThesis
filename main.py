import argparse
import subprocess
import sys
import os

def main():
    parser = argparse.ArgumentParser(
        description="Run training or inspection for RL-based flow control."
    )

    parser.add_argument(
        "--case",
        choices=["bfs", "cylinder"],
        required=True,
        help="Test case to run (bfs or cylinder)",
    )
    parser.add_argument(
        "--algo",
        choices=["ppo", "td3"],
        required=True,
        help="RL algorithm to use (ppo or td3)",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "test"],
        required=True,
        help="Run training or testing",
    )
    parser.add_argument(
        "--submode",
        choices=["default", "mu"],
        default="default",
        help="Inspection type: default (rollout) or mu (stepwise viscosity test)",
    )
    parser.add_argument(
        "--control",
        choices=["amplitude", "ampfreq", "2jets", "3jets"],
        required=True,
        help="Control mode: bfs → amplitude/ampfreq, cylinder → 2jets/3jets",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        help="Name of the run directory under 'runs/' (required for inspection)",
    )

    args = parser.parse_args()

    if args.mode == "test" and not args.run_name:
        parser.error("--run-name is required when mode is 'test'")

    if args.submode == "mu" and args.case != "bfs":
        print(f"❌ Parametric testing (--submode mu) is only implemented for 'bfs'.")
        sys.exit(1)

    if args.case == "bfs" and args.control not in ["amplitude", "ampfreq"]:
        print(f"❌ For bfs, --control must be 'amplitude' or 'ampfreq'")
        sys.exit(1)
    if args.case == "cylinder" and args.control not in ["2jets", "3jets"]:
        print(f"❌ For cylinder, --control must be '2jets' or '3jets'")
        sys.exit(1)

    # Define script mapping
    script_map = {
        ("bfs", "ppo", "train"): "bfs/train/train_ppo.py",
        ("bfs", "td3", "train"): "bfs/train/train_td3.py",
        ("cylinder", "ppo", "train"): "cylinder/train/train_ppo.py",
        ("cylinder", "td3", "train"): "cylinder/train/train_td3.py",
        ("bfs", "ppo", "default"): "bfs/inspection/inspect_ppo.py",
        ("bfs", "ppo", "mu"): "bfs/inspection/inspect_mu_ppo.py",
        ("bfs", "td3", "default"): "bfs/inspection/inspect_td3.py",
        ("bfs", "td3", "mu"): "bfs/inspection/inspect_mu_td3.py",
        ("cylinder", "ppo", "default"): "cylinder/inspection/inspect_ppo.py",
        ("cylinder", "td3", "default"): "cylinder/inspection/inspect_td3.py",
    }

    key = (args.case, args.algo, args.submode if args.mode == "test" else args.mode)
    script = script_map.get(key)

    if not script:
        print(
            f"❌ Unsupported combination: case={args.case}, algo={args.algo}, mode={args.mode}, submode={args.submode}"
        )
        sys.exit(1)

    if not os.path.exists(script):
        print(f"❌ Script '{script}' not found.")
        sys.exit(1)

    cmd = ["python3", script, "--control", args.control]
    if args.mode == "test":
        cmd += ["--run-name", args.run_name]

    print(f"\nLaunching {script}")
    print(
        f"\nRunning {args.mode} [{args.submode}] for {args.case.upper()} using {args.algo.upper()} with control: {args.control}"
    )
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
