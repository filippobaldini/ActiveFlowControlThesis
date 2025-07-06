import csv
import os


def compute_reduction(baseline_recirc, reward):
    """
    Given a baseline recirculation area (baseline_recirc) and a reward,
    where reward = -recirculation_area (i.e., area = -reward),
    compute the percentage reduction:

    reduction = (baseline_area - current_area) / baseline_area * 100%
    """
    # Convert reward -> current recirc area
    current_area = reward

    reduction = (baseline_recirc - current_area) / baseline_recirc * 100.0
    return reduction


def analyze_rewards(
    run_name,
    input_filename="episode_rewards_normalized.csv",
    output_filename="perfromances.txt",
):
    """
    Reads 'csv_path', finds the best reward (i.e., max),
    finds the last reward, and computes the % reduction
    of recirculation area relative to 'baseline_recirc'.
    """
    # Construct input and output paths based on the run_name
    input_csv = os.path.join("runs", run_name, input_filename)
    output_path = os.path.join("runs", run_name, output_filename)
    rewards = []
    with open(input_csv, "r", newline="") as infile:
        reader = csv.reader(infile)
        header = next(reader)

        for row in reader:
            # e.g. row = ["12", "-2.7320513576547834"]
            episode_str, reward_str = row
            reward_val = float(reward_str)
            rewards.append(reward_val)

    if not rewards:
        print("No data found in CSV.")
        return

    # 1) Best reward (maximum, if higher reward => smaller recirc area)
    best_reward = -max(rewards)

    # 2) Last reward
    last_reward = -rewards[-1]

    baseline_recirc = -rewards[0]

    # 3) Compute percentage reductions
    reduction_best = compute_reduction(baseline_recirc, best_reward)
    reduction_last = compute_reduction(baseline_recirc, last_reward)

    # Print or return the results
    print(f"Best reward = {best_reward}")
    print(f"Last reward = {last_reward}")
    print(f"% reduction (best) = {reduction_best:.2f}%")
    print(f"% reduction (last) = {reduction_last:.2f}%")

    # Write to file
    with open(output_path, "w") as outfile:
        outfile.write("Recirculation Area Performance\n")
        outfile.write("--------------------------------\n")
        outfile.write(f"Baseline recirc area  : {-baseline_recirc}\n")
        outfile.write(f"Best reward           : {-best_reward:.5f}\n")
        outfile.write(f"Last reward           : {-last_reward:.5f}\n")
        outfile.write(f"% reduction (best)    : {reduction_best:.2f}%\n")
        outfile.write(f"% reduction (last)    : {reduction_last:.2f}%\n")


if __name__ == "__main__":

    analyze_rewards("BackwardFacingStepv0__train__no_freq__1__20250113_222935")
