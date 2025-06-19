#!/usr/bin/env python3
"""
run_bertopic_force_sweep.py

Force-retrain BERTopic models for multiple collections over a range of topic counts.

Usage:
    python run_bertopic_force_sweep.py [--collections wiki 20ng wsj] [--start 10] [--stop 200] [--step 10] [--train-script bertopic_train.py] [--output-dir Results/BERTOPIC]
"""
import subprocess
import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Force-retrain BERTopic models over a sweep of topic counts."
    )
    parser.add_argument(
        '--collections',
        nargs='+',
        default=['wiki', '20ng', 'wsj'],
        help="List of collections to sweep (default: wiki 20ng wsj)"
    )
    parser.add_argument(
        '--start',
        type=int,
        default=10,
        help="Starting number of topics (inclusive, default: 10)"
    )
    parser.add_argument(
        '--stop',
        type=int,
        default=200,
        help="Stopping number of topics (inclusive, default: 200)"
    )
    parser.add_argument(
        '--step',
        type=int,
        default=10,
        help="Step size for topic counts (default: 10)"
    )
    parser.add_argument(
        '--train-script',
        default='TopicModels/bertopic_train.py',
        help="Path to the bertopic training script (default: bertopic_train.py)"
    )
    parser.add_argument(
        '--output-dir',
        default='Results/BERTOPIC',
        help="Directory to save models and matrices (default: Results/BERTOPIC)"
    )
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    for col in args.collections:
        for k in range(args.start, args.stop + 1, args.step):
            print(f"[FORCE TRAIN] Collection={col}, num_topics={k}")
            cmd = [
                sys.executable,  # current python interpreter
                args.train_script,
                '--dataset', col,
                '--num_topics', str(k),
                '--output_dir', args.output_dir,
                '--force'
            ]
            # call the training script
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error training {col} with k={k}: {e}", file=sys.stderr)


if __name__ == '__main__':
    main()
