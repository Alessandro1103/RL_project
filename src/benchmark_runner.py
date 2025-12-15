import os
import pathlib
import sys
import ray
import numpy
import torch

import muzero
import games.cartpole

def run_benchmarks():
    # --- GLOBAL CONFIGURATION ---
    GAME_NAME = "cartpole"
    NUM_SEEDS = 5  # Number of runs per experiment (32 for exact paper replication)
    DEFAULT_TRAINING_STEPS = 10000 # Standard duration for experiments without pre-training
    
    experiments = {
        "1_Baseline": {
            "reconstruction_loss_weight": 0,
            "consistency_loss_weight": 0,
            "self_supervised_steps": 0
        },
        "2_Reconstruction": {
            "reconstruction_loss_weight": 1,
            "consistency_loss_weight": 0,
            "self_supervised_steps": 0
        },
        "3_Consistency": {
            "reconstruction_loss_weight": 0,
            "consistency_loss_weight": 1,
            "self_supervised_steps": 0
        },
        "4_Hybrid": {
            "reconstruction_loss_weight": 1,
            "consistency_loss_weight": 1,
            "self_supervised_steps": 0
        },
        "5_Hybrid_Pretrained": {
            "reconstruction_loss_weight": 1,
            "consistency_loss_weight": 1,
            "self_supervised_steps": 5000,
            "training_steps": 15000 
        }
    }

    print(f"Starting benchmark for {GAME_NAME} with {NUM_SEEDS} seeds per experiment.")

    # Cycle on each experiment configuration
    for exp_name, exp_config in experiments.items():
        print(f"=== Start of experiment: {exp_name} ===")
        
        for seed in range(NUM_SEEDS):
            print(f"  > Running Seed {seed}/{NUM_SEEDS - 1}...")

            config = games.cartpole.MuZeroConfig()
            
            config.reconstruction_loss_weight = exp_config["reconstruction_loss_weight"]
            config.consistency_loss_weight = exp_config["consistency_loss_weight"]
            config.self_supervised_steps = exp_config["self_supervised_steps"]
            
            config.training_steps = exp_config.get("training_steps", DEFAULT_TRAINING_STEPS)
            
            config.seed = seed
            
            base_dir = pathlib.Path(__file__).parent.parent / "results"
            run_dir = base_dir / GAME_NAME / exp_name / f"seed_{seed}"
            config.results_path = run_dir
            
            config.results_path.mkdir(parents=True, exist_ok=True)

            try:
                mz = muzero.MuZero(GAME_NAME, config, split_resources_in=1)
                mz.train(log_in_tensorboard=True)
                
                mz.terminate_workers()
                del mz
                
            except Exception as e:
                print(f"!!! Errore durante seed {seed} di {exp_name}: {e}")
            
            ray.shutdown()
            
        print(f"=== Experiment completed: {exp_name} ===\n")

    print("All benchmarks have been completed.")

if __name__ == "__main__":
    run_benchmarks()