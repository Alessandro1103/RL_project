import os
import glob
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LOG_DIRS = {
    'MuZero (Baseline)': 'src/results/cartpole/baseline',        
    'Reconstruction': 'src/results/cartpole/reconstruction',
    'Consistency': 'src/results/cartpole/consistency',     
    'Hybrid': 'src/results/cartpole/hybrid'                      
}

WINDOW_SIZE = 500 

def get_data(folder_path):
    files = glob.glob(os.path.join(folder_path, "**", "events.out.tfevents*"), recursive=True)
    if not files: return None, None
    event_file = sorted(files, key=os.path.getmtime, reverse=True)[0]

    try:
        ea = EventAccumulator(event_file)
        ea.Reload()
        
        reward_tag = "1.Total_reward/1.Total_reward"
        step_tag = "2.Workers/2.Training_steps"
        
        if reward_tag not in ea.Tags()['scalars'] or step_tag not in ea.Tags()['scalars']:
            return None, None

        rewards = ea.Scalars(reward_tag)      
        steps_log = ea.Scalars(step_tag)      

        # Mappa step -> reward
        step_map = {e.step: e.value for e in steps_log}
        
        x_vals, y_vals = [], []
        for r in rewards:
            if r.step in step_map:
                x_vals.append(step_map[r.step]) 
                y_vals.append(r.value)
                
        return np.array(x_vals), np.array(y_vals)
    except:
        return None, None

def print_stats():
    print(f"\n{'EXPERIMENT':<25} | {'MEAN':<10} | {'STD DEV':<10} | {'SAMPLES':<10}")
    print("-" * 65)

    for label, path in LOG_DIRS.items():
        if not os.path.exists(path):
            print(f"{label:<25} | {'MISSING':<10} | {'-':<10} | {'0':<10}")
            continue
            
        x, y = get_data(path)
        
        if x is not None and len(x) > 0:

            # Last 500 measurements
            last_step = np.max(x)
            threshold = last_step - WINDOW_SIZE
            mask = x >= threshold
            final_rewards = y[mask]
            
            if len(final_rewards) > 0:
                mean_val = np.mean(final_rewards)
                std_val = np.std(final_rewards)
                samples = len(final_rewards)
                
                print(f"{label:<25} | {mean_val:<10.2f} | {std_val:<10.2f} | {samples:<10}")
            else:
                print(f"{label:<25} | {'NO DATA':<10} | {'-':<10} | {'0':<10}")
        else:
            print(f"{label:<25} | {'EMPTY':<10} | {'-':<10} | {'0':<10}")

    print("-" * 65)
    print(f"Stats calculated on the final {WINDOW_SIZE} training steps.")

if __name__ == "__main__":
    print_stats()