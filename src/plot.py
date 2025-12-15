import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LOG_DIRS = {
    'MuZero (Baseline)': 'results/cartpole/1_Baseline',        
    'Reconstruction': 'results/cartpole/2_Reconstruction',
    'Consistency': 'results/cartpole/3_Consistency',     
    'Hybrid': 'results/cartpole/4_Hybrid',
    'Hybrid+Pre': 'results/cartpole/5_Hybrid_Pretrained',                  
}

DISPLAY_LABELS = {
    'MuZero (Baseline)': 'MuZero',
    'Reconstruction': r'$l^g$',           # l^g
    'Consistency': r'$l^c$',              # l^c
    'Hybrid': r'$l^g + l^c$',             # l^g + l^c
    'Hybrid+Pre': r'$l^g + l^c$ (pre)'    # l^g + l^c pre
}

COLORS = {
    'MuZero (Baseline)': 'black',
    'Reconstruction': 'blue',
    'Consistency': 'green',
    'Hybrid': 'cyan',
    'Hybrid+Pre': 'red'
}

WINDOW_SIZE = 500 
PLOT_OUTPUT_DIR = os.path.join("plots", "cartpole")

def get_data(folder_path, skip_steps=0, rescale=False):
    files = glob.glob(os.path.join(folder_path, "**", "events.out.tfevents*"), recursive=True)
    if not files: return None, None, None

    all_x = []
    all_y = []

    for event_file in files:
        try:
            ea = EventAccumulator(event_file)
            ea.Reload()
            
            reward_tag = "1.Total_reward/1.Total_reward"
            step_tag = "2.Workers/2.Training_steps"
            
            tags = ea.Tags()['scalars']
            if reward_tag not in tags:
                continue

            rewards = ea.Scalars(reward_tag)      
            
            step_map = {}
            if step_tag in tags:
                steps_log = ea.Scalars(step_tag)
                step_map = {e.step: e.value for e in steps_log}
            
            x_vals, y_vals = [], []
            for r in rewards:
                training_step = r.step
                if step_tag in tags and r.step in step_map:
                    training_step = step_map[r.step]
                
                if training_step >= skip_steps:
                    val_step = training_step - skip_steps if rescale else training_step
                    x_vals.append(val_step)
                    y_vals.append(r.value)
            
            if x_vals:
                all_x.append(np.array(x_vals))
                all_y.append(np.array(y_vals))
                
        except Exception as e:
            print(f"Error reading {event_file}: {e}")
            continue

    if not all_x: return None, None, None

    max_common_step = 10000 
    common_x = np.linspace(0, max_common_step, num=200) 

    interpolated_ys = []
    for x, y in zip(all_x, all_y):
        iy = np.interp(common_x, x, y)
        interpolated_ys.append(iy)
    
    stacked_y = np.vstack(interpolated_ys)
    mean_y = np.mean(stacked_y, axis=0)
    std_y = np.std(stacked_y, axis=0)

    return common_x, mean_y, std_y

def create_plot(experiment_list, title, filename):

    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    found_data = False
    for key in experiment_list:
        path = LOG_DIRS.get(key)
        if not path or not os.path.exists(path):
            print(f"Warning: Path not found for {key}")
            continue
            
        skip_steps = 5000 if key == 'Hybrid+Pre' else 0
        rescale = True if key == 'Hybrid+Pre' else False
        
        x, mean_y, std_y = get_data(path, skip_steps=skip_steps, rescale=rescale)
        
        if x is not None and len(x) > 0:
            found_data = True
            mask = x <= 10000
            x_filt = x[mask]
            y_filt = mean_y[mask]
            
            display_label = DISPLAY_LABELS.get(key, key)
            color = COLORS.get(key, 'gray')

            # PLOT DELLA MEDIA
            plt.plot(x_filt, y_filt, 
                     color=color, 
                     label=display_label, 
                     linewidth=1)
        else:
            print(f"No valid data for {key}")

    if found_data:
        plt.xlabel('Training Steps', fontsize=12)
        plt.ylabel('Total Reward (Mean)', fontsize=12)
        plt.title("Cartpole-v1", fontsize=12)
        
        plt.legend(loc='upper left', fontsize=12, frameon=True)
        
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 10000)
        
        plt.tight_layout()
        
        full_path = os.path.join(PLOT_OUTPUT_DIR, filename)
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        print(f"Grafico salvato in: {full_path}")
        plt.close()
    else:
        print(f"Impossibile creare grafico {filename}: nessun dato trovato.")

def print_stats():
    print(f"\n{'EXPERIMENT':<25} | {'MEAN':<10} | {'STD DEV':<10} | {'LABEL':<15}")
    print("-" * 75)

    for key, path in LOG_DIRS.items():
        if not os.path.exists(path):
            continue
        
        skip_steps = 5000 if key == 'Hybrid+Pre' else 0
        rescale = True if key == 'Hybrid+Pre' else False
        
        x, mean_y, std_y = get_data(path, skip_steps=skip_steps, rescale=rescale)
        
        if x is not None and len(x) > 0:
            last_step = np.max(x)
            threshold = last_step - WINDOW_SIZE
            mask = x >= threshold
            
            final_rewards = mean_y[mask]
            if len(final_rewards) > 0:
                final_mean = np.mean(final_rewards)
                final_std = np.mean(std_y[mask]) 
                label_tex = DISPLAY_LABELS.get(key, key)
                
                print(f"{key:<25} | {final_mean:<10.2f} | {final_std:<10.2f} | {label_tex:<15}")

    print("-" * 75)
    print(f"Stats calculated on the final {WINDOW_SIZE} training steps.\n")

if __name__ == "__main__":
    print_stats()
    
    group1 = ['MuZero (Baseline)', 'Reconstruction', 'Hybrid']
    create_plot(group1, 
                "Comparison: Baseline vs Reconstruction vs Hybrid", 
                "comparison_baseline_reconstruction_hybrid.png")
    
    group2 = ['MuZero (Baseline)', 'Consistency', 'Hybrid']
    create_plot(group2,
                "Comparison: Baseline vs Consistency vs Hybrid", 
                "comparison_baseline_concistency_hybrid.png")
    
    group3 = ['MuZero (Baseline)', 'Hybrid', 'Hybrid+Pre']
    create_plot(group3, 
                "Comparison: Baseline vs Hybrid vs Pre-trained", 
                "comparison_baseline_hybrid_pre.png")

    group_all = [
        'MuZero (Baseline)', 
        'Reconstruction', 
        'Consistency', 
        'Hybrid', 
        'Hybrid+Pre'
    ]
    create_plot(group_all, 
                "Comparison: All Experiments", 
                "comparison_all.png")