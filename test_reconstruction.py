import torch
import numpy
from src.models import MuZeroNetwork
from src.games.cartpole import MuZeroConfig

def test_architecture():
    print("--- Inizio Test Architettura (Reconstruction + Consistency) ---")

    config = MuZeroConfig()
    config.network = "resnet"
    config.observation_shape = (3, 96, 96) 
    config.reconstruction_loss_weight = 1.0
    config.consistency_loss_weight = 1.0
    
    config.downsample = "resnet" 
    config.blocks = 1
    config.channels = 16
    config.reduced_channels_reward = 2
    config.reduced_channels_value = 2
    config.reduced_channels_policy = 2
    config.resnet_fc_reward_layers = [8]
    config.resnet_fc_value_layers = [8]
    config.resnet_fc_policy_layers = [8]

    print("\n1. Inizializzazione MuZeroNetwork (ResNet)...")
    try:
        model = MuZeroNetwork(config)
        device = torch.device("cpu")
        model.to(device)
        print("Modello ok")
    except Exception as e:
        print(f"Errore modello. {e}")
        return

    batch_size = 2
    num_unroll_steps = config.num_unroll_steps
    
    observation_batch = torch.randn(batch_size, 3, 96, 96).to(device)
    
    action_batch = torch.randint(0, len(config.action_space), (batch_size, num_unroll_steps + 1)).long().to(device)
    action_batch = action_batch.unsqueeze(-1)
    
    target_observations = torch.randn(batch_size, num_unroll_steps + 1, 3, 96, 96).to(device)    
    gradient_scale_batch = torch.ones(batch_size, num_unroll_steps + 1).to(device)

    try:
        B, T, C, H, W = target_observations.shape
        target_obs_flat = target_observations.view(B * T, C, H, W)
        
        with torch.no_grad():
            target_states_flat = model.representation(target_obs_flat)
            
        target_latent_states = target_states_flat.view(B, T, -1)
        print(f"Target Latenti generati. Shape: {target_latent_states.shape}")
    except Exception as e:
        print(f"Errore consistency: {e}")
        return

    print("\n2. Test Loop...")
    try:
        output = model.initial_inference(observation_batch)
        if len(output) != 5:
            print(f"   -> Errore: initial_inference restituisce {len(output)} valori invece di 5")
            return
        
        value, reward, policy_logits, hidden_state, reconstruction = output 
        
        reconstruction_loss = 0
        consistency_loss = 0
        predictions = [(value, reward, policy_logits, reconstruction)]
        
        for i in range(1, action_batch.shape[1]):
            output = model.recurrent_inference(hidden_state, action_batch[:, i])
            value, reward, policy_logits, hidden_state, reconstruction = output
            predictions.append((value, reward, policy_logits, reconstruction))
            
            curr_rec_loss = torch.nn.functional.mse_loss(reconstruction, target_observations[:, i])
            curr_rec_loss.register_hook(lambda grad: grad / gradient_scale_batch[:, i])
            reconstruction_loss += curr_rec_loss
            
            curr_state_flat = hidden_state.view(batch_size, -1)
            target_state_flat = target_latent_states[:, i]
            
            curr_cons_loss = torch.nn.functional.mse_loss(curr_state_flat, target_state_flat)
            curr_cons_loss.register_hook(lambda grad: grad / gradient_scale_batch[:, i])
            consistency_loss += curr_cons_loss
            
        print(f"Loop ok, reconstraction loss: {reconstruction_loss.item()}, {consistency_loss.item()}")

    except Exception as e:
        print(f"Errore nel loop: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n--- Test ok ---")

if __name__ == "__main__":
    test_architecture()