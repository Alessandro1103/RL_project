import numpy as np
from src.games.cartpole import Game

def test_game_mechanics():
    print("1. Initialization...")
    try:
        game_instance = Game(seed=42)
        print("   -> OK: gym loaded correctly")
    except Exception as e:
        print(f"   -> Error: loading gym {e}")
        return

    game_instance.reset()

    print("\n2. Test of the game (Step random)...")
    try:
        done = False
        steps = 0
        while not done and steps < 10:

            legal_actions = game_instance.legal_actions()
            action = np.random.choice(legal_actions)
            
            _, reward, done = game_instance.step(action)
            steps += 1
            print(f"   Step {steps}: Action {action} -> Reward {reward} -> Done {done}")
        
        print(f"\n   -> OK: The loop works. {steps} steps executed.")
        
    except Exception as e:
        print(f"   -> Error: {e}")
    finally:
        game_instance.close()

if __name__ == "__main__":
    test_game_mechanics()