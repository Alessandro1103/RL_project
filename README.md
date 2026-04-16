# MuZero with Self-Supervised Auxiliary Losses

Questo progetto estende l'implementazione di **MuZero** ([Werner Duvaud](https://github.com/werner-duvaud/muzero-general)) introducendo due loss ausiliarie di tipo self-supervised per migliorare la qualità delle rappresentazioni latenti apprese dal modello:

- **Reconstruction loss** ($l^g$): forza la rete a ricostruire l'osservazione originale dallo stato latente, tramite una rete di decodifica aggiuntiva.
- **Consistency loss** ($l^c$): impone coerenza tra gli stati latenti predetti dalla funzione di dinamica e quelli codificati direttamente dalla funzione di rappresentazione.

L'obiettivo è verificare se queste loss ausiliarie accelerano l'apprendimento o migliorano le performance finali di MuZero su ambienti a bassa dimensionalità (CartPole-v1, LunarLander-v2, MountainCar-v0).

---

## Struttura del progetto

```
muzero-ssl/
├── src/
│   ├── muzero.py                  # Classe principale MuZero
│   ├── models.py                  # Architetture di rete (ResNet, FullyConnected + Reconstruction)
│   ├── trainer.py                 # Loop di training con reconstruction e consistency loss
│   ├── self_play.py               # Self-play e MCTS
│   ├── replay_buffer.py           # Replay buffer con prioritized experience replay
│   ├── shared_storage.py          # Gestione stato condiviso tra worker (Ray)
│   ├── diagnose_model.py          # Strumenti di diagnostica e visualizzazione MCTS
│   ├── plot.py                    # Generazione grafici dai log TensorBoard
│   ├── benchmark_cartpole.py      # Script benchmark per CartPole-v1
│   ├── benchmark_lunarlander.py   # Script benchmark per LunarLander-v2
│   └── games/
│       ├── abstract_game.py       # Interfaccia base per i giochi
│       ├── cartpole.py            # Configurazione CartPole-v1
│       ├── lunarlander.py         # Configurazione LunarLander-v2
│       └── mountaincar.py         # Configurazione MountainCar-v0
├── results/                       # Output del training (checkpoint, log TensorBoard)
├── plots/                         # Grafici generati da plot.py
└── requirements.txt
```

---

## Installazione

**Requisiti:** Python 3.10+, CUDA opzionale.

```bash
git clone https://github.com/<your-username>/muzero-ssl.git
cd muzero-ssl
pip install -r requirements.txt
```

Per LunarLander è necessario il backend Box2D:

```bash
pip install "gymnasium[box2d]"
```

---

## Utilizzo

### Training interattivo

```bash
cd src
python muzero.py
```

Verrà mostrato un menu per selezionare il gioco e la modalità (train, test, ecc.).

È anche possibile avviare il training direttamente da riga di comando:

```bash
python muzero.py cartpole
```

### Benchmark automatico

Per riprodurre gli esperimenti descritti nel progetto, eseguire i benchmark predefiniti:

```bash
# CartPole-v1
python src/benchmark_cartpole.py

# LunarLander-v2
python src/benchmark_lunarlander.py
```

Ogni benchmark esegue i seguenti 5 esperimenti con più seed:

| ID | Configurazione | $l^g$ | $l^c$ | Pre-training |
|----|----------------|:-----:|:-----:|:------------:|
| 1  | Baseline (MuZero standard) | ✗ | ✗ | ✗ |
| 2  | Reconstruction | ✓ | ✗ | ✗ |
| 3  | Consistency | ✗ | ✓ | ✗ |
| 4  | Hybrid | ✓ | ✓ | ✗ |
| 5  | Hybrid + Pre-training | ✓ | ✓ | ✓ |

I risultati vengono salvati in `results/<game>/<experiment>/seed_<n>/`.

### Generazione grafici

```bash
cd src
python plot.py
```

I grafici vengono salvati in `plots/cartpole/`. Lo script produce anche statistiche di reward medio e deviazione standard per ciascun esperimento nella finestra finale di training.

---

## Configurazione

Ogni gioco ha una classe `MuZeroConfig` nel rispettivo file in `games/`. I parametri principali legati alle estensioni self-supervised sono:

| Parametro | Descrizione | Default |
|-----------|-------------|---------|
| `reconstruction_loss_weight` | Peso della reconstruction loss ($l^g$) | `0` |
| `consistency_loss_weight` | Peso della consistency loss ($l^c$) | `0` |
| `self_supervised_steps` | Passi di pre-training self-supervised prima del RL | `0` |

Per sovrascrivere la configurazione a runtime:

```python
from muzero import MuZero
import games.cartpole

config = games.cartpole.MuZeroConfig()
config.reconstruction_loss_weight = 1
config.consistency_loss_weight = 1
config.self_supervised_steps = 5000

mz = MuZero("cartpole", config)
mz.train()
```

---

## Monitoraggio con TensorBoard

```bash
tensorboard --logdir results/
```

Le metriche tracciate includono reward totale, training steps, e le singole componenti della loss (value, policy, reward, reconstruction, consistency).

---

## Dettagli tecnici

### Architettura

La rete fully connected è stata estesa con una **rete di ricostruzione** (`ReconstructionNetwork`) che, a partire dallo stato latente prodotto dalla funzione di rappresentazione o dalla funzione di dinamica, decodifica l'osservazione originale. La loss di ricostruzione è calcolata come MSE tra la ricostruzione e l'osservazione target.

La **consistency loss** è calcolata come MSE tra lo stato latente prodotto dalla funzione di dinamica al passo $t$ e quello prodotto dalla funzione di rappresentazione applicata direttamente all'osservazione reale al passo $t$, entrambi detached dal grafo computazionale del target.

### Funzione di loss complessiva

Durante il training standard:

$$\mathcal{L} = \mathcal{L}_{value} + \mathcal{L}_{reward} + \mathcal{L}_{policy} + \lambda_g \cdot l^g + \lambda_c \cdot l^c$$

Durante il pre-training self-supervised (`self_supervised_steps > 0`), vengono usate **solo** $l^g$ e $l^c$, senza segnale di reward/valore proveniente dall'ambiente.

---

## Ambienti supportati

| Ambiente | Spazio osservazione | Azioni | Training steps |
|----------|---------------------|--------|----------------|
| CartPole-v1 | `(1, 1, 4)` | 2 (discreto) | 10 000 |
| LunarLander-v2 | `(1, 1, 8)` | 4 (discreto) | 30 000 |
| MountainCar-v0 | `(1, 1, 2)` | 3 (discreto) | 20 000 |

---

## Dipendenze principali

- `torch` — framework di deep learning
- `ray` — parallelizzazione self-play e training
- `gymnasium` — ambienti RL
- `tensorboard` — logging e visualizzazione
- `matplotlib` / `numpy` — analisi e plotting dei risultati
- `nevergrad` — ottimizzazione iperparametri

---

## Crediti

Questo progetto si basa sull'implementazione open-source di MuZero di [Werner Duvaud](https://github.com/werner-duvaud/muzero-general), estesa con meccanismi di apprendimento self-supervised ispirati alla letteratura su world models e rappresentazioni latenti strutturate.
