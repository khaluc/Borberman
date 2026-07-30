# Bomberland AI Agents

Three-agent Bomberland simulation for the GDGoC AI Challenge:

- Agent 1: shared PPO policy with a safety shield.
- Agent 2: defensive BFS strategy.
- Agent 3: aggressive A* strategy.

## Demo

![Bomberland AI agents — longest active match](assets/demo-longest.gif)

## PPO benchmark

Agent 1 was evaluated against the Defensive and Aggressive agents over 30
matches, with a maximum of 200 steps per match.

| Metric | Result |
|---|---:|
| Win rate | **36.7%** (11/30) |
| Survival rate | **40.0%** (12/30) |
| Average score | **24.7** |
| Bomb actions | **178** |

The PPO reward encourages bomb avoidance, box destruction, opponent pressure,
well-timed bomb placement, and surviving as the final agent.

## Run the web arena

```bash
pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:8001>.

## Train PPO

```bash
python train_ppo.py --matches 2000 --output bomberland_ppo.pt
```

## Run in the terminal

```bash
python local_engine.py --steps 200 --show-map
```
