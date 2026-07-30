# Bomberland AI Agents

Three-agent Bomberland simulation for the GDGoC AI Challenge:

- Agent 1: shared PPO policy with a safety shield.
- Agent 2: defensive BFS strategy.
- Agent 3: aggressive A* strategy.

## Demo

![Bomberland AI agents — longest active match](assets/demo-longest.gif)

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
