"""Train Agent 1 with PPO against fixed defensive and aggressive opponents."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from agent import Action, BomberlandAgent, _entity_position, normalize_state
from local_engine import LocalBomberland
from ppo_agent import ACTIONS, DEVICE, ActorCritic, encode_observation
from strategies import AggressiveAgent, DefensiveAgent


def action_mask(state: dict, agent_id: str) -> torch.Tensor:
    snapshot = normalize_state(state)
    me = next(player for player in snapshot.players if player["id"] == agent_id)
    x, y = _entity_position(me)
    blocked = set(snapshot.walls | snapshot.boxes)
    blocked.update(_entity_position(bomb) for bomb in snapshot.bombs)
    allowed = {Action.WAIT.value}
    moves = {
        Action.MOVE_UP.value: (0, -1),
        Action.MOVE_DOWN.value: (0, 1),
        Action.MOVE_LEFT.value: (-1, 0),
        Action.MOVE_RIGHT.value: (1, 0),
    }
    for name, (dx, dy) in moves.items():
        target = (x + dx, y + dy)
        if (
            0 <= target[0] < snapshot.width
            and 0 <= target[1] < snapshot.height
            and target not in blocked
        ):
            allowed.add(name)
    radius = int(me.get("radius", 2))
    expert = BomberlandAgent()
    future_blast = expert._blast_from((x, y), radius, snapshot.walls, snapshot.boxes)
    escape = expert._first_step_to(
        (x, y),
        lambda cell: cell not in future_blast,
        snapshot,
        blocked - {(x, y)},
        set(snapshot.explosions),
        max_depth=radius + 5,
    )
    if (
        int(me.get("bombs", 0)) > 0
        and escape is not None
        and not any(_entity_position(bomb) == (x, y) for bomb in snapshot.bombs)
    ):
        allowed.add(Action.PLACE_BOMB.value)
    return torch.tensor(
        [name in allowed for name in ACTIONS], dtype=torch.bool, device=DEVICE
    )


def gae(rewards, values, dones, gamma=.99, lam=.95):
    advantages = np.zeros(len(rewards), np.float32)
    carry = next_value = 0.0
    for index in reversed(range(len(rewards))):
        mask = 1.0 - float(dones[index])
        delta = rewards[index] + gamma * next_value * mask - values[index]
        carry = delta + gamma * lam * mask * carry
        advantages[index] = carry
        next_value = values[index]
    return advantages, advantages + np.asarray(values, np.float32)


def update(model, optimizer, buffer, epochs=6, batch_size=256):
    advantages, returns = gae(
        buffer["rewards"], buffer["values"], buffer["dones"]
    )
    observations = torch.as_tensor(
        np.asarray(buffer["obs"]), dtype=torch.float32, device=DEVICE
    )
    actions = torch.as_tensor(buffer["actions"], dtype=torch.long, device=DEVICE)
    old_logp = torch.as_tensor(buffer["logp"], dtype=torch.float32, device=DEVICE)
    advantages = torch.as_tensor(advantages, dtype=torch.float32, device=DEVICE)
    returns = torch.as_tensor(returns, dtype=torch.float32, device=DEVICE)
    bomb_targets = torch.as_tensor(
        buffer["bomb_targets"], dtype=torch.bool, device=DEVICE
    )
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    final_loss = 0.0
    for _ in range(epochs):
        for batch in torch.randperm(len(actions), device=DEVICE).split(batch_size):
            distribution, values = model(observations[batch])
            logp = distribution.log_prob(actions[batch])
            ratio = (logp - old_logp[batch]).exp()
            policy_loss = -torch.min(
                ratio * advantages[batch],
                torch.clamp(ratio, .8, 1.2) * advantages[batch],
            ).mean()
            value_loss = nn.functional.mse_loss(values, returns[batch])
            useful_bombs = bomb_targets[batch]
            bomb_lesson = (
                -distribution.log_prob(
                    torch.full_like(actions[batch], ACTIONS.index(Action.PLACE_BOMB.value))
                )[useful_bombs].mean()
                if useful_bombs.any()
                else torch.tensor(0.0, device=DEVICE)
            )
            loss = (
                policy_loss
                + .5 * value_loss
                - .02 * distribution.entropy().mean()
                + .18 * bomb_lesson
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), .5)
            optimizer.step()
            final_loss = loss.item()
    return final_loss


def train(matches: int, output: Path, max_steps: int = 300, resume: bool = False):
    model = ActorCritic().to(DEVICE)
    if resume and output.exists():
        model.load_state_dict(
            torch.load(output, map_location=DEVICE, weights_only=True)["model_state"]
        )
        print(f"Resumed {output}")
    optimizer = torch.optim.Adam(model.parameters(), lr=2.5e-4)
    expert = BomberlandAgent()
    recent_rewards, recent_wins = [], []
    last_loss = 0.0

    for match in range(1, matches + 1):
        engine = LocalBomberland(seed=10_000 + match, player_count=3)
        opponents = {"agent_2": DefensiveAgent(), "agent_3": AggressiveAgent()}
        buffer = {
            key: []
            for key in (
                "obs",
                "actions",
                "logp",
                "values",
                "rewards",
                "dones",
                "bomb_targets",
            )
        }
        total_reward = 0.0

        while not engine.finished(max_steps) and engine.players["agent_1"].alive:
            state = engine.state()
            player = engine.players["agent_1"]
            old_score = player.score
            old_pos = player.pos
            old_danger = old_pos in expert._danger_cells(normalize_state(state))
            old_snapshot = normalize_state(state)
            bomb_targets = expert._blast_from(
                old_pos,
                int(player.radius),
                old_snapshot.walls,
                old_snapshot.boxes,
            ).intersection(
                old_snapshot.boxes
                | frozenset(
                    p.pos
                    for key, p in engine.players.items()
                    if key != "agent_1" and p.alive
                )
            )
            enemy_distance = min(
                abs(old_pos[0] - p.pos[0]) + abs(old_pos[1] - p.pos[1])
                for key, p in engine.players.items() if key != "agent_1" and p.alive
            )

            observation = encode_observation(state, "agent_1")
            tensor = torch.as_tensor(observation, dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                distribution, value = model(tensor)
                mask = action_mask(state, "agent_1")
                masked = Categorical(logits=distribution.logits.masked_fill(~mask, -1e9))
                action = masked.sample()

            opponent_actions = {
                key: agent.next_action(state, key)
                for key, agent in opponents.items()
                if engine.players[key].alive
            }
            engine.step({"agent_1": ACTIONS[action.item()], **opponent_actions})
            player = engine.players["agent_1"]
            new_state = engine.state()
            new_danger = player.pos in expert._danger_cells(normalize_state(new_state))
            alive_enemies = [
                p for key, p in engine.players.items() if key != "agent_1" and p.alive
            ]
            new_distance = min(
                (abs(player.pos[0] - p.pos[0]) + abs(player.pos[1] - p.pos[1]) for p in alive_enemies),
                default=0,
            )

            reward = .02  # survival
            reward += (player.score - old_score) * .10  # boxes + kills
            reward += .20 if old_danger and not new_danger else 0
            reward -= .35 if not old_danger and new_danger else 0
            reward += .03 if new_distance < enemy_distance else 0
            if ACTIONS[action.item()] == Action.PLACE_BOMB.value and bomb_targets:
                reward += .30
            reward -= .02 if player.pos == old_pos and ACTIONS[action.item()] != Action.PLACE_BOMB.value else 0
            if not player.alive:
                reward -= 10.0
            if player.alive and not alive_enemies:
                reward += 15.0
            done = not player.alive or engine.finished(max_steps)

            for key, item in (
                ("obs", observation),
                ("actions", action.item()),
                ("logp", masked.log_prob(action).item()),
                ("values", value.item()),
                ("rewards", reward),
                ("dones", done),
                ("bomb_targets", bool(bomb_targets)),
            ):
                buffer[key].append(item)
            total_reward += reward

        last_loss = update(model, optimizer, buffer)
        won = engine.players["agent_1"].alive and sum(
            p.alive for p in engine.players.values()
        ) == 1
        recent_rewards = (recent_rewards + [total_reward])[-50:]
        recent_wins = (recent_wins + [won])[-50:]
        report_every = max(1, matches // 25)
        if match == 1 or match % report_every == 0:
            print(
                f"Match {match:>5}/{matches} | reward {np.mean(recent_rewards):>7.2f} | "
                f"win {np.mean(recent_wins):>6.1%} | loss {last_loss:>7.3f}"
            )

    torch.save(
        {"model_state": model.state_dict(), "actions": ACTIONS, "matches": matches},
        output,
    )
    print(f"\nSaved trained Agent 1 to {output} ({DEVICE})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=Path("bomberland_ppo.pt"))
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(args.matches, args.output, args.max_steps, args.resume)
