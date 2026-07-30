"""Shared PPO policy and challenge-facing inference adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from agent import Action, BomberlandAgent, normalize_state, _entity_position


ACTIONS = tuple(action.value for action in Action)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BOARD_SIZE = 11
CHANNELS = 7
OBS_SIZE = CHANNELS * BOARD_SIZE * BOARD_SIZE + 2


def encode_observation(state: dict[str, Any], agent_id: str) -> np.ndarray:
    """Encode walls, boxes, bombs, fire, self and opponents into fixed channels."""
    snapshot = normalize_state(state)
    grid = np.zeros((CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    def mark(channel: int, position: tuple[int, int], value: float = 1.0) -> None:
        x, y = position
        if 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE:
            grid[channel, y, x] = value

    for position in snapshot.walls:
        mark(0, position)
    for position in snapshot.boxes:
        mark(1, position)
    for bomb in snapshot.bombs:
        fuse = float(bomb.get("fuse", bomb.get("timer", 1)))
        mark(2, _entity_position(bomb), max(0.0, min(1.0, fuse / 4.0)))
    for position in snapshot.explosions:
        mark(3, position)

    me = None
    for player in snapshot.players:
        if str(player.get("id")) == str(agent_id):
            me = player
            mark(4, _entity_position(player))
        elif player.get("alive", player.get("isAlive", True)):
            mark(5, _entity_position(player))

    # Channel 6 predicts imminent blast cells using the safety expert.
    expert = BomberlandAgent()
    for position in expert._danger_cells(snapshot):
        mark(6, position)

    bombs = float(me.get("bombs", 0)) / 4.0 if me else 0.0
    radius = float(me.get("radius", 2)) / 6.0 if me else 0.0
    return np.concatenate((grid.reshape(-1), np.array([bombs, radius], np.float32)))


class ActorCritic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(OBS_SIZE, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
        )
        self.actor = nn.Linear(128, len(ACTIONS))
        self.critic = nn.Linear(128, 1)

    def forward(self, observation: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        features = self.backbone(observation)
        return Categorical(logits=self.actor(features)), self.critic(features).squeeze(-1)


class PPOAgent:
    """Drop-in agent: state + id -> one challenge action."""

    def __init__(self, checkpoint: str | Path = "bomberland_ppo.pt", stochastic: bool = False):
        self.model = ActorCritic().to(DEVICE)
        data = torch.load(checkpoint, map_location=DEVICE, weights_only=True)
        self.model.load_state_dict(data["model_state"])
        self.model.eval()
        self.stochastic = stochastic

    @torch.no_grad()
    def next_action(self, state: dict[str, Any], agent_id: str) -> str:
        # Hard safety shield: PPO cannot gamble while standing in a blast lane.
        snapshot = normalize_state(state)
        me = next(
            (player for player in snapshot.players if str(player.get("id")) == str(agent_id)),
            None,
        )
        expert = BomberlandAgent()
        expert_action = expert.next_action(state, agent_id)
        if me is not None and _entity_position(me) in expert._danger_cells(snapshot):
            return expert_action

        observation = torch.as_tensor(
            encode_observation(state, agent_id), dtype=torch.float32, device=DEVICE
        )
        distribution, _ = self.model(observation)
        legal = self._legal_actions(state, agent_id, expert_action)
        masked_logits = distribution.logits.clone()
        for index, action_name in enumerate(ACTIONS):
            if action_name not in legal:
                masked_logits[index] = -1e9
        masked_distribution = Categorical(logits=masked_logits)
        action = (
            masked_distribution.sample()
            if self.stochastic
            else masked_distribution.probs.argmax()
        )
        return ACTIONS[action.item()]

    @staticmethod
    def _legal_actions(
        state: dict[str, Any], agent_id: str, expert_action: str
    ) -> set[str]:
        snapshot = normalize_state(state)
        me = next(
            (player for player in snapshot.players if str(player.get("id")) == str(agent_id)),
            None,
        )
        if me is None:
            return {Action.WAIT.value}
        x, y = _entity_position(me)
        occupied = set(snapshot.walls | snapshot.boxes)
        occupied.update(_entity_position(bomb) for bomb in snapshot.bombs)
        legal = {Action.WAIT.value}
        moves = {
            Action.MOVE_UP.value: (0, -1),
            Action.MOVE_DOWN.value: (0, 1),
            Action.MOVE_LEFT.value: (-1, 0),
            Action.MOVE_RIGHT.value: (1, 0),
        }
        for action_name, (dx, dy) in moves.items():
            target = (x + dx, y + dy)
            if (
                0 <= target[0] < snapshot.width
                and 0 <= target[1] < snapshot.height
                and target not in occupied
            ):
                legal.add(action_name)
        bombs = int(me.get("bombs", me.get("bombCount", 0)))
        if (
            bombs > 0
            and expert_action == Action.PLACE_BOMB.value
            and not any(_entity_position(bomb) == (x, y) for bomb in snapshot.bombs)
        ):
            legal.add(Action.PLACE_BOMB.value)
        return legal
