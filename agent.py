"""Bomberland challenge agent.

The engine owns the game. This module only maps one state update to one action.
Integrate `BomberlandAgent.next_action(state, agent_id)` with the participant kit.
"""

from __future__ import annotations

import json
import sys
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any


class Action(str, Enum):
    MOVE_UP = "MOVE_UP"
    MOVE_DOWN = "MOVE_DOWN"
    MOVE_LEFT = "MOVE_LEFT"
    MOVE_RIGHT = "MOVE_RIGHT"
    PLACE_BOMB = "PLACE_BOMB"
    WAIT = "WAIT"


Position = tuple[int, int]
MOVES: tuple[tuple[Action, Position], ...] = (
    (Action.MOVE_UP, (0, -1)),
    (Action.MOVE_DOWN, (0, 1)),
    (Action.MOVE_LEFT, (-1, 0)),
    (Action.MOVE_RIGHT, (1, 0)),
)


@dataclass(frozen=True)
class Snapshot:
    width: int
    height: int
    walls: frozenset[Position]
    boxes: frozenset[Position]
    bombs: tuple[dict[str, Any], ...]
    explosions: frozenset[Position]
    players: tuple[dict[str, Any], ...]


def _position(value: Any) -> Position:
    """Accept {x,y}, [x,y], or objects with x/y attributes."""
    if isinstance(value, dict):
        return int(value.get("x", value.get("col", 0))), int(
            value.get("y", value.get("row", 0))
        )
    if isinstance(value, (list, tuple)):
        return int(value[0]), int(value[1])
    return int(value.x), int(value.y)


def _entity_position(entity: dict[str, Any]) -> Position:
    return _position(entity.get("position", entity.get("pos", entity)))


def normalize_state(state: dict[str, Any]) -> Snapshot:
    """Isolate participant-kit field names from the decision algorithm."""
    field = state.get("field", state.get("map", {}))
    width = int(field.get("width", state.get("width", 15)))
    height = int(field.get("height", state.get("height", 15)))
    walls: set[Position] = set()
    boxes: set[Position] = set()

    grid = field.get("tiles", field.get("field", state.get("tiles", [])))
    if grid and isinstance(grid[0], list):
        for y, row in enumerate(grid):
            for x, tile in enumerate(row):
                name = str(tile).upper()
                if name in {"WALL", "BLOCK", "SOLID"}:
                    walls.add((x, y))
                elif name in {"BOX", "CRATE", "BREAKABLE"}:
                    boxes.add((x, y))
    elif grid:
        for index, tile in enumerate(grid):
            name = str(tile).upper()
            position = (index % width, index // width)
            if name in {"WALL", "BLOCK", "SOLID"}:
                walls.add(position)
            elif name in {"BOX", "CRATE", "BREAKABLE"}:
                boxes.add(position)

    for item in state.get("walls", []):
        walls.add(_position(item))
    for item in state.get("boxes", state.get("crates", [])):
        boxes.add(_position(item))

    return Snapshot(
        width=width,
        height=height,
        walls=frozenset(walls),
        boxes=frozenset(boxes),
        bombs=tuple(state.get("bombs", [])),
        explosions=frozenset(
            _position(item) for item in state.get("explosions", [])
        ),
        players=tuple(state.get("agents", state.get("players", []))),
    )


class BomberlandAgent:
    """Fast hybrid agent: safety search first, tactical pressure second."""

    def next_action(self, state: dict[str, Any], agent_id: str | int) -> str:
        snapshot = normalize_state(state)
        me = self._find_player(snapshot.players, agent_id)
        if me is None:
            return Action.WAIT.value

        origin = _entity_position(me)
        radius = int(me.get("radius", me.get("blastRadius", 2)))
        bombs_available = int(
            me.get("bombs", me.get("bombCount", me.get("bombsAvailable", 1)))
        )
        blocked = set(snapshot.walls | snapshot.boxes)
        blocked.update(_entity_position(bomb) for bomb in snapshot.bombs)
        danger = self._danger_cells(snapshot)

        # Survival always has priority over score.
        if origin in danger:
            escape = self._first_step_to(
                origin,
                lambda cell: cell not in danger,
                snapshot,
                blocked,
                snapshot.explosions,
            )
            return escape.value if escape else Action.WAIT.value

        enemies = [
            _entity_position(player)
            for player in snapshot.players
            if str(player.get("id")) != str(agent_id)
            and player.get("alive", player.get("isAlive", True))
        ]

        # Place a bomb only when it creates value and an escape route exists.
        attack_value = self._attack_value(origin, radius, enemies, snapshot.boxes)
        if bombs_available > 0 and attack_value > 0:
            future_danger = danger | self._blast_from(
                origin, radius, snapshot.walls, snapshot.boxes
            )
            escape = self._first_step_to(
                origin,
                lambda cell: cell not in future_danger,
                snapshot,
                blocked - {origin},
                danger,
                max_depth=radius + 4,
            )
            if escape is not None:
                return Action.PLACE_BOMB.value

        targets = self._targets(state, enemies, snapshot.boxes)
        bombing_sites = self._safe_bombing_sites(snapshot, radius, blocked)
        if bombing_sites:
            targets = bombing_sites
        step = self._first_step_to(
            origin,
            lambda cell: cell in targets,
            snapshot,
            blocked,
            danger,
        )
        if step is not None:
            return step.value

        safe_moves = [
            action
            for action, (dx, dy) in MOVES
            if self._valid(
                (origin[0] + dx, origin[1] + dy), snapshot, blocked, danger
            )
        ]
        return (safe_moves[0] if safe_moves else Action.WAIT).value

    @staticmethod
    def _find_player(
        players: tuple[dict[str, Any], ...], agent_id: str | int
    ) -> dict[str, Any] | None:
        return next(
            (player for player in players if str(player.get("id")) == str(agent_id)),
            None,
        )

    def _danger_cells(self, snapshot: Snapshot) -> set[Position]:
        danger = set(snapshot.explosions)
        for bomb in snapshot.bombs:
            timer = int(bomb.get("timer", bomb.get("fuse", 1)))
            if timer <= 3:
                radius = int(bomb.get("radius", bomb.get("blastRadius", 2)))
                danger.update(
                    self._blast_from(
                        _entity_position(bomb),
                        radius,
                        snapshot.walls,
                        snapshot.boxes,
                    )
                )
        return danger

    @staticmethod
    def _blast_from(
        origin: Position,
        radius: int,
        walls: frozenset[Position],
        boxes: frozenset[Position],
    ) -> set[Position]:
        cells = {origin}
        for _, (dx, dy) in MOVES:
            for distance in range(1, radius + 1):
                cell = (origin[0] + dx * distance, origin[1] + dy * distance)
                if cell in walls:
                    break
                cells.add(cell)
                if cell in boxes:
                    break
        return cells

    def _first_step_to(
        self,
        start: Position,
        is_goal,
        snapshot: Snapshot,
        blocked: set[Position],
        danger: set[Position],
        max_depth: int = 30,
    ) -> Action | None:
        queue = deque([(start, None, 0)])
        visited = {start}
        while queue:
            cell, first_action, depth = queue.popleft()
            if cell != start and is_goal(cell):
                return first_action
            if depth >= max_depth:
                continue
            for action, (dx, dy) in MOVES:
                next_cell = (cell[0] + dx, cell[1] + dy)
                if next_cell in visited or not self._valid(
                    next_cell, snapshot, blocked, danger
                ):
                    continue
                visited.add(next_cell)
                queue.append((next_cell, first_action or action, depth + 1))
        return None

    @staticmethod
    def _valid(
        cell: Position,
        snapshot: Snapshot,
        blocked: set[Position],
        danger: set[Position],
    ) -> bool:
        return (
            0 <= cell[0] < snapshot.width
            and 0 <= cell[1] < snapshot.height
            and cell not in blocked
            and cell not in danger
        )

    @staticmethod
    def _attack_value(
        origin: Position,
        radius: int,
        enemies: list[Position],
        boxes: frozenset[Position],
    ) -> int:
        value = 0
        for target in enemies:
            aligned = target[0] == origin[0] or target[1] == origin[1]
            distance = abs(target[0] - origin[0]) + abs(target[1] - origin[1])
            if aligned and distance <= radius:
                value += 5
        value += sum(
            1
            for box in boxes
            if (
                (box[0] == origin[0] or box[1] == origin[1])
                and abs(box[0] - origin[0]) + abs(box[1] - origin[1]) <= radius
            )
        )
        return value

    @staticmethod
    def _targets(
        state: dict[str, Any],
        enemies: list[Position],
        boxes: frozenset[Position],
    ) -> set[Position]:
        powerups = {
            _entity_position(item)
            for item in state.get("powerups", state.get("items", []))
        }
        if powerups:
            return powerups
        # Stand next to a box so the next decision can place a useful bomb.
        box_frontiers = {
            (box[0] + dx, box[1] + dy)
            for box in boxes
            for _, (dx, dy) in MOVES
            if (box[0] + dx, box[1] + dy) not in boxes
        }
        if box_frontiers:
            return box_frontiers
        return set(enemies)

    def _safe_bombing_sites(
        self, snapshot: Snapshot, radius: int, blocked: set[Position]
    ) -> set[Position]:
        """Find useful bomb positions that are not one-way suicide pockets."""
        sites: set[Position] = set()
        for y in range(snapshot.height):
            for x in range(snapshot.width):
                cell = (x, y)
                if cell in blocked:
                    continue
                blast = self._blast_from(
                    cell, radius, snapshot.walls, snapshot.boxes
                )
                if not blast.intersection(snapshot.boxes):
                    continue
                exits = 0
                for _, (dx, dy) in MOVES:
                    neighbor = (x + dx, y + dy)
                    if (
                        0 <= neighbor[0] < snapshot.width
                        and 0 <= neighbor[1] < snapshot.height
                        and neighbor not in blocked
                    ):
                        exits += 1
                if exits >= 2:
                    sites.add(cell)
        return sites


def decide_action(state: dict[str, Any], agent_id: str | int) -> str:
    """Simple function adapter for participant kits that do not use classes."""
    return BomberlandAgent().next_action(state, agent_id)


def main() -> None:
    """JSON-lines adapter: stdin state → stdout action."""
    agent = BomberlandAgent()
    for line in sys.stdin:
        if not line.strip():
            continue
        message = json.loads(line)
        state = message.get("state", message)
        agent_id = message.get("agentId", message.get("agent_id", 0))
        print(agent.next_action(state, agent_id), flush=True)


if __name__ == "__main__":
    main()
