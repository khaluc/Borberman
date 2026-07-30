"""Hand-crafted opponents with clearly different Bomberland strategies."""

from __future__ import annotations

import heapq
from typing import Any

from agent import (
    Action,
    BomberlandAgent,
    MOVES,
    _entity_position,
    normalize_state,
)


Position = tuple[int, int]


class StrategyBase(BomberlandAgent):
    def context(self, state: dict[str, Any], agent_id: str | int):
        snapshot = normalize_state(state)
        me = self._find_player(snapshot.players, agent_id)
        if me is None:
            return snapshot, None, [], set(), set()
        enemies = [
            _entity_position(player)
            for player in snapshot.players
            if str(player.get("id")) != str(agent_id)
            and player.get("alive", player.get("isAlive", True))
        ]
        blocked = set(snapshot.walls | snapshot.boxes)
        blocked.update(_entity_position(bomb) for bomb in snapshot.bombs)
        return snapshot, me, enemies, blocked, self._danger_cells(snapshot)

    def can_escape_after_bomb(
        self, origin: Position, radius: int, snapshot, blocked: set[Position]
    ) -> bool:
        future_blast = self._blast_from(
            origin, radius, snapshot.walls, snapshot.boxes
        )
        # The fuse allows crossing a future blast lane before detonation.
        step = self._first_step_to(
            origin,
            lambda cell: cell not in future_blast,
            snapshot,
            blocked - {origin},
            set(snapshot.explosions),
            max_depth=radius + 5,
        )
        return step is not None

    def path_to(self, origin, targets, snapshot, blocked, danger):
        if not targets:
            return None
        return self._first_step_to(
            origin,
            lambda cell: cell in targets,
            snapshot,
            blocked,
            danger,
        )

    @staticmethod
    def box_frontiers(boxes: frozenset[Position]) -> set[Position]:
        return {
            (box[0] + dx, box[1] + dy)
            for box in boxes
            for _, (dx, dy) in MOVES
        }


class DefensiveAgent(StrategyBase):
    """Survive first; collect resources and counter nearby pressure."""

    def next_action(self, state: dict[str, Any], agent_id: str | int) -> str:
        snapshot, me, enemies, blocked, danger = self.context(state, agent_id)
        if me is None:
            return Action.WAIT.value
        origin = _entity_position(me)

        # 1-2. Build danger map and BFS to the closest safe cell.
        if origin in danger:
            escape = self._first_step_to(
                origin,
                lambda cell: cell not in danger,
                snapshot,
                blocked,
                set(snapshot.explosions),
            )
            return escape.value if escape else Action.WAIT.value

        # 3. Power-ups are the highest priority while safe.
        powerups = {
            _entity_position(item)
            for item in state.get("powerups", state.get("items", []))
        }
        step = self.path_to(origin, powerups, snapshot, blocked, danger)
        if step:
            return step.value

        # 4. Counter-bomb an opponent that enters the defensive perimeter.
        bombs = int(me.get("bombs", me.get("bombCount", 0)))
        radius = int(me.get("radius", me.get("blastRadius", 2)))
        nearest = min(
            (abs(origin[0] - x) + abs(origin[1] - y) for x, y in enemies),
            default=999,
        )
        if (
            bombs > 0
            and nearest <= 3
            and self.can_escape_after_bomb(origin, radius, snapshot, blocked)
        ):
            return Action.PLACE_BOMB.value

        # 5. Open the map by approaching a safe side of the nearest box.
        if (
            bombs > 0
            and self._blast_from(origin, radius, snapshot.walls, snapshot.boxes)
            .intersection(snapshot.boxes)
            and self.can_escape_after_bomb(origin, radius, snapshot, blocked)
        ):
            return Action.PLACE_BOMB.value
        targets = {
            cell
            for cell in self.box_frontiers(snapshot.boxes)
            if self._valid(cell, snapshot, blocked, danger)
        }
        step = self.path_to(origin, targets, snapshot, blocked, danger)
        if step:
            return step.value

        return self._safest_move(origin, enemies, snapshot, blocked, danger)

    def _safest_move(self, origin, enemies, snapshot, blocked, danger) -> str:
        choices = []
        for action, (dx, dy) in MOVES:
            cell = (origin[0] + dx, origin[1] + dy)
            if not self._valid(cell, snapshot, blocked, danger):
                continue
            distance = min(
                (abs(cell[0] - x) + abs(cell[1] - y) for x, y in enemies),
                default=0,
            )
            exits = sum(
                self._valid(
                    (cell[0] + mx, cell[1] + my), snapshot, blocked, danger
                )
                for _, (mx, my) in MOVES
            )
            choices.append((distance + exits * 2, action))
        return max(choices, default=(0, Action.WAIT), key=lambda item: item[0])[1].value


class AggressiveAgent(StrategyBase):
    """Hunt opponents with A* and use bombs to constrain escape routes."""

    def next_action(self, state: dict[str, Any], agent_id: str | int) -> str:
        snapshot, me, enemies, blocked, danger = self.context(state, agent_id)
        if me is None:
            return Action.WAIT.value
        origin = _entity_position(me)

        # 1. Bomb safety overrides every attacking objective.
        if origin in danger:
            escape = self._first_step_to(
                origin,
                lambda cell: cell not in danger,
                snapshot,
                blocked,
                set(snapshot.explosions),
            )
            return escape.value if escape else Action.WAIT.value

        bombs = int(me.get("bombs", me.get("bombCount", 0)))
        radius = int(me.get("radius", me.get("blastRadius", 2)))
        can_bomb = bombs > 0 and self.can_escape_after_bomb(
            origin, radius, snapshot, blocked
        )

        # 2. Bomb immediately if an enemy is lockable in the current blast line.
        if can_bomb and any(
            self._clear_blast_line(origin, enemy, radius, snapshot)
            for enemy in enemies
        ):
            return Action.PLACE_BOMB.value

        # 6. Apply pressure slightly outside blast range to force movement.
        nearest = min(
            (abs(origin[0] - x) + abs(origin[1] - y) for x, y in enemies),
            default=999,
        )
        if can_bomb and nearest <= radius + 2 and self._degree(origin, snapshot, blocked) <= 2:
            return Action.PLACE_BOMB.value

        if can_bomb and self._blast_from(
            origin, radius, snapshot.walls, snapshot.boxes
        ).intersection(snapshot.boxes):
            return Action.PLACE_BOMB.value

        # 3-4. A* pursuit; intersections and narrow corridors receive lower cost.
        step = self._astar(origin, set(enemies), snapshot, blocked, danger)
        if step:
            return step.value

        # Boxes are secondary targets when enemies are unreachable.
        targets = {
            cell
            for cell in self.box_frontiers(snapshot.boxes)
            if self._valid(cell, snapshot, blocked, danger)
        }
        step = self._astar(origin, targets, snapshot, blocked, danger)
        return step.value if step else Action.WAIT.value

    def _astar(self, start, targets, snapshot, blocked, danger):
        if not targets:
            return None
        queue = [(0.0, 0, start, None)]
        best = {start: 0.0}
        serial = 0
        while queue:
            _, cost, cell, first = heapq.heappop(queue)
            if cell != start and cell in targets:
                return first
            for action, (dx, dy) in MOVES:
                nxt = (cell[0] + dx, cell[1] + dy)
                if not self._valid(nxt, snapshot, blocked, danger):
                    continue
                degree = self._degree(nxt, snapshot, blocked)
                tactical_bonus = 0.25 if degree >= 3 else 0.12 if degree == 2 else 0
                new_cost = cost + 1.0 - tactical_bonus
                if new_cost >= best.get(nxt, float("inf")):
                    continue
                best[nxt] = new_cost
                heuristic = min(abs(nxt[0] - x) + abs(nxt[1] - y) for x, y in targets)
                serial += 1
                heapq.heappush(
                    queue, (new_cost + heuristic, new_cost, nxt, first or action)
                )
        return None

    def _degree(self, cell, snapshot, blocked):
        return sum(
            0 <= cell[0] + dx < snapshot.width
            and 0 <= cell[1] + dy < snapshot.height
            and (cell[0] + dx, cell[1] + dy) not in blocked
            for _, (dx, dy) in MOVES
        )

    def _clear_blast_line(self, origin, enemy, radius, snapshot) -> bool:
        if origin[0] != enemy[0] and origin[1] != enemy[1]:
            return False
        if abs(origin[0] - enemy[0]) + abs(origin[1] - enemy[1]) > radius:
            return False
        return enemy in self._blast_from(
            origin, radius, snapshot.walls, snapshot.boxes
        )
