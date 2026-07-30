"""Local 4-agent Bomberland harness for developing the challenge agent.

This is a protocol-compatible test runner, not a replacement for the official
competition engine.
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass

from agent import Action, BomberlandAgent


Position = tuple[int, int]
DELTAS = {
    Action.MOVE_UP.value: (0, -1),
    Action.MOVE_DOWN.value: (0, 1),
    Action.MOVE_LEFT.value: (-1, 0),
    Action.MOVE_RIGHT.value: (1, 0),
}


@dataclass
class Player:
    id: str
    pos: Position
    bombs: int = 1
    radius: int = 2
    score: int = 0
    alive: bool = True


@dataclass
class Bomb:
    pos: Position
    owner: str
    fuse: int
    radius: int


class LocalBomberland:
    def __init__(self, size: int = 11, seed: int = 7, player_count: int = 3) -> None:
        self.size = size
        self.random = random.Random(seed)
        self.step_number = 0
        self.walls = {
            (x, y)
            for y in range(1, size, 2)
            for x in range(1, size, 2)
        }
        spawns = [(0, 0), (size - 1, 0), (0, size - 1), (size - 1, size - 1)]
        safe = set()
        for x, y in spawns:
            safe.update({(x, y), (max(0, x - 1), y), (min(size - 1, x + 1), y)})
            safe.update({(x, max(0, y - 1)), (x, min(size - 1, y + 1))})
        self.boxes = {
            (x, y)
            for y in range(size)
            for x in range(size)
            if (x, y) not in self.walls | safe and self.random.random() < 0.32
        }
        self.players = {
            f"agent_{index + 1}": Player(f"agent_{index + 1}", spawn)
            for index, spawn in enumerate(spawns[:player_count])
        }
        self.bombs: list[Bomb] = []
        self.explosions: set[Position] = set()

    def state(self) -> dict:
        tiles = []
        for y in range(self.size):
            row = []
            for x in range(self.size):
                cell = (x, y)
                row.append("WALL" if cell in self.walls else "BOX" if cell in self.boxes else "AIR")
            tiles.append(row)
        return {
            "step": self.step_number,
            "field": {"width": self.size, "height": self.size, "tiles": tiles},
            "bombs": [
                {
                    "pos": {"x": bomb.pos[0], "y": bomb.pos[1]},
                    "owner": bomb.owner,
                    "fuse": bomb.fuse,
                    "radius": bomb.radius,
                }
                for bomb in self.bombs
            ],
            "explosions": [{"x": x, "y": y} for x, y in self.explosions],
            "agents": [
                {
                    "id": player.id,
                    "pos": {"x": player.pos[0], "y": player.pos[1]},
                    "bombs": player.bombs,
                    "radius": player.radius,
                    "score": player.score,
                    "alive": player.alive,
                }
                for player in self.players.values()
            ],
        }

    def step(self, actions: dict[str, str]) -> None:
        self.step_number += 1
        self.explosions.clear()
        occupied = {player.pos for player in self.players.values() if player.alive}

        # Resolve moves simultaneously against the map.
        proposals: dict[str, Position] = {}
        for player_id, action in actions.items():
            player = self.players[player_id]
            if not player.alive or action not in DELTAS:
                continue
            dx, dy = DELTAS[action]
            target = (player.pos[0] + dx, player.pos[1] + dy)
            if self._walkable(target):
                proposals[player_id] = target
        target_counts = {target: list(proposals.values()).count(target) for target in proposals.values()}
        for player_id, target in proposals.items():
            if target_counts[target] == 1 and target not in occupied:
                self.players[player_id].pos = target

        for player_id, action in actions.items():
            player = self.players[player_id]
            if (
                player.alive
                and action == Action.PLACE_BOMB.value
                and player.bombs > 0
                and not any(bomb.pos == player.pos for bomb in self.bombs)
            ):
                self.bombs.append(Bomb(player.pos, player_id, 4, player.radius))
                player.bombs -= 1

        self._update_bombs()

    def _update_bombs(self) -> None:
        pending: list[Bomb] = []
        exploding: list[Bomb] = []
        for bomb in self.bombs:
            bomb.fuse -= 1
            (exploding if bomb.fuse <= 0 else pending).append(bomb)

        while exploding:
            bomb = exploding.pop()
            blast = self._blast(bomb)
            self.explosions.update(blast)
            destroyed = self.boxes.intersection(blast)
            self.boxes.difference_update(destroyed)
            owner = self.players[bomb.owner]
            owner.bombs += 1
            owner.score += len(destroyed) * 10
            chained = [other for other in pending if other.pos in blast]
            for other in chained:
                pending.remove(other)
                exploding.append(other)

        self.bombs = pending
        for player in self.players.values():
            if player.alive and player.pos in self.explosions:
                player.alive = False
                for bomb in exploding:
                    if bomb.owner != player.id:
                        self.players[bomb.owner].score += 100

    def _blast(self, bomb: Bomb) -> set[Position]:
        cells = {bomb.pos}
        for dx, dy in DELTAS.values():
            for distance in range(1, bomb.radius + 1):
                cell = (bomb.pos[0] + dx * distance, bomb.pos[1] + dy * distance)
                if cell in self.walls or not self._inside(cell):
                    break
                cells.add(cell)
                if cell in self.boxes:
                    break
        return cells

    def _walkable(self, cell: Position) -> bool:
        return (
            self._inside(cell)
            and cell not in self.walls
            and cell not in self.boxes
            and all(bomb.pos != cell for bomb in self.bombs)
        )

    def _inside(self, cell: Position) -> bool:
        return 0 <= cell[0] < self.size and 0 <= cell[1] < self.size

    def finished(self, max_steps: int) -> bool:
        alive = sum(player.alive for player in self.players.values())
        return alive <= 1 or self.step_number >= max_steps

    def render(self) -> str:
        glyphs = [["  " for _ in range(self.size)] for _ in range(self.size)]
        for x, y in self.walls:
            glyphs[y][x] = "##"
        for x, y in self.boxes:
            glyphs[y][x] = "[]"
        for bomb in self.bombs:
            glyphs[bomb.pos[1]][bomb.pos[0]] = f"B{bomb.fuse}"
        for x, y in self.explosions:
            glyphs[y][x] = "**"
        for index, player in enumerate(self.players.values(), 1):
            if player.alive:
                glyphs[player.pos[1]][player.pos[0]] = f"A{index}"
        return "\n".join("".join(row) for row in glyphs)


def run(max_steps: int, delay: float, show_map: bool) -> LocalBomberland:
    engine = LocalBomberland()
    agents = {player_id: BomberlandAgent() for player_id in engine.players}
    while not engine.finished(max_steps):
        state = engine.state()
        actions = {
            player_id: agents[player_id].next_action(state, player_id)
            for player_id, player in engine.players.items()
            if player.alive
        }
        engine.step(actions)
        if show_map:
            print(f"\nSTEP {engine.step_number} | {actions}")
            print(engine.render())
            time.sleep(delay)

    print("\nRESULT")
    for player in engine.players.values():
        status = "ALIVE" if player.alive else "OUT"
        print(f"{player.id:<8} {status:<5} score={player.score}")
    return engine


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run four Bomberland agents locally")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--show-map", action="store_true")
    args = parser.parse_args()
    run(args.steps, args.delay, args.show_map)
