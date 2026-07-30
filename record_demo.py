"""Render a short animated GIF for the GitHub README."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import torch

from local_engine import LocalBomberland
from ppo_agent import PPOAgent
from strategies import AggressiveAgent, DefensiveAgent


COLORS = {
    "background": "#090d0b",
    "panel": "#141a16",
    "floor": "#202a21",
    "wall": "#4b554d",
    "box": "#81583d",
    "blast": "#ffb32e",
    "bomb": "#080a09",
    "text": "#edf0e8",
    "muted": "#819087",
}
AGENT_COLORS = ("#caff39", "#53baff", "#ff7447")


def font(size: int, bold: bool = False):
    names = (
        ["C:/Windows/Fonts/arialbd.ttf", "DejaVuSans-Bold.ttf"]
        if bold
        else ["C:/Windows/Fonts/arial.ttf", "DejaVuSans.ttf"]
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_frame(engine: LocalBomberland, total_step: int, width: int = 560) -> Image.Image:
    padding, header, footer = 30, 82, 72
    board_size = width - padding * 2
    cell = board_size // engine.size
    board_size = cell * engine.size
    height = header + board_size + footer
    image = Image.new("RGB", (width, height), COLORS["background"])
    draw = ImageDraw.Draw(image)

    draw.text((padding, 24), "BOMBERLAND", fill=COLORS["text"], font=font(22, True))
    draw.ellipse((width - 152, 31, width - 140, 43), fill="#caff39")
    draw.text((width - 132, 29), "3 AI AGENTS", fill=COLORS["muted"], font=font(13))

    origin_x, origin_y = padding, header
    for y in range(engine.size):
        for x in range(engine.size):
            left, top = origin_x + x * cell, origin_y + y * cell
            bounds = (left + 1, top + 1, left + cell - 1, top + cell - 1)
            position = (x, y)
            color = COLORS["floor"]
            if position in engine.walls:
                color = COLORS["wall"]
            elif position in engine.boxes:
                color = COLORS["box"]
            draw.rectangle(bounds, fill=color)
            if position in engine.walls:
                draw.line((left + 3, top + cell - 4, left + cell - 4, top + 3), fill="#69746b", width=2)
            elif position in engine.boxes:
                draw.line((left + 8, top + 8, left + cell - 8, top + cell - 8), fill="#b47b54", width=3)
                draw.line((left + cell - 8, top + 8, left + 8, top + cell - 8), fill="#b47b54", width=3)

    for bomb in engine.bombs:
        x, y = bomb.pos
        cx, cy = origin_x + x * cell + cell // 2, origin_y + y * cell + cell // 2
        radius = cell // 3
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=COLORS["bomb"], outline="#d4d9d4", width=2)
        draw.text((cx - 4, cy - 7), str(bomb.fuse), fill="#ffffff", font=font(12, True))

    for x, y in engine.explosions:
        left, top = origin_x + x * cell, origin_y + y * cell
        draw.rectangle((left + 2, top + 2, left + cell - 2, top + cell - 2), fill=COLORS["blast"])
        draw.ellipse((left + 9, top + 9, left + cell - 9, top + cell - 9), fill="#fff29b")

    for index, player in enumerate(engine.players.values()):
        if not player.alive:
            continue
        x, y = player.pos
        cx, cy = origin_x + x * cell + cell // 2, origin_y + y * cell + cell // 2
        radius = cell // 3
        color = AGENT_COLORS[index]
        draw.polygon(
            [(cx, cy - radius), (cx + radius, cy - radius // 2), (cx + radius, cy + radius), (cx - radius, cy + radius), (cx - radius, cy - radius // 2)],
            fill=color,
        )
        label = str(index + 1)
        draw.text((cx - 4, cy - 7), label, fill="#111611", font=font(13, True))

    footer_y = header + board_size + 20
    draw.text(
        (padding, footer_y),
        f"STEP {total_step:04d} / SINGLE MATCH",
        fill=COLORS["muted"],
        font=font(12, True),
    )
    labels = ("PPO", "DEFENSIVE", "AGGRESSIVE")
    x_cursor = 170
    for index, player in enumerate(engine.players.values()):
        draw.ellipse((x_cursor, footer_y + 2, x_cursor + 10, footer_y + 12), fill=AGENT_COLORS[index])
        status = labels[index] if player.alive else f"{labels[index]} OUT"
        draw.text((x_cursor + 15, footer_y), status, fill=COLORS["text"], font=font(11, True))
        x_cursor += 125
    return image


def create_match(seed: int):
    engine = LocalBomberland(seed=seed, player_count=3)
    agents = {
        "agent_1": PPOAgent("bomberland_ppo.pt", stochastic=False),
        "agent_2": DefensiveAgent(),
        "agent_3": AggressiveAgent(),
    }
    return engine, agents


def record(output: Path, frames: int = 400, seed: int = 23) -> None:
    torch.manual_seed(seed)
    engine = LocalBomberland(seed=seed, player_count=3)
    agents = {
        "agent_1": PPOAgent("bomberland_ppo.pt", stochastic=True),
        "agent_2": DefensiveAgent(),
        "agent_3": AggressiveAgent(),
    }
    images = []
    for total_step in range(frames):
        images.append(render_frame(engine, total_step))
        state = engine.state()
        actions = {
            key: agent.next_action(state, key)
            for key, agent in agents.items()
            if engine.players[key].alive
        }
        engine.step(actions)
        if sum(player.alive for player in engine.players.values()) <= 1:
            break
    output.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        output,
        save_all=True,
        append_images=images[1:],
        duration=110,
        loop=0,
        optimize=True,
    )
    print(f"Saved {len(images)} frames to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("assets/demo-longest.gif"))
    parser.add_argument("--frames", type=int, default=400)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()
    record(args.output, args.frames, args.seed)
