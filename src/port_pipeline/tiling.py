from __future__ import annotations

from pathlib import Path

from .models import Tile


def generate_tile_grid(image_width: int, image_height: int, tile_size: int, overlap: int) -> list[Tile]:
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must be >= 0 and < tile_size")

    step = tile_size - overlap
    xs = _axis_positions(image_width, tile_size, step)
    ys = _axis_positions(image_height, tile_size, step)

    tiles: list[Tile] = []
    for row_index, offset_y in enumerate(ys):
        for col_index, offset_x in enumerate(xs):
            width = min(tile_size, image_width - offset_x)
            height = min(tile_size, image_height - offset_y)
            tiles.append(
                Tile(
                    tile_id=f"tile_r{row_index:03d}_c{col_index:03d}",
                    image_path="",
                    offset_x=offset_x,
                    offset_y=offset_y,
                    width=width,
                    height=height,
                )
            )
    return tiles


def save_tiles(image_path: Path, output_dir: Path, tile_size: int, overlap: int) -> list[Tile]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for image tiling. Install requirements.txt first.") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as image:
        image_width, image_height = image.size
        tiles = generate_tile_grid(
            image_width=image_width,
            image_height=image_height,
            tile_size=tile_size,
            overlap=overlap,
        )

        saved_tiles: list[Tile] = []
        for tile in tiles:
            tile_filename = f"{tile.tile_id}.png"
            tile_path = output_dir / tile_filename
            crop_box = (
                tile.offset_x,
                tile.offset_y,
                tile.offset_x + tile.width,
                tile.offset_y + tile.height,
            )
            image.crop(crop_box).save(tile_path)
            saved_tiles.append(
                Tile(
                    tile_id=tile.tile_id,
                    image_path=str(tile_path),
                    offset_x=tile.offset_x,
                    offset_y=tile.offset_y,
                    width=tile.width,
                    height=tile.height,
                )
            )
    return saved_tiles


def _axis_positions(total_size: int, tile_size: int, step: int) -> list[int]:
    if total_size <= tile_size:
        return [0]

    positions: list[int] = []
    position = 0
    while True:
        positions.append(position)
        if position + tile_size >= total_size:
            break
        next_position = position + step
        if next_position + tile_size > total_size:
            next_position = total_size - tile_size
        if next_position == position:
            break
        position = next_position
    return positions
