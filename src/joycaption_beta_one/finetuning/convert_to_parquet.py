#!/usr/bin/env python3
import io
import json
from pathlib import Path

import click
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torchvision.transforms.functional as TVF
from PIL import Image
from tqdm import tqdm


def preprocess_image(image_path: Path) -> torch.Tensor:
    """Preprocesses a single image."""
    img = Image.open(image_path).convert("RGB")
    if img.size != (384, 384):
        img = img.resize((384, 384), Image.LANCZOS)
    return TVF.pil_to_tensor(img)


def save_to_parquet(data: list[dict], output_file: Path):
    """Saves the processed data to a Parquet file."""
    print(f"Saving preprocessed data to {output_file}...")
    data_to_save = []
    for ex in tqdm(data, desc="Serializing tensors"):
        new_ex = ex.copy()
        buffer = io.BytesIO()
        torch.save(new_ex["pixel_values"], buffer)
        new_ex["pixel_values"] = buffer.getvalue()
        del ex["pixel_values"]  # Free up memory
        data_to_save.append(new_ex)

    table = pa.Table.from_pylist(data_to_save)
    pq.write_table(table, output_file)
    print("Done.")


def convert_from_dirs(images_dir: Path, captions_dir: Path, output_file: Path):
    """Converts from a directory of images and a directory of caption files."""
    print(f"Converting from images in '{images_dir}' and captions in '{captions_dir}'")
    image_extensions = [".jpg", ".jpeg", ".png", ".webp"]
    image_files = [p for p in images_dir.iterdir() if p.suffix.lower() in image_extensions]
    
    data = []
    for image_path in tqdm(image_files, desc="Processing images and captions"):
        caption_path = captions_dir / f"{image_path.stem}.txt"
        if not caption_path.exists():
            print(f"Warning: No caption found for {image_path.name}, skipping.")
            continue

        caption = caption_path.read_text().strip()
        messages = [
            {"role": "user", "content": "<image>"},
            {"role": "assistant", "content": caption},
        ]

        pixel_values = preprocess_image(image_path)

        data.append(
            {"images": [image_path.name], "messages": messages, "pixel_values": pixel_values}
        )
    save_to_parquet(data, output_file)


def convert_from_dataset_json(dataset_json: Path, images_dir: Path, output_file: Path):
    """Converts from a dataset.json file and a directory of images."""
    print(f"Converting from '{dataset_json}' and images in '{images_dir}'")
    data = json.loads(dataset_json.read_text())
    for ex in tqdm(data, desc="Preprocessing images"):
        image_filename = ex["images"][0]
        image_path = images_dir / image_filename
        ex["pixel_values"] = preprocess_image(image_path)
        ex["messages"] = [
            {**m, "content": m["content"].replace("<image>", "").strip()}
            for m in ex["messages"]
        ]
    save_to_parquet(data, output_file)


def convert_from_cache(cache_dir: Path, dataset_json: Path, output_file: Path):
    """Converts from a .cache directory of .pt files, using metadata from dataset.json."""
    print(f"Converting from cache '{cache_dir}' using metadata from '{dataset_json}'")
    data = json.loads(dataset_json.read_text())
    
    processed_data = []
    for ex in tqdm(data, desc="Loading cached preprocessed images"):
        image_filename = ex["images"][0]
        cached_image_path = cache_dir / f"{Path(image_filename).stem}.pt"
        if not cached_image_path.exists():
            print(f"Warning: No cached tensor found for {image_filename}, skipping.")
            continue
        
        ex["pixel_values"] = torch.load(cached_image_path)
        ex["messages"] = [
            {**m, "content": m["content"].replace("<image>", "").strip()}
            for m in ex["messages"]
        ]
        processed_data.append(ex)

    save_to_parquet(processed_data, output_file)


@click.command()
@click.option("--images-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), help="Path to the directory of images.")
@click.option("--captions-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), help="Path to the directory of caption files.")
@click.option("--dataset-json", type=click.Path(exists=True, dir_okay=False, path_type=Path), help="Path to the dataset.json file.")
@click.option("--cache-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), help="Path to the .cache directory with .pt files.")
@click.option("--output-file", type=click.Path(dir_okay=False, path_type=Path), required=True, help="Path to the output Parquet file.")
def main(images_dir, captions_dir, dataset_json, cache_dir, output_file):
    """Converts image datasets to Parquet format for faster loading."""
    if cache_dir:
        if not dataset_json:
            raise click.UsageError("`--dataset-json` is required with `--cache-dir`.")
        convert_from_cache(cache_dir, dataset_json, output_file)
    elif dataset_json:
        if not images_dir:
            raise click.UsageError("`--images-dir` is required with `--dataset-json`.")
        convert_from_dataset_json(dataset_json, images_dir, output_file)
    elif images_dir and captions_dir:
        convert_from_dirs(images_dir, captions_dir, output_file)
    else:
        raise click.UsageError(
            "Please specify a source:\n"
            "1. `--images-dir` and `--captions-dir`\n"
            "2. `--dataset-json` and `--images-dir`\n"
            "3. `--cache-dir` and `--dataset-json`"
        )


if __name__ == "__main__":
    main()
