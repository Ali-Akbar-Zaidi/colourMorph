import argparse
import json
import shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from adain import adain_target
from dataset import StyleTransferDataset
from feature_extraction import extract_features, load_feature_extractor
from losses import (
    DEFAULT_LAMBDA,
    DEFAULT_LAMBDA_EDGE,
    content_loss,
    edge_loss,
    style_loss,
    total_loss,
)
from model import Decoder
from style_transfer import decode_to_normalized

def training_step(
    decoder, extractor, optimizer, content, style, lambda_style,
    lambda_edge=DEFAULT_LAMBDA_EDGE,
):
    """One Adam update of the decoder. VGG19 weights are not stepped."""
    decoder.train()
    extractor.eval()
    optimizer.zero_grad()

    with torch.no_grad():
        content_features = extract_features(content, extractor)
        style_features = extract_features(style, extractor)
        target = adain_target(content_features, style_features, alpha=1.0)

    generated = decode_to_normalized(decoder, target)
    generated_features = extract_features(generated, extractor)
    c_loss = content_loss(generated_features, target)
    s_loss = style_loss(generated_features, style_features)
    e_loss = edge_loss(generated, content)
    loss = total_loss(c_loss, s_loss, lambda_style) + lambda_edge * e_loss

    if not loss.requires_grad:
        raise RuntimeError("loss has no gradient; the decoder is not receiving it")

    loss.backward()
    optimizer.step()
    return c_loss.detach(), s_loss.detach(), e_loss.detach(), loss.detach()


@torch.no_grad()
def evaluate(
    decoder, extractor, loader, lambda_style, device, max_steps=0,
    lambda_edge=DEFAULT_LAMBDA_EDGE,
):
    decoder.eval()
    totals = []
    for step, (content, style) in enumerate(loader, start=1):
        content = content.to(device)
        style = style.to(device)
        content_features = extract_features(content, extractor)
        style_features = extract_features(style, extractor)
        target = adain_target(content_features, style_features, alpha=1.0)
        generated = decode_to_normalized(decoder, target)
        generated_features = extract_features(generated, extractor)
        c_loss = content_loss(generated_features, target)
        s_loss = style_loss(generated_features, style_features)
        e_loss = edge_loss(generated, content)
        totals.append(
            (total_loss(c_loss, s_loss, lambda_style) + lambda_edge * e_loss).item()
        )
        if max_steps and step >= max_steps:
            break
    if not totals:
        return None
    return sum(totals) / len(totals)


def run_epoch(
    decoder, extractor, optimizer, loader, lambda_style, device, max_steps=0,
    lambda_edge=DEFAULT_LAMBDA_EDGE,
):
    content_sum = 0.0
    style_sum = 0.0
    edge_sum = 0.0
    total_sum = 0.0
    steps = 0
    for content, style in loader:
        content = content.to(device)
        style = style.to(device)
        c_loss, s_loss, e_loss, loss = training_step(
            decoder, extractor, optimizer, content, style, lambda_style, lambda_edge
        )
        content_sum += c_loss.item()
        style_sum += s_loss.item()
        edge_sum += e_loss.item()
        total_sum += loss.item()
        steps += 1
        if max_steps and steps >= max_steps:
            break
    if steps == 0:
        raise RuntimeError("the data loader produced no batches")
    return (
        content_sum / steps,
        style_sum / steps,
        edge_sum / steps,
        total_sum / steps,
    )


def save_checkpoint(decoder, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(decoder.state_dict(), path)


def copy_to_drive(path, drive_dir):
    if not drive_dir:
        return
    destination = Path(drive_dir)
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy(path, destination / path.name)


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device}")

    train_set = StyleTransferDataset(args.content_dir, args.style_dir, args.image_size)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = None
    if Path(args.val_content_dir).is_dir() and Path(args.val_style_dir).is_dir():
        try:
            val_set = StyleTransferDataset(args.val_content_dir, args.val_style_dir, args.image_size)
        except FileNotFoundError:
            val_set = None
        if val_set is not None:
            val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    extractor = load_feature_extractor(device=device)
    decoder = Decoder().to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)

    checkpoint_dir = Path(args.checkpoint_dir)
    last_path = checkpoint_dir / "last.pth"
    start_epoch = 1
    history = []
    if args.resume and last_path.is_file():
        state = torch.load(last_path, map_location=device, weights_only=True)
        decoder.load_state_dict(state)
        print(f"resumed decoder weights from {last_path}")

    log_path = checkpoint_dir / "loss_log.json"
    if args.resume and log_path.is_file():
        history = json.loads(log_path.read_text())
        start_epoch = len(history) + 1

    best_score = None
    for epoch in range(start_epoch, args.epochs + 1):
        content_value, style_value, edge_value, total_value = run_epoch(
            decoder, extractor, optimizer, train_loader, args.lambda_style, device,
            args.max_steps, args.lambda_edge,
        )
        record = {
            "epoch": epoch,
            "content": content_value,
            "style": style_value,
            "edge": edge_value,
            "total": total_value,
        }
        if val_loader is not None:
            record["val_total"] = evaluate(
                decoder, extractor, val_loader, args.lambda_style, device,
                args.max_steps, args.lambda_edge,
            )
        history.append(record)
        score = record.get("val_total", total_value)
        print(
            f"epoch {epoch}/{args.epochs}  total {total_value:.4f}  "
            f"content {content_value:.4f}  style {style_value:.6f}  "
            f"edge {edge_value:.6f}"
        )

        epoch_path = checkpoint_dir / f"decoder_epoch_{epoch}.pth"
        save_checkpoint(decoder, epoch_path)
        save_checkpoint(decoder, last_path)
        if best_score is None or score < best_score:
            best_score = score
            save_checkpoint(decoder, checkpoint_dir / "best_decoder.pth")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log_path.write_text(json.dumps(history, indent=2))
        copy_to_drive(checkpoint_dir / "best_decoder.pth", args.drive_dir)
        copy_to_drive(log_path, args.drive_dir)

    return history


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train the ColourMorph decoder.")
    parser.add_argument("--content-dir", default="data/content/train")
    parser.add_argument("--style-dir", default="data/style/train")
    parser.add_argument("--val-content-dir", default="data/content/val")
    parser.add_argument("--val-style-dir", default="data/style/val")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda-style", type=float, default=DEFAULT_LAMBDA)
    parser.add_argument("--lambda-edge", type=float, default=DEFAULT_LAMBDA_EDGE)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--drive-dir", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0)
    return parser.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
