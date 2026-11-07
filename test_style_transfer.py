import tempfile
from pathlib import Path

import torch
from PIL import Image

from adain import adaptive_instance_norm, mean_std
from dataset import StyleTransferDataset
from feature_extraction import CONTENT_LAYER, STYLE_LAYERS, load_feature_extractor
from losses import content_loss, style_loss, total_loss
from model import Decoder
from preprocessing import IMAGENET_MEAN, IMAGENET_STD, preprocess_image
from style_transfer import transfer_style
from train import training_step


def make_image(seed, size=128):
    generator = torch.Generator().manual_seed(seed)
    rgb = torch.rand(1, 3, size, size, generator=generator)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (rgb - mean) / std


def test_adain_matches_style_statistics():
    torch.manual_seed(0)
    content = torch.randn(2, 8, 6, 6)
    style = torch.randn(2, 8, 6, 6) * 0.4 - 1.5
    transformed = adaptive_instance_norm(content, style, alpha=1.0)
    got_mean, got_std = mean_std(transformed)
    style_mean, style_std = mean_std(style)
    assert torch.allclose(got_mean, style_mean, atol=1e-4)
    assert torch.allclose(got_std, style_std, atol=1e-4)

    unchanged = adaptive_instance_norm(content, style, alpha=0.0)
    assert torch.equal(unchanged, content)


def test_decoder_output_shape():
    decoder = Decoder()
    output = decoder(torch.randn(2, 512, 16, 16))
    assert output.shape == (2, 3, 128, 128)
    assert output.requires_grad


def test_losses_use_weights():
    features = {}
    for layer, channels in zip(STYLE_LAYERS, (64, 128, 256, 512)):
        features[layer] = torch.randn(2, channels, 4, 4)
    assert content_loss(features, features[CONTENT_LAYER]).item() == 0
    assert style_loss(features, features).item() == 0
    assert total_loss(torch.tensor(2.0), torch.tensor(3.0), 10.0).item() == 32.0


def test_decoder_learns_and_vgg_stays_fixed():
    extractor = load_feature_extractor()
    vgg_before = [param.clone() for param in extractor.parameters()]
    decoder = Decoder()
    decoder_before = [param.clone() for param in decoder.parameters()]
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-4)
    training_step(decoder, extractor, optimizer, make_image(0), make_image(1), 10.0)

    for before, after in zip(vgg_before, extractor.parameters()):
        assert torch.equal(before, after)
    assert any(
        not torch.equal(before, after)
        for before, after in zip(decoder_before, decoder.parameters())
    )


def test_transfer_style_output():
    extractor = load_feature_extractor()
    decoder = Decoder()
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "best_decoder.pth"
        torch.save(decoder.state_dict(), path)
        output = transfer_style(
            make_image(0),
            make_image(1),
            alpha=0.5,
            checkpoint_path=path,
            feature_extractor=extractor,
        )
    assert output.shape == (1, 3, 128, 128)
    assert output.dtype == torch.float32
    assert not output.requires_grad
    assert torch.isfinite(output).all()


def test_dataset_reads_unpaired_folders():
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        content = root / "content"
        style = root / "style"
        content.mkdir()
        style.mkdir()
        Image.new("RGB", (40, 30), (255, 0, 0)).save(content / "a.jpg")
        Image.new("RGB", (20, 50), (0, 0, 255)).save(content / "b.png")
        Image.new("RGB", (33, 33), (0, 255, 0)).save(style / "paint.jpg")
        dataset = StyleTransferDataset(content, style, image_size=128)
        image, painting = dataset[0]
        assert image.shape == (3, 128, 128)
        assert painting.shape == (3, 128, 128)
        roundtrip = preprocess_image(Image.open(content / "a.jpg"), image_size=128).squeeze(0)
        assert torch.allclose(dataset[0][0], roundtrip, atol=1e-5)


if __name__ == "__main__":
    tests = [
        test_adain_matches_style_statistics,
        test_decoder_output_shape,
        test_losses_use_weights,
        test_decoder_learns_and_vgg_stays_fixed,
        test_transfer_style_output,
        test_dataset_reads_unpaired_folders,
    ]
    for test in tests:
        test()
        print("passed", test.__name__)
    print("all tests passed")
