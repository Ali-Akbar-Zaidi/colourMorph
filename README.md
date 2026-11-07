# ColourMorph

ColourMorph applies the visual style of one image to the content of another. You give it a **content image** (the scene, objects or people to keep) and a **style image** (the colours, textures and brush-like patterns to borrow). It produces a new image that keeps the structure of the first and takes on the look of the second.

The project uses adaptive instance normalization (Huang and Belongie, 2017). A frozen pretrained VGG19 encoder reads both images. A custom decoder, trained by the group, turns the transformed features into an RGB image in one forward pass. VGG19's weights never change.

> ColourMorph transfers visual appearance only: colour, texture and artistic style. It does not detect or change anyone's emotional state.

**Team:** Zahra · Muneeb · Hadi

---

## How it works

```
 content photo ──► frozen VGG19 ──► relu4_1 ──┐
                                              ├─► AdaIN ──► trained decoder ──► RGB image
 style painting ─► frozen VGG19 ──► relu4_1 ──┘
```

Training happens before the app is used. The app does not run an optimization loop.

1. **Preprocess.** Both images are converted to RGB, resized to 128 × 128 for the pilot, turned into tensors of shape `[1, 3, 128, 128]` and normalized with ImageNet mean and std.
2. **Extract features.** Frozen VGG19 returns `relu1_1`, `relu2_1`, `relu3_1`, and `relu4_1`. The encoder stops at `relu4_1`.
3. **AdaIN.** The `relu4_1` content features are shifted and scaled so each channel matches the style features' mean and standard deviation. Style strength `alpha` blends that result with the original content features.
4. **Decode.** The trained decoder maps the 512-channel feature map back to a 3-channel image of the same height and width.
5. **Train the decoder only.** Content loss compares `relu4_1` of the generated image with the AdaIN tensor. Style loss compares channel means and standard deviations from `relu1_1` through `relu4_1`. Adam updates the decoder. VGG19 stays frozen.
6. **Display and save.** The app loads `checkpoints/best_decoder.pth`, shows the result, and offers it for download.

---

## Project structure

```
ColourMorph/
├── app.py
├── preprocessing.py
├── feature_extraction.py
├── adain.py
├── model.py
├── losses.py
├── dataset.py
├── train.py
├── style_transfer.py
├── test_style_transfer.py
├── requirements.txt
├── checkpoints/
│   └── best_decoder.pth
├── data/
│   ├── content/
│   └── style/
├── sample_images/
├── outputs/
├── notebooks/
└── README.md
```

---

## Module status

| Module | Owner | Status |
| --- | --- | --- |
| `preprocessing.py`, `dataset.py`, `app.py` | Zahra | Code done; dataset subsets still needed |
| `feature_extraction.py`, `adain.py` | Muneeb | Done |
| `model.py`, `losses.py`, `train.py`, `style_transfer.py` | Hadi | Code done; training run still needed |
| Pilot training and evaluation | Everyone | 12–15 November |

---

## Setup

Requires Python 3.9 or newer.

```bash
git clone <repo-url>
cd ColourMorph
pip install -r requirements.txt
```

The main dependencies are `torch`, `torchvision`, `Pillow`, `numpy`, `matplotlib` and `streamlit`. A GPU is used automatically when CUDA is available. Otherwise everything runs on the CPU, just more slowly.

**Google Colab:** PyTorch and torchvision come preinstalled. Upload the `.py` files to `/content` and run the commands below with a leading `!`.

---

## Usage

### Run the app

Put about 300–500 COCO photographs in `data/content/train` and 300–500 WikiArt paintings in `data/style/train`. Optionally add `data/content/val` and `data/style/val`. Do not commit those images. WikiArt is restricted to non-commercial research use.

```bash
python train.py --epochs 5 --batch-size 4
streamlit run app.py
```

`train.py` saves `checkpoints/best_decoder.pth` and `checkpoints/loss_log.json`. Pass `--drive-dir` on Colab to copy both to Google Drive, and `--resume` to continue from `checkpoints/last.pth`. The app needs the checkpoint file before Generate is enabled.

Upload a content image and a style image (JPG, JPEG or PNG), set the style strength, generate the result, and download it.

### Use the pipeline from Python

```python
from PIL import Image
from preprocessing import preprocess_image, denormalize_image
from style_transfer import transfer_style

content = preprocess_image(Image.open("sample_images/content/city.jpg"), image_size=128)
style = preprocess_image(Image.open("sample_images/style/starry.jpg"), image_size=128)
output = transfer_style(content, style, alpha=1.0)
denormalize_image(output).save("outputs/city_starry.png")
```

---

## Training settings

| Setting | Starting value |
| --- | --- |
| Image size | 128 × 128 |
| Batch size | 4 |
| Optimizer | Adam |
| Learning rate | 0.0001 |
| Epochs | 5 |
| Style loss weight `lambda` | 10 |
| Trainable parameters | Decoder only |

`alpha` in the app is the style strength from 0 to 1. It is not the content-loss weight. `1` uses the full AdaIN features. `0` keeps the content features.

---

## Integration contract

All modules depend on these agreements. Do not change them without telling the team.

| Item | Agreement |
| --- | --- |
| Pilot image size and shape | 128 × 128, tensor `[1, 3, 128, 128]`, RGB |
| Normalization | ImageNet mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]` |
| Feature dictionary keys | `relu1_1`, `relu2_1`, `relu3_1`, `relu4_1` |
| Encoder | `vgg19.features[:21]`, frozen, in-place ReLU replaced |
| AdaIN | Per-channel mean and std at `relu4_1`, then blend with `alpha` |
| Decoder | 512 channels in, 3 channels out, three nearest ×2 upsamples |
| Loss | `content + 10 * style`. Style uses mean and std, not a Gram matrix |
| Checkpoint | `checkpoints/best_decoder.pth` is the decoder `state_dict` |
| Device | CUDA when available, otherwise CPU |

---

## Testing

```bash
python test_style_transfer.py
```

Expected output:

```
passed test_adain_matches_style_statistics
passed test_decoder_output_shape
passed test_losses_use_weights
passed test_decoder_learns_and_vgg_stays_fixed
passed test_transfer_style_output
passed test_dataset_reads_unpaired_folders
all tests passed
```

The tests check AdaIN statistics, the decoder shape, the weighted loss, that a training step updates the decoder and not VGG19, one-pass inference, and the unpaired dataset loader.

---

## Limitations

- The pilot image size is 128 × 128. A checkpoint trained at that size is the one the app loads.
- A few epochs on a few hundred images demonstrate training. They will not match a full research-scale AdaIN model.
- The app needs `checkpoints/best_decoder.pth`. An untrained decoder runs, but it does not transfer style.
- The method matches colour and texture statistics. It does not understand the objects in the picture.

---

## References

- X. Huang and S. Belongie, "Arbitrary Style Transfer in Real-Time with Adaptive Instance Normalization," ICCV 2017.
- L. A. Gatys, A. S. Ecker and M. Bethge, "A Neural Algorithm of Artistic Style," arXiv:1508.06576, 2015.
- K. Simonyan and A. Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition," arXiv:1409.1556, 2014.
- PyTorch, "Neural Transfer Using PyTorch" tutorial, pytorch.org/tutorials.
