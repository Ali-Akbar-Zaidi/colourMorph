# ColourMorph

ColourMorph applies the visual style of one image to the content of another. You give it a **content image** (the scene, objects or people to keep) and a **style image** (the colours, textures and brush-like patterns to borrow). It produces a new image that keeps the structure of the first and takes on the look of the second.

The project uses **Neural Style Transfer** (Gatys et al., 2015). A pretrained VGG19 network serves as a fixed feature extractor. Its weights never change. Only the pixels of the generated image are optimized.

> ColourMorph transfers visual appearance only: colour, texture and artistic style. It does not detect or change anyone's emotional state.

**Team:** Zahra · Muneeb · Hadi

---

## How it works

```
 content image ─┐                                  ┌─> generated tensor ─> RGB image ─> preview / download
                ├─> preprocessing ─> style transfer ┤
 style image ───┘      (Zahra)          (Hadi)      │
                                          ▲         │
                                          │         │
                              VGG19 features + Gram matrices
                                       (Muneeb)
```

1. **Preprocess.** Both images are converted to RGB, resized to 256 × 256, turned into tensors of shape `[1, 3, 256, 256]` and normalized with ImageNet mean and std.
2. **Extract features.** VGG19 returns activations from the content layer (`conv4_2`) and the style layers (`conv1_1`, `conv2_1`, `conv3_1`, `conv4_1`, `conv5_1`).
3. **Represent style.** A Gram matrix of each style layer captures how its feature channels correlate. This describes texture and colour while ignoring where things are in the image.
4. **Compute the losses.**
   - Content loss: MSE between the `conv4_2` features of the generated and content images.
   - Style loss: weighted MSE between the Gram matrices of the generated and style images, with weight 0.2 for each of the five layers.
   - Total loss: `L_total = alpha × L_content + beta × L_style`
5. **Optimize.** The generated image starts as a copy of the content image. Adam updates its pixels to reduce the total loss. After every step the pixels are clamped so they stay in the valid 0 to 1 range once denormalized.
6. **Display and save.** The final tensor is converted back to an RGB image, shown in the Streamlit app and offered for download.

---

## Project structure

```
ColourMorph/
├── app.py                    # Streamlit interface (Zahra)
├── preprocessing.py          # image loading, resizing, normalization (Zahra)
├── feature_extraction.py     # VGG19 loading, feature extraction, Gram matrix (Muneeb)
├── style_transfer.py         # losses and image optimization (Hadi)
├── test_style_transfer.py    # tests for style_transfer.py (Hadi)
├── requirements.txt
├── sample_images/
│   ├── content/
│   └── style/
├── outputs/
├── notebooks/
└── README.md
```

---

## Module status

| Module | Owner | Status |
| --- | --- | --- |
| `preprocessing.py`, `app.py` | Zahra | In progress |
| `feature_extraction.py` | Muneeb | In progress |
| `style_transfer.py`, `test_style_transfer.py` | Hadi | Done, tests pass with a stand-in extractor |
| Integration and evaluation | Everyone | 11–14 November |

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

```bash
streamlit run app.py
```

Upload a content image and a style image (JPG, JPEG or PNG), generate the result and download it.

### Use the pipeline from Python

```python
import torch
from PIL import Image
from preprocessing import preprocess_image, denormalize_image
from feature_extraction import load_feature_extractor
from style_transfer import transfer_style

device = "cuda" if torch.cuda.is_available() else "cpu"

content = preprocess_image(Image.open("sample_images/content/city.jpg"), device=device)
style = preprocess_image(Image.open("sample_images/style/starry.jpg"), device=device)
model = load_feature_extractor(device=device)

output = transfer_style(content, style, model, {"steps": 300})
denormalize_image(output).save("outputs/city_starry.png")
```

The image file names are only examples.

---

## Style transfer settings

`transfer_style(content_tensor, style_tensor, feature_extractor, config)` takes an optional `config` dictionary. Any key you leave out uses its default value.

| Key | Default | Meaning |
| --- | --- | --- |
| `alpha` | `1.0` | Weight of the content loss |
| `beta` | `1e6` | Weight of the style loss |
| `lr` | `0.02` | Adam learning rate |
| `steps` | `300` | Number of optimization steps |
| `style_weights` | `0.2` per layer | Weight of each style layer |
| `device` | device of `content_tensor` | Where the optimization runs |
| `log_every` | `50` | Print the losses every N steps (`0` turns printing off) |
| `callback` | `None` | `callback(step, total_steps, loss)` is called after every step. Useful for a progress bar in the app. |

The function returns a detached, normalized tensor of shape `[1, 3, 256, 256]` on the same device as the input.

**Tuning tips**

- For more style, raise `beta`. For a result closer to the original photo, lower it.
- More `steps` give a cleaner result but take longer.
- These defaults are starting values. They will be tuned with real images during evaluation.

---

## Integration contract

All modules depend on these agreements. Do not change them without telling the team.

| Item | Agreement |
| --- | --- |
| Image size and shape | 256 × 256, tensor `[1, 3, 256, 256]`, RGB |
| Normalization | ImageNet mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`. It must match in `preprocessing.py` and `style_transfer.py`. |
| Feature dictionary keys | `conv1_1`, `conv2_1`, `conv3_1`, `conv4_1`, `conv4_2`, `conv5_1` |
| Gram matrix | Divided by C × H × W. `beta = 1e6` assumes this. Other scaling means `beta` has to be retuned. |
| Gradients | `extract_features` must not run under `torch.no_grad()`, because the optimization needs gradients through it. |
| VGG19 ReLU | In-place ReLU must be turned off. Otherwise the stored feature maps get overwritten silently. |
| Device | CUDA when available, otherwise CPU. Every tensor and the model must be on the same device. |

---

## Testing

```bash
python test_style_transfer.py
```

Expected output:

```
using stand-in feature extractor
passed test_losses_are_zero_for_same_features
passed test_losses_are_positive_for_different_images
passed test_total_loss_uses_weights
passed test_transfer_style_output
passed test_model_weights_stay_fixed
all tests passed
```

The tests check that:
- the losses are zero for identical features and positive for different ones
- the output has the right shape, dtype and device, with finite values in the valid pixel range
- the loss goes down during optimization
- the input tensors and the VGG19 weights are not changed

When `feature_extraction.py` is missing, the tests use a stand-in VGG19 with random weights, so nothing needs to download. Once Muneeb's file is in the folder, the tests use it automatically and the first line changes to `using feature_extraction.py`.

---

## Limitations

- Every image is generated by optimization, so each one takes many forward and backward passes. It is slow on CPU.
- The output is fixed at 256 × 256.
- Results depend a lot on the image pair and on the balance between `alpha` and `beta`.
- Images are compared on texture and colour only. The method doesn't understand what the objects in them are.

---

## References

- L. A. Gatys, A. S. Ecker and M. Bethge, "A Neural Algorithm of Artistic Style," arXiv:1508.06576, 2015.
- L. A. Gatys, A. S. Ecker and M. Bethge, "Image Style Transfer Using Convolutional Neural Networks," CVPR 2016.
- K. Simonyan and A. Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition," arXiv:1409.1556, 2014.
- PyTorch, "Neural Transfer Using PyTorch" tutorial, pytorch.org/tutorials.
