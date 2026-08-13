#!/usr/bin/env python3
"""Build a colour profile from masked image pairs.

Each pair is an image and a copy of it scribbled with magenta (255, 0, 255)
over the areas of interest, e.g. img_15.jpg and img_15_dontcare.jpg. The
pixels under the magenta are collected from the clean image and fitted with a
gaussian colour model (mean + covariance), the same model detect.py uses, then
saved as <folder>.json.

Given a test image as well, that image is scored against the profile with
Mahalanobis distance and everything far from the profile becomes the mask.

Images are read with opencv, so the model lives in BGR: an image scored
against a profile must come from cv2.imread too, not from a reader that
hands back RGB.

Usage: python color_profile.py grass
       python color_profile.py grass --plot --test grass/img_15.jpg
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

MASK_SUFFIX = "_dontcare"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

# Images are read with opencv, so every array here is BGR and the profile is
# fitted in BGR. Anything scored against the profile must be BGR too.
CHANNELS = "BGR"

# The mask is painted as pure magenta, but jpeg compression smears it: only
# ~0.5% of a stroke survives as an exact 255,0,255. Anything within this
# per-channel tolerance counts as masked; the sample set is stable from
# roughly 20 up to 100, so 60 sits in the middle of the plateau.
DEFAULT_TOLERANCE = 60

# Mahalanobis distance from the profile beyond which a pixel is an object.
# On the grass profile 3 leaves the field speckled and 6 eats into the animal,
# while 4 holds it as a solid blob.
DEFAULT_THRESHOLD = 4.0

DEFAULT_MAX_POINTS = 20000
PREVIEW_MAX_PX = 1200


def _pyplot():
    """pyplot, with the faster Qt platform picked before Qt loads."""
    import os

    # Qt's wayland plugin is about a second slower to the first frame
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

    import matplotlib.pyplot as plt

    return plt


def find_pairs(folder):
    """Return (image, mask) pairs in a folder, sorted by image name."""
    pairs = []
    for mask in sorted(folder.iterdir()):
        if not mask.is_file() or mask.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if not mask.stem.endswith(MASK_SUFFIX):
            continue
        image = mask.with_name(mask.stem[: -len(MASK_SUFFIX)] + mask.suffix)
        if image.is_file():
            pairs.append((image, mask))
    return pairs


def read_image(path):
    """Read an image as an (h, w, 3) BGR array, the way opencv sees it."""
    image = cv2.imread(str(path))
    if image is None:
        raise SystemExit(f"could not read image: {path}")
    return image


def collect_pixels(image_path, mask_path, tolerance=DEFAULT_TOLERANCE):
    """Collect the image pixels that are magenta in the mask.

    Returns an (n, 3) uint8 array of RGB values.
    """
    image = read_image(image_path)
    mask_image = read_image(mask_path).astype(np.int16)

    if image.shape[:2] != mask_image.shape[:2]:
        raise ValueError(f"{image_path} and {mask_path} differ in size")

    # magenta is 255,0,255 in BGR just as in RGB - only green sits low
    blue, green, red = mask_image[..., 0], mask_image[..., 1], mask_image[..., 2]
    masked = (blue >= 255 - tolerance) & (green <= tolerance) & (red >= 255 - tolerance)
    return image[masked]


def fit_profile(name, samples):
    """Fit a gaussian colour model to the collected pixels."""
    pixels = samples.astype(np.float32)
    return {
        "name": name,
        "channels": CHANNELS,
        "pixels": len(pixels),
        "mean": np.mean(pixels, axis=0).tolist(),
        "covariance": np.cov(pixels, rowvar=False).tolist(),
    }


def save_profile(profile, path):
    with open(path, "w") as fh:
        json.dump(profile, fh, indent=2)


def load_profile(path):
    with open(path) as fh:
        return json.load(fh)


def mahalanobis_distance(image, profile):
    """Distance from every pixel of an image to the profile."""
    mean = np.array(profile["mean"], dtype=np.float32)
    inv_cov = np.linalg.inv(np.array(profile["covariance"], dtype=np.float32))

    diff = image.astype(np.float32).reshape(-1, 3) - mean
    squared = np.einsum("ij,jk,ik->i", diff, inv_cov, diff)
    return np.sqrt(np.maximum(squared, 0)).reshape(image.shape[:2])


def build_mask(image, profile, threshold=DEFAULT_THRESHOLD):
    """Objects are what the profile does not explain: far from the model."""
    return mahalanobis_distance(image, profile) >= threshold


def plot_profile(samples, title, max_points=DEFAULT_MAX_POINTS):
    """Build the interactive 3D RGB scatter of the collected pixels."""
    plt = _pyplot()

    shown = samples
    if len(shown) > max_points:
        rng = np.random.default_rng(0)
        shown = shown[rng.choice(len(shown), max_points, replace=False)]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection="3d")
    # the axes follow the array, blue first; matplotlib wants the dot colours
    # the other way round
    ax.scatter(
        shown[:, 0], shown[:, 1], shown[:, 2], c=shown[:, ::-1] / 255.0, s=4, depthshade=False
    )
    ax.set_title(f"{title} — {len(samples)} px, showing {len(shown)}")
    ax.set_xlabel("blue")
    ax.set_ylabel("green")
    ax.set_zlabel("red")
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.set_zlim(0, 255)
    # no tight_layout here: it clips the z axis label on 3d axes


def plot_detection(image, mask, title):
    """Build the image and its mask side by side."""
    plt = _pyplot()

    # resampling a full 8 MP frame down to a small window costs ~1s of CPU
    step = max(1, max(image.shape[:2]) // PREVIEW_MAX_PX)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    axes[0].imshow(image[::step, ::step, ::-1])  # BGR array, RGB display
    axes[0].set_title(title)
    axes[1].imshow(mask[::step, ::step], cmap="gray")
    axes[1].set_title(f"mask — {100 * mask.mean():.2f}% of pixels")
    for ax in axes:
        ax.set_axis_off()
    fig.tight_layout()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="folder of image/mask pairs")
    parser.add_argument("--test", type=Path, help="image to score against the profile")
    parser.add_argument(
        "-t", "--tolerance", type=int, default=DEFAULT_TOLERANCE, help="magenta match tolerance"
    )
    parser.add_argument(
        "-T",
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="mahalanobis distance beyond which a pixel is an object",
    )
    parser.add_argument(
        "-n", "--max-points", type=int, default=DEFAULT_MAX_POINTS, help="points to plot"
    )
    parser.add_argument("-o", "--out", type=Path, help="profile file (default <folder>.json)")
    parser.add_argument("--plot", action="store_true", help="show the plots")
    args = parser.parse_args()

    folder = args.folder
    if not folder.is_dir():
        # also allow naming a folder that sits beside this script
        beside = Path(__file__).parent / folder
        if not beside.is_dir():
            sys.exit(f"not a directory: {folder}")
        folder = beside

    pairs = find_pairs(folder)
    if not pairs:
        sys.exit(f"no *{MASK_SUFFIX} image pairs in {folder}")

    collected = []
    for image_path, mask_path in pairs:
        pixels = collect_pixels(image_path, mask_path, args.tolerance)
        collected.append(pixels)
        print(f"{image_path.name}  {mask_path.name}  {len(pixels)} px")

    samples = np.concatenate(collected)
    profile = fit_profile(folder.name, samples)

    out = args.out or folder.with_suffix(".json")
    save_profile(profile, out)

    print(f"total: {len(pairs)} pair(s), {len(samples)} px")
    print("  mean " + "  ".join(f"{c}={v:.1f}" for c, v in zip(CHANNELS, profile["mean"])))
    print("  std  " + "  ".join(f"{c}={v:.1f}" for c, v in zip(CHANNELS, samples.std(0))))
    print(f"profile written to {out}")

    if args.plot:
        plot_profile(samples, folder.name, args.max_points)

    if args.test is not None:
        test = args.test
        if not test.is_file():
            # also allow naming an image below this script, like the folder
            beside = Path(__file__).parent / test
            if not beside.is_file():
                sys.exit(f"no such image: {test}")
            test = beside
        image = read_image(test)
        distance = mahalanobis_distance(image, profile)
        mask = distance >= args.threshold
        print(f"{test}: distance {distance.min():.2f}-{distance.max():.2f}")
        print(f"  {mask.sum()} px ({100 * mask.mean():.2f}%) beyond T={args.threshold}")
        if args.plot:
            plot_detection(image, mask, str(test))

    if args.plot:
        _pyplot().show()

    return profile


if __name__ == "__main__":
    main()
