#!/usr/bin/env python3
"""List image/json pairs found in a capture directory."""

import argparse
import json
import sys
from pathlib import Path
import numpy as np

from analyze import find_animals

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


IMAGE_WIDTH = 3280
IMAGE_HEIGHT = 2464
CX = IMAGE_WIDTH / 2
CY = IMAGE_HEIGHT / 2

# alt_relative / gsd_m_per_px says 2714 px, which is also what a stock Pi
# camera v2 gives: 62.2 deg across 3280 px. The carpet in capture_18/img_15
# measures 400 px, i.e. 2.2 m at that scale, which matches the real thing, so
# both the lens and the 15 m flying height are right. Pass --focal to override.
FX = FY = 2714.0

# The camera looks straight down, image x to the drone's right and image y
# toward the tail, so image up is the drone's forward direction and no extra
# rotation is needed between camera and body.
MOUNT_YAW_DEG = 0.0

# Degrees of latitude per metre; longitude also shrinks with cos(latitude)
METRES_PER_DEG = 111320.0


#Dronespace
# X axis points forward, Y axis points right, Z axis points down. The camera is
# mounted on the bottom of the drone, so the camera's +Z points down. The
# camera's +X points to the right, and the camera's +Y points backward.


def cam2drone(px, py, focal=FX, mount_yaw_deg=MOUNT_YAW_DEG):
    """Direction of a pixel as a ray in drone space (forward, right, down)."""
    x = (px - CX) / focal  # image right
    y = (py - CY) / focal  # image down the image, i.e. toward the tail
    c, s = np.cos(np.radians(mount_yaw_deg)), np.sin(np.radians(mount_yaw_deg))
    forward = -y * c - x * s
    right = x * c - y * s
    return np.array([forward, right, 1.0])


def drone2world(attitude):
    """Rotation from drone space to north/east/down, from yaw, pitch and roll."""
    yaw, pitch, roll = (
        np.radians(attitude["yaw_deg"]),
        np.radians(attitude["pitch_deg"]),
        np.radians(attitude["roll_deg"]),
    )
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    yaw_matrix = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    pitch_matrix = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    roll_matrix = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return yaw_matrix @ pitch_matrix @ roll_matrix


def ground_position(px, py, telemetry, focal=FX):
    """Where the ground point seen at pixel (px, py) is, as (lat, lon).

    The ray is rotated into world axes and stretched until it reaches the
    ground plane below the drone, exactly as in the mapping exercise. Assumes
    flat ground at the drone's takeoff level.
    """
    position = telemetry.get("position") or {}
    attitude = telemetry.get("attitude") or {}
    altitude = position.get("alt_relative_m")
    lat, lon = position.get("lat_deg"), position.get("lon_deg")
    if altitude is None or lat is None or lon is None or not attitude:
        return None

    ray = drone2world(attitude) @ cam2drone(px, py, focal)
    if ray[2] <= 0:  # pointing at the sky, never reaches the ground
        return None

    north, east = ray[:2] * (altitude / ray[2])
    return (
        lat + north / METRES_PER_DEG,
        lon + east / (METRES_PER_DEG * np.cos(np.radians(lat))),
    )


def read_telemetry(meta_path):
    with open(meta_path) as fh:
        return json.load(fh).get("telemetry", {})


def load_gps_log(folder):
    """Read gps_locations.log: image index -> (lat, lon, alt_amsl, heading).

    The json's position block is one image stale - across all three captures
    it sits a median 0.3 m from the PREVIOUS image's log fix but 2.2 m from
    its own, and for 39 of 165 frames it is an exact copy of the previous fix.
    The log line for an image is the fix at its actual exposure, so it wins.
    """
    log = {}
    path = folder / "gps_locations.log"
    if not path.is_file():
        return log
    with open(path) as fh:
        for line in fh:
            idx, lat, lon, alt, heading = line.strip().split(",")
            log[int(idx)] = (float(lat), float(lon), float(alt), float(heading))
    return log


def apply_gps_log(telemetry, fix):
    """Replace the stale json position (and yaw) with the exposure-time fix."""
    if fix is None:
        return telemetry
    lat, lon, _alt, heading = fix
    return dict(
        telemetry,
        position=dict(telemetry.get("position") or {}, lat_deg=lat, lon_deg=lon),
        attitude=dict(telemetry.get("attitude") or {}, yaw_deg=heading),
    )


def find_pairs(folder):
    """Return (image, json) pairs in a folder, sorted by image name."""
    pairs = []
    for image in sorted(folder.iterdir()):
        if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        meta = image.with_suffix(".json")
        if meta.is_file():
            pairs.append((image, meta))
    return pairs


def find_folders(root):
    """Return the root itself plus any subfolders, sorted."""
    folders = [root]
    folders.extend(sorted(p for p in root.iterdir() if p.is_dir()))
    return folders


def place_animals(animals, meta_path, focal=FX, fix=None):
    """Compute gps location of each animal
    Add that gps location as well as the gps point of the drone.
    Returns a new list of dicts."""

    telemetry = apply_gps_log(read_telemetry(meta_path), fix)
    position = telemetry.get("position") or {}

    placed_animals = []
    for a in animals:
        ground = ground_position(a["cX"], a["cY"], telemetry, focal)
        if ground is None:
            continue
        placed_animals.append(
            {
                **a,
                "lat": ground[0],
                "lon": ground[1],
                "drone_lat": position["lat_deg"],
                "drone_lon": position["lon_deg"],
            }
        )
    return placed_animals


            


def plot_observations(drone_locations, observations, out_path):
    """Map the flight: where each image was taken, and what was seen from it.

    A blue dot per image is the drone position, a red dot per animal is where
    it sits on the ground, and a line joins each animal to the image it was
    spotted from.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lats = [a["lat"] for a in observations]
    lons = [a["lon"] for a in observations]

    drone_lats = [pos["lat_deg"] for pos in drone_locations if "lat_deg" in pos]
    drone_lons = [pos["lon_deg"] for pos in drone_locations if "lon_deg" in pos]

    surface = "#fcfcfb"
    ink = "#0b0b0b"
    muted = "#52514e"
    drone_colour = "#2a78d6"
    animal_colour = "#e34948"

    fig, ax = plt.subplots(figsize=(7, 7), facecolor=surface)
    ax.set_facecolor(surface)

    # sight lines first, so the dots sit on top of them
    for a in observations:
        ax.plot(
            [a["drone_lon"], a["lon"]],
            [a["drone_lat"], a["lat"]],
            color=muted,
            linewidth=0.7,
            alpha=0.6,
            zorder=2,
        )

    ax.scatter(
        drone_lons, drone_lats, s=22, color=drone_colour,
        edgecolors=surface, linewidths=0.5, zorder=3, label="image taken here",
    )
    ax.scatter(
        lons, lats, s=34, color=animal_colour,
        edgecolors=surface, linewidths=0.5, zorder=4, label="animal on the ground",
    )
    ax.legend(loc="best", frameon=False, fontsize=9, labelcolor=muted)

    # a small cutout of each animal, floated beside its red dot
    import cv2
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    # cycle the corner the thumbnail hangs in, so clustered dots overlap less
    corners = [(14, 14), (-14, 14), (14, -14), (-14, -14)]
    frames = {}
    for i, a in enumerate(observations):
        if "image" not in a or "x0" not in a:
            continue
        if a["image"] not in frames:
            frames[a["image"]] = cv2.imread(str(a["image"]))
        img = frames[a["image"]]
        if img is None:
            continue
        pad = max(a["x1"] - a["x0"], a["y1"] - a["y0"]) // 4
        x0, y0 = max(a["x0"] - pad, 0), max(a["y0"] - pad, 0)
        x1, y1 = min(a["x1"] + pad, img.shape[1]), min(a["y1"] + pad, img.shape[0])
        cutout = img[y0:y1, x0:x1, ::-1]  # BGR file, RGB plot
        if cutout.size == 0:
            continue
        scale = 90 / max(cutout.shape[:2])
        cutout = cv2.resize(cutout, (0, 0), fx=scale, fy=scale)
        box = AnnotationBbox(
            OffsetImage(cutout, zoom=0.175),
            (a["lon"], a["lat"]),
            xybox=corners[i % len(corners)],
            xycoords="data",
            boxcoords="offset points",
            frameon=True,
            pad=0.15,
            bboxprops=dict(edgecolor=animal_colour, linewidth=0.8),
            arrowprops=dict(arrowstyle="-", color=muted, linewidth=0.6),
            zorder=1.5,  # under the sight lines and dots
        )
        ax.add_artist(box)

    # keep 1 m east equal to 1 m north at this latitude
    ax.set_aspect(1.0 / np.cos(np.radians(sum(lats) / len(lats))))

    ax.set_title(
        f"{len(observations)} animal observations from {len(drone_lats)} images",
        color=ink,
        loc="left",
    )
    ax.set_xlabel("longitude (deg)", color=muted)
    ax.set_ylabel("latitude (deg)", color=muted)
    ax.tick_params(colors=muted, labelsize=8)
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.grid(True, color="#e4e3df", linewidth=0.6, zorder=0)
    for spine in ax.spines.values():
        spine.set_color("#e4e3df")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=surface)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="directory holding capture folders")
    parser.add_argument(
        "-o", "--out", type=Path, default=Path("observations.png"), help="plot output file"
    )
    parser.add_argument(
        "-f", "--focal", type=float, default=FX, help="camera focal length in pixels"
    )
    args = parser.parse_args()

    root = args.directory
    if not root.is_dir():
        sys.exit(f"not a directory: {root}")

    observations = []
    drone_locations = []
    total = 0
    for folder in find_folders(root):
        pairs = find_pairs(folder)
        if not pairs:
            continue
        print(f"{folder}: {len(pairs)} pair(s)")
        gps_log = load_gps_log(folder)
        for image, meta in pairs:
            index = int(image.stem.split("_")[-1])
            fix = gps_log.get(index)
            telemetry = apply_gps_log(read_telemetry(meta), fix)
            drone_locations.append(telemetry.get("position", {}))

            animals = find_animals(image)
            located_animals = place_animals(animals, meta, args.focal, fix)
            for a in located_animals:
                a["image"] = image  # so the plot can crop a cutout
            observations.extend(located_animals)

            for a in located_animals:
                print(f"  {image.name}  {a['lat']:.7f}, {a['lon']:.7f}  area={a['area']:.0f}")
        total += len(pairs)

    print(f"total: {total} pair(s), {len(observations)} observation(s)")

    if observations:
        plot_observations(drone_locations, observations, args.out)
        print(f"plot written to {args.out}")

    return observations


if __name__ == "__main__":
    main()
