#!/usr/bin/env python3
"""Analyze a single capture image and report the objects found in it.

Usage: uv run python analyze.py capture_18/img_3.jpg
UV runner will set up a virtual environment with the dependencies in requirements.txt, and run this script in it. You can also run it directly if you have the dependencies installed.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from color_profiles.color_profile import build_mask, load_profile

PREVIEW_MAX_PX = 1200
DEFAULT_PROFILE = Path(__file__).parent / "color_profiles" / "grass.json"

# the frames are 2464x3280, so a window at full size is taller than the screen
DISPLAY_SCALE = 0.25


def metadata_path(image_path):
    """The json file that belongs to an image."""
    return Path(image_path).with_suffix(".json")


def find_animals(image_path, profile_path=DEFAULT_PROFILE, show=False):
    """Find animals in one image/json pair.

    Everything the background profile cannot explain is a candidate, and the
    candidates are filtered on size and roundness. Returns a list of dicts,
    one per animal. Pass show=True to inspect the mask; it waits for a key,
    so leave it off when walking a whole capture.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise SystemExit(f"could not read image: {image_path}")

    img_reduced = cv2.resize(img, (0, 0), fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)

    grass_profile = load_profile(profile_path)
    # build_mask hands back booleans, findContours wants 0/255 bytes
    grass_mask = build_mask(img_reduced, grass_profile).astype(np.uint8) * 255
    contours, hierarchy = cv2.findContours(grass_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    animals = []
    
    
    for c in contours:
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)

        if perimeter < 120 or area < 300 or area > 2000:
            continue

        circlelike = (4 * np.pi * area) / perimeter**2
        if circlelike < 0.24:
            continue

        M = cv2.moments(c)
        mu12 = M['mu12']
        hu = cv2.HuMoments(M)
    
        cX, cY = 0, 0
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

        a = {}
        a['area'] = area
        a['perimeter'] = perimeter
        a['mu12'] = mu12
        a['hu'] = hu.flatten()
        
        a['cX'] = cX * (1.0/DISPLAY_SCALE)  # scale back to original image coordinates
        a['cY'] = cY * (1.0/DISPLAY_SCALE)  # scale back to original image coordinates
                        
        a['x0'] = int(np.min(c[:, 0, 0]) * (1.0/DISPLAY_SCALE))
        a['y0'] = int(np.min(c[:, 0, 1]) * (1.0/DISPLAY_SCALE))
        a['x1'] = int(np.max(c[:, 0, 0]) * (1.0/DISPLAY_SCALE))
        a['y1'] = int(np.max(c[:, 0, 1]) * (1.0/DISPLAY_SCALE))

        a['width'] = a['x1'] - a['x0']
        a['height'] = a['y1'] - a['y0']
        a['circlelike'] = circlelike              
        a['contour'] = c
        
        animals.append(a)

    if show:
        cv2.imshow("image", img_reduced)
        # nearest keeps the mask binary instead of smearing its edges to grey
        cv2.waitKey(0)
        cv2.imshow("grass_mask", grass_mask)
        cv2.waitKey(0)

        debug = np.zeros_like(img_reduced)
        for a in animals:
            cv2.drawContours(debug, [a['contour']], -1, (0, 255, 0), 1)
            cv2.rectangle(debug, (int(a['x0']*DISPLAY_SCALE), int(a['y0']*DISPLAY_SCALE)), (int(a['x1']*DISPLAY_SCALE), int(a['y1']*DISPLAY_SCALE)), (255, 255, 255), 1)
            cv2.circle(debug, (int(a['cX']*DISPLAY_SCALE), int(a['cY']*DISPLAY_SCALE)), 2, (0, 0, 255), -1)
            #print area, perimeter, circlelike, mu12, hu, cX, cY
            desc = f"area={a['area']:.0f}  perim={a['perimeter']:.0f}  circ={a['circlelike']:.2f}"
            cv2.putText(debug, desc, (int(a['cX']*DISPLAY_SCALE), int(a['cY']*DISPLAY_SCALE)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("outlines", debug)
        cv2.waitKey(0)
        
        cv2.destroyAllWindows()




    return animals

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="image file to analyze")
    parser.add_argument(
        "-p", "--profile", type=Path, default=DEFAULT_PROFILE, help="colour profile to segment with"
    )
    parser.add_argument("--show", action="store_true", help="display image and mask and results")
    args = parser.parse_args()

    image = args.image
    if not image.is_file():
        sys.exit(f"no such image: {image}")
    
    if not args.profile.is_file():
        sys.exit(f"no such profile: {args.profile} — build it with color_profiles/color_profile.py")

    animals = find_animals(image, args.profile, show=args.show)
    print(f"{image}: {len(animals)} animal(s)")
    for animal in animals:
        print(f"  ({animal['cX']}, {animal['cY']})  area={animal['area']:.0f}")


if __name__ == "__main__":
    main()
