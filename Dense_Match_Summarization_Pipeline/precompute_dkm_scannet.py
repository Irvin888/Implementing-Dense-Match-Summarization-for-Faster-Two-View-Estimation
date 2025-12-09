#!/usr/bin/env python
"""
precompute_dkm_scannet.py

Precompute dense matches using DKMv3_outdoor on ScanNet scenes.

Dataset layout:

    scannet_test_1500/
      sceneXXXX_00/
        color/
          0.jpg, 45.jpg, ...
        depth/
          0.png, 45.png, ...           (unused)
        intrinsic/
          intrinsic_color[.txt]        
          intrinsic_depth              (unused)
          extrinsic_color              (unused)
          extrinsic_depth              (unused)
        pose/
          0.txt, 45.txt, ...           

This script creates:

    matches/
      sceneXXXX_00_000***_000***.txt      
    gt_pose/
      sceneXXXX_00_000***_000***_pose.txt 
    intrinsics/
      sceneXXXX_00_intrinsics.txt         
    pair_list_scannet.txt                 
      Program log

Controls:
    MAX_SCENES          : how many scenes to process (-1 = all)
    MAX_PAIRS_PER_SCENE : how many frame pairs per scene (-1 = all)
    MAX_MATCHES_PER_PAIR: cap on matches per pair (-1 = keep all)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch

# =========================================================================
# USER PARAMETERS (adjust as needed)
# =========================================================================

# How many scenes to process:
#   -1 = all scenes
MAX_SCENES: int = 5

# How many pairs to process per scene:
#   -1 = all consecutive pairs within that scene
MAX_PAIRS_PER_SCENE: int = 6

# Maximum number of matches to keep per image pair:
#   -1 = keep all returned by DKM
MAX_MATCHES_PER_PAIR: int = 2000

# Folder names
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "scannet_test_1500"
DEFAULT_PRECOMP_ROOT = PROJECT_ROOT / "precomputed_dkm"

# Subdirectories created under precomputed_dkm
MATCHES_SUBDIR = "matches"
GTPOSE_SUBDIR = "gt_pose"
INTRINSIC_SUBDIR = "intrinsics"
PAIR_LIST_NAME = "pair_list_scannet.txt"

# =========================================================================
# DKM import and model setup
# =========================================================================

try:
    from dkm import DKMv3_outdoor
except ImportError as e:
    raise ImportError(
        "Could not import DKMv3_outdoor from the 'dkm' package.\n"
    ) from e

_DKM_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_DKM_MODEL = None

def _get_dkm_model():
    """Create (once) and cache the DKM model."""
    global _DKM_MODEL
    if _DKM_MODEL is None:
        print(f"[INFO] Initializing DKMv3_outdoor on {_DKM_DEVICE} ...")
        _DKM_MODEL = DKMv3_outdoor(device=_DKM_DEVICE)
        _DKM_MODEL.eval()
    return _DKM_MODEL

# =========================================================================
# Utilities
# =========================================================================

def discover_scenes(dataset_root: Path) -> List[Path]:
    """Return a sorted list of scene directories under dataset_root."""
    scenes = [p for p in dataset_root.iterdir() if p.is_dir()]
    scenes.sort(key=lambda p: p.name)
    return scenes

def discover_frames(color_dir: Path) -> List[int]:
    """
    Discover numeric frame/image indices under color_dir.

    Files are e.g. "0.jpg", "45.jpg", etc..
    """
    frames: List[int] = []
    for entry in color_dir.iterdir():
        if not entry.is_file():
            continue
        stem = entry.stem
        try:
            idx = int(stem)
        except ValueError:
            continue
        frames.append(idx)

    frames.sort()
    return frames

def pair_frames_consecutive(frames: List[int]) -> List[Tuple[int, int]]:
    """
    Form pairs as (frames[0], frames[1]), (frames[2], frames[3]), ...

    If there's an odd number of frames, drop the last one.
    """
    if len(frames) < 2:
        return []

    if len(frames) % 2 != 0:
        print(f"[WARN] Odd number of frames ({len(frames)}); last frame will be ignored.")

    pairs: List[Tuple[int, int]] = []
    for i in range(0, len(frames) - 1, 2):
        pairs.append((frames[i], frames[i + 1]))
    return pairs

def pad6(idx: int) -> str:
    """Zero-pad an integer to 6 digits: 3 -> '000003'."""
    return f"{idx:06d}"

def match_file_base(scene_id: str, idx1: int, idx2: int) -> str:
    """Base file name used for both matches and GT pose."""
    return f"{scene_id}_{pad6(idx1)}_{pad6(idx2)}.txt"

# =========================================================================
# Pose / intrinsics loading
# =========================================================================

def load_pose_matrix(path: Path) -> np.ndarray:
    """
    Load the pose matrix from file.
    Expecting 16 numbers forming a 4x4 matrix.
    """
    data = np.loadtxt(str(path), dtype=np.float64).reshape(-1)
    if data.size != 16:
        raise RuntimeError(f"Expected 16 numbers in pose file {path}, got {data.size}")
    T = data.reshape(4, 4)
    return T

def load_intrinsics_matrix(path: Path) -> np.ndarray:
    """
    Load the camera intrinsics matrix K from intrinsic_color.
    4x4 matrix with [0 0 0 1] in last row/col -> takes top-left 3x3
    """
    data = np.loadtxt(str(path), dtype=np.float64).reshape(-1)

    if data.size == 16:
        M = data.reshape(4, 4)
        K = M[:3, :3]
    elif data.size == 9: # in case just 3x3 directly
        K = data.reshape(3, 3)
    else:
        raise RuntimeError(
            f"Expected 9 or 16 numbers in intrinsics file {path}, got {data.size}"
        )

    return K

def compute_relative_pose(T1: np.ndarray, T2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute relative pose T_rel = T2 * inv(T1) and extract R (3x3) and t (3,).
    """
    # IMPORTANT! ScanNet T1, T2 are Camera-to-World.
    # T_rel (1->2) = inv(T2) @ T1
    T2_inv = np.linalg.inv(T2)
    T_rel = T2_inv @ T1

    R_rel = T_rel[:3, :3]
    t_rel = T_rel[:3, 3]

    R_rel = T_rel[:3, :3]
    t_rel = T_rel[:3, 3]

    # Normalize translation to unit length (relative direction only).
    norm = np.linalg.norm(t_rel)
    if norm > 1e-9:
        t_rel = t_rel / norm

    return R_rel, t_rel

def find_intrinsic_color_file(intrinsic_dir: Path) -> Path | None:
    """
    Find the intrinsic_color file in a scene's intrinsic/ folder.
      - intrinsic_color.txt
      - anything whose name starts with "intrinsic_color"
    """
    if not intrinsic_dir.exists() or not intrinsic_dir.is_dir():
        return None

    # Exact
    candidates = [
        intrinsic_dir / "intrinsic_color.txt",
    ]
    for c in candidates:
        if c.exists():
            return c

    # Fallback: any file starting with "intrinsic_color"
    for p in intrinsic_dir.iterdir():
        if p.is_file() and p.name.startswith("intrinsic_color"):
            return p

    return None

# =========================================================================
# DKM -> dense matches: (x1, y1, x2, y2, score)
# =========================================================================

def compute_dkm_matches(
    img_path1: Path, img_path2: Path
) -> List[Tuple[float, float, float, float, float]]:
    """
    Run DKMv3_outdoor on the two images and return a list of dense matches.

    Each returned match is:
        (x1, y1, x2, y2, score)
    """
    model = _get_dkm_model()

    # Original image sizes for to_pixel_coordinates
    W1, H1 = Image.open(img_path1).size
    W2, H2 = Image.open(img_path2).size

    warp, certainty_map = model.match(
        str(img_path1), str(img_path2), device=_DKM_DEVICE
    )

    matches_tensor, cert_tensor = model.sample(warp, certainty_map)

    kpts1, kpts2 = model.to_pixel_coordinates(
        matches_tensor, H1, W1, H2, W2
    )

    kpts1_np = kpts1.detach().cpu().numpy()  # [N, 2]
    kpts2_np = kpts2.detach().cpu().numpy()  # [N, 2]
    cert_np = cert_tensor.detach().cpu().numpy().reshape(-1)  # [N]

    num_matches = kpts1_np.shape[0]

    # Subsample: keep at most MAX_MATCHES_PER_PAIR matches (may impact RANSAC accruacy)
    if MAX_MATCHES_PER_PAIR > 0 and num_matches > MAX_MATCHES_PER_PAIR:
        idx_sorted = np.argsort(-cert_np)[:MAX_MATCHES_PER_PAIR]
        kpts1_np = kpts1_np[idx_sorted]
        kpts2_np = kpts2_np[idx_sorted]
        cert_np = cert_np[idx_sorted]
        num_matches = kpts1_np.shape[0]

    results: List[Tuple[float, float, float, float, float]] = []
    for i in range(num_matches):
        x1, y1 = float(kpts1_np[i, 0]), float(kpts1_np[i, 1])
        x2, y2 = float(kpts2_np[i, 0]), float(kpts2_np[i, 1])
        score = float(cert_np[i])
        results.append((x1, y1, x2, y2, score))

    return results

# =========================================================================
# Main function
# =========================================================================

def main() -> None:
    dataset_root = DEFAULT_DATASET_ROOT
    out_root = DEFAULT_PRECOMP_ROOT

    print(f"[INFO] Dataset root:         {dataset_root}")
    print(f"[INFO] Precomputed output:   {out_root}")
    print(f"[INFO] MAX_SCENES = {MAX_SCENES}, "
          f"MAX_PAIRS_PER_SCENE = {MAX_PAIRS_PER_SCENE}")
    print(f"[INFO] MAX_MATCHES_PER_PAIR = {MAX_MATCHES_PER_PAIR}")

    if not dataset_root.exists():
        print(f"[ERROR] dataset_root does not exist: {dataset_root}")
        return
    if not dataset_root.is_dir():
        print(f"[ERROR] dataset_root is not a directory: {dataset_root}")
        return

    matches_root = out_root / MATCHES_SUBDIR
    gtpose_root = out_root / GTPOSE_SUBDIR
    intrinsic_root = out_root / INTRINSIC_SUBDIR

    matches_root.mkdir(parents=True, exist_ok=True)
    gtpose_root.mkdir(parents=True, exist_ok=True)
    intrinsic_root.mkdir(parents=True, exist_ok=True)

    scenes = discover_scenes(dataset_root)
    if not scenes:
        print(f"[ERROR] No scene directories found under {dataset_root}")
        return

    if MAX_SCENES > 0:
        scenes = scenes[:MAX_SCENES]

    print(f"[INFO] Found {len(scenes)} scene(s) to process.")

    pair_list_path = out_root / PAIR_LIST_NAME
    pair_list_lines: List[str] = []
    total_pairs = 0

    for scene_idx, scene_dir in enumerate(scenes):
        scene_id = scene_dir.name
        print(f"\n[INFO] ===== Scene {scene_idx+1}/{len(scenes)}: {scene_id} =====")

        color_dir = scene_dir / "color"
        pose_dir = scene_dir / "pose"
        intrinsic_dir = scene_dir / "intrinsic"

        if not color_dir.exists() or not color_dir.is_dir():
            print(f"[WARN] Scene {scene_id} missing color/, skipping.")
            continue
        if not pose_dir.exists() or not pose_dir.is_dir():
            print(f"[WARN] Scene {scene_id} missing pose/, skipping.")
            continue

        # Intrinsics preprocessing (once per scene)
        intrinsic_file = find_intrinsic_color_file(intrinsic_dir)
        intrinsic_rel_path = "NONE"

        if intrinsic_file is not None:
            print(f"[INFO] Found intrinsic_color file for {scene_id}: {intrinsic_file.name}")
            try:
                K = load_intrinsics_matrix(intrinsic_file)
                flat_K = K.reshape(-1)
                intrinsic_out = intrinsic_root / f"{scene_id}_intrinsics.txt"
                with intrinsic_out.open("w") as f:
                    f.write(" ".join(str(x) for x in flat_K))
                    f.write("\n")
                intrinsic_rel_path = f"{intrinsic_root.name}/{intrinsic_out.name}"
                print(f"[INFO] Wrote intrinsics for {scene_id} -> {intrinsic_out}")
            except Exception as e:
                print(f"[WARN] Failed to read intrinsics for {scene_id}: {e}")
        else:
            print(f"[WARN] No intrinsic_color file found for {scene_id} in {intrinsic_dir}")

        # Frame discovery and pairing
        frames = discover_frames(color_dir)
        if not frames:
            print(f"[WARN] Scene {scene_id} has no numeric frames, skipping.")
            continue

        frame_pairs = pair_frames_consecutive(frames)
        if MAX_PAIRS_PER_SCENE > 0:
            frame_pairs = frame_pairs[:MAX_PAIRS_PER_SCENE]

        if not frame_pairs:
            print(f"[WARN] Scene {scene_id} produced no frame pairs, skipping.")
            continue

        print(f"[INFO] Scene {scene_id}: {len(frame_pairs)} pair(s) to process.")

        # Process each pair
        for pair_idx, (f1, f2) in enumerate(frame_pairs):
            print(f"[INFO]  Pair {pair_idx+1}/{len(frame_pairs)}: frames {f1}, {f2}")

            img1 = color_dir / f"{f1}.jpg"
            img2 = color_dir / f"{f2}.jpg"

            if not img1.exists() or not img2.exists():
                print(f"[WARN] Missing image(s) for pair {scene_id} {f1},{f2}, skipping.")
                continue

            base_name = match_file_base(scene_id, f1, f2)
            match_path = matches_root / base_name
            gtpose_path = gtpose_root / base_name.replace(".txt", "_pose.txt")

            # Dense matches via DKM
            print(f"[INFO]     Computing DKM matches for {scene_id}, "
                  f"frames {f1} & {f2} ...")
            try:
                matches = compute_dkm_matches(img1, img2)
            except Exception as e:
                print(f"[WARN]     DKM failed for {scene_id} {f1}-{f2}: {e}")
                matches = []

            if not matches:
                print(f"[WARN]     No matches returned "
                      f"for {scene_id} {f1}-{f2} (writing empty file).")

            with match_path.open("w") as f:
                for x1, y1, x2, y2, score in matches:
                    f.write(f"{x1} {y1} {x2} {y2} {score}\n")

            print(f"[INFO]     Wrote {len(matches)} matches to {match_path}")

            # Ground-truth relative pose
            pose1 = pose_dir / f"{f1}.txt"
            pose2 = pose_dir / f"{f2}.txt"

            if not pose1.exists() or not pose2.exists():
                print(f"[WARN]     Missing pose file(s) for {scene_id} "
                      f"{f1},{f2}, skipping GT pose.")
            else:
                try:
                    T1 = load_pose_matrix(pose1)
                    T2 = load_pose_matrix(pose2)
                    R_rel, t_rel = compute_relative_pose(T1, T2)

                    flat_R = R_rel.reshape(-1)

                    with gtpose_path.open("w") as f:
                        # 9 R entries then 3 t entries on one line
                        f.write(" ".join(str(x) for x in flat_R))
                        f.write(" ")
                        f.write(" ".join(str(x) for x in t_rel))
                        f.write("\n")

                    print(f"[INFO]     Wrote GT relative pose to {gtpose_path}")
                except Exception as e:
                    print(f"[WARN]     Failed to compute GT pose for "
                          f"{scene_id} {f1}-{f2}: {e}")

            # Log this pair in the pair list
            pair_list_lines.append(
                f"{scene_id} {f1} {f2} "
                f"{matches_root.name}/{base_name} "
                f"{gtpose_root.name}/{gtpose_path.name} "
                f"{intrinsic_rel_path}\n"
            )
            total_pairs += 1

    # Write global pair list
    if pair_list_lines:
        with pair_list_path.open("w") as f:
            f.writelines(pair_list_lines)
        print(f"\n[INFO] Wrote pair list to {pair_list_path}")

    print(f"[INFO] Done. Processed {total_pairs} pair(s).")


if __name__ == "__main__":
    main()
