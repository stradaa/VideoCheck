#!/usr/bin/env python3
"""
Compare camera angle/position across .avi files using early-clip motion masks.

- Uses first --duration seconds.
- Builds a low-motion (static) mask to ignore the moving monkey.
- Estimates translation between videos via phase correlation on backgrounds.
- Produces matplotlib plots: per-video diagnostics and pairwise overlays.

Requires: opencv-python, numpy, matplotlib.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# Convenience list so you can run without typing long paths each time.
DEFAULT_VIDEOS = [
    Path("project_test/251030/image_rec010_cam4.avi"),
    Path("project_test/251031/image_rec007_cam4.avi"),
    Path("project_test/251106/image_rec006_cam4.avi")
]


def sample_video(
    path: Path,
    duration_s: float,
    sample_stride: int,
    motion_percentile: float,
    cache_dir: Path | None,
) -> Dict:
    cache_path: Path | None = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_name = (
            f"{path.stem}_dur{duration_s}_stride{sample_stride}_p{motion_percentile}_"
            f"mt{int(path.stat().st_mtime)}.npz"
        )
        cache_path = cache_dir / cache_name
        if cache_path.exists():
            data = np.load(cache_path)
            return {
                "path": path,
                "fps": float(data["fps"]),
                "first_frame": data["first_frame"],
                "median_color": data["median_color"],
                "median_gray": data["median_gray"],
                "mean_motion": data["mean_motion"],
                "static_mask": data["static_mask"].astype(bool),
            }
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    max_frames = int(duration_s * fps)
    frames_color: List[np.ndarray] = []
    frames_gray: List[np.ndarray] = []

    grabbed = 0
    while grabbed < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        grabbed += 1
        if grabbed % sample_stride != 0:
            continue
        frames_color.append(frame)
        frames_gray.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    cap.release()
    if len(frames_gray) < 2:
        raise RuntimeError(f"Not enough frames in {path}")

    frames_gray = [f.astype(np.float32) for f in frames_gray]
    h, w = frames_gray[0].shape
    motion_accum = np.zeros((h, w), dtype=np.float32)
    for prev, curr in zip(frames_gray[:-1], frames_gray[1:]):
        flow = cv2.calcOpticalFlowFarneback(
            prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        motion_accum += mag
    mean_motion = motion_accum / max(1, len(frames_gray) - 1)

    thresh = np.percentile(mean_motion, motion_percentile)
    static_mask = mean_motion <= thresh  # True where background is stable

    median_color = np.median(np.stack(frames_color, axis=0), axis=0).astype(np.uint8)
    median_gray = cv2.cvtColor(median_color, cv2.COLOR_BGR2GRAY).astype(np.float32)

    if cache_path is not None:
        np.savez_compressed(
            cache_path,
            fps=fps,
            first_frame=frames_color[0],
            median_color=median_color,
            median_gray=median_gray,
            mean_motion=mean_motion,
            static_mask=static_mask.astype(np.uint8),
        )

    return {
        "path": path,
        "fps": fps,
        "first_frame": frames_color[0],
        "median_color": median_color,
        "median_gray": median_gray,
        "mean_motion": mean_motion,
        "static_mask": static_mask,
    }


def estimate_shift(ref_gray: np.ndarray, other_gray: np.ndarray) -> Tuple[float, float]:
    # Phase correlation prefers zero-mean.
    ref = (ref_gray - ref_gray.mean()).astype(np.float32)
    other = (other_gray - other_gray.mean()).astype(np.float32)
    (dx, dy), _ = cv2.phaseCorrelate(ref, other)  # returns (x, y)
    return dx, dy


def translate_image(img: np.ndarray, dx: float, dy: float, is_mask: bool = False) -> np.ndarray:
    """Shift image by (dx, dy) pixels. Positive dx moves right, positive dy moves down."""
    h, w = img.shape[:2]
    mat = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    border = cv2.BORDER_REFLECT101
    return cv2.warpAffine(img, mat, (w, h), flags=interp, borderMode=border)


def compute_alignment(ref: Dict, other: Dict) -> Dict[str, np.ndarray | float]:
    dx, dy = estimate_shift(ref["median_gray"], other["median_gray"])

    aligned_color = translate_image(other["median_color"], -dx, -dy)
    aligned_gray = translate_image(other["median_gray"], -dx, -dy)
    aligned_mask = translate_image(
        other["static_mask"].astype(np.uint8), -dx, -dy, is_mask=True
    )
    aligned_mask = aligned_mask > 0.5

    overlap = np.logical_and(ref["static_mask"], aligned_mask)
    union = np.logical_or(ref["static_mask"], aligned_mask)
    iou = overlap.sum() / float(union.sum()) if union.any() else 0.0

    diff = np.abs(ref["median_gray"] - aligned_gray)
    masked_diff = np.where(overlap, diff, np.nan)
    mean_diff = np.nanmean(masked_diff)
    max_diff = np.nanmax(masked_diff)
    coverage = overlap.sum() / float(overlap.size)

    return {
        "dx": dx,
        "dy": dy,
        "aligned_color": aligned_color,
        "aligned_gray": aligned_gray,
        "aligned_mask": aligned_mask,
        "overlap": overlap,
        "masked_diff": masked_diff,
        "mask_iou": iou,
        "coverage": coverage,
        "mean_diff": mean_diff,
        "max_diff": max_diff,
    }


def mask_metrics(ref_mask: np.ndarray, aligned_mask: np.ndarray, trusted: np.ndarray | None = None) -> Dict[str, float]:
    """Compute IoU/overlap limited to a trusted region (bool mask)."""
    if trusted is None:
        trusted = np.ones_like(ref_mask, dtype=bool)
    ref_r = ref_mask & trusted
    aligned_r = aligned_mask & trusted
    overlap = np.logical_and(ref_r, aligned_r)
    union = np.logical_or(ref_r, aligned_r)
    iou = overlap.sum() / float(union.sum()) if union.any() else 0.0
    coverage = overlap.sum() / float(trusted.sum()) if trusted.any() else 0.0
    return {"iou": iou, "coverage": coverage, "overlap": overlap}


def plot_video_diag(stats: Dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(cv2.cvtColor(stats["first_frame"], cv2.COLOR_BGR2RGB))
    axes[0].set_title("First frame")
    axes[1].imshow(cv2.cvtColor(stats["median_color"], cv2.COLOR_BGR2RGB))
    axes[1].set_title("Median background")
    im = axes[2].imshow(stats["mean_motion"], cmap="magma")
    axes[2].set_title("Mean flow magnitude")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    axes[3].imshow(cv2.cvtColor(stats["median_color"], cv2.COLOR_BGR2RGB))
    axes[3].imshow(~stats["static_mask"], cmap="cool", alpha=0.4)
    axes[3].set_title("Dynamic areas (masked out)")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(stats["path"].name)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stats['path'].stem}_diag.png", dpi=200)
    plt.close(fig)


def plot_pairwise(ref: Dict, other: Dict, out_dir: Path) -> Dict[str, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    comp = compute_alignment(ref, other)
    dx, dy = comp["dx"], comp["dy"]
    aligned_color = comp["aligned_color"]
    aligned_gray = comp["aligned_gray"]
    aligned_mask = comp["aligned_mask"]
    # Use the same logic as summary: restrict metrics to pixels static in both videos.
    trusted = np.ones_like(ref["static_mask"], dtype=bool)
    metrics = mask_metrics(ref["static_mask"], aligned_mask, trusted=trusted)
    overlap = metrics["overlap"]
    iou = metrics["iou"]
    coverage = metrics["coverage"]

    diff = np.abs(ref["median_gray"] - aligned_gray)
    masked_diff = np.where(overlap, diff, np.nan)
    mean_diff = np.nanmean(masked_diff)
    max_diff = np.nanmax(masked_diff)

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    axes = axes.ravel()
    axes[0].imshow(cv2.cvtColor(ref["median_color"], cv2.COLOR_BGR2RGB))
    axes[0].imshow(~ref["static_mask"], cmap="cool", alpha=0.35)
    axes[0].set_title(f"Ref: {ref['path'].name}\nCool = dynamic/masked")

    axes[1].imshow(cv2.cvtColor(aligned_color, cv2.COLOR_BGR2RGB))
    axes[1].imshow(~aligned_mask, cmap="cool", alpha=0.35)
    axes[1].set_title(
        f"Other aligned to ref\n(dx={dx:.1f}, dy={dy:.1f})\nCool = dynamic/masked"
    )

    axes[2].imshow(np.logical_xor(ref["static_mask"], aligned_mask), cmap="magma", alpha=0.7)
    axes[2].imshow(overlap, cmap="Greens", alpha=0.5)
    axes[2].set_title(f"Mask agreement (IoU={iou:.3f})\nGreen=overlap, Magenta=mismatch")

    blend = 0.6 * cv2.cvtColor(ref["median_color"], cv2.COLOR_BGR2RGB) + 0.4 * cv2.cvtColor(
        aligned_color, cv2.COLOR_BGR2RGB
    )
    axes[3].imshow(np.clip(blend / 255.0, 0, 1))
    h, w = ref["median_gray"].shape
    axes[3].arrow(
        w / 2,
        h / 2,
        dx,
        dy,
        color="lime",
        width=1,
        head_width=10,
        length_includes_head=True,
    )
    axes[3].set_title("Aligned overlay + shift vector (lime)")

    im = axes[4].imshow(masked_diff, cmap="inferno")
    axes[4].set_title("Abs diff on shared static area\nInferno: higher = bigger change")
    plt.colorbar(im, ax=axes[4], fraction=0.046, pad=0.04)

    axes[5].axis("off")
    axes[5].text(
        0.02,
        0.95,
        "\n".join(
            [
                f"dx, dy (px): {dx:.1f}, {dy:.1f}",
                f"Mask IoU: {iou:.3f}",
                f"Static overlap: {coverage*100:.1f}% of frame",
                f"Mean abs diff (overlap): {mean_diff:.2f}",
                f"Max abs diff (overlap): {max_diff:.2f}",
            ]
        ),
        va="top",
        ha="left",
        fontsize=11,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )

    legend_handles = [
        mpatches.Patch(color="mediumspringgreen", label="Green = static overlap"),
        mpatches.Patch(color="magenta", label="Magenta = mask mismatch"),
        mpatches.Patch(color="lightskyblue", label="Cool mask = dynamic area"),
        mpatches.Patch(color="yellow", label="Inferno heatmap = background diff"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))

    for ax in axes:
        ax.axis("off")

    fig.suptitle(f"{ref['path'].name} vs {other['path'].name}")
    fig.tight_layout()
    fig.savefig(out_dir / f"{ref['path'].stem}_vs_{other['path'].stem}.png", dpi=200)
    plt.close(fig)

    return {
        "ref": ref["path"].name,
        "other": other["path"].name,
        "dx": float(dx),
        "dy": float(dy),
        "mask_iou": float(iou),
        "overlap_pct": float(coverage * 100.0),
        "mean_abs_diff": float(mean_diff),
        "max_abs_diff": float(max_diff),
        "aligned_color": aligned_color,
        "aligned_gray": aligned_gray,
        "aligned_mask": aligned_mask,
        "overlap_mask": overlap,
    }


def plot_summary_grid(ref: Dict, others: List[Dict], out_dir: Path, min_static_count: int) -> None:
    """Create a single figure summarizing ref vs many others, using a consensus static mask."""
    if not others:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    comps = [compute_alignment(ref, other) for other in others]

    # Build consensus "trusted static" mask: pixels that are static in at least min_static_count videos.
    stack_masks = [ref["static_mask"]] + [c["aligned_mask"] for c in comps]
    stack_masks = np.stack(stack_masks, axis=0)
    trusted = stack_masks.sum(axis=0) >= min_static_count

    cols = len(others)
    fig, axes = plt.subplots(2, cols, figsize=(6 * cols, 9))
    if cols == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for idx, (other, comp) in enumerate(zip(others, comps)):
        dx, dy = comp["dx"], comp["dy"]
        aligned_color = comp["aligned_color"]
        aligned_mask = comp["aligned_mask"]

        # Metrics restricted to trusted region
        mask_stats = mask_metrics(ref["static_mask"], aligned_mask, trusted=trusted)
        overlap = mask_stats["overlap"]
        iou = mask_stats["iou"]
        coverage = mask_stats["coverage"]

        diff = np.abs(ref["median_gray"] - comp["aligned_gray"])
        masked_diff = np.where(overlap, diff, np.nan)
        mean_diff = np.nanmean(masked_diff)
        max_diff = np.nanmax(masked_diff)

        # Top row: aligned overlay + shift
        blend = 0.6 * cv2.cvtColor(ref["median_color"], cv2.COLOR_BGR2RGB) + 0.4 * cv2.cvtColor(
            aligned_color, cv2.COLOR_BGR2RGB
        )
        ax_top = axes[0, idx]
        ax_top.imshow(np.clip(blend / 255.0, 0, 1))
        h, w = ref["median_gray"].shape
        ax_top.arrow(
            w / 2,
            h / 2,
            dx,
            dy,
            color="lime",
            width=1,
            head_width=10,
            length_includes_head=True,
        )
        ax_top.set_title(
            f"{other['path'].name}\nshift(lime) dx={dx:.1f}, dy={dy:.1f}\nIoU={iou:.3f}, mean diff={mean_diff:.2f}"
        )
        ax_top.axis("off")

        # Bottom row: mask agreement within trusted area
        ax_bot = axes[1, idx]
        mismatch = np.logical_xor(ref["static_mask"], aligned_mask) & trusted
        color_img = np.zeros((*overlap.shape, 3), dtype=np.float32)
        color_img[overlap] = [0.0, 0.6, 0.0]  # green
        color_img[mismatch] = [1.0, 0.0, 1.0]  # magenta
        ax_bot.imshow(color_img)
        ax_bot.set_title(
            f"Masks (trusted area): Green=overlap, Magenta=mismatch\nOverlap={coverage*100:.1f}%, Max diff={max_diff:.2f}"
        )
        ax_bot.axis("off")

    legend_handles = [
        mpatches.Patch(color="mediumspringgreen", label="Green = static overlap"),
        mpatches.Patch(color="magenta", label="Magenta = mask mismatch"),
        mpatches.Patch(color="lightskyblue", label="Cool mask = dynamic area (on per-video plots)"),
        mpatches.Patch(color="yellow", label="Inferno heatmap = background diff (pair plots)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        f"Summary: {ref['path'].name} vs {len(others)} other day(s)\nConsensus static mask: pixel static in >= {min_static_count} video(s)"
    )
    fig.tight_layout()
    fig.savefig(out_dir / f"summary_{ref['path'].stem}_vs_{len(others)}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare camera stability across .avi files using motion masks."
    )
    p.add_argument(
        "videos",
        nargs="*",
        type=Path,
        help="Paths to .avi files. If omitted, use --use-defaults to run presets.",
    )
    p.add_argument("--duration", type=float, default=10.0, help="Seconds to analyze.")
    p.add_argument(
        "--stride",
        type=int,
        default=3,
        help="Sample every Nth frame to reduce compute (default: 3).",
    )
    p.add_argument(
        "--motion-percentile",
        type=float,
        default=70.0,
        help="Percentile for motion threshold; higher=smaller static region.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("video_compare_out"),
        help="Folder for output plots.",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".video_compare_cache"),
        help="Folder to store/load cached per-video stats (disable with --no-cache).",
    )
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching of per-video stats (forces recompute).",
    )
    p.add_argument(
        "--min-static-count",
        type=int,
        default=2,
        help="Pixel must be static in at least this many videos to count in summary metrics (default: 2).",
    )
    p.add_argument(
        "--use-defaults",
        action="store_true",
        help="Run using the preset DEFAULT_VIDEOS list (edit at top of file).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir: Path = args.out_dir.expanduser().resolve()
    cache_dir: Path | None = None if args.no_cache else args.cache_dir.expanduser().resolve()
    videos: List[Path]
    if args.videos:
        videos = args.videos
    elif args.use_defaults:
        videos = DEFAULT_VIDEOS
    else:
        raise SystemExit("Provide video paths or pass --use-defaults after editing DEFAULT_VIDEOS.")

    stats = [
        sample_video(v, args.duration, args.stride, args.motion_percentile, cache_dir)
        for v in videos
    ]

    for s in stats:
        plot_video_diag(s, out_dir / "per_video")

    # Pairwise comparisons
    pair_metrics = []
    for i in range(len(stats)):
        for j in range(i + 1, len(stats)):
            pair_metrics.append(plot_pairwise(stats[i], stats[j], out_dir / "pairs"))

    # Summary grid: first video as reference vs all others
    if len(stats) > 1:
        plot_summary_grid(stats[0], stats[1:], out_dir / "summary", args.min_static_count)

    if pair_metrics:
        print("Pair metrics (pixels):")
        for m in pair_metrics:
            print(
                f"{m['ref']} vs {m['other']}: dx={m['dx']:.1f}, dy={m['dy']:.1f}, "
                f"IoU={m['mask_iou']:.3f}, overlap={m['overlap_pct']:.1f}%, "
                f"mean|diff|={m['mean_abs_diff']:.2f}, max|diff|={m['max_abs_diff']:.2f}"
            )

    print(f"Wrote diagnostics to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
