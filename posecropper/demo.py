"""Offline end-to-end demo:
    python -m posecropper.demo --image examples/sample.jpg --pose-code M12 --category top
Runs detection -> router -> crop -> writes crops, and (default) renders the visualization
panels to assets/. Core deps only — no GCP."""
import argparse

from posecropper.catalog.static import StaticCategoryProvider
from posecropper.io.local import LocalImageSource, LocalOutputSink
from posecropper.metadata import classify_category, parse_filename
from posecropper.pipeline import process_image
from posecropper.visualize import render_all


def main():
    ap = argparse.ArgumentParser(description="posecropper offline demo")
    ap.add_argument("--image", required=True)
    ap.add_argument("--pose-code", default=None, help="e.g. M12, M20, P3 (overrides filename)")
    ap.add_argument("--category", default=None,
                    help="garment ItemGroupName, e.g. 'T-Shirts', 'Jeans', 'Caps'")
    ap.add_argument("--out", default="out", help="output dir for crops")
    ap.add_argument("--assets", default="assets", help="output dir for visualizations")
    ap.add_argument("--no-viz", action="store_true")
    args = ap.parse_args()

    image_rgb = LocalImageSource().read(args.image)

    meta = parse_filename(args.image)
    pose_code = args.pose_code if args.pose_code is not None else meta["pose"]
    item_group = StaticCategoryProvider(args.category).get_item_group(meta["model"])
    category = classify_category(item_group)

    decision, crops, overlay = process_image(image_rgb, pose_code, category)
    print(f"Router decision: {decision.reason}")

    sink = LocalOutputSink(args.out)
    base = meta["original_file_image"].rsplit(".", 1)[0]
    if overlay is not None:
        sink.write(overlay, f"pose/landmarks_{base}.jpg")
    for name, img in crops.items():
        suffix = f"_{name}" if decision.review else ""
        sub = "review/" if decision.review else ""
        sink.write(img, f"{sub}{base}{suffix}.jpg")
        print(f"  wrote {name} crop ({img.shape[1]}x{img.shape[0]}, ratio {img.shape[0]/img.shape[1]:.2f})")

    if not args.no_viz:
        paths = render_all(image_rgb, pose_code, category, args.assets)
        print(f"Visualizations: {list(paths.values())}")


if __name__ == "__main__":
    main()
