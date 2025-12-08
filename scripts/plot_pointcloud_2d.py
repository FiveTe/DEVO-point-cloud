import argparse
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

PLANE_AXES = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
COLOR_AXES = {"x": 0, "y": 1, "z": 2}
VIRIDIS_STOPS = [
    (68, 1, 84),
    (59, 82, 139),
    (33, 145, 140),
    (94, 201, 98),
    (253, 231, 37),
]


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Create a 2D projection of a DEVO point cloud.")
    parser.add_argument(
        "--pointcloud",
        default="results/corridor_pointcloud_0_40s.npy",
        help="Path to the .npy point cloud file.",
    )
    parser.add_argument(
        "--plane",
        choices=PLANE_AXES.keys(),
        default="xy",
        help="Which plane to project the point cloud onto.",
    )
    parser.add_argument(
        "--color-by",
        choices=list(COLOR_AXES.keys()) + ["none"],
        default="z",
        help="Coordinate used to colorize the scatter plot.",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=1,
        help="Stride for subsampling points before plotting.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=0.9,
        help="Marker radius in pixels for SVG or marker area factor for PNG.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path for saving the figure (extension is set automatically).",
    )
    parser.add_argument(
        "--format",
        choices=["auto", "png", "svg"],
        default="auto",
        help="Force PNG (matplotlib) or SVG (pure Python). Auto picks PNG when matplotlib is available.",
    )
    return parser


def determine_output(format_choice):
    fmt = format_choice.lower()
    if fmt == "auto":
        if plt is not None:
            return "png", True
        return "svg", False
    if fmt == "png":
        if plt is None:
            raise RuntimeError("matplotlib is required for PNG output. Install it or use --format svg.")
        return "png", True
    return "svg", False


def resolve_output_path(pc_path, user_output, plane, extension):
    if user_output:
        out_path = Path(user_output)
        if out_path.suffix.lower() != f".{extension}":
            out_path = out_path.with_suffix(f".{extension}")
    else:
        out_path = Path("results") / f"{pc_path.stem}_{plane}.{extension}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def plot_with_matplotlib(x_vals, y_vals, color_vals, args, pc_path, output_path):
    colors = "k" if color_vals is None else color_vals
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        x_vals,
        y_vals,
        c=colors,
        s=args.point_size,
        cmap="viridis" if color_vals is not None else None,
        linewidths=0,
    )
    plt.xlabel(f"{args.plane[0].upper()} (m)")
    plt.ylabel(f"{args.plane[1].upper()} (m)")
    plt.title(f"{pc_path.stem} - {args.plane.upper()} projection")
    plt.axis("equal")
    plt.grid(True, linestyle="--", linewidth=0.3, alpha=0.5)
    if color_vals is not None:
        cbar = plt.colorbar(scatter, shrink=0.85)
        cbar.set_label(f"{args.color_by.upper()} value (m)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved PNG figure to {output_path}")


def save_svg_plot(x_vals, y_vals, color_vals, args, pc_path, output_path):
    x_vals = np.asarray(x_vals, dtype=np.float64)
    y_vals = np.asarray(y_vals, dtype=np.float64)
    x_min, x_max = x_vals.min(), x_vals.max()
    y_min, y_max = y_vals.min(), y_vals.max()
    x_range = max(x_max - x_min, 1e-9)
    y_range = max(y_max - y_min, 1e-9)

    width = 1200
    height = max(600, int(width * (y_range / x_range))) if x_range > 0 else 600
    padding = 50
    draw_width = width - 2 * padding
    draw_height = height - 2 * padding

    colors, color_limits = compute_colors(color_vals, len(x_vals))

    x_norm = (x_vals - x_min) / x_range
    y_norm = (y_vals - y_min) / y_range
    x_pixels = padding + x_norm * draw_width
    y_pixels = padding + (1.0 - y_norm) * draw_height

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]

    gradient_id = None
    if color_limits is not None:
        gradient_id = "colorbarGradient"
        svg_lines.extend(make_gradient_definition(gradient_id))

    svg_lines.append(
        f'<rect x="{padding}" y="{padding}" width="{draw_width}" height="{draw_height}" fill="none" stroke="#d0d0d0" stroke-width="1"/>'
    )

    radius = max(args.point_size, 0.5)
    for xp, yp, color in zip(x_pixels, y_pixels, colors):
        svg_lines.append(
            f'<circle cx="{xp:.2f}" cy="{yp:.2f}" r="{radius}" fill="{color}" fill-opacity="0.85" stroke="none"/>'
        )

    svg_lines.extend(make_axes_labels(width, height, padding, pc_path, args))

    if color_limits is not None and gradient_id is not None:
        svg_lines.extend(make_colorbar(width, padding, draw_height, gradient_id, color_limits, args.color_by))

    svg_lines.append("</svg>")
    output_path.write_text("\n".join(svg_lines))
    print(f"Saved SVG figure to {output_path}")


def compute_colors(color_vals, count):
    if color_vals is None:
        return ["#2b2b2b"] * count, None
    values = np.asarray(color_vals, dtype=np.float64)
    c_min = values.min()
    c_max = values.max()
    span = max(c_max - c_min, 1e-9)
    normalized = (values - c_min) / span
    color_list = [_viridis_from_unit(val) for val in normalized]
    return color_list, (c_min, c_max)


def _viridis_from_unit(value):
    value = float(np.clip(value, 0.0, 1.0))
    segments = len(VIRIDIS_STOPS) - 1
    pos = value * segments
    idx = min(int(np.floor(pos)), segments - 1)
    frac = pos - idx
    r0, g0, b0 = VIRIDIS_STOPS[idx]
    r1, g1, b1 = VIRIDIS_STOPS[idx + 1]
    r = int(round(r0 + (r1 - r0) * frac))
    g = int(round(g0 + (g1 - g0) * frac))
    b = int(round(b0 + (b1 - b0) * frac))
    return f"#{r:02x}{g:02x}{b:02x}"


def make_gradient_definition(gradient_id):
    lines = [
        "<defs>",
        f'<linearGradient id="{gradient_id}" x1="0%" y1="100%" x2="0%" y2="0%">',
    ]
    stops = len(VIRIDIS_STOPS) - 1
    for idx, (r, g, b) in enumerate(VIRIDIS_STOPS):
        offset = idx / stops * 100 if stops else 0
        lines.append(f'<stop offset="{offset:.1f}%" stop-color="#{r:02x}{g:02x}{b:02x}"/>')
    lines.append("</linearGradient>")
    lines.append("</defs>")
    return lines


def make_colorbar(width, padding, draw_height, gradient_id, limits, color_label):
    bar_height = draw_height * 0.6
    bar_width = 18
    bar_x = width - padding - bar_width
    bar_y = padding + (draw_height - bar_height) / 2
    c_min, c_max = limits
    lines = [
        f'<rect x="{bar_x:.2f}" y="{bar_y:.2f}" width="{bar_width}" height="{bar_height}" fill="url(#{gradient_id})" stroke="#555" stroke-width="0.5"/>' ,
        f'<text x="{bar_x - 6:.2f}" y="{bar_y + bar_height:.2f}" font-size="14" text-anchor="end" fill="#333">{c_min:.2f}</text>',
        f'<text x="{bar_x - 6:.2f}" y="{bar_y + 12:.2f}" font-size="14" text-anchor="end" fill="#333">{c_max:.2f}</text>',
        f'<text x="{bar_x + bar_width / 2:.2f}" y="{bar_y - 8:.2f}" font-size="16" text-anchor="middle" fill="#333">{color_label.upper()} (m)</text>',
    ]
    return lines


def make_axes_labels(width, height, padding, pc_path, args):
    center_x = width / 2
    lines = [
        f'<text x="{center_x:.2f}" y="{padding - 15:.2f}" font-size="20" text-anchor="middle" fill="#111">{pc_path.stem} - {args.plane.upper()} projection</text>',
        f'<text x="{center_x:.2f}" y="{height - padding / 4:.2f}" font-size="16" text-anchor="middle" fill="#333">{args.plane[0].upper()} axis (m)</text>',
        f'<text x="{padding / 3:.2f}" y="{height / 2:.2f}" font-size="16" text-anchor="middle" fill="#333" transform="rotate(-90 {padding / 3:.2f},{height / 2:.2f})">{args.plane[1].upper()} axis (m)</text>',
    ]
    return lines


def plot_pointcloud(args):
    pc_path = Path(args.pointcloud)
    if not pc_path.is_file():
        raise FileNotFoundError(f"Unable to locate point cloud '{pc_path}'.")

    point_cloud = np.load(pc_path)
    if point_cloud.ndim != 2 or point_cloud.shape[1] < 3:
        raise ValueError(f"Expected (N, 3) array but got shape {point_cloud.shape}.")

    if args.downsample > 1:
        point_cloud = point_cloud[:: args.downsample]

    x_idx, y_idx = PLANE_AXES[args.plane]
    x_vals = point_cloud[:, x_idx]
    y_vals = point_cloud[:, y_idx]

    if args.color_by == "none":
        color_vals = None
    else:
        color_vals = point_cloud[:, COLOR_AXES[args.color_by]]

    extension, use_matplotlib = determine_output(args.format)
    output_path = resolve_output_path(pc_path, args.output, args.plane, extension)

    if use_matplotlib:
        plot_with_matplotlib(x_vals, y_vals, color_vals, args, pc_path, output_path)
    else:
        save_svg_plot(x_vals, y_vals, color_vals, args, pc_path, output_path)


if __name__ == "__main__":
    parser = build_arg_parser()
    plot_pointcloud(parser.parse_args())
