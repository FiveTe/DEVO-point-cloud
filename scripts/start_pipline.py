from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys

LOG = logging.getLogger("start_pipline")

def _vector_preprocessed(seq_dir: str, side: str) -> bool:
      """Check for Vector preprocessing outputs needed by export_pointcloud.py."""
      required = [
            os.path.join(seq_dir, f"rectify_map_{side}.h5"),
            os.path.join(seq_dir, f"calib_undist_evs_{side}.txt"),
            os.path.join(seq_dir, f"tss_imgs_us_{side}.txt"),
      ]
      return all(os.path.exists(p) for p in required)


def run(cmd: list[str]) -> None:
      LOG.info("Running: %s", " ".join(cmd))
      # Stream combined stdout/stderr to terminal while capturing it for diagnostics
      proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
      )

      output_lines: list[str] = []
      if proc.stdout is not None:
            for raw_line in proc.stdout:
                  # print live to the user's terminal
                  print(raw_line, end="")
                  output_lines.append(raw_line)

      proc.wait()
      output = "".join(output_lines)
      if proc.returncode != 0:
            LOG.error("Command failed: %s", " ".join(cmd))
            LOG.error("Return code: %s", proc.returncode)
            if output:
                  LOG.error("Output:\n%s", output)
            LOG.error("Working directory: %s", os.getcwd())
            raise subprocess.CalledProcessError(proc.returncode, cmd, output=output)


def main() -> None:
      EPILOG = (
            "Examples:\n"
            "  python scripts/start_pipline.py --indir datasets/robot/\n"
            "  python scripts/start_pipline.py --indir datasets/robot/ --export-side right --cleanup-algorithm statistical\n"
            "  # pass custom config and weights\n"
            "  python scripts/start_pipline.py --indir datasets/robot/ --config config/eval_vector_gradient.yaml --weights \\n+DEVO.pth\n"
      )

      parser = argparse.ArgumentParser(
            description="Run DEVO pointcloud pipeline for an input folder",
            epilog=EPILOG,
            formatter_class=argparse.RawTextHelpFormatter,
      )
      parser.add_argument(
            "--indir",
            required=True,
            help="Input dataset folder (e.g. datasets/robot/). Trailing slash is optional.",
      )
      parser.add_argument(
            "--weights",
            default="DEVO.pth",
            help="Path to model weights used by export_pointcloud.py (default: DEVO.pth)",
      )
      parser.add_argument(
            "--config",
            default="config/eval_vector_gradient.yaml",
            help="Config file passed to export_pointcloud.py (default: config/eval_vector_gradient.yaml)",
      )
      parser.add_argument(
            "--outname",
            default="pointcloud.npy",
            help="Name of the exported numpy pointcloud file placed in OUTDIR (default: pointcloud.npy)",
      )
      parser.add_argument(
            "--frame_out",
            default="frames.npz",
            help="Filename for exported per-frame data placed in OUTDIR (default: frames.npz)",
      )
      parser.add_argument(
            "--export-side",
            default="left",
            choices=["left", "right"],
            help="Which camera side to export (forwarded to export_pointcloud.py --side). Options: left|right (default: left)",
      )
      parser.add_argument(
            "--export-dataset",
            default="vector",
            help="Dataset name forwarded to export_pointcloud.py --dataset (default: vector)",
      )
      parser.add_argument(
            "--export_edge_cloud",
            action="store_true",
            help="If set, also export an edge-aligned point cloud (forwarded to export_pointcloud.py --export_edge_cloud).",
      )
      parser.add_argument(
            "--edge_topk",
            type=int,
            default=6000,
            help="Edge pixels per keyframe at quarter-res (forwarded to export_pointcloud.py --edge_topk).",
      )
      parser.add_argument(
            "--edge_border",
            type=int,
            default=2,
            help="Border suppression at quarter-res (forwarded to export_pointcloud.py --edge_border).",
      )
      parser.add_argument(
            "--edge_knn",
            type=int,
            default=4,
            help="kNN for depth interpolation (forwarded to export_pointcloud.py --edge_knn).",
      )
      parser.add_argument(
            "--edge_max_dist",
            type=float,
            default=6.0,
            help="Max kNN radius in quarter-res pixels (forwarded to export_pointcloud.py --edge_max_dist).",
      )
      parser.add_argument(
            "--cleanup-algorithm",
            default="ror",
            help=(
                  "Cleanup algorithm forwarded to cleanup_pointcloud.py --algorithm. Common options: ror, sor;"
                  " default: ror"
            ),
      )
      # Parameters forwarded to scripts/cleanup_pointcloud.py
      parser.add_argument(
            "--cleanup-output",
            default=None,
            help="Output filename for cleaned pointcloud (overrides default naming). If omitted, uses <input>_cleaned.ply",
      )
      # SOR params
      parser.add_argument(
            "--nb_neighbors",
            type=int,
            default=20,
            help="[SOR] Number of neighbors to analyze for each point (default: 20)",
      )
      parser.add_argument(
            "--std_ratio",
            type=float,
            default=2.0,
            help="[SOR] Standard deviation ratio for SOR (default: 2.0)",
      )
      # ROR params
      parser.add_argument(
            "--radius",
            type=float,
            default=0.05,
            help="[ROR] Radius to search for neighbors (default: 0.05)",
      )
      parser.add_argument(
            "--min_neighbors",
            type=int,
            default=10,
            help="[ROR] Minimum number of neighbors required within the radius (default: 16)",
      )
      parser.add_argument(
            "--display",
            action="store_true",
            help="If set, open a visualization window after cleaning (forwarded to cleanup_pointcloud.py --display)",
      )
      parser.add_argument(
            "--verbose",
            "-v",
            action="store_true",
            help="Enable verbose output for cleanup (forwarded to cleanup_pointcloud.py --verbose / -v)",
      )
      args = parser.parse_args()

      indir = args.indir
      foldername = os.path.basename(os.path.normpath(indir))
      outdir = os.path.join("results", foldername)
      os.makedirs(outdir, exist_ok=True)

      logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
      LOG.info("Input folder: %s", indir)
      LOG.info("Folder name: %s", foldername)
      LOG.info("OUTDIR: %s", outdir)

      py = sys.executable

      try:
            # 1) Preprocess (skip if already done for this sequence)
            if args.export_dataset.lower() == "vector" and _vector_preprocessed(indir, args.export_side):
                  LOG.info("Preprocess outputs already exist for %s (side=%s). Skipping pp_vector.", indir, args.export_side)
            else:
                  run([py, "scripts/pp_vector.py", "--indir", indir])

            # 2) Export pointcloud (npy)
            out_npy = os.path.join(outdir, args.outname)
            frame_out = os.path.join(outdir, args.frame_out)
            dataset_subfolder = os.path.basename(os.path.normpath(indir))
            dataset_path = f"{indir}{dataset_subfolder}"
            export_cmd = [
                  py,
                  "scripts/export_pointcloud.py",
                  "--config",
                  args.config,
                  "--datapath",
                  dataset_path,
                  "--weights",
                  args.weights,
                  "--dataset",
                  args.export_dataset,
                  "--side",
                  args.export_side,
                  "--out",
                  out_npy,
                  "--export_frame_data",
                  "--frame_data_out",
                  frame_out,
            ]
            if args.export_edge_cloud:
                  export_cmd += [
                        "--export_edge_cloud",
                        "--edge_topk",
                        str(args.edge_topk),
                        "--edge_border",
                        str(args.edge_border),
                        "--edge_knn",
                        str(args.edge_knn),
                        "--edge_max_dist",
                        str(args.edge_max_dist),
                  ]
            run(export_cmd)

            # 3) Convert npy -> ply (script writes .ply next to the npy)
            run([py, "scripts/npy2ply.py", out_npy])
            if args.export_edge_cloud:
                  out_edges_npy = os.path.splitext(out_npy)[0] + "_edges.npy"
                  if os.path.exists(out_edges_npy):
                        run([py, "scripts/npy2ply.py", out_edges_npy])

            # assume the ply filename is the same base with .ply
            out_ply = os.path.splitext(out_npy)[0] + ".ply"

            # 4) Cleanup pointcloud
            cleaned_ply = os.path.join(outdir, "pointcloud_cleaned.ply")
            # allow override of cleaned output filename
            if args.cleanup_output:
                  cleaned_ply = os.path.join(outdir, args.cleanup_output)

            cleanup_cmd = [
                  py,
                  "scripts/cleanup_pointcloud.py",
                  "--input_file",
                  out_ply,
                  "--algorithm",
                  args.cleanup_algorithm,
                  "--output_file",
                  cleaned_ply,
                  "--nb_neighbors",
                  str(args.nb_neighbors),
                  "--std_ratio",
                  str(args.std_ratio),
                  "--radius",
                  str(args.radius),
                  "--min_neighbors",
                  str(args.min_neighbors),
            ]
            if args.display:
                  cleanup_cmd.append("--display")
            if args.verbose:
                  cleanup_cmd.append("--verbose")
            run(cleanup_cmd)

            LOG.info("Pipeline finished. Results in %s", outdir)

      except subprocess.CalledProcessError as exc:
            LOG.error("Command failed: %s", exc)
            sys.exit(1)


if __name__ == "__main__":
      main()
