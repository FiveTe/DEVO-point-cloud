from __future__ import annotations

import argparse
import logging
import os
import shutil
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


def _resolve_export_datapath(indir: str) -> str:
      """Support both <root>/<seq> and <root>/<seq>/<seq> layouts."""
      seq_name = os.path.basename(os.path.normpath(indir))
      nested_seq_dir = os.path.join(indir, seq_name)
      if os.path.isdir(nested_seq_dir):
            return nested_seq_dir
      return indir


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
            "  python scripts/start_pipline.py --indir datasets/robot/ --cleanup-algorithm --export-emvs-mono --run-emvs\n"
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
      parser.add_argument(
            "--export-emvs-mono",
            action="store_true",
            help="If set, create a monocular rpg_emvs input bag+conf from Vector events and DEVO poses.",
      )
      parser.add_argument(
            "--emvs-bag-name",
            default="emvs_input.bag",
            help="Output bag name inside OUTDIR when --export-emvs-mono is enabled (default: emvs_input.bag).",
      )
      parser.add_argument(
            "--emvs-conf-name",
            default="emvs_mono.conf",
            help="Output config name inside OUTDIR when --export-emvs-mono is enabled (default: emvs_mono.conf).",
      )
      parser.add_argument(
            "--emvs-side",
            default=None,
            choices=["left", "right"],
            help="Event side for EMVS conversion. Defaults to --export-side when omitted.",
      )
      parser.add_argument(
            "--emvs-event-topic",
            default="/dvs/events",
            help="Event topic stored in EMVS bag (default: /dvs/events).",
      )
      parser.add_argument(
            "--emvs-camera-info-topic",
            default="/dvs/camera_info",
            help="CameraInfo topic stored in EMVS bag (default: /dvs/camera_info).",
      )
      parser.add_argument(
            "--emvs-pose-topic",
            default="/pose",
            help="Pose topic stored in EMVS bag (default: /pose).",
      )
      parser.add_argument(
            "--emvs-min-depth",
            type=float,
            default=None,
            help="Override EMVS min depth in meters. If omitted, estimated from frame seeds when possible.",
      )
      parser.add_argument(
            "--emvs-max-depth",
            type=float,
            default=None,
            help="Override EMVS max depth in meters. If omitted, estimated from frame seeds when possible.",
      )
      parser.add_argument(
            "--emvs-dimZ",
            type=int,
            default=100,
            help="EMVS depth plane count (default: 100).",
      )
      parser.add_argument(
            "--emvs-t-margin-s",
            type=float,
            default=0.5,
            help="Expand EMVS event window around DEVO pose timestamps by this margin in seconds (default: 0.5).",
      )
      parser.add_argument(
            "--emvs-overwrite",
            action="store_true",
            help="Overwrite existing EMVS bag/conf outputs if they already exist.",
      )
      parser.add_argument(
            "--run-emvs",
            action="store_true",
            help="If set, run rpg_emvs after ensuring EMVS bag/conf exist (rosrun mapper_emvs run_emvs --flagfile=...).",
      )
      args = parser.parse_args()

      indir = os.path.abspath(os.path.normpath(args.indir))
      if not os.path.isdir(indir):
            raise FileNotFoundError(f"--indir does not exist or is not a directory: {indir}")
      foldername = os.path.basename(os.path.normpath(indir))
      outdir = os.path.join("results", foldername)
      os.makedirs(outdir, exist_ok=True)

      logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
      LOG.info("Input folder: %s", indir)
      LOG.info("Folder name: %s", foldername)
      LOG.info("OUTDIR: %s", outdir)

      py = sys.executable
      emvs_bag = os.path.join(outdir, args.emvs_bag_name)
      emvs_conf = os.path.join(outdir, args.emvs_conf_name)

      try:
            # 1) Preprocess (skip if already done for this sequence)
            if args.export_dataset.lower() == "vector" and _vector_preprocessed(indir, args.export_side):
                  LOG.info("Preprocess outputs already exist for %s (side=%s). Skipping pp_vector.", indir, args.export_side)
            else:
                  run([py, "scripts/pp_vector.py", "--indir", indir])

            # 2) Export pointcloud (npy)
            out_npy = os.path.join(outdir, args.outname)
            frame_out = os.path.join(outdir, args.frame_out)
            dataset_path = _resolve_export_datapath(indir)
            LOG.info("Export datapath: %s", dataset_path)
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

            # 5) Optional: build single-bag monocular EMVS inputs.
            if args.export_emvs_mono:
                  emvs_side = args.emvs_side or args.export_side
                  poses_npy = os.path.splitext(out_npy)[0] + "_poses.npy"
                  tstamps_npy = os.path.splitext(out_npy)[0] + "_tstamps.npy"

                  emvs_cmd = [
                        py,
                        "scripts/prepare_vector_emvs_mono.py",
                        "--seq_dir",
                        indir,
                        "--devo_poses_npy",
                        poses_npy,
                        "--devo_tstamps_npy",
                        tstamps_npy,
                        "--out_bag",
                        emvs_bag,
                        "--out_conf",
                        emvs_conf,
                        "--side",
                        emvs_side,
                        "--event_topic",
                        args.emvs_event_topic,
                        "--camera_info_topic",
                        args.emvs_camera_info_topic,
                        "--pose_topic",
                        args.emvs_pose_topic,
                        "--dimZ",
                        str(args.emvs_dimZ),
                        "--t_margin_s",
                        str(args.emvs_t_margin_s),
                  ]
                  if os.path.exists(frame_out):
                        emvs_cmd += ["--frames_npz", frame_out]
                  if args.emvs_min_depth is not None:
                        emvs_cmd += ["--min_depth", str(args.emvs_min_depth)]
                  if args.emvs_max_depth is not None:
                        emvs_cmd += ["--max_depth", str(args.emvs_max_depth)]
                  if args.emvs_overwrite:
                        emvs_cmd.append("--overwrite")
                  run(emvs_cmd)

            # 6) Optional: run mapper_emvs when inputs are available.
            if args.run_emvs:
                  missing = [p for p in [emvs_bag, emvs_conf] if not os.path.exists(p)]
                  if missing:
                        raise FileNotFoundError(
                              "Cannot run EMVS because required files are missing: "
                              + ", ".join(missing)
                              + ". Generate them first with --export-emvs-mono or provide matching names via "
                              "--emvs-bag-name/--emvs-conf-name."
                        )
                  if shutil.which("rosrun") is None:
                        raise FileNotFoundError(
                              "Cannot run EMVS because 'rosrun' is not in PATH. Source your ROS environment first."
                        )
                  run(["rosrun", "mapper_emvs", "run_emvs", f"--flagfile={os.path.abspath(emvs_conf)}"])

            LOG.info("Pipeline finished. Results in %s", outdir)


      except subprocess.CalledProcessError as exc:
            LOG.error("Command failed: %s", exc)
            sys.exit(1)


if __name__ == "__main__":
      main()
