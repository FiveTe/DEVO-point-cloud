import argparse
import os
import sys
from typing import Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.export_pointcloud import build_arg_parser, run_export_pointcloud
from scripts.pp_rpg import process_dirs


def preprocess_rosbag(bag_path: str, side: str, delta_ms: Optional[float], output_dir: str) -> str:
    bag_abs = os.path.abspath(bag_path)
    if not os.path.isfile(bag_abs):
        raise FileNotFoundError(f"Bag file '{bag_path}' not found")

    scenedir = os.path.abspath(output_dir or os.path.splitext(bag_abs)[0])
    os.makedirs(scenedir, exist_ok=True)

    # process_dirs expects the .bag file to live beside the target folder
    process_dirs([scenedir], side=side, DELTA_MS=delta_ms)
    return scenedir


def build_rosbag_parser():
    parser = build_arg_parser(datapath_required=False)
    parser.set_defaults(dataset="rpg")
    parser.add_argument("--bag", required=True, help="Path to the input rosbag (.bag)")
    parser.add_argument("--delta-ms", type=float, default=None, help="Optional voxel window duration (overrides per-frame timestamps)")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory where intermediate files will be stored (defaults to bag name)")
    parser.add_argument("--skip-preprocess", action="store_true", help="Assume the bag has already been preprocessed into --output-dir")
    return parser


def main():
    parser = build_rosbag_parser()
    args = parser.parse_args()

    if args.dataset.lower() != "rpg":
        raise ValueError("This wrapper currently supports the RPG dataset format only.")

    datapath = args.output_dir or os.path.splitext(os.path.abspath(args.bag))[0]

    if not args.skip_preprocess:
        datapath = preprocess_rosbag(args.bag, args.side, args.delta_ms, datapath)
    else:
        datapath = os.path.abspath(datapath)
        if not os.path.isdir(datapath):
            raise FileNotFoundError(f"Processed directory '{datapath}' not found. Remove --skip-preprocess or correct the path.")

    args.datapath = datapath
    run_export_pointcloud(args)


if __name__ == "__main__":
    main()
