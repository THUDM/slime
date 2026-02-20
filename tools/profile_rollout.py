import argparse
import sys

import requests


def get_workers(router_url):
    try:
        response = requests.get(f"{router_url}/workers")
        response.raise_for_status()
        return response.json().get("workers", [])
    except Exception as e:
        print(f"Error fetching workers from router: {e}")
        return []


def start_profile(worker_url, args):
    payload = {
        "output_dir": args.output_dir,
        "num_steps": args.num_steps,
        "activities": args.activities,
        "profile_by_stage": args.profile_by_stage,
        "with_stack": args.with_stack,
        "record_shapes": args.record_shapes,
    }
    try:
        print(f"Starting profile on {worker_url} for {args.num_steps} steps...")
        response = requests.post(f"{worker_url}/start_profile", json=payload)
        response.raise_for_status()
        print(f"Successfully started profile on {worker_url}")
    except Exception as e:
        print(f"Failed to start profile on {worker_url}: {e}")


def stop_profile(worker_url):
    try:
        print(f"Stopping profile on {worker_url}...")
        response = requests.post(f"{worker_url}/stop_profile", json={})
        response.raise_for_status()
        print(f"Successfully stopped profile on {worker_url}")
    except Exception as e:
        print(f"Failed to stop profile on {worker_url}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Automate SGLang profiling across all workers via router.")
    parser.add_argument("--router-url", type=str, required=True, help="Router URL (e.g., http://127.0.0.1:3000)")
    parser.add_argument("--action", type=str, choices=["start", "stop"], default="start", help="Action to perform")
    parser.add_argument("--output-dir", type=str, default="/tmp/sglang_profile", help="Output directory for traces")
    parser.add_argument("--num-steps", type=int, default=3, help="Number of steps to profile (default: 3)")
    parser.add_argument("--activities", type=str, nargs="+", default=["GPU"], help="Activities to profile (CPU, GPU)")
    parser.add_argument("--profile-by-stage", action="store_true", help="Profile by stage (prefill/decode)")
    parser.add_argument("--with-stack", action="store_true", help="Record call stack")
    parser.add_argument("--record-shapes", action="store_true", help="Record tensor shapes")

    args = parser.parse_args()

    workers = get_workers(args.router_url)
    if not workers:
        print("No workers found. Ensure the router is running and workers are registered.")
        sys.exit(1)

    print(f"Found {len(workers)} workers.")

    for worker in workers:
        worker_url = worker.get("url")
        if not worker_url:
            continue

        if args.action == "start":
            start_profile(worker_url, args)
        else:
            stop_profile(worker_url)


if __name__ == "__main__":
    main()
