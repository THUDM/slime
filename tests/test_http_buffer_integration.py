#!/usr/bin/env python3
"""
HTTP Buffer Integration Test Script

This script tests the HTTP buffer integration with GRPO training.
It performs basic validation without requiring a full training run.

Usage:
    python test_http_buffer_integration.py [--server-url http://localhost:8889]

Prerequisites:
    1. Start the HTTP buffer server:
       cd slime_plugins/rollout_buffer && python buffer.py
    2. Run this test script to validate the setup
"""

import argparse
import sys
import time
from typing import List, Dict, Any

try:
    import requests
except ImportError:
    print("ERROR: requests library not found. Install with: pip install requests")
    sys.exit(1)


class HTTPBufferTester:
    """Test the HTTP buffer integration"""

    def __init__(self, server_url: str):
        self.server_url = server_url.rstrip('/')
        self.test_results = []

    def run_all_tests(self) -> bool:
        """Run all integration tests"""
        print("="*60)
        print("HTTP Buffer Integration Test")
        print("="*60)
        print(f"Server URL: {self.server_url}")
        print("="*60)
        print()

        tests = [
            ("Health Check", self.test_health_check),
            ("Server Info", self.test_server_info),
            ("Write Sample", self.test_write_sample),
            ("Read Samples", self.test_read_samples),
            ("GRPO Group Write", self.test_grpo_group_write),
            ("GRPO Group Read", self.test_grpo_group_read),
        ]

        all_passed = True
        for test_name, test_func in tests:
            print(f"Running: {test_name}...", end=" ")
            try:
                test_func()
                print("✓ PASSED")
                self.test_results.append((test_name, True, None))
            except Exception as e:
                print(f"✗ FAILED: {e}")
                self.test_results.append((test_name, False, str(e)))
                all_passed = False
            print()

        self.print_summary()
        return all_passed

    def test_health_check(self):
        """Test if the server is responding"""
        response = requests.get(f"{self.server_url}/", timeout=5)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert "status" in data, "Response missing 'status' field"
        assert data["status"] == "healthy", f"Server status: {data['status']}"

    def test_server_info(self):
        """Test server info endpoint"""
        response = requests.get(f"{self.server_url}/", timeout=5)
        data = response.json()
        print(f"\n    Server Info: {data}")

    def test_write_sample(self):
        """Test writing a single sample"""
        sample_data = {
            "instance_id": "test_sample_1",
            "prompt": "What is 2+2?",
            "response": "4",
            "reward": 1.0,
            "policy_version": 0,
            "response_length": 10,
            "timestamp": time.time(),
        }

        response = requests.post(
            f"{self.server_url}/buffer/write",
            json=sample_data,
            timeout=5
        )
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert data.get("success", False), f"Write failed: {data.get('message')}"

    def test_read_samples(self):
        """Test reading samples"""
        response = requests.post(
            f"{self.server_url}/get_rollout_data",
            json={"num_samples": 1},
            timeout=5
        )
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        # It's OK if no data is available, just check the format
        assert "success" in data, "Response missing 'success' field"
        assert "data" in data, "Response missing 'data' field"

    def test_grpo_group_write(self):
        """Test writing a complete GRPO group"""
        # Simulate writing a group of samples with same instance_id
        instance_id = f"grpo_group_test_{int(time.time())}"

        for i in range(8):  # n_samples_per_prompt = 8
            sample_data = {
                "instance_id": instance_id,
                "prompt": "Test prompt for GRPO",
                "response": f"Response {i}",
                "reward": 0.5 + i * 0.1,
                "policy_version": 0,
                "response_length": 20 + i,
                "timestamp": time.time(),
                "tokens": list(range(100)),
                "index": i,
            }

            response = requests.post(
                f"{self.server_url}/buffer/write",
                json=sample_data,
                timeout=5
            )
            assert response.status_code == 200, f"Failed to write sample {i}"

        print(f"\n    Wrote 8 samples for instance_id: {instance_id}")

    def test_grpo_group_read(self):
        """Test reading GRPO groups"""
        # Request 8 samples (one complete group)
        response = requests.post(
            f"{self.server_url}/get_rollout_data",
            json={"num_samples": 8},
            timeout=5
        )
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()

        if data.get("success") and data.get("data", {}).get("data"):
            samples = data["data"]["data"]
            print(f"\n    Read {len(samples)} samples")

            # Check if samples are grouped correctly
            instance_ids = set(s.get("instance_id") for s in samples)
            print(f"    Unique instance_ids: {len(instance_ids)}")

    def print_summary(self):
        """Print test summary"""
        print("="*60)
        print("Test Summary")
        print("="*60)

        passed = sum(1 for _, result, _ in self.test_results if result)
        total = len(self.test_results)

        for test_name, result, error in self.test_results:
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"  {status:8} {test_name}")
            if error:
                print(f"           Error: {error}")

        print("="*60)
        print(f"Results: {passed}/{total} tests passed")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Test HTTP Buffer Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with default server URL
  python test_http_buffer_integration.py

  # Test with custom server URL
  python test_http_buffer_integration.py --server-url http://192.168.1.100:8889

Prerequisites:
  1. Start the HTTP buffer server:
     cd slime_plugins/rollout_buffer && python buffer.py
  2. Run this test script
        """
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://localhost:8889",
        help="URL of the HTTP buffer server (default: http://localhost:8889)"
    )

    args = parser.parse_args()

    # Check if server is reachable
    print(f"Checking if server is reachable at {args.server_url}...")
    try:
        response = requests.get(args.server_url, timeout=5)
        print(f"✓ Server is responding (status: {response.status_code})")
        print()
    except requests.exceptions.RequestException as e:
        print(f"✗ ERROR: Cannot connect to server at {args.server_url}")
        print(f"  Error: {e}")
        print()
        print("Please make sure the HTTP buffer server is running:")
        print("  cd slime_plugins/rollout_buffer && python buffer.py")
        print()
        sys.exit(1)

    # Run tests
    tester = HTTPBufferTester(args.server_url)
    success = tester.run_all_tests()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
