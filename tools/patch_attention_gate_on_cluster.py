#!/usr/bin/env python3
"""
Patch Megatron-LM attention.py on all Ray cluster nodes.

Fix: When num_query_groups < tp_size (e.g., GQA with num_kv_heads < tp_size),
the gate tensor in SelfAttention.get_query_key_value_tensors() needs the same
TP rank indexing as query, otherwise gate.shape != core_attn_out.shape in
_apply_output_gate().

Usage:
    python patch_attention_gate_on_cluster.py              # apply patch
    python patch_attention_gate_on_cluster.py --rollback   # restore from backup
    python patch_attention_gate_on_cluster.py --diagnose   # show target lines on cluster
"""

import argparse
import subprocess
import textwrap

import ray

FILE_PATH = "/root/Megatron-LM/megatron/core/transformer/attention.py"
REMOTE_SCRIPT_PATH = "/tmp/_patch_attention_gate.py"

DIAGNOSE_SCRIPT = textwrap.dedent(
    f"""\
    import sys
    FILE_PATH = {FILE_PATH!r}
    try:
        with open(FILE_PATH, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print("FILE_NOT_FOUND")
        sys.exit(0)
    # Search for the "if output_gate:" block in get_query_key_value_tensors
    # We look for the pattern: gate.reshape(...) followed by return query, key, value, gate
    found = False
    for i, line in enumerate(lines):
        stripped = line.rstrip()
        if "gate = gate.reshape(*gate.shape[:2], -1, self.hidden_size_per_attention_head)" in stripped:
            # Print context: 3 lines before, this line, and 3 lines after
            start = max(0, i - 3)
            end = min(len(lines), i + 4)
            print(f"LINE {{i+1}} (0-indexed {{i}}):")
            for j in range(start, end):
                marker = ">>>" if j == i else "   "
                print(f"  {{marker}} {{j+1}}: {{lines[j].rstrip()}}")
            found = True
            break
    if not found:
        # Fallback: search for any line with "gate.reshape" and "hidden_size_per_attention_head"
        for i, line in enumerate(lines):
            if "gate.reshape" in line and "hidden_size_per_attention_head" in line:
                start = max(0, i - 3)
                end = min(len(lines), i + 4)
                print(f"LINE {{i+1}} (0-indexed {{i}}) [fallback match]:")
                for j in range(start, end):
                    marker = ">>>" if j == i else "   "
                    print(f"  {{marker}} {{j+1}}: {{lines[j].rstrip()}}")
                found = True
                break
    if not found:
        print("PATTERN_NOT_FOUND")
        # Show all lines containing "output_gate" for debugging
        for i, line in enumerate(lines):
            if "output_gate" in line or "gate.reshape" in line:
                print(f"  {{i+1}}: {{line.rstrip()}}")
"""
)

# Patch script: uses robust line-by-line approach instead of string matching
PATCH_SCRIPT = textwrap.dedent(
    f"""\
    import sys, shutil

    FILE_PATH = {FILE_PATH!r}

    with open(FILE_PATH, "r") as f:
        lines = f.readlines()

    # Check if already patched: look for the TP indexing line we add
    already_patched = any(
        "gate needs the same TP rank indexing" in line
        for line in lines
    )
    if already_patched:
        print("ALREADY_PATCHED")
        sys.exit(0)

    # Find the target line: "gate = gate.reshape(*gate.shape[:2], -1, self.hidden_size_per_attention_head)"
    # in the output_gate block of get_query_key_value_tensors
    target_idx = None
    for i, line in enumerate(lines):
        stripped = line.rstrip()
        if "gate = gate.reshape(*gate.shape[:2], -1, self.hidden_size_per_attention_head)" in stripped:
            # Verify this is in the output_gate block by checking the next line
            # should be "return query, key, value, gate"
            next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
            if "return query, key, value, gate" in next_line:
                target_idx = i
                break

    if target_idx is None:
        # Fallback: find any gate.reshape ... followed by return query, key, value, gate
        for i, line in enumerate(lines):
            if "gate.reshape" in line and "hidden_size_per_attention_head" in line:
                next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
                if "return query, key, value, gate" in next_line:
                    target_idx = i
                    break

    if target_idx is None:
        print("TARGET_NOT_FOUND")
        sys.exit(0)

    # Determine the indentation of the reshape line
    indent = ""
    for ch in lines[target_idx]:
        if ch in (" ", "\\t"):
            indent += ch
        else:
            break

    # Build the patch lines (same indent, one level deeper for the if block)
    inner_indent = indent + "    "
    patch_lines = [
        "\\n",
        indent + "if self.config.num_query_groups < self.world_size:\\n",
        inner_indent + "# When num_kv_heads < tp_size, gate needs the same TP rank indexing\\n",
        inner_indent + "# as query (see lines above for query indexing logic).\\n",
        inner_indent + "idx = get_tensor_model_parallel_rank() % (\\n",
        inner_indent + "    self.world_size // self.config.num_query_groups\\n",
        inner_indent + ")\\n",
        inner_indent + "size = self.num_attention_heads_per_partition // (\\n",
        inner_indent + "    self.world_size // self.config.num_query_groups\\n",
        inner_indent + ")\\n",
        inner_indent + "gate = gate[:, :, idx * size : (idx + 1) * size, :]\\n",
    ]

    # Backup
    backup_path = FILE_PATH + ".gate_fix.bak"
    shutil.copy2(FILE_PATH, backup_path)
    print(f"BACKUP: {{backup_path}}")

    # Insert patch lines after the reshape line (before the return line)
    new_lines = lines[: target_idx + 1] + patch_lines + lines[target_idx + 1 :]
    with open(FILE_PATH, "w") as f:
        f.writelines(new_lines)
    print("PATCHED")
"""
)

ROLLBACK_SCRIPT = textwrap.dedent(
    f"""\
    import sys, shutil, os

    FILE_PATH = {FILE_PATH!r}
    backup_path = FILE_PATH + ".gate_fix.bak"

    if not os.path.exists(backup_path):
        print("NO_BACKUP_FOUND")
        sys.exit(0)

    shutil.copy2(backup_path, FILE_PATH)
    print("ROLLED_BACK")
"""
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollback", action="store_true", help="Restore from backup")
    parser.add_argument("--diagnose", action="store_true", help="Show target lines on cluster (no patch)")
    args = parser.parse_args()

    if args.diagnose:
        script_body = DIAGNOSE_SCRIPT
    elif args.rollback:
        script_body = ROLLBACK_SCRIPT
    else:
        script_body = PATCH_SCRIPT

    ray.init(address="auto")

    nodes = [n["NodeManagerAddress"] for n in ray.nodes() if n["Alive"]]
    print(f"Found {len(nodes)} alive nodes")

    # Only check one node for diagnose (they should all be the same)
    target_nodes = nodes[:1] if args.diagnose else nodes

    tasks = []
    for node_ip in target_nodes:

        @ray.remote(resources={f"node:{node_ip}": 0.001})
        def run_on_node(node_ip=node_ip):
            # Step 1: write script to temp file
            write_cmd = ["python3", "-c", f"open({REMOTE_SCRIPT_PATH!r},'w').write({script_body!r})"]
            r1 = subprocess.run(write_cmd, capture_output=True, text=True, timeout=30)
            if r1.returncode != 0:
                return {"node_ip": node_ip, "result": f"WRITE_FAILED: {r1.stderr.strip()}"}

            # Step 2: execute
            r2 = subprocess.run(["python3", REMOTE_SCRIPT_PATH], capture_output=True, text=True, timeout=30)
            return {
                "node_ip": node_ip,
                "result": r2.stdout.strip() if r2.returncode == 0 else f"EXEC_FAILED: {r2.stderr.strip()}",
            }

        tasks.append(run_on_node.remote())

    results = ray.get(tasks)

    if args.diagnose:
        for r in results:
            print(f"\n=== {r['node_ip']} ===")
            print(r["result"])
        return

    success = 0
    for r in results:
        print(f"  {r['node_ip']}: {r['result']}")
        if "PATCHED" in r["result"] or "ALREADY_PATCHED" in r["result"] or "ROLLED_BACK" in r["result"]:
            success += 1

    action = "rolled back" if args.rollback else "patched"
    print(f"\n{success}/{len(nodes)} nodes {action} successfully.")


if __name__ == "__main__":
    main()
