# SPDX-License-Identifier: Apache-2.0
"""Generate a custom NPU-NIC topology file for the Mooncake Transfer Engine.

This script probes the local hardware (``npu-smi info``, ``mst status -v``
and the PCI sysfs tree) and writes ``mooncake_topo.json`` in the format
expected by the Mooncake Transfer Engine when the environment variable
``MC_CUSTOM_TOPO_JSON`` points to the generated file::

    {
        "cpu:0": [["mlx5_bond_1", ...], ["mlx5_bond_5", ...]],
        "cpu:1": [["mlx5_bond_5", ...], ["mlx5_bond_1", ...]],
        "npu:0": [["mlx5_bond_1"], ["mlx5_bond_1"]],
        ...
    }

Each key is a memory location (``npu:<id>`` / ``cpu:<numa_node>``) and the
value is a two-element priority list of NIC bond names: the first list holds
NICs on the same PCIe switch / NUMA node as the location, and the second list
holds the cross-NUMA fallback NICs.

Storage (non-compute) NICs are excluded. They are auto-detected as the NICs
that are not the best match (longest shared PCI sysfs path prefix) of any NPU,
or they can be set manually via ``EXCLUDE_BONDS`` below.

Usage::

    python tools/generate_topo.py
    # then point Mooncake to the generated file:
    export MC_CUSTOM_TOPO_JSON=/path/to/mooncake_topo.json
"""

import json
import os
import re
import subprocess

# ================= Configuration =================
# Set to None to auto-detect storage NICs (based on PCI sysfs topology
# distance). It can also be set manually, e.g. {"mlx5_bond_0"}, to override
# the auto-detection result.
EXCLUDE_BONDS = None
# =================================================


def run_cmd(cmd):
    """Run a shell command and return its output."""
    try:
        return subprocess.check_output(cmd, shell=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {cmd}\nError: {e}")
        return ""


def parse_npu_pci(npu_out):
    """Parse `npu-smi info` output into a map of NPU ID -> PCI Bus-Id."""
    npu_map = {}
    lines = npu_out.strip().split("\n")
    current_npu = None
    for line in lines:
        if re.match(r"^\|\s*\d+\s+\|", line):
            parts = line.split("|")
            current_npu = int(parts[1].strip())
        elif current_npu is not None and "0000:" in line:
            match = re.search(r"(0000:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9a-fA-F])", line)
            if match:
                npu_map[current_npu] = match.group(1).lower()
            current_npu = None
    return npu_map


def parse_nic_list(mst_out):
    """Parse `mst status -v` output into NIC PCI, bond names and NUMA nodes."""
    nic_list = []
    for line in mst_out.strip().split("\n"):
        if "mlx5_bond_" in line:
            parts = line.split()
            pci_short = [p for p in parts if re.match(r"^[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9a-fA-F]$", p)]
            bonds = [p for p in parts if p.startswith("mlx5_bond_")]
            numa_nodes = [p for p in parts if p.isdigit()]

            if pci_short and bonds:
                pci_full = f"0000:{pci_short[0]}".lower()
                bond = bonds[0]
                numa = int(numa_nodes[-1]) if numa_nodes else 0

                # Deduplicate: .0 and .1 functions show up as the same bond
                # in mst output.
                if not any(n["bond"] == bond for n in nic_list):
                    nic_list.append({"pci": pci_full, "bond": bond, "numa": numa})
    return nic_list


def shared_prefix_len(npu_pci, nic_pci):
    """Return the shared sysfs path prefix length between an NPU and a NIC.

    A longer value means the devices are closer in the PCI topology.
    """
    try:
        npu_sysfs = os.path.realpath(f"/sys/bus/pci/devices/{npu_pci}")
        nic_sysfs = os.path.realpath(f"/sys/bus/pci/devices/{nic_pci}")
        return len(os.path.commonprefix([npu_sysfs, nic_sysfs]))
    except (FileNotFoundError, OSError):
        return -1


def auto_detect_storage_bonds(npu_map, nic_list):
    """Auto-detect storage NICs.

    For each NPU, find its best-matching NIC (longest shared PCI sysfs path).
    NICs never chosen as a best match by any NPU (i.e. not under the same PCI
    switch as any NPU) are considered storage NICs.
    """
    if not npu_map or not nic_list:
        return set()

    matched_bonds = set()
    for npu_id, npu_pci in npu_map.items():
        best_bond = None
        max_shared = -1
        for nic in nic_list:
            shared = shared_prefix_len(npu_pci, nic["pci"])
            if shared > max_shared:
                max_shared = shared
                best_bond = nic["bond"]
        if best_bond is not None:
            matched_bonds.add(best_bond)

    all_bonds = {n["bond"] for n in nic_list}
    return all_bonds - matched_bonds


def main():
    # 1. Parse npu-smi info.
    npu_out = run_cmd("npu-smi info")
    npu_map = parse_npu_pci(npu_out)

    # 2. Parse mst status -v.
    mst_out = run_cmd("mst status -v")
    nic_list_all = parse_nic_list(mst_out)

    # 3. Determine the storage NICs to filter out: manual config takes
    #    precedence, otherwise auto-detect.
    if EXCLUDE_BONDS is None:
        exclude_bonds = auto_detect_storage_bonds(npu_map, nic_list_all)
        print(f"[INFO] Auto-detected storage NICs: {sorted(exclude_bonds) if exclude_bonds else 'none'}")
    else:
        exclude_bonds = set(EXCLUDE_BONDS)
        print(f"[INFO] Using manually configured storage NICs: {sorted(exclude_bonds)}")

    # 4. Filter out storage NICs.
    nic_list = [n for n in nic_list_all if n["bond"] not in exclude_bonds]

    # 5. Group compute NICs by NUMA node.
    numa_bonds = {0: [], 1: []}
    for nic in nic_list:
        if nic["numa"] not in numa_bonds:
            numa_bonds[nic["numa"]] = []
        numa_bonds[nic["numa"]].append(nic["bond"])

    for k in numa_bonds:
        # Sort to keep a stable output order (e.g. mlx5_bond_1 -> mlx5_bond_2).
        numa_bonds[k].sort(key=lambda x: int(x.split("_")[-1]))

    # 6. Build the topology.
    topo = {}

    # Fill CPU topology (local NUMA first, then cross NUMA).
    topo["cpu:0"] = [numa_bonds.get(0, []), numa_bonds.get(1, [])]
    topo["cpu:1"] = [numa_bonds.get(1, []), numa_bonds.get(0, [])]

    # 7. Match NPU and NIC affinity by sysfs physical path prefix.
    for npu_id in sorted(npu_map.keys()):
        npu_pci = npu_map[npu_id]
        if not os.path.exists(f"/sys/bus/pci/devices/{npu_pci}"):
            continue

        best_match_bond = None
        max_shared_len = -1
        for nic in nic_list:
            shared = shared_prefix_len(npu_pci, nic["pci"])
            if shared > max_shared_len:
                max_shared_len = shared
                best_match_bond = nic["bond"]

        if best_match_bond:
            topo[f"npu:{npu_id}"] = [[best_match_bond], [best_match_bond]]

    # 8. Print and save the result.
    output_json = json.dumps(topo, indent=4)
    print(output_json)

    with open("mooncake_topo.json", "w") as f:
        f.write(output_json)
    print(
        f"\nSuccessfully generated mooncake_topo.json, "
        f"excluded storage NICs: {sorted(exclude_bonds) if exclude_bonds else 'none'}"
    )


if __name__ == "__main__":
    main()
