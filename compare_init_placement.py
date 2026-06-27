"""Compare zoned neutral-atom initial-placement strategies across benchmarks.

The script mirrors the zoned compiler example from the documentation:

* build the zoned neutral-atom architecture from JSON,
* load benchmark circuits through :func:`mqt.core.load`,
* compile them with :class:`mqt.qmap.na.zoned.RoutingAwareCompiler`, and
* collect the compiler statistics after each run.

The current dummy placer is controlled through the ``strategy_name`` selector:

* ``0`` -> trivial initial placement
* ``1`` -> strategy 1
* ``2`` -> strategy 2

The compiler produces the same ``.naviz`` output shown in the docs, and the
script can optionally write that output to disk for inspection.
"""

from __future__ import annotations

import argparse
from copy import error
from pathlib import Path
from typing import Iterable
import json
import time
from qiskit import QuantumCircuit, transpile

import pandas as pd

from mqt.bench import BenchmarkLevel, get_benchmark
from mqt.core import load
from mqt.qmap.na.zoned import RoutingAwareCompiler, ZonedNeutralAtomArchitecture

DEFAULT_ARCHITECTURE_JSON = """{
  "name": "Architecture with one entanglement and one storage zone",
  "operation_duration": {"rydberg_gate": 0.36, "single_qubit_gate": 52, "atom_transfer": 15},
  "operation_fidelity": {"rydberg_gate": 0.995, "single_qubit_gate": 0.9997, "atom_transfer": 0.999},
  "qubit_spec": {"T": 1.5e6},
  "storage_zones": [{
    "zone_id": 0,
    "slms": [{"id": 0, "site_separation": [3, 3], "r": 20, "c": 100, "location": [0, 0]}],
    "offset": [0, 0],
    "dimension": [297, 57]
  }],
  "entanglement_zones": [{
    "zone_id": 0,
    "slms": [
      {"id": 1, "site_separation": [12, 10], "r": 7, "c": 20, "location": [35, 67]},
      {"id": 2, "site_separation": [12, 10], "r": 7, "c": 20, "location": [37, 67]}
    ],
    "offset": [35, 67],
    "dimension": [230, 60]
  }],
  "aods": [{"id": 0, "site_separation": 2, "r": 100, "c": 100}],
  "rydberg_range": [[[30, 62], [270, 132]]]
}"""

STRATEGIES = (0, 1, 2)


def make_architecture(architecture_json: str) -> ZonedNeutralAtomArchitecture:
    return ZonedNeutralAtomArchitecture.from_json_string(architecture_json)


def make_compiler(
    architecture: ZonedNeutralAtomArchitecture, strategy_name: int
) -> RoutingAwareCompiler:
    config_dict = {
        "layoutSynthesizerConfig": {
            "placerConfig": {
                "strategyName": strategy_name,  # <-- Nests perfectly into HeuristicPlacer::Config!
                "useWindow": True,
            }
        },
        "logLevel": 4  # 4 = Error level integer
    }

    return RoutingAwareCompiler.from_json_string(
        architecture, json.dumps(config_dict)
    )

def benchmark_specs() -> Iterable[tuple[str, int]]:
    families = ("qft", "graphstate", "ghz", "bv", "wstate", "qpeexact")
    sizes = (20, 50, 100) # check sizes
    for family in families:
        for size in sizes:
            yield family, size


def compile_one(
    benchmark_name: str,
    size: int,
    strategy_name: int,
    architecture: ZonedNeutralAtomArchitecture,
    naviz_dir: Path | None,
) -> dict[str, object]:
    print(f"Compiling {benchmark_name} with n={size} and strategy {strategy_name}...")
    raw_circuit = get_benchmark(benchmark_name, BenchmarkLevel.ALG, size)
    transpiled_circuit = transpile(
        raw_circuit, 
        basis_gates=["cz", "id", "u2", "u1", "u3"], 
        optimization_level=1
    )

    stripped_circuit = QuantumCircuit(*transpiled_circuit.qregs, *transpiled_circuit.cregs)
    for instruction in transpiled_circuit.data:
        if instruction.operation.name not in {"measure", "barrier"}:
            stripped_circuit.append(instruction)
    compiler = make_compiler(architecture, strategy_name)
    print(type(compiler))
    # Track execution time manually in case C++ bindings stay stateless
    start_wall = time.perf_counter()
    compiled = compiler.compile(load(stripped_circuit))
    end_wall = time.perf_counter()
    
    stats = compiler.stats()
    
    # If stats comes back empty from the C++ layer, inject our manual benchmark timings
    if not stats:
        print("No stats from compiler, using manual timing.")
        stats = {
            "totalTime": (end_wall - start_wall) * 1000, # time in ms for your pandas mean() aggregation
        }

    if naviz_dir is not None:
        naviz_dir.mkdir(parents=True, exist_ok=True)
        naviz_path = naviz_dir / f"{benchmark_name}_n{size}_strategy{strategy_name}.naviz"
        naviz_path.write_text(compiled)

    row: dict[str, object] = {
        "benchmark": benchmark_name,
        "n_qubits": size,
        "strategy_name": strategy_name,
        "naviz_length": len(compiled),
    }
    row.update(stats)
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="placement_comparison.csv", help="CSV output path")
    parser.add_argument(
        "--naviz-dir",
        default=None,
        help="Optional directory for .naviz outputs",
    )
    parser.add_argument(
        "--arch-json",
        default=DEFAULT_ARCHITECTURE_JSON,
        help="Optional architecture JSON string or path to a JSON file",
    )
    args = parser.parse_args()

    arch_json_arg = args.arch_json.strip()
    if arch_json_arg.startswith("{") or arch_json_arg.startswith("["):
        architecture_json = args.arch_json
    else:
        arch_json_path = Path(args.arch_json)
        architecture_json = arch_json_path.read_text() if arch_json_path.is_file() else args.arch_json
    architecture = make_architecture(architecture_json)
    naviz_dir = Path(args.naviz_dir) if args.naviz_dir is not None else None

    rows = []
    for benchmark_name, size in benchmark_specs():
        for strategy_name in STRATEGIES:
            #try:
            row = compile_one(benchmark_name, size, strategy_name, architecture, naviz_dir)
            #except Exception as exc:  # pragma: no cover - surfaced in the CSV output
            #    row = {
            #        "benchmark": benchmark_name,
            #        "n_qubits": size,
            #        "strategy_name": strategy_name,
            #        "error": repr(exc),
            #    }
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    try:
        agg = df.groupby(["benchmark", "strategy_name"])["totalTime"].mean()
        print(agg.to_string())
    except Exception:
        print("No aggregation available (missing totalTime values)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
