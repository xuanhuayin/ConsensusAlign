"""Run and resume a declared alignment experiment, then aggregate seeds correctly."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .__main__ import evaluate
from .core import ConsensusAlign, WPConfig
from .data import fitting_data, read_manifest, seed_at
from .datasets import fingerprint_manifest, inspect_manifest, sha256_file


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_plan(path: Path) -> dict:
    raw = json.loads(path.read_text())
    allowed = {
        "name",
        "train_manifest",
        "evaluation_manifest",
        "seed_indices",
        "anchor_budgets",
        "latent_dim",
        "weighting",
        "wp",
        "evaluation_block_size",
        "evaluation_vision",
        "evaluation_text",
    }
    if set(raw) - allowed:
        raise ValueError(f"Unknown plan fields: {sorted(set(raw) - allowed)}")
    plan = dict(raw)
    for field in ("train_manifest", "evaluation_manifest"):
        plan[field] = (path.parent / plan[field]).resolve()
    for field, default in [("seed_indices", list(range(5))), ("anchor_budgets", [0])]:
        values = plan.setdefault(field, default)
        if not values or len(set(values)) != len(values) or any(type(x) is not int or x < 0 for x in values):
            raise ValueError(f"{field} must contain unique nonnegative integers")
    for index in plan["seed_indices"]:
        seed_at(index)
    plan.setdefault("latent_dim", 768)
    plan.setdefault("weighting", "uniform")
    plan.setdefault("evaluation_block_size", 256)
    if type(plan["latent_dim"]) is not int or plan["latent_dim"] < 1:
        raise ValueError("latent_dim must be a positive integer")
    if type(plan["evaluation_block_size"]) is not int or plan["evaluation_block_size"] < 1:
        raise ValueError("evaluation_block_size must be a positive integer")
    if plan["weighting"] not in ("uniform", "agreement"):
        raise ValueError("weighting must be uniform or agreement")
    plan["wp"] = asdict(WPConfig(**plan.get("wp", {})))
    return plan


def selected_evaluation(plan: dict) -> dict:
    manifest = read_manifest(plan["evaluation_manifest"])
    if manifest.get("task", "alignment") != "alignment":
        raise ValueError(
            "The experiment runner aggregates alignment experiments. Use evaluate for downstream tasks."
        )
    for modality in ("vision", "text"):
        names = plan.get("evaluation_" + modality)
        if names is not None:
            if not names or len(set(names)) != len(names) or set(names) - set(manifest[modality]):
                raise ValueError(f"Invalid evaluation_{modality} selection")
            manifest[modality] = {name: manifest[modality][name] for name in names}
    return manifest


def tasks(plan: dict) -> list[dict]:
    return [
        {
            "seed_index": index,
            "seed": seed_at(index),
            "anchors": budget,
            "directory": f"seed_{index:02d}/anchors_{budget:04d}",
        }
        for index in plan["seed_indices"]
        for budget in plan["anchor_budgets"]
    ]


def summarize(output: Path, plan: dict) -> dict:
    groups = []
    evaluation = selected_evaluation(plan)
    declared_pairs = {(v, t) for v in evaluation["vision"] for t in evaluation["text"]}
    for budget in plan["anchor_budgets"]:
        seed_rows, pairs = [], {}
        for task in [task for task in tasks(plan) if task["anchors"] == budget]:
            directory = output / task["directory"]
            run = json.loads((directory / "run.json").read_text())
            data = json.loads((directory / "alignment.json").read_text())
            if (
                run.get("status") != "complete"
                or run.get("seed") != task["seed"]
                or run.get("anchors") != budget
            ):
                raise ValueError(f"Incomplete or mismatched run in {task['directory']}")
            for field in ("latent_dim", "weighting", "wp", "evaluation_block_size"):
                if run.get("protocol", {}).get(field) != plan[field]:
                    raise ValueError(f"Run settings do not match the plan in {task['directory']}")
            if run.get("evaluation_sha256") != sha256_file(directory / "alignment.json"):
                raise ValueError(f"Evaluation artifact changed in {task['directory']}")
            results = data["results"]
            keys = [(row["vision"], row["text"]) for row in results]
            if not keys or len(keys) != len(set(keys)):
                raise ValueError("Every seed must report unique encoder pairs")
            if set(keys) != declared_pairs:
                raise ValueError(
                    "Each seed must report every encoder pair declared by the evaluation manifest"
                )
            values = [row["foscttm"] for row in results]
            if any(type(value) not in (int, float) or not 0 <= value <= 1 for value in values):
                raise ValueError("Invalid FOSCTTM result")
            seed_rows.append(
                {
                    "seed_index": task["seed_index"],
                    "seed": task["seed"],
                    "mean": statistics.mean(values),
                    "failed_cells": sum(x > 0.05 for x in values),
                }
            )
            for row in results:
                pairs.setdefault((row["vision"], row["text"]), []).append(row["foscttm"])
        means = [row["mean"] for row in seed_rows]
        groups.append(
            {
                "anchors": budget,
                "n_seeds": len(seed_rows),
                "pairs_per_seed": len(pairs),
                "mean_foscttm": statistics.mean(means),
                "sd_foscttm": statistics.stdev(means) if len(means) > 1 else None,
                "failed_cells": sum(row["failed_cells"] for row in seed_rows),
                "n_cells": sum(len(values) for values in pairs.values()),
                "worst_foscttm": max(max(values) for values in pairs.values()),
                "seed_means": seed_rows,
                "per_pair": [
                    {
                        "vision": v,
                        "text": t,
                        "mean": statistics.mean(values),
                        "sd": statistics.stdev(values) if len(values) > 1 else None,
                    }
                    for (v, t), values in sorted(pairs.items())
                ],
            }
        )
    result = {
        "experiment": plan.get("name", "alignment"),
        "aggregation": "mean pairs within each seed, then sample SD across seed means",
        "groups": groups,
    }
    atomic_json(output / "summary.json", result)
    return result


def run(plan: dict, output: Path, *, dry_run=False, overwrite=False, summarize_only=False) -> dict:
    declared = tasks(plan)
    if dry_run:
        return {
            "experiment": plan.get("name", "alignment"),
            "tasks": declared,
            "bridge_fits": len(declared),
            "latent_dim": plan["latent_dim"],
            "weighting": plan["weighting"],
            "wp": plan["wp"],
        }
    output.mkdir(parents=True, exist_ok=True)
    if summarize_only:
        return summarize(output, plan)
    torch.set_num_threads(4)
    train = read_manifest(plan["train_manifest"])
    evaluation = selected_evaluation(plan)
    inspect_manifest(train)
    inspect_manifest(evaluation)
    cache = {}
    data_identity = {
        "train": fingerprint_manifest(train, cache),
        "evaluation": fingerprint_manifest(evaluation, cache),
    }
    for task in declared:
        directory = output / task["directory"]
        directory.mkdir(parents=True, exist_ok=True)
        identity_fields = {
            "data": data_identity,
            "seed": task["seed"],
            "anchors": task["anchors"],
            "latent_dim": plan["latent_dim"],
            "weighting": plan["weighting"],
            "wp": plan["wp"],
            "evaluation_block_size": plan["evaluation_block_size"],
        }
        identity = hashlib.sha256(json.dumps(identity_fields, sort_keys=True).encode()).hexdigest()
        record_path = directory / "run.json"
        prior = json.loads(record_path.read_text()) if record_path.exists() else {}
        if prior and prior.get("identity") != identity and not overwrite:
            raise ValueError(
                f"Input data or settings changed in {task['directory']}. Choose a new output or --overwrite."
            )
        if prior.get("identity") == identity and prior.get("status") == "complete" and not overwrite:
            if not (directory / "model.pt").is_file() or not (directory / "alignment.json").is_file():
                raise ValueError(
                    f"Completed run is missing artifacts in {task['directory']}. Use --overwrite."
                )
            if prior.get("checkpoint_sha256") != sha256_file(directory / "model.pt") or prior.get(
                "evaluation_sha256"
            ) != sha256_file(directory / "alignment.json"):
                raise ValueError(f"A completed artifact changed in {task['directory']}. Use --overwrite.")
            print(f"Verified existing run: {task['directory']}")
            continue
        lock = directory / ".run.lock"
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as error:
            raise RuntimeError(f"Run is locked: {task['directory']}") from error
        os.close(descriptor)
        try:
            record = {
                **(prior if prior.get("identity") == identity and not overwrite else {}),
                **task,
                "identity": identity,
                "status": "started",
                "protocol": identity_fields,
            }
            atomic_json(record_path, record)
            checkpoint = directory / "model.pt"
            model = None
            if checkpoint.exists() and not overwrite:
                if record.get("checkpoint_sha256") and record["checkpoint_sha256"] != sha256_file(checkpoint):
                    raise ValueError(f"A fitted checkpoint changed in {task['directory']}. Use --overwrite.")
                candidate = ConsensusAlign.load(checkpoint)
                if candidate.state.get("run_identity") == identity:
                    model = candidate
            if model is None:
                start = time.perf_counter()
                data, rows, anchors = fitting_data(train, task["seed"], task["anchors"])
                record["feature_load_seconds"] = time.perf_counter() - start
                start = time.perf_counter()
                model = ConsensusAlign.fit(
                    data["vision"],
                    data["text"],
                    seed=task["seed"],
                    latent_dim=plan["latent_dim"],
                    weighted=plan["weighting"] == "agreement",
                    anchors=anchors,
                    wp=WPConfig(**plan["wp"]),
                )
                record["fit_seconds"] = time.perf_counter() - start
                model.state.update(rows=rows, run_identity=identity)
                temporary = directory / ".model.tmp.pt"
                model.save(temporary)
                temporary.replace(checkpoint)
                del data, anchors
            record["status"] = "fitted"
            record["checkpoint_sha256"] = sha256_file(checkpoint)
            atomic_json(record_path, record)
            start = time.perf_counter()
            values = evaluate(model, evaluation, plan["evaluation_block_size"])
            atomic_json(directory / "alignment.json", values)
            record.update(
                status="complete",
                evaluation_seconds=time.perf_counter() - start,
                evaluation_sha256=sha256_file(directory / "alignment.json"),
            )
            atomic_json(record_path, record)
            print(f"Completed: {task['directory']}")
        finally:
            lock.unlink(missing_ok=True)
    return summarize(output, plan)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/reproduction"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    mode.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            run(
                read_plan(args.plan.resolve()),
                args.output,
                dry_run=args.dry_run,
                overwrite=args.overwrite,
                summarize_only=args.summarize_only,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
