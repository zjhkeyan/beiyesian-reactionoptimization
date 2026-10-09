from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import pandas as pd

from .campaign import CampaignRunner
from .chemistry import build_registry_template
from .config import load_run_config
from .datasets import DatasetSpec, load_reaction_dataset, summarize_topology
from .metrics import summarize_metric
from .oracle import OfflineTableOracle
from .prior import load_candidate_prior
from .serialization import (
    load_torch_submission,
    save_json,
    save_torch_submission,
    validate_submission_payload,
)
from .utils import ensure_directory


def _load_problem(args: argparse.Namespace):
    config = load_run_config(args.config)
    spec = DatasetSpec.from_config(config.dataset)
    dataset = load_reaction_dataset(args.train, args.candidates, spec)
    return config, dataset


def _cmd_audit(args: argparse.Namespace) -> int:
    config, dataset = _load_problem(args)
    payload = {
        "config": asdict(config),
        "topology": summarize_topology(dataset),
        "target_incumbent": dataset.target_incumbent(),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    return 0


def _cmd_make_registry(args: argparse.Namespace) -> int:
    config, dataset = _load_problem(args)
    del config
    frames = [dataset.observed_features(), dataset.candidate_features()]
    output = build_registry_template(frames, dataset.spec.roles, args.output)
    print(output)
    return 0


def _run_one(
    args: argparse.Namespace,
    seed: int,
    output_dir: Path,
) -> dict:
    config, dataset = _load_problem(args)
    prior = (
        None
        if args.prior is None
        else load_candidate_prior(args.prior, dataset.candidate_ids)
    )
    oracle = OfflineTableOracle.from_csv(dataset, args.oracle)
    runner = CampaignRunner(
        dataset,
        config,
        seed=seed,
        backend=args.backend,
        prior_scores=prior,
    )
    snapshot_dir = None
    if args.save_candidate_scores:
        snapshot_dir = output_dir / f"seed_{seed}_candidate_scores"
    result = runner.run(
        oracle,
        budget=args.budget,
        score_snapshot_dir=snapshot_dir,
    )
    payload = result.to_payload()
    save_json(payload, output_dir / f"seed_{seed}.json")
    save_torch_submission(payload, output_dir / f"seed_{seed}.pt")
    return payload


def _cmd_simulate(args: argparse.Namespace) -> int:
    output_dir = ensure_directory(args.output_dir)
    payload = _run_one(args, int(args.seed), output_dir)
    print(json.dumps(payload["metrics"], indent=2))
    return 0


def _cmd_run_all(args: argparse.Namespace) -> int:
    config, _ = _load_problem(args)
    seeds = config.dataset.seeds if not args.seeds else tuple(args.seeds)
    output_dir = ensure_directory(args.output_dir)
    payloads = [_run_one(args, int(seed), output_dir) for seed in seeds]

    keys = ["initial_round_found_best", "best_found", "auc_best_so_far"]
    summary = {
        key: summarize_metric(payload["metrics"][key] for payload in payloads)
        for key in keys
    }
    t95_values = [
        payload["metrics"]["round_to_95_global_best"]
        for payload in payloads
        if payload["metrics"]["round_to_95_global_best"] is not None
    ]
    summary["t95_success_rate"] = len(t95_values) / len(payloads)
    if t95_values:
        summary["t95_conditional"] = summarize_metric(t95_values)
    save_json(summary, output_dir / "summary.json")
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    config, dataset = _load_problem(args)
    payload = load_torch_submission(args.submission)
    expected_budget = int(args.budget or config.dataset.budget)
    validate_submission_payload(
        payload,
        dataset.candidates,
        feature_columns=dataset.spec.feature_columns,
        expected_budget=expected_budget,
    )
    print("Submission is valid.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bgolearn-rxn",
        description="Reaction-specific best-practice extensions for Bgolearn.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_problem_paths(command: argparse.ArgumentParser) -> None:
        command.add_argument("--config", required=True, type=Path)
        command.add_argument("--train", required=True, type=Path)
        command.add_argument("--candidates", required=True, type=Path)

    audit = subparsers.add_parser("audit", help="Validate participant-side files.")
    add_problem_paths(audit)
    audit.set_defaults(func=_cmd_audit)

    registry = subparsers.add_parser(
        "make-registry", help="Create a review-first structure registry template."
    )
    add_problem_paths(registry)
    registry.add_argument("--output", required=True, type=Path)
    registry.set_defaults(func=_cmd_make_registry)

    def add_run_options(command: argparse.ArgumentParser) -> None:
        add_problem_paths(command)
        command.add_argument("--oracle", required=True, type=Path)
        command.add_argument("--backend", choices=("native", "bgolearn"), default=None)
        command.add_argument("--budget", type=int, default=None)
        command.add_argument("--prior", type=Path, default=None)
        command.add_argument("--output-dir", required=True, type=Path)
        command.add_argument("--save-candidate-scores", action="store_true")

    simulate = subparsers.add_parser("simulate", help="Run one offline campaign.")
    add_run_options(simulate)
    simulate.add_argument("--seed", required=True, type=int)
    simulate.set_defaults(func=_cmd_simulate)

    run_all = subparsers.add_parser("run-all", help="Run the configured seed suite.")
    add_run_options(run_all)
    run_all.add_argument("--seeds", nargs="*", type=int, default=None)
    run_all.set_defaults(func=_cmd_run_all)

    validate = subparsers.add_parser("validate", help="Validate a .pt trajectory.")
    add_problem_paths(validate)
    validate.add_argument("--submission", required=True, type=Path)
    validate.add_argument("--budget", type=int, default=None)
    validate.set_defaults(func=_cmd_validate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
