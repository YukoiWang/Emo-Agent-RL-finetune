#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive 5-Level Evaluation Runner.

Usage examples:

  # Evaluate final checkpoint only (Level 1-3)
  python scripts/eval/comprehensive/run_eval.py \
      --output-dir outputs/ppo_emo \
      --levels 1 2 3 \
      --eval-final-only

  # Full learning-curve evaluation (Level 1-4)
  python scripts/eval/comprehensive/run_eval.py \
      --output-dir outputs/ppo_emo \
      --levels 1 2 3 4 \
      --max-samples 20

  # Environment comparison (Level 5)
  python scripts/eval/comprehensive/run_eval.py \
      --output-dir outputs/ppo_emo \
      --levels 5 \
      --env-dirs vanilla=outputs/ppo_vanilla,challenging=outputs/ppo_challenging

  # All levels with custom API
  python scripts/eval/comprehensive/run_eval.py \
      --output-dir outputs/ppo_emo \
      --api-key sk-xxx \
      --api-base https://api.deepseek.com \
      --max-samples 30
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval.comprehensive.config import EvalConfig
from scripts.eval.comprehensive.llm_client import DeepSeekClient
from scripts.eval.comprehensive.checkpoint_manager import (
    discover_checkpoints,
    get_final_checkpoint,
    load_model_and_tokenizer,
    unload_model,
)
from scripts.eval.comprehensive.dialogue_generator import (
    DialogueResult,
    generate_dialogues_for_checkpoint,
    load_dialogues,
    save_dialogues,
)
from scripts.eval.comprehensive.level1_emotion import evaluate_level1
from scripts.eval.comprehensive.level2_strategy import evaluate_level2
from scripts.eval.comprehensive.level3_capability import evaluate_level3
from scripts.eval.comprehensive.level4_stability import evaluate_level4
from scripts.eval.comprehensive.level5_environment import evaluate_level5
from scripts.eval.comprehensive import visualize

from src.data.profile_dataset import load_profiles


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Comprehensive 5-Level Evaluation",
    )
    p.add_argument("--output-dir", type=str, required=True,
                   help="Training output dir containing checkpoints")
    p.add_argument("--test-data", type=str, default="data/data/test_profile.jsonl",
                   help="Test profile JSONL path")
    p.add_argument("--results-dir", type=str, default=None,
                   help="Where to save eval results (default: <output-dir>/eval_comprehensive)")
    p.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--api-key", type=str, default="")
    p.add_argument("--api-base", type=str, default="https://api.deepseek.com")
    p.add_argument("--judge-model", type=str, default="deepseek-chat")
    p.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                   help="Which levels to evaluate (1-5)")
    p.add_argument("--max-samples", type=int, default=50,
                   help="Max test profiles per checkpoint")
    p.add_argument("--max-turns", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--initial-emo", type=float, default=50.0)
    p.add_argument("--checkpoint-interval", type=int, default=50,
                   help="Only evaluate checkpoints at multiples of this step")
    p.add_argument("--eval-final-only", action="store_true",
                   help="Only evaluate the final checkpoint")
    p.add_argument("--skip-dialogue-gen", action="store_true",
                   help="Skip dialogue generation; load from cached files")
    p.add_argument("--env-dirs", type=str, default="",
                   help="For Level 5: env_name=path,... (comma-separated)")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--fig-format", type=str, default="png",
                   choices=["png", "pdf", "svg"])
    p.add_argument("--smooth", type=float, default=5.0)
    return p.parse_args()


def build_config(args: argparse.Namespace) -> EvalConfig:
    cfg = EvalConfig(
        output_dir=args.output_dir,
        test_data_path=args.test_data,
        base_model=args.base_model,
        api_key=args.api_key,
        api_base=args.api_base,
        judge_model=args.judge_model,
        max_turns=args.max_turns,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        initial_emo_point=args.initial_emo,
        max_test_samples=args.max_samples,
        checkpoint_interval=args.checkpoint_interval,
        eval_levels=args.levels,
        eval_final_only=args.eval_final_only,
        device=args.device,
        dpi=args.dpi,
        fig_format=args.fig_format,
        smooth_span=args.smooth,
    )
    if args.results_dir:
        cfg.results_dir = args.results_dir
    else:
        cfg.results_dir = str(Path(args.output_dir) / "eval_comprehensive")
    return cfg


def _dialogues_cache_path(results_dir: Path, step: int) -> Path:
    return results_dir / "dialogues" / f"step_{step}.json"


def run_checkpoint_eval(
    cfg: EvalConfig,
    ckpt_path: str,
    step: int,
    profiles: List[dict],
    client: DeepSeekClient,
    skip_gen: bool = False,
) -> Dict[str, Any]:
    """Evaluate a single checkpoint across requested levels."""
    results_dir = cfg.resolved_results_dir
    cache_path = _dialogues_cache_path(results_dir, step)
    result: Dict[str, Any] = {"step": step, "checkpoint": ckpt_path}

    # ── dialogue generation ─────────────────────────────────────────
    if skip_gen and cache_path.exists():
        print(f"  Loading cached dialogues from {cache_path}")
        dialogues = load_dialogues(cache_path)
    else:
        print(f"  Loading model from {ckpt_path} ...")
        model, tokenizer = load_model_and_tokenizer(
            ckpt_path, cfg.base_model, cfg.device,
        )
        planning_fn = client.as_llm_fn()
        player_fn = client.as_llm_fn()

        print(f"  Generating dialogues ({len(profiles)} samples) ...")
        dialogues = generate_dialogues_for_checkpoint(
            model, tokenizer, profiles,
            planning_llm_fn=planning_fn,
            player_llm_fn=player_fn,
            max_turns=cfg.max_turns,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            initial_emo_point=cfg.initial_emo_point,
            device=cfg.device,
            max_samples=cfg.max_test_samples,
        )
        unload_model(model)
        save_dialogues(dialogues, cache_path)
        print(f"  Cached {len(dialogues)} dialogues to {cache_path}")

    # ── Level 1 ─────────────────────────────────────────────────────
    if 1 in cfg.eval_levels:
        print("  [Level 1] Emotion metrics ...")
        l1 = evaluate_level1(dialogues)
        result.update(l1)
        result["emotion_score"] = l1["emotion_score"]
        result["avg_emo_change"] = l1["avg_emo_change"]

    # ── Level 2 ─────────────────────────────────────────────────────
    if 2 in cfg.eval_levels:
        print("  [Level 2] Strategy analysis ...")
        l2 = evaluate_level2(dialogues, client)
        result["strategy_frequency"] = l2["strategy_frequency"]
        result["strategy_contribution"] = l2["strategy_contribution"]

    # ── Level 3 ─────────────────────────────────────────────────────
    if 3 in cfg.eval_levels:
        print("  [Level 3] SCC + Core Capability ...")
        l3 = evaluate_level3(dialogues, client)
        result["scc"] = l3["scc"]
        result["core_capabilities"] = l3["core_capabilities"]

    return result


def run_full_evaluation(cfg: EvalConfig, args: argparse.Namespace):
    """Orchestrate the full multi-level evaluation."""
    results_dir = cfg.resolved_results_dir
    print(f"Results will be saved to: {results_dir}")

    # ── load test profiles ──────────────────────────────────────────
    test_path = cfg.resolved_test_data
    print(f"Loading test profiles from {test_path} ...")
    profiles = load_profiles(str(test_path.parent), split="test")
    profiles = profiles[: cfg.max_test_samples]
    print(f"  {len(profiles)} profiles loaded")

    # ── LLM client ──────────────────────────────────────────────────
    client = DeepSeekClient(
        api_key=cfg.get_api_key(),
        base_url=cfg.api_base,
        model=cfg.judge_model,
        temperature=0.3,
    )

    # ── discover checkpoints ────────────────────────────────────────
    if cfg.eval_final_only:
        final = get_final_checkpoint(cfg.resolved_output_dir)
        if final is None:
            print("ERROR: no checkpoint found")
            return
        ckpts = [{"step": -1, "path": final}]
    else:
        ckpts = discover_checkpoints(
            cfg.resolved_output_dir, cfg.checkpoint_interval,
        )
    print(f"Checkpoints to evaluate: {len(ckpts)}")
    for c in ckpts:
        print(f"  step={c['step']}  path={c['path']}")

    # ── per-checkpoint evaluation ───────────────────────────────────
    all_results: List[Dict[str, Any]] = []
    t_start = time.time()

    for i, ckpt in enumerate(ckpts):
        step = ckpt["step"]
        label = f"final" if step == -1 else f"step-{step}"
        print(f"\n{'='*60}")
        print(f"Checkpoint {i+1}/{len(ckpts)}: {label}")
        print(f"{'='*60}")

        result = run_checkpoint_eval(
            cfg, ckpt["path"], step, profiles, client,
            skip_gen=args.skip_dialogue_gen,
        )
        all_results.append(result)

        # save incremental
        with open(results_dir / "checkpoint_results.json", "w") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2,
                      default=str)

    elapsed = time.time() - t_start
    print(f"\nAll checkpoints evaluated in {elapsed:.1f}s")

    # ── Level 4: stability across checkpoints ──────────────────────
    if 4 in cfg.eval_levels and len(all_results) > 1:
        print("\n[Level 4] Learning Curve Stability ...")
        l4_input = [
            r for r in all_results
            if "emotion_score" in r and r["step"] != -1
        ]
        if l4_input:
            stability = evaluate_level4(l4_input)
            with open(results_dir / "level4_stability.json", "w") as f:
                json.dump(stability, f, indent=2, default=str)
            print(f"  Score std={stability['score_std']:.2f}, "
                  f"mono={stability['monotonicity']:.2f}, "
                  f"osc={stability['oscillation_detected']}, "
                  f"collapse={stability['collapse_detected']}")
        else:
            stability = None
    else:
        stability = None

    # ── Level 5: environment comparison ────────────────────────────
    if 5 in cfg.eval_levels and args.env_dirs:
        print("\n[Level 5] Environment Comparison ...")
        env_map = {}
        for part in args.env_dirs.split(","):
            name, path = part.strip().split("=", 1)
            env_map[name.strip()] = path.strip()

        env_results_map = {}
        env_dialogues_map = {}
        for env_name, env_dir in env_map.items():
            print(f"  Evaluating env={env_name} ...")
            env_final = get_final_checkpoint(env_dir)
            if env_final is None:
                print(f"    No checkpoint found in {env_dir}, skipping")
                continue
            env_res = run_checkpoint_eval(
                cfg, env_final, -1, profiles, client,
                skip_gen=args.skip_dialogue_gen,
            )
            env_results_map[env_name] = env_res
            cache = _dialogues_cache_path(results_dir, -1)
            if cache.exists():
                env_dialogues_map[env_name] = load_dialogues(cache)
            else:
                env_dialogues_map[env_name] = []

        l5 = evaluate_level5(env_dialogues_map, env_results_map, client)
        with open(results_dir / "level5_environment.json", "w") as f:
            json.dump(l5, f, indent=2, default=str, ensure_ascii=False)
    else:
        l5 = None

    # ── Visualization ───────────────────────────────────────────────
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating plots in {plots_dir} ...")

    if 1 in cfg.eval_levels and len(all_results) > 1:
        ckpt_results = [r for r in all_results if r["step"] != -1 and "emotion_score" in r]
        if ckpt_results:
            visualize.plot_emotion_curves(
                ckpt_results, str(plots_dir),
                smooth=cfg.smooth_span, fmt=cfg.fig_format, dpi=cfg.dpi,
            )

    final_result = all_results[-1] if all_results else {}

    if 1 in cfg.eval_levels and "per_turn_emo_stats" in final_result:
        visualize.plot_per_turn_emo(
            final_result["per_turn_emo_stats"],
            str(plots_dir), fmt=cfg.fig_format, dpi=cfg.dpi,
        )

    if 2 in cfg.eval_levels:
        if "strategy_frequency" in final_result:
            visualize.plot_strategy_frequency(
                final_result["strategy_frequency"],
                str(plots_dir), fmt=cfg.fig_format, dpi=cfg.dpi,
            )
        if "strategy_contribution" in final_result:
            visualize.plot_strategy_contribution(
                final_result["strategy_contribution"],
                str(plots_dir), fmt=cfg.fig_format, dpi=cfg.dpi,
            )

    if 3 in cfg.eval_levels:
        if "scc" in final_result:
            visualize.plot_scc_coordinate(
                final_result["scc"],
                str(plots_dir), fmt=cfg.fig_format, dpi=cfg.dpi,
            )
            # SCC trajectory across checkpoints
            scc_traj = []
            for r in all_results:
                if "scc" in r and r["step"] != -1:
                    coord = r["scc"].get("coordinate", (0, 0))
                    scc_traj.append((r["step"], coord[0], coord[1]))
            if len(scc_traj) > 1:
                visualize.plot_scc_trajectory(
                    scc_traj, str(plots_dir),
                    fmt=cfg.fig_format, dpi=cfg.dpi,
                )
        if "core_capabilities" in final_result:
            visualize.plot_capability_radar(
                final_result["core_capabilities"],
                str(plots_dir), fmt=cfg.fig_format, dpi=cfg.dpi,
            )

    if stability is not None:
        visualize.plot_stability(
            stability, str(plots_dir),
            smooth=cfg.smooth_span, fmt=cfg.fig_format, dpi=cfg.dpi,
        )

    if l5 is not None:
        comp = l5.get("environment_comparison", {})
        if comp:
            visualize.plot_environment_comparison(
                comp, str(plots_dir), fmt=cfg.fig_format, dpi=cfg.dpi,
            )
        sim_q = l5.get("simulator_quality", {})
        if sim_q:
            visualize.plot_simulator_quality(
                sim_q, str(plots_dir), fmt=cfg.fig_format, dpi=cfg.dpi,
            )

    visualize.plot_overview_dashboard(
        final_result, str(plots_dir),
        fmt=cfg.fig_format, dpi=cfg.dpi,
    )

    # ── Print summary ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)
    if "emotion_score" in final_result:
        print(f"  Emotion Score:  {final_result['emotion_score']:.2f}")
    if "avg_emo_change" in final_result:
        print(f"  Avg EmoChange:  {final_result['avg_emo_change']:.2f}")
    if "strategy_frequency" in final_result:
        freq = final_result["strategy_frequency"]
        top = max(freq, key=freq.get) if freq else "N/A"
        print(f"  Top Strategy:   {top} ({freq.get(top, 0):.1%})")
    if "core_capabilities" in final_result:
        caps = final_result["core_capabilities"]
        avg = sum(caps.values()) / len(caps) if caps else 0
        print(f"  Avg Capability: {avg:.2f} / 5.0")
        for k, v in caps.items():
            print(f"    {k:25s} {v:.2f}")
    if "scc" in final_result:
        coord = final_result["scc"]["coordinate"]
        print(f"  SCC Coordinate: ({coord[0]:.2f}, {coord[1]:.2f})")
    if stability:
        print(f"  Stability Std:  {stability['score_std']:.2f}")
        print(f"  Best Step:      {stability['best_step']}")
    print(f"\n  All results saved to: {results_dir}")
    print("=" * 60)


def main():
    args = build_args()
    cfg = build_config(args)
    run_full_evaluation(cfg, args)


if __name__ == "__main__":
    main()
