# -*- coding: utf-8 -*-
"""
Generate multi-turn dialogues by running an NPC model against the
planning-based user simulator.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List

import torch

ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.profile_dataset import build_initial_prompt, load_profiles  # noqa: E402
from src.training.hard_player_simulator_dsv3 import PlayerSimulatorWithPlanning  # noqa: E402


# ── data class ──────────────────────────────────────────────────────────

@dataclass
class DialogueResult:
    """Stores one complete multi-turn dialogue and its emotion trajectory."""
    profile_id: str = ""
    profile: dict = field(default_factory=dict, repr=False)
    dialogue_history: List[Dict[str, str]] = field(default_factory=list)
    emo_point_trajectory: List[float] = field(default_factory=list)
    initial_emo: float = 50.0
    final_emo: float = 50.0
    npc_responses: List[str] = field(default_factory=list)
    num_turns: int = 0

    @property
    def emo_change(self) -> float:
        return self.final_emo - self.initial_emo


# ── model-based generation ──────────────────────────────────────────────

@torch.no_grad()
def generate_npc_response(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    device: str = "cuda",
) -> str:
    """Generate a single NPC response given the raw text prompt."""
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=4096,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    new_tokens = out[0, input_ids.shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    for stop in ("用户：", "用户:", "\nUser:"):
        if stop in text:
            text = text.split(stop)[0].strip()
    return text


# ── single dialogue ─────────────────────────────────────────────────────

def generate_single_dialogue(
    model: Any,
    tokenizer: Any,
    profile: dict,
    planning_llm_fn: Callable,
    player_llm_fn: Callable,
    max_turns: int = 8,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    initial_emo_point: float = 50.0,
    device: str = "cuda",
) -> DialogueResult:
    """
    Run a full multi-turn dialogue for one profile.

    Flow per turn:
      1. NPC model generates a response
      2. Planner scores emotion change  (inside sim.step)
      3. User simulator generates next user utterance
    """
    initial_prompt = build_initial_prompt(profile)
    first_user = profile.get("first_talk", "我最近有些事想和你聊聊。")

    sim = PlayerSimulatorWithPlanning(
        profile=profile,
        player_llm_fn=player_llm_fn,
        planning_llm_fn=planning_llm_fn,
        target="eq",
        initial_emo_point=initial_emo_point,
    )
    sim.dialog.append({"role": "user", "content": first_user})

    dialogue_history: List[Dict[str, str]] = [
        {"role": "user", "content": first_user},
    ]
    npc_responses: List[str] = []
    emo_trajectory: List[float] = [initial_emo_point]

    current_input = f"{initial_prompt}用户：{first_user}\n\nNPC："

    for _turn in range(max_turns):
        npc_reply = generate_npc_response(
            model, tokenizer, current_input,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=device,
        )
        if not npc_reply:
            npc_reply = "我理解你的感受。"

        npc_responses.append(npc_reply)
        dialogue_history.append({"role": "assistant", "content": npc_reply})

        user_reply, done = sim.step(npc_reply)
        emo_trajectory.append(sim.get_emo_point())

        if done:
            break

        dialogue_history.append({"role": "user", "content": user_reply})
        current_input += f"{npc_reply}\n\n用户：{user_reply}\n\nNPC："

    return DialogueResult(
        profile_id=profile.get("id", ""),
        profile=profile,
        dialogue_history=dialogue_history,
        emo_point_trajectory=emo_trajectory,
        initial_emo=initial_emo_point,
        final_emo=sim.get_emo_point(),
        npc_responses=npc_responses,
        num_turns=len(npc_responses),
    )


# ── batch generation ────────────────────────────────────────────────────

def generate_dialogues_for_checkpoint(
    model: Any,
    tokenizer: Any,
    profiles: List[dict],
    planning_llm_fn: Callable,
    player_llm_fn: Callable,
    max_turns: int = 8,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    initial_emo_point: float = 50.0,
    device: str = "cuda",
    max_samples: int = 50,
    verbose: bool = True,
) -> List[DialogueResult]:
    """Run dialogues for many profiles against a single checkpoint."""
    profiles = profiles[:max_samples]
    results: List[DialogueResult] = []

    for i, profile in enumerate(profiles):
        if verbose:
            pid = profile.get("id", i)
            print(f"  [{i+1}/{len(profiles)}] profile={pid}")
        try:
            dr = generate_single_dialogue(
                model, tokenizer, profile,
                planning_llm_fn=planning_llm_fn,
                player_llm_fn=player_llm_fn,
                max_turns=max_turns,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                initial_emo_point=initial_emo_point,
                device=device,
            )
            results.append(dr)
        except Exception as e:
            print(f"    Warning: error on profile {profile.get('id', i)}: {e}")
    return results


# ── serialisation ───────────────────────────────────────────────────────

def save_dialogues(dialogues: List[DialogueResult], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for d in dialogues:
        records.append({
            "profile_id": d.profile_id,
            "dialogue_history": d.dialogue_history,
            "emo_point_trajectory": d.emo_point_trajectory,
            "initial_emo": d.initial_emo,
            "final_emo": d.final_emo,
            "npc_responses": d.npc_responses,
            "num_turns": d.num_turns,
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def load_dialogues(path: str | Path) -> List[DialogueResult]:
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    return [
        DialogueResult(
            profile_id=r["profile_id"],
            dialogue_history=r["dialogue_history"],
            emo_point_trajectory=r["emo_point_trajectory"],
            initial_emo=r["initial_emo"],
            final_emo=r["final_emo"],
            npc_responses=r["npc_responses"],
            num_turns=r["num_turns"],
        )
        for r in records
    ]
