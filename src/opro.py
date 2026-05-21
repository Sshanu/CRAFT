"""
OPRO (Optimization by PROmpting) enhancer — a minimal alternative to sculpt.

Instead of sculpt's critic+actor structural-edit loop, OPRO takes the top-K
prompts (by validation score) seen so far PLUS their scores, packs them into a
meta-prompt, and asks an LLM to generate `n_opro_candidates` new prompts that
should score higher.

Same `enchance()` interface as `sculpt.Sculpt` so the orchestrator can swap
between them via `enhancer.type=opro`.
"""

import os
import traceback
from tqdm import tqdm

import numpy as np

import utils
import llm
import prompt_template


class OPRO:
    """OPRO meta-prompting enhancer (Yang et al., 2023 — simplified)."""

    def __init__(self, config, logger, random_seed=42):
        self.strategy_name = "opro"
        self.config = config
        self.logger = logger
        self.rng = np.random.default_rng(random_seed)
        self.random_seed = random_seed

        template_path = os.path.join(
            "src", "data", "prompt_enhancer", "opro", "opro_template.md"
        )
        with open(template_path, "r", encoding="utf-8") as f:
            self.opro_template = f.read()

        self.current_round = 0
        self.max_rounds = 0
        self.prompt_universe_size = None
        # Set by the orchestrator before each enhance call. List of PromptTree
        # candidates that have been validated this run.
        self.population_candidates = []

    def create_output_dir(self, output_dir, save_prefix):
        out = os.path.join(output_dir, "enchancer", self.strategy_name, save_prefix)
        os.makedirs(out, exist_ok=True)
        return out

    def _build_history_block(self, top_k):
        """Top-k validated past prompts sorted ASCENDING by eval_score (best last)."""
        scored = [c for c in self.population_candidates if getattr(c, "eval_score", None) is not None]
        if not scored:
            return ""
        scored.sort(key=lambda c: c.eval_score)
        chosen = scored[-int(top_k):]
        parts = []
        for c in chosen:
            sys_text, _, _ = c.get_textual_prompt()
            parts.append(f"## Prompt (validation_score={c.eval_score:.4f})\n{sys_text}")
        return "\n\n".join(parts)

    def enchance(self, prompt_universe_size, candidate, error_evaluations,
                 current_round, output_dir, enhancement_id):
        """Generate n_opro_candidates new prompts from history+current.

        Same signature as sculpt.Sculpt.enchance so the orchestrator can call
        either interchangeably. `error_evaluations` is accepted but unused by
        OPRO (history-based, not error-based feedback).
        """
        self.prompt_universe_size = prompt_universe_size
        self.current_round = current_round
        token_usage_tracker = utils.TokenUsageDetails()
        token_usage_tracker.init_components(["actor", "overall"])

        sub_dir = self.create_output_dir(output_dir, f"candidate_{candidate.id}")
        sub_dir = os.path.join(sub_dir, f"enhancement_num_{enhancement_id}")
        os.makedirs(sub_dir, exist_ok=True)

        cur_sys, cur_conv, cur_user = candidate.get_textual_prompt()

        top_k = getattr(self.config, "top_k_history", 5)
        history_block = self._build_history_block(top_k)
        if not history_block:
            history_block = (
                "(no prior validated prompts yet — only the current one is available)\n\n"
                f"## Current prompt (no score yet)\n{cur_sys}"
            )

        meta_prompt = self.opro_template.replace("{history}", history_block).replace(
            "{current_prompt}", cur_sys
        )

        with open(os.path.join(sub_dir, "opro_meta_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(meta_prompt)

        new_candidates = []
        n = int(getattr(self.config, "n_opro_candidates", 4))
        base_temp = float(getattr(self.config, "opro_temperature", 0.7))

        for i in tqdm(range(n), total=n, desc=f"OPRO gen cand_{candidate.id}"):
            try:
                # Each of the n calls sends an IDENTICAL meta_prompt by default,
                # so the LLM cache (keyed on prompt+params) returns the SAME
                # cached response for calls 2..n -> all n candidates collapse to
                # one. Fix: give every call a distinct, SEEDED variation so the
                # cache key differs and the model is nudged to diversify.
                # self.rng is np.random.default_rng(random_seed) -> reproducible.
                var_temp = round(float(self.rng.uniform(base_temp - 0.2, base_temp + 0.2)), 3)
                var_temp = min(max(var_temp, 0.0), 1.5)
                var_meta = meta_prompt + (
                    f"\n\n[Variation {i + 1} of {n}: produce a DISTINCT rewrite — "
                    f"different in structure and wording from the other variations.]"
                )
                _, response, token_usage_i = llm.run_llm(
                    user_message=var_meta,
                    system_prompt="You are an expert prompt engineer.",
                    provider=self.config.provider,
                    model=self.config.model,
                    max_tokens=self.config.max_tokens,
                    temperature=var_temp,
                    top_p=1.0,
                    reasoning_effort=getattr(self.config, "reasoning_effort", "low"),
                    service_tier=getattr(self.config, "service_tier", "fast"),
                )
                token_usage_tracker.add_token_usage_for_component(token_usage_i, "actor")
                response = utils.clean_llm_response(response) if response else ""

                with open(os.path.join(sub_dir, f"opro_response_{i}.txt"), "w", encoding="utf-8") as f:
                    f.write(response or "")

                if not response or response.strip() == "" or response == "Could not get a response":
                    self.logger.log(f"OPRO empty response on attempt {i} for candidate {candidate.id}; skipping")
                    continue

                # Rebuild full chat-template text with the new system prompt,
                # preserving conversation+user_message from the parent.
                new_text = f"<|im_start|>system\n{response}\n<|im_end|>\n"
                for turn in (cur_conv or []):
                    new_text += (
                        f"<|im_start|>user\n{turn.get('User','')}\n<|im_end|>\n"
                        f"<|im_start|>assistant\n{turn.get('Assistant','')}\n<|im_end|>\n"
                    )
                new_text += f"<|im_start|>user\n{cur_user or ''}\n<|im_end|>\n<|im_start|>assistant"

                new_id = self.prompt_universe_size
                new_candidate = prompt_template.PromptTree(
                    new_id, new_text, candidate.model, candidate.provider,
                    output_dir, parent_id=candidate.id, round_created=current_round,
                    formatting=getattr(candidate, "formatting", False),
                )
                self.prompt_universe_size += 1
                new_candidates.append(new_candidate)
            except Exception as e:
                self.logger.log(f"OPRO generation error (cand {candidate.id}, attempt {i}): {e}")
                self.logger.log(traceback.format_exc(), False)

        self.logger.log(f"OPRO: produced {len(new_candidates)}/{n} candidates for parent {candidate.id}")
        return new_candidates, token_usage_tracker
