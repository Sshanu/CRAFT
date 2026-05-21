"""
EvoPrompt enhancer — Differential Evolution (DE) variant.

Guo et al., 2023 ("Connecting LLMs with Evolutionary Algorithms Yields Powerful
Prompt Optimizers"). The LLM acts as the evolutionary operator: given several
population members it identifies differing parts, mutates them, recombines, and
crosses over with a basic prompt.

In THIS codebase EvoPrompt is used *only* as the refiner/operator. The
population, fitness, selection and validation are CRAFT's own machinery:
  - population  = self.population_candidates (the NSGA-retained top_candidates,
                  each carrying eval_score from UCB-MO as its fitness)
  - selection   = NSGA-II (handled by the orchestrator's selector)
  - validation  = UCB-MO (handled by the evaluator)
EvoPrompt's native GA/DE population-update + roulette selection are NOT used —
they are replaced by the above. Same enchance() interface as Sculpt/OPRO.
"""

import os
import re
import traceback
from tqdm import tqdm

import numpy as np

import utils
import llm
import prompt_template


class EvoPrompt:
    """EvoPrompt DE operator as a CRAFT enhancer."""

    def __init__(self, config, logger, random_seed=42):
        self.strategy_name = "evoprompt"
        self.config = config
        self.logger = logger
        self.rng = np.random.default_rng(random_seed)
        self.random_seed = random_seed

        template_path = os.path.join(
            "src", "data", "prompt_enhancer", "evoprompt", "de_template.md"
        )
        with open(template_path, "r", encoding="utf-8") as f:
            self.de_template = f.read()

        self.current_round = 0
        self.max_rounds = 0
        self.prompt_universe_size = None
        # Set by the orchestrator before each enhance call — the NSGA-retained
        # population (PromptTree candidates with eval_score fitness).
        self.population_candidates = []

    def create_output_dir(self, output_dir, save_prefix):
        out = os.path.join(output_dir, "enchancer", self.strategy_name, save_prefix)
        os.makedirs(out, exist_ok=True)
        return out

    @staticmethod
    def _extract_prompt(response):
        """Pull text inside <prompt>...</prompt>. Falls back to the whole response."""
        if not response:
            return ""
        m = re.search(r"<prompt>\s*(.*?)\s*</prompt>", response, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
        # Fallback: strip any stray tags and use the body
        return response.replace("<prompt>", "").replace("</prompt>", "").strip()

    def _population_pool(self, exclude_id=None):
        """Population members as system-prompt text only.

        DE picks 3 RANDOM members — it never uses fitness — so no score (and in
        particular no test_score) is read here. Test scores stay blind to the
        framework; only the orchestrator's NSGA/UCB machinery touches eval_score.
        """
        texts = []
        for c in self.population_candidates:
            if exclude_id is not None and getattr(c, "id", None) == exclude_id:
                continue
            sys_text, _, _ = c.get_textual_prompt()
            texts.append(sys_text)
        return texts

    def enchance(self, prompt_universe_size, candidate, error_evaluations,
                 current_round, output_dir, enhancement_id):
        """Generate n_evoprompt_candidates offspring via the DE operator.

        Same signature as sculpt.Sculpt.enchance. `error_evaluations` is accepted
        but unused (EvoPrompt is population-based, not error-based).
        """
        self.prompt_universe_size = prompt_universe_size
        self.current_round = current_round
        token_usage_tracker = utils.TokenUsageDetails()
        token_usage_tracker.init_components(["actor", "overall"])

        sub_dir = self.create_output_dir(output_dir, f"candidate_{candidate.id}")
        sub_dir = os.path.join(sub_dir, f"enhancement_num_{enhancement_id}")
        os.makedirs(sub_dir, exist_ok=True)

        cur_sys, cur_conv, cur_user = candidate.get_textual_prompt()

        # Population pool (exclude the parent itself where possible). DE needs 3
        # distinct members; if the population is too small, sample with
        # replacement and/or fall back to the parent's own text.
        pool_texts = self._population_pool(exclude_id=candidate.id) or [cur_sys]

        n = int(getattr(self.config, "n_evoprompt_candidates", 4))
        temp = float(getattr(self.config, "evoprompt_temperature", 0.5))

        new_candidates = []
        for i in tqdm(range(n), total=n, desc=f"EvoPrompt-DE cand_{candidate.id}"):
            try:
                replace = len(pool_texts) < 3
                picks = list(self.rng.choice(len(pool_texts), size=3, replace=replace))
                p1, p2, p3 = (pool_texts[picks[0]], pool_texts[picks[1]], pool_texts[picks[2]])
                # Basic prompt = the parent being evolved (p_i).
                de_prompt = (
                    self.de_template
                    .replace("{prompt1}", p1)
                    .replace("{prompt2}", p2)
                    .replace("{prompt3}", p3)
                    .replace("{basic_prompt}", cur_sys)
                )

                with open(os.path.join(sub_dir, f"de_prompt_{i}.txt"), "w", encoding="utf-8") as f:
                    f.write(de_prompt)

                _, response, token_usage_i = llm.run_llm(
                    user_message=de_prompt,
                    system_prompt="You are an expert prompt engineer performing evolutionary prompt optimization.",
                    provider=self.config.provider,
                    model=self.config.model,
                    max_tokens=self.config.max_tokens,
                    temperature=temp,
                    top_p=1.0,
                    reasoning_effort=getattr(self.config, "reasoning_effort", "low"),
                    service_tier=getattr(self.config, "service_tier", "fast"),
                )
                token_usage_tracker.add_token_usage_for_component(token_usage_i, "actor")
                response = utils.clean_llm_response(response) if response else ""

                with open(os.path.join(sub_dir, f"de_response_{i}.txt"), "w", encoding="utf-8") as f:
                    f.write(response or "")

                evolved = self._extract_prompt(response)
                if not evolved or evolved == "Could not get a response":
                    self.logger.log(f"EvoPrompt empty/invalid response on attempt {i} for candidate {candidate.id}; skipping")
                    continue

                new_text = f"<|im_start|>system\n{evolved}\n<|im_end|>\n"
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
                self.logger.log(f"EvoPrompt-DE error (cand {candidate.id}, attempt {i}): {e}")
                self.logger.log(traceback.format_exc(), False)

        self.logger.log(f"EvoPrompt-DE: produced {len(new_candidates)}/{n} candidates for parent {candidate.id}")
        return new_candidates, token_usage_tracker
