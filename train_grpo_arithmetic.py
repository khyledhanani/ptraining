"""
train_grpo_arithmetic.py

GRPO training using arithmetic sampling (Vilnis et al., ICML 2023) in place of
the default i.i.d. completion sampling in TRL's GRPOTrainer.

How it works
------------
TRL's dataloader already repeats each prompt G = num_generations times, so at
generation time the input tensor is [B*G, L] where G consecutive rows share the
same prompt.  The standard path draws G independent uniform code points and
samples tokens from them (i.i.d.).  Arithmetic sampling instead uses a single
shared random shift u ~ Uniform[0,1) per prompt group and spaces the G code
points as an evenly-spaced lattice:

    c_{b,g} = (g + u_b) / G,    g = 0 … G-1

At every autoregressive step, token t is selected as the first token whose
cumulative probability F(t) >= c, and c is renormalized into the chosen
subinterval ready for the next step:

    c_new = (c - F(t-1)) / p(t)

Spreading code points across [0,1) reduces duplicate / near-duplicate outputs
and lowers variance when estimating reward expectations, while each sample is
still decoded independently (embarrassingly parallel).  The estimator is
unbiased because the lattice is randomly shifted.
"""

import torch
import torch.nn.functional as F
from contextlib import nullcontext

from trl import GRPOTrainer, GRPOConfig
from trl.rewards import accuracy_reward
from trl.data_utils import is_conversational
from trl.extras.profiling import profiling_context
from trl.models.utils import unwrap_model_for_generation
from transformers import Trainer as _BaseTrainer
from datasets import load_dataset


# ---------------------------------------------------------------------------
# Arithmetic sampling generation loop
# ---------------------------------------------------------------------------

def arithmetic_generate(
    model,
    input_ids,        # [N, L]   N = B*G; G consecutive rows share the same prompt
    attention_mask,   # [N, L]
    num_generations,  # G  — group size for code-point assignment
    max_new_tokens,
    temperature=1.0,
    top_k=None,
    top_p=None,
    eos_token_id=None,
    pad_token_id=None,
):
    """
    Drop-in replacement for model.generate() that uses an arithmetic code-point
    lattice to produce diverse completions within each prompt group.

    Returns [N, L+T] in the same format as HuggingFace model.generate().
    """
    N, L = input_ids.shape
    G = num_generations
    B = N // G
    device = input_ids.device

    # One shared random shift per prompt group: u_b ~ Uniform[0, 1)
    u = torch.rand(B, device=device)                                    # [B]

    # Code point for row b*G + g:   c = (g + u_b) / G
    g_idx    = torch.arange(G, device=device, dtype=torch.float64)     # [G]
    cp_2d    = (g_idx.unsqueeze(0) + u.double().unsqueeze(1)) / G      # [B, G]
    code_pts = cp_2d.reshape(-1)                                        # [N]

    finished   = torch.zeros(N, dtype=torch.bool, device=device)
    generated  = []
    past_kv    = None
    curr_mask  = attention_mask
    next_token = None

    for step in range(max_new_tokens):
        if finished.all():
            break

        with torch.no_grad():
            if step == 0:
                out  = model(input_ids=input_ids, attention_mask=curr_mask, use_cache=True)
            else:
                out  = model(
                    input_ids=next_token.unsqueeze(1),  # [N, 1]
                    attention_mask=curr_mask,
                    past_key_values=past_kv,
                    use_cache=True,
                )
        past_kv = out.past_key_values
        logits  = out.logits[:, -1, :]   # [N, V]

        # ── Sampling filters ────────────────────────────────────────────────
        if temperature != 1.0:
            logits = logits / temperature

        if top_k is not None and top_k > 0:
            k = min(top_k, logits.size(-1))
            topk_vals, _ = torch.topk(logits, k, dim=-1)
            logits = logits.masked_fill(logits < topk_vals[:, -1:], float('-inf'))

        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove    = (cum_probs - F.softmax(sorted_logits, dim=-1)) > top_p
            sorted_logits = sorted_logits.masked_fill(remove, float('-inf'))
            logits = logits.scatter(1, sorted_idx, sorted_logits)

        # ── Arithmetic token selection ───────────────────────────────────────
        probs = F.softmax(logits, dim=-1)      # [N, V]
        cdf   = torch.cumsum(probs, dim=-1)    # [N, V]

        # Select first token t where F(t) >= code_point
        cp         = code_pts.unsqueeze(1).to(probs.dtype)   # [N, 1]
        next_token = (cdf >= cp).long().argmax(dim=-1)        # [N]

        if pad_token_id is not None:
            next_token = next_token.masked_fill(finished, pad_token_id)

        # Renormalize:  c_new = (c - F(t-1)) / p(t)
        hi    = cdf.gather(1, next_token.unsqueeze(1)).squeeze(1)       # [N]
        lo_i  = (next_token - 1).clamp(min=0)
        lo    = cdf.gather(1, lo_i.unsqueeze(1)).squeeze(1)             # [N]
        lo    = lo.masked_fill(next_token == 0, 0.0)
        p_t   = (hi - lo).clamp(min=1e-10)                              # numerical floor
        code_pts = (
            (code_pts.to(probs.dtype) - lo) / p_t
        ).clamp(0.0, 1.0).double()

        generated.append(next_token)

        # ── EOS / padding tracking ───────────────────────────────────────────
        if eos_token_id is not None:
            eos_ids = [eos_token_id] if isinstance(eos_token_id, int) else list(eos_token_id)
            is_eos  = torch.zeros(N, dtype=torch.bool, device=device)
            for eid in eos_ids:
                is_eos = is_eos | (next_token == eid)
            finished = finished | is_eos

        new_col   = (~finished).long().unsqueeze(1)
        curr_mask = torch.cat([curr_mask, new_col], dim=1)

    completion = (
        torch.stack(generated, dim=1) if generated
        else input_ids.new_zeros(N, 0)
    )
    return torch.cat([input_ids, completion], dim=1)   # [N, L+T]


# ---------------------------------------------------------------------------
# Trainer subclass
# ---------------------------------------------------------------------------

class ArithmeticGRPOTrainer(GRPOTrainer):
    """
    GRPOTrainer with arithmetic sampling replacing i.i.d. sampling.

    Only the standard transformers generation path is modified.
    vLLM, paged-attention, and custom rollout_func paths are forwarded
    to the parent unchanged.
    """

    def _generate_single_turn(self, prompts):
        # Delegate all non-standard paths to the parent
        if (
            self.use_vllm
            or self.use_transformers_paged
            or getattr(self, "rollout_func", None) is not None
        ):
            return super()._generate_single_turn(prompts)

        device = self.accelerator.device
        mode   = "train" if self.model.training else "eval"

        # ── Tokenize (identical to parent) ──────────────────────────────────
        if is_conversational({"prompt": prompts[0]}):
            generate_inputs = self.processing_class.apply_chat_template(
                conversation=prompts,
                tools=self.tools,
                chat_template=self.chat_template,
                add_generation_prompt=True,
                tokenize=True,
                padding=True,
                padding_side="left",
                return_tensors="pt",
                return_dict=True,
                **self.chat_template_kwargs,
            )
        else:
            generate_inputs = self.processing_class(
                text=prompts, padding=True, padding_side="left", return_tensors="pt"
            )
        generate_inputs = _BaseTrainer._prepare_inputs(self, generate_inputs)

        gc              = self.generation_config
        num_generations = self.num_generations if mode == "train" else self.num_generations_eval

        # ── FSDP context (mirrors parent) ────────────────────────────────────
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            fsdp_ctx = (
                FSDP.summon_full_params(self.model_wrapped, recurse=False)
                if self.is_fsdp_enabled else nullcontext()
            )
        except ImportError:
            fsdp_ctx = nullcontext()

        # ── Arithmetic sampling (replaces unwrapped_model.generate) ──────────
        with (
            profiling_context(self, "arithmetic.generate"),
            unwrap_model_for_generation(
                self.model_wrapped,
                self.accelerator,
                gather_deepspeed3_params=self.args.ds3_gather_for_generation,
                generation_kwargs=self.generation_kwargs,
            ) as unwrapped_model,
            torch.no_grad(),
            fsdp_ctx,
        ):
            prompt_completion_ids = arithmetic_generate(
                model=unwrapped_model,
                input_ids=generate_inputs["input_ids"],
                attention_mask=generate_inputs["attention_mask"],
                num_generations=num_generations,
                max_new_tokens=self.max_completion_length,
                temperature=gc.temperature if gc.temperature is not None else 1.0,
                top_k=gc.top_k,
                top_p=gc.top_p,
                eos_token_id=self.eos_token_id,
                pad_token_id=self.pad_token_id,
            )

        # ── Post-process completions (identical to parent) ───────────────────
        prompt_ids, prompt_mask = (
            generate_inputs["input_ids"],
            generate_inputs["attention_mask"],
        )
        prompt_length  = prompt_ids.size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]

        is_eos     = completion_ids == self.eos_token_id
        eos_idx    = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        seq_indices    = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (seq_indices <= eos_idx.unsqueeze(1)).int()

        prompt_ids     = [p[m].tolist() for p, m in zip(prompt_ids,     prompt_mask.bool(),     strict=True)]
        completion_ids = [c[m].tolist() for c, m in zip(completion_ids, completion_mask.bool(), strict=True)]

        return prompt_ids, completion_ids, None, {}


# ---------------------------------------------------------------------------
# Training script
# ---------------------------------------------------------------------------

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
dataset    = load_dataset("trl-lib/DeepMath-103K", split="train").select(range(5000))

training_args = GRPOConfig(
    output_dir="grpo-qwen-math-arithmetic",
    run_name="arithmetic-sampling",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=1,
    learning_rate=1e-6,
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=10,
    save_steps=100,
    save_total_limit=1,
    max_completion_length=512,
    num_generations=8,
    seed=42,
    report_to="wandb",
    dataloader_num_workers=4,
)

trainer = ArithmeticGRPOTrainer(
    model=model_name,
    args=training_args,
    train_dataset=dataset,
    reward_funcs=accuracy_reward,
)

trainer.train()
trainer.save_model()
