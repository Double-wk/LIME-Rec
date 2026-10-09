"""LoRA SFT for the task-adapted controller (Qwen3-4B, validation-only).

Plain PyTorch loop for exact control over the token boundary: the prompt is the
Qwen3 chat rendering of [system, user] with enable_thinking=False (identical to
the frozen vLLM serving configuration); the completion is the target JSON plus
<|im_end|>. Loss is computed on completion tokens only.
"""
from __future__ import annotations

import argparse
import json
import math

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup


class SFTData(Dataset):
    def __init__(self, path, tokenizer, max_examples=0, max_len=10240):
        rows = [json.loads(l) for l in open(path).readlines()]
        if max_examples and len(rows) > max_examples:
            rows = rows[:max_examples]
        self.examples = []
        eos = tokenizer.convert_tokens_to_ids("<|im_end|>")
        for r in rows:
            msgs = [{"role": "system", "content": r["system"]},
                    {"role": "user", "content": r["prompt"]}]
            prompt_ids = tokenizer.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False)
            comp_ids = tokenizer(r["completion"], add_special_tokens=False)["input_ids"] + [eos]
            ids = (prompt_ids + comp_ids)[:max_len]
            labels = ([-100] * len(prompt_ids) + comp_ids)[:max_len]
            self.examples.append((ids, labels))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def collate(batch, pad_id):
    maxlen = max(len(ids) for ids, _ in batch)
    input_ids, labels, attn = [], [], []
    for ids, lab in batch:
        pad = maxlen - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        labels.append(lab + [-100] * pad)
        attn.append([1] * len(ids) + [0] * pad)
    return (torch.tensor(input_ids), torch.tensor(labels), torch.tensor(attn))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-examples", type=int, default=8000)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--attn-impl", default="flash_attention_2",
                        choices=["flash_attention_2", "sdpa", "eager"],
                        help="Attention backend; sdpa avoids a flash-attn install.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation=args.attn_impl).cuda()
    model.config.use_cache = False
    model.gradient_checkpointing_enable()  # needed: 10k-token logits dominate memory
    peft_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
                          task_type="CAUSAL_LM",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                          "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    data = SFTData(args.data, tokenizer, args.max_examples)
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=True,
                        collate_fn=lambda b: collate(b, tokenizer.pad_token_id or 0),
                        num_workers=2, drop_last=False)
    steps_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_steps = int(steps_per_epoch * args.epochs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(optimizer, int(0.03 * total_steps), total_steps)

    model.train()
    step = 0
    accum_loss = 0.0
    for epoch in range(math.ceil(args.epochs)):
        for i, (input_ids, labels, attn) in enumerate(loader):
            input_ids, labels, attn = input_ids.cuda(), labels.cuda(), attn.cuda()
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            (out.loss / args.grad_accum).backward()
            accum_loss += float(out.loss.detach())
            if (i + 1) % args.grad_accum == 0 or i + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                sched.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0:
                    print(f"[sft] step {step}/{total_steps} loss {accum_loss / 10 / args.grad_accum:.4f} "
                          f"lr {sched.get_last_lr()[0]:.2e}", flush=True)
                    accum_loss = 0.0
        if epoch + 1 >= args.epochs:
            break

    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"[sft] adapter saved -> {args.out}")


if __name__ == "__main__":
    main()
