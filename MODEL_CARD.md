---
license: cc-by-sa-4.0
base_model: Qwen/Qwen3.5-4B
library_name: peft
language:
- pl
- en
tags:
- typed-decisions
- decision-model
- classification
- polish
- lora
- qlora
datasets:
- allegro/klej-polemo2-in
- allegro/klej-dyk
- allegro/klej-psc
- ipipan/polqa
- NASK-PIB/PLLuM-Align
- AmazonScience/massive
- mhardalov/exams
---

![Bullfinch 4B, a small decision model for Polish](bullfinch-banner.png)

# Bullfinch 4B (gil)

Bullfinch 4B is a small decision model for Polish. It reads a state (a text) and one typed
question, a choice, a yes/no or a 5-step score, and returns a probability for every option.
The answer comes from the option-letter logits of one forward pass; no text is generated.

It is a LoRA adapter on [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), trained only
on human or source labels from open Polish datasets, and it runs in 4-bit on a laptop GPU.
Code, converters and evaluation: [github.com/takzen/bullfinch](https://github.com/takzen/bullfinch).
Sibling of [Serin 4B](https://huggingface.co/Takzen/serin-4b), the judge for chatbot audits.

## Results

### Full test splits

Every official test split plus Belebele and INCLUDE, 24,168 rows, scored the same way for every
model: Bullfinch and Qwen through the Bullfinch engine (Bullfinch with its calibration), basal
2.0 4.5B through `basal-serve` (`/v1/systemone`, official GGUF Q8_0 build, its own calibration).
Accuracy in %. The first six sets had their training split in Bullfinch's training data (never
the test split); the last three did not.

| Set | Test rows | Qwen3.5-4B untuned | basal 2.0 4.5B | **Bullfinch 4B** |
|---|---|---|---|---|
| PolEmo-IN review sentiment (4) | 722 | 81.6 | 92.2 | **92.4** |
| DYK does the answer fit the question | 1,029 | 82.0 | 89.3 | **92.5** |
| PSC do two summaries describe one article | 1,078 | 76.5 | 92.2 | **98.9** |
| PolQA is the passage evidence | 13,729 | 76.3 | 83.4 | **86.6** |
| MASSIVE assistant scenario (18) | 2,974 | 74.8 | **89.0** | 87.1 |
| MASSIVE intent within the scenario | 2,694 | 87.6 | **93.0** | 92.8 |
| PolEmo-OUT sentiment, other domains | 494 | 76.7 | 77.9 | **78.7** |
| INCLUDE Polish exams (4) | 548 | 56.6 | **63.3** | 57.1 |
| Belebele reading (4) | 900 | 87.4 | **87.6** | 86.4 |
| **Mean of 9** | | 77.7 | 85.3 | **85.8** |
| Mean calibration error (ECE) | | 11.9% | 4.6% | **2.5%** |

- Against basal 2.0 4.5B: +0.5 points, paired bootstrap 95% CI −0.3 to +1.3, so the two models
  are level on accuracy; Bullfinch's probabilities are better calibrated. Bullfinch leads on
  five sets, basal on Polish exam knowledge (INCLUDE), the 18-way scenario and the intent.
- Against the untuned base: +8.1 points (CI +7.4 to +9.0).
- The 46 Polish items of OpenDecisions: 67.4% for Bullfinch and basal, 60.9% for the base.

### Development panel

The panel used to choose between training runs: 500 random test rows per set (seed 7), the same
rows for every model, uncalibrated. Accuracy in %.

| Set | Qwen3.5-4B untuned | basal 2.0 4.5B | **Bullfinch 4B** |
|---|---|---|---|
| PolEmo-IN review sentiment (4) | 81.0 | 91.4 | **92.8** |
| DYK does the answer fit the question | 83.0 | 91.2 | **94.0** |
| PSC do two summaries describe one article | 75.0 | 92.4 | **98.6** |
| PolQA is the passage evidence | 73.6 | 82.0 | **86.4** |
| MASSIVE assistant scenario (18) | 76.0 | **89.0** | 86.8 |
| MASSIVE intent within the scenario | 87.6 | 92.2 | **92.6** |
| PolEmo-OUT sentiment, other domains | 76.7 | 77.9 | **78.7** |
| INCLUDE Polish exams (4) | 56.2 | **63.0** | 56.4 |
| Belebele reading (4) | 87.2 | **87.8** | 85.6 |
| **Mean of 9** | 77.4 | 85.2 | **85.8** |

- With option-order averaging on the two 4-option sets for both models (Bullfinch `--tta`,
  basal `--orders 2`): 86.1 against 85.6, still level.

## How to use

The model reads a chat prompt with lettered options and an assistant turn prefilled with
`{"answer": "`; the decision is a softmax over the logits of the option letters that follow.

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE, REVISION = "Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
PREFILL = '{"answer": "'
FRAMES = {
    "pl": ("Stan", "Pytanie", "Opcje", 'Odpowiedz wyłącznie w formacie JSON: {"answer": "<litera opcji>"}.'),
    "en": ("State", "Question", "Options", 'Reply only with JSON: {"answer": "<option letter>"}.'),
}
TEMPERATURE = {"choice": 1.091, "noul": 1.149}  # fitted on the calibration split

tok = AutoTokenizer.from_pretrained(BASE, revision=REVISION)
model = AutoModelForCausalLM.from_pretrained(
    BASE, revision=REVISION, device_map={"": 0}, dtype=torch.bfloat16,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                           bnb_4bit_compute_dtype=torch.bfloat16,
                                           bnb_4bit_use_double_quant=True))
model = PeftModel.from_pretrained(model, "Takzen/bullfinch-4b").eval()
prefix = tok.encode(PREFILL, add_special_tokens=False)


def decide(state: str, question: str, options: dict[str, str], frame: str = "pl") -> dict[str, float]:
    """Probability of each option key for one choice question."""
    s, q, o, reply = FRAMES[frame]
    codes = dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", options))
    lines = "\n".join(f"{code}) {key}: {options[key]}" for code, key in codes.items())
    content = f"{s}:\n{state}\n\n{q}: {question}\n\n{o}:\n{lines}\n\n{reply}"
    text = tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    ids = tok(text + PREFILL, return_tensors="pt", add_special_tokens=False).input_ids.to(0)
    with torch.inference_mode():
        logits = model(input_ids=ids, logits_to_keep=1).logits[0, -1].float()
    letters = [tok.encode(PREFILL + code, add_special_tokens=False)[len(prefix)] for code in codes]
    probs = torch.softmax(logits[letters] / TEMPERATURE["choice"], dim=0)
    return dict(zip(codes.values(), probs.tolist()))


print(decide(
    state="Dzień dobry, za fakturę 4411 pobrano opłatę dwa razy. Proszę o zwrot nadpłaty jeszcze dziś.",
    question="Który dział powinien obsłużyć tę wiadomość?",
    options={"billing": "faktury, płatności, zwroty",
             "technical": "błędy, awarie, problemy z kontem",
             "sales": "ceny, nowe plany"},
))
# {'billing': 0.968, 'technical': 0.031, 'sales': 0.001}
# with "Od rana nie mogę się zalogować, aplikacja pokazuje błąd 500." as the state:
# {'billing': 0.001, 'technical': 0.998, 'sales': 0.001}
```

- **Yes/no questions** use the options `true` and `false`, listed by description only
  (`A) tak`, `B) nie`), and the `noul` temperature.
- **Scores** list five steps from `bardzo nisko` to `bardzo wysoko` (very low to very high),
  with the low and high anchors in brackets on the first and last line.
- **Option averaging:** score every cyclic rotation of the options in both frame languages,
  average the log-probabilities per option key, then softmax. On 4-option questions it added
  0.8 points on Belebele and 2.2 on INCLUDE, at 8 passes instead of 1.
- The repo has the full engine and a FastAPI server with the `/v1/systemone` interface that
  basal and Jev use (`src/bullfinch/engine.py`, `src/bullfinch/server.py`).

## Training

- QLoRA: base frozen in 4-bit NF4, LoRA r 32, alpha 64, dropout 0.05 on the attention,
  Gated DeltaNet and MLP projections.
- 9,000 examples, one pass, batch 16, learning rate 1e-4 with cosine decay; the adapter with
  the lowest NLL on a 596-row dev panel was kept (step 540 of 563). Trained on a laptop GPU.
- Every example is rendered with a random option order and a random frame language (Polish or
  English), so neither a letter's position nor the frame carries the answer. Each task has
  several Polish phrasings and one English one.
- Mix: PLLuM-Align 16%, PolQA-MC 15%, MASSIVE 14%, PolQA 14%, PolEmo-IN 12%, DYK 11%,
  EXAMS 11%, PSC 7%.
- The same rows trained with Unsloth's `FastDecisionModel` head instead of letter logits
  scored 1.85 points lower on the panel (CI −2.7 to −1.0), mostly on unseen 4-option sets.

## Data

Only sets with human or source labels. No test split, Belebele, INCLUDE or OpenDecisions item
was trained on, and neither was any training row that shares a text with a test row: in PSC
3,023 of the 4,302 training pairs reuse a test summary and 404 Polish EXAMS training questions
are INCLUDE test questions, so they were dropped.

| Source | Licence | Decision |
|---|---|---|
| [allegro/klej-polemo2-in](https://huggingface.co/datasets/allegro/klej-polemo2-in) | CC BY-SA 4.0 | review sentiment, 4 options |
| [allegro/klej-dyk](https://huggingface.co/datasets/allegro/klej-dyk) | CC BY-SA 3.0 | does the answer answer the question |
| [allegro/klej-psc](https://huggingface.co/datasets/allegro/klej-psc) | CC BY-SA 3.0 | do two summaries describe the same article |
| [ipipan/polqa](https://huggingface.co/datasets/ipipan/polqa) | CC BY-SA 4.0 | is the passage evidence; 4-option reading built from its answers (PolQA-MC) |
| [NASK-PIB/PLLuM-Align](https://huggingface.co/datasets/NASK-PIB/PLLuM-Align) | CC BY-SA 4.0 | which reply people preferred; 1–5 ratings of correctness and helpfulness |
| [AmazonScience/massive](https://huggingface.co/datasets/AmazonScience/massive) pl-PL | CC BY 4.0 | assistant scenario, then the intent within it |
| [mhardalov/exams](https://huggingface.co/datasets/mhardalov/exams) pl | CC BY-SA 4.0 | exam questions, multiple choice |

No outputs of commercial models or decision APIs were used. PLLuM-Align replies come from
open models only.

## Limitations

- Polish world knowledge is the base model's: on regional exam questions (INCLUDE) it is 6.2
  points behind basal 2.0 4.5B. Put the facts a decision needs in the state.
- The training covers sentiment, relevance, paraphrase, intent routing, preference and
  multiple choice. Check it on your own decisions before relying on it, and fit your own
  threshold where an error is costly.
- Score questions (5 steps) had few calibration rows; their probabilities are uncalibrated.
- Decisions with serious consequences for people should be reviewed by a person.

## Licence and attribution

The adapter is released under CC BY-SA 4.0, because most of the training data is CC BY-SA. The
base model is Qwen3.5-4B (Apache 2.0). Training data: KLEJ PolEmo 2.0, DYK and PSC (Allegro),
PolQA (IPI PAN), PLLuM-Align (NASK PIB), MASSIVE (Amazon) and EXAMS.
