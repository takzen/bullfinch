<p align="center"><img src="assets/bullfinch-banner.png" alt="Bullfinch 4B, a small decision model for Polish"></p>

# Bullfinch 4B (gil)

A small decision model for Polish. It reads a state and one typed question (choice, yes/no or a
score) and returns a probability for every option from the option-letter logits of a single
forward pass; no text is generated. It is the general-purpose sibling of
[Serin 4B](https://huggingface.co/Takzen/serin-4b), the judge for chatbot security audits, and is
built the same way: QLoRA on Qwen3.5-4B, trained and served on a laptop GPU.

**Status: pilot 3 trained, not released yet.** On a panel of nine Polish test sets it averages
85.8% accuracy, level with basal 2.0 4.5B (85.2%), the open Polish decision model, and +8.4 points
over its untuned base. Full-test numbers and the weights come next.

## Why

Small decision models (Jev and its open alternatives) are trained mostly on English. Their authors
say that in other languages they stay close to the base model. Serin showed that a 4B model tuned
on Polish decisions cuts false alarms of the base model from 131 to 5 out of 183 on its task.
Bullfinch tries the same for everyday Polish decisions, using human labels instead of synthetic
data, which is where Serin hit its ceiling.

## Data

Only sets with human or source labels and a licence that allows an open model. Official test
splits are never trained on, and neither is any training row that shares a text with a test row:
in PSC 3,023 of the 4,302 training pairs reuse a summary from the test split, and 404 of the 1,577
Polish EXAMS training questions are INCLUDE test questions, so they are dropped. The check ignores
case and punctuation and runs across sets that share a source.

| Set | Decision | Train / val / test | Licence |
|---|---|---|---|
| [allegro/klej-polemo2-in](https://huggingface.co/datasets/allegro/klej-polemo2-in) | review sentiment, 4 options | 5,783 / 723 / 722 | CC BY-SA 4.0 |
| [allegro/klej-polemo2-out](https://huggingface.co/datasets/allegro/klej-polemo2-out) | the same, other domains (test only) | – / 494 / 494 | CC BY-SA 4.0 |
| [allegro/klej-dyk](https://huggingface.co/datasets/allegro/klej-dyk) | does the answer answer the question, yes/no | 4,154 / – / 1,029 | CC BY-SA 3.0 |
| [allegro/klej-psc](https://huggingface.co/datasets/allegro/klej-psc) | do two summaries describe the same article, yes/no | 4,302 / – / 1,078 | CC BY-SA 3.0 |
| [ipipan/polqa](https://huggingface.co/datasets/ipipan/polqa) | is the passage evidence for the question, yes/no | 66,770 / 13,681 / 13,729 labelled pairs | CC BY-SA 4.0 |
| [NASK-PIB/PLLuM-Align](https://huggingface.co/datasets/NASK-PIB/PLLuM-Align) | which of two replies people preferred; human 1–5 ratings of correctness and helpfulness (replies from open models only) | 4,300 pairs, 1,834 ratings | CC BY-SA 4.0 |
| [AmazonScience/massive](https://huggingface.co/datasets/AmazonScience/massive) pl-PL | voice-assistant scenario (18 options), then the intent within it (2–9 options) | 11,514 / 2,033 / 2,974 | CC BY 4.0 |
| [mhardalov/exams](https://huggingface.co/datasets/mhardalov/exams) pl | vocational exam questions, multiple choice | 1,577 / 394 / – | CC BY-SA 4.0 |

Derived from PolQA: `polqa_mc` (`scripts/build_polqa_mc.py`) turns a relevant passage, its question
and gold answer plus three answers to other questions of the same entity type into a 4-option
reading item in the Belebele format (train and validation questions only; any question or passage
that matches a PolQA test or Belebele text is dropped).

Licence still to confirm before use: [allegro/klej-cbd](https://huggingface.co/datasets/allegro/klej-cbd)
(cyberbullying, yes/no, 10,041 / 1,000),
[allegro/klej-allegro-reviews](https://huggingface.co/datasets/allegro/klej-allegro-reviews)
(rating 1–5 as a score, 9,577 / 1,002 / 1,006) and
[allegro/klej-nkjp-ner](https://huggingface.co/datasets/allegro/klej-nkjp-ner) (entity type,
15,794 / 1,941 / 2,058).

**Evaluation only, never trained on:**
[facebook/belebele](https://huggingface.co/datasets/facebook/belebele) pol_Latn (900 reading
comprehension questions, CC BY-SA 4.0),
[CohereLabs/include-base-44](https://huggingface.co/datasets/CohereLabs/include-base-44) Polish
(548 regional-knowledge questions, Apache 2.0),
[jgeuter/OpenDecisions](https://huggingface.co/datasets/jgeuter/OpenDecisions) (2,352 Jev-style
decisions; its card asks not to train on it) and the official test split of every set above.

**Left out:** [utahnlp/x-fact](https://huggingface.co/datasets/utahnlp/x-fact), because its evidence
snippets are search results that often quote the fact-checker's verdict, so the label leaks into the
input; [Davlan/sib200](https://huggingface.co/datasets/Davlan/sib200), because its sentences
come from FLORES-200 and 635 of its 701 Polish training sentences appear word for word in the
Belebele passages we evaluate on; CDSC-E and CDSC-R, `sdadas/8tags` and `sdadas/ppc` (non-commercial licences), and
`decisionmodels/sn92-generalist-train-v1`, whose text and labels were written by commercial models.
Bullfinch is never trained on the outputs of commercial models or decision APIs.

## Results so far

Panel: 500 random test rows per set (seed 7, `scripts/evaluate.py --limit 500`), the same rows for
every model, each model asked through its own interface: the Qwen-based models through the
Bullfinch engine, basal through `basal-serve` (`/v1/systemone`, official GGUF Q8_0 build).
Accuracy in %; the first six sets had their training split in pilots 2 and 3, the last three did not.

| Set | Qwen3.5-4B | Serin 4B | basal 2.0 4.5B | Bullfinch pilot 1 | pilot 2 | **pilot 3** |
|---|---|---|---|---|---|---|
| PolEmo-IN sentiment | 81.0 | 80.4 | 91.4 | 91.4 | 90.8 | **92.8** |
| DYK answer fits question | 83.0 | 69.6 | 91.2 | 90.4 | 93.4 | **94.0** |
| PSC same article | 75.0 | 91.4 | 92.4 | 95.2 | 98.4 | **98.6** |
| PolQA passage relevance | 73.6 | 72.0 | 82.0 | 80.4 | 85.2 | **86.4** |
| MASSIVE scenario (18) | 76.0 | 82.8 | **89.0** | 75.0 | 86.8 | 86.8 |
| MASSIVE intent | 87.6 | 88.6 | 92.2 | 84.8 | 92.0 | **92.6** |
| PolEmo-OUT (other domains) | 76.7 | 73.7 | 77.9 | 74.3 | 78.5 | **78.7** |
| INCLUDE exams (4 options) | 56.2 | 54.0 | **63.0** | 45.8 | 59.2 | 56.4 |
| Belebele reading (4 options) | 87.2 | 77.4 | **87.8** | 70.2 | 86.4 | 85.6 |
| **Mean of 9** | 77.4 | 76.7 | 85.2 | 78.6 | 85.6 | **85.8** |

- Pilot 3 against basal: +0.6 points, paired bootstrap 95% CI −0.5 to +1.6, so a tie. Against
  the untuned base: +8.4. Averaging both models over option orders on the two 4-option sets does
  not change that (86.1 against 85.6).
- Pilot 1 (three sets only) picked the first option on 4-option questions (193 of 271 INCLUDE
  errors were "a"); training on every set removed that bias.
- Same rows, same base, two recipes: letter-logit QLoRA (pilot 2, 85.6) beat Unsloth's
  `FastDecisionModel` head (83.8; −1.85, CI −2.7 to −1.0), mostly on the unseen 4-option sets.
- Pilots: 1 = PolEmo, DYK, PSC (6,000 rows); 2 = every training set (8,000 rows, lr 1e-4);
  3 = pilot 2 plus `polqa_mc` (9,000 rows). QLoRA r 32 on Qwen3.5-4B, one pass, laptop GPU.

## Plan

1. Done: converters, baselines, three QLoRA pilots, the comparison with basal and with Unsloth.
2. Full test splits for the release model, calibration, model card.
3. Harder 4-option reading data, where basal still leads.
4. Publish as `Takzen/bullfinch-4b`, with every data source attributed; a 0.8B sibling after that.
