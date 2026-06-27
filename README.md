# Predicting the Helpfulness and Harmlessness of LLM Responses

## Executive summary

When an AI chatbot answers a question, two things matter: was the answer **helpful** (did it actually
address what the person asked?) and was it **harmless** (did it avoid unsafe, misleading, or inappropriate
content?). Today, judging this still relies heavily on people reading responses one by one — slow, expensive,
and impossible to do at the scale modern AI systems operate.

This project asks a practical question: **can a computer learn to predict whether a response is the one a
human would prefer, using only measurable properties of the text itself** — its length, vocabulary, whether
it hedges or refuses — without a human in the loop at the moment of judgment?

Using a large public dataset of real human preferences ([Anthropic
HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf)), the project builds and compares several
machine-learning models. A first finding: **simple text features alone can guess the human-preferred
response correctly about 56–58% of the time — clearly better than a coin flip (50%), but far from perfect.**
That makes this approach valuable as a fast, cheap, fully explainable *first-pass filter*, while confirming
that a truly accurate system needs to understand the *meaning* of the text, not just its surface statistics.

A second, more interesting finding emerged: **the writing that makes a response "helpful" is often the
opposite of what makes it "harmless."** Helpful answers tend to be longer and more detailed; harmless answers
to sensitive questions tend to be shorter and more likely to politely refuse. Any real-world AI safety system
has to balance these two competing goals — and this project makes that tension measurable.

The project then shows how to do meaningfully better. The key realization: the data is a set of
**head-to-head pairs** (two responses to the same question, one preferred), so the model should compare the
two responses against each other rather than judge each in isolation. Making that one change — with no new
data — lifts the model's ranking quality (ROC-AUC) from **0.61 to 0.69 for helpfulness** and from **0.58 to
0.71 for harmlessness**, using only fast, classical models. It also sharpens the finding above: helpfulness
is driven by an answer's *form* (length, structure, staying on topic), while harmlessness is driven by its
*content* (which words and topics appear, and whether it refuses).

## Rationale

Every major AI lab spends heavily on **automated evaluation** of AI outputs — systems that score responses
to guide training and to catch quality or safety problems after a model is deployed, without paying a human
to review every single answer. A lightweight, *explainable* model that scores both helpfulness and
harmlessness, and can point to the specific features behind each score, would be directly useful in these
production pipelines. Just as importantly, it would make those pipelines **auditable** — a reviewer can see
*why* a response was flagged, instead of trusting an opaque black box.

## Research Question

Given a conversational prompt and a candidate AI response, can a machine-learning system predict whether that
response is the one a human would prefer — separately for **helpfulness** and **harmlessness** — and identify
which features of the response drove that prediction, all without a human label at the moment of judgment?

## Data Sources

[**Anthropic HH-RLHF**](https://huggingface.co/datasets/Anthropic/hh-rlhf) (Helpful and Harmless RLHF),
released under an MIT license and loaded directly via the Hugging Face `datasets` library — no manual
download required. The dataset contains roughly 161,000 human preference pairs: a conversational prompt plus
two candidate responses, with a human annotator's choice of the better one.

This project uses the `helpful-base` and `harmless-base` subsets (about 92K and 90K response rows
respectively), keeping the dataset's original train/test split so results are evaluated on data the models
never saw during training. The `helpful-online` and `helpful-rejection-sampled` subsets are left for future
work.

## Methodology

The work is delivered as a single Jupyter notebook that follows the data-science process end to end, from
raw data to an improved, evaluated model.

**Data cleaning and feature engineering.** Empty/malformed responses and exact-duplicate pairs are dropped;
repeated short responses (e.g. the same refusal phrase) are kept, since they reflect real recurring AI
behavior. Each response is then described by 19 interpretable features in three groups: *surface* (length,
vocabulary richness, hedging, refusal phrasing, reading ease), *structure* (code blocks, lists, capitalization,
digits), and *relevance* (how well the response's words overlap with the question).

**Exploratory analysis.** Class balance, response-length distributions (chosen vs. rejected), hedging/refusal
rates by outcome, and a feature-correlation heatmap.

**Baseline and tuned model comparison.** Logistic Regression, Random Forest, and XGBoost are each tuned with
`GridSearchCV` (5-fold cross-validation) and compared on the held-out test split — first using the natural
*pointwise* approach (judge each response on its own).

**The improvement — pairwise preference modeling.** The pointwise approach plateaus, so the notebook reframes
the task the way the data was actually labelled: as a *head-to-head comparison* between the two responses to
the same prompt (learning from their difference, the way real "reward models" are trained). This is tuned with
grouped cross-validation so a pair's two halves can never leak across folds, and includes a *leakage-safe*
check that removes any test response also seen in training. Every model — including the original pointwise one —
is scored on the same ranking task so the comparison is fair.

**Why these evaluation metrics?** Because every preference pair contributes exactly one "chosen" and one
"rejected" response, the classes are perfectly balanced — so **accuracy** is a fair, easy-to-explain headline
metric, with a coin-flip baseline of exactly 50%. **ROC-AUC** is reported alongside it as the metric most
aligned with the real use case (ranking better responses above worse ones, rather than performance at one
fixed cutoff), and **F1** serves as a sanity check against any class bias.

## Results

**In plain terms:** using nothing but measurable text properties — no AI understanding of meaning — the best
models correctly identify the human-preferred response **56–58% of the time**, consistently above a 50% coin
flip but well short of replacing human review. This is a useful, fast, fully explainable first filter, not a
final judge.

- **Tuning and stronger models gave a modest, not dramatic, lift over the simple baseline.** Tuned **XGBoost**
  reached **58.0% accuracy / 0.614 ROC-AUC** on helpfulness and **56.4% accuracy / 0.583 ROC-AUC** on
  harmlessness — a few points above the logistic-regression baseline (56.7% / 0.598 and 56.0% / 0.571).
- **No single model dominates.** The three models land within a couple of points of each other on every
  metric: XGBoost has the best ROC-AUC on helpfulness (0.614) and Random Forest the best on harmlessness
  (0.585), with the other always close behind. XGBoost was used for the detailed view (confusion matrix, ROC, feature
  importance) — but the tiny margins are a reminder that "best model" depends on which metric the downstream
  application actually cares about.
- **Cross-validated and test scores were nearly identical for every model**, confirming the models generalize
  and the comparison is not an artifact of overfitting.
- **There is a real ceiling on what surface features can do.** Across all model families, performance plateaus
  around 56–61% — confirming that the larger lever for improvement is *richer features that capture meaning*,
  not just more powerful models.
- **Helpfulness and harmlessness reward opposite kinds of writing.** Helpful responses tend to be longer and
  more varied; for harmlessness, *refusal* phrasing predicts being preferred while *length* predicts being
  rejected — the reverse of the helpfulness pattern. Where two very different models (linear and tree-based)
  agree on this, as they do for harmlessness, it's a particularly trustworthy finding.

**The improved approach** breaks through much of that ceiling without any new data or any deep learning:

- **Comparing the two responses head-to-head, instead of judging each alone, is the single biggest win — and
  it's essentially free.** Using the *same* features, this one change lifts ROC-AUC from 0.614 to 0.662 on
  helpfulness and from 0.583 to 0.614 on harmlessness. The earlier "ceiling" was largely an artifact of asking
  the model to judge responses in isolation, throwing away the question-matched pairing where the signal lives.
- **The best results raise ROC-AUC to 0.69 (helpfulness) and 0.71 (harmlessness)** — a substantial jump over
  the 0.61 / 0.585 starting point, still using only fast, classical models. The improvement is statistically
  significant: the baseline and improved 95% confidence intervals do not overlap (bootstrap, in the notebook).
- **The two goals are improved by different things — confirming the earlier insight more rigorously.**
  Helpfulness improves most from *form and relevance* features (length, structure, staying on topic);
  harmlessness improves most from *word-level content* (what topics appear, and whether the response refuses).
- **The content result is real, not a measurement artifact.** A "leakage-safe" check that removes any test
  response also seen in training barely changes the harmlessness number (0.714 → 0.711 ROC-AUC), so the gain
  is genuine.

Full code, charts, and detailed interpretation are in the notebook linked below.

## Next steps

- **Add features that capture meaning, not just words** — sentence-embedding features and prompt–response
  similarity, in the same head-to-head framing — the most promising remaining lever, especially for
  helpfulness.
- **Train a single dual-purpose model** that predicts helpfulness and harmlessness together, so it can
  directly balance the two competing objectives this project uncovered.
- **Train a transformer "reward model"** directly on the response pairs (the production-grade approach) and
  compare it against the fast classical models built here.
- **Add per-prediction explanations** (e.g. SHAP) so the model can explain *individual* decisions, not just
  overall feature importance.
- **Extend to the remaining helpfulness subsets** (`helpful-online`, `helpful-rejection-sampled`) to confirm
  the improvement holds beyond the base data.

## Outline of project

- [Predicting the Helpfulness and Harmlessness of LLM Responses](llm-helpfulness-harmlessness-modeling.ipynb)
  — the complete analysis: data preparation, EDA, baseline and tuned model comparison, the pairwise
  improvement, findings, and next steps.

## How to run

The notebook is self-contained and downloads its data automatically via the Hugging Face `datasets`
library — no manual data download needed. To run locally:

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn datasets jupyter
jupyter notebook llm-helpfulness-harmlessness-modeling.ipynb
```

## Contact and further information

For questions about this project, please open an issue on this repository.
