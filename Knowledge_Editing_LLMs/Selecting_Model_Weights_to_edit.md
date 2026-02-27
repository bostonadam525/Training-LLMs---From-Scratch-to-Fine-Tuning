# Selecting Model Weights to Edit
* The paper "Fast Model Editing at Scale" by Mitchel et al. mentions their process for training: "Before training we select the weights of the model that they want to made editable.
* They state this could be the "weight matrices in the last M layers). [Source](https://arxiv.org/pdf/2110.11309)
* So, I thought to myself how do they pick the exact weights to edit?

---
# TL;DR

1. **“How do they know the exact weights?”**
* They usually don’t pick exact elements a priori.
* They localize a matrix or small set of matrices (often mid‑layer MLP projections at the subject token), then apply a low‑rank, closed‑form update (ROME/MEMIT) or a learned low‑rank update (MEND).

2. **“Would the deep learning library WeightWatcher help?”**
* Yes, as a complementary diagnostic: it can rank/prioritize layers and warn about unstable/pathological ones, guiding where to attempt edits — but you’d still use causal patching (task‑conditioned) to pinpoint the right place/time (token) for the edit and a model‑editing algorithm to compute the actual parameter delta.


---
## “Which weights do we edit?” 
* This is the crux of model editing.
* In practice, researchers don’t guess single scalars; they localize where a fact or behavior is computed (layer/module, sometimes token position), then apply a small, targeted update (often low‑rank) there.


### 1) Mechanistic localization (causal/activation patching)
* This involves running the model twice (clean vs. corrupted prompt), patch specific activations (layers, sublayers, tokens) from the clean run into the corrupted run, and see where the output flips back.
* This isolates which components mediate the behavior (e.g., “middle MLP blocks on the subject token”).
* Methods like causal tracing / activation patching are standard for this and were used in ROME’s setup to find that mid‑layer MLPs during subject‑token processing carry factual associations.
* Key Takeaways:
  * You usually don’t pick “exact weights” upfront
  * You need to localize the site (e.g., specific MLP layers) --> then edit a small subset (often one projection matrix) there.

* Sources:
  * Meng et al, 2023. Locating and Editing Factual Associations in GPT. [Link](https://arxiv.org/pdf/2202.05262)
  * Zhang et al, 2024. TOWARDS BEST PRACTICES OF ACTIVATION PATCHING IN LANGUAGE MODELS: METRICS AND METHOD. [Link](https://arxiv.org/pdf/2309.16042v2)

### 2) Method‑specific heuristics (ROME, MEMIT, MEND)
* ROME: After localization, apply a rank‑1 update to a specific feed‑forward (MLP) projection at a chosen mid layer.
  * It computes keys/values for the target subject and writes a new association in closed form into a single matrix. 

* MEMIT: Scales ROME‑style ideas to many edits, distributing updates across multiple layers to maintain specificity and generalization.
  * It chooses layers based on measured edit efficacy in practice. 

* MEND: Trains a small editor network that transforms the fine‑tuning gradient into a low‑rank parameter update; this avoids touching everything and makes edits fast/effective even for 10B+ models.
  * The editor learns where and how to update from data. 

* Key Takeaways:
  * These methods don’t require hand‑picking exact weights.
  * They either:
    * (a) localize components then do a closed‑form low‑rank update (ROME/MEMIT), or
    * (b) learn a mapping from gradients to compact updates (MEND).
   
* Sources:
  * Sources:
  * Meng et al, 2023. Locating and Editing Factual Associations in GPT. [Link](https://arxiv.org/pdf/2202.05262)
  * Meng et al, 2023. Mass-Editing Memory in a Transformer. [Link](https://arxiv.org/pdf/2210.07229)
  * Zhang et al, 2024. TOWARDS BEST PRACTICES OF ACTIVATION PATCHING IN LANGUAGE MODELS: METRICS AND METHOD. [Link](https://arxiv.org/pdf/2309.16042v2)
 
### 3) Prior knowledge about transformer internals
* A consistent result:
  * Transformer MLP layers act like key‑value memories.
  * That’s why many editors target MLP projections in middle layers, where factual retrieval seems to occur.

* Sources
  * Geva et al, 2021. Transformer Feed-Forward Layers Are Key-Value Memories. [Link](https://aclanthology.org/2021.emnlp-main.446.pdf)
 

### 4) Influence / sensitivity diagnostics (optional)
* Some teams use influence functions or related sensitivity tools to rank layers or parameters by their impact on a behavior.
* This is more data‑dependent and heavier, but can guide where to place LoRA/edits or to prioritize layers.
* (Caveat: classical IF has practical limitations; newer variants propose layer‑wise uses.)

* Sources:
  * Askari et al, 2025. LAYERIF: Estimating Layer Quality for Large Language Models using Influence Functions. [Link](https://arxiv.org/pdf/2505.23811)
  * Schioppa et al, 2023. Theoretical and Practical Perspectives on what Influence Functions Do. [Link](https://proceedings.neurips.cc/paper_files/paper/2023/file/57bb27b9be6ad04019ae3cea2b540872-Paper-Conference.pdf)
 
---
# Would `WeightWatcher` help pick the weights?
* [WeightWatcher](https://weightwatcher.ai/) is an open-source, diagnostic tool for analyzing Deep Neural Networks (DNN), without needing access to training or even test data.
* It could be potentially helpful, with caveats.
  * What it does well:
    * Data‑free spectral diagnostics of each weight matrix: heavy‑tailedness, α (alpha) exponents, empirical spectral density (ESD), etc.
    * It flags poorly trained or pathological layers (e.g., “rank collapse”, correlation traps), and shows where layers are in the HTSR “training phases”.
    * This can prioritize layers that are stable, well‑trained, or suspicious. [weightwatcher.ai], [weightwatcher.ai]

## How it could be used in LLM knowlede editing?
* If your goal is robust, minimal‑side‑effect edits, then you may prefer well‑behaved mid‑layer MLPs (e.g., α in a reasonable range), or avoid layers showing pathology.
* That’s a good triage signal for where an edit is likely to “stick” without collateral damage.

## What it does not do:
* WeightWatcher won’t tell you the exact rows/columns to change or which token‑position activations mediate the specific fact.
* It’s model‑centric, not task‑ or prompt‑centric.
* Typically you would pair WW with causal/activation patching to pick which layer/sublayer/token to target, then apply ROME/MEMIT/MEND there. 

## A practical workflow that you could replicate

1. **Localize the site for a specific edit**
  * Run activation patching / causal tracing on a few prompts that elicit the fact.
  * Identify layers+sublayers and token positions where patching restores the correct output.
  * Expect hits in mid‑layer MLPs.


2. **Sanity‑check candidate layers**
  * Run WeightWatcher to inspect those layers’ spectra/α.
  * Prefer layers that look well‑trained (heavy‑tailed, α≈2–4); consider avoiding layers with rank collapse or extreme outliers unless the goal is to repair them. 


3. **Pick an edit method**
* Single fact and you want surgical control → ROME (rank‑1 update on a specific MLP matrix).
* Many facts (hundreds–thousands) → MEMIT (multi‑layer, scaled edits). 
* Fast, repeatable edits with minimal computation → MEND (learned low‑rank gradient transform). 


4. **Validate for side effects**
* Check specificity (only the target changes), generalization (the change holds across phrasings), and fluency/perplexity drawdown (no broad degradation).
* These are the standard evals used in the editing literature. 

* Sources
  * Meng et al, 2023. Mass-Editing Memory in a Transformer. [Link](https://arxiv.org/pdf/2210.07229)
  * Meng et al, 2023. Locating and Editing Factual Associations in GPT. [Link](https://arxiv.org/pdf/2202.05262)
  * [WeightWatcher HTSR and the 5+1 Phases of Training](https://weightwatcher.ai/htsr.html)
  * Zhang et al, 2024. TOWARDS BEST PRACTICES OF ACTIVATION PATCHING IN LANGUAGE MODELS: METRICS AND METHOD. [Link](https://arxiv.org/pdf/2309.16042v2)
 

