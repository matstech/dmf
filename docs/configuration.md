# Configuration

The Deterministic Memory Framework is entirely configurable via the `dmf_settings.toml` file. This centralizes all magic numbers, weights, and thresholds to give you complete control over your agent's memory lifecycle without needing to modify the core code.

The configuration is loaded at startup by `config_loader.load_dmf_config()` and injected into every component via the `DMFConfig` class.

---

## `[nlp]`

Defines the models used for text processing and embedding vectorization. These settings are consumed by `NLPEngine`, `EmbeddingEngine`, and `InteractionMatrix`.

| Parameter     | Type     | Default                            | Description                                                                                                                                                                           |
| ------------- | -------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `spacy_model` | `string` | `"en_core_web_sm"`                 | The spaCy pipeline used by `NLPEngine` for tokenization, POS tagging, NER and dependency parsing. Must be installed separately via `python -m spacy download <model>`.                |
| `model_name`  | `string` | `"nomic-ai/nomic-embed-text-v1.5"` | The FastEmbed model used to generate dense vector embeddings for interactions and queries. Determines embedding quality and dimensionality.                                           |
| `vector_dim`  | `int`    | `768`                              | Native output dimension of the selected embedding model. Used by `EmbeddingEngine` and `InteractionMatrix` for vector shape validation. Must match the actual output of `model_name`. |

---

## `[scoring_weights]`

Calibrates the **Survival Score** ($\Omega$) computation. The survival score determines how "important" each interaction is, and directly drives temporal decay resistance and pruning decisions.

The score is computed as a sigmoid over a weighted linear combination of NLP-extracted signals:

$$
\Omega = \sigma\!\left(
  \alpha \cdot \text{ID}
  + \beta \cdot |S|
  + \gamma \cdot E_{\text{norm}}
  + \delta \cdot D
  + z_{\text{op}}
  - z_0
\right)
$$

where $\sigma(x) = \frac{1}{1 + e^{-x}}$ is the logistic sigmoid and $z_0$ is the `sigmoid_midpoint`.

The **Semantic Divergence** $D$ measures how much a new interaction drifts from the recent conversational context. It is derived from the **centroid** of the sliding embedding window:

$$
C = \frac{1}{W} \sum_{i=1}^{W} v_i
$$

$$
D = 1 - \cos(\theta) = 1 - \frac{v_{\text{new}} \cdot C}{\|v_{\text{new}}\| \cdot \|C\|}
$$

where $W$ is the `window_size`, $v_i$ are the vectors in the current window, and $v_{\text{new}}$ is the embedding of the incoming interaction. $D \in [0, 2]$: a value of 0 means the new interaction is perfectly aligned with the context, 1 means orthogonal, 2 means diametrically opposed.

The weights below are calibrated so that an average interaction (ID ~0.40, |S| ~0.15, $E_{\text{norm}}$ ~0.20, D ~0.15) produces $\Omega$ ~0.50, placing the sigmoid at maximum discriminatory power in the typical signal range.

### Content signal weights

| Parameter         | Type    | Default | Description                                                                                                                                                                                |
| ----------------- | ------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `alpha_density`   | `float` | `3.0`   | Weight for **Information Density** (ID). This is the dominant signal: interactions carrying dense, factual content score higher. Maps to `ScoringConfig.alpha`.                            |
| `beta_entities`   | `float` | `2.0`   | Weight for **Entity count** (E_norm). Higher named-entity density acts as a factual anchor, boosting retention. Maps to `ScoringConfig.gamma`.                                             |
| `gamma_sentiment` | `float` | `0.2`   | Weight for **Absolute Sentiment** (\|S\|). Emotionally charged interactions are slightly more memorable. Maps to `ScoringConfig.beta`.                                                     |
| `delta_technical` | `float` | `-2.5`  | Weight for **Semantic Divergence** (D). This value is intentionally negative: interactions that drift topically from the conversation thread are penalized. Maps to `ScoringConfig.delta`. |

### Sigmoid and normalization

| Parameter          | Type    | Default | Description                                                                                                                                                                          |
| ------------------ | ------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `sigmoid_midpoint` | `float` | `1.5`   | The z-value at which the sigmoid outputs exactly 0.5. Shifting this up makes the scoring more conservative (fewer high-Omega interactions); shifting it down makes it more generous. |
| `entity_cap`       | `int`   | `5`     | Raw entity count is clipped to `[0, entity_cap]` before normalization. Prevents a single entity-heavy message from dominating the score.                                             |

### Social floor

| Parameter          | Type    | Default | Description                                                                                                                                                                                                                                        |
| ------------------ | ------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `social_threshold` | `float` | `0.4`   | If an interaction's raw Omega falls below this value and is classified as social/rapport (e.g., "Thanks!", "Ok"), the social floor kicks in.                                                                                                       |
| `min_social_score` | `float` | `0.25`  | The minimum Omega assigned to social interactions that fall below `social_threshold`. By design this is lower than `social_threshold` -- it provides a floor that prevents premature eviction without inflating the score to an unjustified level. |

### Operational salience weights

These weights contribute to a pre-sigmoid operational channel. They are added to the content score before the final sigmoid is applied, boosting interactions that carry explicit conversational intent.

The operational contribution is computed as:

$$
\begin{aligned}
z_{\text{op}} = \lambda_{\text{op}} \cdot \big(
  & \; \eta_{\text{constraint}} \cdot \mathbb{1}_{\text{constraint}} \\
  & + \eta_{\text{preference}} \cdot \mathbb{1}_{\text{preference}} \\
  & + \eta_{\text{current state}} \cdot \mathbb{1}_{\text{current state}} \\
  & + \eta_{\text{correction}} \cdot \mathbb{1}_{\text{correction}} \\
  & + \eta_{\text{replacement}} \cdot \mathbb{1}_{\text{replacement}} \\
  & + \eta_{\text{past state}} \cdot \mathbb{1}_{\text{past state}} \big)
\end{aligned}
$$

| Parameter            | Type    | Default | Description                                                                                                                                            |
| -------------------- | ------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `lambda_operational` | `float` | `0.75`  | Global scaling factor for the entire operational channel. Controls how much operational signals influence the final score relative to content signals. |
| `eta_constraint`     | `float` | `1.20`  | Weight for **constraint** signals (e.g., "Never use external APIs"). Constraints carry the strongest operational weight by design.                     |
| `eta_preference`     | `float` | `0.70`  | Weight for **preference** signals (e.g., "I prefer dark mode").                                                                                        |
| `eta_current_state`  | `float` | `0.60`  | Weight for **current-state** signals (e.g., "I'm currently working on module X"). Helps recent operative memory persist.                               |
| `eta_correction`     | `float` | `0.90`  | Weight for **correction** signals (e.g., "Actually, the port is 8080, not 3000"). Corrections strongly affect salience.                                |
| `eta_replacement`    | `float` | `0.50`  | Weight for **replacement** signals, when an interaction explicitly supersedes a previous one.                                                          |
| `eta_past_state`     | `float` | `0.0`   | Weight for **past-state** signals. Neutral by default: conflict resolution belongs in memory policy, not in score inflation.                           |

### Structured provenance contributions

| Parameter                   | Type    | Default | Description                                                                                                              |
| --------------------------- | ------- | ------- | ------------------------------------------------------------------------------------------------------------------------ |
| `user_correction_boost`     | `float` | `0.15`  | Pre-sigmoid bonus applied when the interaction is a user-initiated correction.                                           |
| `preference_update_boost`   | `float` | `0.10`  | Pre-sigmoid bonus for interactions that update a user preference.                                                        |
| `constraint_boost`          | `float` | `0.10`  | Pre-sigmoid bonus for interactions that establish a hard constraint.                                                     |
| `corrected_by_user_penalty` | `float` | `0.0`   | Pre-sigmoid penalty applied to interactions that were later corrected by the user. Set to `0.0` by default (no penalty). |

---

## `[temporal_decay]`

Controls the exponential decay applied to the active memory over time. Higher-scoring interactions resist decay thanks to the inertia mechanism.

| Parameter             | Type    | Default | Description                                                                                                                                                                             |
| --------------------- | ------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `lambda_base`         | `float` | `0.035` | Base exponential decay rate. At inertia mu=1 (no resistance), the half-life is `ln(2) / lambda_base`, approximately 19.8 turns with the default value.                                  |
| `inertia_strength`    | `float` | `0.5`   | Inertia coefficient eta, range `[0, 1)`. Controls how strongly a high survival score resists temporal decay. With `0.5`, an interaction with Omega=0.80 decays at 60% of the base rate. |
| `hard_kill_threshold` | `float` | `0.05`  | Absolute eviction floor. Entries whose effective Omega drops below this value are unconditionally removed during periodic cleanup, regardless of tier or capacity pressure.             |

---

## `[memory_tiers]`

Defines the tier boundaries used by `TemporalMemory` to classify interactions for pruning decisions.

| Parameter      | Type    | Default | Description                                                                                                                                                                                            |
| -------------- | ------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `critical_max` | `float` | `0.3`   | Upper bound of the **CRITICAL** tier. Interactions with `Omega_eff <= critical_max` have the highest eviction priority and are removed first under capacity pressure.                                  |
| `unstable_max` | `float` | `0.6`   | Upper bound of the **UNSTABLE** tier. Reserved for future sub-tier reporting. Must satisfy `critical_max < unstable_max <= healthy_min`. Does not currently affect runtime pruning or tier assignment. |
| `healthy_min`  | `float` | `0.75`  | Lower bound of the **HEALTHY** tier. Interactions with `Omega_eff > healthy_min` are protected from pressure-based pruning and will only be removed by natural decay below `hard_kill_threshold`.      |

---

## `[capacity]`

Manages the physical constraints of the active memory window.

| Parameter             | Type  | Default | Description                                                                                                                                                                                    |
| --------------------- | ----- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `token_budget`        | `int` | `4096`  | Maximum total tiktoken tokens allowed across all active queue entries. When exceeded, entries are pressure-evicted in order of lowest `effective_pruning_score` first until the budget is met. |
| `pruning_frequency_x` | `int` | `5`     | Periodic cleanup runs every N turns. Controls how frequently the system scans for entries below `hard_kill_threshold` and enforces capacity limits.                                            |
| `window_size`         | `int` | `10`    | Maximum number of vectors retained in the `InteractionMatrix` sliding window. Affects semantic divergence computation.                                                                         |

---

## `[pruning_priority]`

Retention bonuses applied **only** when the active queue exceeds `token_budget`. These bonuses increase the effective pruning score of contextually important records, making them harder to evict under pressure.

The effective pruning score is computed as:

$$
\begin{aligned}
S_{\text{prune}} = \; & \Omega_{\text{eff}} \\
  & + \rho_{\text{constraint}} \cdot \mathbb{1}_{\text{constraint}} \\
  & + \rho_{\text{preference}} \cdot \mathbb{1}_{\text{preference}} \\
  & + \rho_{\text{current state}} \cdot \mathbb{1}_{\text{current state}} \\
  & + \rho_{\text{correction}} \cdot \mathbb{1}_{\text{correction}} \\
  & + \rho_{\text{replacement}} \cdot \mathbb{1}_{\text{replacement}} \\
  & - \rho_{\text{superseded}} \cdot \mathbb{1}_{\text{topic superseded}}
\end{aligned}
$$

Lower values are evicted first. These bonuses do **not** affect decay, periodic hard-kill, or tier assignment.

| Parameter                 | Type    | Default | Description                                                                                                                                                              |
| ------------------------- | ------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `rho_constraint`          | `float` | `0.20`  | Retention bonus for interactions flagged as hard constraints. Strongest protection.                                                                                      |
| `rho_preference`          | `float` | `0.10`  | Retention bonus for user preference declarations.                                                                                                                        |
| `rho_current_state`       | `float` | `0.10`  | Retention bonus for current-state information.                                                                                                                           |
| `rho_correction`          | `float` | `0.15`  | Retention bonus for corrections.                                                                                                                                         |
| `rho_replacement`         | `float` | `0.08`  | Retention bonus for replacement interactions.                                                                                                                            |
| `superseded_past_penalty` | `float` | `0.35`  | Penalty subtracted from the pruning score of older entries that have been **superseded** by a newer entry on the same topic. Makes outdated information easier to evict. |

---

## `[ltm]`

Configures the **Long-Term Memory** persistence backend. DMF supports multiple backends that can be swapped without changing application code.

| Parameter               | Type     | Default                    | Description                                                                                                                                                                          |
| ----------------------- | -------- | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `storage_type`          | `string` | `"chroma"`                 | Backend type. `"chroma"` for ChromaDB vector store with active recall, `"file"` for a write-only JSONL audit trail via `FileLTMHook`, `"null"` for silent discard (useful in tests). |
| `storage_path`          | `string` | `"data/ltm_archive.jsonl"` | Path to the JSONL archive file. Used when `storage_type = "file"`. Parent directories are created automatically.                                                                     |
| `chroma_path`           | `string` | `"data/ltm_chroma"`        | Persistence directory used only in `embedded` mode. It is not created by the server client.                                                                                          |
| `chroma_mode`           | `string` | `"embedded"`               | Chroma deployment: local `embedded` persistence or remote `server`.                                                                                                                  |
| `chroma_host`           | `string` | `"localhost"`              | Server hostname. Required and non-empty in active server mode.                                                                                                                       |
| `chroma_port`           | `int`    | `8000`                     | Server port in the range 1–65535.                                                                                                                                                    |
| `chroma_ssl`            | `bool`   | `false`                    | Use HTTPS for the server connection.                                                                                                                                                 |
| `chroma_tenant`         | `string` | `"default_tenant"`         | Chroma tenant used by embedded and server clients.                                                                                                                                   |
| `chroma_database`       | `string` | `"default_database"`       | Chroma database used by embedded and server clients.                                                                                                                                 |
| `chroma_auth_token_env` | `string` | `""`                       | Name of the environment variable containing an optional server Bearer token. The token itself must never be stored in TOML.                                                         |
| `collection_name`       | `string` | `"dmf_memory"`             | ChromaDB collection name for raw LTM records. Change this to start a fresh memory namespace.                                                                                         |
| `recall_limit`          | `int`    | `5`                        | Maximum number of raw records returned per active-recall search query.                                                                                                               |
| `distance_threshold`    | `float`  | `0.7`                      | Cosine-distance ceiling for recalled raw records, range `[0, 2]`. A value of `0.7` means `cosine_similarity > 0.3`, filtering to related results only.                               |
| `enabled`               | `bool`   | `true`                     | Master switch. Set to `false` to disable LTM persistence entirely and fall back to `NullLTMHook`.                                                                                    |
| `cards_enabled`         | `bool`   | `false`                    | Enables the auxiliary structured memory-card index. Raw LTM remains canonical; cards provide an additional retrieval path.                                                           |
| `cards_path`            | `string` | `"data/ltm_cards.jsonl"`   | Path to the memory-card JSONL index file.                                                                                                                                            |
| `cards_collection_name` | `string` | `"dmf_cards"`              | ChromaDB collection name for memory cards.                                                                                                                                           |

For deployment examples, direct construction, authentication behavior, retry
semantics, version compatibility, migration guidance, Docker integration, and
the local Ollama benchmark, see [LTM Backends](ltm_backends.md).

---

## `[retrieval]`

Configures the **structured retrieval pipeline** (candidate generation, answerability reranking, and evidence assembly). These settings are separate from the `[ltm]` recall parameters and control the advanced multi-stage retrieval stack.

| Parameter                            | Type   | Default | Description                                                                                                                                        |
| ------------------------------------ | ------ | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `card_prefetch_k`                    | `int`  | `32`    | Number of memory cards to prefetch during the candidate generation phase. A larger value increases recall coverage at the cost of processing time. |
| `raw_prefetch_k`                     | `int`  | `16`    | Number of raw LTM records to prefetch during candidate generation.                                                                                 |
| `symbolic_lookup_k`                  | `int`  | `16`    | Number of candidates retrieved via symbolic (keyword/topic) lookup.                                                                                |
| `final_recall_limit`                 | `int`  | `5`     | Hard cutoff applied after answerability reranking. Only the top N evidence items are returned to the caller.                                       |
| `max_support_turns_per_card`         | `int`  | `3`     | Maximum number of supporting raw turns expanded per winning memory card during evidence assembly.                                                  |
| `include_superseded_when_historical` | `bool` | `true`  | Whether to include superseded (outdated) entries when the query is classified as historical in nature.                                             |
| `include_neighbor_turns`             | `bool` | `false` | Whether to include neighboring turns (adjacent in the original conversation) when expanding evidence for a winning card.                           |
| `enable_raw_semantic`                | `bool` | `true`  | Enables the raw-record semantic retrieval channel (cosine similarity search on raw LTM embeddings).                                                |
| `enable_raw_lexical`                 | `bool` | `true`  | Enables the raw-record lexical retrieval channel (keyword-based matching).                                                                         |
| `enable_card_semantic`               | `bool` | `true`  | Enables the memory-card semantic retrieval channel.                                                                                                |
| `enable_card_symbolic`               | `bool` | `true`  | Enables the memory-card symbolic retrieval channel (topic and entity matching).                                                                    |
