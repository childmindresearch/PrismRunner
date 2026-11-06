#!/usr/bin/env python3
"""
SemanticPrism Stage 4: Synthesis Pipeline (Powered by PrismRunner)
==================================================================
Drop-in replacement for SemanticPrism's Stage 4 Ontological Schema Synthesis pipeline
(https://github.com/childmindresearch/SemanticPrism), replacing Ollama / pydantic-ai
with PrismRunner's local HuggingFace batch inference engine.

Workflow Phases & Output Artifacts:
  - Phase 0 : Dynamic Target Resolution & Architectural Tier Partitioning
              * outputs/03_topology/03_paths/community/resolved_community_targets.json
              * outputs/03_topology/03_paths/community/community_architectural_tiers.json
              * outputs/03_topology/03_paths/community/community_architectural_tiers.csv
              * outputs/03_topology/03_paths/community/tier_partitioning_diagnostics.json
  - Phase 1 : Semantic Enum Vector Clustering & Synthesis
              * outputs/04_synthesis/logs/enum_clusters_community.json
              * outputs/04_synthesis/schemas/community/normalized/enums.py
              * outputs/04_synthesis/schemas/community/raw/enums.py
  - Phase 2 : Community Path Schema Synthesis (Normalized & Raw Passes via PrismRunner Batch Inference)
              * outputs/04_synthesis/schemas/community/normalized/<01_module_name>.py ...
              * outputs/04_synthesis/schemas/community/raw/<01_module_name>.py ...
  - Phase 3 : Global Schema Consolidation & AST Class Deduplication
              * outputs/04_synthesis/schemas/community/normalized/master_ontology.py
              * outputs/04_synthesis/schemas/community/raw/master_ontology.py
  - Phase 4 : Master Comprehensive Ontology Synthesis
              * outputs/04_synthesis/schemas/community/comprehensive_ontology.py
              * outputs/04_synthesis/logs/synthesis_community_errors.log
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import csv
from datetime import datetime
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings

# Prevent PyTorch CUDA allocator fragmentation when switching between embedding and LLM stages
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
warnings.filterwarnings("ignore", message=r".*n_jobs value.*overridden to 1.*", category=UserWarning)

import numpy as np
from pydantic import BaseModel, Field, model_validator
from sklearn.cluster import AgglomerativeClustering

# Ensure local prismrunner package is importable when running from usertests/
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prismrunner import (
    ContextBudget,
    PrismPipeline,
    clear_device_memory,
    get_optimal_device,
    resolve_dynamic_batch_size,
)

try:
    import tiktoken
    _TIKTOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")

    def estimate_tokens(text: str) -> int:
        """Counts exact tokens using tiktoken (cl100k_base)."""
        if not text:
            return 0
        return len(_TIKTOKEN_ENCODER.encode(text))
except ImportError:
    def estimate_tokens(text: str) -> int:
        """Heuristic-based estimation: 1 word ~ 1.3 tokens."""
        if not text:
            return 0
        return int(len(text.split()) * 1.3)


# ============================================================================
# GLOBAL CONFIGURATION VARIABLES
# ============================================================================
# Edit these variables directly to configure models, hardware, paths, and
# synthesis hyperparameters.
#
# NOTE:
#   - MODEL_NAME      : Generative Causal/Coding LLM used by PrismRunner for Pydantic v2
#                       schema synthesis and ontology consolidation (e.g., "Qwen/Qwen3-Coder-Next",
#                       "Qwen/Qwen3.6-27B", or "Qwen/Qwen2.5-Coder-32B-Instruct").
#   - EMBEDDING_MODEL : SentenceTransformer vector encoder used for concept embeddings
#                       (e.g., "BAAI/bge-large-en-v1.5" [1024d] or "all-MiniLM-L6-v2" [384d]).

# --- 1. Generative LLM, Embedding Model & Hardware Settings ---
MODEL_NAME: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct"  # Causal/Coding LLM for Stage 4 (256K native context)
EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"        # Vector encoder for Phase 0 & Phase 1 concept vectors
DEVICE: str = "auto"                                   # "auto", "cuda", "mps", or "cpu"
TORCH_DTYPE: str = "auto"                              # "auto", "bfloat16", "float16", or "float32"
LOAD_IN_4BIT: bool = False                             # Enable 4-bit NF4 quantization (CUDA bitsandbytes)
LOAD_IN_8BIT: bool = False                             # Enable 8-bit quantization (CUDA bitsandbytes)
ATTN_IMPLEMENTATION: str = "auto"                      # "auto", "flash_attention_2", "sdpa", or "eager"
BATCH_SIZE: Union[int, str] = "auto"                   # "auto" (continuous device-relative) or explicit int
EMBEDDING_BATCH_SIZE: Union[int, str] = "auto"         # "auto" (dynamically sized from device memory) or explicit int
CONTEXT_WINDOW_CAP: Union[int, str] = "auto"           # "auto" = full native model context window (e.g. 262,144 tokens), or int cap
MAX_NEW_TOKENS: int = 16384                            # Max generated tokens per Phase 1/2 schema module
CONSOLIDATION_MAX_NEW_TOKENS: int = 32768              # Max generated tokens for Phase 3/4 consolidated master ontologies
TEMPERATURE: float = 0.0                               # 0.0 for deterministic greedy decoding
REPETITION_PENALTY: float = 1.08                       # Mild penalty to prevent combinatorial Enum repetition loops
CONSTRAIN_JSON: bool = True                            # Force JSON '{' prefix & bracket-balanced early stopping
MANAGE_VRAM: bool = True                               # Flush and offload GPU/MPS memory between embedding & LLM phases
DRY_RUN: bool = False                                  # Set True to preview prompts without running LLM generation

# --- 2. Input & Output Paths ---
REFINED_TRIPLETS_PATH: str = "outputs/02_refinement/refined_triplets.json"
ORIGINAL_TRIPLETS_PATH: str = "outputs/01_extraction/original_triplets.json"
TAXONOMIC_MAP_PATH: str = "outputs/02_refinement/taxonomic_map.json"
MASTER_THEMES_PATH: str = "outputs/01_extraction/master_themes.json"
TOPOLOGY_PARTITIONS_PATH: str = "outputs/03_topology/03_paths/community/topology_partitions.json"
RESOLVED_TARGETS_OUTPUT_DIR: str = "outputs/03_topology/03_paths/community"
VECTOR_CACHE_DIR: str = "outputs/02_refinement/cache"
OUTPUTS_DIR: str = "outputs"
SYNTHESIS_OUTPUT_DIR: str = "outputs/04_synthesis"
SAVE_LLM_PAYLOADS: bool = True
PAYLOAD_OUTPUT_DIR: str = "outputs/payload_text/stage_04_synthesis"
PRINT_CHECKPOINTS: bool = True                       # Print real-time timestamped checkpoint messages to console
RESUME_EXISTING_SCHEMAS: bool = False                # If True, skips Phase 2 targets whose .py files already exist

# --- 3. Execution Scope & Target Qualification (Phase 0 & Phase 2) ---
EXECUTION_MODE: str = "community"
SCHEMA_PASS_MODE: str = "both"                       # "both", "normalized", or "raw"
PAYLOAD_VALUES_MODE: str = "raw_text"                # "raw_text", "triples", or "both"
MIN_CLUSTER_SIZE: int = 15                           # Minimum entity count required for a community to receive its own schema
MAX_TRIPLETS_PER_TARGET: int = 1200                  # Maximum triplets per schema target payload (PageRank intra-community cap)

# --- 4. Orphan & Enum Vector Clustering & Atomic Bounds (Phase 1: enums.py) ---
ENUM_CLUSTERING_ENABLED: bool = True
ENUM_MIN_CLUSTER_SIZE: int = 4
ENUM_DISTANCE_THRESHOLD: float = 0.48
ENUM_MIN_COHESION_SCORE: float = 0.55
ENUM_MAX_CLUSTERS_PER_BATCH: int = 25
ENUM_MAX_TERMS_PER_CLUSTER: int = 15                 # Max candidate terms fed per Enum cluster
MAX_ENUM_MEMBER_WORDS: int = 4                       # Max words per Enum member identifier (<= 3 underscores)
MAX_ENUM_MEMBER_CHARS: int = 35                      # Max character length per Enum member identifier
MAX_ENUM_MEMBERS_PER_CLASS: int = 12                 # Max distinct members allowed per Enum class

# --- 5. Domain Term Scoring (C-TF-IDF) ---
TOP_DOMAIN_TERMS_COUNT: int = 5
DOMAIN_TERMS_IDF_EXPONENT: float = 1.5
DOMAIN_TERMS_MULTI_WORD_BOOST: float = 1.2
GENERIC_EXCLUSIONS: List[str] = [
    "person", "name", "support", "organization", "thing", "things", "away", "-",
    "other", "none", "user", "people", "system", "systems", "someone", "something",
    "anyone", "anything", "everyone", "everything", "item", "items", "concept", "concepts",
    "state", "states", "data", "info", "information", "value", "values", "lot", "lots",
    "bit", "bits", "day", "days", "time", "times", "year", "years", "week", "weeks",
    "part", "parts", "case", "cases", "type", "types", "category", "categories", "cat",
]

# --- 6. Thematic Observation Evidence Scoring (Phase 2 Context Quotes) ---
MAX_SENTENCES_PER_TARGET: int = 200
MIN_SENTENCE_LENGTH: int = 15
OBSERVATION_OVERLAP_WEIGHT: float = 0.25
OBSERVATION_OVERLAP_MAX_CAP: int = 4

# --- 7. Topology & Architectural Tier Settings (Used by Phase 0 TargetResolver) ---
CENTER_AGGLOMERATION_EMBEDDINGS: bool = True
TRIPLET_PAYLOAD_MODE: str = "strictly_intra_only"    # "include_boundary" or "strictly_intra_only"
EXCLUDE_HUB_TRIPLETS_FROM_COMMUNITIES: bool = False
ARCHITECTURAL_TIERS_ENABLED: bool = True
TIER_WEIGHT_PROTOTYPICALITY: float = 0.40
TIER_WEIGHT_CENTRALITY: float = 0.40
TIER_WEIGHT_FIDELITY: float = 0.20
TIER_1_PERCENTILE: float = 0.80
TIER_1_MIN_ENTITIES: int = 2
TIER_1_MAX_ENTITIES: int = 8
TIER_2_PERCENTILE: float = 0.30
MIN_FIDELITY_TIER_1: float = 0.10
BOUNDARY_DISCOUNT: float = 0.5
FALLBACK_PROTOTYPICALITY: float = 0.30
FALLBACK_FIDELITY: float = 0.28
EXPORT_TIER_CSV: bool = True
EXPORT_DIAGNOSTICS: bool = True

logger = logging.getLogger("SemanticPrism4")


# ============================================================================
# STAGE 4 PYDANTIC SCHEMAS (Exact match with src/synthesis/schemas.py)
# ============================================================================

class GeneratedModule(BaseModel):
    module_name: str = Field(
        description="The functional title abstracting the core semantic nodes, formatted strictly as a snake_case string (e.g., 'core_database_architecture')."
    )
    source_code: str = Field(
        description="The raw Python source code containing the generated Pydantic BaseModels and Enums, including all necessary import statements."
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_module_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            mod_name = (
                data.get("module_name")
                or data.get("name")
                or data.get("filename")
                or data.get("module")
                or "domain_schema"
            )
            src_code = (
                data.get("source_code")
                or data.get("code")
                or data.get("python_code")
                or data.get("content")
                or ""
            )
            out = dict(data)
            out["module_name"] = str(mod_name).strip()
            out["source_code"] = str(src_code).strip()
            return out
        return data


# ============================================================================
# STAGE 4 SYSTEM PROMPTS (Exact match with src/synthesis/prompts.py)
# ============================================================================

ORPHAN_SYNTHESIS_SYSTEM_PROMPT = """You are an Ontological Structuring Agent.
Analyze the provided semantic clusters and synthesize a cohesive suite of reusable, standardized Python `Enum` and `Literal` classes.

INPUT STRUCTURE:
You will receive `candidate_clusters`, each containing:
- `cluster_id` (int): Numeric identifier of the cluster.
- `category_hint` (str): Representative anchor concept (use as inspiration for the Enum class name).
- `cohesion_score` (float): Semantic density score of the cluster.
- `terms` (list of str): Concept terms to consolidate into discrete standardized members.

CRITICAL MODELING RULES:
1. **Lean into Slight Semantic Abstraction:** Do NOT mechanically transcribe hyper-specific, anecdotal, or idiosyncratic phrases verbatim (e.g., avoid `FOUR_HOURS_OF_SLEEP_IN_FORTY_EIGHT_HOURS` or `TAKING_MEDS_WITH_MY_PRISTIQ`). Instead, abstract them into canonical, generalizable domain concepts (e.g., `ACUTE_SLEEP_DEPRIVATION`, `MORNING_ADMINISTRATION`).
2. **Strict Atomic Enum Bounds (NO Sentence Fragments or Permutations):**
   - Each `Enum` member identifier MUST be an atomic concept of **at most 4 words (`<= 3` underscores, `<= 35` characters)**, such as `VERY_LOW`, `LOW`, `LOW_AVERAGE`, `AVERAGE`, `HIGH_AVERAGE`, `VERY_HIGH`, or `WITHIN_NORMAL_LIMITS`.
   - NEVER emit conversational or prepositional phrase permutations such as `WITHIN_THE_AVERAGE_RANGE_FOR_HIS_AGE_GROUP...`, `..._AS_COMPARED_TO_SAME_AGED_PEERS`, or `..._FOR_THE_SPELLING_SUBTEST`.
   - Each `Enum` class MUST contain **at most 10 distinct members**.
3. **Consolidate Synonyms:** Group terms sharing the same core meaning into a single standardized Enum member or Literal option.
4. **Discard Outliers:** If a term is an ungeneralizable conversational fragment, discard it.
5. **Valid Python Identifiers & Typing:**
   - Define Enums inheriting from `(str, Enum)` for seamless serialization (e.g., `class SymptomSeverity(str, Enum):`).
   - Class names MUST use `PascalCase`.
   - Enum member names MUST be valid Python identifiers in `UPPER_SNAKE_CASE` (alphanumeric and underscores only).
   - Use clean, human-readable string values for members.
6. **Self-Contained Code:** The code must include all required imports (`from enum import Enum`, `from typing import Literal`).

OUTPUT SCHEMA INSTRUCTIONS:
Return a JSON object conforming to the GeneratedModule schema:
- `module_name`: Set strictly to "enums".
- `source_code`: The complete, executable Python code containing all defined Enum and Literal classes."""


SCHEMA_SYNTHESIS_PROMPT = """You are an Expert Domain Modeling and Ontological Architect.
You will receive an extracted thematic cluster of domain concepts and real-world source text observations.
Your job is to interpret the underlying domain workflows, entities, and attributes, and synthesize a clean, practical, and coherent Pydantic (v2) schema module.

INPUT STRUCTURE:
1. `target_domain_focus`: Primary conceptual focus or theme defining this cluster.
2. `top_domain_terms`: Key domain concepts and vocabulary defining this cluster. Use these to guide the naming of classes, fields, and core entities.
3. `thematic_observations`: Authentic source text sentences organized by theme. Study these narrative observations to understand real-world workflows, data types, entity interactions, and field constraints.

CRITICAL MODELING RULES:
1. **Synthesize Domain Workflows, Do Not Transcribe:** Do NOT mechanically turn every term into a separate class. Step back, understand the domain narrative, and design 1 or more intuitive, cohesive Pydantic `BaseModel` classes modeling the core subject matter (e.g., ClinicalEvaluation, AssessmentProtocol, PatientCondition).
2. **Model Real-World Attributes:** Derive concrete attributes and fields directly from the details in `thematic_observations` (e.g., status, duration, severity, modalities, operational metrics).
3. **Use Atomic Enums/Literals with Semantic Abstraction (NO Sentence-Fragment Enums):**
   - Type fields representing discrete categories, severities, or classifications using Python `Enum` or `Literal` types instead of open-ended strings.
   - Every `Enum` member MUST be a short, atomic concept of **at most 4 words (`<= 3` underscores, `<= 35` characters)** (e.g., `LOW`, `LOW_AVERAGE`, `AVERAGE`, `HIGH_AVERAGE`, `VERY_HIGH`), with **at most 10 members per Enum class**.
   - NEVER bake pronouns, age groups, comparison clauses, or specific subtest names into Enum values (e.g., NEVER generate `WITHIN_THE_AVERAGE_RANGE_FOR_HIS_AGE_GROUP_FOR_THE_SPELLING_SUBTEST`). Instead, decompose them into separate Pydantic fields (e.g., `subtest_name: Optional[str]`, `score_range: Optional[PsychometricScoreRange]`, `normative_comparison: Optional[str]`).
4. **Field Descriptions & Optionality:** Declare model attributes using `Field(default=None, description="...")`, providing a concise explanation of the domain meaning derived from `thematic_observations`.
   - Field names MUST be valid Python identifiers in `snake_case` (alphanumeric and underscores only, NEVER hyphens like `field-name`).
5. **Self-Contained & Valid Imports:** The source code must include all necessary imports:
   `from pydantic import BaseModel, Field`
   `from typing import Optional, List, Dict, Any, Union, Literal`
   `from enum import Enum`
6. **Strictly Isolated Module:** Model only the concepts and workflows substantiated by the provided data.
7. **ABSOLUTE BAN ON META-MODELING:** You MUST model the authentic real-world domain phenomena described INSIDE the observations. NEVER define models that describe the prompt itself, the JSON payload wrapper, or the synthesis framework. Specifically:
   - NEVER create classes named `ThemeObservation`, `ThematicObservation`, `DomainTerm`, or `GeneratedModule`.
   - NEVER create fields named `theme_name`, `observations`, or `source_code`.
   Any class that attempts to model the prompt structure itself is strictly invalid.

OUTPUT SCHEMA INSTRUCTIONS:
Return a JSON object conforming to the GeneratedModule schema:
- `module_name` (string): A concise, descriptive `snake_case` name for this module (e.g., "patient_assessment_workflow").
- `source_code` (string): The complete, executable Python code defining the Pydantic BaseModels and any localized Enums."""

LEIDEN_SCHEMA_SYNTHESIS_PROMPT = SCHEMA_SYNTHESIS_PROMPT


CONSOLIDATION_SYSTEM_PROMPT = """You are a Master Ontologist.
Merge, deduplicate, and consolidate the provided fragmented Pydantic schema files into a unified `master_ontology.py` module.

CRITICAL RULES:
1. **Deduplicate & Merge (Strict Class Name Uniqueness):** Identify classes modeling the same core concept across files. Merge their fields into ONE single unified class using the most descriptive and comprehensive names.
   - NEVER define two classes with the exact same name in the file (e.g., do NOT emit multiple `class EmotionalDysregulationAssessment`).
   - NEVER define an Enum and a BaseModel sharing the exact same class name.
2. **Atomic Enum Hygiene:** Keep `Enum` classes concise and canonical (`<= 10` members per Enum, `<= 4` words / `<= 35` characters per member). Discard any verbose sentence-fragment Enum members (e.g., `WITHIN_THE_AVERAGE_RANGE_FOR_HIS_AGE_GROUP...`).
3. **Preserve Hierarchy & Relationships:** Maintain valid class inheritance and typed cross-model references.
4. **Preserve Field Descriptions:** Retain attributes declared as `Field(default=None, description="...")`, preserving the most detailed and descriptive `description` string across merged attributes.
5. **Valid Python Identifiers:** All attribute names must be valid Python identifiers in `snake_case` (alphanumeric and underscores only, NEVER hyphens like `help-seeking_barriers`).
6. **Self-Contained Imports:** Ensure all necessary Pydantic and typing imports are included at the top of the file.

OUTPUT SCHEMA INSTRUCTIONS:
Return a JSON object conforming to the GeneratedModule schema:
- `module_name`: Set strictly to "master_ontology".
- `source_code`: The complete, consolidated Python source code."""


FINAL_ONTOLOGY_SYSTEM_PROMPT = """You are the Lead Master Ontologist.
Merge the provided `enums.py`, `raw_master_ontology.py`, and `normalized_master_ontology.py` into a single standalone `comprehensive_ontology.py` module.

CRITICAL RULES:
1. **Embed Enums Directly (Concise & Atomic Only):** Copy and define `Enum` and `Literal` classes from `enums.py` at the top of the file (`<= 10` atomic members per Enum, `<= 4` words per member; discard any long conversational phrase members). Do not use external or relative imports.
2. **Merge Classes & Strict Uniqueness:** Combine raw and normalized models by unioning their fields into single, comprehensive classes.
   - NEVER declare multiple classes with the same name.
   - NEVER define an Enum and a BaseModel with the same name.
3. **Standardize Fields & Descriptions:** Keep all attributes declared with `Field(default=None, description="...")`, ensuring informative descriptions are retained across models, and leverage embedded Enums/Literals for typed fields.
4. **Valid Python Identifiers:** All attribute names must be valid Python identifiers in `snake_case` (alphanumeric and underscores only, NEVER hyphens).
5. **No Synthetic Identifiers:** Ensure no synthetic `id`, `uuid`, or surrogate key fields are generated unless explicitly substantiated by source domain data.
6. **Formatting:** Output pure, executable Python code with all necessary imports.

OUTPUT SCHEMA INSTRUCTIONS:
Return a JSON object conforming to the GeneratedModule schema:
- `module_name`: Set strictly to "comprehensive_ontology".
- `source_code`: The complete, standalone Python source code."""


SCHEMA_REFORMAT_SYSTEM_PROMPT = """You are an expert Ontological Schema Repair Assistant.
You will receive a Python module attempt and the specific validation or syntax error that occurred.
Your job is to repair the Python code to strictly conform to the expected GeneratedModule schema structure.

TARGET SCHEMA:
```json
{
  "module_name": "string (snake_case identifier)",
  "source_code": "string (valid, executable Python source code)"
}
```

REPAIR RULES:
1. Fix all Python syntax errors, indentation issues, or missing imports (`from pydantic import BaseModel, Field`, `from enum import Enum`, `from typing import ...`).
2. Ensure all Pydantic fields use valid syntax: `field_name: Optional[Type] = Field(default=None, description="...")`.
3. Ensure Enum member names use valid `UPPER_SNAKE_CASE` identifiers, inherit from `(str, Enum)`, and contain **at most 10 concise atomic members (`<= 4` words / `<= 35` characters each)**. Strip any runaway or conversational Enum members (such as `WITHIN_THE_AVERAGE_RANGE_FOR_HIS_AGE_GROUP...`).
4. ABSOLUTE BAN ON META-MODELS: If the malformed input or code contains meta-models like `class ThemeObservation` or `class GeneratedModule` inside `source_code`, DISCARD them completely. Replace them with substantive domain Pydantic models based on the domain concepts and key terms.
5. Return pure JSON matching the GeneratedModule schema only. Do NOT output markdown explanations or conversational text."""


# ============================================================================
# TARGET RESOLVER (Exact match with src/topology/resolver.py)
# ============================================================================

class TargetResolver:
    """
    Target qualification and triplet filtering criteria used by Stage 4 for LLM synthesis.
    Provides methods to resolve qualified spoke targets, rank/cap global hubs, aggregate enum nodes,
    and filter/cap triplet payloads based on PageRank and intra-cluster edge priority.
    """
    _memory_cache: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}
    _raw_matrix_cache: Dict[Tuple[str, str], Tuple[np.ndarray, Dict[str, int], List[str]]] = {}

    @staticmethod
    def qualify_and_resolve_targets(
        topology: Dict[str, Any],
        path_type: str = "community",
        config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Resolves qualified community targets (>= min_cluster_size),
        and aggregates low-density cluster nodes, orphans, and global hubs into all_enum_nodes.
        """
        min_cluster_size = config.get("synthesis", {}).get("min_cluster_size", 15) if config else 15

        # 1. Global hubs are routed directly into Enum synthesis (enums.py)
        raw_hubs = list(topology.get("global_hubs", []))

        # 2. Aggregate Orphans + Low-Density Nodes + Global Hubs (Enums)
        orphans = list(topology.get("orphans", []))
        small_cluster_nodes = []

        for comm in topology.get("communities", []):
            nodes = comm.get("nodes", [])
            if len(nodes) < min_cluster_size:
                small_cluster_nodes.extend(nodes)

        all_enum_nodes = sorted(list(set(orphans + small_cluster_nodes + raw_hubs)))

        # 3. Consolidate Qualified Community Targets (>= min_cluster_size)
        targets = []
        raw_hubs_set = set(raw_hubs)
        for comm in topology.get("communities", []):
            nodes_in_comm = comm.get("nodes", [])
            clean_nodes = [n for n in nodes_in_comm if n not in raw_hubs_set]
            if len(clean_nodes) >= min_cluster_size:
                c_id = comm.get("community_id")
                targets.append({
                    "type": "community",
                    "id": c_id,
                    "nodes": clean_nodes,
                    "merged_from": comm.get("merged_from", [c_id]),
                })

        targets.sort(key=lambda x: len(x["nodes"]), reverse=True)

        # Invariant: Mutually Disjoint Community and Enum Partitioning
        all_target_nodes = set()
        for t in targets:
            all_target_nodes.update(t["nodes"])
        clean_enum_nodes = sorted([n for n in all_enum_nodes if n not in all_target_nodes])

        return targets, clean_enum_nodes

    @staticmethod
    def filter_and_cap_triplets(
        target_nodes: Any,
        triplets: Any,
        topology: Optional[Dict[str, Any]] = None,
        max_triplets_cap: int = MAX_TRIPLETS_PER_TARGET,
        target_id: Any = "global",
        path_type: str = "community",
        config: Optional[Dict[str, Any]] = None,
        exclude_hubs: Optional[bool] = None,
        raw_hubs: Optional[Iterable[str]] = None,
        strictly_intra_only: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """
        Filters triplets incident to target_nodes and prioritizes them using a
        pure Local Community Centrality engine.
        """
        # Gracefully handle if caller passed (triplets, target_nodes) in reverse order
        if isinstance(target_nodes, list) and target_nodes and isinstance(target_nodes[0], dict):
            target_nodes, triplets = triplets, target_nodes

        target_nodes_norm = {str(n).lower().strip() for n in target_nodes}

        if config:
            sem_cfg = (
                config.get("topology", {}).get("semantic_clustering", {})
                if "topology" in config
                else config.get("semantic_clustering", {})
            )
        else:
            sem_cfg = {}
        if exclude_hubs is None:
            exclude_hubs = sem_cfg.get("exclude_hub_triplets_from_communities", False)
        if strictly_intra_only is None:
            strictly_intra = sem_cfg.get("triplet_payload_mode", "include_boundary") == "strictly_intra_only"
        else:
            strictly_intra = bool(strictly_intra_only)

        resolved_hubs = set(raw_hubs) if raw_hubs is not None else (set(topology.get("global_hubs", [])) if topology else set())
        raw_hubs_norm = {str(h).lower().strip() for h in resolved_hubs}

        local_degree = Counter()
        edge_frequencies = Counter()

        for t in triplets:
            subj = str(t.get("subject", "")).lower().strip()
            obj = str(t.get("object", "")).lower().strip()
            pred = str(t.get("predicate", "")).strip()

            if not subj or not obj:
                continue

            if exclude_hubs and (subj in raw_hubs_norm or obj in raw_hubs_norm):
                continue

            is_subj_in = subj in target_nodes_norm
            is_obj_in = obj in target_nodes_norm

            if is_subj_in and is_obj_in:
                local_degree[subj] += 1
                local_degree[obj] += 1
                edge_frequencies[(subj, pred, obj)] += 1
            elif not strictly_intra and (is_subj_in or is_obj_in):
                edge_frequencies[(subj, pred, obj)] += 1

        scored_triplets = []
        for t in triplets:
            subj = str(t.get("subject", "")).lower().strip()
            obj = str(t.get("object", "")).lower().strip()

            if not subj or not obj:
                continue

            if exclude_hubs and (subj in raw_hubs_norm or obj in raw_hubs_norm):
                continue

            is_subj_in = subj in target_nodes_norm
            is_obj_in = obj in target_nodes_norm

            if strictly_intra:
                if not (is_subj_in and is_obj_in):
                    continue
            else:
                if not (is_subj_in or is_obj_in):
                    continue

            pred = str(t.get("predicate", "")).strip()
            freq = edge_frequencies.get((subj, pred, obj), 1)
            freq_boost = 1.0 + 0.1 * min(10, freq - 1)

            if is_subj_in and is_obj_in:
                local_sum = local_degree[subj] + local_degree[obj]
                rank_score = 1000.0 + (local_sum * freq_boost)
            else:
                internal_node = subj if is_subj_in else obj
                anchor_degree = local_degree[internal_node]
                rank_score = 1.0 + (anchor_degree * freq_boost)

            t_copy = dict(t)
            t_copy["triplet_rank_score"] = round(float(rank_score), 4)
            scored_triplets.append((rank_score, t_copy))

        scored_triplets.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored_triplets[:max_triplets_cap]]

    @staticmethod
    def _get_or_load_concept_embeddings(
        nodes: List[str],
        cache_dir: Path = Path("outputs/02_refinement/cache"),
        model_name: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Loads L2-normalized concept embeddings from refinement caches (subject, object, concept),
        and encodes any missing terms using the active SentenceTransformer model, then frees GPU memory.
        """
        cache_dir.mkdir(parents=True, exist_ok=True)
        if not model_name:
            for meta_f in sorted(cache_dir.glob("*_meta.json")):
                try:
                    with open(meta_f, "r", encoding="utf-8") as mf:
                        m_data = json.load(mf)
                    if "model_name" in m_data and m_data["model_name"]:
                        model_name = m_data["model_name"]
                        break
                except Exception:
                    pass
            if not model_name:
                model_name = EMBEDDING_MODEL or "all-MiniLM-L6-v2"

        model_slug = re.sub(r"[^a-zA-Z0-9_]", "_", model_name)
        cache_key = (str(cache_dir.resolve()), model_slug)

        if cache_key in TargetResolver._memory_cache:
            node_map = TargetResolver._memory_cache[cache_key]
            norm_nodes = [str(n).strip() for n in nodes if str(n).strip()]
            missing = [n for n in norm_nodes if n.lower().strip() not in node_map]
            if not missing:
                return node_map
        else:
            node_map = {}

        concept_npy = cache_dir / f"concept_{model_slug}_embeddings.npy"
        concept_meta = cache_dir / f"concept_{model_slug}_meta.json"

        # 1. Try unified concept cache first
        if concept_npy.exists() and concept_meta.exists():
            try:
                with open(concept_meta, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                vocab = meta.get("vocab", [])
                arr = np.load(concept_npy)
                if len(vocab) == len(arr):
                    for idx, term in enumerate(vocab):
                        node_map[str(term).lower().strip()] = arr[idx]
            except Exception as e:
                print(f"      -> Warning reading concept cache: {e}")

        norm_nodes = [str(n).strip() for n in nodes if str(n).strip()]
        missing = [n for n in norm_nodes if n.lower().strip() not in node_map]

        if not missing:
            TargetResolver._memory_cache[cache_key] = node_map
            return node_map

        # 2. Try loading from subject/object caches
        for prefix in ["subject", "object"]:
            sub_npy = cache_dir / f"{prefix}_{model_slug}_embeddings.npy"
            sub_meta = cache_dir / f"{prefix}_{model_slug}_meta.json"
            if sub_npy.exists() and sub_meta.exists():
                try:
                    with open(sub_meta, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    vocab = meta.get("vocab", [])
                    arr = np.load(sub_npy)
                    if len(vocab) == len(arr):
                        for idx, term in enumerate(vocab):
                            k = str(term).lower().strip()
                            if k not in node_map:
                                node_map[k] = arr[idx]
                except Exception:
                    pass

        missing_still = [n for n in norm_nodes if n.lower().strip() not in node_map]
        if missing_still:
            try:
                import torch
                from sentence_transformers import SentenceTransformer

                unique_missing = sorted(list(set(missing_still)))
                models_dir = Path("models/embeddings")
                models_dir.mkdir(parents=True, exist_ok=True)
                try:
                    embedder = SentenceTransformer(
                        model_name, cache_folder=str(models_dir), local_files_only=True
                    )
                except Exception:
                    embedder = SentenceTransformer(
                        model_name, cache_folder=str(models_dir), local_files_only=False
                    )
                emb_batch_size = resolve_dynamic_batch_size(
                    device=get_optimal_device(DEVICE),
                    requested_batch_size=EMBEDDING_BATCH_SIZE,
                    items=[{"prompt": t} for t in unique_missing],
                )
                with torch.inference_mode():
                    new_vecs = embedder.encode(
                        unique_missing,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                        batch_size=emb_batch_size,
                    )
                for idx, term in enumerate(unique_missing):
                    node_map[term.lower().strip()] = new_vecs[idx]

                # Offload and delete SentenceTransformer immediately so GPU VRAM is clean for PrismRunner
                try:
                    embedder.to("cpu")
                except Exception:
                    pass
                del embedder
                if MANAGE_VRAM:
                    clear_device_memory()

                all_terms = list(node_map.keys())
                all_arr = np.array([node_map[t] for t in all_terms])
                np.save(concept_npy, all_arr)
                with open(concept_meta, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "model_name": model_name,
                            "dimensions": int(all_arr.shape[1]),
                            "vocab": all_terms,
                        },
                        f,
                        indent=2,
                    )
                print(f"      -> Unified concept cache saved: {concept_npy.name} ({len(all_terms)} concepts)")
            except Exception as e:
                print(f"      -> Warning: SentenceTransformer encoding error: {e}")

        TargetResolver._memory_cache[cache_key] = node_map
        return node_map

    @staticmethod
    def center_concept_embeddings(
        concept_vecs: Dict[str, np.ndarray],
        mean_vec: Optional[np.ndarray] = None,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Centers concept embeddings by subtracting the corpus mean vector mu and L2-normalizing:
            v_centered = (v - mu) / ||v - mu||_2
        """
        if not concept_vecs:
            return {}, np.zeros(0)

        if mean_vec is None:
            all_arr = np.array(list(concept_vecs.values()))
            mean_vec = np.mean(all_arr, axis=0)

        centered_dict = {}
        for k, v in concept_vecs.items():
            diff = v - mean_vec
            norm = np.linalg.norm(diff)
            centered_dict[k] = diff / norm if norm > 1e-9 else diff

        return centered_dict, mean_vec

    @staticmethod
    def compute_architectural_tiers(
        target_nodes: set,
        triplets: List[Dict[str, Any]],
        config: Dict[str, Any],
        concept_embeddings: Dict[str, np.ndarray],
        community_id: Any = None,
        community_label: str = "Community",
    ) -> Dict[str, Any]:
        """
        Computes hybrid geometric-topological architectural tiers for a community target:
          - Tier 1: Thematic Nucleus
          - Tier 2: Relational Mechanics
          - Tier 3: Contextual Modifiers & Noise
        """
        tier_cfg = (
            config.get("topology", {}).get("architectural_tiers", {})
            if "topology" in config
            else config.get("architectural_tiers", {})
        )
        weights = tier_cfg.get("weights", {})
        w_proto = float(weights.get("prototypicality", 0.40))
        w_cent = float(weights.get("centrality", 0.40))
        w_fid = float(weights.get("fidelity", 0.20))

        p_t1 = float(tier_cfg.get("tier_1_percentile", 0.80))
        min_t1 = int(tier_cfg.get("tier_1_min_entities", 2))
        max_t1 = int(tier_cfg.get("tier_1_max_entities", 8))
        p_t2 = float(tier_cfg.get("tier_2_percentile", 0.30))
        min_fid_t1 = float(tier_cfg.get("min_fidelity_tier_1", 0.10))
        boundary_discount = float(tier_cfg.get("boundary_discount", 0.5))
        fallback_proto = float(tier_cfg.get("fallback_prototypicality", 0.30))
        fallback_fid = float(tier_cfg.get("fallback_fidelity", 0.28))

        target_nodes_norm = {str(n).lower().strip(): str(n) for n in target_nodes}
        node_internal_deg = Counter()
        node_boundary_deg = Counter()
        node_fidelities = defaultdict(list)
        node_quotes = defaultdict(list)

        for t in triplets:
            s_raw = str(t.get("subject", "")).strip()
            o_raw = str(t.get("object", "")).strip()
            s_k = s_raw.lower()
            o_k = o_raw.lower()
            fid = float(t.get("contextual_fidelity", 0.30))
            quote = str(t.get("source_quote", "")).strip()

            is_s_in = s_k in target_nodes_norm
            is_o_in = o_k in target_nodes_norm

            if is_s_in and is_o_in:
                node_internal_deg[s_k] += 1
                node_internal_deg[o_k] += 1
            elif is_s_in:
                node_boundary_deg[s_k] += 1
            elif is_o_in:
                node_boundary_deg[o_k] += 1

            if is_s_in:
                node_fidelities[s_k].append(fid)
                if quote:
                    node_quotes[s_k].append((fid, quote))
            if is_o_in:
                node_fidelities[o_k].append(fid)
                if quote:
                    node_quotes[o_k].append((fid, quote))

        vec_dim = 384
        for vec in concept_embeddings.values():
            vec_dim = len(vec)
            break

        weighted_vecs = []
        weight_sum = 0.0
        for n_k in target_nodes_norm.keys():
            vec = concept_embeddings.get(n_k)
            if vec is not None:
                deg = node_internal_deg[n_k]
                w = 1.0 + math.log(1.0 + deg)
                weighted_vecs.append(vec * w)
                weight_sum += w

        if weighted_vecs and weight_sum > 0:
            c_sum = np.sum(weighted_vecs, axis=0)
            c_norm = np.linalg.norm(c_sum)
            centroid = c_sum / c_norm if c_norm > 0 else c_sum
        else:
            centroid = np.zeros(vec_dim)

        scored_entities = []
        max_effective_deg = 1.0
        effective_degrees = {}
        for n_k in target_nodes_norm.keys():
            deg_int = node_internal_deg[n_k]
            deg_bound = node_boundary_deg[n_k]
            eff_deg = deg_int + boundary_discount * deg_bound
            effective_degrees[n_k] = eff_deg
            if eff_deg > max_effective_deg:
                max_effective_deg = eff_deg

        for n_k, orig_name in target_nodes_norm.items():
            vec = concept_embeddings.get(n_k)
            if vec is not None and np.linalg.norm(centroid) > 0:
                proto = float(np.dot(vec, centroid))
            else:
                proto = fallback_proto

            eff_deg = effective_degrees[n_k]
            cent = eff_deg / max_effective_deg

            fids = node_fidelities.get(n_k, [])
            fid = float(np.mean(fids)) if fids else fallback_fid

            s_proto = max(0.001, proto) ** w_proto
            s_cent = cent ** w_cent
            s_fid = max(0.001, fid) ** w_fid
            composite_s = float(s_proto * s_cent * s_fid)

            quotes = node_quotes.get(n_k, [])
            if quotes:
                quotes.sort(key=lambda x: x[0], reverse=True)
                top_quote = quotes[0][1]
            else:
                top_quote = ""

            scored_entities.append({
                "entity": orig_name,
                "composite_score": round(composite_s, 4),
                "prototypicality": round(proto, 4),
                "centrality": round(cent, 4),
                "fidelity": round(fid, 4),
                "internal_degree": int(node_internal_deg[n_k]),
                "boundary_degree": int(node_boundary_deg[n_k]),
                "exemplar_quote": top_quote[:240],
            })

        scored_entities.sort(key=lambda x: x["composite_score"], reverse=True)

        s_values = np.array([x["composite_score"] for x in scored_entities])
        p80_val = float(np.percentile(s_values, p_t1 * 100)) if len(s_values) > 0 else 0.5
        p30_val = float(np.percentile(s_values, p_t2 * 100)) if len(s_values) > 0 else 0.2

        t1_candidates = []
        demoted_from_t1 = []
        remaining = []

        for item in scored_entities:
            if item["composite_score"] >= p80_val:
                if item["fidelity"] >= min_fid_t1:
                    t1_candidates.append(item)
                else:
                    item_copy = dict(item)
                    item_copy["demoted_due_to_low_fidelity"] = True
                    demoted_from_t1.append(item_copy)
            else:
                remaining.append(item)

        effective_max_t1 = max(min_t1, max_t1)
        if len(t1_candidates) < min_t1 and scored_entities:
            tier_1 = scored_entities[: min(min_t1, len(scored_entities))]
            t1_names = {x["entity"] for x in tier_1}
            pool_for_t2 = [x for x in scored_entities if x["entity"] not in t1_names]
        elif len(t1_candidates) > effective_max_t1:
            tier_1 = t1_candidates[:effective_max_t1]
            t1_names = {x["entity"] for x in tier_1}
            pool_for_t2 = [x for x in (t1_candidates[effective_max_t1:] + remaining) if x["entity"] not in t1_names]
        else:
            tier_1 = t1_candidates
            t1_names = {x["entity"] for x in tier_1}
            pool_for_t2 = [x for x in remaining if x["entity"] not in t1_names]

        tier_2 = []
        tier_3_pool = list(demoted_from_t1)

        for item in pool_for_t2:
            if item["composite_score"] >= p30_val and (item["internal_degree"] > 0 or item["boundary_degree"] > 1):
                tier_2.append(item)
            else:
                tier_3_pool.append(item)

        tier_3 = tier_3_pool
        tier_3.sort(key=lambda x: x["composite_score"], reverse=True)

        return {
            "community_id": community_id,
            "community_label": community_label,
            "total_nodes": len(target_nodes),
            "tier_counts": {
                "tier_1_nucleus": len(tier_1),
                "tier_2_mechanics": len(tier_2),
                "tier_3_modifiers": len(tier_3),
            },
            "tier_1_nucleus": tier_1,
            "tier_2_mechanics": tier_2,
            "tier_3_modifiers": tier_3,
        }

    @staticmethod
    def export_resolved_targets_json(
        topology: Dict[str, Any],
        path_type: str = "community",
        refined_triplets: Optional[List[Dict[str, Any]]] = None,
        config: Optional[Dict[str, Any]] = None,
        output_dir: Path = Path("outputs/03_topology/03_paths/community"),
    ) -> Path:
        """
        Executes target qualification and triplet PageRank capping, and exports
        resolved_<path_type>_targets.json to output_dir with explicit rank scores.
        Also computes and exports geometric-topological architectural tiers.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        synth_cfg = config.get("synthesis", {}) if config else {}
        min_cluster_size = synth_cfg.get("min_cluster_size", 15)
        max_triplets_cap = synth_cfg.get("max_triplets_per_target", 1800)

        targets, all_enum_nodes = TargetResolver.qualify_and_resolve_targets(topology, path_type, config)

        tier_cfg = (
            config.get("topology", {}).get("architectural_tiers", {})
            if config and "topology" in config
            else (config.get("architectural_tiers", {}) if config else {})
        )
        tier_enabled = tier_cfg.get("enabled", True)

        concept_embeddings = {}
        if tier_enabled:
            all_target_nodes = set()
            for target in targets:
                all_target_nodes.update(target.get("nodes", []))
            ref_cfg = config.get("refinement", {}) if config and "refinement" in config else (config or {})
            model_name = ref_cfg.get("embedding_model") or (config.get("embedding_model") if config else None)
            cache_dir = Path(ref_cfg.get("vector_cache_dir", "outputs/02_refinement/cache"))
            concept_embeddings = TargetResolver._get_or_load_concept_embeddings(
                list(all_target_nodes),
                cache_dir=cache_dir,
                model_name=model_name,
            )
            topo_cfg = config.get("topology", {}) if config and "topology" in config else (config or {})
            if topo_cfg.get("center_agglomeration_embeddings", True) and concept_embeddings:
                concept_embeddings, _ = TargetResolver.center_concept_embeddings(concept_embeddings)

        target_payloads = []
        all_tiers_records = []
        csv_rows = []

        for i, target in enumerate(targets, start=1):
            target_type = target["type"]
            c_id = target["id"]
            target_nodes = set(target["nodes"])
            comm_label = f"Community #{c_id}"

            capped_triplets = TargetResolver.filter_and_cap_triplets(
                target_nodes,
                refined_triplets or [],
                topology,
                max_triplets_cap,
                target_id=c_id,
                path_type=path_type,
                config=config,
            )

            if tier_enabled:
                tiers_data = TargetResolver.compute_architectural_tiers(
                    target_nodes=target_nodes,
                    triplets=capped_triplets,
                    config=config or {},
                    concept_embeddings=concept_embeddings,
                    community_id=c_id,
                    community_label=comm_label,
                )
                all_tiers_records.append(tiers_data)

                for tier_key, tier_label in [
                    ("tier_1_nucleus", "Tier 1: Nucleus"),
                    ("tier_2_mechanics", "Tier 2: Mechanics"),
                    ("tier_3_modifiers", "Tier 3: Modifiers"),
                ]:
                    for rank, item in enumerate(tiers_data.get(tier_key, []), start=1):
                        csv_rows.append({
                            "Community ID": f"Community #{c_id}",
                            "Community Label": comm_label,
                            "Tier": tier_label,
                            "Rank in Tier": rank,
                            "Entity Name": item["entity"],
                            "Composite Score": item["composite_score"],
                            "Prototypicality": item["prototypicality"],
                            "Centrality": item["centrality"],
                            "Fidelity": item["fidelity"],
                            "Internal Degree": item["internal_degree"],
                            "Boundary Degree": item["boundary_degree"],
                            "Exemplar Quote": item["exemplar_quote"],
                        })
            else:
                tiers_data = {
                    "community_id": c_id,
                    "community_label": comm_label,
                    "target_node_count": len(target_nodes),
                    "tier_1_nucleus": [],
                    "tier_2_mechanics": [],
                    "tier_3_modifiers": [],
                }

            filename_suffix = f"community_{c_id}"
            output_module_name = f"{i:02d}_{filename_suffix}.py"
            rank_score = float(len(target_nodes))

            payload_item = {
                "target_index": i,
                "target_type": target_type,
                "target_id": c_id,
                "merged_from": target.get("merged_from", [c_id]),
                "output_module_name": output_module_name,
                "rank_score": rank_score,
                "nodes_count": len(target_nodes),
                "nodes": sorted(list(target_nodes)),
                "triplets_count": len(capped_triplets),
                "triplets": capped_triplets,
                "architectural_tiers": tiers_data,
            }
            target_payloads.append(payload_item)

        audit_payload = {
            "path_type": path_type,
            "min_cluster_size": min_cluster_size,
            "max_triplets_per_target": max_triplets_cap,
            "global_hubs": list(topology.get("global_hubs", [])),
            "enum_nodes_count": len(all_enum_nodes),
            "enum_nodes": all_enum_nodes,
            "targets_count": len(target_payloads),
            "targets": target_payloads,
        }

        target_file = output_dir / "resolved_community_targets.json"
        with open(target_file, "w", encoding="utf-8") as f:
            json.dump(audit_payload, f, indent=2)

        print(f"   [OK] Resolved Targets Saved: {target_file}")

        if path_type == "community" and tier_enabled:
            tier_json_file = output_dir / "community_architectural_tiers.json"
            with open(tier_json_file, "w", encoding="utf-8") as f:
                json.dump(all_tiers_records, f, indent=2)
            print(f"   [OK] Community Architectural Tiers JSON Saved: {tier_json_file}")

            if tier_cfg.get("export_tier_csv", True) and csv_rows:
                tier_csv_file = output_dir / "community_architectural_tiers.csv"
                fieldnames = [
                    "Community ID", "Community Label", "Tier", "Rank in Tier", "Entity Name",
                    "Composite Score", "Prototypicality", "Centrality", "Fidelity",
                    "Internal Degree", "Boundary Degree", "Exemplar Quote",
                ]
                with open(tier_csv_file, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(csv_rows)
                print(f"   [OK] Community Architectural Tiers CSV Saved: {tier_csv_file} ({len(csv_rows)} rows)")

            if tier_cfg.get("export_diagnostics", True):
                diag_file = output_dir / "tier_partitioning_diagnostics.json"
                t1_scores = [r["Composite Score"] for r in csv_rows if r["Tier"] == "Tier 1: Nucleus"]
                t2_scores = [r["Composite Score"] for r in csv_rows if r["Tier"] == "Tier 2: Mechanics"]
                t3_scores = [r["Composite Score"] for r in csv_rows if r["Tier"] == "Tier 3: Modifiers"]
                diag = {
                    "total_communities": len(all_tiers_records),
                    "total_partitioned_entities": len(csv_rows),
                    "tier_counts": {
                        "tier_1_nucleus": len(t1_scores),
                        "tier_2_mechanics": len(t2_scores),
                        "tier_3_modifiers": len(t3_scores),
                    },
                    "score_distributions": {
                        "tier_1": {
                            "mean": round(float(np.mean(t1_scores)), 4) if t1_scores else 0.0,
                            "median": round(float(np.median(t1_scores)), 4) if t1_scores else 0.0,
                            "min": round(float(np.min(t1_scores)), 4) if t1_scores else 0.0,
                            "max": round(float(np.max(t1_scores)), 4) if t1_scores else 0.0,
                        },
                        "tier_2": {
                            "mean": round(float(np.mean(t2_scores)), 4) if t2_scores else 0.0,
                            "median": round(float(np.median(t2_scores)), 4) if t2_scores else 0.0,
                            "min": round(float(np.min(t2_scores)), 4) if t2_scores else 0.0,
                            "max": round(float(np.max(t2_scores)), 4) if t2_scores else 0.0,
                        },
                        "tier_3": {
                            "mean": round(float(np.mean(t3_scores)), 4) if t3_scores else 0.0,
                            "median": round(float(np.median(t3_scores)), 4) if t3_scores else 0.0,
                            "min": round(float(np.min(t3_scores)), 4) if t3_scores else 0.0,
                            "max": round(float(np.max(t3_scores)), 4) if t3_scores else 0.0,
                        },
                    },
                    "separation_margin": round(
                        float(np.mean(t1_scores) - np.mean(t3_scores)), 4
                    ) if t1_scores and t3_scores else 0.0,
                }
                with open(diag_file, "w", encoding="utf-8") as f:
                    json.dump(diag, f, indent=2)
                print(f"   [OK] Tier Partitioning Diagnostics Saved: {diag_file}")

        return target_file


# ============================================================================
# AST CLEANING, DEDUPLICATION, AND ENUM CLUSTERING HELPERS
# (Exact match with src/synthesis/synthesizer.py)
# ============================================================================

def _normalize_candidate_enum_term(term: str) -> Optional[str]:
    """
    Strips conversational boilerplate from candidate Enum terms and rejects
    verbose multi-clause sentence fragments (e.g., 'within the average range for his age group...').
    """
    cleaned = str(term).strip()
    if not cleaned:
        return None
    lower = cleaned.lower()

    # Strip leading conversational prefixes
    for prefix in ("within the ", "in the ", "within ", "at the "):
        if lower.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
            lower = cleaned.lower()

    # Strip trailing comparison / age-group clauses
    cleaned = re.sub(
        r"\s+(?:as compared to|compared to|relative to|for (?:his|her|their|its) age group|for the .*?subtest|for the .*?task|for the .*?measure|for the .*?assessment|for the .*?index).*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    cleaned = re.sub(r"\s+range$", "", cleaned, flags=re.IGNORECASE).strip()
    lower = cleaned.lower()

    # Reject if still a multi-clause fragment or overly long
    words = [w for w in re.split(r"[\s_-]+", lower) if w]
    if not words or len(words) > MAX_ENUM_MEMBER_WORDS or len(cleaned) > MAX_ENUM_MEMBER_CHARS:
        return None
    if any(
        marker in f" {lower} "
        for marker in (" for his ", " for her ", " age group ", " same aged ", " subtest ", " compared to ")
    ):
        return None
    return cleaned


def cluster_enum_candidates(
    enum_nodes: List[str],
    concept_embeddings: Dict[str, np.ndarray],
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Groups orphan and low-density nodes into high-cohesion semantic clusters
    using pre-computed concept vectors. Discards isolated terms, conversational
    sentence fragments, and weak clusters.
    """
    enum_cfg = config.get("synthesis", {}).get("enum_clustering", {})
    min_cluster_size = enum_cfg.get("min_cluster_size", ENUM_MIN_CLUSTER_SIZE)
    dist_threshold = enum_cfg.get("distance_threshold", ENUM_DISTANCE_THRESHOLD)
    min_cohesion = enum_cfg.get("min_cohesion_score", ENUM_MIN_COHESION_SCORE)
    max_terms_per_cluster = enum_cfg.get("max_terms_per_cluster", ENUM_MAX_TERMS_PER_CLUSTER)

    valid_terms = []
    vector_list = []
    seen_norm = set()
    for term in enum_nodes:
        raw_key = str(term).lower().strip()
        if raw_key not in concept_embeddings:
            continue
        cleaned_term = _normalize_candidate_enum_term(term)
        if not cleaned_term:
            continue
        norm_cleaned = cleaned_term.lower()
        if norm_cleaned in seen_norm:
            continue
        seen_norm.add(norm_cleaned)
        valid_terms.append(cleaned_term)
        vector_list.append(concept_embeddings[raw_key])

    if len(valid_terms) < min_cluster_size:
        return []

    X = np.vstack(vector_list)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=dist_threshold,
        metric="cosine",
        linkage="average",
    )
    labels = clustering.fit_predict(X)

    clusters_by_id = defaultdict(lambda: {"terms": [], "vectors": []})
    for term, vec, lbl in zip(valid_terms, X, labels):
        clusters_by_id[lbl]["terms"].append(term)
        clusters_by_id[lbl]["vectors"].append(vec)

    strong_clusters = []
    for c_id, data in clusters_by_id.items():
        terms = data["terms"]
        if len(terms) < min_cluster_size:
            continue

        vecs = np.array(data["vectors"])
        centroid = np.mean(vecs, axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm

        sims = np.dot(vecs, centroid)
        avg_cohesion = float(np.mean(sims))

        if avg_cohesion >= min_cohesion:
            ranked_indices = np.argsort(sims)[::-1]
            top_indices = ranked_indices[:max_terms_per_cluster]
            prototypical_idx = int(ranked_indices[0])
            capped_terms = [terms[int(i)] for i in top_indices]
            strong_clusters.append({
                "cluster_id": int(c_id),
                "suggested_category": terms[prototypical_idx],
                "cohesion_score": round(avg_cohesion, 3),
                "member_count": len(capped_terms),
                "terms": capped_terms,
            })

    strong_clusters.sort(key=lambda c: (c["member_count"], c["cohesion_score"]), reverse=True)
    return strong_clusters


def _is_verbose_or_runaway_enum_member(member_name: str, member_val: Optional[str] = None) -> bool:
    """
    Returns True if an Enum member identifier or string value is a verbose sentence fragment
    or combinatorial permutation (e.g. WITHIN_THE_AVERAGE_RANGE_FOR_HIS_AGE_GROUP_FOR_THE_...).
    """
    ident = str(member_name).strip().upper()
    if not ident:
        return True
    words = [w for w in ident.split("_") if w]
    if len(words) > MAX_ENUM_MEMBER_WORDS or len(ident) > MAX_ENUM_MEMBER_CHARS:
        return True
    banned_substrings = (
        "WITHIN_THE_",
        "_FOR_HIS_",
        "_FOR_HER_",
        "_FOR_THEIR_",
        "_FOR_THE_",
        "_AGE_GROUP",
        "_SAME_AGED",
        "_AS_COMPARED_TO",
        "_COMPARED_TO_",
        "_SUBTEST",
        "_IN_THE_RANGE",
    )
    if any(b in ident for b in banned_substrings):
        return True
    if member_val is not None:
        val_str = str(member_val).strip()
        val_words = [w for w in re.split(r"[\s_-]+", val_str) if w]
        if len(val_words) > MAX_ENUM_MEMBER_WORDS + 1 or len(val_str) > MAX_ENUM_MEMBER_CHARS + 5:
            return True
    return False


def _sanitize_ast_enums(code: str) -> str:
    """
    AST post-processor that strips verbose sentence-fragment Enum members, deduplicates
    Enum members within each class, and caps Enum classes to MAX_ENUM_MEMBERS_PER_CLASS.
    """
    try:
        tree = ast.parse(code)
    except Exception:
        return code

    def is_enum_class(node: ast.ClassDef) -> bool:
        for b in node.bases:
            if isinstance(b, ast.Name) and "Enum" in b.id:
                return True
            if isinstance(b, ast.Attribute) and "Enum" in b.attr:
                return True
        return False

    new_body: List[ast.AST] = []
    modified = False

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and is_enum_class(node):
            kept_stmts: List[ast.stmt] = []
            seen_names: set = set()
            seen_vals: set = set()
            member_count = 0

            for stmt in node.body:
                m_name = None
                m_val = None
                if isinstance(stmt, ast.Assign) and stmt.targets and isinstance(stmt.targets[0], ast.Name):
                    m_name = stmt.targets[0].id
                    if isinstance(stmt.value, ast.Constant):
                        m_val = stmt.value.value
                elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    m_name = stmt.target.id
                    if stmt.value and isinstance(stmt.value, ast.Constant):
                        m_val = stmt.value.value

                if m_name is not None:
                    if (
                        _is_verbose_or_runaway_enum_member(m_name, m_val)
                        or m_name in seen_names
                        or (m_val is not None and str(m_val).lower().strip() in seen_vals)
                        or member_count >= MAX_ENUM_MEMBERS_PER_CLASS
                    ):
                        modified = True
                        continue
                    seen_names.add(m_name)
                    if m_val is not None:
                        seen_vals.add(str(m_val).lower().strip())
                    member_count += 1
                    kept_stmts.append(stmt)
                else:
                    # Docstring or pass statement
                    kept_stmts.append(stmt)

            if member_count > 0:
                node.body = kept_stmts
                new_body.append(node)
            else:
                modified = True
        else:
            new_body.append(node)

    if not modified:
        return code

    tree.body = new_body
    try:
        return ast.unparse(tree)
    except Exception:
        return code


def clean_python_code(code: str) -> str:
    """Removes markdown formatting, invalid characters, and escapes that LLMs often inject."""
    if not code:
        return ""
    code = code.strip()
    if code.startswith("```python"):
        code = code[9:]
    elif code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]

    # Fix over-escaped quotes caused by LLM JSON serialization
    code = code.replace('\\"', '"')

    # Replace literal \n escape sequences that some LLMs emit outside raw strings
    code = code.replace("\\n", "\n")

    # Specific fix to prevent un-escaping the one known problematic Enum string
    code = code.replace('"the "firstthen" principle"', '\'the "firstthen" principle\'')

    def fix_line(line: str) -> str:
        # 1. Assignment '=' fix
        m = re.match(r"^(\s+)([a-zA-Z0-9_, ]+?)(\s*=\s*.*)$", line)
        if m:
            indent, var, rest = m.groups()
            stripped_var = var.strip()
            if ("," in stripped_var or " " in stripped_var) and not stripped_var.startswith(
                ("def ", "class ", "return ", "if ", "elif ")
            ):
                clean_var = re.sub(r"[, ]+", "_", stripped_var)
                line = f"{indent}{clean_var}{rest}"

        # 2. Type annotation ':' fix for hyphens (e.g. 'help-seeking_barriers: ...' -> 'help_seeking_barriers: ...')
        m_ann = re.match(r"^(\s+)([a-zA-Z_][a-zA-Z0-9_-]*)(\s*:\s*.*)$", line)
        if m_ann:
            indent, var, rest = m_ann.groups()
            if "-" in var:
                clean_var = var.replace("-", "_")
                line = f"{indent}{clean_var}{rest}"
        return line

    code = "\n".join(fix_line(line_str) for line_str in code.splitlines())

    # Strip rogue CJK / non-ASCII character leakage from bilingual models in comments or docstrings
    code = re.sub(r"[\u4e00-\u9fff]+", "", code)

    # Strip any accidental meta-modeling classes that mirror the prompt/JSON wrapper
    code = re.sub(
        r"(?m)^class\s+(ThemeObservation|ThematicObservation|GeneratedModule)\s*\([^)]*\):.*?(?=(?:\nclass\s+|\n[A-Za-z0-9_]+\s*=|\Z))",
        "",
        code,
        flags=re.DOTALL,
    )
    code = _sanitize_ast_enums(code.strip())
    return code.strip()


def deduplicate_consolidated_classes(code: str) -> str:
    """
    Programmatic AST post-processor for consolidated ontologies (master_ontology.py and comprehensive_ontology.py).
    Merges duplicate Enums and BaseModels, and disambiguates Enum vs BaseModel naming collisions.
    """
    clean_code = clean_python_code(code)
    try:
        tree = ast.parse(clean_code)
    except Exception:
        return clean_code

    def is_enum_class(node: ast.ClassDef) -> bool:
        for b in node.bases:
            if isinstance(b, ast.Name) and "Enum" in b.id:
                return True
            if isinstance(b, ast.Attribute) and "Enum" in b.attr:
                return True
        return False

    def get_assigned_name(stmt: ast.AST) -> Optional[str]:
        if isinstance(stmt, ast.Assign) and stmt.targets and isinstance(stmt.targets[0], ast.Name):
            return stmt.targets[0].id
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            return stmt.target.id
        return None

    enums_by_name: Dict[str, ast.ClassDef] = {}
    models_by_name: Dict[str, ast.ClassDef] = {}
    other_nodes: List[ast.AST] = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            name = node.name
            if is_enum_class(node):
                if name not in enums_by_name:
                    enums_by_name[name] = node
                else:
                    existing_members = {
                        get_assigned_name(stmt)
                        for stmt in enums_by_name[name].body
                        if get_assigned_name(stmt) is not None
                    }
                    for stmt in node.body:
                        m_name = get_assigned_name(stmt)
                        if m_name and m_name not in existing_members:
                            enums_by_name[name].body.append(stmt)
                            existing_members.add(m_name)
            else:
                if name not in models_by_name:
                    models_by_name[name] = node
                else:
                    existing_fields = {
                        get_assigned_name(stmt)
                        for stmt in models_by_name[name].body
                        if get_assigned_name(stmt) is not None
                    }
                    for stmt in node.body:
                        f_name = get_assigned_name(stmt)
                        if f_name and f_name not in existing_fields:
                            models_by_name[name].body.append(stmt)
                            existing_fields.add(f_name)
        else:
            other_nodes.append(node)

    # Disambiguate Enum vs BaseModel name collisions
    for m_name in list(models_by_name.keys()):
        if m_name in enums_by_name:
            new_name = f"{m_name}Detail"
            model_node = models_by_name.pop(m_name)
            model_node.name = new_name
            models_by_name[new_name] = model_node

    tree.body = other_nodes + list(enums_by_name.values()) + list(models_by_name.values())
    try:
        return ast.unparse(tree)
    except Exception:
        return clean_code


def sanitize_module_name(name: str, fallback: str = "domain_schema") -> str:
    """Normalizes a module_name string into a valid snake_case Python module identifier."""
    if not name:
        return fallback
    cleaned = re.sub(r"\.py$", "", str(name).strip().lower())
    cleaned = re.sub(r"[^a-z0-9_]+", "_", cleaned).strip("_")
    return cleaned or fallback


def validate_python_syntax(code: str) -> Tuple[bool, Optional[str]]:
    """Checks whether cleaned Python code parses into a non-empty AST with at least one class."""
    if not code or not code.strip():
        return False, "Empty source_code returned."
    try:
        tree = ast.parse(code)
        has_class = any(isinstance(node, ast.ClassDef) for node in tree.body)
        if not has_class:
            return False, "Generated source_code contains no Python class definitions (BaseModel or Enum)."
        return True, None
    except SyntaxError as err:
        return False, f"SyntaxError: {err}"
    except Exception as err:
        return False, str(err)


# ============================================================================
# STAGE 4 SYNTHESIS PIPELINE (Powered by PrismRunner)
# ============================================================================

class SynthesisPipeline:
    """
    Stage 4 Synthesis Pipeline using PrismRunner for batch local LLM execution.
    Replicates SemanticPrism's 4-phase ontological synthesis workflow:
      - Phase 1: Semantic Enum Clustering & Synthesis (`enums.py`)
      - Phase 2: Community Schema Synthesis (`normalized/<id>_<module>.py` & `raw/<id>_<module>.py`)
      - Phase 3: Global Consolidation (`normalized/master_ontology.py` & `raw/master_ontology.py`)
      - Phase 4: Comprehensive Ontology Synthesis (`comprehensive_ontology.py`)
    """

    def __init__(self, config: Dict[str, Any], prism_pipeline: Optional[PrismPipeline] = None):
        self.config = config
        self.model_name = (
            config.get("synthesis", {}).get("llm", {}).get("model_name")
            or config.get("llm", {}).get("model_name")
            or MODEL_NAME
        )

        # Guard against accidentally passing a SentenceTransformer embedding model as MODEL_NAME
        _lower_llm = self.model_name.lower()
        if any(marker in _lower_llm for marker in ("bge-", "minilm", "e5-", "gte-", "nomic-embed", "sentence-transformers")):
            raise ValueError(
                f"MODEL_NAME is set to '{self.model_name}', which is an encoder-only embedding model, "
                f"not a generative causal LLM.\n"
                f"  -> Keep '{self.model_name}' in EMBEDDING_MODEL.\n"
                f"  -> Set MODEL_NAME to a causal/coding LLM (e.g. 'Qwen/Qwen3-Coder-Next', 'Qwen/Qwen3.6-27B', "
                f"or 'Qwen/Qwen2.5-Coder-32B-Instruct')."
            )

        if prism_pipeline is not None:
            self.prism = prism_pipeline
        else:
            self.prism = PrismPipeline(
                model_name=self.model_name,
                device=DEVICE,
                batch_size=BATCH_SIZE,
                config={
                    "model": {
                        "name": self.model_name,
                        "device": DEVICE,
                        "torch_dtype": TORCH_DTYPE,
                        "load_in_4bit": LOAD_IN_4BIT,
                        "load_in_8bit": LOAD_IN_8BIT,
                        "attn_implementation": ATTN_IMPLEMENTATION,
                    },
                    "inference": {
                        "batch_size": BATCH_SIZE,
                        "max_new_tokens": MAX_NEW_TOKENS,
                        "temperature": TEMPERATURE,
                        "repetition_penalty": REPETITION_PENALTY,
                        "constrain_json": CONSTRAIN_JSON,
                    },
                },
            )

    def _checkpoint(self, phase: str, message: str, file_path: Optional[Path] = None) -> None:
        """Prints a formatted real-time checkpoint notification when PRINT_CHECKPOINTS is enabled."""
        if not PRINT_CHECKPOINTS:
            return
        ts = datetime.now().strftime("%H:%M:%S")
        path_suffix = f" -> {file_path}" if file_path is not None else ""
        print(f"[{ts} | Checkpoint | {phase}] {message}{path_suffix}", flush=True)

    def _apply_context_window_cap(self) -> int:
        """Apply CONTEXT_WINDOW_CAP if specified, otherwise use the model's native context limit."""
        native_ctx = getattr(self.prism, "context_window", 16384) or 16384
        raw_cap = self.config.get("synthesis", {}).get("context_window_cap", CONTEXT_WINDOW_CAP)
        if isinstance(raw_cap, int) and raw_cap > 0:
            effective_ctx = min(native_ctx, raw_cap)
        else:
            effective_ctx = native_ctx
        self.prism.context_window = effective_ctx
        if self.prism.engine is not None:
            self.prism.engine.context_window = effective_ctx
        return effective_ctx

    def _ensure_engine_loaded(self, phase_label: str) -> None:
        """Ensures the PrismPipeline engine is loaded and GPU memory is cleared beforehand."""
        if DRY_RUN:
            return
        if self.prism.engine is None:
            if MANAGE_VRAM:
                clear_device_memory()
            self._checkpoint(phase_label, f"Loading LLM '{self.model_name}' onto device...")
            self.prism.init_engine(
                batch_size=BATCH_SIZE,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                repetition_penalty=REPETITION_PENALTY,
            )
            eff_ctx = self._apply_context_window_cap()
            self._checkpoint(
                phase_label,
                f"Model ready on device '{self.prism.device}' "
                f"(effective context window: {eff_ctx} tokens)",
            )

    def _unload_prism_pipeline(self, phase_label: str = "Cleanup") -> None:
        """Releases LLM weights and flushes GPU cache."""
        if getattr(self.prism, "engine", None) is not None or getattr(self.prism, "model", None) is not None:
            self._checkpoint(phase_label, f"Unloading LLM '{self.model_name}' and clearing VRAM...")
            self.prism.engine = None
            self.prism.model = None
            self.prism.tokenizer = None
        if MANAGE_VRAM:
            clear_device_memory()

    @staticmethod
    def _extract_generated_module(items: List[Any], fallback_name: str) -> Tuple[Optional[str], Optional[str]]:
        """Extracts (module_name, cleaned_source_code) from parsed PrismRunner items."""
        for item in items:
            if isinstance(item, GeneratedModule):
                mod_name = sanitize_module_name(item.module_name, fallback=fallback_name)
                code = clean_python_code(item.source_code)
                if code:
                    return mod_name, code
            elif isinstance(item, dict):
                try:
                    parsed = GeneratedModule.model_validate(item)
                    mod_name = sanitize_module_name(parsed.module_name, fallback=fallback_name)
                    code = clean_python_code(parsed.source_code)
                    if code:
                        return mod_name, code
                except Exception:
                    pass
        return None, None

    def _run_module_batch(
        self,
        items: List[Dict[str, Any]],
        system_prompt: str,
        phase_label: str,
        default_module_name: str = "domain_schema",
        allow_reformat_repair: bool = True,
        max_new_tokens: int = MAX_NEW_TOKENS,
        on_valid_module_ready: Optional[Callable[[Dict[str, Any], str, str], None]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Executes a batch of prompt payloads through PrismRunner using the `GeneratedModule` schema.
        Validates the resulting Python source code with `ast.parse` and automatically runs a
        batch repair pass with `SCHEMA_REFORMAT_SYSTEM_PROMPT` for any items that failed validation
        or produced syntax errors.

        Each item in `items` must contain:
          - `id`: unique string identifier
          - `payload`: user prompt string
          - `fallback_module_name` (optional): fallback snake_case module name
        Returns a list of dicts (aligned with `items`) with keys:
          - `id`, `success`, `module_name`, `source_code`, `error`, `raw_text`
        """
        if not items:
            return []

        self._ensure_engine_loaded(phase_label)

        rows = [
            {"uid": str(item["id"]), "text": str(item["payload"])}
            for item in items
        ]
        meta_by_uid = {str(item["id"]): (idx, item) for idx, item in enumerate(items)}

        module_template = {
            "version": "1.0",
            "task": "schema_synthesis",
            "system_prompt": system_prompt,
            "prompt": "{{ text }}\n\n{schema_block}",
        }

        parsed_outputs: List[Dict[str, Any]] = [
            {
                "id": item["id"],
                "success": False,
                "module_name": item.get("fallback_module_name", default_module_name),
                "source_code": "",
                "error": "Not yet processed",
                "raw_text": "",
            }
            for item in items
        ]
        notified_uids: set = set()
        lock = threading.Lock()

        def _on_batch_start(batch_idx: int, total_batches: int, batch_items: List[Dict[str, Any]]) -> None:
            ids_preview = [str(it.get("record_id") or "") for it in batch_items[:4]]
            self._checkpoint(
                phase_label,
                f"Running PrismRunner batch {batch_idx}/{total_batches} ({len(batch_items)} item(s): {ids_preview})",
            )

        def _on_chunk_processed(chunk_res: Dict[str, Any]) -> None:
            uid = str(chunk_res.get("record_id") or chunk_res.get("id") or "")
            if uid not in meta_by_uid:
                return
            idx, item = meta_by_uid[uid]
            fallback_name = item.get("fallback_module_name", default_module_name)
            p_items = chunk_res.get("parsed_items") or chunk_res.get("extracted_items") or []
            raw_out = str(chunk_res.get("raw_output") or "")
            mod_name, cleaned_code = self._extract_generated_module(p_items, fallback_name)

            with lock:
                if uid in notified_uids:
                    return
                if cleaned_code:
                    is_valid, syntax_err = validate_python_syntax(cleaned_code)
                    if is_valid:
                        notified_uids.add(uid)
                        parsed_outputs[idx] = {
                            "id": item["id"],
                            "success": True,
                            "module_name": mod_name or fallback_name,
                            "source_code": cleaned_code,
                            "error": None,
                            "raw_text": raw_out,
                        }
                        if on_valid_module_ready is not None:
                            on_valid_module_ready(item, mod_name or fallback_name, cleaned_code)
                        return
                    err_reason = syntax_err or "Invalid Python syntax in generated source_code."
                    malformed = cleaned_code or raw_out
                else:
                    err_reason = str(chunk_res.get("parse_error") or "Constrained JSON validation failed.")
                    malformed = raw_out

                parsed_outputs[idx] = {
                    "id": item["id"],
                    "success": False,
                    "module_name": fallback_name,
                    "source_code": "",
                    "error": err_reason,
                    "raw_text": malformed,
                }

        results = self.prism.run(
            task="extract",
            input_data=rows,
            text_column="text",
            id_column="uid",
            schema_cls=GeneratedModule,
            prompt_template=module_template,
            system_prompt=system_prompt,
            chunk=False,
            as_array=False,
            dedup=False,
            batch_size=BATCH_SIZE,
            max_new_tokens=max_new_tokens,
            temperature=TEMPERATURE,
            repetition_penalty=REPETITION_PENALTY,
            constrain_json=CONSTRAIN_JSON,
            dry_run=DRY_RUN,
            on_batch_start=_on_batch_start,
            on_chunk_processed=_on_chunk_processed,
        )

        if DRY_RUN:
            for idx, item in enumerate(items):
                fallback_name = item.get("fallback_module_name", default_module_name)
                stub_code = (
                    "from enum import Enum\n"
                    "from typing import Optional\n"
                    "from pydantic import BaseModel, Field\n\n\n"
                    f"class DryRunModel_{idx + 1}(BaseModel):\n"
                    '    summary: Optional[str] = Field(default=None, description="Dry-run placeholder schema.")\n'
                )
                parsed_outputs[idx] = {
                    "id": item["id"],
                    "success": True,
                    "module_name": fallback_name,
                    "source_code": stub_code,
                    "error": None,
                    "raw_text": "",
                }
                if on_valid_module_ready is not None:
                    on_valid_module_ready(item, fallback_name, stub_code)
            return parsed_outputs

        # Ensure any items not processed by streaming callback are finalized
        for idx, (item, res) in enumerate(zip(items, results)):
            uid = str(item["id"])
            if uid not in notified_uids and not parsed_outputs[idx]["success"]:
                _on_chunk_processed(
                    {
                        "record_id": uid,
                        "parsed_items": res.get("extracted_items") or res.get("parsed_items") or [],
                        "raw_output": (res.get("raw_outputs") or [""])[0] if res.get("raw_outputs") else "",
                        "parse_error": res.get("warning"),
                    }
                )

        # Collect any failed items for the Schema Reformat Repair Agent pass
        repair_candidates: List[Tuple[int, Dict[str, Any], str, str]] = [
            (idx, items[idx], out["raw_text"], str(out["error"]))
            for idx, out in enumerate(parsed_outputs)
            if not out["success"]
        ]

        if repair_candidates and allow_reformat_repair:
            self._checkpoint(
                f"{phase_label} Repair",
                f"Running Schema Reformat Agent via PrismRunner on {len(repair_candidates)} failed item(s)...",
            )
            repair_rows = []
            for idx, item, malformed, err_reason in repair_candidates:
                reformat_prompt = (
                    f"Original Schema Generation Task context was:\n{item['payload'][:4000]}\n\n"
                    f"The malformed Python output from the failed attempts was:\n{malformed[:4000]}\n\n"
                    f"The parsing validation error that occurred was:\n{err_reason}\n\n"
                    f"Please correct and repair this Python code so it strictly maps to the schemas.GeneratedModule structure. "
                    f"The output must consist strictly of the Pydantic GeneratedModule schema (with 'module_name' and 'source_code') "
                    f"and must contain no conversational text, notes, markdown code blocks, or other extraneous words. "
                    f"IMPORTANT: Do NOT generate meta-models representing the prompt envelope (e.g. NEVER output 'class ThemeObservation' or 'class GeneratedModule'). "
                    f"Synthesize authentic domain models reflecting the key terms and observations."
                )
                repair_rows.append({"uid": str(item["id"]), "text": reformat_prompt})

            reformat_template = {
                "version": "1.0",
                "task": "schema_reformat",
                "system_prompt": SCHEMA_REFORMAT_SYSTEM_PROMPT,
                "prompt": "{{ text }}\n\n{schema_block}",
            }

            repair_results = self.prism.run(
                task="extract",
                input_data=repair_rows,
                text_column="text",
                id_column="uid",
                schema_cls=GeneratedModule,
                prompt_template=reformat_template,
                system_prompt=SCHEMA_REFORMAT_SYSTEM_PROMPT,
                chunk=False,
                as_array=False,
                dedup=False,
                batch_size=BATCH_SIZE,
                max_new_tokens=max_new_tokens,
                temperature=TEMPERATURE,
                repetition_penalty=REPETITION_PENALTY,
                constrain_json=CONSTRAIN_JSON,
                dry_run=DRY_RUN,
            )

            for (idx, item, _, _), rep_res in zip(repair_candidates, repair_results):
                fallback_name = item.get("fallback_module_name", default_module_name)
                rep_items = rep_res.get("extracted_items") or rep_res.get("parsed_items") or []
                mod_name, cleaned_code = self._extract_generated_module(rep_items, fallback_name)
                if cleaned_code:
                    parsed_outputs[idx] = {
                        "id": item["id"],
                        "success": True,
                        "module_name": mod_name or fallback_name,
                        "source_code": cleaned_code,
                        "error": None,
                        "raw_text": (rep_res.get("raw_outputs") or [""])[0] if rep_res.get("raw_outputs") else "",
                    }
                    print(f"            -> Reformat successful for {item['id']}!")
                    if on_valid_module_ready is not None:
                        on_valid_module_ready(item, mod_name or fallback_name, cleaned_code)
                else:
                    parsed_outputs[idx]["error"] = str(rep_res.get("warning") or parsed_outputs[idx]["error"])

        return parsed_outputs

    def execute(
        self,
        comm_targets: Optional[Dict[str, Any]] = None,
        refined_triplets: Optional[List[Dict[str, Any]]] = None,
        original_triplets: Optional[List[Dict[str, Any]]] = None,
        taxonomic_map: Optional[Dict[str, str]] = None,
        master_themes: Optional[List[str]] = None,
    ) -> None:
        print("[Synthesis] Starting Stage 4 Pipeline...")
        try:
            if comm_targets:
                print("   ==> Executing Community Path Schema Synthesis...")
                self.execute_path_synthesis(
                    comm_targets,
                    "community",
                    refined_triplets or [],
                    original_triplets or [],
                    master_themes or [],
                )
            else:
                print("   [Warning] Community resolved targets file not found. Skipping synthesis.")
        finally:
            self._unload_prism_pipeline("Stage 4 Complete")

        print("[Synthesis] Stage 4 Pipeline complete.")

    def execute_path_synthesis(
        self,
        targets_data: Dict[str, Any],
        path_type: str,
        refined_triplets: List[dict],
        original_triplets: List[dict],
        master_themes: List[str],
    ) -> None:
        path_label = "Community Workflow" if path_type == "community" else f"Path: {path_type.title()}"
        print(f"   -> Starting Synthesis Pass for [{path_label}]...")

        synth_cfg = self.config.get("synthesis", {})
        schema_pass_mode = synth_cfg.get("schema_pass_mode", "both")
        min_cluster_size = synth_cfg.get("min_cluster_size", 15)
        run_norm = schema_pass_mode in ("both", "normalized")
        run_raw = schema_pass_mode in ("both", "raw")
        print(f"      -> Schema Pass Mode: '{schema_pass_mode}' (Normalized: {run_norm}, Raw: {run_raw})")
        print(f"      -> Entity Threshold (min_cluster_size): {min_cluster_size}")

        # Setup logging
        outputs_dir = self.config.get("directories", {}).get("outputs", OUTPUTS_DIR)
        log_dir = Path(outputs_dir) / "04_synthesis" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"synthesis_{path_type}_errors.log"

        def log_error(msg: str) -> None:
            print(msg)
            with open(log_file, "a", encoding="utf-8") as lf:
                lf.write(f"[{datetime.now().isoformat()}] {msg}\n")

        # Path-isolated Output directories
        output_base_dir = Path(SYNTHESIS_OUTPUT_DIR) / "schemas" / path_type
        norm_dir = output_base_dir / "normalized"
        raw_dir = output_base_dir / "raw"

        if not RESUME_EXISTING_SCHEMAS and output_base_dir.exists():
            shutil.rmtree(output_base_dir)

        output_base_dir.mkdir(parents=True, exist_ok=True)
        if run_norm:
            norm_dir.mkdir(parents=True, exist_ok=True)
        if run_raw:
            raw_dir.mkdir(parents=True, exist_ok=True)

        # Payload dumping directory
        save_llm_payloads = synth_cfg.get("save_llm_payloads", False)
        default_payload_dir = Path(outputs_dir) / "payload_text" / "stage_04_synthesis"
        payload_base_dir = Path(synth_cfg.get("payload_output_dir", default_payload_dir)) / path_type
        if save_llm_payloads:
            if not RESUME_EXISTING_SCHEMAS and payload_base_dir.exists():
                shutil.rmtree(payload_base_dir)
            payload_base_dir.mkdir(parents=True, exist_ok=True)

        def save_payload_file(filename: str, payload_str: str) -> None:
            if save_llm_payloads:
                try:
                    with open(payload_base_dir / filename, "w", encoding="utf-8") as pf:
                        pf.write(payload_str)
                except Exception as err:
                    log_error(f"[Synthesis {path_type}] Error saving payload {filename}: {err}")

        targets = targets_data.get("targets", [])
        all_enum_nodes = targets_data.get("enum_nodes", [])

        # =====================================================================
        # PHASE 1: Semantic Enum Clustering & Synthesis
        # =====================================================================
        print(f"   -> [{path_label}] Phase 1: Semantic Enum Clustering & Synthesis...")
        print(f"      -> Total Raw Enum Candidates: {len(all_enum_nodes)}")
        global_enums_code = ""

        enum_clustering_cfg = synth_cfg.get("enum_clustering", {})
        clustering_enabled = enum_clustering_cfg.get("enabled", True)

        if all_enum_nodes:
            if clustering_enabled:
                ref_cfg = self.config.get("refinement", {})
                emb_model_name = ref_cfg.get("embedding_model", EMBEDDING_MODEL)
                cache_dir = Path(ref_cfg.get("vector_cache_dir", VECTOR_CACHE_DIR))

                concept_embeddings = TargetResolver._get_or_load_concept_embeddings(
                    all_enum_nodes,
                    cache_dir=cache_dir,
                    model_name=emb_model_name,
                )
                topo_cfg = self.config.get("topology", {}) if "topology" in self.config else self.config
                if topo_cfg.get("center_agglomeration_embeddings", True) and concept_embeddings:
                    concept_embeddings, _ = TargetResolver.center_concept_embeddings(concept_embeddings)

                qualified_clusters = cluster_enum_candidates(all_enum_nodes, concept_embeddings, self.config)
                retained_terms = sum(c["member_count"] for c in qualified_clusters)
                discarded_terms = len(all_enum_nodes) - retained_terms

                print(f"      -> Qualified Strong Clusters: {len(qualified_clusters)} ({retained_terms} concepts)")
                print(f"      -> Filtered Outliers / Noise Discarded: {discarded_terms} concepts")

                clusters_audit_file = log_dir / f"enum_clusters_{path_type}.json"
                try:
                    with open(clusters_audit_file, "w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "total_raw_nodes": len(all_enum_nodes),
                                "retained_terms_count": retained_terms,
                                "discarded_noise_count": discarded_terms,
                                "clusters_count": len(qualified_clusters),
                                "clusters": qualified_clusters,
                            },
                            f,
                            indent=2,
                        )
                    self._checkpoint("Phase 1: Enum Clusters", "Saved enum clustering diagnostics", clusters_audit_file)
                except Exception as err:
                    log_error(f"[Synthesis {path_type}] Error saving enum clusters audit: {err}")

                if qualified_clusters:
                    max_clusters_per_batch = enum_clustering_cfg.get("max_clusters_per_batch", 25)
                    selected_clusters = qualified_clusters[:max_clusters_per_batch]
                    if len(qualified_clusters) > max_clusters_per_batch:
                        print(f"      -> Selected Top {max_clusters_per_batch} clusters for primary enum synthesis")

                    payload_data = {
                        "instructions": "Synthesize standardized Python Enum and Literal classes for each strongly grouped semantic cluster.",
                        "candidate_clusters": [
                            {
                                "cluster_id": c["cluster_id"],
                                "category_hint": c["suggested_category"],
                                "cohesion_score": c["cohesion_score"],
                                "terms": c["terms"],
                            }
                            for c in selected_clusters
                        ],
                    }
                    payload = json.dumps(payload_data, indent=2)
                else:
                    payload = ""
            else:
                payload = json.dumps(all_enum_nodes)

            if payload:
                orphan_sys_prompt = (
                    f"{ORPHAN_SYNTHESIS_SYSTEM_PROMPT}\n"
                    f"Master Themes Context: {json.dumps(master_themes)}"
                )
                save_payload_file(
                    "phase1_enums_payload.txt",
                    f"=== SYSTEM PROMPT ===\n{ORPHAN_SYNTHESIS_SYSTEM_PROMPT}\n\n"
                    f"=== SYSTEM CONTEXT ===\nMaster Themes Context: {json.dumps(master_themes)}\n\n"
                    f"=== USER PAYLOAD ===\n{payload}",
                )

                enum_results = self._run_module_batch(
                    items=[{"id": "phase1_enums", "payload": payload, "fallback_module_name": "enums"}],
                    system_prompt=orphan_sys_prompt,
                    phase_label="Phase 1: Enums",
                    default_module_name="enums",
                    allow_reformat_repair=True,
                )
                if enum_results and enum_results[0]["success"]:
                    global_enums_code = enum_results[0]["source_code"]
                else:
                    err_msg = enum_results[0]["error"] if enum_results else "Unknown error"
                    log_error(f"[Synthesis {path_type}] Error generating enums: {err_msg}")
                    global_enums_code = "# Error generating enums"
            else:
                print("      -> No candidate terms passed cohesion thresholds. Skipping enum generation.")
                global_enums_code = "# No strongly grouped enums qualified"

            if run_norm and global_enums_code and not global_enums_code.startswith("# Error"):
                with open(norm_dir / "enums.py", "w", encoding="utf-8") as f:
                    f.write(global_enums_code)
                self._checkpoint("Phase 1: Enums", "Saved normalized enums.py", norm_dir / "enums.py")
            if run_raw and global_enums_code and not global_enums_code.startswith("# Error"):
                with open(raw_dir / "enums.py", "w", encoding="utf-8") as f:
                    f.write(global_enums_code)
                self._checkpoint("Phase 1: Enums", "Saved raw enums.py", raw_dir / "enums.py")

        # =====================================================================
        # PHASE 2: Schema Generation (Community Targets)
        # =====================================================================
        print(f"   -> [{path_label}] Phase 2: Schema Generation...")
        print(f"      -> Target Composition: {len(targets)} qualified schema targets (global hubs routed to enums.py)")

        payload_mode = synth_cfg.get("payload_values", PAYLOAD_VALUES_MODE)
        max_triplets_per_target = int(synth_cfg.get("max_triplets_per_target", MAX_TRIPLETS_PER_TARGET))
        max_sentences_per_target = int(synth_cfg.get("max_sentences_per_target", MAX_SENTENCES_PER_TARGET))
        top_domain_terms_count = int(synth_cfg.get("top_domain_terms_count", TOP_DOMAIN_TERMS_COUNT))
        min_sentence_length = int(synth_cfg.get("min_sentence_length", MIN_SENTENCE_LENGTH))
        domain_terms_idf_exponent = float(synth_cfg.get("domain_terms_idf_exponent", DOMAIN_TERMS_IDF_EXPONENT))
        domain_terms_multi_word_boost = float(synth_cfg.get("domain_terms_multi_word_boost", DOMAIN_TERMS_MULTI_WORD_BOOST))
        observation_overlap_weight = float(synth_cfg.get("observation_overlap_weight", OBSERVATION_OVERLAP_WEIGHT))
        observation_overlap_max_cap = int(synth_cfg.get("observation_overlap_max_cap", OBSERVATION_OVERLAP_MAX_CAP))
        print(
            f"      -> Payload Limits: max_triplets_per_target={max_triplets_per_target}, "
            f"max_sentences_per_target={max_sentences_per_target}, top_domain_terms={top_domain_terms_count}"
        )

        # Precompute global target counts to score community-specificity of terms (C-TF-IDF)
        global_target_counts = Counter()
        for t in targets:
            for n in t.get("nodes", []):
                global_target_counts[str(n).lower().strip()] += 1

        cfg_exclusions = synth_cfg.get("generic_exclusions", GENERIC_EXCLUSIONS)
        generic_exclusions_set = {str(x).lower().strip() for x in cfg_exclusions}

        def extract_top_domain_terms(target_nodes: List[str], triplets: List[dict], top_k: int = 12) -> List[str]:
            target_nodes_norm = {str(n).lower().strip(): str(n) for n in target_nodes}
            local_deg = Counter()
            for tr in triplets:
                s = str(tr.get("subject", "")).lower().strip()
                o = str(tr.get("object", "")).lower().strip()
                if s in target_nodes_norm:
                    local_deg[s] += 1
                if o in target_nodes_norm:
                    local_deg[o] += 1

            scored = []
            for nk, orig in target_nodes_norm.items():
                if nk in generic_exclusions_set or len(nk) <= 2 or nk.isdigit() or not any(c.isalnum() for c in nk):
                    continue
                n_targets = global_target_counts.get(nk, 1)
                deg = local_deg.get(nk, 0)
                multi_word_boost = domain_terms_multi_word_boost if " " in nk else 1.0
                score = ((deg + 1.0) / (n_targets ** domain_terms_idf_exponent)) * multi_word_boost
                scored.append((score, deg, orig))

            scored.sort(key=lambda x: x[0], reverse=True)

            seen_stems = set()
            cleaned_terms = []
            for _, _, orig in scored:
                base = orig.lower().strip()
                stem = base[:-1] if base.endswith("s") else base
                if stem in seen_stems or base in seen_stems:
                    continue
                seen_stems.add(stem)
                seen_stems.add(base)
                cleaned_terms.append(orig)
                if len(cleaned_terms) >= top_k:
                    break
            return cleaned_terms

        def extract_thematic_observations(
            triplets: List[dict],
            target_nodes: List[str],
            max_sentences: int = 60,
            mode: str = "raw_text",
        ) -> Dict[str, List[str]]:
            target_nodes_norm = {str(n).lower().strip() for n in target_nodes}
            quote_info: Dict[str, Dict[str, Any]] = {}

            for tr in triplets:
                subj = str(tr.get("subject", "")).strip()
                pred = str(tr.get("predicate", "")).strip()
                obj = str(tr.get("object", "")).strip()
                svo = f"{subj} {pred} {obj}." if (subj and pred and obj) else ""

                quote = str(tr.get("source_quote", "")).strip()
                if not quote or len(quote) < min_sentence_length:
                    if mode in ["triples", "both"] and svo:
                        quote = svo
                    else:
                        continue

                theme = tr.get("theme_association", "General Domain Observations") or "General Domain Observations"
                fid = float(tr.get("contextual_fidelity", 0.50))

                if quote not in quote_info:
                    quote_info[quote] = {"theme": theme, "fidelities": [fid], "svos": [svo] if svo else []}
                else:
                    quote_info[quote]["fidelities"].append(fid)
                    if svo and svo not in quote_info[quote]["svos"]:
                        quote_info[quote]["svos"].append(svo)

            scored_quotes = []
            for q, info in quote_info.items():
                avg_fid = sum(info["fidelities"]) / len(info["fidelities"])
                q_lower = q.lower()
                node_overlap = sum(1 for n in target_nodes_norm if n in q_lower)
                sentence_score = avg_fid * (
                    1.0 + observation_overlap_weight * min(observation_overlap_max_cap, node_overlap)
                )

                if mode == "triples":
                    text_item = " ".join(info["svos"]) if info["svos"] else q
                elif mode == "both":
                    svo_summary = f" [Facts: {'; '.join(info['svos'][:2])}]" if info["svos"] else ""
                    text_item = f"{q}{svo_summary}"
                else:
                    text_item = q

                scored_quotes.append((sentence_score, text_item, info["theme"]))

            scored_quotes.sort(key=lambda x: x[0], reverse=True)
            selected = scored_quotes[:max_sentences]

            theme_groups = defaultdict(list)
            for _, text_item, theme in selected:
                theme_groups[theme].append(text_item)

            return dict(theme_groups)

        def format_phase2_payload(
            target_id: Any,
            target_nodes: List[str],
            subset: List[dict],
            mode: str,
            merged_from: Optional[List[Any]] = None,
        ) -> str:
            capped_subset = subset[:max_triplets_per_target] if max_triplets_per_target > 0 else subset
            top_terms = extract_top_domain_terms(target_nodes, capped_subset, top_k=top_domain_terms_count)
            observations = extract_thematic_observations(
                capped_subset, target_nodes, max_sentences=max_sentences_per_target, mode=mode
            )

            theme_counts = Counter(
                tr.get("theme_association")
                for tr in capped_subset
                if tr.get("theme_association") and tr.get("theme_association") != "Other"
            )
            primary_theme = theme_counts.most_common(1)[0][0] if theme_counts else ""
            terms_context = f" — Focus: {', '.join(top_terms[:3])}" if top_terms else ""
            domain_focus = (
                f"{primary_theme}{terms_context}"
                if primary_theme
                else (f"Domain Focus: {', '.join(top_terms[:3])}" if top_terms else f"Community #{target_id}")
            )

            payload_dict: Dict[str, Any] = {
                "community_id": target_id,
                "target_domain_focus": domain_focus,
                "top_domain_terms": top_terms,
                "thematic_observations": observations,
            }
            if merged_from and len(merged_from) > 1:
                payload_dict["consolidated_sub_communities"] = merged_from
            return json.dumps(payload_dict, indent=2)

        sem_cfg = (
            self.config.get("topology", {}).get("semantic_clustering", {})
            if "topology" in self.config
            else self.config.get("semantic_clustering", {})
        )
        exclude_hubs = sem_cfg.get("exclude_hub_triplets_from_communities", False)
        strictly_intra = sem_cfg.get("triplet_payload_mode", "include_boundary") == "strictly_intra_only"
        raw_hubs_norm = {str(h).lower().strip() for h in targets_data.get("global_hubs", [])}

        phase2_batch_items: List[Dict[str, Any]] = []
        active_sys_prompt = LEIDEN_SCHEMA_SYNTHESIS_PROMPT

        for i, target in enumerate(targets, start=1):
            target_nodes = set(target["nodes"])
            target_type = target.get("target_type", target.get("type", "spoke"))
            target_id = target.get("target_id", target.get("id"))
            merged_from = target.get("merged_from", [target_id])

            if run_norm:
                norm_subset = list(target.get("triplets", []))
                if max_triplets_per_target > 0:
                    norm_subset = norm_subset[:max_triplets_per_target]
                if norm_subset:
                    existing_norm = list(norm_dir.glob(f"{i:02d}_*.py")) if RESUME_EXISTING_SCHEMAS else []
                    if existing_norm:
                        print(f"         -> [Resume] Skipping normalized Target {i}/{len(targets)} ({existing_norm[0].name} exists)")
                    else:
                        print(
                            f"         -> Queuing Target {i}/{len(targets)} ({target_type} {target_id}) - normalized "
                            f"(Size: {len(target_nodes)} nodes, {len(norm_subset)}/{max_triplets_per_target} max triplets)..."
                        )
                        user_payload = format_phase2_payload(
                            target_id, list(target_nodes), norm_subset, payload_mode, merged_from=merged_from
                        )
                        save_payload_file(
                            f"phase2_target_{i:02d}_normalized_{target_id}_payload.txt",
                            f"=== SYSTEM PROMPT ===\n{active_sys_prompt}\n\n=== USER PAYLOAD ===\n{user_payload}",
                        )
                        phase2_batch_items.append({
                            "id": f"target_{i:02d}_normalized_{target_id}",
                            "target_index": i,
                            "target_type": target_type,
                            "target_id": target_id,
                            "pass_name": "normalized",
                            "target_dir": norm_dir,
                            "payload": user_payload,
                            "fallback_module_name": f"community_{target_id}",
                        })

            if run_raw:
                raw_subset = list(target.get("triplets", []))
                if original_triplets:
                    target_nodes_norm = {str(n).lower().strip() for n in target_nodes}
                    existing_quotes = {tr.get("source_quote") for tr in raw_subset if tr.get("source_quote")}
                    for extra in original_triplets:
                        s = str(extra.get("subject", "")).lower().strip()
                        o = str(extra.get("object", "")).lower().strip()
                        if not s or not o:
                            continue
                        if exclude_hubs and (s in raw_hubs_norm or o in raw_hubs_norm):
                            continue
                        is_s = s in target_nodes_norm
                        is_o = o in target_nodes_norm
                        if strictly_intra:
                            if not (is_s and is_o):
                                continue
                        else:
                            if not (is_s or is_o):
                                continue
                        q = extra.get("source_quote")
                        if q and q not in existing_quotes:
                            raw_subset.append(extra)
                            existing_quotes.add(q)
                if max_triplets_per_target > 0 and len(raw_subset) > max_triplets_per_target:
                    raw_subset = TargetResolver.filter_and_cap_triplets(
                        raw_subset,
                        list(target_nodes),
                        max_triplets_cap=max_triplets_per_target,
                        exclude_hubs=exclude_hubs,
                        raw_hubs=targets_data.get("global_hubs", []),
                        strictly_intra_only=strictly_intra,
                    )
                if max_triplets_per_target > 0 and len(raw_subset) > max_triplets_per_target:
                    raw_subset = TargetResolver.filter_and_cap_triplets(
                        target_nodes,
                        raw_subset,
                        topology=targets_data,
                        max_triplets_cap=max_triplets_per_target,
                        target_id=target_id,
                        path_type=path_type,
                        config=self.config,
                    )
                if raw_subset:
                    existing_raw = list(raw_dir.glob(f"{i:02d}_*.py")) if RESUME_EXISTING_SCHEMAS else []
                    if existing_raw:
                        print(f"         -> [Resume] Skipping raw Target {i}/{len(targets)} ({existing_raw[0].name} exists)")
                    else:
                        print(
                            f"         -> Queuing Target {i}/{len(targets)} ({target_type} {target_id}) - raw "
                            f"(Size: {len(target_nodes)} nodes, {len(raw_subset)}/{max_triplets_per_target} max triplets)..."
                        )
                        user_payload = format_phase2_payload(
                            target_id, list(target_nodes), raw_subset, payload_mode, merged_from=merged_from
                        )
                        save_payload_file(
                            f"phase2_target_{i:02d}_raw_{target_id}_payload.txt",
                            f"=== SYSTEM PROMPT ===\n{active_sys_prompt}\n\n=== USER PAYLOAD ===\n{user_payload}",
                        )
                        phase2_batch_items.append({
                            "id": f"target_{i:02d}_raw_{target_id}",
                            "target_index": i,
                            "target_type": target_type,
                            "target_id": target_id,
                            "pass_name": "raw",
                            "target_dir": raw_dir,
                            "payload": user_payload,
                            "fallback_module_name": f"community_{target_id}",
                        })

        if phase2_batch_items:
            saved_phase2_ids: set = set()

            def _save_phase2_module(item: Dict[str, Any], mod_name: str, code: str) -> None:
                if item["id"] in saved_phase2_ids:
                    return
                saved_phase2_ids.add(item["id"])
                i = item["target_index"]
                pass_name = item["pass_name"]
                target_dir = item["target_dir"]
                filename = f"{i:02d}_{mod_name}.py"
                out_path = target_dir / filename
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(code)
                self._checkpoint(
                    f"Phase 2 ({pass_name})",
                    f"Saved Target {i} schema module",
                    out_path,
                )

            phase2_results = self._run_module_batch(
                items=phase2_batch_items,
                system_prompt=active_sys_prompt,
                phase_label="Phase 2: Community Schemas",
                default_module_name="community_schema",
                allow_reformat_repair=True,
                on_valid_module_ready=_save_phase2_module,
            )
            for item, res in zip(phase2_batch_items, phase2_results):
                i = item["target_index"]
                pass_name = item["pass_name"]
                if res["success"] and res["source_code"]:
                    _save_phase2_module(item, res["module_name"], res["source_code"])
                else:
                    log_error(
                        f"[Synthesis {path_type}] Error generating {pass_name} schema {i} "
                        f"({item['target_type']} {item['target_id']}): {res['error']}"
                    )

        # =====================================================================
        # PHASE 3: Global Consolidation
        # =====================================================================
        print(f"   -> [{path_label}] Phase 3: Consolidation...")

        def strip_comments_and_docstrings(code: str) -> str:
            code = re.sub(r"#.*", "", code)
            code = re.sub(r'""".*?"""', "", code, flags=re.DOTALL)
            code = re.sub(r"'''.*?'''", "", code, flags=re.DOTALL)
            code = re.sub(r"\n\s*\n", "\n", code)
            return code.strip()

        self._ensure_engine_loaded("Phase 3: Consolidation")
        synthesis_cap = self._apply_context_window_cap()
        sys_tokens = estimate_tokens(CONSOLIDATION_SYSTEM_PROMPT)
        max_payload_tokens = max(500, synthesis_cap - sys_tokens - CONSOLIDATION_MAX_NEW_TOKENS)

        def run_consolidation_single(payload: str, label: str) -> str:
            res_list = self._run_module_batch(
                items=[{"id": label, "payload": payload, "fallback_module_name": "master_ontology"}],
                system_prompt=CONSOLIDATION_SYSTEM_PROMPT,
                phase_label=f"Phase 3: Consolidation ({label})",
                default_module_name="master_ontology",
                allow_reformat_repair=True,
                max_new_tokens=CONSOLIDATION_MAX_NEW_TOKENS,
            )
            if res_list and res_list[0]["success"]:
                return res_list[0]["source_code"]
            raise RuntimeError(res_list[0]["error"] if res_list else "Consolidation failed")

        def process_consolidation_sync(files_list: List[dict], schema_dir: Path) -> str:
            payload = "\n".join([f"--- File: {f['name']} ---\n{f['content']}\n" for f in files_list])
            save_payload_file(
                f"phase3_consolidation_{schema_dir.name}_payload.txt",
                f"=== SYSTEM PROMPT ===\n{CONSOLIDATION_SYSTEM_PROMPT}\n\n=== USER PAYLOAD ===\n{payload}",
            )
            if estimate_tokens(payload) <= max_payload_tokens:
                return run_consolidation_single(payload, schema_dir.name)

            for f in files_list:
                f["content"] = strip_comments_and_docstrings(f["content"])
            payload = "\n".join([f"--- File: {f['name']} ---\n{f['content']}\n" for f in files_list])
            if estimate_tokens(payload) <= max_payload_tokens:
                return run_consolidation_single(payload, schema_dir.name)

            sub_batches, current_batch = [], []
            for f in files_list:
                test_batch = current_batch + [f]
                test_payload = "\n".join([f"--- File: {x['name']} ---\n{x['content']}\n" for x in test_batch])
                if estimate_tokens(test_payload) > max_payload_tokens and current_batch:
                    sub_batches.append(current_batch)
                    current_batch = [f]
                else:
                    current_batch.append(f)
            if current_batch:
                sub_batches.append(current_batch)

            intermediate_results = []
            for idx, batch in enumerate(sub_batches, start=1):
                res = process_consolidation_sync(batch, schema_dir)
                intermediate_results.append({"name": f"intermediate_{idx}.py", "content": res})

            return process_consolidation_sync(intermediate_results, schema_dir)

        def consolidate_schemas(schema_dir: Path) -> None:
            py_files = sorted([
                f for f in schema_dir.glob("*.py")
                if f.name not in ("master_ontology.py", "enums.py", "__init__.py")
            ])
            if not py_files:
                return

            files_list = []
            for pf in py_files:
                with open(pf, "r", encoding="utf-8") as f:
                    files_list.append({"name": pf.name, "content": f.read()})

            try:
                master_code = process_consolidation_sync(files_list, schema_dir)
                master_code = deduplicate_consolidated_classes(master_code)
                out_master = schema_dir / "master_ontology.py"
                with open(out_master, "w", encoding="utf-8") as f:
                    f.write(master_code)
                self._checkpoint("Phase 3: Consolidation", f"Saved {schema_dir.name}/master_ontology.py", out_master)
            except Exception as e:
                log_error(f"[Synthesis {path_type}] Error generating master_ontology for {schema_dir.name}: {e}")

        if run_norm:
            consolidate_schemas(norm_dir)
        if run_raw:
            consolidate_schemas(raw_dir)

        # =====================================================================
        # PHASE 4: Path Comprehensive Ontology
        # =====================================================================
        print(f"   -> [{path_label}] Phase 4: Master Ontology Synthesis...")
        norm_master = (
            (norm_dir / "master_ontology.py").read_text(encoding="utf-8")
            if run_norm and (norm_dir / "master_ontology.py").exists()
            else ""
        )
        raw_master = (
            (raw_dir / "master_ontology.py").read_text(encoding="utf-8")
            if run_raw and (raw_dir / "master_ontology.py").exists()
            else ""
        )

        enums_code = ""
        if run_norm and (norm_dir / "enums.py").exists():
            enums_code = (norm_dir / "enums.py").read_text(encoding="utf-8")
        elif run_raw and (raw_dir / "enums.py").exists():
            enums_code = (raw_dir / "enums.py").read_text(encoding="utf-8")

        final_payload = (
            f"--- ENUMS ---\n{enums_code}\n\n"
            f"--- RAW MASTER ONTOLOGY ---\n{raw_master}\n\n"
            f"--- NORMALIZED MASTER ONTOLOGY ---\n{norm_master}"
        )
        save_payload_file(
            "phase4_comprehensive_ontology_payload.txt",
            f"=== SYSTEM PROMPT ===\n{FINAL_ONTOLOGY_SYSTEM_PROMPT}\n\n=== USER PAYLOAD ===\n{final_payload}",
        )

        final_results = self._run_module_batch(
            items=[{"id": "comprehensive_ontology", "payload": final_payload, "fallback_module_name": "comprehensive_ontology"}],
            system_prompt=FINAL_ONTOLOGY_SYSTEM_PROMPT,
            phase_label="Phase 4: Comprehensive Ontology",
            default_module_name="comprehensive_ontology",
            allow_reformat_repair=True,
            max_new_tokens=CONSOLIDATION_MAX_NEW_TOKENS,
        )
        if final_results and final_results[0]["success"]:
            final_code = deduplicate_consolidated_classes(final_results[0]["source_code"])
        else:
            err_msg = final_results[0]["error"] if final_results else "Unknown error"
            log_error(f"[Synthesis {path_type}] Error generating comprehensive_ontology: {err_msg}")
            final_code = "# Error generating comprehensive ontology"

        comp_path = output_base_dir / "comprehensive_ontology.py"
        with open(comp_path, "w", encoding="utf-8") as f:
            f.write(final_code)
        self._checkpoint("Phase 4: Comprehensive Ontology", "Saved comprehensive_ontology.py", comp_path)

        # Finalization (__init__.py & optional ruff format)
        if run_norm:
            with open(norm_dir / "__init__.py", "w", encoding="utf-8") as f:
                f.write("# Normalized Schemas")
            try:
                subprocess.run(["ruff", "format", str(norm_dir)], capture_output=True, text=True, check=False)
            except Exception as e:
                log_error(f"[Synthesis {path_type}] Error running ruff on norm: {e}")

        if run_raw:
            with open(raw_dir / "__init__.py", "w", encoding="utf-8") as f:
                f.write("# Raw Schemas")
            try:
                subprocess.run(["ruff", "format", str(raw_dir)], capture_output=True, text=True, check=False)
            except Exception as e:
                log_error(f"[Synthesis {path_type}] Error running ruff on raw: {e}")

        try:
            subprocess.run(["ruff", "format", str(comp_path)], capture_output=True, text=True, check=False)
        except Exception as e:
            log_error(f"[Synthesis {path_type}] Error running ruff on comprehensive: {e}")

        print(f"   [OK] [{path_label}] Synthesis complete: {output_base_dir}")


# ============================================================================
# CONFIGURATION & CLI ENTRYPOINT (Matches run_stage_4_synthesis.py)
# ============================================================================

def build_default_config() -> Dict[str, Any]:
    """Constructs the SemanticPrism Stage 4 configuration dictionary from global variables."""
    return {
        "directories": {
            "outputs": OUTPUTS_DIR,
        },
        "refinement": {
            "embedding_model": EMBEDDING_MODEL,
            "vector_cache_dir": VECTOR_CACHE_DIR,
        },
        "topology": {
            "center_agglomeration_embeddings": CENTER_AGGLOMERATION_EMBEDDINGS,
            "semantic_clustering": {
                "triplet_payload_mode": TRIPLET_PAYLOAD_MODE,
                "exclude_hub_triplets_from_communities": EXCLUDE_HUB_TRIPLETS_FROM_COMMUNITIES,
            },
            "architectural_tiers": {
                "enabled": ARCHITECTURAL_TIERS_ENABLED,
                "weights": {
                    "prototypicality": TIER_WEIGHT_PROTOTYPICALITY,
                    "centrality": TIER_WEIGHT_CENTRALITY,
                    "fidelity": TIER_WEIGHT_FIDELITY,
                },
                "tier_1_percentile": TIER_1_PERCENTILE,
                "tier_1_min_entities": TIER_1_MIN_ENTITIES,
                "tier_1_max_entities": TIER_1_MAX_ENTITIES,
                "tier_2_percentile": TIER_2_PERCENTILE,
                "min_fidelity_tier_1": MIN_FIDELITY_TIER_1,
                "boundary_discount": BOUNDARY_DISCOUNT,
                "fallback_prototypicality": FALLBACK_PROTOTYPICALITY,
                "fallback_fidelity": FALLBACK_FIDELITY,
                "export_tier_csv": EXPORT_TIER_CSV,
                "export_diagnostics": EXPORT_DIAGNOSTICS,
            },
        },
        "synthesis": {
            "llm": {
                "model_name": MODEL_NAME,
                "temperature": TEMPERATURE,
            },
            "context_window_cap": CONTEXT_WINDOW_CAP,
            "save_llm_payloads": SAVE_LLM_PAYLOADS,
            "payload_output_dir": PAYLOAD_OUTPUT_DIR,
            "execution_mode": EXECUTION_MODE,
            "schema_pass_mode": SCHEMA_PASS_MODE,
            "payload_values": PAYLOAD_VALUES_MODE,
            "min_cluster_size": MIN_CLUSTER_SIZE,
            "max_triplets_per_target": MAX_TRIPLETS_PER_TARGET,
            "enum_clustering": {
                "enabled": ENUM_CLUSTERING_ENABLED,
                "min_cluster_size": ENUM_MIN_CLUSTER_SIZE,
                "distance_threshold": ENUM_DISTANCE_THRESHOLD,
                "min_cohesion_score": ENUM_MIN_COHESION_SCORE,
                "max_clusters_per_batch": ENUM_MAX_CLUSTERS_PER_BATCH,
                "max_terms_per_cluster": ENUM_MAX_TERMS_PER_CLUSTER,
            },
            "top_domain_terms_count": TOP_DOMAIN_TERMS_COUNT,
            "domain_terms_idf_exponent": DOMAIN_TERMS_IDF_EXPONENT,
            "domain_terms_multi_word_boost": DOMAIN_TERMS_MULTI_WORD_BOOST,
            "generic_exclusions": GENERIC_EXCLUSIONS,
            "max_sentences_per_target": MAX_SENTENCES_PER_TARGET,
            "min_sentence_length": MIN_SENTENCE_LENGTH,
            "observation_overlap_weight": OBSERVATION_OVERLAP_WEIGHT,
            "observation_overlap_max_cap": OBSERVATION_OVERLAP_MAX_CAP,
        },
    }


def load_json(filepath: str | Path, required: bool = True) -> Any:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        if required:
            print(f"CRITICAL ERROR: Required file {filepath} not found.")
        return None


def main() -> None:
    global DRY_RUN, RESUME_EXISTING_SCHEMAS
    parser = argparse.ArgumentParser(
        description="SemanticPrism Stage 4: Ontological Schema Synthesis Pipeline (Powered by PrismRunner)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run pipeline in dry-run mode without loading the LLM weights.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing Phase 2 community schema .py files on disk.",
    )
    args, _ = parser.parse_known_args()
    if args.dry_run:
        DRY_RUN = True
    if args.resume:
        RESUME_EXISTING_SCHEMAS = True

    print(f"=== SemanticPrism Stage 4: Synthesis Pipeline (Model: {MODEL_NAME}) ===")

    config = build_default_config()

    refined_triplets = load_json(REFINED_TRIPLETS_PATH)
    original_triplets = load_json(ORIGINAL_TRIPLETS_PATH)
    taxonomic_map = load_json(TAXONOMIC_MAP_PATH, required=False) or {}
    master_themes_raw = load_json(MASTER_THEMES_PATH, required=False) or {}

    if not refined_triplets or not original_triplets:
        print("CRITICAL ERROR: Cannot run Stage 4. Missing required preceding stage outputs (Stages 1 & 2).")
        print("Please run preceding stages first ('python3 run_pipeline.py').")
        sys.exit(1)

    synth_mode = config.get("synthesis", {}).get("execution_mode", "community")
    print(f"   -> Synthesis Execution Mode: '{synth_mode}'")

    # Phase 0: Dynamic Target Resolution using CURRENT synthesis config
    print("   -> Phase 0: Resolving Synthesis Targets from Stage 3 Graph Partitions...")
    comm_file = Path(TOPOLOGY_PARTITIONS_PATH)
    if not comm_file.exists():
        comm_file = Path("outputs/03_topology/community/topology_partitions.json")
    comm_topology = load_json(comm_file, required=False)

    resolved_targets_path = Path(RESOLVED_TARGETS_OUTPUT_DIR) / "resolved_community_targets.json"
    if comm_topology:
        comm_target_file = TargetResolver.export_resolved_targets_json(
            comm_topology,
            "community",
            refined_triplets,
            config,
            Path(RESOLVED_TARGETS_OUTPUT_DIR),
        )
        comm_targets = load_json(comm_target_file)
    elif resolved_targets_path.exists():
        print(f"   -> Using pre-existing resolved targets at: {resolved_targets_path}")
        comm_targets = load_json(resolved_targets_path)
    else:
        print(
            f"CRITICAL ERROR: Cannot run Stage 4. Missing required Stage 3 community topology partition outputs ({comm_file})."
        )
        sys.exit(1)

    master_themes: List[str] = []
    if isinstance(master_themes_raw, dict):
        master_themes = master_themes_raw.get("master_themes", [])
    elif isinstance(master_themes_raw, list):
        master_themes = master_themes_raw

    try:
        pipeline = SynthesisPipeline(config)
        pipeline.execute(
            comm_targets=comm_targets,
            refined_triplets=refined_triplets,
            original_triplets=original_triplets,
            taxonomic_map=taxonomic_map,
            master_themes=master_themes,
        )
        print("=== Stage 4 Synthesis Completed Successfully ===")
    except Exception as e:
        print(f"Error during Stage 4 Synthesis execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
