# SemanticPrism4 AST Sanitization & Code Cleanup Rules

This document catalogs all pre-AST text normalization, Enum AST sanitization, `BaseModel` AST sanitization, and consolidated ontology deduplication rules implemented in [`SemanticPrism4.py`](./SemanticPrism4.py).

All AST sanitization events (stripped or collapsed Enum members and `BaseModel` fields) are logged in real time to the console (when `PRINT_CHECKPOINTS = True`) and persisted to:
- `outputs/04_synthesis/logs/schema_sanitization.log` via `_log_sanitization_detail` (`lines 451–494`).

---

## 1. Pre-AST Text Normalization (`clean_python_code`, `lines 845–899`)

Executed on raw LLM-generated `source_code` before parsing into a Python AST:

1. **Markdown Code Fence Stripping (`lines 850–855`)**
   - Strips leading ```` ```python ```` or ```` ``` ```` and trailing ```` ``` ```` wrappers.
2. **Escape Sequence Unescaping (`lines 858–861`)**
   - Replaces over-escaped quotes (`\"` -> `"`) and literal `\n` sequences with actual newlines.
3. **Invalid Python Identifier Fixes (`lines 863–883`)**
   - **Assignment LHS (`=`)**: Replaces spaces or commas inside variable/member identifiers with underscores (`_`).
   - **Type Annotation LHS (`:`)**: Replaces hyphens in field names with underscores (e.g., `help-seeking_barriers: ...` -> `help_seeking_barriers: ...`).
4. **Non-ASCII / CJK Character Stripping (`line 888`)**
   - Removes CJK characters (`[\u4e00-\u9fff]+`) that bilingual LLMs occasionally emit in comments or docstrings.
5. **Meta-Model Class Regex Removal (`lines 891–896`)**
   - Strips any accidental classes that model the prompt/JSON envelope itself (`ThemeObservation`, `ThematicObservation`, `GeneratedModule`).

---

## 2. Enum AST Sanitizer (`_sanitize_ast_enums`, `lines 519–630`)

Applied to every `Enum` class definition in the parsed AST:

1. **Verbose / Non-Atomic Member Stripping (`_is_verbose_or_runaway_enum_member`, `lines 497–516` & `lines 564–565`)**
   - Strips any `Enum` member whose identifier exceeds **4 words** (`_`-separated) or **35 characters**.
   - Strips any `Enum` member whose string value exceeds **5 words** or **40 characters**.
   - **Log reason**: `verbose/non-atomic member (>4 words or >35 chars)`
2. **Duplicate Member Name Stripping (`lines 566–567`)**
   - Strips repeated member identifiers within the same `Enum` class.
   - **Log reason**: `duplicate member name`
3. **Duplicate Member Value Stripping (`lines 568–569`)**
   - Strips members whose normalized lowercase string value duplicates an earlier member in the same `Enum` class.
   - **Log reason**: `duplicate member value`
4. **Max Enum Member Cap (`lines 570–571`)**
   - Caps each `Enum` class at `MAX_ENUM_MEMBERS_PER_CLASS` (**12** members), stripping any subsequent members.
   - **Log reason**: `exceeded MAX_ENUM_MEMBERS_PER_CLASS (12)`
5. **Empty Enum Removal & Type Annotation Rewriting (`lines 597–619`)**
   - If an `Enum` class has `0` valid members remaining after filtering, the `Enum` class is removed from the AST and `_DroppedEnumRewriter` rewrites any `BaseModel` field type annotations referencing that `Enum` to `str`.
   - **Log note**: `Enum class had 0 valid members remaining and was removed (field references rewritten to 'str').`

---

## 3. BaseModel AST Sanitizer (`_sanitize_ast_models`, `lines 633–842`)

Applied to every non-`Enum` class (`BaseModel`) in the parsed AST:

1. **Duplicate Field Name Stripping (`lines 693–697`)**
   - Strips fields with duplicate normalized lowercase names within the same `BaseModel` class.
   - **Log reason**: `duplicate field name`
2. **Shared 2-Token Prefix Collapse / Cap (`lines 721–755`)**
   - Applies to fields with `>= 3` underscore-separated tokens (`parts[0]_parts[1]_...`):
   - **Collapse (`>= 3` fields sharing a 2-token prefix)**: Collapses all fields sharing `prefix2` into a single `Optional[List[str]]` field named `<prefix2>s` (with `Field(default=None, description="Associated <prefix2>s.")`).
     - **Log reason**: `collapsed N fields sharing prefix '<prefix2>_*'`
   - **Cap (`max_shared_prefix = 2`)**: If a 2-token prefix has already appeared twice on the class, subsequent fields sharing that prefix are stripped.
     - **Log reason**: `exceeded shared prefix cap (2) for '<prefix2>_*'`
3. **Shared Suffix Collapse / Cap (`lines 757–789`)**
   - Applies to fields with `>= 2` tokens whose final token (`suffix1`) is **not** in `exempt_suffixes` (`id`, `name`, `type`, `types`, `status`, `level`, `score`, `range`, `date`, `time`, `plan`, `notes`, `summary`, `description`, `details`, `count`, `rate`, `list`, `items`, `category`, `mode`, `source`):
   - **Collapse (`>= 3` fields sharing a non-exempt suffix)**: Collapses all fields ending in `_*<suffix1>` into a single `Optional[List[str]]` field named `<suffix1>s`.
     - **Log reason**: `collapsed N fields sharing suffix '*_<suffix1>'`
   - **Cap (`max_shared_suffix = 2`)**: Caps non-exempt shared suffixes at 2 fields per class.
     - **Log reason**: `exceeded shared suffix cap (2) for '*_<suffix1>'`
4. **Max BaseModel Field Cap (`lines 798–817`)**
   - Enforces `MAX_BASEMODEL_FIELDS_PER_CLASS` (**12** fields) per `BaseModel` class.
   - When a class has `> 12` fields after Pass 1, fields are prioritized using the sort key:
     1. Non-boolean fields before boolean fields (`1 if is_bool else 0`)
     2. Fields without repeated 2-token prefixes (`1 if len(parts) >= 3 and raw_prefix2_counts[prefix2] > 1 else 0`)
     3. Concise 1–2 token core fields before `3+` token compound fields (`1 if len(parts) >= 3 else 0`)
     4. Original declaration order (`idx`)
   - Keeps the top 12 fields (restored to their original declaration order) and strips the overflow.
   - **Log reason**: `exceeded MAX_BASEMODEL_FIELDS_PER_CLASS (12)`
5. **Empty Class Fallback (`lines 829–833`)**
   - If a class has `0` fields remaining, inserts `pass` so the class definition remains valid Python.

---

## 4. Consolidated Ontology AST Merger (`deduplicate_consolidated_classes`, `lines 902–982`)

Applied in Phase 2 to `master_ontology.py` and `comprehensive_ontology.py`:

1. **Duplicate Class Merging (`lines 933–966`)**
   - Merges multiple `Enum` classes with the same name by unioning their unique members in order of appearance.
   - Merges multiple `BaseModel` classes with the same name by unioning their unique fields in order of appearance.
2. **Enum vs. BaseModel Name Collision Disambiguation (`lines 968–974`)**
   - If both an `Enum` and a `BaseModel` share the same class name `<Name>`, renames the `BaseModel` to `<Name>Detail`.
3. **Post-Merge Re-Sanitization (`lines 978–979`)**
   - Re-runs `_sanitize_ast_enums` and `_sanitize_ast_models` (with context label `"master_ontology (merged)"`) on the merged AST so merged classes strictly respect `MAX_ENUM_MEMBERS_PER_CLASS` and `MAX_BASEMODEL_FIELDS_PER_CLASS`.

