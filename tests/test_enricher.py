"""Integration tests for the Enricher — Pipeline-based, column-oriented enrichment."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

from lattice.core.config import EnrichmentConfig
from lattice.core.enricher import Enricher
from lattice.core.exceptions import FieldValidationError
from lattice.data.fields import FieldManager
from lattice.pipeline.pipeline import Pipeline
from lattice.steps.function import FunctionStep


# -- helpers -----------------------------------------------------------------


def _mock_field_manager(category: str, fields: dict[str, dict]) -> MagicMock:
    """Build a mock FieldManager that validates *category* and returns *fields*."""
    fm = MagicMock(spec=FieldManager)
    fm.validate_category.side_effect = lambda cat: cat == category
    fm.get_categories.return_value = [category]
    fm.get_category_fields.return_value = fields
    return fm


def _identity_step(name: str, fields: list[str], **kwargs) -> FunctionStep:
    """Step that produces static '{field}_value' for each field."""
    def fn(ctx):
        return {f: f"{f}_value" for f in fields}
    return FunctionStep(name=name, fn=fn, fields=fields, **kwargs)


# -- basic execution ---------------------------------------------------------


class TestBasicExecution:
    def test_single_step(self):
        pipeline = Pipeline([_identity_step("s1", ["company_type"])])
        fm = _mock_field_manager("info", {"company_type": {"prompt": "classify"}})
        enricher = Enricher(pipeline, fm)

        df = pd.DataFrame({"company": ["Acme", "Beta"]})
        result = enricher.run(df, "info")

        assert "company_type" in result.columns
        assert list(result["company_type"]) == ["company_type_value", "company_type_value"]
        # Original column preserved
        assert list(result["company"]) == ["Acme", "Beta"]

    def test_multi_step_with_dependencies(self):
        def step_a_fn(ctx):
            return {"raw": ctx.row.get("input", "") + "_raw"}

        def step_b_fn(ctx):
            return {"processed": ctx.prior_results.get("raw", "") + "_done"}

        pipeline = Pipeline([
            FunctionStep("a", fn=step_a_fn, fields=["raw"]),
            FunctionStep("b", fn=step_b_fn, fields=["processed"], depends_on=["a"]),
        ])
        fm = _mock_field_manager("cat", {
            "raw": {"prompt": "r"},
            "processed": {"prompt": "p"},
        })
        enricher = Enricher(pipeline, fm)

        df = pd.DataFrame({"input": ["hello", "world"]})
        result = enricher.run(df, "cat")

        assert list(result["raw"]) == ["hello_raw", "world_raw"]
        assert list(result["processed"]) == ["hello_raw_done", "world_raw_done"]

    def test_row_data_flows_through(self):
        """Steps can read original row data."""
        def fn(ctx):
            return {"greeting": f"Hello, {ctx.row['name']}"}

        pipeline = Pipeline([FunctionStep("g", fn=fn, fields=["greeting"])])
        fm = _mock_field_manager("greet", {"greeting": {"prompt": "g"}})
        enricher = Enricher(pipeline, fm)

        df = pd.DataFrame({"name": ["Alice", "Bob"]})
        result = enricher.run(df, "greet")

        assert list(result["greeting"]) == ["Hello, Alice", "Hello, Bob"]


# -- internal fields filtered ------------------------------------------------


class TestInternalFieldsFiltered:
    def test_double_underscore_not_in_output(self):
        def search_fn(ctx):
            return {"__web_ctx": "search data", "summary": "based on search"}

        pipeline = Pipeline([
            FunctionStep("search", fn=search_fn, fields=["__web_ctx", "summary"]),
        ])
        fm = _mock_field_manager("cat", {"summary": {"prompt": "s"}})
        enricher = Enricher(pipeline, fm)

        df = pd.DataFrame({"q": ["test"]})
        result = enricher.run(df, "cat")

        assert "summary" in result.columns
        assert "__web_ctx" not in result.columns

    def test_internal_fields_pass_between_steps(self):
        """Internal __ fields are available to downstream steps via prior_results."""
        def step_a_fn(ctx):
            return {"__internal": "secret_data"}

        def step_b_fn(ctx):
            return {"output": f"got: {ctx.prior_results.get('__internal', '')}"}

        pipeline = Pipeline([
            FunctionStep("a", fn=step_a_fn, fields=["__internal"]),
            FunctionStep("b", fn=step_b_fn, fields=["output"], depends_on=["a"]),
        ])
        fm = _mock_field_manager("cat", {"output": {"prompt": "o"}})
        enricher = Enricher(pipeline, fm)

        df = pd.DataFrame({"x": [1]})
        result = enricher.run(df, "cat")

        assert result.at[0, "output"] == "got: secret_data"
        assert "__internal" not in result.columns


# -- preserves original columns ----------------------------------------------


class TestPreservesOriginalColumns:
    def test_original_columns_untouched(self):
        pipeline = Pipeline([_identity_step("s", ["new_field"])])
        fm = _mock_field_manager("cat", {"new_field": {"prompt": "n"}})
        enricher = Enricher(pipeline, fm)

        df = pd.DataFrame({"col_a": [1, 2], "col_b": ["x", "y"]})
        result = enricher.run(df, "cat")

        assert list(result["col_a"]) == [1, 2]
        assert list(result["col_b"]) == ["x", "y"]
        assert "new_field" in result.columns

    def test_input_df_not_mutated(self):
        pipeline = Pipeline([_identity_step("s", ["new_field"])])
        fm = _mock_field_manager("cat", {"new_field": {"prompt": "n"}})
        enricher = Enricher(pipeline, fm)

        df = pd.DataFrame({"col": [1]})
        original_cols = list(df.columns)
        enricher.run(df, "cat")

        assert list(df.columns) == original_cols
        assert "new_field" not in df.columns


# -- field routing validation ------------------------------------------------


class TestFieldRoutingValidation:
    def test_missing_field_raises(self):
        """Category has a field that no step produces."""
        pipeline = Pipeline([_identity_step("s", ["field_a"])])
        fm = _mock_field_manager("cat", {
            "field_a": {"prompt": "a"},
            "field_b": {"prompt": "b"},  # not covered by any step
        })
        enricher = Enricher(pipeline, fm)

        with pytest.raises(FieldValidationError, match="not covered"):
            enricher.run(pd.DataFrame({"x": [1]}), "cat")

    def test_duplicate_field_raises(self):
        """Two steps produce the same category field."""
        pipeline = Pipeline([
            FunctionStep("s1", fn=lambda ctx: {"f": 1}, fields=["f"]),
            FunctionStep("s2", fn=lambda ctx: {"f": 2}, fields=["f"]),
        ])
        fm = _mock_field_manager("cat", {"f": {"prompt": "x"}})
        enricher = Enricher(pipeline, fm)

        with pytest.raises(FieldValidationError, match="multiple steps"):
            enricher.run(pd.DataFrame({"x": [1]}), "cat")

    def test_invalid_category_raises(self):
        pipeline = Pipeline([_identity_step("s", ["f"])])
        fm = _mock_field_manager("valid_cat", {"f": {"prompt": "x"}})
        enricher = Enricher(pipeline, fm)

        with pytest.raises(FieldValidationError, match="not found"):
            enricher.run(pd.DataFrame({"x": [1]}), "wrong_cat")

    def test_extra_fields_warn(self):
        """Steps produce fields not in category (and not __) — should warn."""
        def fn(ctx):
            return {"cat_field": "ok", "extra_field": "bonus"}

        pipeline = Pipeline([
            FunctionStep("s", fn=fn, fields=["cat_field", "extra_field"]),
        ])
        fm = _mock_field_manager("cat", {"cat_field": {"prompt": "x"}})
        enricher = Enricher(pipeline, fm)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            enricher.run(pd.DataFrame({"x": [1]}), "cat")
            assert len(w) == 1
            assert "extra_field" in str(w[0].message)


# -- overwrite behaviour -----------------------------------------------------


class TestOverwriteBehaviour:
    def test_overwrite_false_preserves_existing(self):
        pipeline = Pipeline([
            FunctionStep("s", fn=lambda ctx: {"f": "new_val"}, fields=["f"]),
        ])
        fm = _mock_field_manager("cat", {"f": {"prompt": "x"}})
        config = EnrichmentConfig(overwrite_fields=False)
        enricher = Enricher(pipeline, fm, config)

        df = pd.DataFrame({"f": ["existing", None]})
        result = enricher.run(df, "cat")

        assert result.at[0, "f"] == "existing"
        assert result.at[1, "f"] == "new_val"

    def test_overwrite_true_replaces(self):
        pipeline = Pipeline([
            FunctionStep("s", fn=lambda ctx: {"f": "new_val"}, fields=["f"]),
        ])
        fm = _mock_field_manager("cat", {"f": {"prompt": "x"}})
        config = EnrichmentConfig(overwrite_fields=True)
        enricher = Enricher(pipeline, fm, config)

        df = pd.DataFrame({"f": ["existing", None]})
        result = enricher.run(df, "cat")

        assert result.at[0, "f"] == "new_val"
        assert result.at[1, "f"] == "new_val"

    def test_overwrite_param_overrides_config(self):
        pipeline = Pipeline([
            FunctionStep("s", fn=lambda ctx: {"f": "new_val"}, fields=["f"]),
        ])
        fm = _mock_field_manager("cat", {"f": {"prompt": "x"}})
        config = EnrichmentConfig(overwrite_fields=False)
        enricher = Enricher(pipeline, fm, config)

        df = pd.DataFrame({"f": ["existing"]})
        result = enricher.run(df, "cat", overwrite_fields=True)

        assert result.at[0, "f"] == "new_val"


# -- sync wrapper ------------------------------------------------------------


class TestSyncWrapper:
    def test_run_works_outside_async(self):
        """enricher.run() works when no event loop is running."""
        pipeline = Pipeline([_identity_step("s", ["f"])])
        fm = _mock_field_manager("cat", {"f": {"prompt": "x"}})
        enricher = Enricher(pipeline, fm)

        df = pd.DataFrame({"x": [1]})
        result = enricher.run(df, "cat")
        assert "f" in result.columns

    @pytest.mark.asyncio
    async def test_run_raises_in_async_context(self):
        """enricher.run() raises RuntimeError when called inside async."""
        pipeline = Pipeline([_identity_step("s", ["f"])])
        fm = _mock_field_manager("cat", {"f": {"prompt": "x"}})
        enricher = Enricher(pipeline, fm)

        with pytest.raises(RuntimeError, match="run_async"):
            enricher.run(pd.DataFrame({"x": [1]}), "cat")

    @pytest.mark.asyncio
    async def test_run_async_works(self):
        pipeline = Pipeline([_identity_step("s", ["f"])])
        fm = _mock_field_manager("cat", {"f": {"prompt": "x"}})
        enricher = Enricher(pipeline, fm)

        df = pd.DataFrame({"x": [1, 2]})
        result = await enricher.run_async(df, "cat")
        assert list(result["f"]) == ["f_value", "f_value"]


# -- checkpoint resume -------------------------------------------------------


class TestCheckpointResume:
    def test_completed_steps_skipped(self, tmp_path):
        """Manually write a checkpoint file and verify step 1 is skipped on re-run."""
        call_tracker = {"step1_calls": 0, "step2_calls": 0}

        def step1_fn(ctx):
            call_tracker["step1_calls"] += 1
            return {"f1": "from_step1"}

        def step2_fn(ctx):
            call_tracker["step2_calls"] += 1
            return {"f2": ctx.prior_results.get("f1", "") + "_processed"}

        pipeline = Pipeline([
            FunctionStep("step1", fn=step1_fn, fields=["f1"]),
            FunctionStep("step2", fn=step2_fn, fields=["f2"], depends_on=["step1"]),
        ])

        fields_dict = {"f1": {"prompt": "a"}, "f2": {"prompt": "b"}}
        fm = _mock_field_manager("cat", fields_dict)

        config = EnrichmentConfig(
            enable_checkpointing=True,
            auto_resume=True,
            checkpoint_dir=str(tmp_path),
        )

        # Manually write a checkpoint where step1 is already complete
        checkpoint_data = {
            "timestamp": 1000.0,
            "category": "cat",
            "total_rows": 2,
            "fields_dict": fields_dict,
            "completed_steps": ["step1"],
            "step_results": {
                "step1": [{"f1": "cached_val"}, {"f1": "cached_val2"}],
            },
        }

        # Determine the checkpoint path (mirror the manager's logic)
        data_id = f"df_{hash(str(['x']) + str(2))}"
        safe_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in data_id)
        cp_path = tmp_path / f"{safe_id}_cat_checkpoint.json"
        with open(cp_path, "w") as f:
            json.dump(checkpoint_data, f)

        enricher = Enricher(pipeline, fm, config)
        df = pd.DataFrame({"x": [1, 2]})
        result = enricher.run(df, "cat")

        # step1 should NOT have been called (it was checkpointed)
        assert call_tracker["step1_calls"] == 0
        # step2 SHOULD have been called
        assert call_tracker["step2_calls"] == 2
        # step2 should use the cached step1 results
        assert result.at[0, "f2"] == "cached_val_processed"
        assert result.at[1, "f2"] == "cached_val2_processed"

    def test_checkpoint_cleaned_on_success(self, tmp_path):
        """Checkpoint file should be removed after successful completion."""
        pipeline = Pipeline([_identity_step("s", ["f"])])
        fm = _mock_field_manager("cat", {"f": {"prompt": "x"}})
        config = EnrichmentConfig(
            enable_checkpointing=True,
            checkpoint_dir=str(tmp_path),
        )
        enricher = Enricher(pipeline, fm, config)

        df = pd.DataFrame({"x": [1]})
        enricher.run(df, "cat")

        # No checkpoint files should remain
        files = list(tmp_path.glob("*_checkpoint.json"))
        assert len(files) == 0
