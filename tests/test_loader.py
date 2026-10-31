"""Tests for partisan_classifier.data.loader."""

import pandas as pd
import pytest

from partisan_classifier.data.loader import validate_schema

def test_validate_schema_passes_with_required_columns():
    df = pd.DataFrame({"text": ["a"], "publisher": ["x"], "label": ["left"]})
    validate_schema(df)  # should not raise

def test_validate_schema_raises_when_missing_column():
    df = pd.DataFrame({"text": ["a"], "label": ["left"]})
    with pytest.raises(ValueError):
        validate_schema(df)
