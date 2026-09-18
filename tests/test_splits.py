"""
Tests for partisan_classifier.data.splits.

Once publisher_disjoint_split is implemented, the key property to verify is
that no publisher appears in more than one of the returned splits.
"""

import pytest

from partisan_classifier.data.splits import publisher_disjoint_split

@pytest.mark.skip(reason="Implement once publisher_disjoint_split is written")
def test_splits_are_publisher_disjoint():
    pass
