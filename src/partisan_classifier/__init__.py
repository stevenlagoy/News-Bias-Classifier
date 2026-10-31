"""
partisan_classifier
===================

Article-level political partisanship classification (left / center / right),
evaluated with a publisher-disjoint split.

Package layout:
    config      -- config loading (YAML -> dataclass/dict)
    data        -- dataset download, loading, and splitting
    features    -- text featurization
    models      -- model definitions
    training    -- training loops
    evaluation  -- metrics and reporting
    utils       -- shared helpers (logging, seeding, etc.)
"""

__version__ = "0.1.0"
