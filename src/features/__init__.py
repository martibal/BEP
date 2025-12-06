"""
Feature engineering package for ON-CHAIN SUPER SIGNALS™.

Contains:
- base: Base classes and data structures for features
- btc_feature_engine: Bitcoin-specific feature computation
- btc_features: Bitcoin feature definitions
- eth_feature_engine: Ethereum-specific feature computation
- normalizer: Feature normalization utilities
"""

from src.features import base, normalizer

__all__ = ["base", "normalizer"]
