"""Consent-gated, schema-bound feature storage for learning workflows."""

from .client import FeatureStoreClient, FeatureStoreClientError
from .store import FeatureError, FeatureStore

__all__ = ["FeatureStoreClient", "FeatureStoreClientError", "FeatureError", "FeatureStore"]
