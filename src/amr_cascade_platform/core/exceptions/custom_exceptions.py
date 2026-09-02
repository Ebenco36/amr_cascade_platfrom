"""Custom exceptions for the AMR Cascade Platform."""


class AmrCascadeError(Exception):
    """Base application exception."""


class ConfigurationError(AmrCascadeError):
    """Raised when configuration is missing or invalid."""


class DataDiscoveryError(AmrCascadeError):
    """Raised when required datasets cannot be discovered."""


class ValidationError(AmrCascadeError):
    """Raised when runtime validation fails."""
