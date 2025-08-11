"""GPT-OSS specific plugins for Entity Framework.

DEPRECATED: This module location is deprecated. GPT-OSS plugins have been moved
to the 'entity-plugin-gpt-oss' package as part of the Entity Framework modularization.

Please install and import from the new package:
    pip install entity-plugin-gpt-oss
    from entity_plugin_gpt_oss import ReasoningTracePlugin

This compatibility layer will be removed in entity-core 0.1.0.
"""

import warnings

# Issue deprecation warning when this module is imported
warnings.warn(
    "The 'entity.plugins.gpt_oss' module is deprecated. "
    "GPT-OSS plugins have been moved to 'entity-plugin-gpt-oss' package. "
    "Please install 'entity-plugin-gpt-oss' and update your imports. "
    "This compatibility layer will be removed in entity-core 0.1.0.",
    DeprecationWarning,
    stacklevel=2,
)

# Import compatibility shims (plugins and helper classes)
from ..gpt_oss_compat import (
    AdaptiveReasoningPlugin,
    DeveloperOverridePlugin,
    FunctionSchemaRegistryPlugin,
    GPTOSSToolOrchestrator,
    HarmonySafetyFilterPlugin,
    MultiChannelAggregatorPlugin,
    ReasoningAnalyticsDashboardPlugin,
    ReasoningTracePlugin,
    StructuredOutputPlugin,
)


# Use module-level __getattr__ for dynamic imports of helper classes
def __getattr__(name):
    """Forward attribute access to gpt_oss_compat module for helper classes."""
    from ..gpt_oss_compat import __getattr__ as compat_getattr

    return compat_getattr(name)


__all__ = [
    "ReasoningTracePlugin",
    "StructuredOutputPlugin",
    "DeveloperOverridePlugin",
    "AdaptiveReasoningPlugin",
    "GPTOSSToolOrchestrator",
    "MultiChannelAggregatorPlugin",
    "HarmonySafetyFilterPlugin",
    "FunctionSchemaRegistryPlugin",
    "ReasoningAnalyticsDashboardPlugin",
]
