"""Entity Plugin GPT-OSS - GPT-OSS specific plugins for Entity Framework.

This package provides GPT-OSS specific plugin implementations for advanced
AI capabilities and reasoning control.
"""

from .adaptive_reasoning import AdaptiveReasoningPlugin
from .developer_override import DeveloperOverridePlugin
from .function_schema_registry import FunctionSchemaRegistryPlugin
from .harmony_safety_filter import HarmonySafetyFilterPlugin
from .multi_channel_aggregator import MultiChannelAggregatorPlugin
from .native_tools import GPTOSSToolOrchestrator
from .reasoning_analytics_dashboard import ReasoningAnalyticsDashboardPlugin
from .reasoning_trace import ReasoningTracePlugin
from .structured_output import StructuredOutputPlugin

__all__ = [
    "AdaptiveReasoningPlugin",
    "DeveloperOverridePlugin",
    "FunctionSchemaRegistryPlugin",
    "HarmonySafetyFilterPlugin",
    "MultiChannelAggregatorPlugin",
    "GPTOSSToolOrchestrator",
    "ReasoningAnalyticsDashboardPlugin",
    "ReasoningTracePlugin",
    "StructuredOutputPlugin",
]

__version__ = "0.1.0"