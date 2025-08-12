"""Entity Plugin GPT-OSS - GPT-OSS specific plugins for Entity Framework.

This package provides GPT-OSS specific plugin implementations for advanced
AI capabilities and reasoning control.
"""

from .adaptive_reasoning import AdaptiveReasoningController
from .developer_override import DeveloperOverridePlugin
from .function_schema_registry import FunctionSchemaRegistryPlugin
from .harmony_safety_filter import HarmonySafetyFilterPlugin
from .multi_channel_aggregator import MultiChannelAggregatorPlugin
from .native_tools import NativeToolOrchestratorPlugin
from .reasoning_analytics_dashboard import ReasoningAnalyticsDashboardPlugin
from .reasoning_trace import ReasoningTracePlugin
from .structured_output import StructuredOutputValidatorPlugin

__all__ = [
    "AdaptiveReasoningController",
    "DeveloperOverridePlugin",
    "FunctionSchemaRegistryPlugin",
    "HarmonySafetyFilterPlugin",
    "MultiChannelAggregatorPlugin",
    "NativeToolOrchestratorPlugin",
    "ReasoningAnalyticsDashboardPlugin",
    "ReasoningTracePlugin",
    "StructuredOutputValidatorPlugin",
]

__version__ = "0.1.0"