"""
ErrorAnalysis module for MasRouter.

Provides step-level error detection, robustness tracking, and error-augmented
reward shaping for training MAS routing controllers with error signals.
"""

from MAR.ErrorAnalysis.error_taxonomy import (
    ERROR_TYPES, ERROR_TYPE_LIST, NUM_ERROR_TYPES,
    ERROR_CODE_TO_IDX, ERROR_IDX_TO_CODE,
)
from MAR.ErrorAnalysis.robustness_tracker import RobustnessTracker
from MAR.ErrorAnalysis.error_evaluator import LLMErrorEvaluator
from MAR.ErrorAnalysis.error_reward import ErrorRewardComputer
