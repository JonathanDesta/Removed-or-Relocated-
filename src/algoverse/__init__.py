"""Evaluation harness and analysis code for "Removed or Recoverable? Testing
Layer-Local Suppression of Deceptive Behavior Under Retraining".

The scoring and analysis modules (tasks, metrics, sweep, relocation,
recovery_report, figures) import only the standard library, so they run on
a machine with no ML stack. Generation and training (models, eval, train,
sweepdriver, interp) import torch and friends lazily or at their own import.
"""
