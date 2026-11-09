"""Deterministic stratified dataset splitting."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_split(
    frame: pd.DataFrame, target: pd.Series, test_size: float, validation_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    train_frame, test_frame, train_target, test_target = train_test_split(
        frame, target, test_size=test_size, stratify=target, random_state=seed
    )
    relative_validation_size = validation_size / (1 - test_size)
    train_frame, validation_frame, train_target, validation_target = train_test_split(
        train_frame,
        train_target,
        test_size=relative_validation_size,
        stratify=train_target,
        random_state=seed,
    )
    return (
        train_frame,
        validation_frame,
        test_frame,
        train_target,
        validation_target,
        test_target,
    )
