from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from .config import PAVAConfig
from .metadata import SchoolMetadata

Z_95 = 1.959964


@dataclass(slots=True)
class Bucket:
    score_range: str
    lower: int
    upper: int
    center: float
    candidates: int
    admitted: int
    ratio: float
    wilson_low: float
    wilson_high: float


@dataclass(slots=True)
class SteadyScoreResult:
    school: str
    metadata: SchoolMetadata
    buckets_raw: list[Bucket]
    buckets_smoothed: list[Bucket]
    steady_score: float
    steady_bucket_low: Bucket
    steady_bucket_high: Bucket
    warnings: list[str]


@dataclass(slots=True)
class BucketLoadResult:
    metadata: SchoolMetadata
    buckets: list[Bucket]


def load_buckets(csv_path: Path) -> BucketLoadResult:
    df = pd.read_csv(csv_path)
    required_columns = {"score_range", "lower", "upper", "candidates", "admitted"}
    meta_columns = {"school", "college", "major", "code", "study_mode", "province"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} 缺少列: {missing}")
    missing_meta = meta_columns - set(df.columns)
    if missing_meta:
        raise ValueError(f"{csv_path} 缺少元数据列: {missing_meta}")

    first_row = df.iloc[0].to_dict()
    metadata = SchoolMetadata.from_row(first_row)
    if not all([metadata.school, metadata.college, metadata.major, metadata.code, metadata.study_mode, metadata.province]):
        raise ValueError(f"{csv_path} 元数据字段不完整")
    if not re.fullmatch(r"\d{6}", metadata.code):
        raise ValueError(f"{csv_path} 专业代码格式无效: {metadata.code!r}")
    if metadata.study_mode not in {"全日制", "非全日制"}:
        raise ValueError(f"{csv_path} 学习方式无效: {metadata.study_mode!r}")
    if not re.fullmatch(r".+[省市]$|香港|澳门|台湾", metadata.province):
        raise ValueError(f"{csv_path} 省份格式无效: {metadata.province!r}")

    buckets: list[Bucket] = []
    for row in df.to_dict(orient="records"):
        candidates = int(row["candidates"])
        admitted = int(row["admitted"])
        if candidates <= 0:
            continue
        ratio = admitted / candidates
        lower = int(row["lower"])
        upper = int(row["upper"])
        bucket = Bucket(
            score_range=str(row["score_range"]),
            lower=lower,
            upper=upper,
            center=(lower + upper) / 2,
            candidates=candidates,
            admitted=admitted,
            ratio=ratio,
            wilson_low=_wilson_interval(admitted, candidates)[0],
            wilson_high=_wilson_interval(admitted, candidates)[1],
        )
        buckets.append(bucket)
    return BucketLoadResult(metadata=metadata, buckets=buckets)


def compute_steady_score(
    school: str | SchoolMetadata,
    buckets: Sequence[Bucket],
    config: PAVAConfig,
) -> SteadyScoreResult:
    metadata = school if isinstance(school, SchoolMetadata) else SchoolMetadata(
        school=str(school),
        college="",
        major="",
        code="",
        study_mode="",
    )
    buckets_sorted = sorted(buckets, key=lambda b: b.center)
    smoothed = _apply_pava(buckets_sorted)
    steady_score, lo_bucket, hi_bucket = _interpolate_threshold(smoothed, config.target_ratio)

    warnings: list[str] = []
    if any(bucket.candidates < config.min_bin_size for bucket in buckets_sorted):
        warnings.append(f"存在复试人数 < {config.min_bin_size} 的分段，建议人工复核")

    if steady_score is None and config.allow_fallback:
        if smoothed:
            steady_score = smoothed[-1].upper
            lo_bucket = hi_bucket = smoothed[-1]
            warnings.append("未找到满足阈值的区间，退化为最高分段上限")

    if steady_score is None:
        raise ValueError("无法计算稳录取分数，请检查数据完成度")

    return SteadyScoreResult(
        school=metadata.school,
        metadata=metadata,
        buckets_raw=list(buckets_sorted),
        buckets_smoothed=smoothed,
        steady_score=round(steady_score, config.interpolation_digits),
        steady_bucket_low=lo_bucket,
        steady_bucket_high=hi_bucket,
        warnings=warnings,
    )


def _apply_pava(buckets: Sequence[Bucket]) -> list[Bucket]:
    stack: list[Bucket] = []
    for bucket in buckets:
        stack.append(bucket)
        while len(stack) >= 2 and stack[-2].ratio > stack[-1].ratio:
            merged = _merge_buckets(stack[-2], stack[-1])
            stack.pop()
            stack[-1] = merged
    return stack


def _merge_buckets(left: Bucket, right: Bucket) -> Bucket:
    total_candidates = left.candidates + right.candidates
    total_admitted = left.admitted + right.admitted
    center = (left.center * left.candidates + right.center * right.candidates) / total_candidates
    ratio = total_admitted / total_candidates
    score_range = f"{min(left.lower, right.lower)}-{max(left.upper, right.upper)}"
    wilson_low, wilson_high = _wilson_interval(total_admitted, total_candidates)
    return Bucket(
        score_range=score_range,
        lower=min(left.lower, right.lower),
        upper=max(left.upper, right.upper),
        center=center,
        candidates=total_candidates,
        admitted=total_admitted,
        ratio=ratio,
        wilson_low=wilson_low,
        wilson_high=wilson_high,
    )


def _interpolate_threshold(buckets: Sequence[Bucket], target: float):
    if not buckets:
        return None, None, None
    if buckets[0].ratio >= target:
        return buckets[0].lower, buckets[0], buckets[0]
    for left, right in zip(buckets, buckets[1:]):
        if left.ratio <= target <= right.ratio:
            if right.ratio == left.ratio:
                return right.center, left, right
            t = (target - left.ratio) / (right.ratio - left.ratio)
            score = left.center + t * (right.center - left.center)
            return score, left, right
    return None, buckets[-1], buckets[-1]


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    z = Z_95
    p_hat = successes / total
    denominator = 1 + z * z / total
    center = p_hat + z * z / (2 * total)
    margin = z * ((p_hat * (1 - p_hat) + (z * z) / (4 * total)) / total) ** 0.5
    lower = (center - margin) / denominator
    upper = (center + margin) / denominator
    return max(lower, 0.0), min(upper, 1.0)
