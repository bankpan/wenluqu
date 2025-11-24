from __future__ import annotations

from pathlib import Path

import pandas as pd

from .pava import SteadyScoreResult


def build_summary_dataframe(results: list[SteadyScoreResult]) -> pd.DataFrame:
    records = []
    for result in results:
        warnings = "; ".join(result.warnings) if result.warnings else ""
        records.append(
            {
                "院校": result.metadata.school,
                "省份": result.metadata.province,
                "学院": result.metadata.college,
                "专业": result.metadata.major,
                "代码": result.metadata.code,
                "学习方式": result.metadata.study_mode,
                "稳录取分数": result.steady_score,
                "触发分段(低)": result.steady_bucket_low.score_range,
                "触发分段(高)": result.steady_bucket_high.score_range,
                "p_low": round(result.steady_bucket_low.ratio, 4),
                "p_high": round(result.steady_bucket_high.ratio, 4),
                "样本量_low": result.steady_bucket_low.candidates,
                "样本量_high": result.steady_bucket_high.candidates,
                "备注": warnings,
            }
        )
    return pd.DataFrame(records)


def build_detail_dataframe(results: list[SteadyScoreResult]) -> pd.DataFrame:
    detail_records = []
    for result in results:
        for bucket in result.buckets_smoothed:
            detail_records.append(
                {
                    "院校": result.metadata.school,
                    "省份": result.metadata.province,
                    "学院": result.metadata.college,
                    "专业": result.metadata.major,
                    "代码": result.metadata.code,
                    "学习方式": result.metadata.study_mode,
                    "分数段": bucket.score_range,
                    "中心分": bucket.center,
                    "复试人数": bucket.candidates,
                    "录取人数": bucket.admitted,
                    "p": round(bucket.ratio, 4),
                    "Wilson_low": round(bucket.wilson_low, 4),
                    "Wilson_high": round(bucket.wilson_high, 4),
                }
            )
    return pd.DataFrame(detail_records)


def write_report(results: list[SteadyScoreResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df = build_summary_dataframe(results)
    detail_df = build_detail_dataframe(results)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="汇总", index=False)
        detail_df.to_excel(writer, sheet_name="PAVA详情", index=False)
