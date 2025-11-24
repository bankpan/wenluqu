from __future__ import annotations

import argparse
import sys
from pathlib import Path

from steady_score.config import OCRConfig, PAVAConfig
from steady_score.logging_utils import setup_file_logger
from steady_score.ocr_pipeline import OCRProcessor
from steady_score.pava import compute_steady_score, load_buckets
from steady_score.reporting import write_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="考研稳录取分数线自动化工具")
    subparsers = parser.add_subparsers(dest="command", required=True)

    process_parser = subparsers.add_parser("process-images", help="批量识别图片并导出CSV")
    process_parser.add_argument("image_dir", type=Path, help="待处理图片目录")
    process_parser.add_argument("output_dir", type=Path, help="输出目录")
    process_parser.add_argument("--lang", default="ch", help="PaddleOCR 语言 (默认: ch)")

    batch_parser = subparsers.add_parser("batch-calculate", help="批量计算稳录取分数")
    batch_parser.add_argument("csv_dir", type=Path, help="OCR生成的CSV目录")
    batch_parser.add_argument("output_excel", type=Path, help="最终报告输出路径")
    batch_parser.add_argument("--threshold", type=float, default=0.75, help="稳录取阈值T (默认0.75)")
    batch_parser.add_argument("--min-bin", type=int, default=8, help="最小样本量约束 (默认8)")

    return parser.parse_args()


def process_images(args: argparse.Namespace) -> int:
    output_dir = args.output_dir
    if not args.image_dir.exists():
        raise FileNotFoundError(f"图片目录不存在: {args.image_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = OCRConfig(
        image_dir=args.image_dir,
        output_dir=output_dir,
        raw_dir=output_dir / "raw",
        log_dir=output_dir / "logs",
        error_file=output_dir / "errors.txt",
        lang=args.lang,
    )
    logger = setup_file_logger(config.log_dir)
    logger.info("开始处理目录: %s", config.image_dir)
    processor = OCRProcessor(config, logger)
    results = processor.process_directory()

    error_lines: list[str] = []
    for result in results:
        if result.errors:
            for err in result.errors:
                error_lines.append(f"{result.image_path.name}\t{err}")

    config.error_file.parent.mkdir(parents=True, exist_ok=True)
    with config.error_file.open("w", encoding="utf-8") as fh:
        if error_lines:
            fh.write("\n".join(error_lines))
        else:
            fh.write("所有图片均识别成功\n")

    total = len(results)
    success = sum(1 for r in results if r.status == "success")
    print(f"处理完成: {success}/{total} 张图片")
    print(f"CSV输出目录: {config.raw_dir}")
    print(f"错误清单: {config.error_file}")
    return 0 if success else 1


def batch_calculate(args: argparse.Namespace) -> int:
    csv_dir: Path = args.csv_dir
    if not csv_dir.exists():
        raise FileNotFoundError(f"CSV目录不存在: {csv_dir}")
    csv_files = sorted(csv_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"在 {csv_dir} 未找到CSV文件")

    config = PAVAConfig(target_ratio=args.threshold, min_bin_size=args.min_bin, allow_fallback=True)
    results = []
    skipped: list[str] = []
    for csv_file in csv_files:
        school = csv_file.stem
        try:
            buckets = load_buckets(csv_file)
            result = compute_steady_score(school, buckets, config)
            results.append(result)
            if result.warnings:
                print(f"[警告] {school}: {'; '.join(result.warnings)}")
        except Exception as exc:  # noqa: BLE001
            skipped.append(f"{school}: {exc}")

    if skipped:
        print("以下院校计算失败:")
        for item in skipped:
            print(f"  - {item}")

    if not results:
        raise RuntimeError("没有成功的院校可以生成报告")

    args.output_excel.parent.mkdir(parents=True, exist_ok=True)
    write_report(results, args.output_excel)
    print(f"报告已生成: {args.output_excel}")
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "process-images":
        return process_images(args)
    if args.command == "batch-calculate":
        return batch_calculate(args)
    raise ValueError("未知命令")


if __name__ == "__main__":
    sys.exit(main())
