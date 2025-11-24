"""测试简单OCR集成到ocr_pipeline"""
from pathlib import Path
import logging
from steady_score.config import OCRConfig
from steady_score.logging_utils import setup_file_logger
from steady_score.ocr_pipeline import OCRProcessor

def test_simple_ocr():
    """测试简单OCR方法"""
    # 设置路径
    image_path = Path("test_images/微信图片_20251112160618_717_2.png")
    output_dir = Path("test_output_simple")

    if not image_path.exists():
        print(f"错误: 测试图片不存在: {image_path}")
        return

    # 创建配置
    config = OCRConfig(
        image_dir=image_path.parent,
        output_dir=output_dir,
        raw_dir=output_dir / "raw",
        log_dir=output_dir / "logs",
        error_file=output_dir / "errors.txt",
        lang="ch"
    )

    # 设置日志
    logger = setup_file_logger(config.log_dir)
    logger.setLevel(logging.INFO)

    print(f"开始测试简单OCR方法")
    print(f"图片: {image_path}")
    print(f"输出目录: {output_dir}")
    print()

    try:
        # 创建处理器
        processor = OCRProcessor(config, logger)

        # 处理单张图片
        result = processor._process_single_image(image_path)

        # 输出结果
        print(f"处理状态: {result.status}")
        print(f"识别行数: {len(result.rows)}")

        if result.rows:
            print("\n提取的数据:")
            for i, row in enumerate(result.rows, 1):
                print(f"  {i}. {row.score_range}: 复试={row.candidates}, 录取={row.admitted}")

        if result.csv_path:
            print(f"\nCSV输出: {result.csv_path}")

        if result.errors:
            print(f"\n错误信息:")
            for err in result.errors:
                print(f"  - {err}")

        # 验证结果
        print("\n验证:")
        if len(result.rows) == 11:
            print("✓ 行数正确 (11行)")
        else:
            print(f"✗ 行数不正确，期望11行，实际{len(result.rows)}行")

        if result.status == "success":
            print("✓ 处理成功")
        else:
            print(f"✗ 处理失败: {result.status}")

    except Exception as exc:
        print(f"\n错误: {exc}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple_ocr()
