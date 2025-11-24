"""简单的OCR表格提取模块 - 使用基础OCR + 正则匹配"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from paddleocr import PaddleOCR

from .validators import ScoreRow


class SimpleOCRExtractor:
    """简单的OCR提取器 - 专门处理格式统一的表格截图"""

    # 分数段匹配：201分-205分 或 201-205 或 201分～205分
    SCORE_PATTERN = re.compile(r'(\d{3})\s*分?[-~～]\s*(\d{3})\s*分?')

    # 纯数字匹配（用于识别人数）
    NUMBER_PATTERN = re.compile(r'^\d+$')

    def __init__(self, lang: str = "ch", logger: logging.Logger | None = None):
        """初始化简单OCR提取器

        Args:
            lang: OCR语言，默认中文
            logger: 日志记录器
        """
        self.lang = lang
        self.logger = logger or logging.getLogger(__name__)
        self._ocr_engine: PaddleOCR | None = None

    @property
    def ocr(self) -> PaddleOCR:
        """懒加载OCR引擎"""
        if self._ocr_engine is None:
            self.logger.info(f"初始化基础OCR引擎 (lang={self.lang})")
            self._ocr_engine = PaddleOCR(lang=self.lang, use_angle_cls=True)
        return self._ocr_engine

    def extract_table_from_image(self, image_path: Path) -> list[ScoreRow]:
        """从图片提取表格数据

        Args:
            image_path: 图片路径

        Returns:
            提取的分数行列表

        Raises:
            ValueError: 图片无法打开或OCR失败
        """
        # 1. 读取图片
        try:
            img = Image.open(image_path).convert('RGB')
            img_array = np.array(img)
        except Exception as exc:
            raise ValueError(f"无法打开图片: {exc}") from exc

        # 2. 运行OCR识别
        self.logger.info(f"开始OCR识别: {image_path.name}")
        try:
            result = self.ocr.ocr(img_array)
        except Exception as exc:
            raise ValueError(f"OCR识别失败: {exc}") from exc

        if not result or not result[0]:
            raise ValueError("OCR未返回任何结果")

        # 3. 解析OCR结果 - 处理新版PaddleOCR的字典格式
        first_result = result[0]
        
        # 检查是否是新版API的字典格式
        if isinstance(first_result, dict):
            # 新版格式: {'rec_texts': [...], 'rec_polys': [...], 'rec_scores': [...]}
            texts = first_result.get('rec_texts', [])
            polys = first_result.get('rec_polys', [])
            scores = first_result.get('rec_scores', [])
            
            self.logger.info(f"识别到 {len(texts)} 个文本框（新版API）")
            
            # 转换为boxes格式
            ocr_boxes = []
            for i, (text, poly) in enumerate(zip(texts, polys)):
                try:
                    # 计算中心坐标
                    x_coords = poly[:, 0]
                    y_coords = poly[:, 1]
                    x_center = float(np.mean(x_coords))
                    y_center = float(np.mean(y_coords))
                    
                    conf = scores[i] if i < len(scores) else 1.0
                    
                    ocr_boxes.append({
                        'text': text.strip(),
                        'x': x_center,
                        'y': y_center,
                        'conf': conf
                    })
                except Exception as exc:
                    self.logger.warning(f"处理文本框失败: {exc}")
                    continue
        else:
            # 旧版格式: list of [[bbox], (text, conf)]
            ocr_boxes = self._parse_ocr_result(first_result)
            self.logger.info(f"识别到 {len(ocr_boxes)} 个文本框（旧版API）")

        # 4. 按Y坐标分组成行
        text_rows = self._group_into_rows(ocr_boxes)
        self.logger.info(f"分组为 {len(text_rows)} 行")

        # 5. 提取表格数据行
        score_rows = self._extract_score_rows(text_rows)
        self.logger.info(f"提取到 {len(score_rows)} 行有效数据")

        return score_rows

    def _parse_ocr_result(self, ocr_result: list[Any]) -> list[dict[str, Any]]:
        """解析OCR结果，提取文本和坐标

        Args:
            ocr_result: PaddleOCR返回的结果

        Returns:
            包含text, x, y的字典列表
        """
        boxes = []

        for item in ocr_result:
            try:
                # PaddleOCR格式: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]], (text, confidence)
                bbox = item[0]
                text_info = item[1]

                text = text_info[0] if isinstance(text_info, (list, tuple)) else str(text_info)
                conf = text_info[1] if isinstance(text_info, (list, tuple)) and len(text_info) > 1 else 1.0

                # 计算中心坐标
                x_coords = [p[0] for p in bbox]
                y_coords = [p[1] for p in bbox]
                x_center = sum(x_coords) / len(x_coords)
                y_center = sum(y_coords) / len(y_coords)

                boxes.append({
                    'text': text.strip(),
                    'x': x_center,
                    'y': y_center,
                    'conf': conf
                })
            except Exception as exc:
                self.logger.warning(f"解析OCR结果项失败: {exc}, item={item}")
                continue

        return boxes

    def _group_into_rows(self, boxes: list[dict[str, Any]], y_threshold: int = 40) -> list[list[dict[str, Any]]]:
        """按Y坐标将文本框分组成行

        Args:
            boxes: 文本框列表
            y_threshold: Y坐标阈值（像素），同一行的文本框Y坐标差异小于此值

        Returns:
            分组后的行列表，每行按X坐标排序
        """
        # 按Y坐标分组
        rows_dict: dict[float, list[dict[str, Any]]] = defaultdict(list)

        for box in boxes:
            # 查找最接近的Y坐标组
            matched_y = None
            min_diff = y_threshold

            for existing_y in rows_dict.keys():
                diff = abs(box['y'] - existing_y)
                if diff < min_diff:
                    matched_y = existing_y
                    min_diff = diff

            # 如果没找到匹配的，创建新组
            if matched_y is None:
                matched_y = box['y']

            rows_dict[matched_y].append(box)

        # 排序：先按Y坐标（行），再按X坐标（列）
        sorted_rows = sorted(rows_dict.items(), key=lambda x: x[0])

        result = []
        for y_pos, row_boxes in sorted_rows:
            sorted_boxes = sorted(row_boxes, key=lambda x: x['x'])
            result.append(sorted_boxes)

        return result

    def _extract_score_rows(self, text_rows: list[list[dict[str, Any]]]) -> list[ScoreRow]:
        """从文本行中提取分数数据行

        Args:
            text_rows: 文本行列表

        Returns:
            ScoreRow列表
        """
        score_rows = []

        for row_idx, row_boxes in enumerate(text_rows):
            # 合并同一行的文本，用空格分隔
            row_texts = [box['text'] for box in row_boxes]
            row_text = ' '.join(row_texts)

            # 查找分数段
            score_match = self.SCORE_PATTERN.search(row_text)
            if not score_match:
                # 没有分数段，跳过这行（可能是标题或其他内容）
                continue

            score_lower = int(score_match.group(1))
            score_upper = int(score_match.group(2))
            score_range = f"{score_lower}-{score_upper}"

            # 在分数段右侧查找数字（复试人数、录取人数）
            # 找到分数段在row_boxes中的位置
            score_x = None
            for box in row_boxes:
                if score_match.group(0) in box['text'] or \
                   f"{score_lower}" in box['text'] or \
                   f"{score_upper}" in box['text']:
                    score_x = box['x']
                    break

            # 收集分数段右侧的所有数字
            numbers = []
            for box in row_boxes:
                # 必须在分数段右侧
                if score_x and box['x'] <= score_x:
                    continue

                # 检查是否是纯数字
                text = box['text'].strip()
                if self.NUMBER_PATTERN.match(text):
                    try:
                        num = int(text)
                        numbers.append(num)
                    except ValueError:
                        continue

            # 根据数字个数判断列格式
            if len(numbers) >= 2:
                # 至少有2个数字：复试人数、录取人数
                candidates = numbers[0]
                admitted = numbers[1]

                # 基本验证
                if score_lower >= score_upper:
                    self.logger.warning(
                        f"行 {row_idx} 分数范围无效: {score_lower} >= {score_upper}"
                    )
                    continue

                if admitted > candidates:
                    self.logger.warning(
                        f"行 {row_idx} 录取人数({admitted})大于复试人数({candidates})"
                    )
                    continue

                # 创建ScoreRow对象
                score_row = ScoreRow(
                    score_range=score_range,
                    lower=score_lower,
                    upper=score_upper,
                    candidates=candidates,
                    admitted=admitted
                )
                score_rows.append(score_row)

                self.logger.debug(
                    f"提取行 {row_idx}: {score_range}, "
                    f"复试={candidates}, 录取={admitted}"
                )
            else:
                self.logger.debug(
                    f"行 {row_idx} 数字不足: {score_range}, "
                    f"numbers={numbers}, row_text='{row_text}'"
                )

        return score_rows
