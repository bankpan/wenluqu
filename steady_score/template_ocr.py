"""模板定位法OCR识别 - 专门针对考研院校截图的固定格式优化"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from paddleocr import PaddleOCR

from .validators import ScoreRow


class ValidationError(Exception):
    """数据验证失败异常 - 数据不可信"""
    pass


@dataclass
class TableHeader:
    """表头信息"""
    col1_x_range: tuple[float, float]  # 分数段列的X坐标范围
    col2_x_range: tuple[float, float]  # 复试人数列的X坐标范围
    col3_x_range: tuple[float, float]  # 录取人数列的X坐标范围
    header_y: float  # 表头Y坐标
    image_width: float  # 图片宽度


@dataclass
class DataRegion:
    """数据区域"""
    top_y: float  # 数据区域上边界
    bottom_y: float  # 数据区域下边界
    height: float  # 区域高度

    def __post_init__(self):
        self.height = self.bottom_y - self.top_y


@dataclass
class RowRegion:
    """单行区域"""
    y_start: float
    y_end: float
    y_center: float


class TemplateOCRExtractor:
    """模板定位法OCR提取器"""

    # 表头关键词
    HEADER_KEYWORDS = ["分数段", "复试人数", "录取人数"]

    # 底部导航栏元素（用于定位数据区域下边界）
    NAV_BAR_MARKERS = ["收藏", "院校PK", "本校话题", "话题"]

    # 分数段正则
    SCORE_PATTERN = re.compile(r'(\d{3})\s*分?[-~～]\s*(\d{3})\s*分?')
    NUMBER_PATTERN = re.compile(r'^\d+$')

    def __init__(self, lang: str = "ch", logger: logging.Logger | None = None):
        """初始化模板OCR提取器

        Args:
            lang: OCR语言
            logger: 日志记录器
        """
        self.lang = lang
        self.logger = logger or logging.getLogger(__name__)
        self._ocr_engine: PaddleOCR | None = None

    @property
    def ocr(self) -> PaddleOCR:
        """懒加载OCR引擎"""
        if self._ocr_engine is None:
            self.logger.info(f"初始化PaddleOCR引擎 (lang={self.lang})")
            self._ocr_engine = PaddleOCR(lang=self.lang, use_textline_orientation=True)
        return self._ocr_engine

    def extract_table_from_image(self, image_path: Path) -> list[ScoreRow]:
        """从图片提取表格数据（主入口）

        Args:
            image_path: 图片路径

        Returns:
            提取的分数行列表

        Raises:
            ValidationError: 数据验证失败
        """
        self.logger.info(f"开始处理图片: {image_path.name}")

        # 1. 读取图片
        try:
            img = Image.open(image_path).convert('RGB')
            img_array = np.array(img)
        except Exception as exc:
            raise ValidationError(f"无法打开图片: {exc}") from exc

        # 2. 运行OCR识别
        self.logger.info("运行OCR识别...")
        try:
            result = self.ocr.ocr(img_array)
        except Exception as exc:
            raise ValidationError(f"OCR识别失败: {exc}") from exc

        if not result or not result[0]:
            raise ValidationError("OCR未返回任何结果")

        # 3. 解析OCR结果
        ocr_boxes = self._parse_ocr_result(result[0])
        self.logger.info(f"识别到 {len(ocr_boxes)} 个文本框")

        if len(ocr_boxes) < 10:
            raise ValidationError(f"识别到的文本框过少({len(ocr_boxes)}个)，可能截图不完整")

        # ============ 检查点1：表头验证 ============
        self.logger.info("检查点1: 验证表头...")
        header = self._find_table_header(ocr_boxes, img.width)
        if not header:
            raise ValidationError("未找到表头（分数段/复试人数/录取人数），可能不是正确的截图")

        self.logger.info(
            f"表头定位成功: Y={header.header_y:.0f}, "
            f"列1=[{header.col1_x_range[0]:.0f}-{header.col1_x_range[1]:.0f}], "
            f"列2=[{header.col2_x_range[0]:.0f}-{header.col2_x_range[1]:.0f}], "
            f"列3=[{header.col3_x_range[0]:.0f}-{header.col3_x_range[1]:.0f}]"
        )

        # ============ 检查点2：数据区域验证 ============
        self.logger.info("检查点2: 定位数据区域...")
        data_region = self._locate_data_region(ocr_boxes, header, img.height)
        if data_region.height < 50:
            raise ValidationError(
                f"数据区域过小(高度={data_region.height:.0f}px)，可能截图不完整"
            )

        self.logger.info(
            f"数据区域: Y=[{data_region.top_y:.0f}-{data_region.bottom_y:.0f}], "
            f"高度={data_region.height:.0f}px"
        )

        # ============ 检查点3：行数验证 ============
        self.logger.info("检查点3: 分割数据行...")
        rows = self._split_into_rows(img_array, data_region, header)

        if len(rows) < 3:
            raise ValidationError(f"识别到的行数过少({len(rows)}行)，可能识别失败")

        if len(rows) > 30:
            raise ValidationError(f"识别到的行数过多({len(rows)}行)，可能分割错误")

        self.logger.info(f"分割为 {len(rows)} 行")

        # ============ 检查点4：逐行数据提取和验证 ============
        self.logger.info("检查点4: 提取单元格数据...")
        score_rows = []
        failed_rows = []

        for i, row_region in enumerate(rows):
            try:
                score_row = self._extract_row_data(
                    img_array, row_region, header, ocr_boxes
                )
                # 单行验证
                self._validate_single_row(score_row, i + 1)
                score_rows.append(score_row)

            except Exception as e:
                failed_rows.append((i + 1, str(e)))
                self.logger.warning(f"行 {i+1} 提取失败: {e}")

        # ============ 检查点5：整体数据验证 ============
        self.logger.info("检查点5: 整体数据验证...")

        if not score_rows:
            raise ValidationError("所有行识别失败，无有效数据")

        failure_rate = len(failed_rows) / len(rows)
        if failure_rate > 0.2:
            raise ValidationError(
                f"识别失败率过高({failure_rate:.0%})，数据不可信。"
                f"失败行: {failed_rows[:5]}..."  # 只显示前5个
            )

        # 验证分数段递增
        if not self._is_score_increasing(score_rows):
            raise ValidationError("分数段未递增，数据异常")

        # 验证录取率合理性
        if not self._is_admission_ratio_valid(score_rows):
            raise ValidationError("录取率异常（录取>复试或全为0），数据异常")

        # ============ 检查点6：尝试修复失败行 ============
        if failed_rows:
            self.logger.info(f"检查点6: 尝试修复 {len(failed_rows)} 个失败行...")
            if failure_rate <= 0.1:  # 只修复少量失败行
                try:
                    score_rows = self._repair_missing_rows(score_rows, rows, failed_rows)
                    self.logger.info(f"成功修复失败行，最终行数: {len(score_rows)}")
                except Exception as e:
                    self.logger.warning(f"修复失败: {e}")
                    # 不抛出异常，使用未修复的数据
            else:
                self.logger.warning(f"失败行过多({len(failed_rows)}行)，放弃修复")

        self.logger.info(f"✓ 提取成功: {len(score_rows)} 行有效数据")
        return score_rows

    def _parse_ocr_result(self, ocr_result: list | dict) -> list[dict[str, Any]]:
        """解析OCR结果，提取文本和坐标

        Args:
            ocr_result: PaddleOCR返回的结果

        Returns:
            包含text, x, y, bbox的字典列表
        """
        boxes = []

        # 检查是否是新版API的字典格式
        if isinstance(ocr_result, dict):
            texts = ocr_result.get('rec_texts', [])
            polys = ocr_result.get('rec_polys', [])
            scores = ocr_result.get('rec_scores', [])

            for i, (text, poly) in enumerate(zip(texts, polys)):
                try:
                    x_coords = poly[:, 0]
                    y_coords = poly[:, 1]
                    x_center = float(np.mean(x_coords))
                    y_center = float(np.mean(y_coords))

                    boxes.append({
                        'text': text.strip(),
                        'x': x_center,
                        'y': y_center,
                        'bbox': poly,
                        'conf': scores[i] if i < len(scores) else 1.0
                    })
                except Exception as exc:
                    self.logger.warning(f"处理文本框失败: {exc}")
                    continue
        else:
            # 旧版格式: list of [[bbox], (text, conf)]
            for item in ocr_result:
                try:
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
                        'bbox': np.array(bbox),
                        'conf': conf
                    })
                except Exception as exc:
                    self.logger.warning(f"解析OCR结果项失败: {exc}")
                    continue

        return boxes

    def _find_table_header(self, ocr_boxes: list[dict], image_width: float) -> TableHeader | None:
        """识别表头，确定列边界

        Args:
            ocr_boxes: OCR识别的文本框列表
            image_width: 图片宽度

        Returns:
            TableHeader对象，如果未找到返回None
        """
        # 查找表头关键词的位置
        header_positions = {}

        for box in ocr_boxes:
            text = box['text']
            for keyword in self.HEADER_KEYWORDS:
                if keyword in text:
                    header_positions[keyword] = {
                        'x': box['x'],
                        'y': box['y'],
                        'text': text
                    }
                    break

        # 验证是否找到所有3个表头
        if len(header_positions) != 3:
            missing = set(self.HEADER_KEYWORDS) - set(header_positions.keys())
            self.logger.error(f"表头缺失: {missing}")
            return None

        # 获取表头Y坐标（取平均值，应该在同一行）
        y_coords = [pos['y'] for pos in header_positions.values()]
        header_y = sum(y_coords) / len(y_coords)

        # 根据X坐标确定列边界
        col1_x = header_positions["分数段"]['x']
        col2_x = header_positions["复试人数"]['x']
        col3_x = header_positions["录取人数"]['x']

        # 计算列的X坐标范围（列1从0开始，列3到图片宽度结束）
        # 列之间的分界线取中点
        mid_12 = (col1_x + col2_x) / 2
        mid_23 = (col2_x + col3_x) / 2

        return TableHeader(
            col1_x_range=(0, mid_12),
            col2_x_range=(mid_12, mid_23),
            col3_x_range=(mid_23, image_width),
            header_y=header_y,
            image_width=image_width
        )

    def _locate_data_region(
        self,
        ocr_boxes: list[dict],
        header: TableHeader,
        image_height: float
    ) -> DataRegion:
        """定位数据区域

        Args:
            ocr_boxes: OCR识别的文本框列表
            header: 表头信息
            image_height: 图片高度

        Returns:
            DataRegion对象
        """
        # 数据区域上边界：表头下方一定距离
        top_y = header.header_y + 40  # 表头下方40像素

        # 数据区域下边界：使用底部导航栏元素（优先级从高到低）
        # 注意：不再使用"一对一择校"广告，因为它会与数据行重叠
        bottom_y = None

        # 优先级1: 查找"收藏"按钮（底部导航栏最左侧，从不重叠）
        for box in ocr_boxes:
            if "收藏" in box['text']:
                # 获取bbox顶部Y坐标，减去安全间距
                if 'bbox' in box:
                    bbox = box['bbox']
                    if isinstance(bbox, np.ndarray):
                        y_coords = bbox[:, 1] if bbox.ndim == 2 else [p[1] for p in bbox]
                        min_y = float(np.min(y_coords))
                        bottom_y = min_y - 30  # 安全间距30px
                        self.logger.info(f"使用'收藏'按钮定位底部边界: Y={bottom_y:.0f}")
                        break

        # 优先级2: 查找"院校PK"或"PK"按钮
        if bottom_y is None:
            for box in ocr_boxes:
                text = box['text']
                if "院校PK" in text or (text == "PK" and box['y'] > image_height * 0.8):
                    if 'bbox' in box:
                        bbox = box['bbox']
                        if isinstance(bbox, np.ndarray):
                            y_coords = bbox[:, 1] if bbox.ndim == 2 else [p[1] for p in bbox]
                            min_y = float(np.min(y_coords))
                            bottom_y = min_y - 30
                            self.logger.info(f"使用'PK'按钮定位底部边界: Y={bottom_y:.0f}")
                            break

        # 优先级3: 查找"本校话题"按钮（绿色按钮）
        if bottom_y is None:
            for box in ocr_boxes:
                if "本校话题" in box['text'] or (box['text'] == "话题" and box['y'] > image_height * 0.8):
                    # 这个按钮在数据下方，不会重叠
                    if 'bbox' in box:
                        bbox = box['bbox']
                        if isinstance(bbox, np.ndarray):
                            y_coords = bbox[:, 1] if bbox.ndim == 2 else [p[1] for p in bbox]
                            min_y = float(np.min(y_coords))
                            bottom_y = min_y - 20
                            self.logger.info(f"使用'本校话题'按钮定位底部边界: Y={bottom_y:.0f}")
                            break

        # 优先级4: 使用固定偏移（典型底部导航栏高度）
        if bottom_y is None:
            bottom_y = image_height - 120
            self.logger.warning(f"未找到底部导航栏标记，使用固定偏移: Y={bottom_y:.0f}")

        # 验证数据区域高度
        if bottom_y - top_y < 200:
            self.logger.warning(
                f"数据区域过小(top={top_y:.0f}, bottom={bottom_y:.0f}, height={bottom_y-top_y:.0f})，"
                f"使用默认值"
            )
            bottom_y = image_height - 120

        # 确保下边界不超过图片高度
        bottom_y = min(bottom_y, image_height - 50)

        return DataRegion(top_y=top_y, bottom_y=bottom_y, height=0)

    def _split_into_rows(
        self,
        image: np.ndarray,
        data_region: DataRegion,
        header: TableHeader
    ) -> list[RowRegion]:
        """分割数据行

        Args:
            image: 图片数组
            data_region: 数据区域
            header: 表头信息

        Returns:
            RowRegion列表
        """
        # 裁剪出数据区域
        top = int(data_region.top_y)
        bottom = int(data_region.bottom_y)
        data_image = image[top:bottom, :]

        # 转换为灰度图
        if len(data_image.shape) == 3:
            gray = np.mean(data_image, axis=2).astype(np.uint8)
        else:
            gray = data_image

        # 检测水平网格线
        horizontal_lines = self._detect_horizontal_lines(gray)

        if len(horizontal_lines) >= 3:
            # 使用检测到的网格线分割
            self.logger.info(f"检测到 {len(horizontal_lines)} 条水平网格线")
            rows = []
            for i in range(len(horizontal_lines) - 1):
                y_start = horizontal_lines[i] + top
                y_end = horizontal_lines[i + 1] + top
                y_center = (y_start + y_end) / 2
                rows.append(RowRegion(y_start, y_end, y_center))
        else:
            # 使用固定行高分割
            self.logger.info("网格线检测失败，使用固定行高分割")
            row_height = 60  # 估算的行高
            rows = []
            y = data_region.top_y
            while y + row_height <= data_region.bottom_y:
                rows.append(RowRegion(y, y + row_height, y + row_height / 2))
                y += row_height

        return rows

    def _detect_horizontal_lines(self, gray_image: np.ndarray) -> list[float]:
        """检测水平网格线

        Args:
            gray_image: 灰度图像

        Returns:
            水平线的Y坐标列表
        """
        # 计算每行的平均亮度
        row_brightness = np.mean(gray_image, axis=1)

        # 寻找亮度低的行（网格线）
        threshold = np.percentile(row_brightness, 30)  # 亮度最低的30%

        # 找到所有低于阈值的行
        dark_rows = np.where(row_brightness < threshold)[0]

        if len(dark_rows) == 0:
            return []

        # 合并相邻的暗行（网格线有宽度）
        lines = []
        current_line_start = dark_rows[0]

        for i in range(1, len(dark_rows)):
            if dark_rows[i] - dark_rows[i-1] > 5:  # 间隔超过5像素，认为是新线
                line_center = (current_line_start + dark_rows[i-1]) / 2
                lines.append(line_center)
                current_line_start = dark_rows[i]

        # 添加最后一条线
        line_center = (current_line_start + dark_rows[-1]) / 2
        lines.append(line_center)

        return lines

    def _extract_row_data(
        self,
        image: np.ndarray,
        row_region: RowRegion,
        header: TableHeader,
        ocr_boxes: list[dict]
    ) -> ScoreRow:
        """提取单行数据

        Args:
            image: 图片数组
            row_region: 行区域
            header: 表头信息
            ocr_boxes: 全图OCR结果（用于查找该行的文本）

        Returns:
            ScoreRow对象
        """
        # 找到该行范围内的所有文本框
        row_boxes = [
            box for box in ocr_boxes
            if row_region.y_start <= box['y'] <= row_region.y_end
        ]

        # 按列分组
        col1_texts = [
            box['text'] for box in row_boxes
            if header.col1_x_range[0] <= box['x'] < header.col1_x_range[1]
        ]
        col2_texts = [
            box['text'] for box in row_boxes
            if header.col2_x_range[0] <= box['x'] < header.col2_x_range[1]
        ]
        col3_texts = [
            box['text'] for box in row_boxes
            if header.col3_x_range[0] <= box['x'] < header.col3_x_range[1]
        ]

        # 提取分数段（第1列）
        score_range, score_lower, score_upper = self._extract_score_range(col1_texts)

        # 提取复试人数（第2列）
        candidates = self._extract_number(col2_texts, "复试人数")

        # 提取录取人数（第3列）
        admitted = self._extract_number(col3_texts, "录取人数")

        return ScoreRow(
            score_range=score_range,
            lower=score_lower,
            upper=score_upper,
            candidates=candidates,
            admitted=admitted
        )

    def _extract_score_range(self, texts: list[str]) -> tuple[str, int, int]:
        """从文本列表中提取分数段

        Args:
            texts: 文本列表

        Returns:
            (score_range字符串, lower, upper)

        Raises:
            ValueError: 无法提取分数段
        """
        # 合并所有文本
        combined_text = ' '.join(texts)

        # 正则匹配
        match = self.SCORE_PATTERN.search(combined_text)
        if match:
            lower = int(match.group(1))
            upper = int(match.group(2))
            return f"{lower}-{upper}", lower, upper

        # 尝试分步匹配（如"206分" + "-210"）
        numbers = []
        for text in texts:
            # 提取所有3位数字
            nums = re.findall(r'\d{3}', text)
            numbers.extend([int(n) for n in nums])

        if len(numbers) >= 2:
            lower = numbers[0]
            upper = numbers[1]
            return f"{lower}-{upper}", lower, upper

        raise ValueError(f"无法提取分数段: {texts}")

    def _extract_number(self, texts: list[str], field_name: str = "") -> int:
        """从文本列表中提取数字

        Args:
            texts: 文本列表
            field_name: 字段名称（用于错误提示）

        Returns:
            提取的整数

        Raises:
            ValueError: 无法提取数字
        """
        for text in texts:
            # 查找纯数字
            if self.NUMBER_PATTERN.match(text.strip()):
                return int(text.strip())

            # 提取文本中的数字
            nums = re.findall(r'\d+', text)
            if nums:
                return int(nums[0])

        raise ValueError(f"无法提取{field_name}: {texts}")

    def _validate_single_row(self, row: ScoreRow, row_num: int):
        """验证单行数据

        Args:
            row: 分数行
            row_num: 行号

        Raises:
            ValueError: 验证失败
        """
        # 分数段范围验证
        if row.lower >= row.upper:
            raise ValueError(f"分数段下限≥上限: {row.score_range}")

        if not (150 <= row.lower <= 500 and 150 <= row.upper <= 500):
            raise ValueError(f"分数超出合理范围(150-500): {row.score_range}")

        # 人数验证
        if row.candidates < 0 or row.admitted < 0:
            raise ValueError(f"人数为负数: candidates={row.candidates}, admitted={row.admitted}")

        if row.admitted > row.candidates:
            raise ValueError(f"录取人数({row.admitted})>复试人数({row.candidates})")

        if row.candidates > 1000:
            raise ValueError(f"复试人数异常大: {row.candidates}")

    def _is_score_increasing(self, rows: list[ScoreRow]) -> bool:
        """验证分数段递增"""
        for i in range(len(rows) - 1):
            if rows[i].upper > rows[i+1].lower:
                self.logger.error(
                    f"分数段重叠: {rows[i].score_range} vs {rows[i+1].score_range}"
                )
                return False
        return True

    def _is_admission_ratio_valid(self, rows: list[ScoreRow]) -> bool:
        """验证录取率合理性"""
        total_candidates = sum(r.candidates for r in rows)
        total_admitted = sum(r.admitted for r in rows)

        if total_candidates == 0:
            return False

        if total_admitted == 0:
            # 全为0也可能是真实情况（某些年份无人录取）
            # 但至少要有复试人数
            return total_candidates > 0

        ratio = total_admitted / total_candidates
        if ratio > 1.0 or ratio < 0.001:  # 录取率<0.1%可能异常
            self.logger.error(f"录取率异常: {ratio:.2%}")
            return False

        return True

    def _repair_missing_rows(
        self,
        score_rows: list[ScoreRow],
        all_rows: list[RowRegion],
        failed_rows: list[tuple[int, str]]
    ) -> list[ScoreRow]:
        """尝试修复缺失的行

        Args:
            score_rows: 已成功提取的行
            all_rows: 所有行区域
            failed_rows: 失败的行列表

        Returns:
            修复后的行列表
        """
        # 简单策略：根据分数段推断缺失行
        # 更复杂的修复逻辑可以后续添加

        self.logger.info("当前暂不支持自动修复，返回原数据")
        return score_rows
