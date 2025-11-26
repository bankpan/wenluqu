"""简单的OCR表格提取模块 - 使用基础OCR + 正则匹配"""
from __future__ import annotations

import logging
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from paddleocr import PaddleOCR

from .metadata import SchoolMetadata
from .validators import ScoreRow


@dataclass
class TableHeader:
    """表头列定位结果。"""

    col1_x_range: tuple[float, float]
    col2_x_range: tuple[float, float]
    col3_x_range: tuple[float, float]
    header_y: float


@dataclass
class DataRegion:
    """数据区域的上下边界。"""

    top_y: float
    bottom_y: float

    @property
    def height(self) -> float:
        return self.bottom_y - self.top_y


@dataclass
class RowRegion:
    """单行的上下边界。"""

    y_start: float
    y_end: float
    y_center: float


class SimpleOCRExtractor:
    """简单的OCR提取器 - 专门处理格式统一的表格截图"""

    # 分数段匹配：201分-205分 或 201-205 或 201分～205分
    SCORE_PATTERN = re.compile(r'(\d{3})\s*分?[-~～]\s*(\d{3})\s*分?')

    # 纯数字匹配（用于识别人数）
    NUMBER_PATTERN = re.compile(r'^\d+$')
    SCHOOL_PATTERN = re.compile(r"[\u4e00-\u9fa5]{2,}(大学|学院)")
    CODE_PATTERN = re.compile(r"(\d{6})")
    SEP_PATTERN = re.compile(r"[·•∙⋅．·]+")
    STUDY_MODE_ALIASES = {
        "全日制": "全日制",
        "非全日制": "非全日制",
        "非全": "非全日制",
    }
    HEADER_MIN_CONFIDENCE = 0.78
    TRAILING_SEPARATORS = re.compile(r"[／/|\\\\]+$")
    TRAILING_DOTS = re.compile(r"[·•∙⋅．·]+$")
    PROVINCES = {
        "北京", "天津", "上海", "重庆",
        "河北", "山西", "辽宁", "吉林", "黑龙江",
        "江苏", "浙江", "安徽", "福建", "江西", "山东",
        "河南", "湖北", "湖南", "广东", "海南",
        "四川", "贵州", "云南", "陕西", "甘肃", "青海",
        "台湾",
        "广西", "内蒙古", "宁夏", "新疆", "西藏",
        "香港", "澳门",
    }

    def __init__(self, lang: str = "ch", logger: logging.Logger | None = None, debug_dir: Path | None = None):
        """初始化简单OCR提取器

        Args:
            lang: OCR语言，默认中文
            logger: 日志记录器
            debug_dir: 调试输出目录，如果设置则保存调试可视化图片
        """
        self.lang = lang
        self.logger = logger or logging.getLogger(__name__)
        self.debug_dir = debug_dir
        self._ocr_engine: PaddleOCR | None = None

    @property
    def ocr(self) -> PaddleOCR:
        """懒加载OCR引擎"""
        if self._ocr_engine is None:
            self.logger.info(f"初始化基础OCR引擎 (lang={self.lang})")
            self._ocr_engine = PaddleOCR(lang=self.lang, use_angle_cls=True)
        return self._ocr_engine




    def extract_table_from_image(self, image_path: Path) -> list[ScoreRow]:
        """从图片提取表格数据 - 基于分数段驱动的新方案

        Args:
            image_path: 图片路径

        Returns:
            提取的分数行列表

        Raises:
            ValueError: 图片无法打开或OCR失败
        """
        try:
            img = Image.open(image_path).convert('RGB')
            img_array = np.array(img)
        except Exception as exc:
            raise ValueError(f"无法打开图片: {exc}") from exc

        # 1. OCR识别
        ocr_boxes = self._ocr_boxes(img_array)
        self.logger.info(f"识别到 {len(ocr_boxes)} 个文本框")

        if not ocr_boxes:
            raise ValueError("OCR未识别到任何文本框")

        # 2. 尝试定位表头（用于确定数据区域，不是必须的）
        header = self._find_table_header(ocr_boxes, img.width)
        if header:
            self.logger.info(
                "表头定位成功: y=%.0f, col1=[%.0f-%.0f], col2=[%.0f-%.0f], col3=[%.0f-%.0f]",
                header.header_y,
                header.col1_x_range[0], header.col1_x_range[1],
                header.col2_x_range[0], header.col2_x_range[1],
                header.col3_x_range[0], header.col3_x_range[1],
            )
            data_top_y = header.header_y + 30  # 表头下方30像素开始是数据区
        else:
            self.logger.warning("未找到表头，使用估算的数据区域")
            data_top_y = img.height * 0.25  # 估算：图片上方25%是表头区域

        # 3. 过滤出数据区域的文本框（排除底部按钮）
        button_top = self._find_button_top(ocr_boxes)
        if button_top:
            data_bottom_y = button_top - 20
        else:
            data_bottom_y = img.height - 80

        data_boxes = [
            b for b in ocr_boxes
            if data_top_y < b.get("y", 0.0) < data_bottom_y
        ]

        self.logger.info(f"数据区域内有 {len(data_boxes)} 个文本框")

        # 4. 以分数段为锚点，分组成行（行带分组，传入全部OCR结果）
        row_groups = self._group_boxes_by_score_anchors(data_boxes, ocr_boxes)
        self.logger.info(f"找到 {len(row_groups)} 个分数段，对应 {len(row_groups)} 行数据")

        if not row_groups:
            raise ValueError("未找到任何分数段，无法提取表格数据")

        # 5. 从每个行分组中提取数据
        score_rows, failed_rows = self._extract_rows_from_groups(row_groups, img_array, header)

        if not score_rows:
            raise ValueError(f"所有行识别失败，无有效数据。失败详情: {failed_rows[:5]}")

        # 6. 严格验证 - 零容忍策略：任何行失败都不输出
        if failed_rows:
            failure_rate = len(failed_rows) / len(row_groups)
            raise ValueError(
                f"表格提取不完整，丢弃了 {len(failed_rows)}/{len(row_groups)} 行 ({failure_rate:.0%})。"
                f"失败行详情: {failed_rows[:10]}"
            )

        if not self._is_score_increasing(score_rows):
            raise ValueError("分数段未递增，数据异常")

        self.logger.info(
            "✓ 提取成功: %d 行有效数据，零失败",
            len(score_rows),
        )

        # 7. 保存调试可视化
        if self.debug_dir:
            try:
                self._save_debug_visualization(
                    image_path, img_array, ocr_boxes, header,
                    row_groups, score_rows, failed_rows, self.debug_dir
                )
            except Exception as e:
                self.logger.warning(f"保存调试可视化失败: {e}")

        return score_rows

    def _group_boxes_by_score_anchors(
        self,
        boxes: list[dict[str, Any]],
        all_boxes: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """以分数段为锚点，将文本框分组成行（行带分组，末行更稳健）

        Args:
            boxes: 数据区域内的所有文本框（用于找分数段锚点）
            all_boxes: 全部OCR文本框（用于按行带收集，覆盖末行偏移）

        Returns:
            行分组列表，每个元素包含：
            - anchor: 分数段文本框（锚点）
            - boxes: 该行所有文本框
            - y: 该行的y坐标（分数段的y）
        """
        # 1. 找到所有包含分数段的文本框（仅在数据区域内查找，避免误识别）
        score_boxes = [
            b for b in boxes
            if self.SCORE_PATTERN.search(b.get("text", ""))
        ]

        if not score_boxes:
            self.logger.error("数据区域内未找到任何分数段")
            return []

        # 按y坐标排序，确定行顺序
        score_boxes.sort(key=lambda b: b.get("y", 0.0))
        score_ys = [sb.get("y", 0.0) for sb in score_boxes]

        self.logger.info(f"找到 {len(score_boxes)} 个分数段文本框")

        # 2. 计算行距/行带参数
        y_gaps = [
            score_ys[i + 1] - score_ys[i]
            for i in range(len(score_ys) - 1)
            if score_ys[i + 1] > score_ys[i]
        ]
        if y_gaps:
            y_gaps_sorted = sorted(y_gaps)
            median_gap = y_gaps_sorted[len(y_gaps_sorted) // 2]
            min_gap = min(y_gaps_sorted)
            avg_gap = sum(y_gaps_sorted) / len(y_gaps_sorted)
        else:
            median_gap = 80.0
            min_gap = 80.0
            avg_gap = 80.0

        # 行带期望高度（用于首/末行下界估计）
        base_gap = max(60.0, min(140.0, median_gap))
        self.logger.info(
            "行间距: min=%.1f, avg=%.1f, median=%.1f, base_gap=%.1f",
            min_gap, avg_gap, median_gap, base_gap
        )

        # 数据区顶部估计，用于首行上界
        data_top_y = min((b.get("y", 0.0) for b in boxes), default=0.0)
        data_top_y = max(0.0, data_top_y - min(base_gap * 0.4, 30.0))

        # 按钮上缘用于末行下界保护
        button_top = self._find_button_top(all_boxes)
        button_guard = button_top - 10.0 if button_top is not None else None

        # 3. 对每个分数段，使用行带收集所有文本框
        row_groups = []
        for idx, score_box in enumerate(score_boxes, start=1):
            score_y = score_box.get("y", 0.0)

            prev_y = score_ys[idx - 2] if idx >= 2 else None
            next_y = score_ys[idx] if idx < len(score_ys) else None

            # 行带上下界（用前后锚点中点，首/末行用估计行高）
            if prev_y is None:
                top_boundary = data_top_y
            else:
                top_boundary = (prev_y + score_y) / 2

            if next_y is None:
                bottom_boundary = score_y + base_gap * 0.8
            else:
                bottom_boundary = (score_y + next_y) / 2

            if button_guard is not None:
                bottom_boundary = min(bottom_boundary, button_guard)

            if bottom_boundary <= top_boundary:
                bottom_boundary = top_boundary + max(20.0, base_gap * 0.5)

            row_boxes = [
                b for b in all_boxes
                if top_boundary <= b.get("y", 0.0) <= bottom_boundary
            ]

            row_groups.append({
                "row_index": idx,
                "anchor": score_box,
                "boxes": row_boxes,
                "y": score_y,
                "top": top_boundary,
                "bottom": bottom_boundary,
            })

            self.logger.debug(
                "行 %d: y=%.1f, 带=[%.1f, %.1f], 收集到 %d 个文本框",
                idx, score_y, top_boundary, bottom_boundary, len(row_boxes)
            )

        return row_groups

    def _extract_rows_from_groups(
        self,
        row_groups: list[dict[str, Any]],
        img_array: np.ndarray,
        header: TableHeader | None,
    ) -> tuple[list[ScoreRow], list[tuple[int, str]]]:
        """从行分组中提取数据

        Args:
            row_groups: 行分组列表
            img_array: 图片数组
            header: 表头信息（可选，用于回退方案）

        Returns:
            (成功的分数行列表, 失败的行列表)
        """
        score_rows: list[ScoreRow] = []
        failed: list[tuple[int, str]] = []
        image_width = img_array.shape[1]
        col_windows = self._column_windows(header, image_width)

        for group in row_groups:
            idx = group["row_index"]
            row_boxes = group["boxes"]
            anchor = group["anchor"]
            row_is_last = (idx == len(row_groups))
            row_top = float(group.get("top", group.get("y", 0.0) - 40.0))
            row_bottom = float(group.get("bottom", group.get("y", 0.0) + 40.0))

            # 末行：仅保留列通道内的文本，屏蔽底部按钮类文本
            if row_is_last:
                row_boxes = [
                    b for b in row_boxes
                    if self._box_in_columns(b, col_windows, pad=25.0)
                    and not self._looks_like_button_text(b.get("text", ""))
                ]

            if not row_boxes:
                failed.append((idx, "该行没有文本框"))
                continue

            # 方法1：尝试用间隙检测分成3列
            columns = self._split_row_into_columns_by_gap(row_boxes, gap_threshold=60.0)

            if columns is None:
                # 方法2：如果间隙检测失败，尝试用表头固定边界（回退方案）
                if header:
                    self.logger.debug(f"行 {idx}: 间隙检测失败，使用表头边界回退")
                    score_texts, cand_texts, admit_texts = self._collect_by_header_boundary(
                        row_boxes, header
                    )
                else:
                    # 方法3：最后的回退，按x坐标简单三等分
                    self.logger.debug(f"行 {idx}: 间隙检测失败且无表头，使用三等分回退")
                    score_texts, cand_texts, admit_texts = self._collect_by_thirds(
                        row_boxes, img_array.shape[1]
                    )
            else:
                # 间隙检测成功
                col1_boxes, col2_boxes, col3_boxes = columns
                score_texts = self._collect_texts_from_boxes(col1_boxes, merge_numeric=row_is_last)
                cand_texts = self._collect_texts_from_boxes(col2_boxes, merge_numeric=row_is_last)
                admit_texts = self._collect_texts_from_boxes(col3_boxes, merge_numeric=row_is_last)

                self.logger.debug(
                    f"行 {idx}: 间隙检测成功，3列文本数 [{len(score_texts)}, {len(cand_texts)}, {len(admit_texts)}]"
                )

            # 解析数据
            parsed_score = self._parse_score_from_texts(score_texts)
            # 严格数字共识：多重OCR + 高置信度，不一致则失败
            cand_cell = self._crop_cell_from_group(img_array, row_top, row_bottom, col_windows[1])
            admit_cell = self._crop_cell_from_group(img_array, row_top, row_bottom, col_windows[2])
            candidates_val = self._extract_number_strict(cand_texts, cand_cell)
            admitted_val = self._extract_number_strict(admit_texts, admit_cell)

            # 末行缺列时，尝试窄裁剪重跑OCR兜底（不放松校验）
            if row_is_last and (parsed_score is None or candidates_val is None or admitted_val is None):
                recovered = self._recover_last_row_via_crop(
                    img_array=img_array,
                    row_group=group,
                    header=header,
                )
                if recovered:
                    score_texts, cand_texts, admit_texts = recovered
                    parsed_score = self._parse_score_from_texts(score_texts)
                    candidates_val = self._extract_number_strict(cand_texts, cand_cell)
                    admitted_val = self._extract_number_strict(admit_texts, admit_cell)

            # 严格验证
            if parsed_score is None:
                failed.append((idx, f"无法解析分数段，文本: {score_texts}"))
                continue
            if candidates_val is None:
                failed.append((idx, f"无法解析复试人数，文本: {cand_texts}"))
                continue
            if admitted_val is None:
                failed.append((idx, f"无法解析录取人数，文本: {admit_texts}"))
                continue

            score_range, lower, upper = parsed_score

            # 分数段验证
            if lower >= upper:
                failed.append((idx, f"分数段无效 {score_range} (下限>=上限)"))
                continue
            if lower < 100 or upper > 500:
                failed.append((idx, f"分数段超出合理范围 {score_range}"))
                continue

            # 人数逻辑验证
            if admitted_val > candidates_val:
                failed.append((idx, f"录取人数({admitted_val})大于复试人数({candidates_val})"))
                continue
            if candidates_val > 1000 or admitted_val > 1000:
                failed.append((idx, f"人数异常大: 复试={candidates_val}, 录取={admitted_val}"))
                continue

            # 与前一行的递增检查
            if score_rows and score_rows[-1].upper > lower:
                failed.append((idx, f"分数段未递增: 前行{score_rows[-1].score_range} vs 当前{score_range}"))
                continue

            # 通过所有验证，添加到结果
            score_rows.append(
                ScoreRow(
                    score_range=score_range,
                    lower=lower,
                    upper=upper,
                    candidates=candidates_val,
                    admitted=admitted_val,
                )
            )

            self.logger.debug(
                f"✓ 行 {idx}: {score_range}, 复试={candidates_val}, 录取={admitted_val}"
            )

        return score_rows, failed

    def _recover_last_row_via_crop(
        self,
        img_array: np.ndarray,
        row_group: dict[str, Any],
        header: TableHeader | None,
    ) -> tuple[list[str], list[str], list[str]] | None:
        """末行窄裁剪单独OCR，尝试补全缺失列（不放松数值校验）。"""
        top = float(row_group.get("top", row_group.get("y", 0.0) - 30.0))
        bottom = float(row_group.get("bottom", row_group.get("y", 0.0) + 80.0))
        h = img_array.shape[0]
        if h <= 0:
            return None

        margin = 8.0
        top = max(0.0, top - margin)
        bottom = min(float(h), bottom + margin)
        if bottom - top < 20.0:
            return None

        try:
            crop = img_array[int(top): int(bottom), :, :]
            boxes = self._ocr_boxes(crop)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("末行裁剪OCR失败: %s", exc)
            return None

        # y坐标补偿回全图坐标
        for b in boxes:
            b["y"] = b.get("y", 0.0) + top
            if "y_min" in b:
                b["y_min"] = b.get("y_min", 0.0) + top
            if "y_max" in b:
                b["y_max"] = b.get("y_max", 0.0) + top

        col_windows = self._column_windows(header, img_array.shape[1])
        boxes = [
            b for b in boxes
            if self._box_in_columns(b, col_windows, pad=15.0)
            and not self._looks_like_button_text(b.get("text", ""))
        ]
        if not boxes:
            return None

        columns = self._split_row_into_columns_by_gap(boxes, gap_threshold=60.0)
        if columns is None:
            if header:
                score_texts, cand_texts, admit_texts = self._collect_by_header_boundary(boxes, header)
            else:
                score_texts, cand_texts, admit_texts = self._collect_by_thirds(boxes, img_array.shape[1])
        else:
            col1_boxes, col2_boxes, col3_boxes = columns
            score_texts = [b.get("text", "").strip() for b in col1_boxes if b.get("text")]
            cand_texts = [b.get("text", "").strip() for b in col2_boxes if b.get("text")]
            admit_texts = [b.get("text", "").strip() for b in col3_boxes if b.get("text")]

        if not (score_texts or cand_texts or admit_texts):
            return None

        self.logger.debug(
            "末行窄裁剪OCR补充: score=%s, cand=%s, admit=%s",
            score_texts, cand_texts, admit_texts,
        )
        return score_texts, cand_texts, admit_texts

    def _collect_by_header_boundary(
        self,
        row_boxes: list[dict[str, Any]],
        header: TableHeader,
    ) -> tuple[list[str], list[str], list[str]]:
        """使用表头边界收集列数据（回退方案）"""
        def collect_texts(x_range: tuple[float, float]) -> list[str]:
            texts = [
                b.get("text", "").strip()
                for b in row_boxes
                if x_range[0] <= b.get("x", 0.0) <= x_range[1]
            ]
            return [t for t in texts if t]

        score_texts = collect_texts(header.col1_x_range)
        cand_texts = collect_texts(header.col2_x_range)
        admit_texts = collect_texts(header.col3_x_range)

        return score_texts, cand_texts, admit_texts

    def _collect_by_thirds(
        self,
        row_boxes: list[dict[str, Any]],
        image_width: int,
    ) -> tuple[list[str], list[str], list[str]]:
        """按图片宽度三等分收集列数据（最后回退方案）"""
        third = image_width / 3

        def collect_texts(x_min: float, x_max: float) -> list[str]:
            texts = [
                b.get("text", "").strip()
                for b in row_boxes
                if x_min <= b.get("x", 0.0) < x_max
            ]
            return [t for t in texts if t]

        score_texts = collect_texts(0, third)
        cand_texts = collect_texts(third, third * 2)
        admit_texts = collect_texts(third * 2, image_width)

        return score_texts, cand_texts, admit_texts

    def _column_windows(self, header: TableHeader | None, image_width: int) -> list[tuple[float, float]]:
        """获取三列的x窗口（用于末行过滤/回退）。"""
        if header:
            return [
                header.col1_x_range,
                header.col2_x_range,
                header.col3_x_range,
            ]
        third = image_width / 3
        return [
            (0.0, third),
            (third, third * 2),
            (third * 2, float(image_width)),
        ]

    def _box_in_columns(self, box: dict[str, Any], windows: list[tuple[float, float]], pad: float = 0.0) -> bool:
        x = box.get("x", 0.0)
        return any(x_range[0] - pad <= x <= x_range[1] + pad for x_range in windows)

    def _looks_like_button_text(self, text: str) -> bool:
        text = text or ""
        noisy_tokens = ("一对一", "择校", "点我", "PK", "话题", "咨询")
        return any(token in text for token in noisy_tokens)

    def _is_score_increasing(self, rows: list[ScoreRow]) -> bool:
        for i in range(len(rows) - 1):
            if rows[i].upper > rows[i + 1].lower:
                self.logger.error(
                    "分数段重叠: %s vs %s", rows[i].score_range, rows[i + 1].score_range
                )
                return False
        return True

    def extract_header_metadata(self, image_path: Path) -> SchoolMetadata:
        """仅从表头区域提取院校元信息，缺失/置信度不足即报错。"""
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as exc:
            raise ValueError(f"无法打开图片: {exc}") from exc

        header_height = max(int(img.height * 0.4), 1)
        header_crop = img.crop((0, 0, img.width, header_height))
        boxes = self._ocr_boxes(np.array(header_crop))
        if not boxes:
            raise ValueError("表头区域未识别到任何文本")

        rows = self._group_into_rows(boxes, y_threshold=28)
        lines: list[dict[str, Any]] = []
        for row_boxes in rows:
            texts = [box["text"].strip() for box in row_boxes if box.get("text")]
            if not texts:
                continue
            raw = " ".join(texts)
            norm = self._normalize_header_text(raw)
            confs = [float(box.get("conf", 0.0)) for box in row_boxes if box.get("conf") is not None]
            avg_conf = sum(confs) / len(confs) if confs else 0.0
            lines.append(
                {
                    "raw": raw,
                    "norm": norm,
                    "boxes": row_boxes,
                    "avg_conf": avg_conf,
                }
            )

        raw_header_text = "\n".join(line["raw"] for line in lines)

        school_line = self._pick_school_line(lines)
        school_match = self.SCHOOL_PATTERN.search(school_line["norm"])
        if not school_match:
            raise ValueError("未找到院校名称")
        school = school_match.group(0)
        self._ensure_conf(
            self._pattern_conf(school_line["boxes"], self.SCHOOL_PATTERN),
            "院校名称置信度不足",
        )

        code_line = self._pick_code_line(lines)
        code_match = self.CODE_PATTERN.search(code_line["norm"])
        if not code_match:
            raise ValueError("未找到专业代码")
        code = code_match.group(1)
        code_conf = self._pattern_conf(code_line["boxes"], self.CODE_PATTERN)
        self._ensure_conf(code_conf, "专业代码置信度不足")

        study_mode, study_conf = self._extract_study_mode(code_line)
        if not study_mode:
            raise ValueError("未识别到学习方式")
        self._ensure_conf(study_conf, "学习方式置信度不足")

        college, major = self._parse_college_major(code_line, code_match.start())
        if not college or not major:
            raise ValueError("学院/专业解析失败")

        province, province_conf = self._extract_province(lines)
        if not province:
            raise ValueError("未识别到省份信息")
        self._ensure_conf(province_conf, "省份置信度不足")

        return SchoolMetadata(
            school=school,
            college=college,
            major=major,
            code=code,
            study_mode=study_mode,
            province=province,
            raw_header=raw_header_text,
        )

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

                # 计算中心坐标和边界
                x_coords = [p[0] for p in bbox]
                y_coords = [p[1] for p in bbox]
                x_center = sum(x_coords) / len(x_coords)
                y_center = sum(y_coords) / len(y_coords)

                boxes.append({
                    'text': text.strip(),
                    'x': x_center,
                    'y': y_center,
                    'x_min': min(x_coords),
                    'x_max': max(x_coords),
                    'y_min': min(y_coords),
                    'y_max': max(y_coords),
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

    def _collect_texts_from_boxes(
        self,
        boxes: list[dict[str, Any]],
        merge_numeric: bool = False,
    ) -> list[str]:
        """按x排序收集文本，可选合并数字碎片。"""
        texts = [
            b.get("text", "").strip()
            for b in sorted(boxes, key=lambda b: b.get("x", 0.0))
            if b.get("text")
        ]
        texts = [t for t in texts if t]
        if not merge_numeric or len(texts) <= 1:
            return texts

        if all(re.fullmatch(r"\d{1,3}", t) for t in texts):
            return ["".join(texts)]
        return texts

    def _ocr_boxes(self, img_array: np.ndarray) -> list[dict[str, Any]]:
        """运行 OCR 并统一为 box 结构。"""
        self.logger.info("开始OCR识别图片")
        try:
            result = self.ocr.ocr(img_array)
        except Exception as exc:
            raise ValueError(f"OCR识别失败: {exc}") from exc

        if not result or not result[0]:
            raise ValueError("OCR未返回任何结果")

        first_result = result[0]
        if isinstance(first_result, dict):
            texts = first_result.get("rec_texts", [])
            polys = first_result.get("rec_polys", [])
            scores = first_result.get("rec_scores", [])
            boxes: list[dict[str, Any]] = []
            for i, (text, poly) in enumerate(zip(texts, polys)):
                try:
                    x_vals = poly[:, 0]
                    y_vals = poly[:, 1]
                    x_center = float(np.mean(x_vals))
                    y_center = float(np.mean(y_vals))
                    conf = scores[i] if i < len(scores) else 1.0
                    boxes.append({
                        "text": text.strip(),
                        "x": x_center,
                        "y": y_center,
                        "x_min": float(np.min(x_vals)),
                        "x_max": float(np.max(x_vals)),
                        "y_min": float(np.min(y_vals)),
                        "y_max": float(np.max(y_vals)),
                        "conf": conf,
                    })
                except Exception as exc:  # noqa: BLE001
                    self.logger.warning(f"处理文本框失败: {exc}")
                    continue
            return boxes

        boxes = self._parse_ocr_result(first_result)
        return boxes

    def _normalize_header_text(self, text: str) -> str:
        cleaned = re.sub(r"\s+", "", text)
        for bad, good in {
            "﹒": "·",
            "・": "·",
            "•": "·",
            "∙": "·",
            "．": "·",
        }.items():
            cleaned = cleaned.replace(bad, good)
        return cleaned

    def _pick_school_line(self, lines: list[dict[str, Any]]) -> dict[str, Any]:
        candidates = [line for line in lines if self.SCHOOL_PATTERN.search(line["norm"])]
        if not candidates:
            raise ValueError("表头未找到院校名称行")

        def score_line(line: dict[str, Any]) -> tuple[int, float, int]:
            norm = line["norm"]
            has_university = 1 if "大学" in norm else 0
            conf = self._pattern_conf(line["boxes"], self.SCHOOL_PATTERN)
            length = len(norm)
            return has_university, conf, length

        best = max(candidates, key=score_line)
        # 如果最佳行仍然只有“学院”且不含“大学”，视为不可信，直接报错
        if "大学" not in best["norm"]:
            raise ValueError("表头院校名称缺失或被学院字段占用")
        return best

    def _pick_code_line(self, lines: list[dict[str, Any]]) -> dict[str, Any]:
        candidates = [line for line in lines if self.CODE_PATTERN.search(line["norm"])]
        if not candidates:
            raise ValueError("表头未找到代码行")
        return max(
            candidates,
            key=lambda line: (
                self._pattern_conf(line["boxes"], self.CODE_PATTERN),
                len(line["norm"]),
            ),
        )

    def _pattern_conf(self, boxes: list[dict[str, Any]], pattern: re.Pattern[str]) -> float:
        confs = []
        for box in boxes:
            text = self._normalize_header_text(box.get("text", ""))
            if pattern.search(text):
                try:
                    confs.append(float(box.get("conf", 0.0)))
                except Exception:  # noqa: BLE001
                    continue
        return max(confs) if confs else 0.0

    def _ensure_conf(self, confidence: float, message: str) -> None:
        if confidence < self.HEADER_MIN_CONFIDENCE:
            raise ValueError(message)

    def _extract_study_mode(self, line: dict[str, Any]) -> tuple[str | None, float]:
        norm = line["norm"]
        best_mode: str | None = None
        best_conf = 0.0
        for key, canonical in self.STUDY_MODE_ALIASES.items():
            if key in norm:
                conf = self._pattern_conf(line["boxes"], re.compile(re.escape(key)))
                if conf > best_conf:
                    best_conf = conf
                    best_mode = canonical
        return best_mode, best_conf

    def _parse_college_major(self, code_line: dict[str, Any], code_start: int) -> tuple[str, str]:
        before_code = code_line["norm"][:code_start]
        before_code = before_code.replace("全国", "")
        before_code = before_code.rstrip("/").rstrip("-")
        parts = [p for p in self.SEP_PATTERN.split(before_code) if p]
        if len(parts) < 2:
            raise ValueError("未找到“学院·专业”格式")
        college = parts[0].strip()
        major = parts[1].strip()
        if not re.search(r"(学院|系|研究院)", college):
            raise ValueError("学院字段缺少关键字")
        major_clean = self._clean_major(major)
        if not major_clean:
            raise ValueError("专业字段为空或无效")
        return college, major_clean

    def _clean_major(self, text: str) -> str:
        """去除专业字段的尾部分隔符/特殊符号，保留中文/字母/数字等主体。"""
        cleaned = text.strip()
        cleaned = self.TRAILING_SEPARATORS.sub("", cleaned)
        cleaned = self.TRAILING_DOTS.sub("", cleaned)
        cleaned = cleaned.strip()
        return cleaned

    # ===== 表格结构定位 =====
    def _find_table_header(self, boxes: list[dict[str, Any]], image_width: float) -> TableHeader | None:
        header_positions: dict[str, dict[str, float]] = {}
        for box in boxes:
            text = box.get("text", "")
            for key in ("分数段", "复试人数", "录取人数"):
                if key in text:
                    header_positions[key] = {
                        "x": float(box.get("x", 0.0)),
                        "y": float(box.get("y", 0.0)),
                    }
                    break

        if len(header_positions) != 3:
            missing = {"分数段", "复试人数", "录取人数"} - set(header_positions.keys())
            self.logger.error("表头缺失: %s", missing)
            return None

        header_y = sum(pos["y"] for pos in header_positions.values()) / 3
        col1_x = header_positions["分数段"]["x"]
        col2_x = header_positions["复试人数"]["x"]
        col3_x = header_positions["录取人数"]["x"]

        mid_12 = (col1_x + col2_x) / 2
        mid_23 = (col2_x + col3_x) / 2

        return TableHeader(
            col1_x_range=(0.0, mid_12),
            col2_x_range=(mid_12, mid_23),
            col3_x_range=(mid_23, float(image_width)),
            header_y=header_y,
        )

    def _locate_data_region(
        self,
        boxes: list[dict[str, Any]],
        header: TableHeader,
        image_height: float,
    ) -> DataRegion:
        top_y = max(0.0, header.header_y + 30.0)
        button_top = self._find_button_top(boxes)
        if button_top is None:
            bottom_y = image_height - 80.0
        else:
            bottom_y = float(button_top) - 12.0

        if bottom_y - top_y < 150:
            bottom_y = min(image_height - 40.0, top_y + max(180.0, image_height * 0.35))

        bottom_y = min(bottom_y, image_height - 10.0)
        return DataRegion(top_y=top_y, bottom_y=bottom_y)

    def _detect_horizontal_lines(self, gray_image: np.ndarray) -> list[float]:
        """通过行亮度检测水平网格线。"""
        if gray_image.size == 0:
            return []
        row_brightness = np.mean(gray_image, axis=1)
        threshold = np.percentile(row_brightness, 30)
        dark_rows = np.where(row_brightness < threshold)[0]
        if len(dark_rows) == 0:
            return []

        lines: list[float] = []
        start = dark_rows[0]
        for i in range(1, len(dark_rows)):
            if dark_rows[i] - dark_rows[i - 1] > 5:
                lines.append((start + dark_rows[i - 1]) / 2)
                start = dark_rows[i]
        lines.append((start + dark_rows[-1]) / 2)
        return lines

    def _segment_rows(
        self,
        img_array: np.ndarray,
        data_region: DataRegion,
        ocr_boxes: list[dict[str, Any]],
    ) -> list[RowRegion]:
        """使用网格线/投影或行高估计切分行。"""
        height, width = img_array.shape[:2]
        top = int(max(0.0, data_region.top_y))
        bottom = int(min(float(height), data_region.bottom_y))
        bottom = max(bottom, top + 1)

        gray = img_array[top:bottom, :]
        if gray.ndim == 3:
            gray = np.mean(gray, axis=2).astype(np.uint8)

        lines = self._detect_horizontal_lines(gray)
        rows: list[RowRegion] = []

        if len(lines) >= 3:
            for i in range(len(lines) - 1):
                y_start = top + lines[i]
                y_end = top + lines[i + 1]
                if y_end - y_start < 12:
                    continue
                y_center = (y_start + y_end) / 2
                rows.append(RowRegion(y_start, y_end, y_center))
            if rows:
                return rows

        # 回退：根据分数段的y估计行高，均匀切分
        score_boxes = [
            b for b in ocr_boxes
            if self.SCORE_PATTERN.search(b.get("text", ""))
            and data_region.top_y <= b.get("y", 0.0) <= data_region.bottom_y
        ]
        score_ys = sorted(b.get("y", 0.0) for b in score_boxes)
        row_height = self._estimate_row_height(score_ys)
        if row_height <= 0:
            row_height = 40.0

        y = data_region.top_y
        max_rows = 80
        count = 0
        while y + row_height <= data_region.bottom_y and count < max_rows:
            rows.append(RowRegion(y, y + row_height, y + row_height / 2))
            y += row_height
            count += 1

        return rows

    def _isolate_table_region(self, boxes: list[dict[str, Any]], image_height: int) -> tuple[list[dict[str, Any]], float]:
        """根据按钮和分数段位置截取表格区域，避免按钮干扰。"""
        if not boxes:
            return [], 40.0

        score_boxes = [b for b in boxes if self.SCORE_PATTERN.search(b.get("text", ""))]
        score_ys = sorted(b["y"] for b in score_boxes) if score_boxes else []
        row_height = self._estimate_row_height(score_ys)

        button_top = self._find_button_top(boxes)
        if button_top is None:
            # 无按钮时不裁剪，仅使用行高阈值
            return boxes, row_height

        # padding 自适应：行高的 0.7，至少 8，最多行高
        padding = max(8.0, min(row_height, row_height * 0.7))
        bottom_limit = button_top - padding

        last_score_y = score_ys[-1] if score_ys else 0.0
        min_bottom = last_score_y + row_height * 0.6
        if bottom_limit < min_bottom:
            bottom_limit = min(button_top, min_bottom)

        # 若裁剪下界侵入表格末行（数字区域可能低于分数段），则放弃裁剪以保留末行
        guard_bottom = last_score_y + row_height * 0.9
        if bottom_limit < guard_bottom:
            return boxes, row_height

        filtered = [b for b in boxes if b.get("y", 0) <= bottom_limit]
        # 如果裁剪后分数段数量比原来少，怀疑截断，则不裁剪
        if score_boxes:
            kept_scores = [b for b in filtered if self.SCORE_PATTERN.search(b.get("text", ""))]
            if len(kept_scores) < len(score_boxes):
                return boxes, row_height
        return filtered, row_height

    def _estimate_row_height(self, score_ys: list[float]) -> float:
        if len(score_ys) < 2:
            return 40.0
        diffs = sorted(abs(b - a) for a, b in zip(score_ys, score_ys[1:]) if b > a)
        if not diffs:
            return 40.0
        mid = diffs[len(diffs) // 2]
        return max(24.0, min(60.0, mid))

    def _find_button_top(self, boxes: list[dict[str, Any]]) -> float | None:
        """查找“一对一择校”按钮的上边界。"""
        button_texts = ("一对一择校", "一对一", "择校", "点我", "咨询")
        candidates: list[float] = []
        for b in boxes:
            text = b.get("text", "")
            if any(token in text for token in button_texts):
                y_min = b.get("y_min")
                if y_min is not None:
                    candidates.append(float(y_min))
        return min(candidates) if candidates else None

    def _adaptive_row_threshold(self, row_height: float) -> int:
        if row_height <= 0:
            return 40
        return int(max(18.0, min(50.0, row_height * 0.8)))

    def _extract_province(self, lines: list[dict[str, Any]]) -> tuple[str | None, float]:
        """在表头文本行中提取省份/直辖市/自治区，未找到即返回 (None, 0.0)。"""
        if not lines:
            return None, 0.0
        # 构造匹配所有省份名称的正则，优先长度长的匹配
        pattern = re.compile("|".join(sorted(self.PROVINCES, key=len, reverse=True)))
        best_province: str | None = None
        best_conf = 0.0
        for line in lines:
            match = pattern.search(line["norm"])
            if not match:
                continue
            prov_raw = match.group(0)
            canonical = self._canonical_province(prov_raw)
            if not canonical:
                continue
            conf = self._pattern_conf(line["boxes"], re.compile(re.escape(prov_raw)))
            if conf > best_conf:
                best_conf = conf
                best_province = canonical
        return best_province, best_conf

    def _canonical_province(self, text: str) -> str:
        cleaned = text.strip()
        cleaned = cleaned.replace("省", "").replace("市", "")
        cleaned = re.sub(r"(壮族自治区|维吾尔自治区|回族自治区|自治区|特别行政区)", "", cleaned)
        base = cleaned.strip()
        if not base or base not in self.PROVINCES:
            return ""

        # 直辖市
        if base in {"北京", "天津", "上海", "重庆"}:
            return f"{base}市"

        # 自治区/特别行政区
        autonomous_map = {
            "广西": "广西壮族自治区",
            "内蒙古": "内蒙古自治区",
            "宁夏": "宁夏回族自治区",
            "新疆": "新疆维吾尔自治区",
            "西藏": "西藏自治区",
        }
        if base in autonomous_map:
            return autonomous_map[base]
        if base in {"香港", "澳门"}:
            return base
        if base == "台湾":
            return "台湾省"

        # 其他省份
        return f"{base}省"

    # ===== 行/列切分 + 单元格OCR =====
    def _split_row_into_columns_by_gap(
        self,
        row_boxes: list[dict[str, Any]],
        gap_threshold: float = 80.0,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]] | None:
        """基于x坐标间隙将一行的文本框动态分成3列

        Args:
            row_boxes: 一行内的所有文本框
            gap_threshold: 判定为列分界的最小间隙（像素）

        Returns:
            (第1列boxes, 第2列boxes, 第3列boxes) 或 None（分组失败）
        """
        if not row_boxes:
            return None

        # 按x坐标排序
        sorted_boxes = sorted(row_boxes, key=lambda b: b.get("x", 0.0))

        # 计算相邻文本框的间隙
        gaps: list[tuple[float, int]] = []  # (gap_size, position_after_gap)
        for i in range(len(sorted_boxes) - 1):
            x1_max = sorted_boxes[i].get("x_max", sorted_boxes[i].get("x", 0.0))
            x2_min = sorted_boxes[i + 1].get("x_min", sorted_boxes[i + 1].get("x", 0.0))
            gap = x2_min - x1_max
            if gap > gap_threshold:
                gaps.append((gap, i + 1))

        # 需要恰好2个间隙来分成3列
        if len(gaps) < 2:
            # 尝试降低阈值
            gaps_relaxed: list[tuple[float, int]] = []
            for i in range(len(sorted_boxes) - 1):
                x1_max = sorted_boxes[i].get("x_max", sorted_boxes[i].get("x", 0.0))
                x2_min = sorted_boxes[i + 1].get("x_min", sorted_boxes[i + 1].get("x", 0.0))
                gap = x2_min - x1_max
                if gap > gap_threshold * 0.6:  # 降低到60%
                    gaps_relaxed.append((gap, i + 1))

            if len(gaps_relaxed) < 2:
                return None
            gaps = gaps_relaxed

        # 选择最大的2个间隙作为列分界
        gaps.sort(reverse=True, key=lambda x: x[0])
        split_positions = sorted([gaps[0][1], gaps[1][1]])

        # 分成3组
        col1 = sorted_boxes[:split_positions[0]]
        col2 = sorted_boxes[split_positions[0]:split_positions[1]]
        col3 = sorted_boxes[split_positions[1]:]

        # 验证每组都有内容
        if not col1 or not col2 or not col3:
            return None

        return col1, col2, col3

    def _extract_rows_structured(
        self,
        img_array: np.ndarray,
        row_regions: list[RowRegion],
        header: TableHeader,
        ocr_boxes: list[dict[str, Any]],
    ) -> tuple[list[ScoreRow], list[tuple[int, str]]]:
        """基于间隙检测的动态列分组提取，单行失败不拖累整表。"""
        score_rows: list[ScoreRow] = []
        failed: list[tuple[int, str]] = []

        for idx, region in enumerate(row_regions, start=1):
            row_boxes = [
                b for b in ocr_boxes
                if region.y_start <= b.get("y", 0.0) <= region.y_end
            ]

            if not row_boxes:
                failed.append((idx, "该行未识别到任何文本框"))
                continue

            # 使用间隙检测动态分列
            columns = self._split_row_into_columns_by_gap(row_boxes)
            if columns is None:
                # 分组失败，尝试使用表头的固定边界作为回退
                self.logger.debug(f"行 {idx}: 间隙检测失败，使用固定边界回退")

                def collect_texts(x_range: tuple[float, float]) -> list[str]:
                    texts = [
                        b.get("text", "").strip()
                        for b in row_boxes
                        if x_range[0] <= b.get("x", 0.0) <= x_range[1]
                    ]
                    return [t for t in texts if t]

                score_texts = collect_texts(header.col1_x_range)
                cand_texts = collect_texts(header.col2_x_range)
                admit_texts = collect_texts(header.col3_x_range)

                # 回退到单元格OCR
                if not score_texts:
                    crop = self._crop_cell(img_array, region, header.col1_x_range)
                    score_texts = self._ocr_cell_text(crop)
                if not cand_texts:
                    crop = self._crop_cell(img_array, region, header.col2_x_range)
                    cand_texts = self._ocr_cell_text(crop)
                if not admit_texts:
                    crop = self._crop_cell(img_array, region, header.col3_x_range)
                    admit_texts = self._ocr_cell_text(crop)
            else:
                col1_boxes, col2_boxes, col3_boxes = columns
                score_texts = [b.get("text", "").strip() for b in col1_boxes if b.get("text")]
                cand_texts = [b.get("text", "").strip() for b in col2_boxes if b.get("text")]
                admit_texts = [b.get("text", "").strip() for b in col3_boxes if b.get("text")]

                self.logger.debug(
                    f"行 {idx}: 间隙检测成功，分成3列 [{len(col1_boxes)}, {len(col2_boxes)}, {len(col3_boxes)}]"
                )

            # 解析数据
            parsed_score = self._parse_score_from_texts(score_texts)
            candidates_val = self._extract_number_from_texts(cand_texts)
            admitted_val = self._extract_number_from_texts(admit_texts)

            # 严格验证
            if parsed_score is None:
                failed.append((idx, f"无法解析分数段，文本: {score_texts}"))
                continue
            if candidates_val is None:
                failed.append((idx, f"无法解析复试人数，文本: {cand_texts}"))
                continue
            if admitted_val is None:
                failed.append((idx, f"无法解析录取人数，文本: {admit_texts}"))
                continue

            score_range, lower, upper = parsed_score

            # 分数段验证
            if lower >= upper:
                failed.append((idx, f"分数段无效 {score_range} (下限>=上限)"))
                continue
            if lower < 100 or upper > 500:
                failed.append((idx, f"分数段超出合理范围 {score_range}"))
                continue

            # 人数逻辑验证
            if admitted_val > candidates_val:
                failed.append((idx, f"录取人数({admitted_val})大于复试人数({candidates_val})"))
                continue
            if candidates_val > 1000 or admitted_val > 1000:
                failed.append((idx, f"人数异常大: 复试={candidates_val}, 录取={admitted_val}"))
                continue

            # 与前一行的递增检查
            if score_rows and score_rows[-1].upper > lower:
                failed.append((idx, f"分数段未递增: 前行{score_rows[-1].score_range} vs 当前{score_range}"))
                continue

            score_rows.append(
                ScoreRow(
                    score_range=score_range,
                    lower=lower,
                    upper=upper,
                    candidates=candidates_val,
                    admitted=admitted_val,
                )
            )

            self.logger.debug(
                "行 %d: %s 复试=%s 录取=%s (boxes=%d)",
                idx,
                score_range,
                candidates_val,
                admitted_val,
                len(row_boxes),
            )

        return score_rows, failed

    def _crop_cell(
        self,
        img_array: np.ndarray,
        region: RowRegion,
        x_range: tuple[float, float],
        pad: int = 4,
    ) -> np.ndarray:
        """裁剪单元格区域，加入少量padding。"""
        height, width = img_array.shape[:2]
        x0 = max(0, int(x_range[0]) - pad)
        x1 = min(width, int(x_range[1]) + pad)
        y0 = max(0, int(region.y_start) - pad)
        y1 = min(height, int(region.y_end) + pad)
        if x1 <= x0 or y1 <= y0:
            return np.zeros((0, 0, 3), dtype=np.uint8)
        return img_array[y0:y1, x0:x1]

    def _crop_cell_from_group(
        self,
        img_array: np.ndarray,
        row_top: float,
        row_bottom: float,
        x_range: tuple[float, float],
        pad: int = 6,
    ) -> np.ndarray:
        """按行带与列窗口裁剪单元格，用于数字重识别。"""
        height, width = img_array.shape[:2]
        x0 = max(0, int(x_range[0]) - pad)
        x1 = min(width, int(x_range[1]) + pad)
        y0 = max(0, int(row_top) - pad)
        y1 = min(height, int(row_bottom) + pad)
        if x1 <= x0 or y1 <= y0:
            return np.zeros((0, 0, 3), dtype=np.uint8)
        return img_array[y0:y1, x0:x1]

    def _ocr_cell_text(self, cell_image: np.ndarray) -> list[str]:
        """对单元格局部进行OCR，提升小号数字召回。"""
        if cell_image.size == 0:
            return []
        try:
            h, w = cell_image.shape[:2]
        except ValueError:
            return []
        scale = 1.6 if max(h, w) < 160 else 1.2
        try:
            pil_img = Image.fromarray(cell_image)
            resized = pil_img.resize(
                (max(1, int(w * scale)), max(1, int(h * scale))),
                resample=Image.BICUBIC,
            )
            result = self.ocr.ocr(np.array(resized))
        except Exception as exc:  # noqa: BLE001
            self.logger.debug("单元格OCR失败: %s", exc)
            return []

        if not result or not result[0]:
            return []

        first = result[0]
        texts: list[str] = []
        if isinstance(first, dict):
            texts.extend([t.strip() for t in first.get("rec_texts", []) if t.strip()])
        else:
            parsed = self._parse_ocr_result(first)
            texts.extend([b["text"] for b in parsed if b.get("text")])
        return texts

    def _ocr_crop_boxes(self, image_array: np.ndarray) -> list[dict[str, Any]]:
        """对裁剪图运行OCR并返回box结构，不写日志。"""
        try:
            result = self.ocr.ocr(image_array)
        except Exception as exc:  # noqa: BLE001
            self.logger.debug("裁剪OCR失败: %s", exc)
            return []
        if not result or not result[0]:
            return []
        first = result[0]
        if isinstance(first, dict):
            texts = first.get("rec_texts", [])
            polys = first.get("rec_polys", [])
            scores = first.get("rec_scores", [])
            boxes: list[dict[str, Any]] = []
            for i, (text, poly) in enumerate(zip(texts, polys)):
                try:
                    x_vals = poly[:, 0]
                    y_vals = poly[:, 1]
                    x_center = float(np.mean(x_vals))
                    y_center = float(np.mean(y_vals))
                    conf = scores[i] if i < len(scores) else 1.0
                    boxes.append({
                        "text": text.strip(),
                        "x": x_center,
                        "y": y_center,
                        "x_min": float(np.min(x_vals)),
                        "x_max": float(np.max(x_vals)),
                        "y_min": float(np.min(y_vals)),
                        "y_max": float(np.max(y_vals)),
                        "conf": conf,
                    })
                except Exception:  # noqa: BLE001
                    continue
            return boxes
        return self._parse_ocr_result(first)

    def _ocr_number_variants(self, cell_image: np.ndarray) -> list[tuple[int, float]]:
        """对数字单元格多尺度/二值化OCR，输出(数值,置信度)。"""
        if cell_image.size == 0:
            return []
        h, w = cell_image.shape[:2]
        variants: list[np.ndarray] = []
        variants.append(cell_image)
        scale = 1.8 if max(h, w) < 160 else 1.3
        try:
            pil_img = Image.fromarray(cell_image)
            variants.append(np.array(pil_img.resize(
                (max(1, int(w * scale)), max(1, int(h * scale))),
                resample=Image.BICUBIC,
            )))
        except Exception:  # noqa: BLE001
            pass
        try:
            gray = Image.fromarray(cell_image).convert("L")
            bin_img = gray.point(lambda p: 0 if p < 180 else 255).convert("RGB")
            variants.append(np.array(bin_img))
        except Exception:  # noqa: BLE001
            pass

        candidates: list[tuple[int, float]] = []
        for var in variants:
            boxes = self._ocr_crop_boxes(var)
            for b in boxes:
                text = b.get("text", "").strip()
                if self.NUMBER_PATTERN.match(text):
                    try:
                        num = int(text)
                        conf = float(b.get("conf", 0.0))
                        candidates.append((num, conf))
                    except ValueError:
                        continue
        return candidates

    def _extract_number_strict(self, texts: list[str], cell_image: np.ndarray | None) -> int | None:
        """高门槛数字提取：多路一致且置信度达标，否则返回None。"""
        candidates: list[tuple[int, float]] = []
        for t in texts:
            t = t.strip()
            if not t:
                continue
            if self.NUMBER_PATTERN.match(t):
                try:
                    candidates.append((int(t), 0.5))  # 没有置信度时给中等权重
                except ValueError:
                    continue
            else:
                digits = re.findall(r"\d+", t)
                if digits:
                    try:
                        candidates.append((int(digits[0]), 0.4))
                    except ValueError:
                        continue

        if cell_image is not None:
            candidates.extend(self._ocr_number_variants(cell_image))

        if not candidates:
            return None

        # 汇总置信度
        agg: dict[int, float] = {}
        for num, conf in candidates:
            agg[num] = agg.get(num, 0.0) + float(conf)

        # 选置信度最高的数字，要求明显领先且达阈值
        sorted_nums = sorted(agg.items(), key=lambda x: x[1], reverse=True)
        top_num, top_conf = sorted_nums[0]
        second_conf = sorted_nums[1][1] if len(sorted_nums) > 1 else 0.0

        # 单字符要求更高置信度
        conf_threshold = 0.8 if top_num < 10 else 0.6
        if top_conf < conf_threshold:
            return None
        if second_conf and top_conf < second_conf * 1.5:
            return None

        return top_num

    def _parse_score_from_texts(self, texts: list[str]) -> tuple[str, int, int] | None:
        combined = " ".join(t for t in texts if t)
        match = self.SCORE_PATTERN.search(combined)
        if match:
            lower = int(match.group(1))
            upper = int(match.group(2))
            return f"{lower}-{upper}", lower, upper

        numbers = re.findall(r"\d{3}", combined)
        if len(numbers) >= 2:
            lower = int(numbers[0])
            upper = int(numbers[1])
            return f"{lower}-{upper}", lower, upper
        return None

    def _extract_number_from_texts(self, texts: list[str], min_conf: float | None = None) -> int | None:
        for text in texts:
            text = text.strip()
            if not text:
                continue
            if self.NUMBER_PATTERN.match(text):
                try:
                    return int(text)
                except ValueError:
                    continue
            digits = re.findall(r"\d+", text)
            if digits:
                try:
                    return int(digits[0])
                except ValueError:
                    continue
        return None

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
            if len(numbers) < 2:
                # 尝试从相邻行补充数字，处理按钮/间距导致的行拆分
                try:
                    current_y = statistics.median([b.get("y", 0.0) for b in row_boxes])
                except statistics.StatisticsError:
                    current_y = 0.0
                try:
                    heights = [
                        max(4.0, float(b.get("y_max", 0.0) - b.get("y_min", 0.0)))
                        for b in row_boxes
                    ]
                    row_height_guess = statistics.median(heights)
                except statistics.StatisticsError:
                    row_height_guess = 30.0
                # 放宽邻行合并范围，覆盖按钮挤压导致的垂直偏移
                delta_y = max(16.0, min(60.0, row_height_guess * 1.3))
                neighbor_indices = [row_idx + 1, row_idx - 1, row_idx + 2, row_idx - 2]
                for n_idx in neighbor_indices:
                    if n_idx < 0 or n_idx >= len(text_rows):
                        continue
                    neighbor = text_rows[n_idx]
                    # 避免把下一行的分数段行合并进来
                    if any(self.SCORE_PATTERN.search(b.get("text", "")) for b in neighbor):
                        continue
                    for b in neighbor:
                        if score_x and b.get("x", 0.0) <= score_x:
                            continue
                        if abs(b.get("y", 0.0) - current_y) > delta_y:
                            continue
                        text = b.get("text", "").strip()
                        if self.NUMBER_PATTERN.match(text):
                            try:
                                numbers.append(int(text))
                            except ValueError:
                                continue
                    if len(numbers) >= 2:
                        break

            if len(numbers) < 2:
                # 尝试从包含数字的混合文本中提取（例如被按钮挤压导致粘连）
                candidate_rows = [row_boxes]
                for n_idx in [row_idx + 1, row_idx - 1, row_idx + 2, row_idx - 2]:
                    if 0 <= n_idx < len(text_rows):
                        candidate_rows.append(text_rows[n_idx])
                for row_group in candidate_rows:
                    for b in row_group:
                        if score_x and b.get("x", 0.0) <= score_x:
                            continue
                        digits = re.findall(r"\d+", b.get("text", ""))
                        for d in digits:
                            try:
                                numbers.append(int(d))
                            except ValueError:
                                continue
                        if len(numbers) >= 2:
                            break
                    if len(numbers) >= 2:
                        break

            if len(numbers) < 2:
                raise ValueError(
                    f"行 {row_idx} 分数段 {score_range} 缺少复试/录取人数，识别结果: {numbers}"
                )

            # 至少有2个数字：复试人数、录取人数
            candidates = numbers[0]
            admitted = numbers[1]

            # 基本验证
            if score_lower >= score_upper:
                raise ValueError(
                    f"行 {row_idx} 分数范围无效: {score_lower} >= {score_upper}"
                )

            if admitted > candidates:
                raise ValueError(
                    f"行 {row_idx} 录取人数({admitted})大于复试人数({candidates})"
                )

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

        return score_rows

    def _save_debug_visualization(
        self,
        image_path: Path,
        img_array: np.ndarray,
        ocr_boxes: list[dict[str, Any]],
        header: TableHeader | None,
        row_groups: list[dict[str, Any]],
        score_rows: list[ScoreRow],
        failed: list[tuple[int, str]],
        output_dir: Path,
    ) -> None:
        """保存调试可视化图片，标注文本框、行分组、列分组等信息"""
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            self.logger.warning("PIL未安装，跳过调试可视化")
            return

        # 创建debug目录
        debug_dir = output_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

        # 复制图片并准备绘制
        img = Image.fromarray(img_array)
        draw = ImageDraw.Draw(img)

        # 尝试加载字体
        try:
            font = ImageFont.truetype("arial.ttf", 20)
            small_font = ImageFont.truetype("arial.ttf", 14)
        except Exception:
            font = ImageFont.load_default()
            small_font = font

        # 1. 绘制所有OCR文本框（浅灰色）
        for box in ocr_boxes:
            x_min = box.get("x_min", box.get("x", 0) - 10)
            y_min = box.get("y_min", box.get("y", 0) - 10)
            x_max = box.get("x_max", box.get("x", 0) + 10)
            y_max = box.get("y_max", box.get("y", 0) + 10)
            draw.rectangle([x_min, y_min, x_max, y_max], outline="gray", width=1)

        # 2. 绘制表头位置（蓝色）
        if header:
            y = header.header_y
            draw.line([(0, y), (img.width, y)], fill="blue", width=3)
            draw.text((10, y - 25), "TABLE HEADER", fill="blue", font=font)

            # 绘制列边界（虚线效果）
            for x in [header.col1_x_range[1], header.col2_x_range[1]]:
                for y_dash in range(0, img.height, 20):
                    draw.line([(x, y_dash), (x, min(y_dash + 10, img.height))], fill="cyan", width=2)

        # 3. 绘制行分组（绿色=成功，红色=失败）
        failed_indices = {f[0] for f in failed}
        for group in row_groups:
            idx = group["row_index"]
            boxes = group["boxes"]
            y = group["y"]

            # 计算这一行的y范围
            if boxes:
                y_min = min(b.get("y_min", b.get("y", 0) - 10) for b in boxes)
                y_max = max(b.get("y_max", b.get("y", 0) + 10) for b in boxes)
            else:
                y_min = y - 20
                y_max = y + 20

            color = "red" if idx in failed_indices else "green"
            draw.rectangle([0, y_min, img.width, y_max], outline=color, width=2)

            # 标注行号
            status = "FAIL" if idx in failed_indices else "OK"
            draw.text((5, y - 10), f"Row {idx} [{status}]", fill=color, font=small_font)

        # 4. 标注成功提取的数据（绿色文字）
        success_idx = 0
        for group in row_groups:
            idx = group["row_index"]
            if idx not in failed_indices and success_idx < len(score_rows):
                score_row = score_rows[success_idx]
                y = group["y"]
                text = f"{score_row.score_range} | {score_row.candidates}/{score_row.admitted}"
                draw.text((img.width - 300, y - 10), text, fill="darkgreen", font=small_font)
                success_idx += 1

        # 5. 标注失败原因（红色文字）
        for row_idx, reason in failed[:10]:  # 最多显示10个失败
            # 找到对应的行分组
            matching_groups = [g for g in row_groups if g["row_index"] == row_idx]
            if matching_groups:
                y = matching_groups[0]["y"]
                # 截断过长的原因
                short_reason = reason[:40] + "..." if len(reason) > 40 else reason
                draw.text((img.width - 500, y + 10), f"X {short_reason}", fill="red", font=small_font)

        # 保存图片
        output_path = debug_dir / f"{image_path.stem}_debug.png"
        img.save(output_path)
        self.logger.info(f"调试可视化已保存: {output_path}")

