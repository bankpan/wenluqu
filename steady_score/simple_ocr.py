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

from .metadata import SchoolMetadata
from .validators import ScoreRow


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

        ocr_boxes = self._ocr_boxes(img_array)
        self.logger.info(f"识别到 {len(ocr_boxes)} 个文本框")

        # 4. 按Y坐标分组成行
        text_rows = self._group_into_rows(ocr_boxes)
        self.logger.info(f"分组为 {len(text_rows)} 行")

        # 5. 提取表格数据行
        score_rows = self._extract_score_rows(text_rows)
        self.logger.info(f"提取到 {len(score_rows)} 行有效数据")

        return score_rows

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
                    x_center = float(np.mean(poly[:, 0]))
                    y_center = float(np.mean(poly[:, 1]))
                    conf = scores[i] if i < len(scores) else 1.0
                    boxes.append({"text": text.strip(), "x": x_center, "y": y_center, "conf": conf})
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
        return max(
            candidates,
            key=lambda line: (
                self._pattern_conf(line["boxes"], self.SCHOOL_PATTERN),
                len(line["norm"]),
            ),
        )

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
        base = text.replace("省", "").replace("市", "").strip()
        if not base or base not in self.PROVINCES:
            return ""
        # 直辖市用“市”，自治区/特别行政区不用后缀，其余补“省”
        if base in {"北京", "天津", "上海", "重庆"}:
            return f"{base}市"
        if base in {"广西", "内蒙古", "宁夏", "新疆", "西藏", "香港", "澳门", "台湾"}:
            return base
        return f"{base}省"

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
