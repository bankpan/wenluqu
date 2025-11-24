from __future__ import annotations

import inspect
import logging
import traceback
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, List

import numpy as np
import pandas as pd
import paddle
from PIL import Image
from tqdm import tqdm

# Default to PP-StructureV2; fall back to other exports if V2不可用
DEFAULT_STRUCTURE_VERSION = "PP-StructureV2"
try:  # PaddleOCR version compatibility
    # Prefer the lightweight V2 table pipeline to avoid heavy DocLayout models.
    from paddleocr import PPStructureV2 as PPStructure
except ImportError:  # pragma: no cover - fallback for new versions
    try:
        from paddleocr import PPStructure
    except ImportError:
        from paddleocr import PPStructureV3 as PPStructure

from .config import OCRConfig
from .simple_ocr import SimpleOCRExtractor
from .table_parser import parse_table_html
from .validators import (
    INTEGER_PATTERN,
    RowValidationError,
    SCORE_RANGE_PATTERN,
    ScoreRow,
    parse_integer,
    parse_score_range,
    validate_row,
)

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"}


@dataclass(slots=True)
class ImageProcessResult:
    image_path: Path
    csv_path: Path | None
    rows: list[ScoreRow]
    errors: list[str]
    status: str


class OCRProcessor:
    _ENGINE_CACHE: dict[tuple[Any, ...], Any] = {}

    def __init__(self, config: OCRConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger
        # Initialize simple OCR extractor (new simpler method)
        self.simple_extractor = SimpleOCRExtractor(lang=config.lang, logger=logger)
        logger.info("使用简单OCR方法（基础OCR + 正则匹配）")

    def _patch_paddle_analysis_config(self) -> None:
        """兼容旧版本 Paddle 中缺失的 set_optimization_level 接口。"""
        try:
            from paddle.base import libpaddle
        except Exception:  # noqa: BLE001
            return
        analysis_cls = getattr(libpaddle, "AnalysisConfig", None)
        if analysis_cls is None:
            return
        if hasattr(analysis_cls, "set_optimization_level"):
            return

        def _set_optimization_level(self, *_, **__):  # pragma: no cover - runtime patch
            # 旧版 AnalysisConfig 不支持该接口，静默吞掉调用即可
            return None

        try:
            analysis_cls.set_optimization_level = _set_optimization_level  # type: ignore[attr-defined]
            self.logger.warning(
                "Paddle AnalysisConfig 缺少 set_optimization_level，已注入兼容实现。"
            )
        except Exception:  # noqa: BLE001
            self.logger.warning(
                "Paddle AnalysisConfig 兼容补丁注入失败，可能仍会触发 set_optimization_level 错误。",
                exc_info=True,
            )

    def _get_or_create_engine(self, lang: str):
        kwargs = self._build_engine_kwargs(lang)
        key = (lang, tuple(sorted(kwargs.items())))
        cached = self._ENGINE_CACHE.get(key)
        if cached is not None:
            self.logger.info("复用已加载的 PPStructure 引擎。")
            return cached
        engine = self._create_engine(kwargs)
        self._ENGINE_CACHE[key] = engine
        return engine

    def _build_engine_kwargs(self, lang: str) -> dict[str, Any]:
        # 尽量使用 PP-StructureV2（不依赖 PaddleX 大模型）
        try:
            supported = set(inspect.signature(PPStructure.__init__).parameters)
        except (ValueError, TypeError):
            supported = {"self"}
        supported.discard("self")

        preferred: dict[str, Any] = {
            "lang": lang,
            "structure_version": DEFAULT_STRUCTURE_VERSION or "PP-StructureV2",
            "layout": False,
            "table": True,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "use_seal_recognition": False,
            "use_formula_recognition": False,
            "use_chart_recognition": False,
            "use_region_detection": False,
        }
        return {
            name: value
            for name, value in preferred.items()
            if name in supported
        }

    def _create_engine(self, kwargs: dict[str, Any]):
        engine_name = getattr(PPStructure, "__name__", "PPStructure")
        self.logger.info("创建 %s 引擎参数: %s", engine_name, kwargs)
        # Force disable layout analysis to prevent crashes
        safe_kwargs = {"layout": False, "table": True, "show_log": False}
        safe_kwargs.update(kwargs)

        try:
            engine = PPStructure(**safe_kwargs)
            self.logger.info("PPStructure 引擎创建成功，layout已禁用")
        except TypeError:  # 某些版本不支持上述参数，回退为无参初始化
            self.logger.warning("PPStructure 参数不兼容，已回退至默认构造。", exc_info=True)
            try:
                engine = PPStructure(layout=False, table=True, show_log=False)
            except:
                engine = PPStructure()

        instance_name = type(engine).__name__
        if "V3" in instance_name:
            self.logger.warning(
                "当前加载的是 PPStructureV3，可能较慢且在手机截图上易截断表格，"
                "建议安装 paddleocr<=2.7.0.3 以启用轻量 PPStructureV2。"
            )
        return engine

    def process_directory(
        self,
        progress_callback: Callable[[int, int, Path], None] | None = None,
    ) -> list[ImageProcessResult]:
        image_paths = sorted(
            path for path in self.config.image_dir.iterdir()
            if path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not image_paths:
            raise FileNotFoundError(f"未在 {self.config.image_dir} 找到图片")

        self.config.raw_dir.mkdir(parents=True, exist_ok=True)
        results: list[ImageProcessResult] = []

        total = len(image_paths)
        iterator: Iterable[Path]
        if progress_callback is None:
            iterator = tqdm(image_paths, desc="识别进度", unit="张")
        else:
            iterator = image_paths

        for idx, image_path in enumerate(iterator, start=1):
            result = self._process_single_image(image_path)
            results.append(result)
            if progress_callback is not None:
                progress_callback(idx, total, image_path)
        return results

    def _process_single_image(self, image_path: Path) -> ImageProcessResult:
        errors: list[str] = []
        csv_path: Path | None = None
        normalized_rows: list[ScoreRow] = []
        try:
            # Use simple OCR extractor - much simpler and more reliable
            self.logger.info("[%s] 使用简单OCR方法识别", image_path.name)
            normalized_rows = self.simple_extractor.extract_table_from_image(image_path)
            
            if normalized_rows:
                # Write CSV
                csv_path = self.config.raw_dir / f"{image_path.stem}.csv"
                self.config.raw_dir.mkdir(parents=True, exist_ok=True)
                
                import pandas as pd
                df = pd.DataFrame([
                    {
                        "score_range": row.score_range,
                        "lower": row.lower,
                        "upper": row.upper,
                        "candidates": row.candidates,
                        "admitted": row.admitted,
                    }
                    for row in normalized_rows
                ])
                df.to_csv(csv_path, index=False, encoding="utf-8-sig")
                
                status = "success"
                self.logger.info(
                    "[%s] 成功识别 %d 行，输出: %s",
                    image_path.name,
                    len(normalized_rows),
                    csv_path,
                )
            else:
                status = "no_data"
                errors.append("未识别到有效表格数据")
                self.logger.warning("[%s] %s", image_path.name, errors[-1])
        except Exception as exc:  # noqa: BLE001
            status = "failed"
            msg = f"识别失败: {exc}"
            errors.append(msg)
            self.logger.exception("[%s] %s", image_path.name, msg)

        return ImageProcessResult(
            image_path=image_path,
            csv_path=csv_path,
            rows=normalized_rows,
            errors=errors,
            status=status,
        )

    def _run_engine(self, image: np.ndarray) -> Iterable[dict]:
        engine = self.engine
        candidates: list[tuple[str, Callable[[np.ndarray], Any]]] = []
        if callable(engine):
            candidates.append(("__call__", engine))
        for name in ("predict", "run", "run_ocr", "infer", "process"):
            method = getattr(engine, name, None)
            if callable(method):
                candidates.append((name, method))

        if not candidates:
            raise TypeError("当前 PaddleOCR 引擎缺少可调用或 predict 接口，无法运行识别。")

        last_error: Exception | None = None
        for name, method in candidates:
            try:
                result = method(image)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue

            normalized = self._normalize_engine_output(result)
            if normalized is not None:
                if name != "__call__":
                    self.logger.debug("通过 engine.%s() 兼容调用 OCR 引擎。", name)
                return normalized
            last_error = TypeError(f"engine.{name} 返回未知数据格式: {type(result)!r}")

        raise RuntimeError("OCR 引擎调用失败，详见日志中的堆栈信息。") from last_error

    def _gather_table_rows(self, image: np.ndarray) -> list[list[str]]:
        crops = self._generate_crops(image)
        aggregated: list[list[str]] = []
        seen: set[tuple[str, ...]] = set()
        for crop in crops:
            results = self._run_engine(crop)
            rows = self._extract_rows(results)
            for row in rows:
                key = tuple(row)
                if len(row) >= 3 and key not in seen:
                    seen.add(key)
                    aggregated.append(row)
        return aggregated


    def _generate_crops(self, image: np.ndarray) -> list[np.ndarray]:
        # Use full image only to reduce OCR engine crashes
        # Multiple crops can cause PaddleOCR memory issues
        return [image]


    def _normalize_engine_output(self, result: Any) -> list[dict] | None:
        if result is None:
            return []
        normalized = self._flatten_engine_output(result)
        if normalized is not None:
            return normalized
        return None

    def _flatten_engine_output(self, payload: Any) -> list[dict] | None:
        if payload is None:
            return []
        if isinstance(payload, list):
            if not payload:
                return []
            if all(self._is_structure_block(item) for item in payload if isinstance(item, Mapping)):
                return payload  # type: ignore[return-value]
            flattened: list[dict] = []
            for item in payload:
                converted = self._flatten_engine_output(item)
                if converted is not None:
                    flattened.extend(converted)
            return flattened if flattened else []
        if isinstance(payload, tuple):
            flattened: list[dict] = []
            for item in payload:
                converted = self._flatten_engine_output(item)
                if converted is not None:
                    flattened.extend(converted)
            return flattened if flattened else []
        if isinstance(payload, Mapping):
            if self._is_structure_block(payload):
                return [payload]  # type: ignore[list-item]
            tables = self._convert_ppstructure_v3_tables(payload)
            if tables is not None:
                return tables
            for key in ("structure_res", "result", "res", "data"):
                if key in payload:
                    converted = self._flatten_engine_output(payload[key])
                    if converted is not None:
                        return converted
            return None
        return None

    def _is_structure_block(self, block: Mapping[str, Any]) -> bool:
        return "type" in block and isinstance(block.get("res"), Mapping)

    def _convert_ppstructure_v3_tables(self, payload: Mapping[str, Any]) -> list[dict] | None:
        if "table_res_list" not in payload:
            return None
        table_list = payload.get("table_res_list") or []
        converted: list[dict] = []
        for item in table_list:
            table_block = self._convert_single_table(item)
            if table_block is not None:
                converted.append(table_block)
        return converted

    def _convert_single_table(self, table: Any) -> dict | None:
        if not isinstance(table, Mapping):
            return None
        html = self._extract_table_html(table)
        if not html:
            return None
        block: dict[str, Any] = {"type": "table", "res": {"html": html}}
        bbox = (
            table.get("table_box")
            or table.get("table_bbox")
            or table.get("bbox")
            or table.get("coordinate")
        )
        if bbox is not None:
            block["bbox"] = bbox
        if "table_region_id" in table:
            block["id"] = table["table_region_id"]
        if "cell_box_list" in table:
            block["res"]["cell_box_list"] = table["cell_box_list"]
        return block

    def _extract_table_html(self, table: Mapping[str, Any]) -> str | None:
        html = table.get("pred_html")
        if isinstance(html, str) and html:
            return html
        html_field = table.get("html")
        if isinstance(html_field, Mapping):
            candidate = html_field.get("pred") or html_field.get("html")
            if isinstance(candidate, str) and candidate:
                return candidate
        res = table.get("res")
        if isinstance(res, Mapping):
            candidate = res.get("html") or res.get("pred_html")
            if isinstance(candidate, str) and candidate:
                return candidate
        return None

    def _prepare_image(self, image_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
        with Image.open(image_path) as img:
            rgb = img.convert("RGB")
            metadata = self._extract_metadata(rgb)
            table_region = self._crop_table_region(rgb)
            cropped = self._auto_crop(table_region)
            return np.array(cropped), metadata

    def _auto_crop(self, image: Image.Image) -> Image.Image:
        gray = image.convert("L")
        np_gray = np.array(gray)
        mask = np_gray < self.config.binarize_threshold

        coords = np.argwhere(mask)
        if coords.size == 0:
            return image

        top = max(int(coords[:, 0].min()) - self.config.crop_margin, 0)
        bottom = min(int(coords[:, 0].max()) + self.config.crop_margin, image.height)
        left = max(int(coords[:, 1].min()) - self.config.crop_margin, 0)
        right = min(int(coords[:, 1].max()) + self.config.crop_margin, image.width)
        return image.crop((left, top, right, bottom))

    def _crop_table_region(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        gray = np.array(image.convert("L"))
        mask = gray < 240
        row_mask = mask.any(axis=1)
        col_mask = mask.any(axis=0)
        if not row_mask.any() or not col_mask.any():
            return image
        row_indices = np.where(row_mask)[0]
        col_indices = np.where(col_mask)[0]
        top = max(row_indices[0] - 20, 0)
        bottom = min(row_indices[-1] + 20, height)
        left = max(col_indices[0] - 20, 0)
        right = min(col_indices[-1] + 20, width)
        return image.crop((left, top, right, bottom))

    def _extract_rows(self, table_results: Iterable[dict]) -> List[List[str]]:
        rows: list[list[str]] = []
        for block in table_results:
            if block.get("type") != "table":
                continue
            html = block.get("res", {}).get("html")
            if not html:
                continue
            rows.extend(parse_table_html(html))
        if not rows:
            return rows

        flat_cells: list[str] = []
        for row in rows:
            for cell in row:
                text = (cell or "").strip()
                if text:
                    flat_cells.append(text)

        structured: list[list[str]] = []
        idx = 0
        while idx < len(flat_cells):
            cell = flat_cells[idx]
            try:
                parse_score_range(cell)
            except RowValidationError:
                idx += 1
                continue

            # Skip if this cell contains multiple score ranges (e.g., "231分-235分 236分-240分")
            if cell.count("分-") > 1 or cell.count("分～") > 1:
                self.logger.info(f"Skipping merged score range: {cell}")
                idx += 1
                continue

            score_text = cell
            values: list[int] = []
            j = idx + 1
            while j < len(flat_cells):
                candidate = flat_cells[j]
                if SCORE_RANGE_PATTERN.search(candidate) and len(values) < 2:
                    break
                if INTEGER_PATTERN.search(candidate):
                    values.append(parse_integer(candidate))
                    if len(values) == 2:
                        j += 1
                        break
                j += 1
            if len(values) == 2:
                structured.append(
                    [score_text, str(values[0]), str(values[1])]
                )
            idx = j if j > idx else idx + 1

        return structured

    def _normalize_rows(self, raw_rows: list[list[str]], errors: list[str]) -> list[ScoreRow]:
        normalized: list[ScoreRow] = []
        for idx, row in enumerate(raw_rows):
            try:
                normalized.append(validate_row(row))
            except RowValidationError as err:
                errors.append(f"行{idx+1}: {err}")
        return normalized

    def _write_csv(
        self,
        image_path: Path,
        rows: list[ScoreRow],
        metadata: dict[str, Any],
    ) -> Path:
        df = pd.DataFrame([asdict(r) for r in rows])
        if metadata:
            for key, value in metadata.items():
                df[key] = value
        output_name = image_path.with_suffix(".csv").name
        csv_path = self.config.raw_dir / output_name
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        return csv_path

    def _extract_metadata(self, image: Image.Image) -> dict[str, Any]:
        width, height = image.size
        header_box = (0, 0, width, int(height * 0.3))
        header = image.crop(header_box)
        text = self._run_text_ocr(header)
        school = None
        if text:
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            if lines:
                school = lines[0]
        return {"school": school} if school else {}

    def _run_text_ocr(self, image: Image.Image) -> str:
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            self.logger.warning("未找到 PaddleOCR，无法提取学校名称。")
            return ""
        if not hasattr(self, "_header_ocr"):
            self._header_ocr = PaddleOCR(lang="ch", use_angle_cls=True)
        result = self._header_ocr.ocr(np.array(image))
        texts: list[str] = []
        for line in result[0] if result else []:
            texts.append(line[1][0])
        return "\n".join(texts)
