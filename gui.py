from __future__ import annotations

import os
import shlex
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

from steady_score.config import PAVAConfig
from steady_score.pava import compute_steady_score, load_buckets
from steady_score.reporting import write_report


class SteadyScoreGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("考研稳录取分数线工具")
        self.geometry("860x620")
        self.ocr_image_dir = tk.StringVar()
        self.ocr_output_dir = tk.StringVar()
        self.ocr_lang = tk.StringVar(value="ch")

        self.calc_csv_dir = tk.StringVar()
        self.calc_output_excel = tk.StringVar()
        self.calc_threshold = tk.StringVar(value="0.75")
        self.calc_min_bin = tk.StringVar(value="8")

        self.status_var = tk.StringVar(value="ready")
        self._running_task = None
        self._current_proc: subprocess.Popen | None = None
        self.project_root = Path(__file__).resolve().parent
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build_layout()

    # UI building ---------------------------------------------------------
    def _build_layout(self) -> None:
        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # OCR tab
        ocr_frame = ttk.Frame(notebook)
        notebook.add(ocr_frame, text="1. OCR 识别")

        self._add_path_row(
            parent=ocr_frame,
            row=0,
            label="图片目录:",
            var=self.ocr_image_dir,
            is_directory=True,
        )
        self._add_path_row(
            parent=ocr_frame,
            row=1,
            label="输出目录:",
            var=self.ocr_output_dir,
            is_directory=True,
        )
        ttk.Label(ocr_frame, text="语言:").grid(row=2, column=0, sticky="e", pady=5, padx=5)
        ttk.Combobox(
            ocr_frame,
            textvariable=self.ocr_lang,
            values=["ch", "en", "chinese_cht"],
            state="readonly",
        ).grid(row=2, column=1, sticky="ew", padx=5)

        run_ocr_btn = ttk.Button(ocr_frame, text="运行OCR", command=self._start_ocr_task)
        run_ocr_btn.grid(row=3, column=0, columnspan=3, pady=15)

        for i in range(3):
            ocr_frame.grid_columnconfigure(i, weight=1)

        # PAVA tab
        calc_frame = ttk.Frame(notebook)
        notebook.add(calc_frame, text="2. 稳录取计算")

        self._add_path_row(
            parent=calc_frame,
            row=0,
            label="CSV目录:",
            var=self.calc_csv_dir,
            is_directory=True,
        )
        self._add_path_row(
            parent=calc_frame,
            row=1,
            label="Excel输出:",
            var=self.calc_output_excel,
            is_directory=False,
            filetypes=[("Excel 文件", "*.xlsx")],
        )

        ttk.Label(calc_frame, text="阈值T:").grid(row=2, column=0, sticky="e", padx=5, pady=5)
        ttk.Entry(calc_frame, textvariable=self.calc_threshold).grid(row=2, column=1, sticky="ew", padx=5)
        ttk.Label(calc_frame, text="最小样本:").grid(row=3, column=0, sticky="e", padx=5, pady=5)
        ttk.Entry(calc_frame, textvariable=self.calc_min_bin).grid(row=3, column=1, sticky="ew", padx=5)

        run_calc_btn = ttk.Button(calc_frame, text="运行PAVA计算", command=self._start_calc_task)
        run_calc_btn.grid(row=4, column=0, columnspan=3, pady=15)

        for i in range(3):
            calc_frame.grid_columnconfigure(i, weight=1)

        # Log panel
        log_frame = ttk.Frame(self)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        ttk.Label(log_frame, text="执行日志:").pack(anchor="w")
        self.log_widget = scrolledtext.ScrolledText(log_frame, height=12, state="disabled")
        self.log_widget.pack(fill=tk.BOTH, expand=True)

        ttk.Label(self, textvariable=self.status_var, anchor="w").pack(fill=tk.X, padx=10, pady=(0, 10))

    def _add_path_row(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        var: tk.StringVar,
        is_directory: bool,
        filetypes: list[tuple[str, str]] | None = None,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="e", padx=5, pady=5)
        entry = ttk.Entry(parent, textvariable=var)
        entry.grid(row=row, column=1, sticky="ew", padx=5, pady=5)
        ttk.Button(
            parent,
            text="浏览",
            command=lambda: self._choose_path(var, is_directory=is_directory, filetypes=filetypes),
        ).grid(row=row, column=2, sticky="w", padx=5, pady=5)

    def _choose_path(
        self,
        var: tk.StringVar,
        is_directory: bool,
        filetypes: list[tuple[str, str]] | None = None,
    ) -> None:
        if is_directory:
            selected = filedialog.askdirectory()
        else:
            selected = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=filetypes)
        if selected:
            var.set(selected)

    # Logging helpers -----------------------------------------------------
    def _append_log(self, message: str) -> None:
        def write():
            self.log_widget.configure(state="normal")
            self.log_widget.insert(tk.END, message + "\n")
            self.log_widget.see(tk.END)
            self.log_widget.configure(state="disabled")

        self.after(0, write)

    def _set_status(self, message: str) -> None:
        self.after(0, lambda: self.status_var.set(message))

    def _show_info(self, title: str, message: str) -> None:
        self.after(0, lambda: messagebox.showinfo(title, message))

    def _show_warning(self, title: str, message: str) -> None:
        self.after(0, lambda: messagebox.showwarning(title, message))

    def _show_error(self, title: str, message: str) -> None:
        self.after(0, lambda: messagebox.showerror(title, message))

    def _run_in_thread(self, target) -> None:
        if self._running_task:
            messagebox.showinfo("执行中", "已有任务在运行，请稍候。")
            return
        self._set_status("执行中...")
        self._running_task = threading.Thread(target=self._wrap_task(target), daemon=True)
        self._running_task.start()

    def _wrap_task(self, target):
        def runner():
            try:
                target()
            finally:
                self._running_task = None
                self._set_status("ready")

        return runner

    # Subprocess helpers -------------------------------------------------
    def _run_cli_command(
        self,
        command: list[str | os.PathLike[str]],
        task_label: str,
        capture_keys: list[str] | None = None,
    ) -> dict[str, str]:
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        capture = capture_keys or []
        summary_hits: dict[str, str] = {}
        cmd_list = [str(part) for part in command]
        cmd_display = shlex.join(cmd_list)
        self._append_log(f"[{task_label}] 命令: {cmd_display}")
        try:
            self._current_proc = subprocess.Popen(
                cmd_list,
                cwd=str(self.project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
            )
            assert self._current_proc.stdout is not None
            for raw_line in self._current_proc.stdout:
                line = raw_line.rstrip()
                if not line:
                    continue
                self._append_log(line)
                for key in capture:
                    if key in line:
                        summary_hits[key] = line
            self._current_proc.wait()
            if self._current_proc.returncode != 0:
                raise RuntimeError(f"{task_label} 失败，退出码 {self._current_proc.returncode}")
        except FileNotFoundError as exc:
            raise RuntimeError(f"无法启动 {task_label} 命令: {exc}") from exc
        finally:
            self._current_proc = None
        return summary_hits

    # OCR task ------------------------------------------------------------
    def _start_ocr_task(self) -> None:
        self._run_in_thread(self._run_ocr)

    def _run_ocr(self) -> None:
        try:
            image_dir = Path(self.ocr_image_dir.get())
            output_dir = Path(self.ocr_output_dir.get())
            if not image_dir.exists():
                raise FileNotFoundError(f"图片目录不存在: {image_dir}")
            output_dir.mkdir(parents=True, exist_ok=True)

            lang = (self.ocr_lang.get() or "ch").strip() or "ch"
            python_exec = sys.executable or "python"
            command = [
                python_exec,
                self.project_root / "tool.py",
                "process-images",
                image_dir,
                output_dir,
                "--lang",
                lang,
            ]
            summary = self._run_cli_command(
                command,
                task_label="OCR",
                capture_keys=["处理完成", "CSV输出目录", "错误清单"],
            )
            message_lines = [
                summary[key]
                for key in ("处理完成", "CSV输出目录", "错误清单")
                if key in summary
            ]
            info_message = "\n".join(message_lines) if message_lines else "OCR命令执行完成，详情见日志。"
            self._show_info("OCR完成", info_message)
        except Exception as exc:  # noqa: BLE001
            self._append_log(f"[OCR] 错误: {exc}")
            self._show_error("OCR失败", str(exc))

    # Calculation task ----------------------------------------------------
    def _start_calc_task(self) -> None:
        self._run_in_thread(self._run_calc)

    def _run_calc(self) -> None:
        try:
            csv_dir = Path(self.calc_csv_dir.get())
            excel_path = Path(self.calc_output_excel.get())
            if not csv_dir.exists():
                raise FileNotFoundError(f"CSV目录不存在: {csv_dir}")
            if not excel_path.parent.exists():
                excel_path.parent.mkdir(parents=True, exist_ok=True)

            threshold = float(self.calc_threshold.get() or 0.75)
            min_bin = int(float(self.calc_min_bin.get() or 8))
            config = PAVAConfig(target_ratio=threshold, min_bin_size=min_bin, allow_fallback=True)

            csv_files = sorted(csv_dir.glob("*.csv"))
            if not csv_files:
                raise FileNotFoundError(f"在 {csv_dir} 未找到CSV文件")

            results = []
            skipped = []
            self._append_log("[PAVA] 开始计算")
            for csv_file in csv_files:
                school = csv_file.stem
                try:
                    buckets = load_buckets(csv_file)
                    result = compute_steady_score(school, buckets, config)
                    results.append(result)
                    if result.warnings:
                        self._append_log(f"[PAVA] {school} 警告: {'; '.join(result.warnings)}")
                except Exception as exc:  # noqa: BLE001
                    skipped.append(f"{school}: {exc}")
                    self._append_log(f"[PAVA] {school} 失败: {exc}")

            if not results:
                raise RuntimeError("没有成功的院校可写入报告")

            write_report(results, excel_path)
            self._append_log(f"[PAVA] 报告生成: {excel_path}")
            if skipped:
                self._show_warning("计算完成", f"部分院校失败: {'; '.join(skipped)}")
            else:
                self._show_info("计算完成", f"报告已生成: {excel_path}")
        except Exception as exc:  # noqa: BLE001
            self._append_log(f"[PAVA] 错误: {exc}")
            self._show_error("计算失败", str(exc))

    # Lifecycle ----------------------------------------------------------
    def _terminate_current_proc(self) -> None:
        proc = self._current_proc
        if proc is None:
            return
        if proc.poll() is not None:
            return
        try:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        except Exception:
            # Swallow termination errors; process may have exited.
            pass
        finally:
            self._current_proc = None

    def _on_close(self) -> None:
        self._terminate_current_proc()
        self.destroy()


def main() -> None:
    app = SteadyScoreGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
