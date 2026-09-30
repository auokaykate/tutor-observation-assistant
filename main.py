"""Запуск пайплайна на dataset.json одной командой: python main.py.

Пишет output/results.json и output/report.md, затем собирает RESULTS.md
из output/report.md. Флаг --llm включает LLM-режим check_zone (нужен
ANTHROPIC_API_KEY); без ключа и сети он молча работает на правилах.
"""

import argparse
import sys
from pathlib import Path

from tutor_assistant import config
from tutor_assistant.pipeline import (
    FLAGS,
    load_json,
    run_pipeline,
    save_json,
)
from tutor_assistant.report import (
    FLAG_TITLES,
    NEWLINE,
    render_report,
    write_results_markdown,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разобрать аргументы командной строки.

    Args:
        argv: Аргументы; по умолчанию берутся из sys.argv.

    Returns:
        Разобранные аргументы.
    """
    parser = argparse.ArgumentParser(
        description="AI-помощник тьютора: прогон пайплайна на датасете."
    )
    parser.add_argument(
        "--dataset", type=Path, default=config.DATASET_PATH,
        help="путь к dataset.json",
    )
    parser.add_argument(
        "--expected", type=Path, default=config.EXPECTED_PATH,
        help="путь к ручной разметке expected.json",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=config.OUTPUT_DIR,
        help="каталог для results.json и report.md",
    )
    parser.add_argument(
        "--results-md", type=Path, default=config.RESULTS_MD_PATH,
        help="куда записать RESULTS.md",
    )
    parser.add_argument(
        "--llm", action="store_true",
        help="пробовать LLM в check_zone (нужен ANTHROPIC_API_KEY)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Прогнать пайплайн и записать отчёты.

    Args:
        argv: Аргументы командной строки.

    Returns:
        Код завершения процесса.
    """
    args = parse_args(argv)
    results = run_pipeline(
        load_json(args.dataset), load_json(args.expected), use_llm=args.llm
    )
    results_path = args.output_dir / config.RESULTS_JSON_FILENAME
    report_path = args.output_dir / config.REPORT_FILENAME
    save_json(results, results_path)
    report_path.write_text(
        render_report(results), encoding=config.FILE_ENCODING, newline=NEWLINE
    )
    write_results_markdown(report_path, args.results_md)

    print(f"Результаты: {results_path}")
    print(f"Отчёт: {report_path}")
    print(f"RESULTS.md: {args.results_md}")
    sources = ", ".join(results["settings"]["zone_sources"])
    print(f"Источник ответов check_zone: {sources}")
    for flag in FLAGS:
        data = results["summary"]["flags"][flag]
        share = data["share"] * config.PERCENT_MULTIPLIER
        print(
            f"{FLAG_TITLES[flag].capitalize()}: {data['count']} из "
            f"{data['total']} ({share:.{config.PERCENT_DIGITS}f}%)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
