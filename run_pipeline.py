"""
run_pipeline.py
============================================================
Мастер-скрипт: запускает весь пайплайн этап за этапом.

Этапы:
  0. (опционально) extract_video_meta.py — проверка параметров видео
  1. auto_label.py — авторазметка из GT (уже сделано, генерирует labels_2d.csv)
  2. prepare_dataset.py — извлечение кадров, train/val split (уже сделано)
  3. train_yolo.py — обучение YOLO (уже сделано, есть best.pt)
  4. run_inference.py — инференс на 5 видео -> detections.csv
  5. triangulate.py — триангуляция 3D -> trajectory_raw.csv
  6. postprocess.py — сглаживание -> trajectory_smooth.csv
  7. validate.py — валидация против GT
  8. visualize.py — видео с наложениями, 3D GIF, графики ошибок

Запуск:
  python run_pipeline.py              # все этапы 4-8
  python run_pipeline.py --from 4     # с этапа 4 (инференс)
  python run_pipeline.py --only 6     # только этап 6 (постобработка)
  python run_pipeline.py --list       # показать этапы
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STAGES = {
    0: ("extract_video_meta", "pipeline/extract_video_meta.py", "Проверка параметров видео"),
    1: ("auto_label", "auto_label.py", "Авторазметка из GT (labels_2d.csv + yolo_labels/)"),
    2: ("prepare_dataset", "prepare_dataset.py", "Извлечение кадров, train/val split (dataset/)"),
    3: ("train_yolo", "train_yolo.py", "Обучение YOLOv8n (best.pt)"),
    4: ("run_inference", "run_inference.py", "Инференс на 5 видео (detections.csv)"),
    5: ("triangulate", "triangulate.py", "Триангуляция 3D (trajectory_raw.csv)"),
    6: ("postprocess", "postprocess.py", "Сглаживание траектории (trajectory_smooth.csv)"),
    7: ("validate", "validate.py", "Валидация против GT (метрики, репроекция)"),
    8: ("visualize", "visualize.py", "Визуализация: видео, 3D GIF, графики"),
}


def run_stage(script: str, args: list[str] = None) -> bool:
    """Запускает этап как подпроцесс."""
    script_path = ROOT / script
    if not script_path.exists():
        # Проверяем в pipeline/
        script_path = ROOT / "pipeline" / script
        if not script_path.exists():
            print(f"  ❌ Скрипт не найден: {script}")
            return False

    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)

    print(f"  Запуск: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, cwd=ROOT, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Ошибка (код {e.returncode})")
        return False
    except KeyboardInterrupt:
        print(f"  ⚠ Прервано пользователем")
        return False


def check_prerequisites(stage: int) -> bool:
    """Проверяет, готовы ли входные данные для этапа."""
    checks = {
        4: [ROOT / "runs" / "peleng_detect" / "weights" / "best.pt"],
        5: [ROOT / "runs" / "detections" / "all_detections.csv"],
        6: [ROOT / "runs" / "trajectory" / "trajectory_raw.csv"],
        7: [ROOT / "runs" / "trajectory" / "trajectory_smooth.csv",
            ROOT / "runs" / "trajectory" / "trajectory_raw.csv"],
        8: [ROOT / "runs" / "validation" / "validation_report.csv"],
    }
    for path in checks.get(stage, []):
        if not path.exists():
            print(f"  ❌ Не найден файл для этапа {stage}: {path}")
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Run full ML pipeline")
    parser.add_argument("--from", type=int, dest="start", default=4,
                        help="Начать с этапа (default: 4 = inference)")
    parser.add_argument("--to", type=int, default=8, help="Закончить на этапе (default: 8)")
    parser.add_argument("--only", type=int, default=None, help="Запустить только один этап")
    parser.add_argument("--list", action="store_true", help="Показать список этапов")
    parser.add_argument("--force", action="store_true", help="Игнорировать проверки пререквизитов")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence для инференса")
    parser.add_argument("--draw", action="store_true", help="Рисовать боксы на видео (инференс)")
    parser.add_argument("--no-videos", action="store_true", help="Пропустить создание видео в визуализации")
    parser.add_argument("--no-3d", action="store_true", help="Пропустить 3D анимацию")
    parser.add_argument("--no-plots", action="store_true", help="Пропустить графики ошибок")
    parser.add_argument("--max-frames", type=int, default=None, help="Лимит кадров для видео наложений")

    args = parser.parse_args()

    if args.list:
        print("\nДоступные этапы:")
        for num, (name, script, desc) in STAGES.items():
            marker = " →" if num >= 4 else " ✓"  # 0-3 уже сделаны
            print(f"  {num}: {name:20s} {desc}{marker}")
        print("\nПримеры:")
        print("  python run_pipeline.py              # этапы 4-8")
        print("  python run_pipeline.py --from 5     # с триангуляции")
        print("  python run_pipeline.py --only 7     # только валидация")
        return

    if args.only is not None:
        start = end = args.only
    else:
        start = args.start
        end = args.to

    print(f"\n{'='*60}")
    print(f"ML PIPELINE: этапы {start} – {end}")
    print(f"{'='*60}\n")

    success = True
    for stage in range(start, end + 1):
        if stage not in STAGES:
            print(f"⚠ Этап {stage} не определён, пропуск")
            continue

        name, script, desc = STAGES[stage]
        print(f"\n[{stage}] {name.upper()} — {desc}")

        # Проверка пререквизитов
        if not args.force and not check_prerequisites(stage):
            print(f"  ⏭ Пропуск (нет входных данных). Используйте --force для принудительного запуска.")
            success = False
            break

        # Аргументы для конкретных этапов
        stage_args = []
        if stage == 4:
            stage_args = ["--conf", str(args.conf)]
            if args.draw:
                stage_args.append("--draw")
        elif stage == 8:
            if args.no_videos:
                stage_args.append("--no-videos")
            if args.no_3d:
                stage_args.append("--no-3d")
            if args.no_plots:
                stage_args.append("--no-plots")
            if args.max_frames:
                stage_args.extend(["--max-frames", str(args.max_frames)])

        if run_stage(script, stage_args):
            print(f"  ✅ Этап {stage} завершён успешно")
        else:
            print(f"  ❌ Этап {stage} завершён с ошибкой")
            success = False
            break

    print(f"\n{'='*60}")
    if success:
        print(f"🎉 ПАЙПЛАЙН ЗАВЕРШЁН УСПЕШНО (этапы {start}-{end})")
    else:
        print(f"💥 ПАЙПЛАЙН ОСТАНОВЛЕН С ОШИБКОЙ")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()