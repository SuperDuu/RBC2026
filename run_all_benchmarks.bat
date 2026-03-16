@echo off
setlocal enabledelayedexpansion

REM ==============================================================================
REM 1. SpearHead_HighPerformance (YOLOv8n INT8)
REM ==============================================================================
set DIR_HP=.\SpearHead_HighPerformance\benchmark_report_yolov8n
set MODEL_HP=.\SpearHead_HighPerformance\models\best_openvino_model_int8\best_int8.xml

echo [1/4] Running YOLOv8n CPU Benchmark...
if not exist "%DIR_HP%\cpu" mkdir "%DIR_HP%\cpu"
benchmark_app ^
  -m "%MODEL_HP%" ^
  -d CPU ^
  -api sync ^
  -niter 100 ^
  -report_type detailed_counters ^
  -report_folder "%DIR_HP%\cpu"

echo [2/4] Running YOLOv8n GPU Benchmark...
if not exist "%DIR_HP%\gpu" mkdir "%DIR_HP%\gpu"
benchmark_app ^
  -m "%MODEL_HP%" ^
  -d GPU ^
  -api sync ^
  -niter 100 ^
  -report_type detailed_counters ^
  -report_folder "%DIR_HP%\gpu"


REM ==============================================================================
REM 2. SpearHead_YOLO26 (YOLO26n INT8)
REM ==============================================================================
set DIR_YOLO26=.\SpearHead_YOLO26\benchmark_report_yolo26n
set MODEL_YOLO26=.\SpearHead_YOLO26\models\yolo26n_openvino_int8\best_int8.xml

echo [3/4] Running YOLO26n CPU Benchmark...
if not exist "%DIR_YOLO26%\cpu" mkdir "%DIR_YOLO26%\cpu"
benchmark_app ^
  -m "%MODEL_YOLO26%" ^
  -d CPU ^
  -api sync ^
  -niter 100 ^
  -report_type detailed_counters ^
  -report_folder "%DIR_YOLO26%\cpu"

echo [4/4] Running YOLO26n GPU Benchmark...
if not exist "%DIR_YOLO26%\gpu" mkdir "%DIR_YOLO26%\gpu"
benchmark_app ^
  -m "%MODEL_YOLO26%" ^
  -d GPU ^
  -api sync ^
  -niter 100 ^
  -report_type detailed_counters ^
  -report_folder "%DIR_YOLO26%\gpu"

echo =================================================
echo All done! Reports saved to:
echo  - %DIR_HP%
echo  - %DIR_YOLO26%
pause
