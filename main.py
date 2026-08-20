import json
import os
import argparse
import time
from llm_analysis_python.evaluation.assess_func import evaluate_predictions, print_evaluation_report, save_evaluation_report
from llm_analysis_python.auto_coding_by_division import code_full_transcript, PROMPT_VERSION
from llm_analysis_python.data_clean_to_json import clean_excel_to_json
from llm_analysis_python.data_clean_to_json import CLEANING_VERSION
from llm_analysis_python.llm_config import get_model_name
from llm_analysis_python.evaluation.error_analysis import error_analysis
from llm_analysis_python.utils import pct

# 默认配置（可被命令行覆盖）
DEFAULT_EXCEL_PATH = "coded_discourse/excel/chris moon video 1 transcription_susan_5_22.xlsx"
DEFAULT_JSON_DIR = "coded_discourse/json"
DEFAULT_EVAL_DIR = "coded_discourse/evaluation"
DEFAULT_ERROR_DIR = "coded_discourse/error_analysis"
DEFAULT_FILE_ID = "chris_moon_v1"
DEFAULT_CONCURRENCY = 3

def parse_args():
    parser = argparse.ArgumentParser(description="APT 话语自动编码工具")
    parser.add_argument("--excel", default=DEFAULT_EXCEL_PATH, help="Excel 文件路径")
    parser.add_argument("--file-id", default=DEFAULT_FILE_ID, help="输出文件标识符")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="最大并发数")
    parser.add_argument("--json-dir", default=DEFAULT_JSON_DIR, help="JSON 输出目录")
    return parser.parse_args()


def auto_coding(excel_path: str, file_id: str, max_concurrency: int, json_dir: str):
    """执行完整的自动编码流程"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # 1. 检查是否有已清洗好的json， 且 excel未修改
    os.makedirs(json_dir, exist_ok=True)
    clean_json_path = os.path.join(json_dir, f"{file_id}.json")
    need_clean = True
    if os.path.exists(clean_json_path):
        excel_mtime = os.path.getmtime(excel_path)
        json_mtime = os.path.getmtime(clean_json_path)
        if json_mtime >= excel_mtime: # 检查修改时间：只有 Excel 比 JSON 新时才重新清洗
            print(f"发现已清洗数据 {clean_json_path}，跳过清洗步骤。")
            with open(clean_json_path, "r", encoding="utf-8") as f:
                result = json.load(f)
            need_clean = result.get("cleaning_version") != CLEANING_VERSION
            if need_clean:
                print("清洗规则已更新，重新清洗数据。")
    if need_clean:
        print("开始清洗数据...")
        result = clean_excel_to_json(excel_path, file_id)
        with open(clean_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"清洗后的数据已保存: {clean_json_path}")

    # 2. 自动编码教师话轮
    transcript = result["transcript"]
    gold = result["gold"]
    print("开始自动编码教师话轮...")
    predictions = code_full_transcript(transcript, max_concurrency=max_concurrency)

    # 保存预测结果
    pred_output = {
        "file_id": file_id,
        "model": get_model_name(),
        "prompt_version": PROMPT_VERSION,
        "method": "v10_immediate_response_aware_single_code_review",
        "timestamp": timestamp,
        "predictions": predictions,
    }
    prediction_dir = os.path.join(json_dir, "predictions", PROMPT_VERSION)
    os.makedirs(prediction_dir, exist_ok=True)
    pred_json_path = os.path.join(prediction_dir, f"predictions_{file_id}_{timestamp}.json")
    with open(pred_json_path, "w", encoding="utf-8") as f:
        json.dump(pred_output, f, indent=2, ensure_ascii=False)
    print(f"预测结果已保存: {pred_json_path}")

    # 3. 统计确认率
    total = len(predictions)
    reviewed_count = sum(1 for p in predictions if p.get("reviewed", False))
    review_count = sum(1 for p in predictions if p.get("needs_review", False))
    first_pass_count = total - reviewed_count - review_count

    print(f"\n--- 编码统计 ---")
    print(f"总教师话轮: {total}")
    print(f"首次确信通过: {first_pass_count} ({pct(first_pass_count, total):.1f}%)")
    print(f"经复查确认: {reviewed_count} ({pct(reviewed_count, total):.1f}%)")
    print(f"仍需人工复查: {review_count} ({pct(review_count, total):.1f}%)")

    # 4. 评估
    if gold:
        print("开始评估...")
        pred_for_eval = [
            {"turn_id": p["turn_id"], "codes": p["codes"]}
            for p in predictions
        ]
        eval_result = evaluate_predictions(gold, pred_for_eval)
        eval_result["prompt_version"] = PROMPT_VERSION
        print_evaluation_report(eval_result)
        evaluation_dir = os.path.join(DEFAULT_EVAL_DIR, PROMPT_VERSION)
        error_analysis_dir = evaluation_dir
        save_evaluation_report(eval_result, evaluation_dir, file_id, timestamp)

        # 5. 错误分析
        print("开始错误分析...")
        error_analysis(
            gold, predictions, error_analysis_dir, file_id, timestamp,
            prompt_version=PROMPT_VERSION,
        )


if __name__ == '__main__':
    args = parse_args()
    auto_coding(args.excel, args.file_id, args.concurrency, args.json_dir)