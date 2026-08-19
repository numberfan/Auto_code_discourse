import json
import os
import argparse
import time
from auto_analysis.evaluation.assess_func import evaluate_predictions, print_evaluation_report, save_evaluation_report
from auto_analysis.auto_coding import code_full_transcript
from auto_analysis.data_clean_to_json import clean_excel_to_json
from auto_analysis.llm_config import get_model_name
from auto_analysis.evaluation.error_analysis import error_analysis
import auto_analysis.utils as utils

# 默认配置（可被命令行覆盖）
DEFAULT_EXCEL_PATH = "coded_discourse/excel/chris moon video 1 transcription_susan_5_22.xlsx"
DEFAULT_JSON_DIR = "coded_discourse/json"
DEFAULT_EVAL_DIR = "coded_discourse/evaluation"
DEFAULT_FILE_ID = "chris_moon_v1"
DEFAULT_CONCURRENCY = 5


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
            need_clean = False
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
        "method": "single_predict_with_confirm",
        "timestamp": timestamp,
        "predictions": predictions,
    }
    pred_json_path = os.path.join(json_dir, f"predictions_{file_id}.json")
    with open(pred_json_path, "w", encoding="utf-8") as f:
        json.dump(pred_output, f, indent=2, ensure_ascii=False)
    print(f"预测结果已保存: {pred_json_path}")

    # 3. 统计确认率
    total = len(predictions)
    confirmed_count = sum(1 for p in predictions if p.get("confirmed", False))
    review_count = sum(1 for p in predictions if p.get("needs_review", False))
<<<<<<< HEAD
    confident_first_pass = total - confirmed_count - review_count    

    print(f"\n--- 编码统计 ---")
    print(f"  总教师话轮: {total}")
    print(f"  首次确信通过: {confident_first_pass} ({utils.pct(confident_first_pass):.1f}%)")
    print(f"  经复查确认: {confirmed_count} ({utils.pct(confirmed_count):.1f}%)")
    print(f"  仍需人工复查: {review_count} ({utils.pct(review_count):.1f}%)")
=======
    confident_first_pass = total - confirmed_count - review_count

    def pct(count: int) -> float:
        return count / total * 100 if total else 0.0

    print(f"\n--- 编码统计 ---")
    print(f"  总教师话轮: {total}")
    print(f"  首次确信通过: {confident_first_pass} ({pct(confident_first_pass):.1f}%)")
    print(f"  经复查确认: {confirmed_count} ({pct(confirmed_count):.1f}%)")
    print(f"  仍需人工复查: {review_count} ({pct(review_count):.1f}%)")
>>>>>>> 82fa97e9f6019bf5a2218d1a62a9e10093bf31d5

    # 4. 评估
    if gold:
        print("开始评估...")
        pred_for_eval = [
            {"turn_id": p["turn_id"], "codes": p["codes"]}
            for p in predictions
        ]
        eval_result = evaluate_predictions(gold, pred_for_eval)
        print_evaluation_report(eval_result)
        save_evaluation_report(eval_result, DEFAULT_EVAL_DIR, file_id, timestamp)

        # 5. 错误分析
        print("开始错误分析...")
        error_analysis(gold, predictions, DEFAULT_EVAL_DIR, file_id, timestamp)


if __name__ == '__main__':
    args = parse_args()
    auto_coding(args.excel, args.file_id, args.concurrency, args.json_dir)