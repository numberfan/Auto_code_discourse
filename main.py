import json
import os
import argparse

from auto_analysis.assess_func import evaluate_predictions, print_evaluation_report
from auto_analysis.auto_coding import code_full_transcript
from auto_analysis.data_clean_to_json import clean_excel_to_json
from error_analysis.error_analysis import error_analysis

# 默认配置（可被命令行覆盖）
DEFAULT_EXCEL_PATH = "coded_discourse/excel/chris moon video 1 transcription_susan_5_22.xlsx"
DEFAULT_JSON_DIR = "coded_discourse/json"
DEFAULT_FILE_ID = "chris_moon_v1"
DEFAULT_N_VOTES = 3


def parse_args():
    parser = argparse.ArgumentParser(description="APT 话语自动编码工具")
    parser.add_argument("--excel", default=DEFAULT_EXCEL_PATH, help="Excel 文件路径")
    parser.add_argument("--file-id", default=DEFAULT_FILE_ID, help="输出文件标识符")
    parser.add_argument("--votes", type=int, default=DEFAULT_N_VOTES, help="每个话轮的投票次数")
    parser.add_argument("--json-dir", default=DEFAULT_JSON_DIR, help="JSON 输出目录")
    return parser.parse_args()


def auto_coding(excel_path: str, file_id: str, n_votes: int, json_dir: str):
    """执行完整的自动编码流程"""
    # 1. 检查是否有已清洗好的json， 且 excel未修改
    os.makedirs(json_dir, exist_ok=True)
    clean_json_path = os.path.join(json_dir, f"{file_id}.json")
    need_clean = True
    if os.path.exists(clean_json_path):
        # 检查修改时间：只有 Excel 比 JSON 新时才重新清洗
        excel_mtime = os.path.getmtime(excel_path)
        json_mtime = os.path.getmtime(clean_json_path)
        if json_mtime >= excel_mtime:
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
    predictions = code_full_transcript(transcript, n_votes=n_votes)

    pred_output = {
        "file_id": file_id,
        "predictions": predictions,
        "generated_by": "gpt-4o",  # 建议改为从 config 读取模型名
        "n_votes": n_votes,
    }
    pred_json_path = os.path.join(json_dir, f"predictions_{file_id}.json")
    with open(pred_json_path, "w", encoding="utf-8") as f:
        json.dump(pred_output, f, indent=2, ensure_ascii=False)
    print(f"预测结果已保存: {pred_json_path}")

    # 3. 评估
    if gold:
        print("开始评估...")
        pred_for_eval = [
            {"turn_id": p["turn_id"], "codes": p["codes"]}
            for p in predictions
        ]
        eval_result = evaluate_predictions(gold, pred_for_eval)
        print_evaluation_report(eval_result)


if __name__ == '__main__':
    args = parse_args()
    # 调用llm进行自动分析 （清洗、编码、评估）
    auto_coding(args.excel, args.file_id, args.votes, args.json_dir)

    # result = clean_excel_to_json(args.excel, args.file_id)
    # gold_data = result["gold"]
    #
    # with open("coded_discourse/json/predictions_chris_moon_v1.json", "r", encoding="utf-8") as f:
    #      pred_data = json.load(f)["predictions"]
    #
    # # # 评估具体错误的细节
    # error_analysis(gold_data, pred_data)