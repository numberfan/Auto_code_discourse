import json
import os

import pandas as pd

from auto_analysis.assess_func import evaluate_predictions, print_evaluation_report
from auto_analysis.auto_coding import code_full_transcript
from auto_analysis.data_clean_to_json import clean_excel_to_json
from error_analysis import error_analysis

EXCEL_PATH = "coded_discourse/excel/chris moon video 1 transcription_susan_5_22.xlsx"
JSON_DIR = "coded_discourse/json"
FILE_ID = "chris_moon_v1" # 测试数据
N_VOTES = 3            # 投票次数，建议 3~5，如果测试速度可用 1

def auto_coding():

    # 1.数据清洗，保存为json （样例数据）
    print("开始清洗数据...")
    result = clean_excel_to_json(EXCEL_PATH, FILE_ID)

    os.makedirs(JSON_DIR, exist_ok=True)
    clean_json_path = os.path.join(JSON_DIR, f"{FILE_ID}.json")
    with open(clean_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"清洗后的数据已保存: {clean_json_path}")

    # 2.开始分析（加载后分析）
    transcript = result["transcript"]
    gold = result["gold"]

    print("开始自动编码教师话轮...")
    predictions = code_full_transcript(transcript, n_votes=N_VOTES)

    pred_output = {
        "file_id": FILE_ID,
        "predictions": predictions,
        "generated_by": "gpt-4o",
        "n_votes": N_VOTES,
    }
    pred_json_path = os.path.join(JSON_DIR, f"predictions_{FILE_ID}.json")
    with open(pred_json_path, "w", encoding="utf-8") as f:
        json.dump(pred_output, f, indent=2, ensure_ascii=False)
    print(f"预测结果已保存: {pred_json_path}")

    # 3. 评估结果
    if gold:
        print("开始评估...")
        pred_for_eval = [
            {"turn_id": p["turn_id"], "codes": p["codes"]}
            for p in predictions
        ]
        eval_result = evaluate_predictions(gold, pred_for_eval)
        print_evaluation_report(eval_result)


if __name__ == '__main__':
    result = clean_excel_to_json(EXCEL_PATH, FILE_ID)
    gold_data = result["gold"]

    with open("coded_discourse/json/predictions_chris_moon_v1.json", "r", encoding="utf-8") as f:
        pred_data = json.load(f)["predictions"]
    # 直接评估
    # result = evaluate_predictions(gold_data, pred_data)
    # print_evaluation_report(result)

    confusions = error_analysis(gold_data, pred_data)


    # auto_coding()
