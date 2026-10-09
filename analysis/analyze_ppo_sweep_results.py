#!/usr/bin/env python3
"""Create reproducible summaries and an evidence audit from the uploaded repo data.

Run from any directory: python3 analyze_ppo_sweep_results.py
The repository root is inferred from this script's parent directory.
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"
OUT_CSV = HERE / "sweep_summary.csv"
OUT_PNG = HERE / "sweep_mean_score.png"
OUT_MD = HERE / "evidence_audit.md"
GROUPS = list("ABCDE")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fmt(x: float) -> str:
    return f"{x:.4f}".rstrip("0").rstrip(".")


def main() -> None:
    source_csv = RESULTS / "sweep_1m" / "leaderboard.csv"
    leaderboard = {}
    with source_csv.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            leaderboard[r["exp_id"]] = r

    summaries: list[dict] = []
    series: dict[str, list[dict]] = {}
    for exp in GROUPS:
        path = RESULTS / "sweep_1m" / f"{exp}.jsonl"
        rows = load_jsonl(path)
        series[exp] = rows
        best_mean = max(rows, key=lambda r: r["mean_score"])
        peak_max = max(r["max_score"] for r in rows)
        peak_max_rows = [r for r in rows if r["max_score"] == peak_max]
        first_nonzero = next((r["step"] for r in rows if r["mean_score"] > 0), "")
        last = rows[-1]
        lb = leaderboard.get(exp, {})
        summaries.append({
            "exp_id": exp,
            "checkpoints": len(rows),
            "first_step": rows[0]["step"],
            "last_step": last["step"],
            "seed_values": ";".join(str(s) for s in sorted({r["seed"] for r in rows})),
            "first_nonzero_mean_step": first_nonzero,
            "final_mean_score": last["mean_score"],
            "best_logged_mean_score": best_mean["mean_score"],
            "step_at_best_mean": best_mean["step"],
            "recorded_max_score": peak_max,
            "step_at_recorded_max_score": ";".join(str(r["step"]) for r in peak_max_rows),
            "final_p90_score": last["p90_score"],
            "leaderboard_reported_peak_mean": lb.get("peak_mean_score", ""),
            "leaderboard_reported_step_at_peak": lb.get("step_at_peak", ""),
        })

    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0]))
        w.writeheader()
        w.writerows(summaries)

    # Figure: training-time evaluation game score, not TensorBoard episodic reward.
    plt.rcParams.update({
        "font.family": "Noto Sans CJK TC",
        "font.size": 11,
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
    })
    fig, ax = plt.subplots(figsize=(11.5, 6.7), constrained_layout=True)
    palette = {"A": "#0072B2", "B": "#D55E00", "C": "#009E73", "D": "#CC79A7", "E": "#E69F00"}
    for exp, rows in series.items():
        x = [r["step"] / 1_000_000 for r in rows]
        y = [r["mean_score"] for r in rows]
        ax.plot(x, y, marker="o", markersize=3.8, linewidth=1.7, color=palette[exp], label=f"Group {exp}")
    ax.set_title("A–E 評估紀錄：平均遊戲分數隨訓練步數變化")
    ax.set_xlabel("訓練步數（百萬步；依 JSONL 的 step 欄位）")
    ax.set_ylabel("每次評估的 mean_score（原始欄位；單位／聚合方式待核）")
    ax.set_xlim(0, 1.03)
    ax.set_ylim(bottom=0)
    ax.set_xticks([i / 10 for i in range(0, 11, 1)])
    ax.grid(True, color="#D9DEE5", linewidth=0.8, alpha=0.85)
    ax.legend(ncol=5, loc="upper left", frameon=False)
    fig.text(0.01, -0.015, "註：每組僅記錄 seed=42；每個 checkpoint 的評估回合數未寫入原始 JSONL。這不是 TensorBoard reward 曲線。", fontsize=9, color="#444444")
    fig.savefig(OUT_PNG, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    lines = []
    lines += [
        "# Flappy Bird PPO：現有實驗證據稽核",
        "",
        "> 本文件僅整理上傳至 `1123leo/small` 儲存庫中可讀取的原始檔；不推送或修改 GitHub 儲存庫。",
        "",
        "## 核心結論",
        "",
        "儲存庫確實包含一個 180 維 Lidar PPO 訓練腳本與相符 checkpoint；但 A–E 的組別映射仍缺失、各維度訓練設定不一致、TensorBoard event 檔缺失，且 A 組獨立 benchmark 與訓練期評估互相矛盾。因此，現有資料仍不足以驗證高維與低維方法的比較結論。",
        "",
        "## 可直接引用的描述性結果",
        "",
        "來源：`results/sweep_1m/A.jsonl` 至 `E.jsonl`。每檔 20 筆，step 由 50,000 至 1,000,000，紀錄欄位為 `step`, `mean_score`, `max_score`, `p90_score`, `seed`, `exp_id`；五組紀錄中的 seed 均為 42。",
        "",
        "| 組別 | 最後一步 mean_score（1,000,000） | 紀錄中最高 mean_score（步數） | 原始紀錄最高 max_score | 首次 mean_score > 0 |",
        "|---|---:|---:|---:|---:|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['exp_id']} | {fmt(s['final_mean_score'])} | {fmt(s['best_logged_mean_score'])}（{int(s['step_at_best_mean']):,}） | {fmt(s['recorded_max_score'])} | {int(s['first_nonzero_mean_step']):,} |"
        )
    lines += [
        "",
        "圖表 `sweep_mean_score.png` 畫的是 JSONL 原始 `mean_score` 評估欄位；由於產生評估摘要的程式未上傳，分數單位、每次評估回合數及彙整方式仍待核對。它**不是**可由現有證據確認的 TensorBoard `rollout/ep_rew_mean`，也不應直接稱為 reward 曲線。",
        "",
        "## 重大限制與矛盾",
        "",
        "1. **實驗組別無法對應研究條件。** 儲存庫沒有找到 A–E 的實驗設定表或產生這些 JSONL 的 sweep 訓練／評估程式。因此目前不能把 A、B 等標成高維、低維或優化版，也不能把差異歸因於 Observation 或 PPO 超參數。",
        "2. **只有一個 seed。** 五組皆為 seed=42，沒有跨 seed 重複，不能用來估計訓練隨機性，也不能支持「穩定性較佳」的統計結論。原始 JSONL 沒有逐回合分數，不能重算標準差、信賴區間或顯著性。每筆評估的回合數亦未記錄；分數呈現 1/30 的小數步距只是線索，不可當成已證實的 30 回合。",
        "3. **A 組的獨立 benchmark 與訓練期紀錄矛盾。** `results/sweep_1m/champion_benchmark.json` 報告 A 的 100 回合 mean、median、max、p90、std 全為 0；但 A.jsonl 記錄有非零平均分，最高 mean_score=1.0。benchmark 指向的 `models/sweep_1m/A/best_score_model.zip` 不在儲存庫中，故無法確認是否測了同一模型、同一環境／wrapper 或同一評分程式。",
        "4. **排行榜摘要錯誤或定義不明。** 對照 A–E 原始 JSONL，`leaderboard.csv` 的 A、C、E `peak_mean_score`／`step_at_peak` 和最大 mean_score 不符：A 原始最大值 1.0@950,000（排行榜 .8@500,000）；C .9333@850,000（排行榜 .4333@900,000）；E .9@800,000 與 950,000（排行榜 .4333@900,000）。B、D 的欄位與原始最大值相符。不要直接沿用排行榜名次或這三列摘要。",
        "5. **12 維實驗目前也未能支持成功主張。** `results/cv12_match_180/eval_scores.jsonl` 三個檢查點（10k、20k、30k）皆為 mean_score=0；`benchmark_100eps.json` 亦報告 100 回合平均／最高分為 0。雖然有 `models/cv12_match_180/final_model.zip`，但沒有每回合原始分數或可核對的評估程式；其命名中的 `180` 不足以證明它是 180 維雷達模型。",
        "6. **TensorBoard 主張沒有原始 event 證據。** 在儲存庫中未找到任何 `events.out.tfevents*` 檔案。故 `ep_rew_mean`、`ep_len_mean`、loss、entropy、approx_kl、clip_fraction 等曲線，目前無法由上傳檔案重現或驗證。",
        "7. **已找到 180 維 Lidar 訓練腳本及 checkpoint。** `train_ppo_lidar_180d_reward_shaped_legacy.py` 明確設 `use_lidar=True`、`SEED=42`、4 個訓練環境、1,000,000 訓練步數，使用 +0.05 存活獎勵、+20 過管獎勵及 terminated 時 -15，並儲存為 `ppo_flappybird_final`。對應的 `ppo_flappybird_final.zip` 靜態 SB3 metadata 顯示 observation shape=[180]、實際記錄 1,007,616 timesteps、seed=42；`多於/showcase_ppo_vs_human_full.py` 也使用同一模型路徑與 `use_lidar=True`。因此「找不到 180 維模型」的先前判斷錯誤，應予更正。",
        "8. **找到 180 維模型，不代表既有高低維實驗已公平比較。** 180 維程式使用 4 個環境及 +20/-15 獎勵；可見的 36 維 `train_ppo_state36d_variant_07.py` 使用 8 個環境、不同獎勵（+0.2 過管、中心對齊等）與不同 PPO 設定，且對應的 10M checkpoint 訓練步數也不同。9D、12D 模型同樣來自不同程式／訓練階段。這些模型不能不加控制地直接比較，也沒有 A–E 組別映射。",
        "9. **Reward shaping 程式版本不一致。** 可見程式中的過管獎勵與死亡懲罰有 +0.2/-0.15、+20/-15、+10/-5 等不同設定。報告必須指明每一組實際執行的程式版本／設定，不能把不同版本的 reward 數值合併敘述。",
        "",
        "## 對論文目前可作的修正",
        "",
        "- 暫時刪除或改寫「高維雷達較震盪、低維較穩定」、「更有利於泛化」、「優化配置使曲線穩定上升」等因果／比較結論；目前資料不能辨認組別與條件，也沒有多 seed 或泛化測試。",
        "- 第五章若先採用現有數據，只能寫成**描述性觀察**：A–E 在 seed 42 的訓練期評估平均分均曾非零，但各組最後一點皆低於或等於其歷程中的高點；A 的獨立 100 回合紀錄為 0，與訓練期分數矛盾，故目前結論不確定。配圖須標成「評估遊戲分數」，不要稱為 TensorBoard reward。",
        "- 不可把最高單次分數或 leaderboard 排名當成穩定學習、普遍優越或模型泛化的證據。",
        "",
        "## 要補到能支持高維／低維比較，最少需補的資料",
        "",
        "1. A–E 每組的完整訓練程式／commit、觀測維度與定義、reward、PPO 超參數、環境版本及模型檔路徑對照表。",
        "2. 每個條件至少 3 個獨立訓練 seed（資源許可時更多），所有條件採相同訓練步數與一致評估流程；保存每個 seed 的結果，不只保存平均。",
        "3. 固定且獨立於訓練的評估 seeds；保存每回合 score、存活步數及失敗原因，報告平均／中位數、標準差或信賴區間。至少 100 回合可以改善測量精度，但不能取代多個訓練 seed。",
        "4. 上傳 TensorBoard event 檔或可重畫的 scalar CSV；同時提供 `ep_rew_mean`、`ep_len_mean` 與實際遊戲分數，避免把兩種量混用。",
        "5. 重跑 A 組 100 回合 benchmark，核對模型檔、Observation wrapper、`deterministic` 設定、環境版本及 seed；在矛盾解決前，不應把 champion benchmark 當作論文結果。",
        "",
        "## 來源與可追溯性",
        "",
        f"- 儲存庫目前 clone 的 HEAD：`{subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()}`。",
        f"- A–E 原始 JSONL SHA-256：" + ", ".join(f"`{g}.jsonl` `{sha256(RESULTS / 'sweep_1m' / f'{g}.jsonl')[:16]}…`" for g in GROUPS) + ".",
        f"- 輸出產生時間（UTC）：{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}。",
        "- 可重現腳本：`analysis/analyze_ppo_sweep_results.py`；執行後會重建 CSV、PNG 與本報告。",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_CSV}\nWrote {OUT_PNG}\nWrote {OUT_MD}")


if __name__ == "__main__":
    main()
