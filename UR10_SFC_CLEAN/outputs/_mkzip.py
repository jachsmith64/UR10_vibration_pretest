"""交付打包：CODE ZIP（只放真正在跑的代码）+ DATA ZIP（只放实验数据）。

刻意写死显式清单，不用通配符扫目录 —— 避免把 __pycache__ / 图 / 中间产物
/ 上一版的冻结包混进去。

V1.3 相对 V1.2 的清单变化：
  CODE  − slowoffset.py（LOWESS 随 X 变化基线，已废止；备份在 git 分支
                         UR10_SFC_dedrift_2026_0915 的提交 98a2a4d 里）
        + drift.py（四个稳定常值 + 三段余弦过渡）
  DATA  − fd_packet/{b_slow_offset,b_plus_x,b_minus_x,e_raw_um,e_vib_um}.csv
        + fd_packet/{raw_reference,drift_baseline,vibration_reference,
                     raw_plus_baseline_table}.csv   （§七 要求）
        + fd_packet/drift_baseline_summary.json     （§三/§八 要求）
        + fd_packet/plant_frf.npz                   （测试B 复用的同一份 Plant 标定）
        + sweep/summary_main.csv                    （5 个正式组主汇总）
        + sweep/summary_disturbance_scale.csv       （测试A）
        + sweep/summary_bias_robustness.csv         （测试B）
        + sweep/summary_event_response.csv          （测试C）
        + sweep/event_response_detail.csv           （测试C 逐事件明细）
        + sweep/summary_phasewise.csv               （测试D）
        + sweep/test_results.json                   （四项测试的结论汇总）
        + bias_packets/*/fd_meta.json               （测试B 四次重建各自的元数据）
        − sweep/recon_source.json（口径说明已并入 fd_meta 的 drift_baseline 段）
  明确不含图片：本轮全部代码路径都不产生图片，交付树里也没有任何图片文件。
  说明：测试B 的 4 份 fd_frozen.csv（各约 1 MB）未收录，只收录它们的 fd_meta.json
        —— 需要复算时按 fd_meta 里的参数跑 --reconstruct 即可重生成。
"""
from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(r"C:\Users\PC\Desktop\code\pre_program\UR10_SFC_CLEAN")
OUT = Path(r"C:\Users\PC\Desktop\code\pre_program")

CODE_FILES = [
    "analysis.py", "config.py", "controller.py", "disturbance.py", "drift.py",
    "main.py", "record.py", "sfc.py", "sim_env.py", "sweep.py",
    "README.md", "requirements.txt",
    "assets_ur10e/scene.xml", "assets_ur10e/ur10e.xml",
]

DATA_FILES = [
    # 原始预实验数据（慢偏移与净振动的构造输入）
    "data/P05R01/D1_时序.csv",
    # §七：慢偏移参考序列（记录网格，带原始相位名）
    "outputs/fd_packet/raw_reference.csv",
    "outputs/fd_packet/drift_baseline.csv",
    "outputs/fd_packet/vibration_reference.csv",
    "outputs/fd_packet/raw_plus_baseline_table.csv",
    # §三/§八：四个稳定常值 + 分段统计 + 机器可判验收
    "outputs/fd_packet/drift_baseline_summary.json",
    # 冻结扰动重建结果
    "outputs/fd_packet/fd_frozen.csv",
    "outputs/fd_packet/fd_meta.json",
    "outputs/fd_packet/e_des_um.csv",
    "outputs/fd_packet/schedule.csv",
    "outputs/fd_packet/plant_frf.npz",
    # 模板网格参考列（与 Fd 等长，时序 CSV 的 e_raw/b/e_vib 三列来源）
    "outputs/fd_packet/ref_e_raw_um.csv",
    "outputs/fd_packet/ref_b_um.csv",
    "outputs/fd_packet/ref_e_vib_um.csv",
    # 重建验收
    "outputs/sweep/disturbance_fit_summary.json",
    # §十三(5)：五张总汇总表 + 逐事件明细
    "outputs/sweep/summary_main.csv",
    "outputs/sweep/summary_disturbance_scale.csv",
    "outputs/sweep/summary_bias_robustness.csv",
    "outputs/sweep/summary_event_response.csv",
    "outputs/sweep/event_response_detail.csv",
    "outputs/sweep/summary_phasewise.csv",
    # 运行档案
    "outputs/sweep/sweep_summary.csv",
    "outputs/sweep/baseline_summary.json",
    "outputs/sweep/run_manifest.json",
    "outputs/sweep/config_snapshot.json",
    "outputs/sweep/test_results.json",
    "outputs/sweep/errors.txt",
    "outputs/compare/summary.json",
]

DATA_GLOBS = [
    "outputs/sweep/traces/*.csv",           # 各组完整时序（200 Hz 抽稀）
    "outputs/bias_packets/*/fd_meta.json",  # 测试B 四次重建的元数据
    "outputs/log_reconstruct.txt",
    "outputs/log_sweep.txt",
    "outputs/log_check.txt",
]

IMG_SUFFIX = (".png", ".jpg", ".jpeg", ".bmp", ".svg", ".pdf", ".gif", ".tif", ".tiff")


def build(zip_path: Path, rel_paths: list[str], root_name: str) -> dict:
    missing = [p for p in rel_paths if not (ROOT / p).is_file()]
    if missing:
        raise FileNotFoundError(f"缺文件，拒绝打包：{missing}")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for rel in rel_paths:
            z.write(ROOT / rel, f"{root_name}/{rel}")
    return {"zip": str(zip_path), "n": len(rel_paths)}


def main() -> None:
    globbed = sorted(p for g in DATA_GLOBS for p in ROOT.glob(g))
    data = DATA_FILES + [str(p.relative_to(ROOT)).replace("\\", "/") for p in globbed]

    # 交付前最后一道闸：
    #   CODE 里不许有结果/图/缓存；DATA 里不许有源码/图（任何图片格式，含 svg/pdf）。
    bad_code = [p for p in CODE_FILES
                if Path(p).suffix.lower() in IMG_SUFFIX
                or Path(p).suffix.lower() == ".csv"
                or "outputs/" in p or "__pycache__" in p]
    bad_data = [p for p in data
                if Path(p).suffix.lower() == ".py"
                or Path(p).suffix.lower() in IMG_SUFFIX
                or Path(p).suffix.lower() == ".obj"
                or "/figs/" in p]
    if bad_code or bad_data:
        raise SystemExit(f"清单违规：code={bad_code} data={bad_data}")

    for p in data:
        if Path(p).suffix.lower() not in (".csv", ".json", ".txt", ".npz"):
            raise SystemExit(f"DATA 清单出现非白名单后缀：{p}")

    tag = "V1_3"
    info_c = build(OUT / f"SFC_sim_{tag}_CODE.zip", CODE_FILES, f"SFC_sim_{tag}_CODE")
    info_d = build(OUT / f"SFC_sim_{tag}_DATA.zip", data, f"SFC_sim_{tag}_DATA")
    for i in (info_c, info_d):
        p = Path(i["zip"])
        print(f"{p.name}: {i['n']} files, {p.stat().st_size} bytes "
              f"({p.stat().st_size / 1048576:.2f} MB)")
    print(f"\nCODE：{len(CODE_FILES)} 项（含 README.md / requirements.txt / 2 个 XML）")
    print(f"DATA：{len(data)} 项（显式 {len(DATA_FILES)} + 通配 {len(globbed)}）")
    n_img = sum(1 for p in data + CODE_FILES if Path(p).suffix.lower() in IMG_SUFFIX)
    print(f"图片后缀检查：{'未命中' if n_img == 0 else f'命中 {n_img} 个！'}")
    for g, n in zip(DATA_GLOBS, [sum(1 for _ in ROOT.glob(g)) for g in DATA_GLOBS]):
        print(f"  通配 {g} → {n} 个")


if __name__ == "__main__":
    main()
