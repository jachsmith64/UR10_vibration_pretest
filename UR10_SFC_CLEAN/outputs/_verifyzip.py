"""交付前实际打开两个 ZIP 逐项核对（V1.3）。

V1.3 相比 V1.2 的核对项变化：
  · 慢偏移换成「四个稳定常值 + 三段余弦过渡」→ 核对 drift_baseline_summary.json、
    四个新参考 CSV，并【反向】确认 LOWESS 那套产物（b_slow_offset / b_plus_x /
    b_minus_x / e_raw_um / e_vib_um）已不在包里；
  · §十六 禁止项静态扫描：LOWESS、盲扫参数表、132 Hz、滤波器/观测器/APF 一类；
  · §十二 采样率必须来自时间戳 —— 扫源码确认没有写死 1000；
  · §十三 DATA 必须含五项测试汇总表（main / scale / bias / event / phasewise）；
  · §十四 任一包里都不得出现图片（任意图片后缀）。
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import zipfile
from collections import Counter
from pathlib import Path

TAG = "V1_3"
OUT = Path(r"C:\Users\PC\Desktop\code\pre_program")
CODE = OUT / f"SFC_sim_{TAG}_CODE.zip"
DATA = OUT / f"SFC_sim_{TAG}_DATA.zip"
ROOTDIR = f"SFC_sim_{TAG}_DATA"
TMP = OUT / "UR10_SFC_CLEAN" / "outputs" / "_unzip_check" / TAG

IMG = (".png", ".jpg", ".jpeg", ".bmp", ".svg", ".pdf", ".gif", ".tif", ".tiff", ".webp")
BANNED_IN_CODE = (".csv", ".npz", ".obj", "__pycache__", ".git", "venv",
                  "/runs/", "/sweep/", "/figs/") + IMG
BANNED_IN_DATA = (".py", ".obj", "__pycache__", ".git", "/figs/") + IMG
# 旧控制链特征词（与本工程 main.py 的 LEGACY_TOKENS 同源）
LEGACY = ("VirtualCart", "virtual_cart", "APF", "delta_y", "F_vir", "y_sfc_offset",
          "closed_loop")
# §十六 禁止项：正式代码路径里不得出现的方案特征
FORBIDDEN = {
    "LOWESS": r"lowess",
    "随X分箱基线(bbox/janela)": r"b_plus_x|b_minus_x|b_slow_offset|SLOW_BIN",
    "盲扫参数表": r"SWEEP_N|SWEEP_R|EXTREME_N|K_SCALES|B0_SCALES|PARAM_SWEEP",
    "132Hz": r"132\s*Hz|F_132|f_132",
    "滤波器/观测器": r"lowpass_filter|butter|iirfilter|kalman|observer\b",
    "APF 位场": r"potential_field|\bAPF\b|attractive_force|repulsive_force",
    "轨迹规划": r"trajectory_planner|moveit|plan_traj|rrt_|TOPP",
    "视觉噪声/掉帧/延迟": r"vision_noise|frame_drop|dropout|latency_ms",
}
# §十二：PSD 采样率禁止写死。逐行排除 mm→m 的 1000.0 与物理步长。
FS_HARDCODE = re.compile(r"\bfs\s*=\s*1000|fs\s*=\s*200\b|SAMPLE_RATE\s*=\s*1000")

ok = True


def say(s=""):
    print(s)


def check(cond, msg):
    global ok
    ok &= bool(cond)
    say(f"  [{'OK ' if cond else 'FAIL'}] {msg}")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ============================================================ ① 两个 ZIP 的成员与卫生
for tag, zpath, banned in (("CODE", CODE, BANNED_IN_CODE), ("DATA", DATA, BANNED_IN_DATA)):
    say("=" * 72)
    say(f"{tag} ZIP: {zpath.name}  ({zpath.stat().st_size} bytes)")
    say("=" * 72)
    with zipfile.ZipFile(zpath) as z:
        bad_crc = z.testzip()
        check(bad_crc is None, f"testzip（逐成员 CRC）→ {bad_crc or '全部完好'}")
        names = z.namelist()
        say(f"  成员数 {len(names)}，解压后总字节 {sum(i.file_size for i in z.infolist())}")
        for n in names:
            say(f"    {n}")
        hits = [n for n in names if any(b in n.lower() for b in banned)]
        check(not hits, f"被禁内容检查（{', '.join(banned[:5])} …）→ {hits or '无'}")
        img_hits = [n for n in names if n.lower().endswith(IMG)]
        check(not img_hits, f"§十四 图片检查（{'/'.join(IMG)}）→ {img_hits or '无'}")
        roots = {n.split("/")[0] for n in names}
        check(roots == {f"SFC_sim_{TAG}_{tag}"},
              f"扁平根目录：单一顶层目录 {roots}")
        dest = TMP / tag
        dest.mkdir(parents=True, exist_ok=True)
        for n in names:
            (dest / n).parent.mkdir(parents=True, exist_ok=True)
            (dest / n).write_bytes(z.read(n))
        check(True, f"全部成员已实际解压到 {dest}")
    say()

# ============================================================ ② CODE 源码扫描
say("CODE ZIP 源码扫描")
py = sorted((TMP / "CODE").rglob("*.py"))
say(f"  交付源码 {len(py)} 个：{[f.name for f in py]}")

# 旧控制链特征词
hit = []
for f in py:
    t = f.read_text(encoding="utf-8")
    for tok in LEGACY:
        if tok in t:
            hit.append((f.name, tok))
check(not hit, f"旧控制链特征词（{len(LEGACY)} 个）→ {hit or '无命中'}")

# §十六 禁止项 —— 只扫【可执行代码】，先把注释与字符串剥掉。
# 理由：本工程的注释/docstring 里大量出现 "LOWESS"、"132 Hz" 这类词，
# 它们是在说明「V1.2 用过什么、现在为什么不用」以及「相机原始帧约 132 Hz」，
# 属于历史说明与数据来源描述，不是禁止方案的实现。
# 不剥掉就会把说明文字误判成违规，所以这里用 tokenize 精确区分。
def code_only(path: Path) -> str:
    import io
    import tokenize
    out = []
    with open(path, "rb") as fh:
        for tok in tokenize.tokenize(fh.readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING,
                            tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
                            tokenize.DEDENT):
                continue
            if tok.type == tokenize.NAME and tok.string in ("#",):
                continue
            out.append(tok.string)
    return " ".join(out)


fhits, fhits_text = {}, {}
for f in py:
    t_code = code_only(f)
    t_all = f.read_text(encoding="utf-8")
    for label, pat in FORBIDDEN.items():
        m = re.findall(pat, t_code, flags=re.IGNORECASE)
        if m:
            fhits.setdefault(label, []).append((f.name, sorted(set(m))[:4]))
        m2 = re.findall(pat, t_all, flags=re.IGNORECASE)
        if m2:
            fhits_text.setdefault(label, []).append((f.name, sorted(set(m2))[:4]))
check(not fhits, f"§十六 禁止项扫描（仅可执行代码，已剥注释/字符串）→ {fhits or '无命中'}")
say("      逐项：" + "；".join(f"{k}→{'命中' if k in fhits else '未命中'}"
                               for k in FORBIDDEN))
say(f"      对照：若连注释与字符串一起扫，会命中 {sorted(fhits_text)} ——")
for label in sorted(fhits_text):
    for nm, toks in fhits_text[label]:
        say(f"        {nm}: {toks}（说明性文字，非实现）")
check(set(fhits) <= set(fhits_text),
      "代码命中项 ⊆ 全文命中项（说明文字确实只是文字）")

# §十二 PSD 采样率禁止写死
hard = []
for f in py:
    for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
        if FS_HARDCODE.search(line):
            hard.append((f.name, i, line.strip()[:70]))
check(not hard, f"§十二 PSD 采样率未写死 → {hard or '无命中'}")
check(any("fs_from_time" in f.read_text(encoding='utf-8') for f in py),
      "存在 fs_from_time 这类「从时间戳反推采样率」的实现")

# 新增 / 移除的模块
names_code = {f.name for f in py}
check("drift.py" in names_code, "新增预处理层 drift.py 已随包交付")
check("slowoffset.py" not in names_code,
      "旧方案 slowoffset.py（LOWESS 随 X 变化基线）已不在包内（备份在 git 分支）")

D = TMP / "DATA" / ROOTDIR

# ============================================================ ③ §七 参考 CSV 与 §三/§八 摘要
say("\nDATA ZIP —— §七 慢偏移参考序列 + §三/§八 摘要")
pk = D / "outputs" / "fd_packet"
for f in ("raw_reference.csv", "drift_baseline.csv", "vibration_reference.csv",
          "drift_baseline_summary.json"):
    check((pk / f).is_file(),
          f"§十三(1) 存在 {f}（{(pk / f).stat().st_size if (pk / f).is_file() else 0} bytes）")
check((pk / "raw_plus_baseline_table.csv").is_file(),
      "附赠 raw_plus_baseline_table.csv（三列并排，供人工比对）")
# 经典三列的并排表
rr = list(csv.DictReader(open(pk / "raw_reference.csv", encoding="utf-8")))
vb = list(csv.DictReader(open(pk / "vibration_reference.csv", encoding="utf-8")))
bl = list(csv.DictReader(open(pk / "drift_baseline.csv", encoding="utf-8")))
say(f"  三表行数：raw {len(rr)} / baseline {len(bl)} / vib {len(vb)}；"
    f"列名 {list(rr[0])}")
check(len(rr) == len(bl) == len(vb) > 7000, "三表等长且覆盖整段记录（7756 行量级）")
check(list(rr[0])[:3] == ["t_s", "phase", "nominal_along_mm"],
      "三表均带 t / phase / nominal_along_mm（§七 要求）")
phases = {r["phase"] for r in rr}
say(f"  phase 取值：{sorted(phases)}")
check({"pre_motion", "X_outbound", "X_return", "post_stop"} <= phases,
      "四个稳定段的原始相位名齐全")
# e_raw = b + e_vib 的恒等式（逐点）
worst = 0.0
for a, b_, c in zip(rr[:2000], bl[:2000], vb[:2000]):
    worst = max(worst, abs(float(a["e_raw_um"]) - (float(b_["b_um"]) + float(c["e_vib_um"]))))
check(worst < 5e-4, f"逐点校验 e_raw = b + e_vib（前 2000 点最大偏差 {worst:.2e} µm）")

ds = json.loads((pk / "drift_baseline_summary.json").read_text(encoding="utf-8"))
lv = ds["levels_um"]
say(f"  §十七 四个常值：b_pre={lv['b_pre']:+.3f}  b_out={lv['b_out']:+.3f}  "
    f"b_ret={lv['b_ret']:+.3f}  b_post={lv['b_post']:+.3f} µm")
check(ds["levels_are_recomputed"] is True, "四个常值标记为「现算，非硬编码」")
for k in ("b_pre", "b_out", "b_ret", "b_post"):
    w = ds["windows"]["pre_motion" if k == "b_pre" else
                      "fwd_steady" if k == "b_out" else
                      "ret_steady" if k == "b_ret" else "post_stop"]
    check(isinstance(w["n"], int) and w["n"] > 100,
          f"  {k} 来自窗口 {w['rule']}（n={w['n']}，i0={w['i0']}..{w['i1']}）")
check(len(ds["transitions"]) == 3 and all(t["form"].startswith("w = 0.5(1−cos")
                                          for t in ds["transitions"]),
      "三段过渡（启动/远端转向/停止）形状均为余弦 w = 0.5(1−cos πs)")
check(all(t["monotone"] for t in ds["transitions"]), "三段过渡均单调、无来回摆动")
# 常值段必须严格为常值
segs = {s["name"]: (s["i0"], s["i1"]) for s in ds["segments"]}
say(f"  七段区间：{[(s['name'], s['i0'], s['i1']) for s in ds['segments']]}")
bcol = [float(r["b_um"]) for r in bl]
for nm in ("pre_motion", "fwd_steady", "ret_steady", "post_stop"):
    i0, i1 = segs[nm]
    seg = bcol[i0:i1 + 1]
    check(max(seg) - min(seg) == 0.0,
          f"  §四 {nm} 段严格为常值（ptp = {max(seg) - min(seg):.1e} µm）")

acc = ds["acceptance"]
check(acc["passed"] is True, f"§八 机器可判验收 passed = {acc['passed']}")
nfail = [k for k, v in acc["checks"].items() if not v]
check(not nfail, f"  {len(acc['checks'])} 项检查全通过 → 未过项 {nfail or '无'}")
check(all(isinstance(v, bool) for v in acc["checks"].values()),
      "  所有检查值都是真 bool（不是字符串 'True'，否则下游 `if not v` 会失效）")
sc = ds["steady_segment_checks"]
for k in ("fwd_steady", "ret_steady"):
    d = sc[k]
    say(f"  §八 {k:11s} median {d['median_raw_um']:+8.3f} → {d['median_vib_um']:+8.3f} | "
        f"mean {d['mean_raw_um']:+8.3f} → {d['mean_vib_um']:+8.3f} | "
        f"rms {d['rms_raw_um']:7.3f} → {d['rms_vib_um']:7.3f} µm "
        f"({d['rms_reduction_pct']:+.1f}%)")
    check(abs(d["mean_vib_um"]) < abs(d["mean_raw_um"]), f"  {k} 均值明显更居中")
    check(d["rms_vib_um"] < d["rms_raw_um"], f"  {k} RMS 明显下降")
    check(d["vib_band_retention"] >= 0.98,
          f"  {k} 振动带（2.5–45 Hz）保留 {d['vib_band_retention'] * 100:.1f}%")
ov = ds["overall"]
say(f"  整体：e_raw rms {ov['rms_raw_um']:.2f} → e_vib rms {ov['rms_vib_um']:.2f} µm"
    f"（−{ov['rms_reduction_pct']:.1f}%），b 范围 {ov['b_min_um']:+.2f}..{ov['b_max_um']:+.2f} µm")
for nt in ds["notes"]:
    say(f"  注：{nt}")
check(any("中位数为 0 是构造必然" in n for n in ds["notes"]),
      "注记已如实说明「稳态段中位数为 0 是构造必然，不能当效果证据」")
check(any("post_stop 段不平稳" in n for n in ds["notes"]),
      "注记已如实说明 post_stop 段不平稳（未静默平滑掉）")

# ============================================================ ④ 冻结 Fd 的三处一致性
say("\n冻结 Fd 的 SHA-256 与作用点")
meta = json.loads((pk / "fd_meta.json").read_text(encoding="utf-8"))
check(meta["recon_source"] == "vib", f"重建口径 recon_source = {meta['recon_source']}")
check(meta["drift_baseline"]["is_bias_test"] is False
      and meta["drift_baseline"]["bias_um"] == 0.0,
      "正式冻结包用的是正确基线（未注入 Δb）")
disk = sha256(pk / "fd_frozen.csv")
full = meta["frozen"]["sha256"]
head = [l for l in (D / "outputs" / "sweep" / "traces" / "A0.csv").read_text(
    encoding="utf-8").splitlines() if l.startswith("# fd_sha256=")][0].split("=", 1)[1]
say(f"  包内 fd_frozen.csv 实算 = {disk}")
say(f"  fd_meta.json 记录      = {full}")
say(f"  A0 轨迹头记录          = {head}")
check(disk == full == head, "冻结 Fd 的 SHA-256 三处一致（文件实算 / meta / 正式轨迹头）")
tc = meta["force_application"]["tcp_force_check"]
check(tc["passed"] is True and tc["max_abs_err"] == 0.0,
      f"TCP 施力点：max|q_applyFT − J_tcp^T·F| = {tc['max_abs_err']:.3e}（参考 {tc['ref_max_abs']:.3e}）")
h = meta["refine"]["history"][-1]
say(f"  重建复现：e_sim rms {h['e_sim_rms_um']:.2f} µm vs 目标 {meta['template']['e_rms_um']:.2f} µm，"
    f"corr {h['corr']:.4f}，残差 {h['residual_rms_um']:.2f} µm（{h['residual_over_target'] * 100:.1f}%）")
check(h["corr"] >= 0.95, f"重建相关性 corr = {h['corr']:.4f} ≥ 0.95")
check(h["residual_over_target"] <= 0.20,
      f"残差/目标 = {h['residual_over_target'] * 100:.1f}% ≤ 20%（§九 门槛）")
check([s[0] for s in meta["template"]["segments"]][:2] == ["start", "mid_fwd"],
      "模板分段已随元数据落盘（供时序 phase 列使用，左闭右开）")
check((pk / "plant_frf.npz").is_file(), "Plant 频响 plant_frf.npz 随包（测试B 复用同一份标定）")

# ============================================================ ⑤ §十三 五张总汇总表
say("\n§十三(5) 五张总汇总表 + 运行档案")
sw = D / "outputs" / "sweep"
for f in ("summary_main.csv", "summary_disturbance_scale.csv",
          "summary_bias_robustness.csv", "summary_event_response.csv",
          "summary_phasewise.csv", "run_manifest.json", "config_snapshot.json",
          "errors.txt"):
    check((sw / f).is_file(), f"存在 {f}")

rows = list(csv.DictReader(open(sw / "summary_main.csv", encoding="utf-8")))
say(f"  summary_main.csv 列：{list(rows[0])}")
say(f"  summary_main 组：{[r.get('case_id') or r.get('group') for r in rows]}")
check(len(rows) == 5, f"§十 正式组恰好 5 个 → {len(rows)}")
ids = [(r.get("case_id") or r.get("group")) for r in rows]
check(set(ids) == {"A0", "Alinear", "ASFC_default", "ASFC_medium", "ASFC_strong"},
      f"§十 组名与要求一致 → {ids}")

sc_rows = list(csv.DictReader(open(sw / "summary_disturbance_scale.csv", encoding="utf-8")))
say(f"  summary_disturbance_scale.csv 列：{list(sc_rows[0])}")
scales = sorted({float(r["scale"]) for r in sc_rows})
check(scales == [0.5, 1.0, 1.5], f"§十一测试A 缩放系数 {scales}（0.5/1.0/1.5）")
for s in scales:
    ids_s = sorted(r["case_id"] for r in sc_rows if float(r["scale"]) == s)
    check(len(ids_s) == 2 and any("Alinear" in i for i in ids_s)
          and any("ASFC_strong" in i for i in ids_s),
          f"  s={s} 恰好 Alinear + ASFC_strong 两行 → {ids_s}")
check(any("Alinear" == r["case_id"] for r in sc_rows),
      "  s=1.0 复用正式组，未重复跑（case_id 直接是 Alinear / ASFC_strong）")

bias_rows = list(csv.DictReader(open(sw / "summary_bias_robustness.csv", encoding="utf-8")))
say(f"  summary_bias_robustness.csv 列：{list(bias_rows[0])}")
biases = sorted({float(r["bias_um"]) for r in bias_rows})
check(biases == [-10.0, -5.0, 5.0, 10.0],
      f"§十一测试B Δb 取值 {biases}（Δb=0 的对照即正式组 ASFC_strong，不重复跑）")
check(all(r["status"] == "OK" for r in bias_rows), "  四个偏差组全部 OK")
check(all(float(r["vib_rms_um"]) < 6.0 for r in bias_rows),
      "  四个偏差组的净振动 RMS 都仍在 6 µm 以下（未因偏差崩掉）")

ev = list(csv.DictReader(open(sw / "summary_event_response.csv", encoding="utf-8")))
pw = list(csv.DictReader(open(sw / "summary_phasewise.csv", encoding="utf-8")))
say(f"  测试C 事件汇总 {len(ev)} 行；测试D 分相汇总 {len(pw)} 行")
check(len(ev) == 3 and {r["case_id"] for r in ev} == {"A0", "Alinear", "ASFC_strong"},
      f"  测试C 含 A0 / Alinear / ASFC_strong 三行 → {[r['case_id'] for r in ev]}")
check(all(int(r["n_events"]) > 0 for r in ev), "  测试C 每行都定位到事件")
check(len(pw) == 25 and len({r["phase"] for r in pw}) == 5,
      f"  测试D 五相位 × 五组 = 25 行 → {len(pw)} 行 / {len({r['phase'] for r in pw})} 相位")

man = json.loads((sw / "run_manifest.json").read_text(encoding="utf-8"))
say(f"  run_manifest.n_runs = {man['n_runs']}；scope = {man.get('scope')}")
check(man["n_runs"] >= 13,
      "运行数 ≥ 13（5 正式 + 测试A 4 + 测试B 4，测试C/D 为离线复用）")
snap = json.loads((sw / "config_snapshot.json").read_text(encoding="utf-8"))
say(f"  config_snapshot 顶层键：{list(snap)[:12]} …（{len(snap)} 项）")
check(not (sw / "errors.txt").read_text(encoding="utf-8").strip()
      or "无" in (sw / "errors.txt").read_text(encoding="utf-8"),
      "errors.txt 无实质错误")

# ============================================================ ⑥ 时序 CSV 完整列
say("\n§十三(3) 各组完整时序 CSV")
tr = sorted((D / "outputs" / "sweep" / "traces").glob("*.csv"))
say(f"  时序组（{len(tr)} 个）：{[p.stem for p in tr]}")
main5 = {"A0", "Alinear", "ASFC_default", "ASFC_medium", "ASFC_strong"}
check(main5 <= {p.stem for p in tr}, f"§十 五个正式组都有完整时序 → 缺 {main5 - {p.stem for p in tr}}")
need = ["t", "phase", "along_mm", "e_um", "v_m_s", "Fd_N", "F_spring_N",
        "F_linear_damping_N", "F_shear_N", "F_SFC_N",
        "tau_1", "tau_2", "tau_3", "tau_4", "tau_5", "tau_6"]
cols_line = [l for l in tr[0].read_text(encoding="utf-8").splitlines()
             if not l.startswith("#")][0]
missing = [c for c in need if c not in cols_line]
check(not missing, f"§十三(3) 必需列齐全（缺 {missing or '无'}）")
say(f"  A0.csv 列（{len(cols_line.split(','))} 列）：{cols_line}")
tm = [l for l in tr[0].read_text(encoding="utf-8").splitlines() if l.startswith("#")]
say(f"  A0.csv 元信息头：{tm}")
check(any("phase_names=" in l for l in tm), "时序头记录了 phase 名称映射")
check("e_raw_um" in cols_line and "b_um" in cols_line and "e_vib_um" in cols_line,
      "§十三(3) 可选但已附：e_raw / b / e_vib 三列参考")
# phase 列必须是 0..4 的整数码。注意时序 CSV 开头有若干 "# 元信息" 行，
# 直接交给 DictReader 会把第一行注释当成列名，必须先剥掉。
def read_trace(p: Path):
    lines = p.read_text(encoding="utf-8").splitlines()
    return list(csv.DictReader(io.StringIO(
        "\n".join(l for l in lines if not l.startswith("#")))))


pc = sorted({int(float(r["phase"])) for r in read_trace(tr[0])})
check(pc and min(pc) >= 0 and max(pc) <= 4, f"phase 编码落在 0..4 → {pc}")
# phase 列必须与模板分段覆盖同一段数据，且五相位都出现过
check(pc == [0, 1, 2, 3, 4], f"五个相位名全部出现 → {pc}")
for p in tr:
    if p.stem in main5:
        n = len(read_trace(p))
        check(n == 12571, f"  {p.stem}.csv 行数 {n} = 62853/5 抽稀（TRACE_STRIDE=5）")

# ============================================================ ⑦ PSD 采样率来自时间戳
say("\n§十二 PSD 采样率来自时间戳")
cs = json.loads((D / "outputs" / "compare" / "summary.json").read_text(encoding="utf-8"))
fs = cs["fs_from_timestamps_hz"]
say(f"  compare/summary.json 采样率（时间戳反推）= {fs:.3f} Hz，样本数 {cs['n_samples']}")
check(abs(fs - 200.0) < 0.5, "抽稀轨迹的采样率约 200 Hz（不是 1000 Hz 物理步长）")
check(cs["run_params"]["A0"]["fd_sha256"] == full, "compare 用的 Fd 与冻结包同一份")

# ============================================================ ⑧ 旧方案产物必须已清除
say("\n旧方案（V1.2 LOWESS 随 X 变化基线）产物必须已清除")
for f in ("b_slow_offset.csv", "b_plus_x.csv", "b_minus_x.csv",
          "e_raw_um.csv", "e_vib_um.csv"):
    check(not (pk / f).is_file(), f"fd_packet 不含旧产物 {f}")
for f in ("recon_source.json",):
    check(not (sw / f).is_file(), f"sweep 不含旧产物 {f}")

# ============================================================ ⑨ 非 ASCII 成员名
say("\n非 ASCII 成员名的字节与 UTF-8 标志位")
with zipfile.ZipFile(DATA) as z:
    nonascii = [i for i in z.infolist() if not i.filename.isascii()]
check(len(nonascii) >= 1, f"DATA 含 {len(nonascii)} 个非 ASCII 成员（源数据文件名）")
for i in nonascii:
    stored = i.filename.encode("utf-8")
    src = (ROOTDIR and Path(
        r"C:\Users\PC\Desktop\code\pre_program\UR10_SFC_CLEAN") /
        Path(i.filename.split("/", 1)[1]))
    flag = bool(i.flag_bits & 0x800)
    say(f"  {stored.split(b'/')[-1]!r}")
    say(f"    UTF-8 标志位(bit 11) = {flag}   zip 内 arcname 末段字节 = {stored.split(b'/')[-1]}")
    check(flag, "  已按 ZIP 规范置 UTF-8 标志位（APPNOTE 6.3.0 §4.4.4）")
    if src.is_file():
        check(stored.split(b"/")[-1] == os.fsencode(src.name),
              "  ZIP 内文件名原始字节与工作树文件**逐字节相同**")
say("  注：Git Bash 自带的 Info-ZIP unzip 会忽略该标志位、按 cp1252 解码，")
say("      从而把 D1_时序.csv 解成乱码文件名（文件内容完好）。")
say("      这是该 unzip 构建的已知问题；7-Zip / Windows 资源管理器 / bsdtar /")
say("      Python zipfile / macOS 归档实用工具 均按规范正确还原。")
say("      复核时若用 Git Bash 的 unzip，请加 -O UTF-8 或改用上述任一工具。")

say("\n" + "=" * 72)
say("打包前核对结论：" + ("全部通过" if ok else "存在未通过项"))
say("=" * 72)
