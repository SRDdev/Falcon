#!/usr/bin/env python
"""Build the shareable results page (report/index.html) from experiment CSVs. No hand-entered numbers.
   python analysis/build_report_html.py"""
import html
import json
import shutil
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
E, OUT = ROOT / "experiments", ROOT / "report"
(OUT / "figs").mkdir(parents=True, exist_ok=True)
WM = {"wam": "WAM", "trustmark": "TrustMark", "dwtdct": "DWT-DCT-SVD", "none": "none (B0)"}
ED = {"ip2p": "InstructPix2Pix", "sdedit": "SDEdit s=0.5"}


def fig(name, caption):
    src = ROOT / "figures" / f"{name}.png"
    if not src.exists():
        return ""
    shutil.copy(src, OUT / "figs" / src.name)
    return f'<figure class="plate"><img src="figs/{src.name}" alt="{html.escape(caption)}" loading="lazy"><figcaption>{caption}</figcaption></figure>'


def table(df, fmt=None, heat=None):
    fmt = fmt or {}
    h = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        tds = []
        for c in df.columns:
            v = r[c]
            if isinstance(v, float):
                s = format(v, fmt.get(c, ".3f")) if pd.notna(v) else "–"
                style = f' style="--v:{max(0.0, min(1.0, v)):.3f}"' if heat and c in heat and pd.notna(v) else ""
                cls = ' class="num heat"' if style else ' class="num"'
                tds.append(f"<td{cls}{style}>{s}</td>")
            else:
                tds.append(f"<td>{html.escape(str(WM.get(v, ED.get(v, v))))}</td>")
        rows.append("<tr>" + "".join(tds) + "</tr>")
    return f'<div class="tbl"><table><thead><tr>{h}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def levels(sw, val):
    t = sw.pivot_table(index=["editor", "method", "watermark"], columns="level", values=val).reset_index()
    t.columns = [c if isinstance(c, str) else f"L{c}" for c in t.columns]
    return t


p = E / "pilot" / "results"
sw = pd.read_csv(p / "summary_wm.csv")
d = pd.read_csv(p / "rows_derived.csv", dtype={"id": str})
fp = pd.read_csv(p / "fpr.csv").groupby("detector")[["false_positives", "n"]].sum().reset_index()
n_pilot = int(sw.n.min())
det = levels(sw, "det_rate")
lv = [c for c in det.columns if c.startswith("L")]
edit = d[d.level > 0].groupby(["editor", "method", "watermark"])[["edit_success", "clip_dir", "dino", "lpips"]].mean().reset_index()
imp = d[(d.level == 0) & (d.watermark != "none")].drop_duplicates(["watermark", "id"]).groupby("watermark")[["wm_psnr", "wm_ssim"]].mean().reset_index()

# headline numbers, all computed
b = sw[(sw.method == "baseline")]
ip_l4 = b[(b.editor == "ip2p") & (b.level == b.level.max())].set_index("watermark").det_rate
sd_l1 = b[(b.editor == "sdedit") & (b.level == 1)].det_rate.max()
re = sw[(sw.method == "reembed") & (sw.level > 0)]
re_l0 = sw[(sw.method == "reembed") & (sw.level == 0)].set_index(["editor", "watermark"]).det_rate
re_drop = max(abs(r.det_rate - re_l0[(r.editor, r.watermark)]) for r in re.itertuples())
# relative to what was detectable at L0 (DWT-DCT-SVD starts below 100%)
re_min = min(r.det_rate / re_l0[(r.editor, r.watermark)] for r in re.itertuples() if re_l0[(r.editor, r.watermark)] > 0)
fp_total, fp_n = int(fp.false_positives.sum()), int(fp.n.sum())

# probes (partial allowed)
pr_rows = [r for f in sorted((E / "probes" / "probes" / "rows").glob("*.json")) for r in json.loads(f.read_text())]
probe_html = ""
if pr_rows:
    pr = pd.DataFrame(pr_rows)
    partial = pr.groupby("probe").id.nunique().min() < 27
    order = ["vae_roundtrip", "ip2p_identity", "ip2p_imgcfg_4.0", "ip2p_imgcfg_2.5", "ip2p_imgcfg_1.5", "ip2p_imgcfg_1.0",
             "ip2p_steps_25", "ip2p_steps_10", "sdedit_0.1", "sdedit_0.2", "sdedit_0.3", "sdedit_0.5", "sdedit_0.7"]
    g = pr.pivot_table(index="probe", columns="watermark", values="detected", aggfunc="mean")
    g = g.reindex([o for o in order if o in g.index])
    g.insert(0, "images", pr.groupby("probe").id.nunique().reindex(g.index).astype(float))
    g = g.reset_index().rename(columns={c: WM.get(c, c) for c in g.columns})
    vae = pr[pr.probe == "vae_roundtrip"].groupby("watermark").detected.mean()
    ident = pr[pr.probe == "ip2p_identity"].groupby("watermark").detected.mean()
    _det = pr.groupby(["probe", "watermark"]).detected.mean()
    dr = lambda p, w: _det.get((p, w), float("nan"))  # noqa: E731
    e0 = lambda w: pr[pr.watermark == w]["band0_energy0"].mean()  # noqa: E731
    probe_html = f"""
<section id="probes"><h2>Where the watermark is lost</h2>
<p class="lede">Each probe applies one piece of the editing pipeline to the watermarked original and decodes again.
{'<span class="pill warn">partial: run in progress</span>' if partial else ''}</p>
<ul class="findings">
<li>The SD-1.5 VAE alone (encode → decode, no diffusion) keeps WAM detectable on {vae.get('wam', float('nan')):.0%} and TrustMark on {vae.get('trustmark', float('nan')):.0%} of images, but DWT-DCT-SVD on only {vae.get('dwtdct', float('nan')):.0%}.</li>
<li>An InstructPix2Pix edit told to <em>keep the image unchanged</em> already drops WAM to {ident.get('wam', float('nan')):.0%} and TrustMark to {ident.get('trustmark', float('nan')):.0%}. The diffusion regeneration step, not the requested change, does most of the damage for the neural watermarks.</li>
<li>Image guidance (how strongly InstructPix2Pix sticks to its input) is effectively the watermark dial: at 1.0, WAM survives on {dr('ip2p_imgcfg_1.0','wam'):.0%} and TrustMark on {dr('ip2p_imgcfg_1.0','trustmark'):.0%}; at 4.0, on {dr('ip2p_imgcfg_4.0','wam'):.0%} and {dr('ip2p_imgcfg_4.0','trustmark'):.0%}.</li>
<li>SDEdit has a sharp threshold: at noise strength 0.1, WAM survives on {dr('sdedit_0.1','wam'):.0%}; at 0.3, on {dr('sdedit_0.3','wam'):.0%}; from 0.5, on {dr('sdedit_0.5','wam'):.0%}, with no change to the prompt.</li>
<li>The VAE keeps the low-frequency part of the watermark residual and loses the rest. TrustMark puts {e0('trustmark'):.0%} of its residual energy in the lowest frequency band (WAM {e0('wam'):.0%}, DWT-DCT-SVD {e0('dwtdct'):.0%}), which matches its robustness to the VAE.</li>
</ul>
{table(g, {"images": ".0f"}, heat=set(WM.values()))}
{fig('probes_partial_fig9_probes' if partial else 'probes_fig9_probes', 'Detection rate and residual correlation per probe.')}
{'' if partial else fig('probes_fig10_frequency_survival', 'Fraction of the watermark residual surviving per radial frequency band (lines) and where its energy sits (grey bars).')}
</section>"""

progress = []
for name in ["pilot", "probes", "guided_test", "stage2"]:
    dd = E / name
    if dd.exists():
        progress.append((name, len(list(dd.glob("images/*/*/*/*/L*.png"))), (dd / "results" / "rows.csv").exists()))

doc = f"""<title>Watermark Retention Study</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{{--bg:#f6f7f5;--panel:#ffffff;--ink:#1b2129;--ink2:#56606b;--rule:#dde1dc;--accent:#2a78d6;--warn:#a35b00;--warnbg:#fff1dc;--ok:#1d7a4f;--okbg:#e3f4ea;--bad:#b3261e;--badbg:#fbe5e3;--heat:42,120,214}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{color-scheme:dark;--bg:#12161b;--panel:#1a2027;--ink:#e6eaee;--ink2:#9aa5b1;--rule:#2b333c;--accent:#5a9ef0;--warn:#f0b35c;--warnbg:#3a2a12;--ok:#5fcf98;--okbg:#14321f;--bad:#f28b82;--badbg:#3b1715;--heat:90,158,240}}}}
:root[data-theme="dark"]{{color-scheme:dark;--bg:#12161b;--panel:#1a2027;--ink:#e6eaee;--ink2:#9aa5b1;--rule:#2b333c;--accent:#5a9ef0;--warn:#f0b35c;--warnbg:#3a2a12;--ok:#5fcf98;--okbg:#14321f;--bad:#f28b82;--badbg:#3b1715;--heat:90,158,240}}
body{{background:var(--bg);color:var(--ink);font:15px/1.6 "IBM Plex Sans",system-ui,sans-serif;padding-inline:20px;padding-block:28px 64px}}
main{{max-width:1000px;margin:0 auto;display:grid;gap:44px}}
h1,h2{{font-family:"IBM Plex Sans Condensed","IBM Plex Sans",sans-serif;text-wrap:balance;margin:0}}
h1{{font-size:2.1rem;line-height:1.15;font-weight:600}} h2{{font-size:1.45rem;font-weight:600;margin-bottom:6px}}
.eyebrow{{font:500 .75rem "IBM Plex Mono",monospace;letter-spacing:.08em;text-transform:uppercase;color:var(--ink2)}}
.lede,p{{max-width:68ch;color:var(--ink2);margin:.3em 0 1em}}
header{{display:grid;gap:10px;border-bottom:1px solid var(--rule);padding-bottom:22px}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}}
.kpi{{background:var(--panel);border:1px solid var(--rule);border-radius:6px;padding:14px 16px;display:grid;gap:4px}}
.kpi b{{font:500 1.7rem "IBM Plex Mono",monospace;font-variant-numeric:tabular-nums}}
.kpi span{{color:var(--ink2);font-size:.88rem;line-height:1.4}}
.pill{{display:inline-block;font:500 .72rem "IBM Plex Mono",monospace;padding:2px 8px;border-radius:99px;margin-left:6px;vertical-align:middle}}
.warn{{background:var(--warnbg);color:var(--warn)}} .ok{{background:var(--okbg);color:var(--ok)}} .bad{{background:var(--badbg);color:var(--bad)}}
.findings{{margin:0 0 14px;padding-left:1.1em;max-width:72ch;display:grid;gap:6px}}
.tbl{{overflow-x:auto;border:1px solid var(--rule);border-radius:6px;background:var(--panel);margin:12px 0}}
table{{border-collapse:collapse;width:100%;font-size:.86rem}}
th{{font:500 .72rem "IBM Plex Mono",monospace;letter-spacing:.04em;text-transform:uppercase;color:var(--ink2);text-align:left;padding:8px 10px;border-bottom:1px solid var(--rule);white-space:nowrap}}
td{{padding:6px 10px;border-bottom:1px solid var(--rule);white-space:nowrap}} tr:last-child td{{border-bottom:0}}
td.num{{font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums;text-align:right}}
td.heat{{background:rgba(var(--heat),calc(var(--v)*.38))}}
.plate{{margin:14px 0;background:#fff;border:1px solid var(--rule);border-radius:6px;padding:12px}}
.plate img{{display:block;width:100%;height:auto}}
figcaption{{color:#56606b;font-size:.82rem;margin-top:6px}}
.status{{display:grid;gap:8px}} .row{{display:flex;flex-wrap:wrap;gap:8px 16px;align-items:baseline;border-bottom:1px solid var(--rule);padding-bottom:8px}}
.row code{{font-family:"IBM Plex Mono",monospace}}
.note{{border-left:3px solid var(--warn);padding:4px 0 4px 14px;max-width:72ch}}
</style>
<main>
<header>
<div class="eyebrow">Research progress · generated {time.strftime('%d %b %Y, %H:%M')} from experiment outputs</div>
<h1>Do invisible watermarks survive repeated text-based image edits?</h1>
<p class="lede">We watermark PIE-Bench++ photos with three post-hoc watermarks, edit them four times in a row with
InstructPix2Pix and SDEdit, and decode after every round (L0 = watermarked original, L1–L4 = after each edit).
Detection is a one-sided binomial test at a false-positive rate of 1 in 1,000.</p>
</header>

<section class="kpis" aria-label="Headline results">
<div class="kpi"><b>{ip_l4.get('wam', float('nan')):.0%} / {ip_l4.get('trustmark', float('nan')):.0%}</b><span>WAM / TrustMark still detected after 4 InstructPix2Pix edits, plain editing (pilot, n={n_pilot})</span></div>
<div class="kpi"><b>{sd_l1:.0%}</b><span>of watermarks survive one SDEdit pass (strength 0.5) with plain editing, no protection</span></div>
<div class="kpi"><b>{re_min:.0%}</b><span>of the watermarks detectable before editing are still detected after every round with decode → edit → re-embed (worst editor/watermark pair)</span></div>
<div class="kpi"><b>{fp_total}/{fp_n}</b><span>false alarms on unwatermarked images (all detectors, rounds, editors)</span></div>
</section>

<section><h2>Retention over four edits</h2>
<p class="lede">Pilot, {n_pilot} images, 4 rounds. Shaded bands are 95% Wilson intervals; with n={n_pilot} they are wide.
<span class="pill warn">pilot · underpowered</span></p>
{fig('pilot_fig2_retention_det_rate', 'Detection rate per editing round, baseline editing.')}
{table(det, {c: ".2f" for c in lv}, heat=set(lv))}
</section>

<section><h2>A trivial fix already works</h2>
<p class="lede">Decoding the payload from the input, editing, then embedding the same payload into the output keeps
detection at its starting level in every round. In the pilot this costs no visible edit quality.</p>
{fig('pilot_fig3_methods_ip2p_det_rate', 'Baseline vs decode → edit → re-embed, InstructPix2Pix.')}
<p class="note">Consequence: a "watermark-aware editor" that steers the edit with the watermark decoder needs the decoder inside
the editor, and whoever has the decoder for these watermarks has the encoder too. Our guided editor is therefore being
tested against re-embedding, not presented as the contribution.</p>
</section>

<section><h2>Edit quality</h2>
<p class="lede">Mean over rounds 1–4. <em>edit success</em> = the image moved toward the target caption in CLIP space
(directional and absolute), as fixed in the pre-registration. <code>none</code> is the editor on the unwatermarked image.</p>
{table(edit)}
{table(imp, {"wm_psnr": ".1f", "wm_ssim": ".3f"})}
{fig('pilot_fig7_qualitative_wam_ip2p', 'WAM under InstructPix2Pix: baseline and re-embed rows for three images (chosen by rule: first image of three categories).')}
</section>
{probe_html}
<section class="status"><h2>Runs</h2>
{''.join(f'<div class="row"><code>{n}</code><span>{k} edited images on disk</span><span class="pill {"ok" if done else "warn"}">{"evaluated" if done else "running"}</span></div>' for n, k, done in progress)}
<p>Stage 2 (90 images, 10 per PIE-Bench++ category) runs on an NVIDIA L4; the guided-editing test (18 held-out images) and the
forgery test run on an RTX 3060. This page is rebuilt from their outputs when they finish.</p>
</section>
</main>"""
(OUT / "index.html").write_text(doc)
print(OUT / "index.html", [p.name for p in (OUT / "figs").iterdir()])

# Standalone local copy: full document, figures embedded as data URIs (opens anywhere, no folder needed).
import base64, re  # noqa: E402

def _inline(m):
    data = base64.b64encode((OUT / m.group(1)).read_bytes()).decode()
    return f'src="data:image/png;base64,{data}"'

head = '<!doctype html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
title, rest = doc.split("</title>", 1)
standalone = f"{head}{title}</title>{rest.split('<main>', 1)[0]}</head><body><main>{rest.split('<main>', 1)[1]}</body></html>"
standalone = re.sub(r'src="(figs/[^"]+)"', _inline, standalone).replace(' loading="lazy"', "")
(OUT / "watermark_retention_report.html").write_text(standalone)
print(OUT / "watermark_retention_report.html")
