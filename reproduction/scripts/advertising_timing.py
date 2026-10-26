"""Analyze anonymized packet tables for gap and advertising-event timing."""

from __future__ import annotations

import csv
import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PDF_METADATA = {"Creator": "PhantomChannel artifact release", "CreationDate": None, "ModDate": None}

# The source analyzer subtracted large epoch-valued floats before converting
# intervals to milliseconds. Keep a synthetic 2**30 anchor (not a captured
# timestamp) so the release reproduces that floating-point path without
# retaining wall-clock time.
NUMERICAL_COMPATIBILITY_ANCHOR_S = 1_073_741_824.0


@dataclass
class Packet:
    timestamp_s: float
    group: str
    counter: Optional[int]
    delta_us: Optional[int]


@dataclass
class Series:
    gaps_ms: list[float]
    events_ms: list[float]
    event_count: int


def percentile(values: list[float], p: float) -> Optional[float]:
    if not values: return None
    values=sorted(values); rank=(len(values)-1)*p/100; lo=math.floor(rank); hi=math.ceil(rank)
    return values[lo] if lo==hi else values[lo]+(values[hi]-values[lo])*(rank-lo)


def mean(values: list[float]) -> Optional[float]: return sum(values)/len(values) if values else None
def std(values: list[float]) -> Optional[float]:
    if not values: return None
    m=mean(values); return 0.0 if len(values)==1 else math.sqrt(sum((x-m)**2 for x in values)/len(values))
def iqr(values: list[float]) -> Optional[float]:
    q1=percentile(values,25); q3=percentile(values,75); return None if q1 is None or q3 is None else q3-q1


def read_packets(path: Path) -> list[Packet]:
    packets=[]
    with path.open(newline="",encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try: timestamp=NUMERICAL_COMPATIBILITY_ANCHOR_S + float(row["relative_time_us"])/1_000_000.0
            except (TypeError,ValueError): continue
            try: counter=int(row["packet_counter"]) if row["packet_counter"] else None
            except ValueError: counter=None
            try: delta=int(row["delta_time_us"]) if row["delta_time_us"] else None
            except ValueError: delta=None
            packets.append(Packet(timestamp,row["anonymous_group"],counter,delta))
    return packets


def cluster(packets: list[Packet]) -> dict[str,list[list[float]]]:
    by={}
    for packet in packets: by.setdefault(packet.group,[]).append(packet.timestamp_s)
    result={}
    for group,times in by.items():
        times.sort(); events=[]; current=[]; last=None
        for t in times:
            if last is None or (t-last)<0.010: current.append(t)
            else: events.append(current); current=[t]
            last=t
        if current: events.append(current)
        result[group]=events
    return result


def compute_series(packets: list[Packet], events: dict[str,list[list[float]]]) -> dict[str,Series]:
    by={}
    for p in packets: by.setdefault(p.group,[]).append(p)
    for group in by: by[group].sort(key=lambda p:p.timestamp_s)
    out={}
    for group,group_events in events.items():
        starts=[event[0] for event in group_events if event]
        fallback=[]
        for event in group_events:
            fallback.extend((event[i+1]-event[i])*1000 for i in range(len(event)-1))
        counter_gaps=[]
        for a,b in zip(by.get(group,[])[:-1],by.get(group,[])[1:]):
            if a.counter is not None and b.counter is not None and ((a.counter+1)%65536)==b.counter and b.delta_us is not None:
                counter_gaps.append(b.delta_us/1000.0)
        gaps=counter_gaps if counter_gaps else fallback
        event_gaps=[(starts[i+1]-starts[i])*1000 for i in range(len(starts)-1)]
        out[group]=Series(gaps,event_gaps,len(group_events))
    return out


def summarize(series: dict[str,Series], trim: bool) -> list[dict[str,object]]:
    rows=[]
    for group in sorted(series):
        item=series[group]; gaps=list(item.gaps_ms); events=list(item.events_ms)
        if trim:
            gcut=percentile(gaps,99.5); ecut=percentile(events,99.5); gaps=[v for v in gaps if gcut is not None and v<=gcut]; events=[v for v in events if ecut is not None and v<=ecut]
        include=len(gaps)>=50 and len(events)>=50
        reason="" if include else (f"gap samples {len(gaps)} < 50" if len(gaps)<50 else f"adv-event samples {len(events)} < 50")
        rows.append({"adv_address":group,"include":include,"exclude_reason":reason,"gap_count":len(gaps),"gap_median_ms":percentile(gaps,50),"gap_mean_ms":mean(gaps),"gap_std_ms":std(gaps),"gap_iqr_ms":iqr(gaps),"gap_p90_ms":percentile(gaps,90),"gap_p95_ms":percentile(gaps,95),"event_count":item.event_count,"adv_count":len(events),"adv_median_ms":percentile(events,50),"adv_mean_ms":mean(events),"adv_std_ms":std(events),"adv_iqr_ms":iqr(events),"adv_p90_ms":percentile(events,90),"adv_p95_ms":percentile(events,95),"raw_gap_count":len(item.gaps_ms),"raw_adv_count":len(item.events_ms)})
    return rows


def write_csv(path: Path, rows: list[dict[str,object]]) -> None:
    if not rows:return
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def sample_check(series: dict[str, Series]) -> dict[str, object]:
    """Summarize packet/event samples without retaining a second raw copy.

    Values are rounded to 1 microsecond (0.001 ms), the precision of the
    capture timestamps and packet-counter delta field.  Sorting the per-group
    payloads makes the check independent of the anonymized group names while
    still checking every group's complete T_gap/T_advEvent sample vector.
    """
    chunks = []
    gap_samples = event_samples = event_count = 0
    for item in series.values():
        gaps = ",".join(f"{value:.3f}" for value in item.gaps_ms)
        events = ",".join(f"{value:.3f}" for value in item.events_ms)
        chunks.append(f"{gaps}|{events}")
        gap_samples += len(item.gaps_ms)
        event_samples += len(item.events_ms)
        event_count += item.event_count
    payload = "\n".join(sorted(chunks)).encode("utf-8")
    return {
        "groups": len(series),
        "event_count": event_count,
        "gap_samples": gap_samples,
        "event_samples": event_samples,
        "gap_event_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _length(label: str) -> Optional[int]:
    match=re.search(r"-(\d+)B$",label); return int(match.group(1)) if match else None


def plot_compact(root: Path, normal_rows: list[dict[str,object]], embedded_rows: list[dict[str,object]]) -> None:
    benign=[r for r in normal_rows if str(r.get("include")).lower() in {"true","1"}]
    every=[r for r in embedded_rows if r.get("family")=="every"]; last=[r for r in embedded_rows if r.get("family")=="last"]
    lengths=sorted({v for r in every+last for v in [_length(str(r.get("label","")))] if v is not None}); cmap=plt.get_cmap("viridis"); colors={v:cmap(i/max(1,len(lengths)-1)) for i,v in enumerate(lengths)}; jitter={v:(0 if len(lengths)==1 else -0.16+0.32*i/(len(lengths)-1)) for i,v in enumerate(lengths)}; xpos={"benign":0.0,"every":1.0,"last":2.0}
    plt.rcParams.update({"font.size":12,"axes.titlesize":12,"axes.labelsize":12,"xtick.labelsize":11,"ytick.labelsize":11,"legend.fontsize":11})
    def draw(ax,key,logy=False):
        vals=[float(r[key]) for r in benign if r.get(key) not in (None,"")]
        if vals: ax.boxplot(vals,positions=[0],widths=0.38,patch_artist=True,showfliers=True,boxprops={"facecolor":"#d9e2f3","edgecolor":"#4c78a8"},medianprops={"color":"#1f4e79","linewidth":1.6},whiskerprops={"color":"#4c78a8"},capprops={"color":"#4c78a8"},flierprops={"marker":"o","markersize":2.2,"markeredgecolor":"#666666","alpha":0.6})
        for family,items,marker in (("every",every,"^"),("last",last,"s")):
            for r in items:
                if r.get(key) in (None,""): continue
                length=_length(str(r.get("label",""))); x=xpos[family]+jitter.get(length,0); ax.scatter([x],[float(r[key])],marker=marker,s=26,color=colors.get(length,"#333333"),edgecolors="none",zorder=3)
        if logy: ax.set_yscale("log")
        ax.set_ylabel("ms"); ax.grid(alpha=0.25,linestyle="-",linewidth=0.5); ax.set_xticks([0,1,2]); ax.set_xticklabels(["benign","every","last"]); ax.tick_params(axis="both",which="both",direction="in")
    fam=[Line2D([0],[0],marker="^",color="w",markerfacecolor="#666",markersize=6,label="every-*"),Line2D([0],[0],marker="s",color="w",markerfacecolor="#666",markersize=6,label="last-*")]; lens=[Line2D([0],[0],marker="o",color="w",markerfacecolor=colors[v],markersize=5,label=f"{v}B") for v in lengths]; handles=fam+lens
    out=root/"plots"; out.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(7.0,2.9),constrained_layout=True); draw(axes[0],"gap_median_ms"); draw(axes[1],"adv_median_ms",True)
    for ax in axes: ax.legend(handles=handles,loc="upper right",ncol=1,frameon=True,borderaxespad=0.2,handletextpad=0.4,labelspacing=0.3)
    fig.savefig(out/"median_compact_singlecolumn.png",dpi=300); fig.savefig(out/"median_compact_singlecolumn.pdf",metadata=PDF_METADATA); plt.close(fig)
    for key,name,logy in (("gap_median_ms","gap_median_compact_single",False),("adv_median_ms","advevent_median_compact_single",True)):
        fig,ax=plt.subplots(1,1,figsize=(3.45,2.9),constrained_layout=True); draw(ax,key,logy); ax.legend(handles=handles,loc="upper right",bbox_to_anchor=(0.995,0.965),ncol=1,frameon=True,borderaxespad=0.2,handletextpad=0.4,labelspacing=0.3); fig.savefig(out/f"{name}.png",dpi=300); fig.savefig(out/f"{name}.pdf",metadata=PDF_METADATA); plt.close(fig)


def run(root: Path) -> dict[str,int]:
    packet_dir=root/"data/advertising_timing/packets"; normal_packets=read_packets(packet_dir/"benign_population.csv"); normal_series=compute_series(normal_packets,cluster(normal_packets)); normal_full=summarize(normal_series,False); normal_trim=summarize(normal_series,True)
    outputs=root/"outputs"; outputs.mkdir(parents=True,exist_ok=True); write_csv(outputs/"summary_per_address_normal.csv",normal_full); write_csv(outputs/"summary_per_address_normal_full.csv",normal_full); write_csv(outputs/"summary_per_address_normal_trimmed.csv",normal_trim)
    embedded=[]
    for path in sorted(packet_dir.glob("every-*.csv"))+sorted(packet_dir.glob("last-*.csv")):
        packets=read_packets(path); series=compute_series(packets,cluster(packets)); rows=summarize(series,False)
        if not rows: continue
        row=dict(rows[0]); row.update({"label":path.stem,"family":path.stem.split("-",1)[0],"length_B":_length(path.stem),"source_file":path.name}); embedded.append(row)
    base=[r for r in normal_full if r["adv_address"]=="group_baseline" and r["include"]]
    if base:
        row=dict(base[0]); row.update({"label":"embedded-normal","family":"normal","length_B":0,"source_file":"benign_population.csv"}); embedded.append(row)
    write_csv(outputs/"summary_embedded_points.csv",embedded)
    plot_compact(root,normal_full,embedded)
    write_csv(outputs/"timing_counts.csv",[{"normal_packet_records":len(normal_packets),"normal_groups":len(normal_series),"included_benign_groups":sum(bool(r["include"]) for r in normal_full),"embedded_points":len(embedded)}])
    sample_rows = [{"table_file":"benign_population.csv","packet_records":len(normal_packets),**sample_check(normal_series)}]
    for path in sorted(packet_dir.glob("every-*.csv"))+sorted(packet_dir.glob("last-*.csv")):
        packets = read_packets(path)
        series = compute_series(packets, cluster(packets))
        sample_rows.append({"table_file":path.name,"packet_records":len(packets),**sample_check(series)})
    write_csv(outputs/"timing_sample_checks.csv", sample_rows)
    return {"normal_packets":len(normal_packets),"normal_groups":len(normal_series),"included_benign_groups":sum(bool(r["include"]) for r in normal_full),"embedded_points":len(embedded)}
