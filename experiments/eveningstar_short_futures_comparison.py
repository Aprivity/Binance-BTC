"""USD-M BTCUSDT futures 4h: mirrored Evening Star SHORT vs Morning Star LONG.

RESEARCH ONLY, no credential, futures order or real trading. Eight predeclared
short exit configurations ATR14x[1.5,2] × TP[3,5]% × max[24,48]h,
plus unchanged long ATR1.5/TP5/48h and chronological nonoverlapping long+short.

Actual Binance public USD-M futures archive 4h OHLC, not spot prices. Funding
archives attempted for a funding *proxy* diagnostic; headline gross-to-net
price account results include fees/slippage, but NOT exact margin/liquidation
and NOT funding unless explicitly marked as a proxy.
Selection: choose highest development 2020-23 compounded account result across
all methods, then display 2024-26 historical evaluation; this evaluation
period is NOT untouched due to previous long-side parameter research.

No optimized selection is activated in the old spot forward paper program.
"""
from __future__ import annotations
import io
import itertools
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from experiments.candlestick_study import START, TEST_START, END,signals
from experiments.atr_risk_sized_comparison import wilder_atr
from experiments.fixed_bracket_comparison import simulate_bracket as old_long_bracket

OUT=Path("outputs/eveningstar-short-comparison")
ARCH="https://data.binance.vision/data/futures/um"
STEP=4*3600*1000
FEE=.001
SLIP=.0005
ALLOC=.30
ALL_SHORT_CONFIGS=tuple(itertools.product((1.5,2.0),(3.,5.),(6,12)))
SEED=20261010
HORIZON_RESERVE=12
MAX_KLINE_ARCHIVE_MB=10
MAX_FUND_ARCHIVE_MB=6


def download(session,url,max_mb):
    for attempt in range(3):
        try:
            r=session.get(url,timeout=40,stream=True)
            if r.status_code==404:raise FileNotFoundError(url)
            r.raise_for_status()
            max_bytes=int(max_mb*1024*1024)
            if int(r.headers.get("Content-Length",0))>max_bytes:
                raise ValueError("Archive larger than bounded budget")
            payload=bytearray()
            for chunk in r.iter_content(chunk_size=1<<18):
                payload.extend(chunk)
                if len(payload)>max_bytes:
                    raise ValueError("Actual downloaded archive exceeds budget")
            if not payload:raise ValueError("Empty public archive")
            return bytes(payload)
        except (requests.RequestException,ValueError):
            if attempt==2:raise
            time.sleep(2**attempt)
    raise RuntimeError("Unreachable")


def csv_zip(blob):
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        csv=[name for name in z.namelist() if name.endswith(".csv")]
        if len(csv)!=1:raise ValueError("Expected exactly one CSV")
        with z.open(csv[0]) as file:
            return pd.read_csv(file,header=None,low_memory=False)


def epoch_ms(values):
    # 2025+ public archives may use MICROseconds, not ms.
    a=pd.to_numeric(values,errors="raise").to_numpy(dtype=np.int64)
    if len(a) and np.median(a)>1e14:a=a//1000
    return a


def parse_klines(blob):
    raw=csv_zip(blob)
    if raw.shape[1]<6:raise ValueError("Unexpected archive fields")
    first=raw.iloc[:,0].astype(str)
    if first.iloc[0].strip().lower() in ("open_time","opentime"):
        raw=raw.iloc[1:]
    opens=epoch_ms(raw.iloc[:,0])
    out=pd.DataFrame({
        "open_time_ms":opens,
        "open":pd.to_numeric(raw.iloc[:,1],errors="raise").to_numpy(dtype=float),
        "high":pd.to_numeric(raw.iloc[:,2],errors="raise").to_numpy(dtype=float),
        "low":pd.to_numeric(raw.iloc[:,3],errors="raise").to_numpy(dtype=float),
        "close":pd.to_numeric(raw.iloc[:,4],errors="raise").to_numpy(dtype=float),
        "base_volume":pd.to_numeric(raw.iloc[:,5],errors="raise").to_numpy(dtype=float)
    })
    return out


def load_futures(session):
    pieces=[]
    archive_issues=[]
    months=pd.period_range("2020-01","2026-09",freq="M")
    for k,month in enumerate(months,1):
        mon=month.strftime("%Y-%m")
        url=f"{ARCH}/monthly/klines/BTCUSDT/4h/BTCUSDT-4h-{mon}.zip"
        try:
            pieces.append(parse_klines(download(session,url,MAX_KLINE_ARCHIVE_MB)))
        except Exception as err:
            archive_issues.append(f"{mon} {type(err).__name__}:{str(err)[:90]}")
        if k%12==0:
            print("SHORT_FUTURES_DOWNLOAD "+json.dumps(
                {"months":k,"success":len(pieces),"errors":len(archive_issues)}),flush=True)
    for day in pd.date_range("2026-10-01","2026-10-09",freq="D",tz="UTC"):
        stamp=day.strftime("%Y-%m-%d")
        url=f"{ARCH}/daily/klines/BTCUSDT/4h/BTCUSDT-4h-{stamp}.zip"
        try:
            pieces.append(parse_klines(download(session,url,MAX_KLINE_ARCHIVE_MB)))
        except Exception as err:
            archive_issues.append(f"{stamp} {type(err).__name__}:{str(err)[:90]}")
    if not pieces:raise ValueError("No genuine futures data")
    f=pd.concat(pieces,ignore_index=True).sort_values("open_time_ms")
    f=f.drop_duplicates(subset="open_time_ms",keep="first").reset_index(drop=True)
    arr=f[["open","high","low","close","base_volume"]].to_numpy(float)
    if (not np.isfinite(arr).all() or (arr[:,:4]<=0).any() or
            (arr[:,4]<0).any() or
            (f.high<f[["open","close","low"]].max(axis=1)).any() or
            (f.low>f[["open","close","high"]].min(axis=1)).any()):
        raise ValueError("Invalid futures OHLCV")
    stamps=f.open_time_ms.to_numpy(np.int64)
    if np.any(np.diff(stamps)<=0) or (stamps%STEP!=0).any():
        raise ValueError("Misaligned futures data")
    valid=(stamps>=int(START.timestamp()*1000))&(stamps<int(END.timestamp()*1000))
    f=f.loc[valid].reset_index(drop=True)
    stamps=f.open_time_ms.to_numpy(dtype=np.int64)
    gaps=int(np.sum(np.diff(stamps)!=STEP))
    if len(f)<14000 or f.open_time_ms.iloc[0]>int(START.timestamp()*1000) or (
            int(f.open_time_ms.iloc[-1])+STEP<int(END.timestamp()*1000)):
        raise RuntimeError("Insufficient full-period Binance futures archive coverage")
    # Fail closed on MONTH/DAY archive coverage. Individual 4h gaps are excluded.
    if archive_issues:
        raise RuntimeError("Missing futures archived days/months; refuse parameter ranking: "+
                           json.dumps(archive_issues[:8]))
    seg=np.cumsum(np.r_[False,np.diff(stamps)!=STEP])
    return f,seg,gaps


def read_funding(session):
    """Attempt genuine historical funding archives for proxy calculation.

    Most Binance files have header symbol,calc_time,funding_interval_hours,last_funding_rate.
    Require an identifiable timestamp and rate; never silently assume 8h
    cadence or fund=0 if archive fails. Funding proxy does not equal mark-based
    exact actual USDT cashflow, as exit timing within 4h OHLC is unknown.
    """
    chunks=[];missing=[];examples=[]
    for i,month in enumerate(pd.period_range("2020-01","2026-09",freq="M"),1):
        s=month.strftime("%Y-%m")
        url=f"{ARCH}/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-{s}.zip"
        try:
            blob=download(session,url,MAX_FUND_ARCHIVE_MB)
            raw=csv_zip(blob)
            chunks.append(parse_funding(raw))
        except Exception as err:
            missing.append(f"{s}: {type(err).__name__} {str(err)[:90]}")
        if i%24==0:
            print("SHORT_FUNDING_DOWNLOAD "+json.dumps({
                "months_attempted":i,"success":len(chunks),"missing_or_invalid":len(missing)}),flush=True)
    for day in pd.date_range("2026-10-01","2026-10-09",tz="UTC"):
        s=day.strftime("%Y-%m-%d")
        url=f"{ARCH}/daily/fundingRate/BTCUSDT/BTCUSDT-fundingRate-{s}.zip"
        try:chunks.append(parse_funding(csv_zip(download(session,url,MAX_FUND_ARCHIVE_MB))))
        except Exception as err:missing.append(f"{s}: {type(err).__name__} {str(err)[:90]}")
    if not chunks:return pd.DataFrame(columns=["funding_ms","rate"]),missing
    rates=pd.concat(chunks,ignore_index=True).drop_duplicates(subset="funding_ms")
    rates=rates.sort_values("funding_ms").reset_index(drop=True)
    return rates,missing


def parse_funding(raw):
    if raw.empty:raise ValueError("Empty funding archive")
    top=[str(x).strip().lower() for x in raw.iloc[0].tolist()]
    has_header=any("time" in h or "rate" in h for h in top)
    if has_header:
        col={key:i for i,key in enumerate(top)}
        time_ix=next((i for i,v in enumerate(top) if
            v in ("calc_time","fundingtime","funding_time","time")),None)
        rate_ix=next((i for i,v in enumerate(top) if
            v in ("last_funding_rate","fundingrate","funding_rate","rate")),None)
        if time_ix is None or rate_ix is None:
            raise ValueError("Unrecognized funding archive header: "+str(top)[:150])
        frame=raw.iloc[1:]
    else:
        # Most archived USDM fundingRate CSV files are:
        # symbol, calc_time, funding_interval_hours, last_funding_rate.
        if raw.shape[1]<4:raise ValueError("Ambiguous headerless funding schema")
        time_ix,rate_ix=1,3
        frame=raw
    ms=epoch_ms(frame.iloc[:,time_ix])
    rate=pd.to_numeric(frame.iloc[:,rate_ix],errors="raise").to_numpy(float)
    if not np.isfinite(rate).all() or np.any(np.abs(rate)>.03):
        raise ValueError("Implausible funding values")
    return pd.DataFrame({"funding_ms":ms,"rate":rate})


def evening_star(frame):
    """Exact price-inverted mirror of frozen three-candle Morning Star."""
    op=frame.open.to_numpy(float);cl=frame.close.to_numpy(float)
    hi=frame.high.to_numpy(float);lo=frame.low.to_numpy(float)
    b=np.abs(op-cl);ranges=np.maximum(hi-lo,1e-12)
    mask=np.zeros(len(frame),dtype=bool)
    for t in range(20,len(frame)):
        strong_first=(cl[t-2]>op[t-2] and b[t-2]>=.5*ranges[t-2])
        star=(b[t-1]<=.35*b[t-2]) if b[t-2] else False
        strong_third=(op[t]>cl[t] and b[t]>=.5*ranges[t])
        trend=(cl[t-2]>cl[t-6])
        if (strong_first and star and strong_third and trend and
                hi[t-1]>=cl[t-2] and
                cl[t]<=(op[t-2]+cl[t-2])/2):
            mask[t]=True
    return mask


def select_events(mask,segments,phase_start,phase_end):
    """Freeze full 48h reservation for every signal; 24h exits don't refill."""
    stamps=segments.index.to_numpy() if isinstance(segments,pd.Series) else None
    raise NotImplementedError if False else None


def pick_signals(mask,frame,seg,start,end):
    stamps=frame.open_time_ms.to_numpy(dtype=np.int64)
    start_ms=int(start.timestamp()*1000);end_ms=int(end.timestamp()*1000)
    kept=[];reserved=-1
    for raw in np.flatnonzero(mask):
        i=int(raw);entry=i+1;ending=entry+HORIZON_RESERVE
        if i<20 or ending>=len(frame) or stamps[i]<start_ms or stamps[i]>=end_ms:
            continue
        if stamps[ending]>=end_ms or seg[i-20]!=seg[ending]:
            continue
        if entry<=reserved:continue
        kept.append(i)
        reserved=ending
    return kept


def price_bracket(frame,side,signal_i,atr,multiple,tp_pct,hold_bars):
    """Mirror long spot bracket from study; short futures is a modeled trade.

    Intrabar both thresholds hit => STOP first. Gap at unfavorable opening;
    favorable gap at TP limit. All account arithmetic in units of fraction of
    fixed 30% of equity, not return on 1x isolated margin liquidation model.
    """
    op=frame.open.to_numpy(float)
    highs=frame.high.to_numpy(float);lows=frame.low.to_numpy(float)
    i=int(signal_i);entry=i+1
    if side=="long":
        # Exact existing spot-model long convention, applied to futures OHLC.
        buy=op[entry]*(1+SLIP)
        initial_distance=100*atr[i]*multiple/buy
        net,gross,reason,exit_bar,amb=old_long_bracket(
            op,highs,lows,entry,tp_pct,initial_distance,hold_bars)
        raw_exit= op[exit_bar+hold_bars] if False else None
        return float(net),reason,int(exit_bar),bool(amb)
    sell_fill=op[entry]*(1-SLIP)
    stop=sell_fill+multiple*atr[i]
    take=sell_fill*(1-tp_pct/100)
    exit_raw=float(op[entry+hold_bars])
    reason="time_exit";exit_bar=entry+hold_bars;amb=False
    for j in range(entry,entry+hold_bars):
        opening=float(op[j]);high=float(highs[j]);low=float(lows[j])
        if opening>=stop:
            exit_raw,reason,exit_bar=opening,"gap_stop",j;break
        if opening<=take:
            exit_raw,reason,exit_bar=take,"take_profit",j;break
        if high>=stop:
            exit_raw,reason,exit_bar=stop,"stop_loss",j
            amb=bool(low<=take)
            break
        if low<=take:
            exit_raw,reason,exit_bar=take,"take_profit",j;break
    cover_fill=exit_raw*(1+SLIP)
    net=(sell_fill*(1-FEE)-cover_fill*(1+FEE))/sell_fill
    return float(net),reason,int(exit_bar),amb


def make_trades(frame,seg,atr,side,mask,start,end,multiple,tp_pct,hold_bars):
    picked=pick_signals(mask,frame,seg,start,end)
    timestamps=frame.open_time_ms.to_numpy(np.int64)
    result=[]
    for i in picked:
        if not np.isfinite(atr[i]) or atr[i]<=0:raise ValueError("Invalid preentry ATR")
        net,reason,exit_idx,amb=price_bracket(frame,side,i,atr,
                                               multiple,tp_pct,hold_bars)
        result.append({
            "signal_open_ms":int(timestamps[i]),
            "entry_open_ms":int(timestamps[i+1]),
            "exit_bar_open_ms":int(timestamps[exit_idx]),
            "side":side,
            "position_price_net_fraction":net,
            "position_price_net_pct":round(net*100,5),
            "account_net_pct":round(net*ALLOC*100,5),
            "reason":reason,
            "ambiguous_stop_first":bool(amb),
            "entry_year":int(pd.Timestamp(timestamps[i+1],unit="ms",tz="UTC").year),
            "entry_month":pd.Timestamp(timestamps[i+1],unit="ms",tz="UTC").strftime("%Y-%m"),
            "atr_signal":float(atr[i]),
            "initial_stop_pct":round(100*multiple*atr[i]/frame.open.iloc[i+1],4),
            "max_holding_hours":hold_bars*4,
            "stop_multiple":multiple,
            "tp_pct":tp_pct,
        })
    return result


def combine(long_trades,short_trades):
    """Chronological nonoverlap with 48h reservation, long wins equal-time ties."""
    candidates=sorted([*long_trades,*short_trades],
       key=lambda t:(t["entry_open_ms"],0 if t["side"]=="long" else 1))
    out=[];reservation_ends=-1
    for t in candidates:
        if t["entry_open_ms"]<=reservation_ends:continue
        out.append(t)
        reservation_ends=t["entry_open_ms"]+12*STEP
    return out


def add_funding_proxy(trades,frame,rates,complete):
    """Funding before exit bar open only (intrabar cash flows unresolved).

    Uses closest previous completed 4h futures trade close, NOT true mark
    price at funding interval, so it's an explanatory approximation.
    """
    stamps=frame.open_time_ms.to_numpy(np.int64)
    fts=rates.funding_ms.to_numpy(np.int64)
    rs=rates.rate.to_numpy(float)
    closes=frame.close.to_numpy(float)
    for t in trades:
        entry=int(t["entry_open_ms"]);exit_ms=int(t["exit_bar_open_ms"])
        a=int(np.searchsorted(fts,entry,side="right"))
        b=int(np.searchsorted(fts,exit_ms,side="left"))
        if not complete:
            t["funding_proxy_fraction"]=None
            t["position_with_funding_proxy_pct"]=None
            continue
        if a>=b:
            fund=0.
        else:
            indices=np.maximum(0,np.searchsorted(stamps,fts[a:b],side="left")-1)
            entry_idx=int(np.searchsorted(stamps,entry,side="left"))
            entry_p=float(frame.open.iloc[entry_idx])
            side=1 if t["side"]=="long" else -1
            fund=float(np.sum(-side*rs[a:b]*(closes[indices]/entry_p)))
        t["funding_proxy_fraction"]=fund
        t["position_with_funding_proxy_pct"]=round(
            (t["position_price_net_fraction"]+fund)*100,5)


def performance(trades):
    if not trades:
        return {"trades":0,"account_compounded_pct":0,"win_rate_pct":None}
    arr=np.asarray([t["position_price_net_fraction"] for t in trades],dtype=float)
    a=ALLOC*arr;equity=np.r_[1.,np.cumprod(1+a)]
    mdd=equity/np.maximum.accumulate(equity)-1
    lose=longest=0
    for v in arr:
        if v<0:
            lose+=1;longest=max(longest,lose)
        else:lose=0
    fund=[t["funding_proxy_fraction"] for t in trades]
    fund_equity=(np.prod(1+ALLOC*(arr+np.array(fund))) if
                 all(v is not None for v in fund) else None)
    tail=max(1,int(np.ceil(len(arr)*.05)))
    return {
       "trades":len(arr),
       "long_count":sum(t["side"]=="long" for t in trades),
       "short_count":sum(t["side"]=="short" for t in trades),
       "win_rate_pct":round(float(np.mean(arr>0)*100),2),
       "position_avg_net_pct":round(float(np.mean(arr)*100),4),
       "position_median_net_pct":round(float(np.median(arr)*100),4),
       "account_compounded_pct":round(float((equity[-1]-1)*100),4),
       "account_trade_close_mdd_pct":round(float(mdd.min()*100),4),
       "account_worst_event_pct":round(float(np.min(a)*100),4),
       "account_bottom5pct_mean_pct":round(float(np.sort(a)[:tail].mean()*100),4),
       "max_loss_streak":longest,
       "take_count":sum(t["reason"]=="take_profit" for t in trades),
       "stop_count":sum("stop" in t["reason"] for t in trades),
       "time_count":sum(t["reason"].startswith("time") for t in trades),
       "ambiguous":sum(t["ambiguous_stop_first"] for t in trades),
       "account_with_funding_proxy_pct":round(float((fund_equity-1)*100),4) if
          fund_equity is not None else None
    }


def unit_tests():
    # Mirrored pattern passes ONLY the short detector, not long.
    f=pd.DataFrame({
       "open":np.full(40,100.),
       "close":np.full(40,100.),
       "high":np.full(40,101.),
       "low":np.full(40,99.),
    })
    t=30
    f.loc[t-6,"close"]=95.
    f.loc[t-2,["open","close","high","low"]]=[97.,103.,104.,96.]
    f.loc[t-1,["open","close","high","low"]]=[103.,103.3,104.,102.8]
    f.loc[t,["open","close","high","low"]]=[103.,99.,104.,98.]
    assert evening_star(f)[t] and not signals(f)["morning_star"][t]
    # All short trade payoffs adverse for increasing BTC price, good for fall.
    n=25
    x=pd.DataFrame({"open":np.full(n,100.),"high":np.full(n,101.),
             "low":np.full(n,99.),"close":np.full(n,100.)})
    a=np.full(n,2.)
    val,reason,*_=price_bracket(x,"short",0,a,1.5,5.,6)
    assert reason=="time_exit" and val<0
    x.loc[1,"low"]=94.
    val,reason,*_=price_bracket(x,"short",0,a,1.5,5.,6)
    assert reason=="take_profit" and val>0
    x.loc[1,"high"]=105.
    val,reason,*_=price_bracket(x,"short",0,a,1.5,5.,6)
    assert reason=="stop_loss" and val<0


def main():
    unit_tests()
    OUT.mkdir(parents=True,exist_ok=True)
    session=requests.Session()
    frame,seg,gaps=load_futures(session)
    longs=signals(frame)["morning_star"]
    shorts=evening_star(frame)
    atr=wilder_atr(frame.high.to_numpy(float),frame.low.to_numpy(float),
        frame.close.to_numpy(float),seg)
    rates,missing_funding=read_funding(session)
    funding_complete=not missing_funding
    print("SHORT_SOURCE "+json.dumps({
        "future_bars":len(frame),"gaps":gaps,
        "funding_entries":len(rates),"funding_archives_missing":len(missing_funding),
        "funding_complete":funding_complete,
    }),flush=True)
    if int(np.sum(longs & shorts))!=0:raise ValueError("Long and short same signal")
    records=[];yearly=[];detail=[];evidence=[];phase_runs={}
    arms=[("long_original_4h_ATR1.5_TP5_48h","long",1.5,5.,12)]
    for mult,tp,bars in ALL_SHORT_CONFIGS:
        arms.append((f"short_eveningstar_ATR{mult:g}_TP{tp:g}_{bars*4}h",
                    "short",mult,tp,bars))
    for phase,start,end in [
       ("development_2020_2023",START,TEST_START),
       ("evaluation_2024_2026",TEST_START,END)
    ]:
        by_name={}
        for label,side,mult,tp,bars in arms:
            arr=make_trades(frame,seg,atr,side,
                longs if side=="long" else shorts,start,end,mult,tp,bars)
            add_funding_proxy(arr,frame,rates,funding_complete)
            by_name[label]=arr
            for t in arr:detail.append({"phase":phase,"method":label,**t})
            result={"phase":phase,"method":label,"direction":side,
                    "atr_multiple":mult,"take_profit_pct":tp,"max_holding_hours":bars*4,
                    **performance(arr)}
            records.append(result)
            print("SHORT_METHOD "+json.dumps(result),flush=True)
        for short_name,side,mult,tp,bars in arms[1:]:
            label="both_"+short_name
            arr=combine(by_name[arms[0][0]],by_name[short_name])
            by_name[label]=arr
            result={"phase":phase,"method":label,"direction":"long_plus_short",
                "atr_multiple":mult,"take_profit_pct":tp,
                "max_holding_hours":bars*4,**performance(arr)}
            records.append(result)
            print("SHORT_METHOD "+json.dumps(result),flush=True)
        for name,arr in by_name.items():
            for yr in sorted(set(t["entry_year"] for t in arr)):
                sub=[t for t in arr if t["entry_year"]==yr]
                out={"phase":phase,"method":name,"year":yr,**performance(sub)}
                yearly.append(out)
        phase_runs[phase]=by_name
    summary=pd.DataFrame(records)
    dev=summary.loc[summary.phase=="development_2020_2023"]
    ev=summary.loc[summary.phase=="evaluation_2024_2026"]
    best_dev=dev.sort_values(["account_compounded_pct","trades"],
               ascending=[False,False]).iloc[0].to_dict()
    best_ev=ev.sort_values(["account_compounded_pct","trades"],
               ascending=[False,False]).iloc[0].to_dict()
    frozen=ev.loc[ev.method==best_dev["method"]].iloc[0].to_dict()
    print("SHORT_WINNERS "+json.dumps({
       "highest_development":best_dev,
       "chosen_development_winner_on_2024_2026":frozen,
       "highest_inspected_2024_2026_hindsight":best_ev,
       "WARNING":"2024+ previously inspected for existing LONG choices; no untouched test. Does NOT include exact liquidation and funding/mark at exit."
    },ensure_ascii=False),flush=True)
    summary.to_csv(OUT/"all_method_comparison.csv",index=False)
    pd.DataFrame(yearly).to_csv(OUT/"yearly_method_results.csv",index=False)
    pd.DataFrame(detail).to_csv(OUT/"single_side_trade_audit.csv",index=False)
    (OUT/"winners.json").write_text(json.dumps({
       "selected_by_2020_2023_development":best_dev,
       "frozen_selected_on_2024_2026":frozen,
       "retrospective_highest_2024_2026":best_ev
    },ensure_ascii=False,indent=2),encoding="utf8")
    notes={
      "research_only":True,"none_of_this_trades_or_changes_paper":True,
      "market":"Binance BTCUSDT USD-M perpetual futures ORIGINAL real 4h traded-price OHLC archive",
      "history":["2020-01-01","2026-10-10 excluded"],"split":"2024-01-01",
      "spot_baseline_old_study":"Original BTC SPOT 4h long 2020-23 53 trades +11.2231%, 2024-26 44 trades +16.2238%; DO NOT directly equate with futures quotes",
      "signal_long":"Existing exact 3-candle morning star from experiments/candlestick_study.py",
      "signal_short":"Three-candle EVENING STAR reverse of original: strong bullish first, small middle, strong bearish third; prior uptrend; third close <= bullish midpoint, middle high >= bullish first close; 24/7 no gap",
      "long_frozen":{"atr_period":14,"stop_atr_multiple":1.5,"take_profit_pct":5,
                     "max_hold_hours":48,"position_fraction_of_equity":.30},
      "short_grid":{"atr_stop_multiple":[1.5,2],"take_profit_pct":[3,5],
                    "max_hold_hours":[24,48],"kline_hours":4,"atr_period":14},
      "direction":"1x NOTIONAL synthetic USD-M long/short proxy; 30% equity notional per trade, 70% idle, no leverage optimization",
      "entry":"Confirmation 4h bar closes, enter NEXT 4h bar OPEN",
      "cohort":"All single-direction parameter variants freeze the SAME 48h reserved nonoverlapping signal cohort (even 24h exit variants), and reject gaps in previous20 bars and following full48h; combined chronological picks whichever long or short first, reserves 48h after entry.",
      "stop":"Fixed initial price distance ATR(14) Wilder computed on completed third signal bar; not trailing",
      "exit":"ATR stop, fixed % TP, or at next bar OPEN when max 24/48h passes. STOP first for intrabar TP/SL collision, gap stop at worse open, TP favorable gap caps at limit.",
      "transaction_cost":"Fee10bps + slippage5bps on entry AND exit, no modeled maker discounts; short entry sells below spot future OHLC open, cover above exit price",
      "funding":"FundingRate archive attempted each month/day; account_with_funding_proxy_pct only if 100% archive coverage. Funding proxy uses funding rates known over complete bars and previous completed 4h traded close/entry price instead of true mark, and does not resolve funding in exit bar.",
      "short_market_risks":["Short futures not enabled in repo, position prices and fees are model assumptions",
        "No account margin maintenance/liquidation engine, funding cash-flow may be incomplete, contracts mark/last difference and ADL not simulated",
        "Short has theoretically uncapped price downside exposure and liquidation risk; ATR stop execution is not guaranteed",
        "Binance price kline archives are last-trade data, not a complete executable order book",
        "Returns are account simulation from 30% position notional, not margin ROI",
        "Equity curve MDD is TRADE CLOSE only, not intratrade MTM",
        "Short historical profitability changes significantly with funding and spreads; do not deploy from these results"],
      "selection_bias":["2020-2023 development parameter grid chosen highest development account compounded; 2024-2026 evaluation repeated prior LONG-side research and not untouched",
        "2024-2026 highest observed is a HINDSIGHT winner, not statistically valid or recommended live parameter selection",
        "A 24h max exit with frozen 48h signal reservation deliberately for fairness, missing potential extra 24h entries",
        "No corrected multiple-testing inference and no true forward/OOS proof"],
      "data_archives": {"futures_4h_candles":int(len(frame)),
            "4h_gaps":gaps,"funding_entries":len(rates),
            "funding_missing_count":len(missing_funding),
            "funding_missing_examples":missing_funding[:12]},
    }
    (OUT/"methodology.json").write_text(json.dumps(notes,ensure_ascii=False,indent=2),encoding="utf8")
    print("SHORT_COMPARISON_COMPLETE",flush=True)


if __name__=="__main__":
    main()
