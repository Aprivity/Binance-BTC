"""BTC 4h Morning Star: point-in-time Volume Profile exploratory study.

No orders, no active strategy, no main merge. A full 7/30d REAL aggTrades
reconstruction is NOT claimed: broad-sample VP here distributes each real
Binance 5m candle's executed BTC volume at its HLC3 typical price, a PROXY.
Exact aggregate-trade 24h snapshots are optionally checked on two event dates
to quantify proxy mismatch; neither is a historical orderbook/heatmap.

Input: original non-overlapping 4h signal cohort 53+44 events, unchanged
ATR(14)*1.5 initial stop / TP5% / max 48h / 30% fixed equity exposure.
"""
from __future__ import annotations

import hashlib
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from experiments.candlestick_study import START, TEST_START, END, fetch_gap_aware, signals
from experiments.fixed_bracket_comparison import cohort, simulate_bracket
from experiments.atr_risk_sized_comparison import wilder_atr

OUT=Path("outputs/morningstar-volume-profile-pilot")
ARCHIVE="https://data.binance.vision/data/spot"
STEP5=300_000
STEP4=14_400_000
WINDOWS=(7,30)
BIN_LOG=float(np.log1p(.0025))
ALLOCATION=.30
TP_PCT=5.
ATR_MULT=1.5
MAX_HOLD=12
SEED=20261010
MAX_ZIP_MB=85
MIN_PROFILE_COVERAGE=.995


def get_bytes(s,url,max_mb=MAX_ZIP_MB):
    for retry in range(3):
        try:
            r=s.get(url,timeout=55,stream=True)
            if r.status_code==404:
                raise FileNotFoundError(url)
            r.raise_for_status()
            declared=int(r.headers.get("Content-Length","0") or "0")
            if declared>max_mb*1024*1024:
                raise ValueError(f"Archive size exceeds {max_mb}MB safety cap: {declared}")
            chunks=[];size=0
            for part in r.iter_content(chunk_size=1<<18):
                size+=len(part)
                if size>max_mb*1024*1024:
                    raise ValueError("Downloaded archive exceeded size budget")
                chunks.append(part)
            blob=b"".join(chunks)
            if not blob:
                raise ValueError("Empty exchange archive")
            return blob
        except (requests.RequestException,ValueError) as err:
            if retry==2: raise
            time.sleep(2**retry)
    raise RuntimeError("Unreachable")


def read_archive_zip(blob):
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        files=[x for x in z.namelist() if x.endswith(".csv")]
        if len(files)!=1:
            raise ValueError("Expected exactly one Binance archive CSV")
        with z.open(files[0]) as f:
            return pd.read_csv(f,header=None)


def epoch_milliseconds(series):
    raw=np.asarray(pd.to_numeric(series,errors="raise"),dtype=np.int64)
    if len(raw)==0: return raw
    if np.nanmedian(raw)>1e14:
        if np.any(raw%1000!=0):
            # aggTrades microsecond values are not always aligned to a ms:
            # sub-ms precision is discarded only for UTC day comparison.
            pass
        raw=raw//1000
    return raw


def monthly_candles(session,month):
    ym=month.strftime("%Y-%m")
    url=f"{ARCHIVE}/monthly/klines/BTCUSDT/5m/BTCUSDT-5m-{ym}.zip"
    raw=read_archive_zip(get_bytes(session,url,30))
    if raw.shape[1]<6: raise ValueError("Malformed 5m public archive")
    dates=epoch_milliseconds(raw.iloc[:,0])
    if len(dates)>1 and (np.diff(dates)<=0).any():
        raise ValueError("Nonmonotonic 5m archive timestamps")
    return pd.DataFrame({
        "open_ms":dates,
        "high":pd.to_numeric(raw.iloc[:,2],errors="raise").to_numpy(float),
        "low":pd.to_numeric(raw.iloc[:,3],errors="raise").to_numpy(float),
        "close":pd.to_numeric(raw.iloc[:,4],errors="raise").to_numpy(float),
        "volume":pd.to_numeric(raw.iloc[:,5],errors="raise").to_numpy(float),
    })


def end_of_history_5m(session):
    """Last incomplete archive month from public REST; no future candles."""
    end_ms=int(END.timestamp()*1000)
    current_month=END.normalize().replace(day=1)
    cursor=int(current_month.timestamp()*1000)
    url="https://data-api.binance.vision/api/v3/klines"
    records=[]
    while cursor<end_ms:
        r=session.get(url,params={"symbol":"BTCUSDT","interval":"5m",
                        "startTime":cursor,"endTime":end_ms-1,"limit":1000},timeout=35)
        r.raise_for_status()
        raw=r.json()
        if not raw:break
        for item in raw:
            ms=int(item[0])
            if ms+STEP5>end_ms:continue
            records.append((ms,float(item[2]),float(item[3]),float(item[4]),float(item[5])))
        new=int(raw[-1][0])+STEP5
        if new<=cursor:raise ValueError("REST pagination stall")
        cursor=new
        time.sleep(.07)
    return pd.DataFrame(records,columns=["open_ms","high","low","close","volume"])


def fetch_5m_history(session):
    # Includes 2019-12 for early 2020 signals' full previous 30d windows.
    periods=pd.period_range("2019-12","2026-09",freq="M")
    pieces=[];issues=[]
    for count,p in enumerate(periods,1):
        try:
            piece=monthly_candles(session,p)
            pieces.append(piece)
        except Exception as e:
            issues.append({"month":str(p),"error":f"{type(e).__name__}: {str(e)[:180]}"})
        if count%12==0:
            print("PROFILE_ARCHIVE_PROGRESS "+json.dumps({
                "months_checked":count,"months_with_data":len(pieces),
                "failures":len(issues)}),flush=True)
    # October 2026 is not yet a published monthly archive.
    try:
        pieces.append(end_of_history_5m(session))
    except Exception as e:
        issues.append({"month":"2026-10 REST","error":f"{type(e).__name__}: {str(e)[:180]}"})
    if not pieces:raise RuntimeError("No historical 5m candles available")
    merged=pd.concat(pieces,ignore_index=True).sort_values("open_ms")
    merged=merged.drop_duplicates(subset=["open_ms"],keep="first").reset_index(drop=True)
    if not np.isfinite(merged[["high","low","close","volume"]].to_numpy()).all():
        raise ValueError("Non-finite 5m OHLCV")
    if (merged.volume<0).any() or (merged.close<=0).any() or (
        merged.high<merged[["low","close"]].max(axis=1)).any() or (
        merged.low>merged[["high","close"]].min(axis=1)).any():
        raise ValueError("Corrupt historical 5m bars")
    print("PROFILE_5M_COVERAGE "+json.dumps({
        "bars":int(len(merged)),"missing_archives":len(issues),
        "first_utc":pd.to_datetime(merged.open_ms.iloc[0],unit="ms",utc=True).isoformat(),
        "last_utc":pd.to_datetime(merged.open_ms.iloc[-1],unit="ms",utc=True).isoformat(),
        "issues":issues[:12]}),flush=True)
    return merged,issues


def profile_from_price_qty(prices,qty,reference):
    """Relative log-price bins 0.25%, anchored to known confirmation CLOSE."""
    if len(prices)==0 or reference<=0:raise ValueError("Empty profile")
    b=np.floor(np.log(prices/reference)/BIN_LOG).astype(np.int64)
    positions,rev=np.unique(b,return_inverse=True)
    sums=np.bincount(rev,weights=qty)
    weights=np.asarray(sums,float)
    mass=float(weights.sum())
    if mass<=0:raise ValueError("Zero historical traded volume")
    return positions,weights,mass


def profile_features(prices,volumes,reference,window_days,method):
    bins,weights,total=profile_from_price_qty(prices,volumes,reference)
    k=int(np.argmax(weights))
    poc=int(bins[k])
    level_mid=lambda idx:float(reference*np.exp((idx+.5)*BIN_LOG))
    poc_price=level_mid(poc)
    # HVN is a *local maximum* >=70% of this window's POC volume,
    # strictly on the appropriate side of signal reference.
    left=np.r_[-np.inf,weights[:-1]]
    right=np.r_[weights[1:],-np.inf]
    peaks=(weights>=left)&(weights>=right)&(weights>=.70*weights.max())
    above=np.where(peaks&(bins>=0))[0]
    below=np.where(peaks&(bins<0))[0]
    next_above=level_mid(int(bins[above[0]])) if len(above) else None
    next_below=level_mid(int(bins[below[-1]])) if len(below) else None
    # On all historical executed *volume*, percent within [entry, entry+5%].
    span_hi=np.log(1.05)/BIN_LOG
    span_lo=np.log(.95)/BIN_LOG
    above5=float(weights[(bins>=0)&(bins<span_hi)].sum()/total)
    below5=float(weights[(bins<0)&(bins>=span_lo)].sum()/total)
    # "density" signals crowded market area, not actual order-book sell walls.
    return {
        "window_days":window_days,"method":method,
        "poc_distance_pct":float(100*(poc_price/reference-1)),
        "poc_volume_share_pct":float(100*weights[k]/total),
        "above_0_to_5pct_volume_share_pct":above5*100,
        "below_minus5_to_0pct_volume_share_pct":below5*100,
        "upper_minus_lower_5pct_share_pp":(above5-below5)*100,
        "nearest_upper_hvn_distance_pct":float(100*(next_above/reference-1)) if next_above else None,
        "nearest_lower_hvn_distance_pct":float(100*(next_below/reference-1)) if next_below else None,
        "volume_total_btc":total,
        "bins":len(bins)
    }


def lookback_slice(five,decision_ms,days):
    start=decision_ms-days*86_400_000
    ms=five.open_ms.to_numpy(dtype=np.int64)
    i=int(np.searchsorted(ms,start,side="left"))
    j=int(np.searchsorted(ms,decision_ms,side="left"))
    sl=five.iloc[i:j]
    # Do not fill missing historical spot 5m candles. If coverage incomplete,
    # mark the feature undefined, avoid fake POC or invisible missing data.
    expected=days*288
    complete=(len(sl)==expected and (len(sl)==0 or (
        int(sl.open_ms.iloc[0])==start and
        int(sl.open_ms.iloc[-1])==decision_ms-STEP5 and
        np.all(np.diff(sl.open_ms.to_numpy(np.int64))==STEP5)
    )))
    return sl,complete,expected


def correlation_and_median(frame,feature,phase,days):
    vals=pd.to_numeric(frame[feature],errors="coerce").to_numpy(float)
    ret=frame["trade_net_pct"].to_numpy(float)
    ok=np.isfinite(vals)&np.isfinite(ret)
    n=int(ok.sum())
    if n<8:return {"phase":phase,"days":days,"feature":feature,"n":n,"too_few":True}
    rank_corr=float(np.corrcoef(pd.Series(vals[ok]).rank(method="average").to_numpy(),pd.Series(ret[ok]).rank(method="average").to_numpy())[0,1])
    median=float(np.median(vals[ok]))
    a=ret[ok&(vals<median)]
    b=ret[ok&(vals>=median)]
    return {
        "phase":phase,"window_days":days,"feature":feature,
        "valid_pairs":n,"observed_spearman":round(rank_corr,4),
        "median_threshold_descriptive_only":round(median,5),
        "below_median_n":int(len(a)),
        "above_or_equal_median_n":int(len(b)),
        "below_median_mean_net_pct":round(float(a.mean()),4) if len(a) else None,
        "above_median_mean_net_pct":round(float(b.mean()),4) if len(b) else None,
        "WARNING":"Descriptive within-period median splits, NOT trade filters or independent predictive evidence"
    }


def exact_trade_day(session,signal,day,five):
    """One public daily spot aggTrades CSV for REAL volume-at-price audit.

    Compare event's actual 24h aggregate-trade volume profile to approximate
    5m HLC3-as-single-price profile, using same price bin reference.
    """
    date=day.strftime("%Y-%m-%d")
    url=f"{ARCHIVE}/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-{date}.zip"
    blob=get_bytes(session,url,max_mb=MAX_ZIP_MB)
    raw=read_archive_zip(blob)
    if raw.shape[1]<7:raise ValueError("Malformed aggTrades CSV")
    prices=pd.to_numeric(raw.iloc[:,1],errors="raise").to_numpy(float)
    quantities=pd.to_numeric(raw.iloc[:,2],errors="raise").to_numpy(float)
    trade_ts=epoch_milliseconds(raw.iloc[:,5])
    day_open=int(day.timestamp()*1000)
    day_end=day_open+86_400_000
    valid=(trade_ts>=day_open)&(trade_ts<day_end)
    if np.mean(valid)<.999:
        raise ValueError("Archive contains unexpected out-of-day trades")
    if not np.isfinite(prices).all() or not np.isfinite(quantities).all():
        raise ValueError("Bad daily public aggTrade values")
    if (prices<=0).any() or (quantities<0).any():raise ValueError("Nonpositive trade data")
    market=five[(five.open_ms>=day_open)&(five.open_ms<day_end)]
    if len(market)!=288 or not np.all(np.diff(market.open_ms.to_numpy(np.int64))==STEP5):
        raise ValueError("Incomplete 5m day for true-trade comparison")
    price5=(market.high.to_numpy(float)+market.low.to_numpy(float)+market.close.to_numpy(float))/3
    qty5=market.volume.to_numpy(float)
    ref=float(signal["signal_close"])
    a,w,total=profile_from_price_qty(prices,quantities,ref)
    b,u,proxy_total=profile_from_price_qty(price5,qty5,ref)
    all_bins=np.union1d(a,b)
    exact_v=np.zeros(len(all_bins));proxy_v=np.zeros(len(all_bins))
    exact_v[np.searchsorted(all_bins,a)]=w/total
    proxy_v[np.searchsorted(all_bins,b)]=u/proxy_total
    tvd=float(.5*np.abs(exact_v-proxy_v).sum())
    fine=profile_features(prices,quantities,ref,1,"exact_public_aggTrades")
    crude=profile_features(price5,qty5,ref,1,"5m_hlc3_volume_proxy")
    row={
        "signal_utc":signal["signal_utc"],"trade_day":date,
        "agg_trades":int(len(prices)),"agg_archive_bytes":len(blob),
        "spot_agg_volume_btc":round(total,6),
        "spot_5m_volume_btc":round(proxy_total,6),
        "aggregate_trades_minus_5m_volume_pct":round(100*(total/proxy_total-1),5),
        "distribution_total_variation_distance":round(tvd,5),
        "exact_POC_distance_pct":round(fine["poc_distance_pct"],5),
        "proxy_POC_distance_pct":round(crude["poc_distance_pct"],5),
        "exact_above5_volume_share_pct":round(fine["above_0_to_5pct_volume_share_pct"],4),
        "proxy_above5_volume_share_pct":round(crude["above_0_to_5pct_volume_share_pct"],4),
        "representativeness":"ONE historical UTC day only; not proof that 7/30 day proxies are accurate",
    }
    return row


def tests():
    ms=np.arange(100,100+7*288,dtype=np.int64)*STEP5
    frame=pd.DataFrame({"open_ms":ms,"high":np.full(len(ms),101.),
        "low":np.full(len(ms),99.),"close":np.full(len(ms),100.),
        "volume":np.ones(len(ms))})
    decision=int(ms[-1]+STEP5)
    sl,complete,n=lookback_slice(frame,decision,7)
    assert complete and len(sl)==2016
    dropped=frame.drop(index=2).reset_index(drop=True)
    assert not lookback_slice(dropped,decision,7)[1]
    ex=profile_features(np.array([100.,101.,105.]),
          np.array([10.,2.,1.]),100.,7,"synthetic_test")
    assert ex["poc_distance_pct"]<1 and ex["volume_total_btc"]==13.
    # zero hindsight: candle beginning at decision is explicitly excluded.
    future=pd.concat([frame,pd.DataFrame({"open_ms":[decision],
       "high":[1e8],"low":[1.],"close":[1e8],"volume":[1e9]})],ignore_index=True)
    assert lookback_slice(future,decision,7)[0].volume.sum()==2016


def main():
    tests()
    OUT.mkdir(parents=True,exist_ok=True)
    session=requests.Session()
    frame,gaps,segments=fetch_gap_aware("4h")
    ts=pd.to_datetime(frame.open_time_ms,unit="ms",utc=True)
    signal=signals(frame)["morning_star"]
    atr=wilder_atr(frame.high.to_numpy(float),
              frame.low.to_numpy(float),frame.close.to_numpy(float),segments)
    candles5,archive_issues=fetch_5m_history(session)
    op=frame.open.to_numpy(float)
    hi=frame.high.to_numpy(float)
    lo=frame.low.to_numpy(float)
    close=frame.close.to_numpy(float)
    all_rows=[];coverage=[]
    for phase,start,end in (("development_2020_2023",START,TEST_START),
            ("evaluation_2024_2026",TEST_START,END)):
        indices=cohort(frame,segments,signal,start,end)
        expected=53 if phase.startswith("development") else 44
        if len(indices)!=expected:raise RuntimeError("Baseline cohort changed")
        for i in indices:
            entry=int(i)+1
            signal_end_ms=int(frame.open_time_ms.iloc[i])+STEP4
            entry_ts=int(frame.open_time_ms.iloc[entry])
            if signal_end_ms!=entry_ts:raise RuntimeError("Unexpected 4h gap")
            fill=op[entry]*1.0005
            sl_pct=float(ATR_MULT*atr[i]/fill*100)
            net,gross,reason,idx,amb=simulate_bracket(
                op,hi,lo,entry,TP_PCT,sl_pct,MAX_HOLD)
            rec={
                "phase":phase,"signal_utc":ts.iloc[i].isoformat(),
                "signal_confirmed_utc":pd.to_datetime(signal_end_ms,unit="ms",utc=True).isoformat(),
                "entry_utc":ts.iloc[entry].isoformat(),
                "signal_close":float(close[i]),
                "entry_open":float(op[entry]),
                "trade_net_pct":round(float(net*100),5),
                "account_return_pct":round(float(.30*net*100),5),
                "trade_exit_reason":reason,
                "stop_pct":round(sl_pct,5),
                "ambiguous_exit":bool(amb)
            }
            for days in WINDOWS:
                window,complete,n=lookback_slice(candles5,signal_end_ms,days)
                rec[f"w{days}_coverage_bars"]=len(window)
                rec[f"w{days}_expected_bars"]=n
                rec[f"w{days}_complete"]=bool(complete)
                if complete:
                    typical=(window.high.to_numpy(float)+window.low.to_numpy(float)
                              +window.close.to_numpy(float))/3
                    feat=profile_features(typical,window.volume.to_numpy(float),
                                    close[i],days,"Binance_5m_HLC3_volume_proxy")
                    for key,value in feat.items():
                        if key not in ("window_days","method"):
                            rec[f"w{days}_{key}"]=round(value,6) if isinstance(value,float) else value
                else:
                    coverage.append({"phase":phase,"signal_utc":rec["signal_utc"],
                             "window_days":days,"observed":len(window),"expected":n})
            all_rows.append(rec)
        print("PROFILE_COHORT "+json.dumps({"phase":phase,"signals":len(indices),
            "profiles7_complete":int(sum(x["w7_complete"] for x in all_rows if x["phase"]==phase)),
            "profiles30_complete":int(sum(x["w30_complete"] for x in all_rows if x["phase"]==phase))}),flush=True)
    study=pd.DataFrame(all_rows)
    fields=[
        "poc_distance_pct","poc_volume_share_pct",
        "above_0_to_5pct_volume_share_pct",
        "below_minus5_to_0pct_volume_share_pct",
        "upper_minus_lower_5pct_share_pp",
        "nearest_upper_hvn_distance_pct","nearest_lower_hvn_distance_pct",
    ]
    diagnostics=[]
    for phase,group in study.groupby("phase"):
        for days in WINDOWS:
            for f in fields:
                row=correlation_and_median(group,f"w{days}_{f}",phase,days)
                diagnostics.append(row)
                print("PROFILE_DIAGNOSTIC "+json.dumps(row),flush=True)
    # Exact aggTrades daily audit uses a small date-selected sample,
    # never uses post-entry transaction data for signal features.
    audit=[]
    for phase in ("development_2020_2023","evaluation_2024_2026"):
        g=study[(study["phase"]==phase)&(study["w7_complete"])]
        if g.empty:continue
        # 2 deterministic trades nearest 33% and 67% of cohort chronology.
        for ix in sorted(set((max(0,len(g)//3-1),min(len(g)-1,2*len(g)//3)))):
            record=g.iloc[ix].to_dict()
            confirmed=pd.Timestamp(record["signal_confirmed_utc"])
            # Use previous FULL UTC day; never let realized next-trade
            # 24h volume enter a pre-entry "exact" diagnostic.
            prior_day=(confirmed-pd.Timedelta(days=1)).normalize()
            if prior_day+pd.Timedelta(days=1)>confirmed:
                raise AssertionError("Exact audit observes future")
            try:
                result=exact_trade_day(session,record,prior_day,candles5)
                audit.append({"status":"ok",**result})
                print("PROFILE_AGGTRADE_AUDIT "+json.dumps({"status":"ok",**result}),flush=True)
            except Exception as e:
                failure={"status":"not_verified","phase":phase,"signal_utc":record["signal_utc"],
                    "trade_day":str(prior_day.date()),"error":f"{type(e).__name__}: {str(e)[:160]}"}
                audit.append(failure)
                print("PROFILE_AGGTRADE_AUDIT "+json.dumps(failure),flush=True)
    study.to_csv(OUT/"signal_preentry_profile_features.csv",index=False)
    pd.DataFrame(diagnostics).to_csv(OUT/"descriptive_feature_diagnostics.csv",index=False)
    pd.DataFrame(coverage).to_csv(OUT/"missing_5m_lookback_windows.csv",index=False)
    pd.DataFrame(audit).to_csv(OUT/"exact_aggtrades_proxy_daily_audit.csv",index=False)
    # Representative legible 7d VP chart, only if complete. The plotted
    # histogram uses real Kline volumes assigned to HLC3, NOT actual trades.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ex=study[(study.phase=="evaluation_2024_2026")&study.w7_complete].iloc[-1]
        end=int(pd.Timestamp(ex["signal_confirmed_utc"]).timestamp()*1000)
        small,valid,_=lookback_slice(candles5,end,7)
        p=(small.high.to_numpy(float)+small.low.to_numpy(float)
            +small.close.to_numpy(float))/3
        bins,mass,total=profile_from_price_qty(p,small.volume.to_numpy(float),
                         float(ex["signal_close"]))
        prices=float(ex["signal_close"])*np.exp((bins+.5)*BIN_LOG)
        fig,ax=plt.subplots(figsize=(7.4,7.2))
        ax.barh(prices,mass,height=prices*.0023,alpha=.72)
        ax.axhline(float(ex["signal_close"]),linestyle="--",
                   linewidth=1.5,label="Signal confirmation close")
        ax.axhline(float(ex["signal_close"])*1.05,linestyle=":",
                   linewidth=1.2,label="+5% target reference")
        ax.set_xlabel("Proxy historical BTC volume (5-minute HLC3 assignment)")
        ax.set_ylabel("BTCUSDT price (USDT)")
        ax.set_title("Pre-signal 7d volume-at-price proxy; NOT a trade heatmap")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(OUT/"sample_7d_volume_profile_PROXY.png",dpi=135)
        plt.close(fig)
    except Exception as err:
        print("PROFILE_CHART_WARNING "+str(err)[:250],flush=True)
    methods={
      "research_only":True,"trade_execution":"none","main_branch_modified":False,
      "market":"Binance BTCUSDT spot","source":"Binance public 4h OHLCV and 5m historical monthly archives + Oct 5m REST",
      "date_range":"2020-01-01 through 2026-10-09 UTC",
      "cohorts":{"development_original_n":53,"inspected_evaluation_original_n":44},
      "frozen_trading_params":"Original 3x4h candle Morning Star next open; ATR14 Wilder from confirmed candle x1.5 initial static stop; +5% take; 48h max; account 30%; 10bps fees and 5bps slippage per side.",
      "lookback_windows_days":[7,30],"price_bucket_percent":0.25,
      "signal_reference_price":"Last fully completed 4h confirmation candle CLOSE, NEVER next-open or future close",
      "profile_method":"PROXY: assign entire REAL 5m BTC traded base volume to that candle's HLC3 typical price; 0.25% relative logarithmic bucket; this is NOT actual trade execution-price VP",
      "POC":"0.25%-bin with the most 5m-proxy assigned traded BTC",
      "HVN":"local 0.25%-price-bin maximum >=70% window POC mass, for nearest upper/lower distances",
      "feature_set":["POC distance","POC share","above signal close up to +5% share",
        "below close down to -5% share","above-minus-below shares","nearest higher/lower HVN"],
      "fairness":"Each profile strictly ends at signal candle CLOSE. 5m candle timestamps less than confirmed candle boundary, lookback exactly 7/30 calendar days; do NOT forward fill missing 5m files; incomplete profiles excluded.",
      "inference":"Descriptive Spearman correlations and within-cohort median split only; NO optimization or trade filter and NO independent holdout; 2024-26 repeatedly used to tune strategy; no multivariate risk confound correction yet.",
      "daily_aggTrades_validation":"Attempt small number of full prior-day real executed price/quantity snapshots from public archives, compute profile total-variation and POC error versus corresponding 5m HLC3 proxy. 24h audit does NOT validate 7/30d profile precision.",
      "limitations":[
        "Broader sample uses real traded 5m bar BTC volume but a PROXY price attribution at HLC3, not actual executions nor real liquidity/book or liquidation heatmaps.",
        "A 5-minute OHLC bar volume may trade across a broad price range. Mapping all to HLC3 can distort POC/HVN; pilot daily aggTrade comparison tests error but cannot fully calibrate all regimes.",
        "All features use only past data by construction. Archive 2025+ spot timestamps can be microseconds, normalized to UTC milliseconds.",
        "No actual order book/depth/liquidation futures information, no orders.",
        "Correlation, bucket medians, and descriptive thresholds are not predictive causality or proof of edge; strategy baseline already heavily inspected in historic period.",
        "No new filter or tuned threshold enabled; frozen main strategy remains unchanged.",
        "Stop-first 4h OHLC fill ambiguity; no 1m execution simulation or intratrade MTM maximum drawdown.",
        "Market history can have missing days; fail closed on noncontiguous 7/30d lookbacks."
      ],
      "archive_fetch_issues":archive_issues,
      "aggTrades_audit_did_not_use_post_entry_market_data":True
    }
    (OUT/"methodology.json").write_text(json.dumps(methods,ensure_ascii=False,indent=2),encoding="utf8")
    print("PROFILE_RESEARCH_COMPLETE "+json.dumps({
        "signals":len(study),"complete_7d":int(study.w7_complete.sum()),
        "complete_30d":int(study.w30_complete.sum()),
        "archive_issue_count":len(archive_issues),
        "agg_trade_audit_successes":sum(x["status"]=="ok" for x in audit),
    }),flush=True)


if __name__=="__main__":
    main()
