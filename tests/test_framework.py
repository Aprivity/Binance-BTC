"""BTC-only framework regression, safety, plugin and offline integration tests."""
import json
import math
import pandas as pd
import pytest
from btc_quant.__main__ import main, synthetic
from btc_quant.config import Config, STEPS
from btc_quant.core import Account, backtest
from btc_quant.data import normalize, validate, load_csv
from btc_quant.paper import state_for, paper_tick, load_state, atomic_save
from btc_quant.report import metrics
from btc_quant.research import rolling_validate
from btc_quant.strategy import no_trades,prepare_signals,resolve_strategy


def candles(n=560):
    return validate(synthetic("4h",n),"4h")


def pulse_signals(x,cfg):
    # Deterministic fixture, no real trading claim. Signals only on past rows.
    return pd.DataFrame({"buy_signal":[i%25==0 for i in range(len(x))],
                         "sell_signal":[i%25==5 for i in range(len(x))]},index=x.index,dtype=bool)


def test_btc_only_guards():
    assert Config().symbols==("BTC/USDT",)
    with pytest.raises(ValueError):
        Config(symbols=("ETH/USDT",))
    with pytest.raises(ValueError):
        Config(symbols=("BTC/USDT","ETH/USDT"))
    with pytest.raises(ValueError):
        Config(max_symbol_pct=80)


def test_backtest_requires_explicit_strategy():
    with pytest.raises(ValueError,match="explicit"):
        backtest({"BTC/USDT":candles()},Config())
    with pytest.raises(SystemExit):
        main(["backtest","--csv-dir","data/market"])


def test_inert_plugin_is_always_flat():
    book=backtest({"BTC/USDT":candles()},Config(),signal_fn=no_trades)
    m=metrics(book,Config())
    assert m["closed_trades"]==0
    assert m["net_return_pct"]==0
    assert not book.positions and book.cash==1000 and not book.events


def test_explicit_fixture_produces_simulated_trades():
    book=backtest({"BTC/USDT":candles()},Config(),signal_fn=pulse_signals)
    assert len(book.trades)>0
    assert book.fees>0 and book.slippage>0
    assert all(t["symbol"]=="BTC/USDT" for t in book.trades)
    assert 0<=book.cash


def test_signal_executes_next_bar_only():
    frame=candles()
    allowed={pd.to_datetime(int(frame.iloc[i+1].open_time_ms),unit="ms",utc=True).isoformat()
             for i in range(len(frame)-1) if i%25==0}
    book=backtest({"BTC/USDT":frame},Config(),signal_fn=pulse_signals)
    assert {e["time"] for e in book.events if e["side"]=="BUY"}.issubset(allowed)


def test_fresh_window_rejects_carry_signal():
    frame=candles()
    open_i=251 # prior bar 250 may be a buy signal, must not carry into new fold
    starting=int(frame.iloc[open_i].open_time_ms)
    book=backtest({"BTC/USDT":frame},Config(),signal_fn=pulse_signals,start_ms=starting,
                  end_ms=int(frame.iloc[open_i+10].open_time_ms))
    assert not any(e["side"]=="BUY" and e["time"]==pd.to_datetime(starting,unit="ms",utc=True).isoformat() for e in book.events)


def test_strategy_interface_refuses_non_bool_and_misalignment():
    frame=candles()
    with pytest.raises(ValueError):
        prepare_signals(frame,Config(),lambda x,c:pd.DataFrame({"buy_signal":[1]*len(x),"sell_signal":[False]*len(x)}))
    with pytest.raises(ValueError):
        prepare_signals(frame,Config(),lambda x,c:pd.DataFrame({"buy_signal":[True]*len(x),"sell_signal":[True]*len(x)}))
    with pytest.raises(ValueError):
        prepare_signals(frame,Config(),lambda x,c:pd.DataFrame({"buy_signal":[False]*(len(x)-1),"sell_signal":[False]*(len(x)-1)}))


def test_fee_and_stop_are_engine_level():
    cfg=Config()
    book=Account.create(cfg)
    book.prices["BTC/USDT"]=100
    assert book.buy("BTC/USDT","t",100,2,cfg)
    assert book.protective_exit("BTC/USDT",100,200,20)[1]=="stop_loss"
    assert book.sell("BTC/USDT","u",100,cfg,"exit")
    assert book.cash<1000 and book.fees>0 and book.slippage>0
    assert book.trades[0]["pnl_usdt"]<0


def test_drawdown_halt_still_allows_exits():
    cfg=Config()
    book=Account.create(cfg)
    book.prices["BTC/USDT"]=100
    assert book.buy("BTC/USDT","t",100,2,cfg)
    book.prices["BTC/USDT"]=20
    book.mark("later",cfg)
    assert book.halted and book.halted_at=="later"
    assert not book.buy("BTC/USDT","next",100,2,cfg)
    assert book.sell("BTC/USDT","next",20,cfg,"gap_stop")


def test_observer_bootstrap_no_trades_or_strategy(tmp_path):
    cfg=Config()
    df=candles()
    state=state_for(cfg)
    assert state["strategy_ref"] is None
    one=paper_tick(state,cfg,{"BTC/USDT":df},100,"first")
    two=paper_tick(one,cfg,{"BTC/USDT":df},101,"second")
    assert one["last_closed_ms"]==two["last_closed_ms"]
    assert two["account"]["events"]==[]
    assert two["account"]["positions"]=={}
    path=tmp_path/"state.json"
    atomic_save(path,two)
    assert load_state(path,cfg)==two


def test_old_state_rejected(tmp_path):
    path=tmp_path/"v2.json"
    path.write_text(json.dumps({"schema_version":2,"config":Config().to_dict()}))
    with pytest.raises(ValueError,match="Incompatible"):
        load_state(path,Config())


def test_plugin_paper_bootstrap_no_historical_buy():
    cfg=Config()
    state=state_for(cfg,"tests.strategy_fixture:signals")
    df=candles()
    st=paper_tick(state,cfg,{"BTC/USDT":df},100,"first",signal_fn=pulse_signals,
                  strategy_ref="tests.strategy_fixture:signals")
    assert st["account"]["events"]==[]
    with pytest.raises(ValueError):
        paper_tick(st,cfg,{"BTC/USDT":df},100,"bad",signal_fn=pulse_signals,strategy_ref="another:signals")


def test_invalid_data_and_eth_rejected(tmp_path):
    df=candles()
    with pytest.raises(ValueError):
        validate(df.drop(index=10),"4h")
    csv=tmp_path/"BTCUSDT_4h.csv"
    df.to_csv(csv,index=False)
    assert len(load_csv(csv,"BTC/USDT","4h"))==len(df)
    with pytest.raises(ValueError):
        load_csv(csv,"ETH/USDT","4h")


def test_rolling_no_strategy_is_rejected(tmp_path):
    with pytest.raises(ValueError,match="explicit"):
        rolling_validate({"BTC/USDT":candles()},Config(),signal_fn=None,strategy_ref="",output_dir=tmp_path)


def test_rolling_synthetic_folds_and_isolation(tmp_path):
    df=candles(1200)
    result,agg=rolling_validate({"BTC/USDT":df},Config(),signal_fn=pulse_signals,
                                strategy_ref="test:signals",train_days=45,test_days=14,
                                step_days=14,output_dir=tmp_path,plot=False)
    assert agg["folds"]>1
    assert set(result.stage)=={"train","test"}
    assert (tmp_path/"rolling_folds.csv").exists()
    assert (tmp_path/"rolling_summary.json").exists()
    assert (result.starting_usdt==1000).all()
    assert not result[result.stage=="test"].empty


def test_offline_cli_demo_is_inert(tmp_path):
    main(["demo","--output-dir",str(tmp_path)])
    data=json.loads((tmp_path/"metrics.json").read_text())
    assert data["metrics"]["closed_trades"]==0
    assert data["metrics"]["ending_usdt"]==1000


def test_explicit_file_plugin_import(tmp_path):
    path=tmp_path/"user_signal.py"
    path.write_text("import pandas as pd\ndef signals(df,cfg):\n return pd.DataFrame({'buy_signal':False,'sell_signal':False},index=df.index)\n")
    fn=resolve_strategy(f"{path}:signals")
    assert callable(fn)
    with pytest.raises(ValueError):
        resolve_strategy("triple_ema")