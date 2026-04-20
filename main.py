import os
import sys
import subprocess

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def run(cmd, desc):
    print(f"\n{'='*70}")
    print(f"  {desc}")
    print(f"{'='*70}")
    result = subprocess.run(cmd, shell=True, cwd=PROJECT_DIR)
    if result.returncode != 0:
        print(f"\n  ERROR: {desc} failed with code {result.returncode}")
        return False
    return True


def fetch():
    return run("python -c \"from data_loader import build_dataset; build_dataset(force_download=True)\"", "Fetching data from broker")


def train_direction():
    return run("python trainer_xgb.py", "Training direction model (XGBoost)")


def train_sub():
    return run("python trainer_multi.py", "Training sub-models (Vol + HMM + MeanRev)")


def backtest():
    return run("python backtest_chart.py", "Running backtest with trailing stop")


def all_steps():
    steps = [
        ("Fetch data", fetch),
        ("Train direction model", train_direction),
        ("Train sub-models", train_sub),
        ("Backtest", backtest),
    ]
    for name, fn in steps:
        if not fn():
            print(f"\n  Pipeline FAILED at: {name}")
            return False
    print(f"\n{'='*70}")
    print(f"  ALL STEPS COMPLETE")
    print(f"{'='*70}")
    return True


def signal():
    return run("python trader_d1.py signal", "Getting D1 signal")


def trade():
    return run("python trader_d1.py trade", "Executing trade")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python main.py all       - Fetch data + train all models + backtest")
        print("  python main.py fetch      - Download fresh data from broker")
        print("  python main.py train      - Train direction model only")
        print("  python main.py train_sub - Train sub-models (Vol, HMM, MeanRev)")
        print("  python main.py backtest  - Run backtest with trailing stop")
        print("  python main.py signal    - Get current trading signal")
        print("  python main.py trade     - Execute one trade")
        print("  python main.py bot       - Run trading bot 24/7")
        sys.exit(0)

    cmd = sys.argv[1].lower()
    if cmd == "all":
        all_steps()
    elif cmd == "fetch":
        fetch()
    elif cmd == "train":
        train_direction()
    elif cmd == "train_sub":
        train_sub()
    elif cmd == "backtest":
        backtest()
    elif cmd == "signal":
        signal()
    elif cmd == "trade":
        trade()
    elif cmd == "bot":
        run("python trader_d1.py bot", "Starting D1 trading bot")
    else:
        print(f"Unknown command: {cmd}")
        print("Use: all, fetch, train, train_sub, backtest, signal, trade, bot")