import argparse
from config import DEVICE

print(f"Device: {DEVICE}")


def main():
    parser = argparse.ArgumentParser(description="EUR/USD Predictive Analytics")
    parser.add_argument("command", choices=["fetch", "train", "backtest", "predict"],
                        help="Command to run")
    args = parser.parse_args()

    if args.command == "fetch":
        from data_loader import build_dataset
        build_dataset()

    elif args.command == "train":
        from trainer import run_training
        run_training()

    elif args.command == "backtest":
        from backtest import run_backtest
        run_backtest()

    elif args.command == "predict":
        from data_loader import build_dataset, load_raw_prices
        from trainer import run_training
        print("Use: python trader.py signal  — for live signal generation")


if __name__ == "__main__":
    main()