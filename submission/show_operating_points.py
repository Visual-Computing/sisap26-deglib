import argparse
import pandas as pd

# show best performing parameters exceeding threshold

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--algorithm',)
    parser.add_argument(
        '--threshold',
        default=0.8,
        help='minimum recall',
        type=float)
    parser.add_argument(
        'csv',
        metavar='CSV',
        help='input csv')
    parser.add_argument(
        '--task',
        choices=['task1', 'task2', 'task3'],
        required=True,
    )
    parser.add_argument(
        '--dataset',
        default=None,
        help="dataset name to filter on; inferred from the CSV if only one dataset is present",
    )

    args = parser.parse_args()
    df = pd.read_csv(args.csv)
    df = df[df.task == args.task]

    if args.dataset is None:
        datasets = set(df.dataset.unique())
        if len(datasets) == 0:
            print(f"No results found for task={args.task!r}")
            raise SystemExit(1)
        if len(datasets) > 1:
            print(f"Multiple datasets found for task={args.task!r}: {sorted(datasets)}")
            print("Please specify --dataset explicitly.")
            raise SystemExit(1)
        args.dataset = datasets.pop()
        print(f"Inferred dataset: {args.dataset}")

    df = df[df.dataset == args.dataset]

    if args.algorithm:
        algorithms = [args.algorithm]
    else:
        algorithms = set(df.algo.values)
    for algo in algorithms:
        print(f'show {algo}')
        if (len(df[(df.recall > args.threshold) & (df.algo == algo)].groupby(['algo', 'dataset']).min()[['querytime']])) == 0:
            print("didn't exceed recall, print highest recall:")
            print(df[(df.algo == algo)].groupby(['algo', 'dataset']).max()[['recall', 'querytime']])
    
        else:
            print(df[(df.recall > args.threshold) & (df.algo == algo)].groupby(['algo', 'dataset']).min()[['querytime']])

    print("Overview passing threshold")

    print(df[(df.recall >= args.threshold - 1e-6)][['algo', 'dataset', 'querytime', 'params']].sort_values(by=['dataset', 'algo', 'querytime']))

    print("Overview NOT passing threshold")

    print(df[(df.recall < args.threshold - 1e-6)][['algo', 'dataset', 'querytime', 'params']].sort_values(by=['dataset', 'algo', 'querytime']))