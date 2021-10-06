import json
import os

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdirs', type=str, nargs='+')
    parser.add_argument('--tags', type=str)
    parser.add_argument('--target', type=str)
    args = parser.parse_args()

    for logdir in args.logdirs:
        event_acc = EventAccumulator(logdir)
        tags = args.tags.split(',')
        event_acc.Reload()
        data = {}
        for tag in tags:
            scalars = event_acc.Scalars(tag)
            data[tag] = scalars
        with open(f'{args.target}/{os.path.basename(logdir)}.json', 'w') as f:
            json.dump(data, f)


if __name__ == '__main__':
    main()