"""Train a model on cleaned feature windows.

uv run python scripts/04_train.py --model lstm_uni_attention --window-seconds 10
"""

from attention_lapse_detection.training.train import create_arg_parser, train_model


if __name__ == "__main__":
    args = create_arg_parser(__doc__).parse_args()
    train_model(**vars(args))
