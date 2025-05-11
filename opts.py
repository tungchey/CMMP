import argparse

def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')

    parser.add_argument('--base-model', default='GRU', help='base Model')

    parser.add_argument('--use-model', action='store_true', default=True,
                        help='whether to use Model')

    parser.add_argument('--dropout', type=float, default=0.4, metavar='dropout', help='dropout rate')

    parser.add_argument('--batch-size', type=int, default=16, metavar='BS', help='batch size')

    parser.add_argument('--epochs', type=int, default=1, metavar='E', help='number of epochs')

    parser.add_argument('--model_type', default='hyper', help='restruct')

    parser.add_argument('--use_residue', action='store_true', default=False,
                        help='whether to use residue information or not')

    parser.add_argument('--multi_modal', action='store_true', default=True,
                        help='whether to use multimodal information')

    parser.add_argument('--fusion', default='TF',
                        help='method to use multimodal information')

    parser.add_argument('--modals', default='avl', help='modals to fusion')

    parser.add_argument('--Dataset', default='IEMOCAP', help='dataset to train and test')

    parser.add_argument('--use_speaker', action='store_true', default=True, help='whether to use speaker embedding')

    parser.add_argument('--use_modal', action='store_true', default=False, help='whether to use modal embedding')

    parser.add_argument('--norm', default='BN', help='NORM type')

    parser.add_argument('--testing', action='store_true', default=False, help='testing')

    args = parser.parse_args()
    return args
