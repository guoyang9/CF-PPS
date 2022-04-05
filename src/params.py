def parser_add_data_arguments(parser):
    # ------------------------------------Dataset Parameters-------------------- #
    parser.add_argument('--data_path',
                        type=str, default='/temp/guoyy/Amazon/data/data_raw/',
                        help="raw downloaded path")
    parser.add_argument('--processed_path',
                        type=str, default='/temp/guoyy/Amazon/data/data_run/',
                        help="after processed path")
    parser.add_argument('--dataset',
                        type=str, default='Office_Products',
                        help="the chosen dataset name")
    parser.add_argument('--model',
                        type=str,
                        help="the model name (for AEM and CFSearch only)")
    # ------------------------------------Process Parameters-------------------- #
    parser.add_argument('--seed',
                        type=int, default=11,
                        help="for code reproduction")
    parser.add_argument('--word_count',
                        type=int, default=10,
                        help="remove the words number less than count")
    parser.add_argument("--doc2vec_size",
                        type=int,
                        default=512,
                        help="doc2vec model embedding dimension")
    parser.add_argument('--candidate',
                        type=int, default=100,
                        help="rank results on 100 candidate items")
    # ------------------------------------Experiment Setups -------------------- #
    parser.add_argument('--debug',
                        default=True,
                        action='store_true',
                        help="enable debug")
    parser.add_argument('--gpu',
                        default='0',
                        help="using device")
    parser.add_argument('--worker_num',
                        default=4,
                        type=int,
                        help='number of workers for data loading')
    parser.add_argument('--top_k',
                        default=10,
                        type=int,
                        help='truncated at top_k products')
    parser.add_argument('--max_query_len',
                        default=20,
                        type=int,
                        help='max length for each query')
    parser.add_argument('--max_sent_len',
                        default=100,
                        type=int,
                        help='max length for each review')
    parser.add_argument('--embedding_size',
                        default=128,
                        type=int,
                        help="embedding size for possibly word, user and item")
    parser.add_argument('--lr',
                        default=1e-3,
                        type=float,
                        help='learning rate')
    parser.add_argument('--regularization',
                        default=1e-3,
                        type=float,
                        help='regularization factor')
    parser.add_argument('--batch_size',
                        default=256,
                        type=int,
                        help='batch size for training')
    parser.add_argument('--neg_sample_num',
                        default=5,
                        type=int,
                        help='negative sample number')
    parser.add_argument('--epochs',
                        default=30,
                        type=int,
                        help="training epochs")

