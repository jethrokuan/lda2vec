BASE_DIR = "/Users/jethrokuan/Documents/Code/hash-lda2vec/lda2vec/"

twenty_newsgroups = {
    "summary_dir": "{}/summaries/".format(BASE_DIR),
    "checkpoint_dir": "{}/checkpoints/".format(BASE_DIR),
    "max_to_keep": 3,
    "batch_size": 50,
    "file_path": "experiments/lorem_ipsum/train_data.csv",
    "num_epochs": 10000,
    "num_iter_per_epoch": 20,
}
