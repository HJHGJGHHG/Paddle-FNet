from reprod_log import ReprodDiffHelper

if __name__ == "__main__":
    diff_helper = ReprodDiffHelper()
    torch_info = diff_helper.load_info("torch_train/train_align_benchmark.npy")
    paddle_info = diff_helper.load_info("paddle_train/train_align_paddle.npy")

    diff_helper.compare_info(torch_info, paddle_info)

    diff_helper.report(path="train_align_diff.log", diff_threshold=0.0015)