from reprod_log import ReprodDiffHelper

if __name__ == "__main__":
    diff_helper = ReprodDiffHelper()
    torch_info = diff_helper.load_info("./forward_torch.npy")
    paddle_info = diff_helper.load_info("./forward_paddle.npy")

    diff_helper.compare_info(torch_info, paddle_info)

    diff_helper.report(path="forward_diff.log", diff_threshold=2e-6)