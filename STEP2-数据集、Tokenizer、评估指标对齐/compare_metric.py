from reprod_log import ReprodDiffHelper

if __name__ == "__main__":
    # ACC
    diff_helper = ReprodDiffHelper()
    torch_info = diff_helper.load_info("./acc_torch.npy")
    paddle_info = diff_helper.load_info("./acc_paddle.npy")

    diff_helper.compare_info(torch_info, paddle_info)
    diff_helper.report(path="acc_diff.log")
    
    #F1
    diff_helper = ReprodDiffHelper()
    torch_info = diff_helper.load_info("./f1_torch.npy")
    paddle_info = diff_helper.load_info("./f1_paddle.npy")

    diff_helper.compare_info(torch_info, paddle_info)
    diff_helper.report(path="f1_diff.log")
    