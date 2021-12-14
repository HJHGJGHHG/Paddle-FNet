import numpy as np
import paddle
import torch
import torch.fft as fft
from collections import OrderedDict


def convert_pytorch_checkpoint_to_paddle(
        pytorch_checkpoint_path="pytorch_model.bin",
        paddle_dump_path="model_state.pdparams",
        version="old", ):
    hf_to_paddle = {
        "embeddings.LayerNorm": "embeddings.layer_norm",
        ".LayerNorm.": ".layer_norm.",
        "encoder.layer": "encoder.layers",
    }
    do_not_transpose = []
    if version == "old":
        hf_to_paddle.update({
            "predictions.bias": "predictions.decoder_bias",
            ".gamma": ".weight",
            ".beta": ".bias",
        })
        do_not_transpose = do_not_transpose + ["predictions.decoder.weight"]
    
    pytorch_state_dict = torch.load(
        pytorch_checkpoint_path, map_location="cpu")
    paddle_state_dict = OrderedDict()
    for k, v in pytorch_state_dict.items():
        is_transpose = False
        if k[-7:] == ".weight":
            # embeddings.weight and LayerNorm.weight do not transpose
            if all(d not in k for d in do_not_transpose):
                # if ".embeddings." not in k and ".LayerNorm." not in k:
                if (
                        "embeddings." not in k and ".LayerNorm." not in k) or "embeddings.projection" in k or "seq_relationship" in k or "classifier" in k:
                    if v.ndim == 2:
                        v = v.transpose(0, 1)
                        is_transpose = True
        oldk = k
        for hf_name, pd_name in hf_to_paddle.items():
            k = k.replace(hf_name, pd_name)
        
        # add prefix `fnet.`
        # if "fnet." not in k and "cls." not in k and "classifier" not in k:
        #    k = "fnet." + k
        
        print(f"Converting: {oldk} => {k} | is_transpose {is_transpose}")
        paddle_state_dict[k] = v.data.numpy()
    
    paddle.save(paddle_state_dict, paddle_dump_path)
    return paddle_state_dict


def compare(out_torch, out_paddle):
    out_torch = out_torch.detach().numpy()
    out_paddle = out_paddle.detach().numpy()
    assert out_torch.shape == out_paddle.shape
    abs_dif = np.abs(out_torch - out_paddle)
    mean_dif = np.mean(abs_dif)
    max_dif = np.max(abs_dif)
    min_dif = np.min(abs_dif)
    print("mean_dif:{}".format(mean_dif))
    print("max_dif:{}".format(max_dif))
    print("min_dif:{}".format(min_dif))


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch_file", default=None, required=True)
    parser.add_argument("--paddle_file", default=None, required=True)
    
    args = parser.parse_args([])
    convert_pytorch_checkpoint_to_paddle(args.torch_file, args.paddle_file)
