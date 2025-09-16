import torch
import torch.nn as nn
import traceback

import sampling
from a_myLM import Function, LinearModel, LayerNorm, RMSNorm, SwiGLU, CausalMultiheadAttention
from a_tokenizer import BPETokenizer
from a_transformer import TransformerBlock, TransformerStack

# try:
# tokenizer = BPETokenizer(
#     vocab_id2token_file='vocab_5M_id2token.json',
#     merges_file='merges_5M.txt',
#     special_tokens=["<|endoftext|>"],
#     space_replacement="Ġ"
# )
#
# texts = [
#     "hello, world",
#     "tell me a story.",
#     "the quick brown fox",
#     "jumps over the lazy dog",
#     "this is a longer text to ensure sufficient runtime",
#     "another sample text for performance analysis",
#     "deep learning models require sufficient data"
# ]
#
# device = torch.device('cuda')
# batch_tensor, padding_mask = tokenizer.batch_encode(texts, padding=True, device=device)
#
# vocab_size = tokenizer.vocab_size
# d_model = 768
# weights = nn.Parameter(torch.randn(vocab_size, d_model, device=device))
# nn.init.normal_(weights, mean=0, std=0.02)
#
# dense_vectors = Function.embedding(vocab_size, d_model, weights, batch_tensor)
# rope_vectors = Function.apply_rope(dense_vectors)
# norm_vectors = RMSNorm(d_model, device=device)(rope_vectors)
#
# num_layers = 64
# num_heads = 8
#
# transformer_stack = TransformerStack(
#     num_layers=num_layers,
#     embed_dim=d_model,
#     num_heads=num_heads,
#     dropout=0.1,
#     batch_first=True,
#     device=device,
#     key_padding_mask=padding_mask
# )
#
# kv_caches = None
# norm_vectors, kv_caches = transformer_stack(
#     norm_vectors,
#     kv_caches=kv_caches)
#
# final_norm = RMSNorm(d_model)(norm_vectors)
# logits = LinearModel(d_model, vocab_size)(final_norm)  # LM Head
# probs = Function.softmax(logits, dim=-1)
#
# predicted_ids = sampling.top_p_sampling(probs, p=0.9)
# if predicted_ids.dim() == 1:
#     predicted_ids = predicted_ids.unsqueeze(0)  # 单样本转为多样本
# id_lists = predicted_ids.tolist()
#
# output_text = [tokenizer.decode(ids) for ids in id_lists]
# for i, text in enumerate(output_text):
#     print(f"样本 {i + 1} 预测文本: {text}")
#
#     # 确保程序不会提前退出
#     print("程序执行完成")
#
# except Exception as e:
#     print(f"程序出错: {e}")
#     traceback.print_exc()
#     # 确保程序不会立即退出，给Nsys时间收集数据
#     import time
#
#     time.sleep(15)

def generate(tokenizer, initial_ids, padding_mask=None, max_new_tokens=100, device='cuda'):
    vocab_size = tokenizer.vocab_size
    d_model = 768
    num_layers = 64
    num_heads = 8

    weights = nn.Parameter(torch.randn(vocab_size, d_model, device=device))
    nn.init.normal_(weights, mean=0, std=0.02)

    rms_norm = RMSNorm(d_model, device=device)
    final_norm = RMSNorm(d_model, device=device)
    lm_head = LinearModel(d_model, vocab_size, device=device)

    transformer_stack = TransformerStack(
        num_layers=num_layers,
        embed_dim=d_model,
        num_heads=num_heads,
        dropout=0.1,
        batch_first=True,
        device=device,
        key_padding_mask=padding_mask
    )

    kv_caches = None
    generated_ids = initial_ids.clone()
    current_ids = initial_ids

    for step in range(max_new_tokens):

        if step == 0:
            input_step = current_ids
        else:
            input_step = current_ids[:, -1:]

        embeddings = Function.embedding(vocab_size, d_model, weights, input_step)

        rope_vectors = Function.apply_rope(embeddings)

        norm_vectors = rms_norm(rope_vectors)

        norm_vectors, kv_caches = transformer_stack(
            norm_vectors,
            kv_caches=kv_caches
        )

        final_output = final_norm(norm_vectors)

        logits = lm_head(final_output)

        next_token_logits = logits[:, -1, :]

        probs = Function.softmax(next_token_logits, dim=-1)
        probs = probs.unsqueeze(1)

        next_token = sampling.top_p_sampling(probs, p=0.9)

        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        current_ids = next_token

    return generated_ids

try:
    tokenizer = BPETokenizer(
        vocab_id2token_file='vocab_5M_id2token.json',
        merges_file='merges_5M.txt',
        special_tokens=["<|endoftext|>"],
        space_replacement="Ġ"
    )

    texts = [
        "hello, world",
        # "tell me a story.",
        # "the quick brown fox",
        # "jumps over the lazy dog",
        # "this is a longer text to ensure sufficient runtime",
        # "another sample text for performance analysis",
        # "deep learning models require sufficient data"
    ]

    device = torch.device('cuda')
    batch_tensor, padding_mask = tokenizer.batch_encode(texts, padding=True, device=device)

    generated_ids = generate(
        tokenizer=tokenizer,
        initial_ids=batch_tensor,
        padding_mask=padding_mask,
        max_new_tokens=100,
        device=device
    )

    id_lists = generated_ids.tolist()
    output_text = [tokenizer.decode(ids) for ids in id_lists]
    for i, text in enumerate(output_text):
        print(f"样本 {i + 1} 预测文本: {text}")

except Exception as e:
    print(f"程序出错: {e}")
    traceback.print_exc()

