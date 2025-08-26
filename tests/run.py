from adapters import run_train_bpe


input_path = "D:\Model\cs336\\assignment1-basics-main\\tests\\fixtures\\tinystories_sample.txt"
vocab_size = 100
special_tokens = ["<unk>", "<bos>", "<eos>", "<|endoftext|>"]
print("run")
run_train_bpe(input_path, vocab_size, special_tokens, num_merges=100)

# with open(input_path, "rb") as f:
#     text = f.read()  # text 是 bytes 对象
#     int_text = list(text)
#     print(text)
#     print(int_text)
#     print(len(text))
#     end_index = text.decode('utf-8').find("<|endoftext|>")
#     print(end_index)
#     print(text[:end_index])
#
# print(hex(ord('中')))
# print(list('中'.encode('utf-8')))
# print(chr(97))