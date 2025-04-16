import sentencepiece as spm

if __name__ == "__main__":

    spm.SentencePieceTrainer.Train(
        input='corpus.txt',
        model_prefix='bpe_tokenizer',
        vocab_size=10000,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        user_defined_symbols=",".join(["<bos>", "<eos>", "<pad>"])
    )

    print("Tokenizer training complete! Files generated:")
    print("- bpe_tokenizer.model")
    print("- bpe_tokenizer.vocab")

