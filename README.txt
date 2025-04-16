Instructions:

    1) Models

        The models [RNN, LSTM, Transformer Encoder] are all named with the Module suffix. (ex. LSTM_Module.py)
        These files contain a main method that can be used if you want to train the model yourself

    2) Tokenizer

        If you want to create a new tokenizer just run the BPEtokenizer.py file.
        You are free to use a new corpus if you want to test on a different text set,
        but that function is not implemented into this project

    3) Prompting

        To prompt the models run the Prompt_and_Bleu scripts for the respective model that 
        you want to run (ex. Prompt_and_Bleu_LSTM.py).
        The prompts can be customized at the bottom of the file if you want to try your own.
