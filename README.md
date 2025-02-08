#GPT-2 Like Model for High School Project
Project Overview

This project aims to develop a GPT-2-like language model with reduced computational requirements. It is implemented using TensorFlow/Keras and leverages Hugging Face's datasets and transformers libraries for data handling and tokenization.

The model is built with self-attention layers, embedding layers, and LayerNorm while utilizing mixed precision training for efficiency. Various training strategies, including learning rate scheduling, early stopping, and model checkpointing, are employed to improve performance.
Features

    Transformer-based architecture (inspired by GPT-2 but with Grpuped Query Attention instead of Multi-Head attention)
    Hugging Face datasets integration for loading and preprocessing text data
    Mixed precision training to improve computational efficiency
    Multiple Keras callbacks for better training management
    Colab-compatible (easy to run in Google Colab)

Implementation Details

    Model Architecture: The model consists of self-attention layers, feed-forward layers, and dropout layers to enhance generalization.
    Training Strategy:
        CosineDecay: Adjusts learning rate based on a cosine function.
        EarlyStopping: Stops training if validation loss stops improving
        ModelCheckpoint: Saves the best-performing model
    Dataset Handling: The project uses tokenization from Hugging Face's AutoTokenizer for efficient text processing (The tokenizer was trained from the dataset used for training).

Contributing

If you'd like to contribute, feel free to fork the repository and submit a pull request! Right now the model is not learning and I am not able to figure out why, I would appreciate any help :D

License

This project is for educational purposes and follows an open-source approach.
