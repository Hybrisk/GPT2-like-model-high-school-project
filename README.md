# GPT-2 Like Model for High School Project  

## Project Overview  
This project aims to develop a **GPT-2-like language model** with reduced computational requirements.  
It is implemented using **TensorFlow/Keras** and leverages **Hugging Face's `datasets` and `transformers`**  
for efficient data handling and tokenization.  

The model incorporates **self-attention layers, embedding layers, and LayerNorm**, while utilizing  
**mixed precision training** for improved efficiency. Various training strategies, such as  
**cosine learning rate decay, early stopping, and model checkpointing**, are used to optimize performance.  

---

##  Features  
 **Transformer-based architecture** (inspired by GPT-2 but with **Grouped Query Attention** instead of Multi-Head Attention)  
 **Custom Tokenizer** trained directly on the dataset  
 **Hugging Face `datasets` integration** for seamless data loading and preprocessing  
 **Mixed precision training** for improved computational efficiency  
 **Advanced Keras callbacks** for better training management  
 **Colab-compatible** (easily run the model in Google Colab)  

---

## Implementation Details  

### Model Architecture  
The model consists of:  
- **Self-Attention Layers** (Grouped Query Attention)  
- **Feed-Forward Layers**  
- **Dropout Layers** to improve generalization  

### Training Strategy  
- **Cosine Learning Rate Decay** – Adjusts learning rate dynamically  
- **Early Stopping** – Stops training if validation loss stops improving  
- **Model Checkpointing** – Saves the best-performing model automatically  

### Dataset Handling  
- Uses **Hugging Face's `AutoTokenizer`**, trained on the dataset for optimal tokenization.  

> ** Note:**  
> During the upload to GitHub, the **dataset preprocessing cell was corrupted**.  
> Corrected dataset preprocessing code below:  

```python
# Preprocessing logic
import re

def truncate_tokens(tokens, max_len):
    return tokens[:max_len]

def split_and_truncate(text, tokenizer, max_seq_len):
    chunks = []
    current_chunk = []

    pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s'
    sentences = re.split(pattern, text.replace('\n', ''))

    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    current_chunk.append(bos_token_id)

    for sentence in sentences:
        if not sentence.strip():
          continue
        sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
        sentence_tokens = truncate_tokens(sentence_tokens, max_seq_len - 2)
        if len(current_chunk) + len(sentence_tokens) <= max_seq_len - 1:
            current_chunk.extend(sentence_tokens)
        elif len(current_chunk) > 1:
            current_chunk.append(eos_token_id)
            chunks.append(current_chunk)
            current_chunk = [bos_token_id] + sentence_tokens

    if len(current_chunk) > 1:
        current_chunk.append(eos_token_id)
        chunks.append(current_chunk)

    return chunks


def preprocess_function(examples, tokenizer, max_seq_len):
    pad_token_id = tokenizer.pad_token_id
    input_ids_list = []
    labels_list = []

    for text in examples["text"]:
        tokenized_chunks = split_and_truncate(text, tokenizer, max_seq_len)

        for seq in tokenized_chunks:
            input_ids = seq + [pad_token_id] * (max_seq_len - len(seq))

            labels = seq[1:] + [pad_token_id] * (max_seq_len - len(seq) + 1)

            input_ids_list.append(input_ids)
            labels_list.append(labels)

    return {
        "input_ids": input_ids_list,
        "labels": labels_list
    }


train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=["text"], fn_kwargs={"tokenizer": tokenizer, "max_seq_len": max_seq_len})
val_dataset = val_dataset.map(preprocess_function, batched=True, remove_columns=["text"], fn_kwargs={"tokenizer": tokenizer, "max_seq_len": max_seq_len})
```
## Contributing

If you'd like to contribute, feel free to fork the repository and submit a pull request!

---

### Current Issue: The model is not learning correctly, and I haven’t been able to figure out why.
### Any help would be greatly appreciated! :D

---

## License

This project is for educational purposes and follows an open-source approach.
