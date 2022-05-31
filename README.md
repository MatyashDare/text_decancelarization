# Text Decancelarization
## Abstract
Despite the abundance of studies in style transfer, to our knowledge, there is no existing research connected with “translation” of official documents into neurally styled texts. In many previous works style transfer models were able to change the style of a text, but they tended to distort the original meaning, or, vise versa, the meaning of the source texts’, while the style remained unchanged. Reading official papers such as legal texts can be challenging for a person without special education due to their syntactic and morphological complexity. Therefore, there is a distinct need for a tool which would be able to convert a legal text into sentences which have the same meaning but are more accessible and comprehensible for an average reader. This paper formulates a new style transfer task and suggests several possible solutions. The paper gives a detailed overview of supervised approach to text decancelization. A new parallel corpus of sentences originally written in formal language and one or more variants of it written in neutral style was collected. We fine-tuned different style transfer models on our dataset. We developed a style transfer model which is able to translate legal texts into utterances in neutral style, while preserving the meaning of the transformed words.
# Task
The [thesis paper](https://github.com/MatyashDare/text_decancelarization/blob/master/materials/thesis.pdf) proposes a new text style transfer task - text decancelarization. 
## Data

We collected a parallel dataset for supervised experiments. It contained 5200 pairs of sentences in official style and their manual translations to the neutral style collected from
almost 300 official documents. An [Instruction](https://github.com/MatyashDare/text_decancelarization/blob/master/materials/Instruction.pdf) for manual translation from official style to neuntral one was also developed.

## Models and experiments

**Baseline**

As a baseline, we used a paraphrasing model based on `T5` architecture introduced
by [David Dale](https://habr.com/ru/post/564916/). This model was trained on a large neutrally styled [ParaNMT-Ru-Leipzig
dataset](https://storage.yandexcloud.net/nlp/paranmt_ru_leipzig.zip), so we do not expect the output of the model to be generated in official style.

Then we modified the outputs of the Baseline model using the `bad_word_ids` parameter. The words that were banned to generated were obtained from the top-100 `max_features` from the binary classifier based on `Naive Bayes` and `TfidfVectorizer` and top-100 important features from the Attention matrices
of the Self-Attention layer obtained from the second best binary classifier - fine-tuned
transformer model `ruBERT-base-cased`.


The experiments were conducted on the following models:

**ruT5**

The `T5` [(Raffel et al., 2019)](https://arxiv.org/pdf/1910.10683.pdf) is an encoder-decoder architecture that could be applied
on a number of purposes, including text style transfer. We fine-tuned `ruT5-base` and
`ruT5-large` models on our parallel corpus on text style transfer task. Due to the similarity
of the decancelarized text to the original one, it is likely for the model to copy the original
input to get the low value of the `Cross-Entropy loss`. To focus on the rare dissimilarities
during the optimization, we adopted the `Focal loss` [(Lin et al., 2017)](https://arxiv.org/pdf/1708.02002.pdf) for our task
   
   - [ruT5-base](https://huggingface.co/sberbank-ai/ruT5-base)
   
   -  [ruT5-large](https://huggingface.co/sberbank-ai/ruT5-large)
   
   - [ruT5-large](https://huggingface.co/sberbank-ai/ruT5-large) +  `Focal loss`
   
**ruGPT3**

`GPT` is a transformer decoder model that uses attention instead of earlier recurrence-
and convolution-based architectures [(Sherstinsky, 2020)](https://www.researchgate.net/publication/338723814_Fundamentals_of_Recurrent_Neural_Network_RNN_and_Long_Short-Term_Memory_LSTM_network), [(O’Shea and Nash, 2015)](https://arxiv.org/pdf/1511.08458.pdf). The
model’s attention mechanisms allow it to selectively focus on parts of input text that it
predicts would be the most relevant. We fine-tuned `ruGPT3-small`, `ruGPT3-medium`,
`ruGPT3-large` for text generation given a new token <DECANC> that was used as special token for generation decancelarized variant of the text before the mentioned token.
We used optuna15 open-source library for searching for the best hyperparameters. It took
us about 40 GPU hours to find the best parameters for all 3 models. `Promt-tuning` [(Lesteret al., 2021)](https://arxiv.org/pdf/2104.08691.pdf) is a technique of adding trainable embeddings to a sequence of tokens embeddings and optimizing them given a frozen network. We used this technique in order to
improve the outputs of the `ruGPT3` models.
    
- [ruGPT3-small](https://huggingface.co/sberbank-ai/rugpt3small_based_on_gpt2) 
    
- [ruGPT3-medium](https://huggingface.co/sberbank-ai/rugpt3medium_based_on_gpt2)
    
- [ruGPT3-large](https://huggingface.co/sberbank-ai/rugpt3large_based_on_gpt2)
    
- [ruGPT3-large](https://huggingface.co/sberbank-ai/rugpt3large_based_on_gpt2) + promt-tuning
    
## Evaluation
In order to conduct a thorough evaluation of a style transfer model, we have to consider the following output of the model:
- the style of the source sentence is changed;
- the content of the source sentence is preserved;
- it yields a grammatically correct sentence. 
 
Individual metrics are used in the majority of works on style transfer to evaluate these
parameters. [(Pang and Gimpel, 2018)](https://arxiv.org/pdf/1810.11878.pdf) points out, however, that these three components
are frequently inversely correlated, thus they must be combined to obtain the balance. As
we need a compound metric to find a balance between them, our evaluation setup follows
this principle. 

That is why we computed three parameters:

1. **Style transfer accuracy (STA)**

We trained a [binary classifier](https://github.com/MatyashDare/text_decancelarization/blob/master/notebooks/binary%20classifiers.ipynb) to predict the style of the text (1 - neutral
style, 0 - official language). We count an average of all the predicted classes for each sentence and the level of officialese. The higher the
level of STA, the better style was transferred.

2. **Content preservation (CP)**
We look into content preservation from two perspectives. First, we compute well-
established word-based metrics in order to count the number of matching substrings in the
fastest and most efficient way:

- **BLEU** score - ngram precision for n from 1 to 4. It calculates how similar a candidate sentence is to a reference sentence based on n-gram matches between sentences. The larger
the metric, the closer generated texts to their references;
- **METEOR** - this metric is also based on word overlap (WO) calculation. METEOR is determined as the harmonic mean of unigram precision and recall, with
recall weighted more heavily [(Denkowski and Lavie, 2014)](https://aclanthology.org/W14-3348.pdf).

Secondly, we calculate cosine similarity (CS) between sentence-level embeddings of both
source and transformed texts. We used [LaBSE](https://huggingface.co/cointegrated/LaBSE-en-ru) model pre-trained on Russian texts.

3. **Fluency (FL)**
We utilized the [ruGPT2-large](https://github.com/vlarine/ruGPT2) language model for computing Perplexity (PPL) of the generated and original texts.
The higher 1/PPL, the more natural a sentence is generated.

4. **Aggregated metric (GM)**

Following [(Dementieva et al., 2021)](https://aclanthology.org/2021.emnlp-main.629.pdf) and [(Pang and Gimpel, 2018)](https://arxiv.org/pdf/1810.11878.pdf), we use a combination of STA, CS and FL.

$$GM = (\max(STA, 0) \times \max(CP, 0) \times \max({1/PPL}, 0))^{\frac{1}{3}}$$

## Results

**model**|**STA**  | **CP** |**BLEU**|**METEOR**|**PPL**|**GM**
------------- | ------------- | -------------| -------------| -------------| -------------| -------------
Baseline  | 0.43 | 0.7 | 0.64| 0.76| 0.85| 0.25
Baseline + `bad_word_ids`| 0.52 | 0.75 | 0.81| 0.64| 0.69| 0.29 
ruGPT3-small |0.66|0.8|0.7|0.7|0.61|0.33
ruGPT3-large |0.7|0.88|0.74|0.77|0.77|0.43
ruGPT3-large |0.72|0.9|0.88|0.73|0.75|0.49
ruGPT3-large + `promt-tuning` |0.71|0.9|0.78|0.78|0.82|0.52
ruT5-large |0.74|0.9|0.75|0.78|0.82|0.55
**ruT5-large** + `Focal loss` |**0.74**|**0.93**|**0.89**|**0.8**|**0.83**|**0.57**
