# GPT-2: [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

## Summary

This paper examines whether large language models can learn to perform downstream tasks without the need for explicit supervision. The authors train a series of large language models on a massive dataset of text and code called WebText, and evaluate their performance on a variety of tasks, including language modeling, question answering, summarization, and translation.

Here are the key findings from the paper:

* **Language models can be trained to perform a variety of tasks in a zero-shot setting, without any parameter or architecture modification.** This is accomplished by conditioning the language model on a context that includes examples of the desired task. For example, to perform translation, the language model is conditioned on a context of example pairs of the format "English sentence = French sentence", and then after a final prompt of "English sentence =", the model samples from its distribution to generate a French translation.
* **The performance of language models on downstream tasks improves with model size.** The authors find that their largest model, GPT-2, achieves state-of-the-art results on 7 out of 8 tested language modeling datasets in a zero-shot setting. GPT-2 also achieves promising results on other tasks, such as question answering and summarization.
* **The diversity of the training data is important for zero-shot task transfer.** The authors find that their models trained on WebText, a dataset that includes text from a variety of domains, outperform models trained on more specialized datasets.
* **Data overlap between the training data and the evaluation datasets can inflate the reported performance of language models.** The authors analyze the overlap between their WebText training data and several common evaluation datasets and find that the overlap is small but consistent. They recommend the use of n-gram overlap based de-duplication as a verification step during the creation of training and test splits for new NLP datasets.

**Overall, the paper presents evidence that large language models trained on sufficiently large and diverse datasets are able to perform well across many domains and datasets in a zero-shot setting. These findings suggest that high-capacity models trained to maximize the likelihood of a sufficiently varied text corpus begin to learn how to perform a surprising amount of tasks without the need for explicit supervision.**

