# BSC-loss
Code for the paper "Batch-Softmax Contrastive Loss for Pairwise Sentence Scoring Tasks"

```
@inproceedings{chernyavskiy-etal-2022-batch,
    title = "Batch-Softmax Contrastive Loss for Pairwise Sentence Scoring Tasks",
    author = "Chernyavskiy, Anton  and
      Ilvovsky, Dmitry  and
      Kalinin, Pavel  and
      Nakov, Preslav",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.naacl-main.9",
    doi = "10.18653/v1/2022.naacl-main.9",
    pages = "116--126",
    abstract = "The use of contrastive loss for representation learning has become prominent in computer vision, and it is now getting attention in Natural Language Processing (NLP).Here, we explore the idea of using a batch-softmax contrastive loss when fine-tuning large-scale pre-trained transformer models to learn better task-specific sentence embeddings for pairwise sentence scoring tasks.We introduce and study a number of variations in the calculation of the loss as well as in the overall training procedure; in particular, we find that a special data shuffling can be quite important.Our experimental results show sizable improvements on a number of datasets and pairwise sentence scoring tasks including classification, ranking, and regression.Finally, we offer detailed analysis and discussion, which should be useful for researchers aiming to explore the utility of contrastive loss in NLP.",
}
```
