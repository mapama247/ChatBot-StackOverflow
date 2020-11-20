# ChatBot-StackOverflow

Dialog system built as the final project of the Natural Language Processing [course](https://www.coursera.org/learn/language-processing) from HSE.

The goal was to develop a StackOverflow Assistant able to answer programming-related questions and hold conversations on random topics. For the “chit-chat mode”, the conversational bot uses a pre-trained neural network engine called [ChatterBot](https://github.com/gunthercox/ChatterBot), which ranks Twitter replicas by their relevance.

FILE | DESCRIPTION
------------ | -------------
programming_assistant.py | main code
dialogues.tsv | dialogue phrases from movie subtitles (used as negative samples).
tagged_posts.tsv  | StackOverflow posts, tagged with one programming language (used as positive samples).
starspace_embeddings.tsv  | Word embeddings trained with StarSpace for the task of finding duplicate questions on StackOverflow. The embeddings were trained using duplicates with high similarity as “positive” examples and randomly selected questions (low similarity assumed) as “negative” examples. 
