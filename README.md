# ChatBot-StackOverflow

Dialog system built as the final project of the Natural Language Processing [course](https://www.coursera.org/learn/language-processing) from HSE.

The goal was to develop a StackOverflow Assistant able to answer programming-related questions and hold conversations on random topics. For the “chit-chat mode”, the conversational bot uses a pre-trained neural network engine called [ChatterBot](https://github.com/gunthercox/ChatterBot), which ranks Twitter replicas by their relevance.

## Files

NAME | DESCRIPTION
------------ | -------------
*programming_assistant.py* | Main code
*utils.py* | Auxiliary functions
*dialogue_manager.py* | Creates an instance of the ChatBot and trains it to generate the best answer to any question
*dialogues.tsv* | Dialogue phrases from movie subtitles (used as negative samples). [source](https://github.com/hse-aml/natural-language-processing/releases/download/project/dialogues.tsv)
*tagged_posts.tsv* | StackOverflow posts, tagged with one programming language (positive samples). Original file: [source](https://github.com/hse-aml/natural-language-processing/releases/download/project/tagged_posts.tsv)
*starspace_embeddings.tsv* | Word embeddings trained with StarSpace for the task of finding duplicate questions on StackOverflow. The embeddings were trained using duplicates with high similarity as “positive” examples and randomly selected questions (low similarity assumed) as “negative” examples. Splitted in two parts and compressed due to space issues (uncompress and join them into "*starspace_embeddings.tsv*").

## Execution example

Q: Hey<br/>
A: Hi 

Q: How are you doing?<br/>
A: I am doing well. 

Q: What's your hobby?<br/>
A: Playing Soccer, Painting, Writing are my hobbies.How about you? 

Q: How to write a loop in python?<br/>
A: This thread might help you: https://stackoverflow.com/questions/26736277 

Q: How to empty workspace in r studio?<br/>
A: This thread might help you: https://stackoverflow.com/questions/25841549 

Q: How to delete rows in pandas?<br/>
A: This thread might help you: https://stackoverflow.com/questions/24612584 

Q: python3 re<br/>
A: This thread might help you: https://stackoverflow.com/questions/10769394 

Q: What is the difference between c and c++<br/>
A: This thread might help you: https://stackoverflow.com/questions/25180069 

Q: How to make histogram with matplotlib<br/>
A: This thread might help you: https://stackoverflow.com/questions/8369584 

Q: Multithreading in Java<br/>
A: This thread might help you: https://stackoverflow.com/questions/8318 

Q: Catch exceptions C++<br/>
A: This thread might help you: https://stackoverflow.com/questions/336475 

Q: How to transpose hashmap?<br/>
A: This thread might help you: https://stackoverflow.com/questions/2941997 

Q: Initialize empty vector numpy<br/>
A: This thread might help you: https://stackoverflow.com/questions/25276965 

Q: What language do you prefer?<br/>
A: I use Python, Java and C++ quite often. 

Q: What is AI?<br/>
A: Artificial Intelligence is the branch of engineering and science devoted to constructing machines that think. 
