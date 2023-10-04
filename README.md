# NanoPhi
[1/10/23] - first model trained
[3/10/23] - Code was acting up, and sadly, I can't switch to GPT-4 tokenizer without a pretrain, but now we use gpt-2-medium, and the models really improved!

Finetuning GPT-2-medium (355m) with Phi-1.5 like Data, to see how the model improves, and how strong tiny LMs can get
Data is tiny-textbooks and the sciphi dataset

Goals
 - Hope to pretrain GPT-2 instead of finetune, GPU funding would be amazing!
 - total of ~640 million tokens(20x, need to be at 7.1B), undertrained by chinchilla scaling, need more data
 - work with GPT-2 355m, build up strong tiny LMs
 - try integrating Orca's student teacher style with GPT-2
 - Combine Phi informational capabilities, Orca, instruct/chat model, ultimate tinyLM?

## samples
 - Tuning Hyperparams, and messing with top_k and temp, has led to some beautiful outputs, the model is best prompted with part of a sentence, or a textbook topic

Small, textbook-like paragraph
```
User: What is the Python Programming Language?
Bot: What is the Python Programming Language?

Python is a programming language designed for data processing and data manipulation. Its main features are the following:
It's a powerful, easy-to-understand and powerful programming language, making it well-suited for data science applications
The language is designed to make it easier to develop complex, real-time systems and data structures, such as data frames and data-driven systems
The language provides powerful tools for data analysis and data visualization, such as plotting, data analysis, and data-driven design
The language's syntax and features allow for the development of complex systems, data frames, and data-driven systems
```

Kind works edge topics not in the data
```
User: What are memes?
Bot: What are memes?

Merely a phrase or expression that has a specific meaning, often in a simple, memorable and easy-to-digest form, that can be repeated and repeated over and over.
What does this have to do with music?
Music is a visual language that is used to convey ideas, feelings, messages, and feelings. It's a way to convey ideas and feelings, and it can be used both in art and in everyday life.
How does music relate to memes?
Music can be a powerful tool for communicating and communicating ideas and feelings, and it can inspire and encourage people to create their own memes.
Music can also be a great way to connect with others, such as through
```

Can show topic differentiation
```
User: Whats the difference between Astronomy and Astrology?
Bot: Whats the difference between Astronomy and Astrology?

Astrology is a branch of astrological studies. It deals with the principles of celestial movement and the planets and the stars. On the other hand Astronomy is a branch of astronomy. It deals with the principles of astronomy and the stars and the planets. Astronomy deals directly with the laws of celestial movement, while Astrology deals with the laws of the stars and the planets.
```

Solved a simple logic puzzle!
```
User: If a beaver makes wooden Dams, and Dams are on rivers, the beavers dam must be on a
Bot: If a beaver makes wooden Dams, and Dams are on rivers, the beavers dam must be on a river.
```
