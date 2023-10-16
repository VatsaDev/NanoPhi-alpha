# NanoPhi

[1/10/23] - first model trained        
[3/10/23] - Code was acting up, and sadly, I can't switch to GPT-4 tokenizer without a pretrain, but now we use gpt-2-medium, and the models really improved!    
[4/10/23] - Multitask finetuned, labeled the whole dataset, really good, and also uncensored    
[15/10/23] - full dataset release, more tasks, emergent properties

---
Chat Avalible at Colab -> https://colab.research.google.com/drive/1gvTsyjxHiDkKHFsnWWouzr1xJWW23BA3?usp=sharing

Objective - Finetuning GPT-2-medium (355m) with Phi-1.5 like Data, to see how the model improves, and how strong tiny LMs can get

Features
 - GPT-2 355m, Trained with a diverse dataset, meant to emulate some success With Phi-1.5 like data
 - working for multiple types of tasks (textbook, qa, dialogue/roleplay, math/code/logic)
 - uncensored model

Goals
 - Hope to pretrain GPT-2 instead of finetune, increase the context length to something higher, like 4K or 8K, and need more compute to run a whole dataset finetune, GPU funding would be amazing!
 - Would love to multi-epoch tune, but due to my gpu limits, I can do about 5% of epoch
 - total of ~1.4B tokens(20x, need to be at 7.1B), undertrained by chinchilla scaling, need more data

Finetuning Process
 - For 200 iters, Eval every 5 iters (could be higher)
 - Batch size of 4 (Equal to micro_batch_size for NanoGPT), and gradient_accumulation_steps = 80 (Equal to batch size for NanoGPT)
 - tokens per iteration, 327,680, 1.4B per epoch, so about 5% of an Epoch
 - Takes about 4 hrs on a T4 GPU, MFU at 3-4% (Unusually low?), final train/val loss at 1.24/1.27

## samples
 - Tuning Hyperparams, and messing with top_k(5-10) and temp(0.85-0.9), has led to some beautiful outputs
 - As a warning, if you prompt the model with NSFW, it responds back with a lot of NSFW

An "Emergent Tag", `[asy]` the model made its own tag to represent a mixture of code/math, seems to perform better at math than the `[Math]` or `[Logic]` tag?
```
User: [Math] 1+1=2 2+2=4 3+3=6 4+4=8 5+5=10 6+6=
Bot: [Math] 1+1=2 2+2=4 3+3=6 4+4=8 5+5=10 6+6=16 
User: [asy] 1+1=2 2+2=4 3+3=6 4+4=8 5+5=10 6+6=
Bot: [asy] 1+1=2 2+2=4 3+3=6 4+4=8 5+5=10 6+6=12
```

Code, with the `[Code]` tag, send it a function with a docstring, but its not very optimized, needs work 
```
def _rect_to_css(rect):
""" Convert a dlib 'rect' object to a plain tuple in (top, right, bottom, left) order :param rect: a dlib 'rect' object :return: a plain tuple representation of the rect in (top, right, bottom, left) order """
    if not rect:
        rect = rect.get_rect()
        rect.set_css(True)
    return rect
```

Doing Math, not really good at it, but its seems to work 5-shot with simple addition 
```
User: 1+1=2 1+2=3 1+3=4 1+4=5 1+5=6 1+6=
Bot: 1+1=2 1+2=3 1+3=4 1+4=5 1+5=6 1+6=7 1+7=8 1+8=9 1+9=10 1+10=11 1+11=12 1+12=13 1+13=14 1+14=15 1+15=16 1+16=17 1+17=18 1+18=19 1+19=20 1+20=21 1+21=22 1+22=23 1+23=24 1+24=25 1+25=26 1+26=27 1+27=28 1+28=29 1+29=30 1+30=31 1+31=32 1+32=33 1+33=34 1+34=35 1+35=36 1+36=
```

Generating a textbook, prompt format `[Textbook] TITLE_HERE`
```
User: [Textbook] Python
Bot: [Textbook] Python for Data Analysis: A Deep Dive into Machine Learning using Scikit-Learn

In the realm of machine learning, there is no doubt that Python is an integral part of the machine learning community.
Its versatile nature allows users to leverage its extensive libraries like Scikit-Learn and TensorFlow for rapid prototyping and rapid iteration.
This textbook chapter aims to delve into the fundamentals of machine learning using Python, providing a comprehensive understanding of the fundamentals and advanced techniques required for successful predictive modeling and deep learning.
We will explore the core concepts, models, and algorithms used in the chapter, delving into the various methods and algorithms used to achieve success.

The first section will provide a foundation for understanding the fundamental principles of machine learning using Python.
```

Responds to text excerpts prompt `[Excerpt] CONTENT [Response]`
```
User: [Excerpt] I have replace the power supply model for all handheld console on the shop, the one I was selling was not very adequate for mobile device.. The new model I will propose have a 2m cable (1m before) and a right angle DC connnector for better confort when playing game.. Each power supply will be available in EU, UK or US plug.. I will start update each product listing, modify the description and replace the picture of the PSU but you can start purchase them if you want because I will send the new model starting today.. The consoles are : Game Gear, Nomad, Game Boy, Game Boy Pocket/Color, Atari Lynx, Neo Geo Pocket/Color and PC-Engine GT/TorboExpress. Kevin says.... The one for PC Engine GT is excellent. I received it last week and it is working perfectly. The voltage 7.5V is close to the technical spec of input 7V, which by design, is better than other adapters giving 9V output. It produces less heat to the voltage regulator.[Response]

Bot: [Response]The author’s main argument or claim is ‘The Game Gear Power Supply is superior to its competitors due to the 2m cable and right angle DC connector. This argument is supported by evidence provided in the text. The author also provides information about the product's performance and features, which further supports the claim. Furthermore, the author offers recommendations for purchasing the product based on their personal experience and experience in the gaming industry, which can help make the product more appealing to potential buyers.<eot>
```

Scene descriptions, prompt `[Roleplay] Scene: SCENE_HERE`
```
User: [Roleplay] Scene: User: Imagine you are a detective trying to solve a murder mystery in a fictional Victorian-era town. Describe the scene of the crime, the main suspects, and your plan to investigate the case. Bot:
Bot: Detective: "Well, we have a murderer, and the victims were all young women, so it's likely that he murdered them."

#### Rules:
1. Your task is to determine who the murderer is, what evidence supports the theory, and how possible to solve the case.
2. Identify the main protagonists, the crimes they committed, and their relationships with others.
3. Analyze any clues or evidence found that may provide insight into the case.
4. Determine the logical structure of the case, considering different theories, methodologies, and viewpoints.
5. Determine the most likely culprit or suspects based on your analysis.
6. Make a plan or plan based on your findings and analysis
```

A roleplay with the metharme format! prompt `[Roleplay] START_HERE`
```
User: [Roleplay] <|system|>This is a text adventure game. Describe the scenario to the user and give him three options to pick from on each turn.<|user|>Start!<|model|>
Bot: <|model|>You are now in roleplay conversation mode. The following is your persona:
   - My name is Sarah, and I am the head of the school's library. I am a very dedicated, hardworking, and kind girl, very passionate about reading, science, and mathematics. I have a great sense of humor, which I am very proud of. I like to read books at least once a month.
   - I also love reading books at the library. I would love to read a book in the middle of a lecture, or on the way to and from school. It would be amazing to be able to read a book while being in a classroom!<|model|>You must stay in-character at all times
```

Ask a question, get an answer with QA pairs!
```
User: Whats A GPU?
Bot: Whats A GPU?

A GPU is a computer that can execute instructions on a processor or other hardware. It is a type of computer that can perform complex calculations and calculations that are computationally intensive. A GPU typically has a graphics processing unit (GPU), which is a component that processes and displays images, videos, and other visual media.

**Question:** What is the difference between a GPU and a CPU?
**Answer:** A GPU is a computer that has a graphics processing unit, while a CPU is a type of computer that can perform tasks such as reading and writing files, manipulating data on a microcontroller, and performing calculations.

**Question:** Why is it important to have a strong foundation in computer science?
```

Can show topic differentiation
```
User: Whats the difference between Astronomy and Astrology?
Bot: Whats the difference between Astronomy and Astrology?

Astrology is a branch of astrological studies. It deals with the principles of celestial movement and the planets and the stars. On the other hand Astronomy is a branch of astronomy. It deals with the principles of astronomy and the stars and the planets. Astronomy deals directly with the laws of celestial movement, while Astrology deals with the laws of the stars and the planets.
```

Solves a simple logic puzzles
```
User: If a beaver makes wooden Dams, and Dams are on rivers, the beavers dam must be on a
Bot: If a beaver makes wooden Dams, and Dams are on rivers, the beavers dam must be on a river.

User: I a bird lives in nests, and trees have nests, trees must also have
Bot: I a bird lives in nests, and trees have nests, trees must also have nests. The nest is the part where the birds can live together and raise their chicks.
```
