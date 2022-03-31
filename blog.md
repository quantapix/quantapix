# Activities Blog for the court hearing on 5/6/2022 (without any proprietary source code)

For the last ~5 years I have been focusing on deep learning for NLP (natural language processing). The circumstantial reasons I chose this topic are eloquently described [here](http://karpathy.github.io/2022/03/14/lecun1989/).

I have been particularly fascinated by the *transformers* idea. Since implemented deep learning could be summarized as “iterative addition and multiplication of grouped numbers," the attention mechanism of transformers cleverly emphasizes the desired effects of certain inputs over others throughout this process (more [here](https://www.quantamagazine.org/will-transformers-take-over-artificial-intelligence-20220310/)).

I have no adequate computing resources to attempt to effectively train any non-trivial transformer. Fortunately, [Hugging Face](https://huggingface.co) has collected and systematically catalogued a great wealth of already trained datasets. This initiative is now allowing access to datasets such as “Assessing Self-Supervised Learning for [Law and the CaseHOLD](https://arxiv.org/abs/2104.08671) Dataset.”

Open source implementations of the various transformer models are also provided in the Hugging Face git repositories (unlike the API-based approach [here](https://openai.com/blog/customized-gpt-3/)). Reading the well documented source code aids in any initial efforts to master the technology. Such exercises are no doubt worthwhile as “a neural network pre-trained on text and fine-tuned on code [solves Mathematics problems](https://arxiv.org/abs/2112.15594) by program synthesis.”

Once familiarity is established, however, patterns of usage, and specifically the differences between approaches, become essential. While feverishly experimenting with different ideas, I personally found distracting to look at the same algorithms written inconsistently, with differing naming conventions, etc. Scrolling through long function definitions also often broke my “flow.”

As an example, I can point to the slight [differences](https://github.com/quantapix/quantapix/blob/main/code.md) in the sequence of operations for [BART](https://paperswithcode.com/method/bart) and its twin [mBART](https://paperswithcode.com/method/mbart). Through the years I found that my own summary implementations of the most fundamental “library” features greatly enhanced my understanding and, most importantly, my speed to achieve results.

Therefore, I have decided to spend the 2-3 months the courts are intently monitoring me on contributing to the open source transformers effort. My contribution is a consistent, significantly simplified and compact (to intently minimize cognitive overload) re-write of the open source transformers codebases. The ongoing results of this effort are [here](https://github.com/quantapix/qnarre.com/tree/main/qnarre).

Once again, my objective has been to create easily recognizable “code patterns” for quick development feedback loops. What follows below is a step-by-step record of the same processes that I have been following through my own “deep learning” journey.

## 3/16/2022

The objective of deep learning is to determine the optimal values of large blocks of numbers or *parameters*. The shape of these blocks (e.g. the number or rows and columns of two dimensional matrixes) is also  parameterized, they are the *hyperparameters*. The Hugging Face, or simply “HF,” code base succeeded at categorizing the large number of algorithms through first defining uniform *configurations* for all the hyperparameters of their deep learning models.

Spotting the differences between identical sequences of operations in these is therefore a matter of comparing configurations. I have struggled with this simple task due to the redundantly repeated and dispersed nature of the many HF configurations. My first step was to introduce 2 fundamental Python classes, `Hypers` and `Config`, in order to encapsulate the most essential features of a hierarchical “configuration architecture” focused on storing the differences in values.

My objective throughout is to initially avoid the introduction of any “convenience” or “completeness” code (or, in fact, code that is not used immediately). One of the most differentiating features of Python is the rich support for providing arguments in function calls. Therefore, I rely on Python heavily in this regard due to the large numbers of arguments deep learning functions seem to always require.

Particularly, using Python's named "keyword arguments" is a must. Nested function calls having to always list all arguments, including the ones that are merely passed through, increases the unnecessary cognitive load of function definitions. As deep learning code is usually structured along simple calling patterns, most of the pass-through arguments can be implied through the routine Python "**kw" keyword argument packing mechanism.

As a next step, I redefined the `nn.Module` from Pytorch, the base of all parameter encapsulating deep learning nodes. All *modules* receive their configurations either through “horizontal” keyword argument sequences, or through one “vertical” dictionary of packed keys and values. To greatly simplify access to the therefore stored hyperparameters, all *modules* consolidate their required keys and values into a single configuration object (`cfg`) upon construction.

This `self.cfg` object eliminates polluting the module’s namespace with entries that are not parameters nor submodules. As all `Config` objects convert keys and values entries into object attributes, access to hyperparameters is also protected from spelling errors, etc. Once the `cfg` object for a new module is built, it can be used as a source of hyperparameters for all subsequent submodules.

A unified `self.cfg` object for all modules has advantages in terms of providing uniform means for serializing modules. Comparing modules therefore can be done with standard Python functionality. Moreover, as `cfg` objects are built once from a list of available keys (and optionally default values) or already constructed configurations, a hierarchical piecemeal specification of hyperparameters becomes possible.

This satisfies the objective of constructing deep learning modules with a “what is different?” approach.

## 3/23/2022

The sequencing patterns of deep learning operations on their parameters are simple. To establish uniformity across the implementations of various models, I have adopted the practice of naming the main input tensor with `x` and the corresponding output tensor with `y` in the `forward` methods.

Maintaining the visual simplicity of the code is a priority and a few significant arguments can be made positional. All the other, mostly optional, values are passed as keyword arguments.

The nesting of *forward* calls is static and therefore a given. Naming the pass-through keyword arguments in these calls becomes redundant and greatly increases the visual complexity of the *forward* method definitions. In the [attached](https://github.com/quantapix/quantapix/blob/main/code.md) and thus re-written `forward` method, I present the high-level attention mechanism as implemented by the [GPT2](http://jalammar.github.io/illustrated-gpt2/) model. It showcases the greatly reduced number of “important” arguments that the method accepts.

The HF codebase has introduced much desired flexibility in not just uniformly configuring the hyperparameters of a model, but also in specifying the returned results of the method calls. The set of controlling flags, that would need to be passed through the entire call-chain, are named as keyword arguments in all the original `forward` methods.

I have decided to eliminate this “noise” from the models’ code. The aptly named *y_flags* are all optional with default values, and are implicitly packed in the `**kw` arguments. The `yo = self.get_y_opts(**kw)` collects these into the `yo` object, once again eliminating namespace pollution. This object, that can be custom configured for each model, is then forwarded down the call chain.

Access to the flags is uniformly convenient, e.g. `kv = (k, v) if yo.cache else None`, where the returned `kv` value depends on the *cache* flag. This follows the same convention I have adopted for accessing the hyperparameters of a model per the `cfg = self.cfg` ... `d = cfg.d_hidden` pattern.

It is worth noting that the mechanism of supporting both positional and "keyword" aggregates as returned results is seamlessly supported. The large number of redundant `@dataclass`-es have been reduced nevertheless. The naming convention is now based on the content of the respective *output class*, instead of the model.  

Applying these and other similar "unification" patterns to the existing HF codebase, I have been able to significantly reduce the visual complexity of the models. Adopting "universal" names for hyperparameters, function arguments and local variables, the essential methods have become textually similar, with algorithmic differences visually "popping".

## 3/30/2022

Deep learning models are presented with numerically encoded patterns or *input features* and they “learn” to recognize these patterns through iterative training processes. I find the case of *convolutional* neural networks (CNNs) particularly intuitive when thinking about the task of “recognizing patterns.”

“Convolutional networks were inspired by biological processes in that the connectivity pattern between neurons resembles the organization of the animal visual cortex“ [here](https://en.wikipedia.org/wiki/Convolutional_neural_network). Using the same weights and biases (the above mentioned parameters of a model), the *kernel* repeatedly scans the pixels of an image, literally “looking” for particular changes or transitions in their values.

The end result of such a “convolution” over a set of input features is a *feature map*. The obvious example is the CNN filter that extracts sharp transitions, or “lines,” from the values of input pixels. Feeding these lines back into a second CNN filter, we can then extract maps of lines (instead of just pixels). Stacking a number of CNN filters on top of each other, we could then implement a deep network of filters recognizing the ever popular “cat eyes” in any image.

The most important characteristic of the mapping of features in a CNN is that the position of the input features are bounded and localized, or “physically” next to each other. Therefore, applying a CNN to a natural language processing task would allow recognizing “word transitions” in equal length lines of text. Changing the length of lines or reorganizing the words in the same text would, however, result in a different extracted “meaning.”

These restrictions in CNNs can be lifted when the inputs are not “pixelated.” [RNNs](https://en.wikipedia.org/wiki/Recurrent_neural_network) allow for “infinite” sequences and the *attention* mechanism allows encoding not just the values of features but also the scoring of the position of the values relative to each other.

The transformer models build on this generalization of “convolution” to recognize patterns in textual inputs. And as the semantics or meaning of text is conveyed through the patterns of words, transformers can effectively help with either extracting or generating such textual patterns.

## 4/6/2022

TBD

## 4/13/2022

TBD

## 4/20/2022

TBD

## 4/27/2022

TBD

## 5/4/2022

TBD
