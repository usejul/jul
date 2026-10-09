# Glossary

**Adapter (LoRA).** Small extra weights attached to a model's layers. JuL's cross models on the preset's
own weights are LoRA adapters, switched on only while a question type they read is answered.

**autotune.** Trains a head per question from labeled examples and keeps it only if it beats
zero-shot in cross-validation. [More](tuning.md#autotune--when-it-is-off-key).

**Backend.** What runs the model: `mlx`, `torch`, `onnx` or `api`. [More](hub.md#backends).

**Bundle.** A directory written by `jul pack`: a fixed set of questions with everything that does not
depend on the state precomputed. Holds no weights. [More](deployment.md).

**Calibration.** Making probabilities mean what they say. JuL fits a temperature per model, and
`autotune` a temperature and a bias per option on your labels.

**Center.** The vector subtracted before comparing the state with the options: set per preset (the mean of
the options for most, a generic one for `minicpm5-2b`), the mean of a context's examples when it has some.

**Choice.** A question that picks one option among several. Answers `choice`, `probabilities`, `confidence`.

**Confidence.** For a `Choice`, the chosen option's probability. For a `Score`, how peaked the levels'
distribution is (1 = one level takes everything). A `Noul`'s is `max(noul, 1 − noul)` where JuL needs one
(escalation).

**Context.** Unlabeled examples (and optionally a description) of your data, saved under a name; also where
heads are stored. [More](tuning.md#context--what-the-data-looks-like).

**Cross model.** A reading that takes the question and the text together, in one pass, for the questions an
embedding model answers near chance. [More](models.md#cross-models-reading-the-question-and-the-text-together).

**Decision bench.** 2,108 typed questions over 12 task families, in English and French, used to compare
JuL with Jev on Choice, Noul and Score. [Results](https://github.com/usejul/jul#decision-bench-against-jev).

**Decision model.** A model trained to answer typed questions, which brings its own format in a
`decision.json` (pointer reading). [More](models.md#decision-models).

**Description.** What the model reads for an option. The key is what you get back.

**ECE (expected calibration error).** How far the probabilities are from the observed accuracy; lower is
better. The default model: 0.084, Jev: 0.156.

**Encoder.** A small bidirectional model (BERT, e5), read as its sentence embedding. Fast on CPU.
[More](models.md#micro-models-encoders).

**Escalation.** Answering locally and sending only the questions below a confidence bar to the next
decider. [More](serve.md#escalation-local-first-a-bigger-decider-when-unsure).

**Formulation.** One prompt a state is read with, at one layer: `one_word` (the state alone, shared by every
question), `question_options`, `question`. A preset uses two by default.

**Head.** A small classifier trained on the vectors (or on TF-IDF, or both) to correct a model on your task.
The model itself is never changed.

**Hidden state.** The vector a model builds at a layer for a token. JuL reads it instead of letting the model
write.

**Hybrid head.** A head that reads both the model's vectors and the TF-IDF of the text.

**Input limit.** The most tokens a reading takes (a `cross.json`'s `max_length`, a `decision.json`'s
`max_state_tokens`, an encoder's positions). Past it the end is dropped, logged and counted in
`usage.truncated_tokens`; `on_long="error"` refuses the call instead. [More](models.md#input-limits).

**Jev.** TypeSafe's hosted decision model, whose SDK and HTTP protocol JuL follows. JuL is not affiliated
with TypeSafe.

**Jev bench.** Jev's published benchmark: 300 examples over AG News, Banking77 and Emotion, all `Choice`.

**Letter-readout model.** A decision model that answers with an option letter after its own prompt
(JevK5, plumb-4b, Quyet, spark-s1), read from the letters' logits as its runtime reads them.
[More](models.md#letter-readout-decision-models-jevk5-plumb-quyet-spark-s1).

**Noul.** A yes/no question. Answers `noul`, the probability of yes.

**Option.** One possible answer of a question: a key and a description.

**Pointer reading.** How a decision model answers: a trained head over delimiter tokens of its own format.

**Preset.** A model plus its reading settings (layers, prompts, temperature, center), built in or fitted by
`jul models add`. [More](hub.md#models-you-run).

**Reading.** How a model is turned into an answer: vector, cross, pointer, contrastive, letters, a tuned head.
[More](models.md#every-reading-and-every-setting).

**Score.** A question that places the text on an ordered scale. Answers `score`, the expected level.

**State.** The text (or object) a call asks about.

**System One.** Jev's name for its typed-decision API: `POST /v1/systemone`. `jul serve` speaks it,
`SystemOneHTTP` calls it.

**tau (τ).** The temperature of `softmax(cosine / tau)`, fitted per model on development sets.

**Zero-shot.** Answering with no example of the task: only the question and its options.
