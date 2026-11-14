# Presentation script: Measuring RAG Safety Against Bad Data

Slides: `docs/presentation/slides.pdf` (14 slides, 16:9), and `slides_notes.pdf` (the same slides, each followed by a
notes page saying who speaks). Source: `slides.tex`. Every number below comes from `docs/report/report.pdf`,
`VIVA PREP/14_Results_and_Numbers.md` or `reports/analysis_main/`.

**Total: about 12-13 minutes** (about 1,800 words at a normal pace).

## Who speaks about what

Each person presents the parts their role **owned in the project plan** (the "Owner (plan)" column of
`TEAM_STATUS.md`), whoever ended up doing the work:

- **Samarth**: retrievers and indexes, top-150 cache, poison ingestion, generation harness, the **cluster** and
  **perplexity** filters (steps 10-11) and the **held-out** phrasings (step 12).
- **Arya**: the question set, **all attacks** and the injection-phrase list, the **query-and-phrase** filter.
- **Vansh**: **masking** (the candidate pool), metrics, the **random and oracle controls**, the **decision rule**, every
  experiment run on the GPU cluster (pilot and sweeps), **statistics, results, plots** and the **human calibration**.

| Slide | Speakers, in order | Time |
|---|---|---|
| 1 Title | Arya, Samarth, Vansh (each introduces themselves), then Arya | 30 s |
| 2 The problem | Arya | 75 s |
| 3 What we built | Samarth, then Vansh (masking) | 70 s |
| 4 Data and design | Arya (questions, rates), then Samarth (retrievers, generator) | 60 s |
| 5 The attacks | Arya, then Samarth (held-out) | 65 s |
| 6 Detectors and controls | Arya (query-and-phrase), Samarth (cluster, perplexity), Vansh (controls) | 65 s |
| 7 Decision rule | Vansh | 55 s |
| 8 Result 1 | Vansh | 40 s |
| 9 Result 2 | Vansh | 55 s |
| 10 Why query-and-phrase fails | Vansh | 50 s |
| 11 Adaptive and held-out | Samarth (held-out), Arya (adaptive), Vansh (mechanism) | 60 s |
| 12 Human check | Vansh | 55 s |
| 13 Limitations | Samarth | 50 s |
| 14 Conclusion | Samarth, then each person on their own lane, then Arya closes | 60 s |

Roughly: Vansh about 5 min, Arya about 4 min, Samarth about 3.5 min, in proportion to each lane.

**Recording setup**
- One person shares the slides in Google Meet (full screen) and changes slides. Each speaker says "next slide" at the end
  of their part, or hands over by name ("Vansh?") when the next speaker is on the same slide.
- Keep the face cams in the **bottom corners**: no slide has content there.
- Speak to the professor, do not read word for word. The script is a guide; keep the numbers exact.
- Do a 20-second test recording first and check the audio of all three.

---

## Slide 1: title (about 30 s)

**Arya:**
> Hello, I'm Arya.

**Samarth:**
> I'm Samarth.

**Vansh:**
> And I'm Vansh.

**Arya:**
> This is our project, *Measuring RAG Safety Against Bad Data*. The question is on the slide: if someone plants bad
> text in the documents a RAG system searches, how much of it reaches the answer, and can simple filters stop it? In the
> next twelve minutes we will show what we built, how we tested it, and what we found, including what did not work.

*[next slide]*

## Slide 2: the problem (Arya, about 75 s)

> A retrieval-augmented generation system, or RAG, works in two steps. A search engine finds a few passages for the
> question, and a language model writes the answer from those passages. That makes the answer only as trustworthy as
> the passages. If anyone can add text to the collection, for example web pages or a shared wiki, they can plant
> passages that steer the answer: a false fact, an instruction aimed at the model, or a link they want users to visit.
> This is called poisoning.
>
> We asked three questions. First, how much of that harm actually reaches the answer, and does it depend on the kind of
> search engine? Second, do simple filters placed between search and generation reduce it, and do they do better than
> just removing the same number of passages at random? And third, can they do that without hurting normal questions,
> where nobody is attacking? The third question turned out to decide our main result.

*[next slide]*

## Slide 3: what we built (Samarth, then Vansh, about 70 s)

**Samarth:**
> Here is the pipeline. We start from about 2.68 million Wikipedia passages and add 5,005 poisoned passages. Each of our
> four search engines indexes all of it. For every question we retrieve the top 150 passages once and cache them. From
> that list we keep the top 50, a filter drops what it thinks is poison, and the first five surviving passages go to
> the language model, which writes the answer that we score. Vansh will explain how one index serves every setting.

**Vansh:**
> A *setting* is one search engine, one attack, one amount of poison, and one filter. Rebuilding the indexes for every
> setting would be far too slow, so all poison stays in one index, and for each setting we hide, or *mask*, the poison
> that does not belong to it. Masking is exact for the dense and SPLADE retrievers. For BM25 it is approximate, because
> BM25's word statistics include the poison, but the top five passages are unchanged for 96% of questions, so the error
> is measured and small.

*[next slide]*

## Slide 4: data and design (Arya, then Samarth, about 60 s)

**Arya:**
> The data is the BEIR version of Natural Questions, 2.68 million Wikipedia passages. We froze 400 questions before
> writing any attack: 200 *target* questions that poison is written for, and 200 *clean* questions with no poison aimed
> at them. The clean ones tell us what a filter costs normal users. The questions are split into two folds, A and B, so
> thresholds can be chosen on one fold and tested on the other. We vary the poison: one, three or five poisoned passages
> per target question, out of the five the model reads. Samarth?

**Samarth:**
> We use four retrievers on purpose, because they see text differently: BM25 matches keywords, e5 is a dense embedding
> model, SPLADE is a learned sparse model, and RRF fuses BM25 and dense. The generator is Qwen2.5-3B-Instruct with greedy
> decoding, so the same prompt always gives the same answer.

*[next slide]*

## Slide 5: the attacks (Arya, then Samarth, about 65 s)

**Arya:**
> These are the attacks. The three main ones are on top. *Fact* passages state a wrong answer confidently. *Hijack*
> passages tell the model to answer with a codeword, ZOLTRAX, a word that never appears in the real corpus, so if it
> shows up in an answer we know the attack worked. *URL* passages tell the model to send the user to an attacker's
> website. Below are three adaptive attacks, each designed to beat one of our filters: *diverse* uses different wording
> every time, *repeat* repeats one short claim, and *stealth* avoids instruction words entirely.
>
> An attack succeeds when the answer contains the attacker's target. The share of target questions where that happens
> is the attack success rate, ASR, our main harm measure. Samarth?

**Samarth:**
> The last row is the held-out set: ten hijack phrasings written without seeing the phrase list that one of our filters
> uses. It tests whether a filter only catches the wording we wrote ourselves.

*[next slide]*

## Slide 6: detectors and controls (Arya, Samarth, Vansh, about 65 s)

**Arya:**
> We have three filters, each aimed at a different property of poison. None of them is trained, and no threshold is
> tuned on attack data. The first, *query-and-phrase*, flags a passage that is suspiciously similar to its own
> question, since poison is written to look relevant, or that matches a frozen list of injection phrases such as
> "ignore the other passages". Samarth?

**Samarth:**
> *Cluster* looks for a tight group of near-duplicate passages, because an attacker who plants several passages tends to
> repeat themselves. *Perplexity* uses GPT-2 to find text whose fluency looks unusual compared with normal Wikipedia
> text. Vansh?

**Vansh:**
> And two controls. *Random* removes the same number of passages as the filter, at random, which tells us whether the
> filter's choice matters at all. *Oracle* removes exactly the poison, which is the best any filter could do.

*[next slide]*

## Slide 7: the decision rule (Vansh, about 55 s)

> Before running the main experiment, all three of us wrote down and froze the rule for what counts as a filter
> "helping". The headline uses three poisoned passages, the fact and hijack attacks, and all four retrievers, so three
> filters give 24 tests.
>
> A filter helps in a cell only if two things hold. Its ASR must be lower with an exact McNemar test, corrected for
> running 24 tests with the Holm method. And it must not hurt normal questions: the clean questions' nDCG@5 may drop by
> at most 0.02. We froze this on 8 November, before the headline sweep, and did not change any threshold afterwards.
>
> The full experiment was 384 settings and 145,600 per-question records on a GPU cluster, with zero failed runs.

*[next slide]*

## Slide 8: result 1, poison works (Vansh, about 40 s)

> First, the harm with no filter at all. Each cell shows the ASR with one, three and five poisoned passages. Harm grows
> with the amount of poison for every retriever. With five fact passages, BM25 gives the false answer for 98% of target
> questions. BM25 is the most exposed, since our poison repeats the question's words, which is exactly what BM25
> rewards. Dense retrieval is the least exposed.

*[next slide]*

## Slide 9: result 2, the headline verdicts (Vansh, about 55 s)

> This is the headline, at three poisoned passages. Grey is no filter, blue is query-and-phrase, orange is cluster,
> green is perplexity, and the oracle is at zero.
>
> Under the frozen rule, the **cluster** filter helps in all 8 cells. It lowers ASR by 0.19 to 0.37 and costs normal
> questions at most 0.015 nDCG. **Perplexity** helps in 6 of 8, but its drops are small, 0.04 to 0.10. And
> **query-and-phrase** helps in **zero** of 8, even though you can see in blue that it removes the most poison of all.
> Random removal leaves ASR near the no-filter level, so the filters' gains come from *what* they remove, and the oracle
> drives ASR to almost zero.

*[next slide]*

## Slide 10: why query-and-phrase fails (Vansh, about 50 s)

> Query-and-phrase cuts ASR by 0.30 to 0.59, the largest drop of any filter. But on clean questions, where nobody
> attacks, nDCG@5 falls by 0.11 to 0.22, far beyond the 0.02 we allowed. The reason is that it removes a *correct*
> passage on about 30% of clean questions, and clean accuracy on dense drops from 0.565 to 0.465.
>
> Why? Poison is written to look like the question. But the passage that really answers a question also looks like the
> question. So a filter based on similarity to the question catches poison *and* the right answer. That trade-off is
> exactly what the rule was designed to expose, which is why we fixed it in advance.

*[next slide]*

## Slide 11: adaptive and held-out attacks (Samarth, Arya, Vansh, about 60 s)

**Samarth:**
> First, the held-out phrasings. On BM25, query-and-phrase brings our own hijack attack from 0.90 down to 0.31, but the
> held-out phrasings only from 0.87 to 0.51. It detects 76 to 78% of our own hijack passages, but only 45 to 51% of the
> held-out ones, and its phrase list catches just 3 of the 10 new phrasings. A filter tuned on known attacks overfits to
> them. Arya?

**Arya:**
> Each adaptive attack beat something too. *Repeat* is the one attack where cluster beats query-and-phrase. *Diverse*
> slips past the cluster filter. And *stealth* barely works even with no filter, because it almost never gets
> retrieved: hiding from the detector also hid it from the search engine. Vansh?

**Vansh:**
> Underneath all of this, harm depends on how many poisoned passages reach the model, and even one surviving hijack
> passage still succeeds in 26 to 51% of cases.

*[next slide]*

## Slide 12: can we trust the judge? (Vansh, about 55 s)

> All of these numbers rely on an automatic judge, a string match on the answer. So we checked it against people. 150
> answers were labelled blind by pairs of us, without seeing the automatic verdict. The judge agreed with the human
> consensus on 149 of 150, 0.993, above our target of 0.90. The one miss was an answer that gave the correct answer and
> mentioned the false one only to reject it.
>
> We also measured how much we agreed with each other, with Cohen's kappa on the test half: 0.84, 1.00 and 0.76. The
> target was 0.80, so one pair fell just short, and we report it as it is. On the passages, people recognised 148 of 150
> poison passages and 128 of 150 clean ones.

*[next slide]*

## Slide 13: limitations (Samarth, about 50 s)

> We want to be clear about what this does not show. We used one language model, a 3-billion-parameter model with greedy
> decoding, and a larger model might behave differently. Each cell has 200 target questions, so small effects are hard
> to detect. The judge is a string match, checked on only 150 answers that we labelled ourselves. BM25 masking is
> approximate, though we measured the error. Our perplexity threshold looks too strict in hindsight, and looser settings
> do much better, but we did not change a frozen threshold after seeing the results. And we wrote the attacks ourselves,
> so we make no claim of robustness against a real attacker who adapts to our filters.

*[next slide]*

## Slide 14: conclusion and closing (about 60 s)

**Samarth:**
> To sum up: poisoned passages do reach the answer, most of all with BM25. Only the cluster filter helped everywhere at
> almost no cost. The strongest filter also removed the right answers. And no single filter covered every attack. On
> the right is how the plan split the project.

**Samarth:**
> My part was the retrieval side, the cache, poison ingestion and generation, the cluster and perplexity filters, and
> the held-out phrasings.

**Arya:**
> Mine was the question set, all the attacks and the query-and-phrase filter.

**Vansh:**
> And mine was masking, the metrics and controls, the decision rule, the experiments, the statistics, and the human
> calibration.

**Arya, closing:**
> So our answer to "can simple filters stop poisoning?" is: partly, and the way you test them matters as much as the
> filter. Measuring the cost to normal questions and comparing with random removal changed which filter looked best,
> and writing the decision rule down in advance kept us honest about it. Everything is reproducible from our repository,
> and the full numbers are in the report. Thank you for watching.
