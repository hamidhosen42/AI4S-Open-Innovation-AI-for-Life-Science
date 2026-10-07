# ChipStain — video script

Deck: `presentation/ChipStain_presentation.html` (14 slides). Narration: 610 words ≈ 4.2 min at a calm pace — the finished video must stay **under 5:00**.

## How to record

1. Start the demo: `python demo/app.py --weights weights/chipstain.pt`, open http://localhost:7860, and run the first example once (warm-up). Keep *Test-time augmentation* ticked.
2. Open the deck in Chrome, press **F** for fullscreen. Press **N** to see these lines as speaker notes (hide them before recording).
3. Record the screen with QuickTime (File → New Screen Recording) or Loom, with your microphone.
4. On **slides 7–8**, switch to the browser with the demo and do it live (the judges want to see the real system running):
   - slide 7: click the example `example_bf_dense_t150.tif` → **Predict**; point at the three panels and the nuclei count.
   - slide 8: click `example_bf_dense_t150_blur1px.tif` → **Predict**; point at the ⚠ warning and the higher σ.
5. Trim, export 1080p, upload to YouTube as **Unlisted**, check it plays in a private window, and paste the link into the Kaggle Writeup.

## Narration

| # | Slide | Say |
|---|---|---|
| 1 | ChipStain | We are team Hack2Publish — Hamid, Esfer and Foysal — and this is ChipStain. It predicts a nuclear stain from a plain bright-field image, and tells you where not to trust that prediction. |
| 2 | Every nuclear read-out on a chip starts with a stain — and every stain has a cost | On an organ-on-a-chip, cell counts, growth and viability all start from a nuclear stain. But fixation ends the experiment, live dyes harm the cells, and reporter lines are rare for stem-cell tissue. In-silico labeling predicts the stain — but never says when it is wrong. |
| 3 | ChipStain | So our question is: can we predict the stain and say, per pixel and per frame, how much to trust it — and does that signal beat simple alternatives? |
| 4 | Public data, a held-out well, and three training seeds per model | All data are public. We train on HeLa cells and test on a whole held-out well. We add a 60-hour time-lapse, stem-cell-derived neurons, and a competing segmentation model. Every model is trained three times — twenty-one runs. |
| 5 | One network, two outputs: the stain μ and its uncertainty σ | The model is a U-Net with a ResNet-34 encoder. Its head predicts two maps: the stain, mu, and its log-variance, the learned uncertainty, trained with the beta-NLL loss. At inference we average eight rotated and flipped views and add their disagreement to the variance. |
| 6 | From bright-field to a trusted biological read-out — and how we validate it | The whole system: data feed the training runs; the model turns one image into a stain and an uncertainty map; downstream we count nuclei, flag unusual frames, and fit growth curves from trusted frames. Everything is validated, and one command regenerates every result. |
| 7 | 183 nuclei found — the reference annotation has 180 | Here is an image from a well the model never saw, on a laptop CPU. Sigma is high on nuclei and near zero on background. The app counts 183 nuclei; the reference annotation has 180. |
| 8 | The prediction fails — and this model's σ says so | Blurred by one pixel, the prediction fails, and this model's uncertainty jumps from 0.08 to 0.33. But honestly, across three training runs sigma rose under blur only once — so it is not a drift alarm. What does catch it is a cheap check on the input itself: image sharpness. The demo runs it on every upload, and it flags every blurred image in every run. |
| 9 | Same fidelity as an equally strong baseline — a much better uncertainty | Over three runs on 125 test images, ChipStain matches the same ImageNet U-Net trained with an L1 loss and the same augmentation in correlation, 0.768 versus 0.768, at 11 percent higher pixel error. Its uncertainty ranks errors far better — 0.47 versus 0.32 — and is close to calibrated: the 95 percent interval covers 92 percent of pixels, while the control's spread must be scaled 18-fold. |
| 10 | σ finds the errors better than anything you could compute without it | Errors sit on bright nuclei, so brightness and edges are honest rivals. Over the whole ranking curve sigma wins clearly; at a single cut-off the margin is small. |
| 11 | 60 hours of growth from bright-field alone | The biological payoff: on a 60-hour time-lapse, label-free counts give a doubling time of 24.9 hours against 23.9 from the real stain — without a single fluorescence exposure. A sigma frame gate cuts the error from 9.1 to 4.1 percent — mostly by sending failing fields to human review; on the same fields without the gate it is 5.8 percent. |
| 12 | New imaging and new cells: what holds, and what does not | Outside the training data: added noise always raises sigma, but blur or a new modality breaks every model and sigma warns only sometimes. A cheap check on the input — image sharpness and distance from the training features — catches every one of those images in every run, at the cost of some false alarms on a new well. On stem-cell-derived neurons, fine-tuning on one well raises the correlation from 0.59 to 0.76. |
| 13 | When to use ChipStain — and when not to | Our limits: for a count alone, a dedicated segmenter is better — F1 0.831 versus 0.708. ChipStain is for when you need a stain-like image with a trust signal. And we had no organ-on-chip images. |
| 14 | Everything is public — and one command rebuilds it | Code, weights, data and the report are public, and one command regenerates every result. Thank you. |
