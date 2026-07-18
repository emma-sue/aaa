# SRSC-Lite v1.2 — Reassessment

## Main model

One shared encoder, D1 coarse restoration, deterministic evidence-comparative state head, and D2 correction. Fixed K=2; no tool library, agent, MoE, VLM, recurrence, or dynamic stopping.

State head outputs: signed progress p; normalized deviation magnitude m; frozen 4–8D direction code d=P(e/(||e||+epsilon)). P is fitted/frozen from training labels or fixed random orthogonal projection; no end-to-end P in the main version.

## Exact network topology

SRSC-Lite is a U-shaped macro-architecture, but SRSC is not a replacement for every Transformer block and is not the minimum network block. The restoration operators remain ordinary Restormer blocks. SRSC is an inter-stage state branch whose multi-scale outputs modulate D2.

Input x first passes through a 3×3 overlap patch embedding and a four-level Restormer encoder E, producing feature pyramid {F1,F2,F3,F4} at resolutions {H,W}, {H/2,W/2}, {H/4,W/4}, {H/8,W/8}. E is executed once.

D1 is a standard coarse decoder with its own weights. Starting from F4, every level performs upsample → concatenate the matching encoder skip Fi → 1×1 fusion → Ni Restormer blocks. A 3×3 reconstruction head predicts residual R1 and y1=x+R1.

The Evidence-Comparative State Assessor is placed between D1 and D2. It receives x, y1, y1−x and stop-gradient copies of {Fi}. A shallow three-level convolutional/state encoder outputs a pyramid {S1,S2,S3,S4}; each Si contains predicted p, m and direction code d resized/encoded for that resolution. Ground truth is used only to construct state labels during training.

D2 is a separate correction decoder with its own weights. At level i it performs:

1. upsample the deeper D2 feature;
2. concatenate encoder skip Fi and a shallow y1 feature Gi;
3. fuse by 1×1 convolution;
4. apply SRSC-Mod using Si;
5. process the modulated feature with ordinary Restormer blocks.

SRSC-Mod is inserted after skip/y1 fusion and before the Restormer blocks at each D2 scale:

Fmod=(1+gamma_i(Si))⊙Ffuse+beta_i(Si).

The last convolutions of gamma_i and beta_i are zero-initialized, so D2 begins as an ordinary unconditioned correction decoder. A final 3×3 head predicts Delta1 and y2=y1+Delta1.

Main inference is strictly one-way and fixed-depth:

x → E → D1 → y1 → State Assessor → {Si} → D2 → y2.

There is no recurrent loop in the conference main model. A dotted y2→assessor→D2 extension with shared D2 weights may be drawn as future/K=3 ablation only, not as part of the claimed method.

## Recommended training order

Stage A: pretrain E+D1 as a normal single-stage AiOIR model.

Stage B (the kill experiment): freeze E+D1, initialize one D2 template, and train separate capacity-matched D2 variants for uncertainty, residual, v1.0 U/D, signed progress, SRSC-Lite, and equal-dimensional residual code. This isolates feedback information from changes in y1.

Stage C (only after GO): attach the predicted SRSC-Lite assessor and jointly fine-tune E/D1/assessor/D2, using a smaller learning rate for the pretrained E+D1. The state labels use stop-gradient y1 so D1 cannot change them merely to simplify the assessor task.

## Headline contributions

1. Magnitude is not a restoration state: p distinguishes under-correction from overshoot, while d distinguishes equal-magnitude off-path edits.
2. Evidence-comparative coordinate-conditioned refinement predicts (p,m,d) from x,y1,y1-x and injects them through zero-initialized FiLM/SFT into one ordinary D2.
3. Equal-information state audit: compare SRSC-Lite against equal-dimensional compressed GT residual codes and matched predicted-residual heads, not only uncertainty/residual maps.

## Required controls

- unsigned |p| versus signed p;
- p+m versus p+m+d;
- direction shuffled across samples during training;
- zero direction and equal-dimensional random noise;
- frozen random P versus training-only PCA P;
- equal-dimensional projected direct correction r*=psi(y*-y1);
- matched predictor of that residual code from the same evidence;
- full e only as a leakage diagnostic, never the main model.

## Go criterion

Proceed only when signed p beats unsigned p, direction adds beyond p+m under retrained distribution-matched controls, predicted SRSC-Lite beats matched predicted residual code or demonstrates a clear learnability/robustness advantage, and results are stable across reasonable psi/P choices. Otherwise fall back to signed-progress-only or v1.0.
