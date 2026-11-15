# Method

## 1. Sequential retention benchmark
Given an image x, a watermark W = (Embed, Decode) with payload m, and an editor E, we form
x₀ = Embed(x, m) and xᵣ = E(xᵣ₋₁, pᵣ; sᵣ) for rounds r = 1..R with prompts pᵣ and seeds sᵣ. At every level we
record Decode(xᵣ) against m, and for r ≥ 1 the edit metrics of (xᵣ₋₁ → xᵣ, pᵣ). Seeds sᵣ depend only on
(image, round), so all watermarks and methods see identical editing noise and the comparison is paired.
Retention decay is summarised per (editor, watermark) by the excess bit accuracy
e(L) = (acc(L) − ½)/(acc(0) − ½), fitted by a one-parameter linear (1 − bL) and exponential (e^{−kL}) model.

## 2. Decode → edit → re-embed (control)
mᵣ = Decode(xᵣ₋₁); xᵣ = Embed(E(xᵣ₋₁, pᵣ), mᵣ). The editor never sees the ground-truth payload, so a
decoding error at round r propagates. This needs the watermark encoder at edit time, which is the same access
SafeMark and any editor-side method assume.

## 3. Watermark-guided editing (tested hypothesis)
We adapt decoder-gradient guidance (Gesny et al., ICLR 2026; generation) to editing. At InstructPix2Pix step t
with latent z_t, noise level σ_t and guided noise estimate ε̂ (three-way classifier-free guidance), the clean
latent estimate is x̂₀ = z_t − σ_t ε̂. For steps t in the last (1 − start_frac) fraction of the schedule we compute

  L_wm = BCE( D(VAE_dec(x̂₀)), m̂ ),  g = ∂L_wm / ∂x̂₀,  z_t ← z_t − λ · g / rms(g),

holding ε̂ fixed so that ∂x̂₀/∂z_t = I, then take the scheduler step. D is the watermark's differentiable
decoder (WAM: mask-weighted average of per-pixel bit logits; TrustMark: its decoder network on the centre crop
at its native resolution), m̂ is the payload decoded from the round input, and λ is the step size in latent
units. λ = 0 reproduces the unguided editor exactly (verified bitwise). DWT-DCT-SVD, Tree-Ring and Gaussian
Shading have no differentiable decoder in this form and are not guided.

**What the method can and cannot do.** Guidance changes the editor's output so that the decoder reads m̂. That
is the same outcome re-embedding produces, obtained through the decoder rather than the encoder. Whether it
*preserves* the original watermark signal or *writes* a new one is tested directly: we run guidance on
unwatermarked images with a random target payload (forgery test). If detection rises there, guidance is a
decoder-only re-embedding mechanism and inherits re-embedding's forgery properties.

## 4. Choosing one contribution
After the pilot, re-embedding matched L0 detection in every round at no measurable edit cost, so we did not
claim the guided editor as the contribution. The paper's contribution is the paired sequential benchmark and the
failure analysis; the guided editor is evaluated as a falsification test of the original hypothesis on held-out
images with settings fixed in advance.
