1:# Vipassana Meditation Model: Hierarchical Active Inference
2:
3:**A minimal implementation of three-layer active inference for meditation attention dynamics.**
4:
5:---
6:
7:## Architecture
8:![Thoughtseeds Framework](Thoughtseeds_Framework.jpg)
9:```
10:+--------------------------------------------------------------+
11:| Layer 3: Metacognitive Monitor                               |
12:| - Tracks meta-awareness from L2 thoughtseeds                 |
13:| - Sends meta-awareness to L2 (sensory)                       |
14:+------------------------------+-------------------------------+
15:               | Markov Blanket L2<->L3
16:               | Sensory: meta_awareness
17:               | Active:  precision_sensory, policy_precision
18:+------------------------------v-------------------------------+
19:| Layer 2: Attentional Agent (Thoughtseeds)                    |
20:| - Compresses neural dynamics into 5 thoughtseeds             |
21:| - VAE encoder/decoder + forward dynamics model               |
22:| - Policy posterior q(pi) via softmax of G(pi)                |
23:| - Policy precision from entropy(q_pi)|
24:| - Sensory precision from forward prediction error            |
25:+------------------------------+-------------------------------+
26:               | Markov Blanket L1<->L2
27:               | Sensory: DMN, VAN, DAN, FPN activations
28:               | Active:  mu_x, policy_drive, precision_gain, noise_reduction
29:+------------------------------v-------------------------------+
30:| Layer 1: Neural Generative Process (MVOU)                    |
31:| - 4 brain networks (DMN, VAN, DAN, FPN)                      |
32:| - 4 meditation states (BF, MW, MA, RA)                       |
33:| - Multivariate Ornstein-Uhlenbeck dynamics                   |
34:| - State-dependent coupling (Theta matrices)                 |
35:+--------------------------------------------------------------+
36:```
37:
38:## Core Components
39:
40:### States (4)
41:- **BF** (Breath Focus): Stable attention on breath sensations
42:- **MW** (Mind Wandering): Spontaneous thought proliferation
43:- **MA** (Meta-Awareness): Recognition of mind-wandering
44:- **RA** (Redirect Attention): Volitional return to breath
45:
46:### Networks (4)
47:- **DMN** (Default Mode): Self-referential processing
48:- **VAN** (Ventral Attention): Stimulus-driven detection
49:- **DAN** (Dorsal Attention): Goal-directed control
50:- **FPN** (Frontoparietal): Cognitive flexibility
51:
52:### Thoughtseeds (5)
53:Compress high-dimensional neural state into interpretable mental content:
54:- `attend_breath`: Breath sensation focus
55:- `pain_discomfort`: Physical discomfort awareness
56:- `pending_tasks`: Task-related rumination
57:- `aha_moment`: Insight/creative thought
58:- `equanimity`: Acceptance/non-reactivity
59:
60:---
61:
62:## Installation
63:
64:```bash
65:pip install torch numpy matplotlib
66:```
67:
68:**Requirements:**
69:- Python 3.9+
70:- PyTorch 1.13+
71:- NumPy 1.25+, Matplotlib 3.8+
72:
73:---
74:
75:## Usage
76:
77:### Train Both Phenotypes (Expert & Novice)
78:
79:```bash
80:python run_enactive_inference.py run
81:```
82:
83:This trains both expert and novice models for 10,000 timesteps (default), saves results to `data/`, and generates all plots.
84:
85:**Options:**
86:```bash
87:python run_enactive_inference.py run --timesteps 10000  # Custom training length
88:```
89:
90:### Generate Plots from Existing Data
91:
92:```bash
93:python run_enactive_inference.py plot
94:```
95:
96:Generates 10 publication-quality figures from saved training results.
97:
98:---
99:
100:## Output
101:
102:### Training Results
103:Saved to `data/`:
104:- `training_results_expert_seed42.json`
105:- `training_results_novice_seed42.json`
106:
107:Each contains:
108:- Full state/network/thoughtseed trajectories
109:- Free energy history
110:- Meta-awareness evolution
111:- Transition statistics
112:- Action prediction error summaries
113:
114:### Plots
115:Generated in `plots/`:
116:
117:**Convergence:**
118:- `FigS1_Convergence_Expert.png` - Free energy stabilization, state occupancy
119:- `FigS1_Convergence_Novice.png`
120:
121:**Comparison:**
122:- `Fig3A_Network_Radar.png` - Network profiles across states (Expert vs Novice)
123:- `Fig3B_FE_and_Dwell.png` - Free energy and dwell times per state
124:- `Fig3C_Transitions.png` - State transition probability matrices
125:
126:**Dynamics:**
127:- `Fig4A_Hierarchy_Novice.png` - 3-layer hierarchical dynamics over time
128:- `Fig4B_Hierarchy_Expert.png`
129:
130:**State Space:**
131:- `Fig5_PCA_Trajectories.png` - PCA trajectories across the hierarchy (L2 thoughtseeds + L1 networks)
132:
133:---
134:
135:## Model Features
136:
137:### 1. Hierarchical Markov Blankets
138:Each layer interfaces through Markov blankets defining:
139:- **Sensory states**: What the layer observes from below
140:- **Active states**: How the layer influences layers below
141:
142:### 2. Thoughtseeds as Tractable Bottleneck
143:Layer 2 compresses 4 network activations -> 5 thoughtseeds, making neural state "tractable" for conscious access and metacognitive monitoring.
144:
145:### 3. Forward Dynamics Model
146:Layer 2 learns to predict future network activations from (x, z), enabling:
147:- Anticipatory action selection
148:- Policy evaluation beyond immediate outcomes
149:- Counterfactual reasoning ("what if I stay in MW?")
150:
151:### 4. BPTT Learning
152:Backpropagation Through Time optimizes:
153:- VAE encoder/decoder (representation learning)
154:- Forward model (dynamics prediction)
155:- Loss = VFE + forward prediction error (+ recognition loss for expert)
156:- Policy precision is derived from policy posterior entropy (q_pi)
157:- Sensory precision is derived from forward prediction error with noise floor
158:
159:### 5. Expert vs Novice Phenotypes
160:**Expert:**
161:- **Unfrozen Encoder:** Learns "Amortized Inference" (fast, intuitive state recognition).
162:- **Universal Priors:** Accesses the "Universal/Goal" priors effectively.
163:- **Physiology:** Stronger FPN activation, longer BF dwell times.
164:
165:**Novice:**
166:- **Weak Amortized Inference:** Learns more slowly due to lower learning rate, so recognition is less reliable.
167:- **Universal Priors:** Holds the same goal (Focus) but lacks the intuition to recognize it.
168:- **Physiology:** DMN-dominant profile, shorter BF dwell times.
169:
170:---
171:
172:## Key Results
173:
174:**Behavioral Signatures (run-dependent):**
175:- Expert typically shows lower free energy and a more stable breath-focus basin
176:- Novice shows broader excursions and shallower basins
177:- See `data/` and `plots/` for the current run's quantitative summaries
178:
179:---
180:
181:## File Structure
182:
183:```
184:.
185:+-- run_enactive_inference.py  # Main entry point
186:+-- model/                     # Core Logic
187:|   +-- training_loop.py       # MeditationTrainer class
188:|   +-- l1_generative_process.py  # Layer1Process (MVOU dynamics)
189:|   +-- l2_recognition.py         # Layer2Agent (VAE + forward model)
190:|   +-- l3_metacognition.py       # Layer3Monitor (meta-awareness tracking)
191:|   +-- markov_blankets.py        # Markov blanket interfaces
192:+-- utils/                     # Utilities & Config
193:|   +-- config.py              # Constants and universal priors
194:|   +-- math_utils.py          # Tensor/math operations
195:|   +-- analysis_utils.py      # Metrics computation
196:+-- data/                      # Training results (JSON)
197:+-- plots/                     # Generated figures (PNG)
198:+-- viz/                       # Plotting modules
199:    +-- analysis.py
200:    +-- attractors.py
201:    +-- convergence.py
202:    +-- diagnostics.py
203:    +-- hierarchy.py
204:    +-- radar_plot.py
205:    +-- plotting_utils.py
206:```
207:
208:---
209:
210:## Technical Details
211:
212:### Notation
213:- `x in R^4`: L1 network activations ordered {DMN, VAN, DAN, FPN}
214:- `z in [0,1]^5`: L2 thoughtseed activations
215:- `s in {BF, MW, MA, RA}`: meditation state
216:- `mu_x(s)`, `mu_z(s)`: state-conditioned network and thoughtseed priors
217:- `Theta(s)`: state-conditioned coupling matrix
218:- `NOISE_LEVEL`: L1 process noise variance
219:
220:### Layer 1: Generative Process (MVOU)
221:Continuous-time dynamics:
222:```
223:dx = -Theta(s) (x - mu_x(s)) dt + sigma dW
224:```
225:with `sigma^2 = NOISE_LEVEL`. Euler integration is used with state-specific `Theta(s)`.
226:
227:### Layer 2: Recognition + Variational Inference (VAE)
228:VAE components:
229:- **Encoder**: `q(z|x)` (networks -> thoughtseeds)
230:- **Decoder**: `p(x|z)` (reconstruction)
231:- **Forward model**: `f(x, z)` predicts next networks
232:
233:Per-step VFE:
234:```
235:F(z) = MSE(decode(z), x) + KL_Bernoulli(z || mu_z(s))
236:```
237:
238:Fixed-step VI (2 steps, lr=0.2) optimizes:
239:```
240:L(z) = recon_loss + KL
241:     + precision_w * MSE(z, z_rec)
242:     + (1 - precision_w) * MSE(z, z_prev)
243:```
244:Initialization:
245:```
246:z_init = 0.5 * z_prev + 0.5 * z_rec
247:z_init = precision_w * z_init + (1 - precision_w) * mu_z(s)
248:```
249:where `precision_w = clip(precision_sensory)` in [0, 1].
250:
251:### Sensory Precision (from prediction error)
252:Forward prediction:
253:```
254:x_pred = f(x_{t-1}, a_{t-1})
255:epsilon_fwd = mean((x_t - x_pred)^2)
256:```
257:Precision update:
258:```
259:precision_sensory = NOISE_LEVEL / (NOISE_LEVEL + epsilon_fwd + eps)
260:```
261:
262:### Layer 3: Meta-Awareness
263:Meta-awareness is a weighted sum of thoughtseeds:
264:```
265:meta = sum_i w_i z_i / sum_i w_i
266:```
267:clipped to [0, 1], with state-dependent weights.
268:
269:### Policy Inference (L2)
270:Candidate policies: stay in `s` or transition to each other state.
271:Hazard from dwell:
272:```
273:h = clip(dwell_progress^2)
274:```
275:Policy prior:
276:```
277:E(stay) = 1 - h
278:E(s') = h * P(s' | s)
279:```
280:Expected free energy:
281:```
282:G(pi) = KL_Bernoulli(x_pred || C_s) + H(x_pred)
283:```
284:Policy posterior:
285:```
286:q(pi) = softmax(log E(pi) - gamma * G(pi))
287:```
288:Policy precision:
289:```
290:gamma = 1 - H(q(pi)) / log(N_pi)
291:```
292:Action target:
293:```
294:mu = sum_pi q(pi) * mu_z(s_pi)
295:mu = (1 - h) * mu_current + h * mu
296:mu_x = decode(mu)
297:```
298:
299:### Learning Objective
300:Auto-balanced forward weight:
301:```
302:w_fwd = sum_t F_t / (sum_t epsilon_fwd + eps)
303:```
304:Total loss:
305:```
306:L_total = F + w_fwd * epsilon_fwd + alpha_rec * L_rec
307:```
308:where `L_rec = MSE(encode(x), z*)`. Experts use `alpha_rec = 1`. Novices use a weak
309:weight tied to learning rate: `alpha_rec = lr_novice / lr_expert`.
310:
311:### Training Loop
312:BPTT windows of 50 steps; gradients accumulated per window.
313:
314:---
315:
316:## Configuration
317:
318:Edit `config.py` to modify:
319:- Network/state parameters (Theta matrices, mu attractors)
320:- Thoughtseed priors (THOUGHTSEED_STATE_PRIORS)