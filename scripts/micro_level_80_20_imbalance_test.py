import torch
import torch.nn.functional as F
import copy
from Freedream_sft import GPT, tokenizer, CONFIG
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_batch(text):
    text = text.replace("<|user|>", "").replace("<|assistant|>", "").replace("<|end|>", "")
    ids = tokenizer.encode(text)
    if len(ids) > 256: ids = ids[:256]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    y = torch.tensor([ids[1:] + [tokenizer.token_id("<|end|>")]], dtype=torch.long, device=device)
    return x, y


def run_fixed_100_test():
    tokenizer.load("tokenizer_v2.json")
    vocab = tokenizer.size()
    base_model = GPT(vocab).to(device).to(torch.bfloat16)

    ck = torch.load("pretrain_v2.pt", map_location="cpu", weights_only=False)
    clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()}
    base_model.load_state_dict(clean_state_dict, strict=True)
    del ck

    # 你的 50 条 Easy 数据池
    easy_pool = [
        "Hello! How are you doing today?", "I am fine, thank you. And you?",
        "What is the weather like in London?", "It is usually raining and quite cold.",
        "Can you recommend a good restaurant?", "Sure, there's a great Italian place nearby.",
        "I like to read books in my free time.", "That's a wonderful hobby to have.",
        "Do you have any pets?", "Yes, I have a golden retriever named Max.",
        "I am going to the supermarket.", "Don't forget to buy some milk.",
        "What time is it right now?", "It is currently 3:30 PM.",
        "I love listening to music.", "What kind of music do you like?",
        "Have a great weekend!", "You too, take care!",
        "Good morning, did you sleep well?", "Yes, I slept like a baby.",
        "The wind whispered through the ancient trees.",
        "She looked at the horizon, where the sun melted into the ocean.",
        "He was a man of few words.", "The library was a sanctuary of silence.",
        "Time is like a river, flowing endlessly.", "The old bookstore smelled of dust and old paper.",
        "The abandoned mansion stood on the hill.", "The train rattled through the countryside.",
        "Dreams are the subconscious mind's way.", "The scent of blooming jasmine filled the air.",
        "The artist dipped his brush into the vibrant paint.", "The ancient city was a labyrinth of narrow streets.",
        "A gentle breeze rustled the autumn leaves.", "The ocean is a vast and endless expanse.",
        "Her laughter was like a clear bell.", "The mountain peak was veiled in thick white clouds.",
        "A single candle flickered in the dark room.", "The path was lined with cherry blossoms.",
        "The sun set behind the mountains.", "The river flowed gently through the valley.",
        "The stars twinkled in the night sky.", "The moon cast a pale glow on the water.",
        "The forest was alive with the sound of birds.", "The city bustled with activity.",
        "A butterfly landed on the flower.", "The rain fell softly on the roof.",
        "The snow covered the ground in a white blanket.", "The wind howled outside the window.",
        "The fire crackled in the fireplace.", "The room was warm and inviting."
    ]

    # 你的 50 条 Hard 数据池
    hard_pool = [
        "Solve the differential equation: dy/dx = 2xy.", "The integral of x^2 dx from 0 to 1 is 1/3.",
        "Prove that the sum of the first n odd numbers is n^2.", "By induction, the base case n=1 holds.",
        "What is the Fourier transform of a Gaussian function?", "It remains a Gaussian function.",
        "State the Heisenberg uncertainty principle.", "You cannot know position and momentum precisely.",
        "Calculate the determinant of a 3x3 matrix.", "Use the Laplace expansion along the first row.",
        "Explain the concept of a Riemann manifold.", "It locally resembles Euclidean space.",
        "What is the Schwarzschild radius?", "It defines the event horizon of a black hole.",
        "Derive the quadratic formula.", "Start with ax^2 + bx + c = 0 and complete the square.",
        "Explain the Monty Hall problem.", "Switching gives a 2/3 probability of winning.",
        "What is the Navier-Stokes equation?", "It describes the motion of viscous fluids.",
        "def quicksort(arr): return arr if len(arr) <= 1 else quicksort([x for x in arr[1:] if x < arr[0]]) + [arr[0]] + quicksort([x for x in arr[1:] if x >= arr[0]])",
        "class Node: def __init__(self, val): self.val = val; self.left = None; self.right = None",
        "import numpy as np; a = np.zeros((3,3)); print(a.T)",
        "public static void main(String[] args) { System.out.println(\"Hello\"); }",
        "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
        "SELECT * FROM users WHERE age > 18 AND country = 'US';",
        "const fetchData = async () => { const res = await fetch(url); return res.json(); }",
        "int* p = (int*)malloc(sizeof(int) * 10); free(p);",
        "try { int x = 10 / 0; } catch (ArithmeticException e) { print(e); }",
        "The Industrial Revolution was the transition to new manufacturing processes.",
        "Photosynthesis is a biological process used by many organisms to convert light energy.",
        "The Roman Empire was one of the largest and most influential empires in history.",
        "Quantum mechanics is a fundamental theory in physics.",
        "The global supply chain is a network of organizations and resources.",
        "Artificial neural networks are computing systems inspired by biological brains.",
        "The French Revolution was a period of radical political and societal change.",
        "Climate change includes global warming driven by human emissions.",
        "The stock market is a public entity for trading company stock.",
        "Cognitive psychology is the scientific study of mental processes.",
        "The Renaissance was a period of European cultural and economic rebirth.",
        "Blockchain is a type of distributed ledger comprised of blocks.",
        "The human genome is a complete set of nucleic acid sequences.",
        "Machine learning is the study of algorithms that improve automatically.",
        "The Cold War was a period of geopolitical tension.",
        "Evolution is the change in heritable characteristics of populations.",
        "Macroeconomics deals with the performance and behavior of an economy.",
        "The solar system is the gravitationally bound system of the Sun.",
        "Cryptography is the practice of secure communication.",
        "The Bill of Rights comprises the first ten amendments.",
        "Calculus is the mathematical study of continuous change.",
        "Linear algebra is the branch of mathematics concerning linear equations.",
        "Thermodynamics is a branch of physics that deals with heat and work.",
        "Electromagnetism is a branch of physics involving the study of the electromagnetic force.",
        "The theory of relativity was developed by Albert Einstein.",
        "String theory is a theoretical framework in which point-like particles are replaced by one-dimensional objects.",
        "The Higgs boson is an elementary particle in the Standard Model.",
        "Dark matter is a hypothetical form of matter thought to account for approximately 85% of the matter in the universe.",
        "A Turing machine is a mathematical model of computation."
    ]

    # 🚨 物理截断，无论上面写了多少条，统统只取前 50 条
    easy_pool = easy_pool[:50]
    hard_pool = hard_pool[:50]

    assert len(easy_pool) == 50 and len(hard_pool) == 50

    total_steps = 100
    accumulation = 10  # 8 Easy + 2 Hard

    # ================= Path 1: AdamW =================
    model_adamw = copy.deepcopy(base_model)
    optimizer_adamw = torch.optim.AdamW(model_adamw.parameters(), lr=2e-5)
    model_adamw.train()

    adamw_grad_norms = []
    adamw_loss_h = []  # 只记录 Hard 的 Loss

    print("\n" + "=" * 50)
    print("🚀 AdamW Baseline (8 Easy + 2 Hard)")
    print("=" * 50)

    for step in range(total_steps):
        optimizer_adamw.zero_grad()
        loss_h_record = 0.0

        # 8 个 Easy
        for i in range(8):
            idx = (step * 8 + i) % 50
            b = get_batch(easy_pool[idx])
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model_adamw(*b)
            (loss / accumulation).backward()

        # 2 个 Hard
        for i in range(2):
            idx = (step * 2 + i) % 50
            b = get_batch(hard_pool[idx])
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model_adamw(*b)
            (loss / accumulation).backward()
            if i == 0: loss_h_record = loss.item()

        g_norm = torch.nn.utils.clip_grad_norm_(model_adamw.parameters(), float('inf')).item()
        adamw_grad_norms.append(g_norm)
        optimizer_adamw.step()
        adamw_loss_h.append(loss_h_record)

        if step < 5 or step % 20 == 0:
            print(f"Step {step + 1:03d} | Hard L: {loss_h_record:.3f} | Grad N: {g_norm:.2f}")

    del model_adamw, optimizer_adamw
    torch.cuda.empty_cache()

    # ================= Path 2: LAP-SDR =================
    model_lap = copy.deepcopy(base_model)
    optimizer_lap = torch.optim.AdamW(model_lap.parameters(), lr=2e-5)
    model_lap.train()

    lap_grad_norms = []
    lap_loss_h = []

    print("\n" + "=" * 50)
    print("🛡️ LAP-SDR (8 Easy + 2 Hard)")
    print("=" * 50)

    for step in range(total_steps):
        optimizer_lap.zero_grad()
        loss_h_record = 0.0

        step_batches = []
        for i in range(8):
            idx = (step * 8 + i) % 50
            step_batches.append(get_batch(easy_pool[idx]))
        for i in range(2):
            idx = (step * 2 + i) % 50
            step_batches.append(get_batch(hard_pool[idx]))

        # Phase 1: Probe
        probe_losses = []
        with torch.no_grad():
            for b in step_batches:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pl = model_lap(*b)
                probe_losses.append(pl.item())

        # Phase 2: Route
        loss_tensor = torch.tensor(probe_losses, device=device)
        weights = F.softmax(loss_tensor / 1.0, dim=0) * accumulation
        weights = torch.clamp(weights, min=0.6, max=1.5)
        final_weights = weights * (accumulation / weights.sum())

        # Phase 3: Train
        for i, b in enumerate(step_batches):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model_lap(*b)
            (loss * final_weights[i] / accumulation).backward()
            if i == 8: loss_h_record = loss.item()

        g_norm = torch.nn.utils.clip_grad_norm_(model_lap.parameters(), float('inf')).item()
        lap_grad_norms.append(g_norm)
        optimizer_lap.step()
        lap_loss_h.append(loss_h_record)

        if step < 5 or step % 20 == 0:
            print(f"Step {step + 1:03d} | Hard L: {loss_h_record:.3f} | Grad N: {g_norm:.2f}")

    # ================= Results & Plotting =================
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    axs[0].plot(adamw_grad_norms, label='AdamW', color='red', alpha=0.7)
    axs[0].plot(lap_grad_norms, label='LAP-SDR', color='blue', alpha=0.7)
    axs[0].set_title('Gradient Norm')
    axs[0].set_xlabel('Steps')
    axs[0].legend()

    axs[1].plot(adamw_loss_h, label='AdamW (Hard)', color='red', alpha=0.7)
    axs[1].plot(lap_loss_h, label='LAP-SDR (Hard)', color='blue', alpha=0.7)
    axs[1].set_title('Hard Data Loss (Convergence Speed)')
    axs[1].set_xlabel('Steps')
    axs[1].legend()

    plt.tight_layout()
    plt.savefig('split_loss_comparison.png')
    print("\n✅ Plot saved as 'split_loss_comparison.png'")


if __name__ == "__main__":
    run_fixed_100_test()
