import numpy as np
import matplotlib.pyplot as plt
import time

# ============== CONFIGURATION ==============
PLOT_FAMILY = "double"  # "single", "double", or "rational"
TEST_SPEED = False

SINGLE_AMOUNTS = [
    0, 0.17, 0.2833, 0.3667, 0.43, 0.4767,
    0.5233, 0.57, 0.6333, 0.7167, 0.83, 1
]

DOUBLE_AMOUNTS = [
    0, 0.17, 0.3, 0.3833, 0.44,
    0.5333, 0.5767, 0.64, 0.72, 0.83, 1
]

RATIONAL_AMOUNTS = [
    0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1
]

RATIONAL_AMOUNTS = [i/11 for i in range(12)]
# ===========================================


def a(p):
    """Smoother parameter transformation."""
    return p ** 3


def ease_in(x, p):
    return x ** a(p)


def ease_out(x, p):
    return 1 - (1 - x) ** a(2 - p)


def single_curve(x, p):
    """Exponential family: easeIn for p > 1, easeOut for p <= 1."""
    if p > 1:
        return ease_in(x, p)
    else:
        return ease_out(x, p)


def plateau1(x, p):
    ap = a(1 / p)
    if x < 0.5:
        return 0.5 * (2 * x) ** ap
    else:
        return 1 - 0.5 * (2 * (1 - x)) ** ap


def plateau2(x, p):
    ap = a(p)
    if x < 0.5:
        return 0.5 * (1 - (1 - 2 * x) ** ap)
    else:
        return 0.5 + 0.5 * (2 * x - 1) ** ap


def plateau(x, p):
    return 0.3 * plateau1(x, p) + 0.7 * plateau2(x, p)


def s_tan(x, p):
    """Tanh-based s-curve. Undefined at p = 1."""
    k = 5 * (1 - p)
    return 0.5 * (np.tanh(k * (2 * x - 1)) / np.tanh(k) + 1)


def double_curve(x, p):
    """S-curve/plateau family: plateau for p >= 1, s_tan for p < 1."""
    if p >= 1:
        return plateau(x, p)
    else:
        return s_tan(x, p)


def rational_curve(x, p):
    """Cheap rational family: only one division, no exp/log.
    p < 1: ease-out (concave), p = 1: linear, p > 1: ease-in (convex)."""
    return x / (x + p * (1 - x))

'''
def rational_reparam(t):
    """Maps t in [0,1] to p in [0.02, 50] with t=0.5 -> p=1.
    Ensures visually uniform curve spacing."""
    return 50 ** (2 * t - 1)
'''

def rational_reparam(t):
    """
    Approximates the exponential feel using only elementary operations 
    (add/mul/div). Slightly faster than pow().
    """
    # 1. Map to intermediate curve
    # These constants are derived to hit sqrt(0.02) and sqrt(50)
    val = (0.1414 + 0.8586 * t) / (1 - 0.8586 * t)
    
    # 2. Square it to get the ease-in curve on the parameter itself
    return val * val


def rational_curve_normalized(x, t):
    """Rational curve with normalized parameter t in [0,1].
    t=0: max ease-out, t=0.5: linear, t=1: max ease-in."""
    return rational_curve(x, rational_reparam(t))


def plot_single_family():
    x = np.linspace(0, 1, 1000)
    plt.figure(figsize=(5.6, 3.6))
    colors = plt.cm.RdYlGn(np.linspace(0, 1, len(SINGLE_AMOUNTS)))

    for i, p in enumerate(SINGLE_AMOUNTS):
        p_param = p * 3 - 0.5
        y = np.array([single_curve(xi, p_param) for xi in x])
        label = f"p={p:.2f}"
        plt.plot(x, y, label=label, color=colors[i], linewidth=2, alpha=0.8)

    plt.plot(x, x, 'k--', linewidth=1, alpha=0.3, label='Linear (y=x)')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Single Curve Family (easeIn/easeOut)', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_double_family():
    x = np.linspace(0, 1, 1000)
    plt.figure(figsize=(5.6, 3.6))
    colors = plt.cm.RdYlGn(np.linspace(0, 1, len(DOUBLE_AMOUNTS)))

    for i, p in enumerate(DOUBLE_AMOUNTS):
        p_param = p * 3 - 0.5
        y = np.array([double_curve(xi, p_param) for xi in x])
        label = f"p={p:.2f}"
        plt.plot(x, y, label=label, color=colors[i], linewidth=2, alpha=0.8)

    plt.plot(x, x, 'k--', linewidth=1, alpha=0.3, label='Linear (y=x)')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Double Curve Family (S-curve/Plateau)', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_rational_family():
    x = np.linspace(0, 1, 1000)

    plt.figure(figsize=(5.6, 3.6))
    colors = plt.cm.RdYlGn(np.linspace(0, 1, len(RATIONAL_AMOUNTS)))

    for i, t in enumerate(RATIONAL_AMOUNTS):
        p = rational_reparam(t)
        y = rational_curve(x, p)
        label = f"p={t:.2f}"
        plt.plot(x, y, label=label, color=colors[i], linewidth=2, alpha=0.8)

    plt.plot(x, x, 'k--', linewidth=1, alpha=0.3, label='Linear (y=x)')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Single Family Alternative', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def rational_reparam_vec(t):
    """Vectorized reparametrization."""
    val = (0.1414 + 0.8586 * t) / (1 - 0.8586 * t)
    return val * val


def rational_curve_vec(x, p):
    """Vectorized rational curve."""
    return x / (x + p * (1 - x))


def test_speed():
    """
    Benchmark segment evaluation speed using realistic numbers:
    - Training loss: batch_size=50, seq_len=30 -> 1500 segments per batch
    - Simpson's rule: 6 evaluation points per segment
    - Preprocessing optimization: 30 iterations per segment fit
    """
    np.random.seed(42)

    # Scenario 1: Loss computation (one forward pass)
    num_segments_loss = 15000000  # 50 batch * 30 seq
    points_per_segment = 6  # Simpson's rule points

    # Scenario 2: Preprocessing (segment fitting)
    num_segments_preprocess = 2000000  # segments per MIDI file
    optimizer_iterations = 30

    # Generate random data
    t_loss = np.random.uniform(0, 1, num_segments_loss)
    x_loss = np.random.uniform(0, 1, (num_segments_loss, points_per_segment))

    t_preprocess = np.random.uniform(0, 1, (num_segments_preprocess, optimizer_iterations))
    x_preprocess = np.random.uniform(0, 1, (num_segments_preprocess, optimizer_iterations, points_per_segment))

    # Benchmark 1: Loss computation (vectorized)
    start = time.perf_counter()
    p = rational_reparam_vec(t_loss)[:, None]
    _ = rational_curve_vec(x_loss, p)
    loss_time = time.perf_counter() - start

    # Benchmark 2: Preprocessing (vectorized)
    start = time.perf_counter()
    p = rational_reparam_vec(t_preprocess)[:, :, None]
    _ = rational_curve_vec(x_preprocess, p)
    preprocess_time = time.perf_counter() - start

    total_evals_loss = num_segments_loss * points_per_segment
    total_evals_preprocess = num_segments_preprocess * optimizer_iterations * points_per_segment

    print("=" * 60)
    print(f"SEGMENT EVALUATION SPEED TEST" + PLOT_FAMILY)
    print("=" * 60)
    print(f"\n[Loss Computation - 1 batch]")
    print(f"  Segments: {num_segments_loss}, Points/segment: {points_per_segment}")
    print(f"  Total evaluations: {total_evals_loss:,}")
    print(f"  Time: {loss_time*1000:.3f} ms")

    print(f"\n[Preprocessing - 1 MIDI file]")
    print(f"  Segments: {num_segments_preprocess}, Iterations: {optimizer_iterations}, Points/iter: {points_per_segment}")
    print(f"  Total evaluations: {total_evals_preprocess:,}")
    print(f"  Time: {preprocess_time*1000:.3f} ms")
    print("=" * 60)


if __name__ == "__main__":
    if TEST_SPEED:
        test_speed()
    elif PLOT_FAMILY == "single":
        plot_single_family()
    elif PLOT_FAMILY == "double":
        plot_double_family()
    else:
        plot_rational_family()
