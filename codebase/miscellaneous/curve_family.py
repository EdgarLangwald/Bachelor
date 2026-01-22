import numpy as np
import matplotlib.pyplot as plt

# ============== CONFIGURATION ==============
PLOT_FAMILY = "double"  # "single" for exponential family, "double" for s-curve/plateau family

SINGLE_AMOUNTS = [
    -0.50, 0.01, 0.35, 0.60, 0.79, 0.93,
    1.07, 1.21, 1.40, 1.65, 1.99, 2.50
]

DOUBLE_AMOUNTS = [
    -0.50, 0.01, 0.4, 0.65, 0.82,
    1.1, 1.23, 1.42, 1.66, 1.99, 2.50
]
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


def plot_single_family():
    x = np.linspace(0, 1, 1000)
    plt.figure(figsize=(14, 9))
    colors = plt.cm.RdYlGn(np.linspace(0, 1, len(SINGLE_AMOUNTS)))

    for i, p in enumerate(SINGLE_AMOUNTS):
        y = np.array([single_curve(xi, p) for xi in x])
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
    plt.figure(figsize=(14, 9))
    colors = plt.cm.RdYlGn(np.linspace(0, 1, len(DOUBLE_AMOUNTS)))

    for i, p in enumerate(DOUBLE_AMOUNTS):
        y = np.array([double_curve(xi, p) for xi in x])
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


if __name__ == "__main__":
    if PLOT_FAMILY == "single":
        plot_single_family()
    else:
        plot_double_family()
