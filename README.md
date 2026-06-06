
import tkinter as tk
from tkinter import ttk, messagebox, font
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import math
import re
from abc import ABC, abstractmethod


# ─────────────────────────────────────────────
#  COLOR PALETTE  (lab / warm parchment theme)
# ─────────────────────────────────────────────
BG_DARK      = "#0D1B2A"   # deep navy
BG_PANEL     = "#112233"   # dark blue-slate
BG_CARD      = "#1A2F45"   # mid blue-slate
ACCENT_GOLD  = "#4FC3C3"   # lab teal
ACCENT_AMBER = "#7DE8E8"   # bright cyan
ACCENT_CREAM = "#D6F0F0"   # pale cyan-white
ACCENT_RED   = "#C0494A"   # lab red
ACCENT_GREEN = "#3A9E6F"   # flask green
TEXT_BRIGHT  = "#E8F4F8"   # cool white
TEXT_DIM     = "#6B9AAF"   # muted steel blue
BORDER       = "#1E4060"   # dark border
HOVER_BG     = "#1F3D5C"   # hover state

FONT_TITLE   = ("Georgia", 42, "bold")
FONT_HEAD    = ("Georgia", 28, "bold")
FONT_BODY    = ("Courier New", 22)
FONT_BOLD    = ("Courier New", 22, "bold")
FONT_LABEL   = ("Courier New", 20)
FONT_BTN     = ("Georgia", 22, "bold")
FONT_BIG     = ("Georgia", 48, "bold")


# ══════════════════════════════════════════════
#  EMBEDDED CHEMISTRY MODULES  (no external imports)
# ══════════════════════════════════════════════

class Laws:
    def validate_positive(self, *args):
        for v in args:
            if v is None or v < 0:
                raise ValueError("Invalid input: negative value")

class BasicLaws(Laws):
    def moles(self, mass, molar_mass):
        self.validate_positive(mass, molar_mass)
        return mass / molar_mass
    def mass(self, moles, molar_mass):
        self.validate_positive(moles, molar_mass)
        return moles * molar_mass

class Solutions(Laws):
    def molarity(self, n, v):
        self.validate_positive(n, v); return n / v
    def molality(self, n_solute, kg_solvent):
        self.validate_positive(n_solute, kg_solvent); return n_solute / kg_solvent
    def normality(self, eq, v):
        self.validate_positive(eq, v); return eq / v
    def percent_w_w(self, mass_solute, mass_solution):
        self.validate_positive(mass_solute, mass_solution)
        return (mass_solute / mass_solution) * 100
    def dilution(self, C1=None, V1=None, C2=None, V2=None):
        if   C1 is None: return (C2 * V2) / V1
        elif V1 is None: return (C2 * V2) / C1
        elif C2 is None: return (C1 * V1) / V2
        elif V2 is None: return (C1 * V1) / C2

class AcidsBases(Laws):
    def ph(self, H):   return -math.log10(H)
    def poh(self, OH): return -math.log10(OH)
    def henderson(self, pKa, A, HA):
        if HA == 0: raise ValueError("HA cannot be zero")
        return pKa + math.log10(A / HA)

class GasLaws(Laws):
    R = 0.0821
    def ideal_gas_pressure(self, n, T, V):
        self.validate_positive(n, T, V); return (n * self.R * T) / V
    def ideal_gas_volume(self, n, T, P):
        self.validate_positive(n, T, P); return (n * self.R * T) / P
    def boyle(self, p1=None, p2=None, v1=None, v2=None):
        if   p1 is None: return (p2 * v2) / v1
        elif p2 is None: return (p1 * v1) / v2
        elif v1 is None: return (p2 * v2) / p1
        elif v2 is None: return (p1 * v1) / p2
    def charles(self, v1=None, v2=None, t1=None, t2=None):
        if   v1 is None: return (v2 * t1) / t2
        elif v2 is None: return (v1 * t2) / t1
        elif t1 is None: return (v1 * t2) / v2
        elif t2 is None: return (v2 * t1) / v1

class Thermo(Laws):
    def temp_c_to_k(self, c): return c + 273
    def temp_k_to_c(self, k): return k - 273


# Bond
class Bond:
    def __init__(self, element1, element2, en1, en2):
        self.element1 = element1
        self.element2 = element2
        self._en1 = None
        self._en2 = None
        self.en1 = en1
        self.en2 = en2

    @property
    def en1(self): return self._en1
    @en1.setter
    def en1(self, v):
        if v < 0: raise ValueError("Electronegativity cannot be negative")
        self._en1 = v
    @property
    def en2(self): return self._en2
    @en2.setter
    def en2(self, v):
        if v < 0: raise ValueError("Electronegativity cannot be negative")
        self._en2 = v
    def delta_en(self): return abs(self.en1 - self.en2)
    def bond_type(self):
        d = self.delta_en()
        if d > 1.7:  return "Ionic Bond"
        if d > 0.4:  return "Polar Covalent Bond"
        return "Nonpolar Covalent Bond"
    def polarity(self):
        d = self.delta_en()
        if d > 1.7:  return "High Polarity (Ionic)"
        if d > 0.4:  return "Moderate Polarity"
        return "Nonpolar"


# Reaction
class Compound(ABC):
    def __init__(self, formula):
        self._formula = formula.upper()
    @property
    def formula(self): return self._formula
    @abstractmethod
    def get_atoms(self): pass
    @abstractmethod
    def __str__(self): pass

class Molecule(Compound):
    def get_atoms(self):
        pattern = r'([A-Z][a-z]?)(\d*)'
        elements = re.findall(pattern, self.formula)
        counts = {}
        for (elem, num) in elements:
            num = int(num) if num else 1
            counts[elem] = counts.get(elem, 0) + num
        return counts
    def __str__(self): return self.formula
    def __eq__(self, other): return self.get_atoms() == other.get_atoms()

class CombustionReaction:
    def __init__(self, reactants):
        self._reactants = [Molecule(r) for r in reactants]
    @property
    def reactants(self): return self._reactants
    def predict_products(self): return [Molecule("CO2"), Molecule("H2O")]

class SynthesisReaction:
    def __init__(self, reactants):
        self._reactants = [Molecule(r) for r in reactants]
    @property
    def reactants(self): return self._reactants
    def predict_products(self):
        combined = "".join([r.formula for r in self._reactants])
        return [Molecule(combined)]

class Balancer:
    def __init__(self, reactants, products):
        self.reactants = reactants
        self.products  = products
    def _count_atoms(self, coeffs):
        left, right = {}, {}
        for i, comp in enumerate(self.reactants):
            for e, v in comp.get_atoms().items():
                left[e] = left.get(e, 0) + v * coeffs[i]
        for i, comp in enumerate(self.products):
            for e, v in comp.get_atoms().items():
                right[e] = right.get(e, 0) + v * coeffs[len(self.reactants)+i]
        return left, right
    def _backtrack(self, index, coeffs, n):
        if index == n:
            left, right = self._count_atoms(coeffs)
            return coeffs if left == right else None
        for i in range(1, 10):
            result = self._backtrack(index+1, coeffs+[i], n)
            if result: return result
        return None
    def solve(self):
        n = len(self.reactants) + len(self.products)
        sol = self._backtrack(0, [], n)
        if sol: return sol[:len(self.reactants)], sol[len(self.reactants):]
        return None, None

class ChemicalSystem:
    def __init__(self, reactants_str):
        self.reaction   = None
        self.products   = []
        self.r_coeffs   = []
        self.p_coeffs   = []
        mols = [Molecule(r.strip()) for r in reactants_str]
        if any(r.formula == "O2" for r in mols):
            self.reaction = CombustionReaction([r.formula for r in mols])
        else:
            self.reaction = SynthesisReaction([r.formula for r in mols])
        self.products = self.reaction.predict_products()
        balancer = Balancer(self.reaction.reactants, self.products)
        self.r_coeffs, self.p_coeffs = balancer.solve()
    def __str__(self):
        if not self.r_coeffs: return "Balancing failed"
        left  = " + ".join(f"{c if c>1 else ''}{r}" for c,r in zip(self.r_coeffs, self.reaction.reactants))
        right = " + ".join(f"{c if c>1 else ''}{p}" for c,p in zip(self.p_coeffs, self.products))
        return left + "  →  " + right


# ══════════════════════════════════════════════
#  QUIZ DATA
# ══════════════════════════════════════════════
QUIZ_QUESTIONS = [
    {"q": "What is H2O commonly known as?",
     "opts": ["Hydrogen Peroxide", "Water", "Oxygen Gas", "Acid"],
     "ans": 1, "exp": "H2O is the chemical formula for water (two hydrogen + one oxygen)."},
    {"q": "NaCl is the chemical formula for?",
     "opts": ["Sugar", "Baking Soda", "Table Salt", "Vinegar"],
     "ans": 2, "exp": "NaCl (Sodium Chloride) is common table salt."},
    {"q": "What is the pH of a neutral solution?",
     "opts": ["0", "14", "7", "1"],
     "ans": 2, "exp": "pH 7 is neutral — neither acidic nor basic."},
    {"q": "Which gas law states P1V1 = P2V2?",
     "opts": ["Charles's Law", "Avogadro's Law", "Boyle's Law", "Gay-Lussac's Law"],
     "ans": 2, "exp": "Boyle's Law: at constant temperature, pressure and volume are inversely proportional."},
    {"q": "What does an ionic bond form between?",
     "opts": ["Two nonmetals", "A metal and a nonmetal", "Two metals", "Two gases"],
     "ans": 1, "exp": "Ionic bonds form when a metal transfers electrons to a nonmetal (e.g. Na + Cl → NaCl)."},
    {"q": "What is the unit of molarity?",
     "opts": ["g/L", "mol/kg", "mol/L", "atm"],
     "ans": 2, "exp": "Molarity = moles of solute per liter of solution (mol/L)."},
    {"q": "An exothermic reaction releases energy as?",
     "opts": ["Light only", "Sound only", "Heat to surroundings", "Cold to surroundings"],
     "ans": 2, "exp": "Exothermic reactions release heat energy into the surroundings (negative ΔH)."},
    {"q": "What is the Henderson-Hasselbalch equation used for?",
     "opts": ["Gas pressure", "Buffer pH", "Molarity", "Temperature conversion"],
     "ans": 1, "exp": "Henderson-Hasselbalch: pH = pKa + log([A⁻]/[HA]) — used for buffer calculations."},
    {"q": "Convert 25°C to Kelvin:",
     "opts": ["248 K", "298 K", "273 K", "325 K"],
     "ans": 1, "exp": "K = °C + 273, so 25 + 273 = 298 K."},
    {"q": "A solution with pH = 2 is:",
     "opts": ["Strongly basic", "Neutral", "Weakly acidic", "Strongly acidic"],
     "ans": 3, "exp": "pH 2 is very acidic — far below the neutral pH of 7."},
]


# ══════════════════════════════════════════════
#  BONDS DATA
# ══════════════════════════════════════════════
BONDS_DATA = [
    ("Ionic Bond",
     "Formed by complete transfer of electrons from a metal to a nonmetal.",
     "ΔEN > 1.7",
     "NaCl, MgO, CaCl₂, KBr",
     "Very high melting point, conducts electricity when dissolved, forms crystals."),
    ("Polar Covalent Bond",
     "Electrons shared unequally between two nonmetals with different electronegativities.",
     "0.4 < ΔEN ≤ 1.7",
     "H₂O, HCl, NH₃, HF",
     "Partial charges (δ+ and δ−), creates dipole moment, often soluble in water."),
    ("Nonpolar Covalent Bond",
     "Electrons shared equally between atoms of same or similar electronegativity.",
     "ΔEN ≤ 0.4",
     "H₂, O₂, N₂, CH₄, Cl₂",
     "No partial charges, lower boiling points, often insoluble in water."),
    ("Metallic Bond",
     "Sea of delocalized electrons surrounding a lattice of positive metal ions.",
     "Metal-Metal",
     "Fe, Cu, Au, Al, Na",
     "High electrical/thermal conductivity, malleable, ductile, lustrous appearance."),
    ("Hydrogen Bond",
     "Weak electrostatic attraction between H bonded to F, O, or N and another electronegative atom.",
     "Special dipole-dipole",
     "H₂O···H₂O, DNA base pairs, proteins",
     "Responsible for high boiling point of water, protein folding, DNA double helix structure."),
    ("Van der Waals / London Dispersion",
     "Temporary induced dipoles between nonpolar molecules due to electron fluctuations.",
     "Weakest intermolecular",
     "Noble gases, Cl₂, I₂, hydrocarbons",
     "Strength increases with molar mass and surface area; responsible for liquefaction of noble gases."),
]


# ══════════════════════════════════════════════
#  HELPER WIDGETS
# ══════════════════════════════════════════════

def styled_button(parent, text, command, width=18, bg=ACCENT_GOLD, fg=BG_DARK, active_bg=ACCENT_AMBER):
    btn = tk.Button(
        parent, text=text, command=command,
        font=FONT_BTN, bg=bg, fg=fg,
        activebackground=active_bg, activeforeground=BG_DARK,
        relief="flat", bd=0, cursor="hand2",
        padx=14, pady=8, width=width
    )
    def on_enter(e): btn.config(bg=active_bg)
    def on_leave(e): btn.config(bg=bg)
    btn.bind("<Enter>", on_enter)
    btn.bind("<Leave>", on_leave)
    return btn

def section_label(parent, text):
    return tk.Label(parent, text=text, font=FONT_HEAD,
                    bg=BG_PANEL, fg=ACCENT_AMBER)

def divider(parent):
    return tk.Frame(parent, bg=BORDER, height=1)

def card_frame(parent, **kw):
    f = tk.Frame(parent, bg=BG_CARD, bd=0, relief="flat", **kw)
    return f


# ══════════════════════════════════════════════
#  MAIN APPLICATION
# ══════════════════════════════════════════════

class ChemLabDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Chemistry Laboratory Dashboard")
        self.root.configure(bg=BG_DARK)
        self.root.geometry("1600x950")
        self.root.minsize(1300, 800)

        self.quiz_index   = 0
        self.quiz_score   = 0
        self.quiz_active  = False

        self._build_ui()
        self._show_section("plots")

    # ── MAIN LAYOUT ──────────────────────────
    def _build_ui(self):
        # TOP HEADER BAR
        header = tk.Frame(self.root, bg=BG_PANEL, height=64)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        tk.Label(header, text="⬡", font=("Georgia", 26), bg=BG_PANEL, fg=ACCENT_GOLD).pack(side="left", padx=(18,6), pady=10)
        tk.Label(header, text="CHEMISTRY LABORATORY", font=("Georgia", 18, "bold"),
                 bg=BG_PANEL, fg=ACCENT_CREAM).pack(side="left", pady=10)
        tk.Label(header, text="  —  Interactive Dashboard",
                 font=("Georgia", 12, "italic"), bg=BG_PANEL, fg=TEXT_DIM).pack(side="left", pady=14)

        # status bar (right side of header)
        self.status_var = tk.StringVar(value="Select a section below")
        tk.Label(header, textvariable=self.status_var,
                 font=FONT_LABEL, bg=BG_PANEL, fg=TEXT_DIM).pack(side="right", padx=20)

        # HORIZONTAL NAV BAR
        nav = tk.Frame(self.root, bg=BG_DARK, height=52)
        nav.pack(fill="x", side="top")
        nav.pack_propagate(False)

        self.nav_buttons = {}
        sections = [
            ("plots",     "  Plots  "),
            ("reactions", "  Reactions  "),
            ("laws",      "  Laws  "),
            ("bonds",     "  Bonds  "),
            ("quiz",      "  Quiz  "),
        ]
        for key, label in sections:
            btn = tk.Button(
                nav, text=label, font=FONT_BTN,
                bg=BG_DARK, fg=TEXT_DIM,
                activebackground=BG_CARD, activeforeground=ACCENT_AMBER,
                relief="flat", bd=0, cursor="hand2",
                padx=10, pady=12,
                command=lambda k=key: self._show_section(k)
            )
            btn.pack(side="left", fill="y")
            self.nav_buttons[key] = btn

        # thin gold line under nav
        tk.Frame(self.root, bg=ACCENT_GOLD, height=2).pack(fill="x")

        # CONTENT AREA
        self.content = tk.Frame(self.root, bg=BG_DARK)
        self.content.pack(fill="both", expand=True)

        # Build all section frames
        self.sections = {}
        self.sections["plots"]     = self._build_plots_section()
        self.sections["reactions"] = self._build_reactions_section()
        self.sections["laws"]      = self._build_laws_section()
        self.sections["bonds"]     = self._build_bonds_section()
        self.sections["quiz"]      = self._build_quiz_section()

    # ── SECTION SWITCHER ─────────────────────
    def _show_section(self, key):
        for k, frame in self.sections.items():
            frame.pack_forget()
        self.sections[key].pack(fill="both", expand=True)
        # highlight active nav btn
        for k, btn in self.nav_buttons.items():
            if k == key:
                btn.config(bg=BG_CARD, fg=ACCENT_AMBER,
                           relief="flat",
                           font=("Georgia", 11, "bold"))
            else:
                btn.config(bg=BG_DARK, fg=TEXT_DIM,
                           font=FONT_BTN)
        labels = {"plots":"Plots","reactions":"Reactions",
                  "laws":"Laws","bonds":"Bonds","quiz":"Quiz"}
        self.status_var.set(f"Section: {labels[key]}")


    # ════════════════════════════════════════
    #  SECTION 1 — PLOTS
    # ════════════════════════════════════════
    def _build_plots_section(self):
        outer = tk.Frame(self.content, bg=BG_DARK)

        # Left panel — plot list
        left = tk.Frame(outer, bg=BG_PANEL, width=480)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)

        tk.Label(left, text="PLOTS", font=FONT_HEAD,
                 bg=BG_PANEL, fg=ACCENT_AMBER, pady=14).pack(fill="x")
        divider(left).pack(fill="x", padx=8)

        self.plot_names = [
            "1. Exothermic Profile",
            "2. Endothermic Profile",
            "3. Titration End Point",
            "4. Reaction Rate Curve",
            "5. Rate Gradient (dC/dt)",
        ]
        self.active_plot_var = tk.IntVar(value=0)
        self.plot_btns = []
        for i, name in enumerate(self.plot_names):
            btn = tk.Button(
                left, text=name, font=FONT_BODY,
                bg=BG_PANEL, fg=TEXT_DIM, anchor="w",
                activebackground=HOVER_BG, activeforeground=ACCENT_AMBER,
                relief="flat", bd=0, cursor="hand2",
                padx=14, pady=9,
                command=lambda idx=i: self._show_plot(idx)
            )
            btn.pack(fill="x")
            self.plot_btns.append(btn)

        divider(left).pack(fill="x", padx=8, pady=6)
        styled_button(left, "Show All Plots", self._show_all_plots, width=16).pack(pady=10, padx=12)

        # Right panel — matplotlib canvas
        right = tk.Frame(outer, bg=BG_DARK)
        right.pack(side="left", fill="both", expand=True)

        self.plot_fig, self.plot_ax = plt.subplots(facecolor=BG_DARK)
        self.plot_canvas = FigureCanvasTkAgg(self.plot_fig, master=right)
        self.plot_canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

        # show first plot by default
        self._show_plot(0)
        return outer

    def _highlight_plot_btn(self, idx):
        for i, btn in enumerate(self.plot_btns):
            if i == idx:
                btn.config(bg=HOVER_BG, fg=ACCENT_AMBER, font=FONT_BOLD)
            else:
                btn.config(bg=BG_PANEL, fg=TEXT_DIM, font=FONT_BODY)

    def _show_plot(self, idx):
        self._highlight_plot_btn(idx)
        self.plot_fig.clf()
        ax = self.plot_fig.add_subplot(111)
        ax.set_facecolor(BG_CARD)
        self.plot_fig.patch.set_facecolor(BG_DARK)
        ax.tick_params(colors=TEXT_DIM, labelsize=14)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        ax.title.set_color(ACCENT_CREAM)
        ax.xaxis.label.set_color(TEXT_DIM)
        ax.yaxis.label.set_color(TEXT_DIM)

        t = np.linspace(0, 10, 200)

        if idx == 0:
            y = 30*np.exp(-((t-3)**2)) - 2*t
            ax.plot(t, y, color="#E05050", lw=2.5, label="Exothermic")
            ax.fill_between(t, y, alpha=0.12, color="#E05050")
            ax.set_title("Exothermic Energy Profile", fontsize=18, fontweight="bold")
            ax.set_xlabel("Reaction Coordinate", fontsize=15)
            ax.set_ylabel("Energy", fontsize=15)
            ax.legend(facecolor=BG_PANEL, labelcolor=TEXT_BRIGHT, edgecolor=BORDER, fontsize=14)

        elif idx == 1:
            y = 30*np.exp(-((t-3)**2)) + 2*t
            ax.plot(t, y, color="#5090E0", lw=2.5, label="Endothermic")
            ax.fill_between(t, y, alpha=0.12, color="#5090E0")
            ax.set_title("Endothermic Energy Profile", fontsize=18, fontweight="bold")
            ax.set_xlabel("Reaction Coordinate", fontsize=15)
            ax.set_ylabel("Energy", fontsize=15)
            ax.legend(facecolor=BG_PANEL, labelcolor=TEXT_BRIGHT, edgecolor=BORDER, fontsize=14)

        elif idx == 2:
            x_t = np.linspace(0, 10, 100)
            y_t = np.tanh(x_t - 5)*7 + 7
            dy  = np.diff(y_t)
            ep_idx = np.argmax(np.abs(dy))
            ax.plot(x_t, y_t, color="#E8B84B", lw=2.5, label="Titration Curve")
            ax.scatter(x_t[ep_idx], y_t[ep_idx], color=ACCENT_RED, s=90, zorder=5, label="End Point")
            ax.annotate(f"  End Point\n  ({x_t[ep_idx]:.1f}, {y_t[ep_idx]:.1f})",
                        xy=(x_t[ep_idx], y_t[ep_idx]),
                        xytext=(x_t[ep_idx]+1, y_t[ep_idx]-2),
                        color=ACCENT_CREAM, fontsize=14,
                        arrowprops=dict(arrowstyle="->", color=ACCENT_CREAM))
            ax.set_title("Titration End Point Detection", fontsize=18, fontweight="bold")
            ax.set_xlabel("Volume Added (mL)", fontsize=15)
            ax.set_ylabel("pH", fontsize=15)
            ax.legend(facecolor=BG_PANEL, labelcolor=TEXT_BRIGHT, edgecolor=BORDER, fontsize=14)

        elif idx == 3:
            reactant = np.exp(-0.5*t)
            product  = 1 - np.exp(-0.5*t)
            ax.plot(t, reactant, color=ACCENT_RED,   lw=2.5, label="Reactant")
            ax.plot(t, product,  color=ACCENT_GREEN, lw=2.5, label="Product")
            ax.set_title("Reaction Rate Curve", fontsize=18, fontweight="bold")
            ax.set_xlabel("Time (s)", fontsize=15)
            ax.set_ylabel("Concentration (mol/L)", fontsize=15)
            ax.legend(facecolor=BG_PANEL, labelcolor=TEXT_BRIGHT, edgecolor=BORDER, fontsize=14)

        elif idx == 4:
            reactant = np.exp(-0.5*t)
            rate = np.gradient(reactant, t)
            ax.plot(t, rate, color="#80C080", lw=2.5, label="Rate (dC/dt)")
            ax.axhline(0, color=BORDER, lw=1, linestyle="--")
            ax.fill_between(t, rate, where=(rate < 0), alpha=0.15, color="#80C080")
            ax.set_title("Rate of Reaction — dC/dt", fontsize=18, fontweight="bold")
            ax.set_xlabel("Time (s)", fontsize=15)
            ax.set_ylabel("Rate", fontsize=15)
            ax.legend(facecolor=BG_PANEL, labelcolor=TEXT_BRIGHT, edgecolor=BORDER, fontsize=14)

        ax.grid(True, color=BORDER, alpha=0.5, linestyle="--")
        self.plot_canvas.draw()

    def _show_all_plots(self):
        win = tk.Toplevel(self.root)
        win.title("All Plots")
        win.configure(bg=BG_DARK)
        win.geometry("1100x700")

        fig, axes = plt.subplots(2, 3, figsize=(14, 8), facecolor=BG_DARK)
        fig.subplots_adjust(hspace=0.45, wspace=0.35)
        t = np.linspace(0, 10, 200)
        plot_data = [
            (t, 30*np.exp(-((t-3)**2)) - 2*t, "#E05050", "Exothermic Profile", "Coord", "Energy"),
            (t, 30*np.exp(-((t-3)**2)) + 2*t, "#5090E0", "Endothermic Profile", "Coord", "Energy"),
        ]
        x_t = np.linspace(0, 10, 100)
        y_t = np.tanh(x_t - 5)*7 + 7
        reactant = np.exp(-0.5*t)
        product  = 1 - np.exp(-0.5*t)
        rate     = np.gradient(reactant, t)

        for ax in axes.flatten(): ax.set_visible(False)

        configs = [
            (axes[0,0], t, 30*np.exp(-((t-3)**2)) - 2*t, "#E05050", "Exothermic Profile"),
            (axes[0,1], t, 30*np.exp(-((t-3)**2)) + 2*t, "#5090E0", "Endothermic Profile"),
            (axes[0,2], x_t, y_t,      ACCENT_AMBER, "Titration Curve"),
            (axes[1,0], t,   reactant,  ACCENT_RED,   "Reactant Decay"),
            (axes[1,1], t,   product,   ACCENT_GREEN, "Product Formation"),
            (axes[1,2], t,   rate,      "#80C080",    "Rate dC/dt"),
        ]
        for ax, x, y, color, title in configs:
            ax.set_visible(True)
            ax.set_facecolor(BG_CARD)
            ax.plot(x, y, color=color, lw=2)
            ax.set_title(title, color=ACCENT_CREAM, fontsize=9, fontweight="bold")
            ax.tick_params(colors=TEXT_DIM, labelsize=7)
            for spine in ax.spines.values(): spine.set_edgecolor(BORDER)
            ax.grid(True, color=BORDER, alpha=0.4, linestyle="--")

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)
        canvas.draw()


    # ════════════════════════════════════════
    #  SECTION 2 — REACTIONS
    # ════════════════════════════════════════
    def _build_reactions_section(self):
        outer = tk.Frame(self.content, bg=BG_DARK)

        # Title
        tk.Label(outer, text="CHEMICAL REACTIONS", font=FONT_HEAD,
                 bg=BG_DARK, fg=ACCENT_AMBER, pady=14).pack()
        divider(outer).pack(fill="x", padx=20, pady=(0,10))

        # Input card
        card = card_frame(outer)
        card.pack(fill="x", padx=30, pady=(0,12))

        tk.Label(card, text="Enter reactants (comma-separated, e.g.  C3H8, O2)",
                 font=FONT_BOLD, bg=BG_CARD, fg=ACCENT_CREAM, pady=10).pack(pady=(10,2))

        entry_row = tk.Frame(card, bg=BG_CARD)
        entry_row.pack(pady=6)
        self.reaction_entry = tk.Entry(
            entry_row, font=("Courier New", 18), width=28,
            bg=BG_PANEL, fg=ACCENT_CREAM, insertbackground=ACCENT_GOLD,
            relief="flat", bd=6
        )
        self.reaction_entry.insert(0, "C3H8, O2")
        self.reaction_entry.pack(side="left", padx=(0,10))

        styled_button(entry_row, "Balance", self._run_reaction, width=12).pack(side="left")

        # Result display
        self.reaction_result = tk.Text(
            card, font=("Courier New", 20, "bold"),
            bg=BG_PANEL, fg=ACCENT_AMBER,
            relief="flat", bd=0, height=3,
            state="disabled", wrap="word"
        )
        self.reaction_result.pack(fill="x", padx=16, pady=(4,14))

        divider(outer).pack(fill="x", padx=20, pady=8)

        # Preset reactions
        tk.Label(outer, text="PRESET REACTIONS", font=FONT_BODY,
                 bg=BG_DARK, fg=TEXT_DIM).pack(pady=(6,4))

        presets_frame = tk.Frame(outer, bg=BG_DARK)
        presets_frame.pack(fill="x", padx=20)

        presets = [
            ("Combustion of Propane", "C3H8, O2"),
            ("Combustion of Methane", "CH4, O2"),
            ("Combustion of Ethane",  "C2H6, O2"),
            ("Synthesis: Na + Cl",    "Na, Cl"),
            ("Combustion of Ethanol", "C2H5OH, O2"),
        ]
        for name, formula in presets:
            row = tk.Frame(presets_frame, bg=BG_CARD)
            row.pack(fill="x", pady=2, padx=4)
            tk.Label(row, text=name, font=FONT_BOLD, bg=BG_CARD,
                     fg=ACCENT_CREAM, width=26, anchor="w", padx=12, pady=8).pack(side="left")
            tk.Label(row, text=formula, font=("Courier New", 16, "italic"),
                     bg=BG_CARD, fg=ACCENT_GOLD, width=18, anchor="w").pack(side="left")
            styled_button(row, "Run", lambda f=formula: self._run_preset(f), width=8).pack(side="right", padx=10, pady=4)

        return outer

    def _run_reaction(self):
        raw = self.reaction_entry.get().strip()
        reactants = [r.strip() for r in raw.split(",") if r.strip()]
        if not reactants:
            messagebox.showwarning("Input Error", "Please enter at least one reactant.")
            return
        try:
            system = ChemicalSystem(reactants)
            result = str(system)
        except Exception as e:
            result = f"Error: {e}"
        self.reaction_result.config(state="normal")
        self.reaction_result.delete("1.0", "end")
        self.reaction_result.insert("end", f"  {result}")
        self.reaction_result.config(state="disabled")

    def _run_preset(self, formula):
        self.reaction_entry.delete(0, "end")
        self.reaction_entry.insert(0, formula)
        self._run_reaction()


    # ════════════════════════════════════════
    #  SECTION 3 — LAWS
    # ════════════════════════════════════════
    def _build_laws_section(self):
        outer = tk.Frame(self.content, bg=BG_DARK)

        tk.Label(outer, text="CHEMISTRY LAWS — CALCULATOR",
                 font=FONT_HEAD, bg=BG_DARK, fg=ACCENT_AMBER, pady=14).pack()
        divider(outer).pack(fill="x", padx=20, pady=(0,10))

        # Tabbed interface using ttk.Notebook with custom style
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Lab.TNotebook",        background=BG_DARK,    borderwidth=0)
        style.configure("Lab.TNotebook.Tab",    background=BG_PANEL,   foreground=TEXT_DIM,
                        font=FONT_BODY, padding=[14, 6])
        style.map("Lab.TNotebook.Tab",
                  background=[("selected", BG_CARD)],
                  foreground=[("selected", ACCENT_AMBER)])

        nb = ttk.Notebook(outer, style="Lab.TNotebook")
        nb.pack(fill="both", expand=True, padx=20, pady=6)

        tabs = [
            ("Basic",     self._build_basic_tab),
            ("Solutions", self._build_solutions_tab),
            ("Acids/Bases", self._build_acids_tab),
            ("Gas Laws",  self._build_gas_tab),
            ("Thermo",    self._build_thermo_tab),
        ]
        for label, builder in tabs:
            frame = tk.Frame(nb, bg=BG_CARD)
            builder(frame)
            nb.add(frame, text=f"  {label}  ")

        return outer

    def _law_row(self, parent, label, var, unit=""):
        row = tk.Frame(parent, bg=BG_CARD)
        row.pack(fill="x", padx=20, pady=4)
        tk.Label(row, text=label, font=FONT_BOLD, bg=BG_CARD,
                 fg=ACCENT_CREAM, width=22, anchor="w").pack(side="left")
        entry = tk.Entry(row, textvariable=var, font=FONT_BODY, width=12,
                         bg=BG_PANEL, fg=ACCENT_CREAM, insertbackground=ACCENT_GOLD,
                         relief="flat", bd=4)
        entry.pack(side="left", padx=6)
        if unit:
            tk.Label(row, text=unit, font=FONT_LABEL, bg=BG_CARD, fg=TEXT_DIM, width=8).pack(side="left")
        return entry

    def _result_label(self, parent, var):
        frm = tk.Frame(parent, bg=BG_PANEL, pady=6)
        frm.pack(fill="x", padx=20, pady=8)
        tk.Label(frm, text="Result:", font=FONT_BOLD, bg=BG_PANEL, fg=TEXT_DIM).pack(side="left", padx=10)
        tk.Label(frm, textvariable=var, font=("Courier New", 22, "bold"),
                 bg=BG_PANEL, fg=ACCENT_GOLD).pack(side="left")

    def _build_basic_tab(self, parent):
        tk.Label(parent, text="Moles  =  mass / molar mass", font=FONT_BOLD,
                 bg=BG_CARD, fg=ACCENT_CREAM, pady=12).pack()
        self.basic_mass  = tk.StringVar(value="10")
        self.basic_mm    = tk.StringVar(value="2")
        self.basic_res   = tk.StringVar(value="—")
        self._law_row(parent, "Mass (g):", self.basic_mass, "g")
        self._law_row(parent, "Molar Mass (g/mol):", self.basic_mm, "g/mol")
        self._result_label(parent, self.basic_res)
        styled_button(parent, "Calculate Moles", self._calc_basic).pack(pady=6)

    def _calc_basic(self):
        try:
            m  = float(self.basic_mass.get())
            mm = float(self.basic_mm.get())
            r  = BasicLaws().moles(m, mm)
            self.basic_res.set(f"{r:.4f} mol")
        except Exception as e:
            self.basic_res.set(f"Error: {e}")

    def _build_solutions_tab(self, parent):
        tk.Label(parent, text="Molarity  =  n (mol) / V (L)", font=FONT_BOLD,
                 bg=BG_CARD, fg=ACCENT_CREAM, pady=12).pack()
        self.sol_n   = tk.StringVar(value="2")
        self.sol_v   = tk.StringVar(value="1")
        self.sol_res = tk.StringVar(value="—")
        self._law_row(parent, "Moles (n):", self.sol_n, "mol")
        self._law_row(parent, "Volume (V):", self.sol_v, "L")
        self._result_label(parent, self.sol_res)
        styled_button(parent, "Calculate Molarity", self._calc_solutions).pack(pady=6)

    def _calc_solutions(self):
        try:
            n = float(self.sol_n.get())
            v = float(self.sol_v.get())
            r = Solutions().molarity(n, v)
            self.sol_res.set(f"{r:.4f} mol/L")
        except Exception as e:
            self.sol_res.set(f"Error: {e}")

    def _build_acids_tab(self, parent):
        tk.Label(parent, text="pH  =  −log[H⁺]", font=FONT_BOLD,
                 bg=BG_CARD, fg=ACCENT_CREAM, pady=12).pack()
        self.acid_h   = tk.StringVar(value="0.001")
        self.acid_res = tk.StringVar(value="—")
        self._law_row(parent, "[H⁺] concentration:", self.acid_h, "mol/L")
        self._result_label(parent, self.acid_res)
        styled_button(parent, "Calculate pH", self._calc_acids).pack(pady=6)

        divider(parent).pack(fill="x", padx=20, pady=10)
        tk.Label(parent, text="Henderson-Hasselbalch:  pH = pKa + log([A⁻]/[HA])",
                 font=FONT_BOLD, bg=BG_CARD, fg=ACCENT_CREAM, pady=4).pack()
        self.hh_pka = tk.StringVar(value="4.75")
        self.hh_A   = tk.StringVar(value="0.1")
        self.hh_HA  = tk.StringVar(value="0.1")
        self.hh_res = tk.StringVar(value="—")
        self._law_row(parent, "pKa:", self.hh_pka)
        self._law_row(parent, "[A⁻]:", self.hh_A, "mol/L")
        self._law_row(parent, "[HA]:", self.hh_HA, "mol/L")
        self._result_label(parent, self.hh_res)
        styled_button(parent, "Calculate Buffer pH", self._calc_hh).pack(pady=4)

    def _calc_acids(self):
        try:
            h = float(self.acid_h.get())
            r = AcidsBases().ph(h)
            self.acid_res.set(f"pH = {r:.4f}")
        except Exception as e:
            self.acid_res.set(f"Error: {e}")

    def _calc_hh(self):
        try:
            pKa = float(self.hh_pka.get())
            A   = float(self.hh_A.get())
            HA  = float(self.hh_HA.get())
            r   = AcidsBases().henderson(pKa, A, HA)
            self.hh_res.set(f"pH = {r:.4f}")
        except Exception as e:
            self.hh_res.set(f"Error: {e}")

    def _build_gas_tab(self, parent):
        tk.Label(parent, text="Ideal Gas:  PV = nRT   (R = 0.0821 L·atm/mol·K)",
                 font=FONT_BOLD, bg=BG_CARD, fg=ACCENT_CREAM, pady=12).pack()
        self.gas_n   = tk.StringVar(value="1")
        self.gas_T   = tk.StringVar(value="300")
        self.gas_V   = tk.StringVar(value="22.4")
        self.gas_res = tk.StringVar(value="—")
        self._law_row(parent, "n (mol):", self.gas_n, "mol")
        self._law_row(parent, "T (Kelvin):", self.gas_T, "K")
        self._law_row(parent, "V (Liters):", self.gas_V, "L")
        self._result_label(parent, self.gas_res)
        styled_button(parent, "Calculate Pressure", self._calc_gas).pack(pady=6)

    def _calc_gas(self):
        try:
            n = float(self.gas_n.get())
            T = float(self.gas_T.get())
            V = float(self.gas_V.get())
            r = GasLaws().ideal_gas_pressure(n, T, V)
            self.gas_res.set(f"P = {r:.4f} atm")
        except Exception as e:
            self.gas_res.set(f"Error: {e}")

    def _build_thermo_tab(self, parent):
        tk.Label(parent, text="Temperature Conversion", font=FONT_BOLD,
                 bg=BG_CARD, fg=ACCENT_CREAM, pady=12).pack()
        self.thermo_c   = tk.StringVar(value="25")
        self.thermo_k   = tk.StringVar(value="298")
        self.thermo_res = tk.StringVar(value="—")
        self._law_row(parent, "Celsius (°C):", self.thermo_c, "°C")
        styled_button(parent, "°C  →  K", lambda: self._conv_thermo("ctok")).pack(pady=4)
        divider(parent).pack(fill="x", padx=20, pady=6)
        self._law_row(parent, "Kelvin (K):", self.thermo_k, "K")
        styled_button(parent, "K  →  °C", lambda: self._conv_thermo("ktoc")).pack(pady=4)
        self._result_label(parent, self.thermo_res)

    def _conv_thermo(self, direction):
        try:
            t = Thermo()
            if direction == "ctok":
                r = t.temp_c_to_k(float(self.thermo_c.get()))
                self.thermo_res.set(f"{r:.2f} K")
            else:
                r = t.temp_k_to_c(float(self.thermo_k.get()))
                self.thermo_res.set(f"{r:.2f} °C")
        except Exception as e:
            self.thermo_res.set(f"Error: {e}")


    # ════════════════════════════════════════
    #  SECTION 4 — BONDS
    # ════════════════════════════════════════
    def _build_bonds_section(self):
        outer = tk.Frame(self.content, bg=BG_DARK)

        tk.Label(outer, text="CHEMICAL BONDS", font=FONT_HEAD,
                 bg=BG_DARK, fg=ACCENT_AMBER, pady=14).pack()
        divider(outer).pack(fill="x", padx=20, pady=(0,10))

        # ── Bond Calculator ──
        calc_card = card_frame(outer)
        calc_card.pack(fill="x", padx=24, pady=(0,12))

        tk.Label(calc_card, text="Bond Type Calculator  —  Enter Electronegativities",
                 font=FONT_BOLD, bg=BG_CARD, fg=ACCENT_CREAM, pady=10).pack()

        row = tk.Frame(calc_card, bg=BG_CARD)
        row.pack(pady=6)
        self.bond_el1  = tk.StringVar(value="Na")
        self.bond_el2  = tk.StringVar(value="Cl")
        self.bond_en1  = tk.StringVar(value="0.9")
        self.bond_en2  = tk.StringVar(value="3.0")

        for label, var, w in [("Element 1:", self.bond_el1, 6),
                               ("EN1:", self.bond_en1, 6),
                               ("Element 2:", self.bond_el2, 6),
                               ("EN2:", self.bond_en2, 6)]:
            tk.Label(row, text=label, font=FONT_BODY, bg=BG_CARD, fg=TEXT_DIM).pack(side="left", padx=(10,2))
            tk.Entry(row, textvariable=var, font=FONT_BODY, width=w,
                     bg=BG_PANEL, fg=ACCENT_CREAM, insertbackground=ACCENT_GOLD,
                     relief="flat", bd=4).pack(side="left", padx=(0,6))

        styled_button(calc_card, "Analyze Bond", self._analyze_bond, width=16).pack(pady=8)

        self.bond_result_var = tk.StringVar(value="—")
        res_row = tk.Frame(calc_card, bg=BG_PANEL, pady=6)
        res_row.pack(fill="x", padx=20, pady=(0,10))
        tk.Label(res_row, text="Result:", font=FONT_BOLD, bg=BG_PANEL, fg=TEXT_DIM).pack(side="left", padx=10)
        tk.Label(res_row, textvariable=self.bond_result_var,
                 font=("Courier New", 13, "bold"), bg=BG_PANEL, fg=ACCENT_GOLD).pack(side="left")

        divider(outer).pack(fill="x", padx=20, pady=6)
        tk.Label(outer, text="BOND TYPES — REFERENCE",
                 font=FONT_BODY, bg=BG_DARK, fg=TEXT_DIM).pack(pady=(4,6))

        # ── Bond Cards ──
        scroll_outer = tk.Frame(outer, bg=BG_DARK)
        scroll_outer.pack(fill="both", expand=True, padx=20, pady=(0,10))

        canvas  = tk.Canvas(scroll_outer, bg=BG_DARK, highlightthickness=0)
        scrollbar = tk.Scrollbar(scroll_outer, orient="vertical", command=canvas.yview)
        scroll_frame = tk.Frame(canvas, bg=BG_DARK)

        scroll_frame.bind("<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw", tags="frame")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.bind("<Configure>",
            lambda e: canvas.itemconfig("frame", width=e.width))

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        for bond_name, description, en_range, examples, properties in BONDS_DATA:
            bcard = card_frame(scroll_frame)
            bcard.pack(fill="x", pady=5, padx=4)

            header_row = tk.Frame(bcard, bg=BG_CARD)
            header_row.pack(fill="x", padx=14, pady=(10,4))
            tk.Label(header_row, text=bond_name, font=FONT_HEAD,
                     bg=BG_CARD, fg=ACCENT_AMBER).pack(side="left")
            tk.Label(header_row, text=f"  [{en_range}]",
                     font=FONT_LABEL, bg=BG_CARD, fg=ACCENT_GOLD).pack(side="left")

            tk.Label(bcard, text=description, font=FONT_BODY,
                     bg=BG_CARD, fg=ACCENT_CREAM, wraplength=1300, justify="left",
                     padx=14).pack(anchor="w")
            tk.Label(bcard, text=f"Examples:  {examples}",
                     font=("Courier New", 16, "italic"), bg=BG_CARD, fg=TEXT_DIM, padx=14).pack(anchor="w")
            tk.Label(bcard, text=f"Properties:  {properties}",
                     font=FONT_LABEL, bg=BG_CARD, fg=TEXT_DIM, wraplength=1300, justify="left",
                     padx=14).pack(anchor="w", pady=(0,10))

        return outer

    def _analyze_bond(self):
        try:
            el1 = self.bond_el1.get().strip()
            el2 = self.bond_el2.get().strip()
            en1 = float(self.bond_en1.get())
            en2 = float(self.bond_en2.get())
            b   = Bond(el1, el2, en1, en2)
            self.bond_result_var.set(
                f"{el1}-{el2}  |  ΔEN = {b.delta_en():.2f}  |  {b.bond_type()}  |  {b.polarity()}"
            )
        except Exception as e:
            self.bond_result_var.set(f"Error: {e}")


    # ════════════════════════════════════════
    #  SECTION 5 — QUIZ
    # ════════════════════════════════════════
    def _build_quiz_section(self):
        outer = tk.Frame(self.content, bg=BG_DARK)

        tk.Label(outer, text="CHEMISTRY QUIZ", font=FONT_HEAD,
                 bg=BG_DARK, fg=ACCENT_AMBER, pady=14).pack()
        divider(outer).pack(fill="x", padx=20, pady=(0,10))

        # Score / status bar
        score_bar = tk.Frame(outer, bg=BG_PANEL)
        score_bar.pack(fill="x", padx=20)

        self.quiz_score_var   = tk.StringVar(value="Score: 0 / 0")
        self.quiz_status_var  = tk.StringVar(value="Press  Start Quiz  to begin")
        tk.Label(score_bar, textvariable=self.quiz_score_var,
                 font=FONT_BOLD, bg=BG_PANEL, fg=ACCENT_GOLD, pady=8, padx=14).pack(side="left")
        tk.Label(score_bar, textvariable=self.quiz_status_var,
                 font=FONT_BODY, bg=BG_PANEL, fg=TEXT_DIM, pady=8).pack(side="left", padx=20)

        # Question card
        q_card = card_frame(outer)
        q_card.pack(fill="x", padx=20, pady=12)

        self.quiz_progress_var = tk.StringVar(value="Question 0 of 10")
        tk.Label(q_card, textvariable=self.quiz_progress_var,
                 font=FONT_LABEL, bg=BG_CARD, fg=TEXT_DIM, pady=6).pack(anchor="e", padx=14)

        self.quiz_q_var = tk.StringVar(value="Press Start to load the first question.")
        tk.Label(q_card, textvariable=self.quiz_q_var,
                 font=("Georgia", 14, "bold"), bg=BG_CARD, fg=ACCENT_CREAM,
                 wraplength=820, justify="left",
                 padx=20, pady=14).pack(anchor="w")

        # Options
        self.quiz_opt_vars = [tk.IntVar(value=-1)] * 4
        self.quiz_opt_btns = []
        opts_frame = tk.Frame(q_card, bg=BG_CARD)
        opts_frame.pack(fill="x", padx=20, pady=(0,12))
        self.quiz_selected = tk.IntVar(value=-1)

        for i in range(4):
            btn = tk.Button(
                opts_frame, text=f"  Option {i+1}",
                font=FONT_BODY, bg=BG_PANEL, fg=ACCENT_CREAM,
                activebackground=HOVER_BG, activeforeground=ACCENT_AMBER,
                relief="flat", bd=0, cursor="hand2",
                anchor="w", padx=16, pady=10,
                command=lambda idx=i: self._select_option(idx)
            )
            btn.pack(fill="x", pady=2)
            self.quiz_opt_btns.append(btn)

        # Explanation box
        self.quiz_exp_var = tk.StringVar(value="")
        self.quiz_exp_lbl = tk.Label(q_card, textvariable=self.quiz_exp_var,
                 font=("Courier New", 10, "italic"), bg=BG_CARD, fg=ACCENT_AMBER,
                 wraplength=820, justify="left",
                 padx=20)
        self.quiz_exp_lbl.pack(anchor="w", pady=(0, 10))

        # Buttons row
        btn_row = tk.Frame(outer, bg=BG_DARK)
        btn_row.pack(pady=10)

        self.quiz_start_btn  = styled_button(btn_row, "Start Quiz",   self._start_quiz,  width=14, bg=ACCENT_GREEN, fg=TEXT_BRIGHT, active_bg="#5EA870")
        self.quiz_start_btn.pack(side="left", padx=10)

        self.quiz_submit_btn = styled_button(btn_row, "Submit Answer", self._submit_answer, width=16)
        self.quiz_submit_btn.pack(side="left", padx=10)
        self.quiz_submit_btn.config(state="disabled")

        self.quiz_next_btn   = styled_button(btn_row, "Next Question", self._next_question, width=16, bg=BG_CARD, fg=ACCENT_AMBER, active_bg=HOVER_BG)
        self.quiz_next_btn.pack(side="left", padx=10)
        self.quiz_next_btn.config(state="disabled")

        styled_button(btn_row, "Restart", self._restart_quiz, width=10, bg=BG_PANEL, fg=TEXT_DIM, active_bg=HOVER_BG).pack(side="left", padx=10)

        return outer

    def _start_quiz(self):
        self.quiz_index  = 0
        self.quiz_score  = 0
        self.quiz_active = True
        self.quiz_score_var.set("Score: 0 / 0")
        self.quiz_status_var.set("Good luck!")
        self._load_question()
        self.quiz_submit_btn.config(state="normal")
        self.quiz_start_btn.config(state="disabled")

    def _load_question(self):
        if self.quiz_index >= len(QUIZ_QUESTIONS):
            self._end_quiz()
            return
        q = QUIZ_QUESTIONS[self.quiz_index]
        self.quiz_progress_var.set(f"Question {self.quiz_index+1} of {len(QUIZ_QUESTIONS)}")
        self.quiz_q_var.set(q["q"])
        self.quiz_exp_var.set("")
        self.quiz_selected.set(-1)
        for i, btn in enumerate(self.quiz_opt_btns):
            btn.config(text=f"  {chr(65+i)}.  {q['opts'][i]}",
                       bg=BG_PANEL, fg=ACCENT_CREAM, state="normal")
        self.quiz_next_btn.config(state="disabled")

    def _select_option(self, idx):
        self.quiz_selected.set(idx)
        for i, btn in enumerate(self.quiz_opt_btns):
            btn.config(bg=HOVER_BG if i == idx else BG_PANEL,
                       fg=ACCENT_AMBER if i == idx else ACCENT_CREAM)

    def _submit_answer(self):
        sel = self.quiz_selected.get()
        if sel == -1:
            messagebox.showinfo("No Selection", "Please select an answer first.")
            return
        q = QUIZ_QUESTIONS[self.quiz_index]
        correct = q["ans"]
        answered = self.quiz_index + 1
        if sel == correct:
            self.quiz_score += 1
            self.quiz_status_var.set("Correct!")
            self.quiz_opt_btns[sel].config(bg=ACCENT_GREEN, fg=TEXT_BRIGHT)
        else:
            self.quiz_status_var.set("Incorrect.")
            self.quiz_opt_btns[sel].config(bg=ACCENT_RED, fg=TEXT_BRIGHT)
            self.quiz_opt_btns[correct].config(bg=ACCENT_GREEN, fg=TEXT_BRIGHT)
        self.quiz_exp_var.set(f"Explanation:  {q['exp']}")
        self.quiz_score_var.set(f"Score: {self.quiz_score} / {answered}")
        for btn in self.quiz_opt_btns: btn.config(state="disabled")
        self.quiz_submit_btn.config(state="disabled")
        self.quiz_next_btn.config(state="normal")

    def _next_question(self):
        self.quiz_index += 1
        if self.quiz_index >= len(QUIZ_QUESTIONS):
            self._end_quiz()
        else:
            self._load_question()
            self.quiz_submit_btn.config(state="normal")
        self.quiz_next_btn.config(state="disabled")

    def _end_quiz(self):
        total = len(QUIZ_QUESTIONS)
        pct   = (self.quiz_score / total) * 100
        self.quiz_active = False
        self.quiz_q_var.set(f"Quiz Complete!  You scored {self.quiz_score} out of {total}  ({pct:.0f}%)")
        self.quiz_exp_var.set("")
        for btn in self.quiz_opt_btns: btn.config(text="", state="disabled", bg=BG_PANEL)
        self.quiz_submit_btn.config(state="disabled")
        self.quiz_next_btn.config(state="disabled")
        self.quiz_start_btn.config(state="normal")
        self.quiz_status_var.set(
            "Excellent!" if pct >= 80 else "Good effort!" if pct >= 50 else "Keep studying!")

    def _restart_quiz(self):
        self.quiz_index  = 0
        self.quiz_score  = 0
        self.quiz_active = False
        self.quiz_score_var.set("Score: 0 / 0")
        self.quiz_status_var.set("Press  Start Quiz  to begin")
        self.quiz_progress_var.set("Question 0 of 10")
        self.quiz_q_var.set("Press Start to load the first question.")
        self.quiz_exp_var.set("")
        for btn in self.quiz_opt_btns:
            btn.config(text="", bg=BG_PANEL, fg=ACCENT_CREAM, state="disabled")
        self.quiz_submit_btn.config(state="disabled")
        self.quiz_next_btn.config(state="disabled")
        self.quiz_start_btn.config(state="normal")


# ══════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════

if __name__ == "__main__":
    root = tk.Tk()
    app  = ChemLabDashboard(root)
    root.mainloop()

