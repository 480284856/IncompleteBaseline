"""Render the three slide flowcharts as high-resolution PNG images."""

from pathlib import Path
from math import atan2, cos, sin, pi
from PIL import Image, ImageDraw, ImageFont


OUT = Path(__file__).parent / "assets"
W, H = 2400, 970
INK = "#14283f"
LINE = "#202a32"
MUTED = "#52677b"
TEAL = "#087f8c"
NET = "#edf4fa"
VALUE = "#fff4d9"
LABEL = "#e8f2f2"
LOSS = "#fceceb"
WHITE = "#ffffff"

FONT_REG = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
FONT_UNI = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"


def font(size, bold=False, unicode=False):
    return ImageFont.truetype(FONT_UNI if unicode else FONT_BOLD if bold else FONT_REG, size)


def canvas():
    im = Image.new("RGB", (W, H), WHITE)
    return im, ImageDraw.Draw(im)


def txt(d, xy, value, size=39, bold=False, color=INK, anchor="mm"):
    d.multiline_text(xy, value, font=font(size, bold, True), fill=color,
                     anchor=anchor, align="center", spacing=7)


def box(d, xy, value, fill=WHITE, outline=LINE, size=38, bold=False, width=4):
    d.rectangle(xy, fill=fill, outline=outline, width=width)
    x0, y0, x1, y1 = xy
    txt(d, ((x0 + x1) / 2, (y0 + y1) / 2), value, size, bold)


def arrow(d, points, color=LINE, width=5, head=17):
    d.line(points, fill=color, width=width, joint="curve")
    x0, y0 = points[-2]
    x1, y1 = points[-1]
    ang = atan2(y1 - y0, x1 - x0)
    wings = [
        (x1 - head * cos(ang - pi / 6), y1 - head * sin(ang - pi / 6)),
        (x1 - head * cos(ang + pi / 6), y1 - head * sin(ang + pi / 6)),
    ]
    d.polygon([(x1, y1), *wings], fill=color)


def dot(d, x, y):
    d.ellipse((x - 8, y - 8, x + 8, y + 8), fill=LINE)


def vector(d, xy, title, selected=None, size=35):
    x0, y0, x1, y1 = xy
    txt(d, ((x0 + x1) / 2, y0 - 27), title, 33, color=INK)
    cell = (x1 - x0) / 4
    for i in range(4):
        color = "#e0bd63" if i == selected else VALUE
        d.rectangle((x0 + i * cell, y0, x0 + (i + 1) * cell, y1),
                    fill=color, outline=LINE, width=4)
        txt(d, (x0 + (i + .5) * cell, (y0 + y1) / 2), f"q{i}", size)


def flow_header(d, text):
    txt(d, (70, 42), text, 29, color=MUTED, anchor="lm")


def dqn():
    im, d = canvas()
    flow_header(d, "REPLAY TRANSITION   (s, a, r, s', done)")
    txt(d, (77, 116), "TARGET BRANCH", 29, True, TEAL, "lm")
    txt(d, (77, 603), "ONLINE BRANCH", 29, True, TEAL, "lm")

    box(d, (80, 194, 160, 270), "s'", LABEL, size=43, bold=True)
    box(d, (260, 160, 595, 300), "Target Q\nnetwork", NET, size=43, bold=True)
    vector(d, (700, 197, 1080, 269), "Q_T(s', all actions)")
    box(d, (1200, 194, 1370, 270), "max", LABEL, size=42, bold=True)
    box(d, (1480, 156, 1900, 310),
        "TD target\ny = r + γ(1 − done)\n· max Q_T(s', ·)", LABEL, size=31, bold=True)
    box(d, (1510, 350, 1840, 420), "r, done", WHITE, size=35)
    box(d, (2060, 407, 2340, 570), "Regression\nloss (MSE)", LOSS, size=40, bold=True)

    box(d, (80, 672, 160, 748), "s", LABEL, size=45, bold=True)
    box(d, (260, 640, 595, 780), "Online Q\nnetwork", NET, size=43, bold=True)
    vector(d, (700, 677, 1080, 749), "Q(s, all actions)", selected=2)
    box(d, (1200, 672, 1370, 752), "select a", LABEL, size=34, bold=True)
    box(d, (1480, 664, 1900, 760), "Q(s, a)", VALUE, size=42, bold=True)
    box(d, (1228, 842, 1342, 912), "a", WHITE, size=42, bold=True)

    for p in [((160, 232), (260, 232)), ((595, 232), (700, 232)),
              ((1080, 232), (1200, 232)), ((1370, 232), (1480, 232)),
              ((160, 710), (260, 710)), ((595, 710), (700, 710)),
              ((1080, 710), (1200, 710)), ((1370, 712), (1480, 712))]:
        arrow(d, list(p))
    arrow(d, [(1675, 350), (1675, 310)])
    arrow(d, [(1285, 842), (1285, 752)])
    arrow(d, [(1900, 232), (1980, 232), (1980, 450), (2060, 450)])
    arrow(d, [(1900, 712), (1980, 712), (1980, 525), (2060, 525)])
    im.save(OUT / "model-dqn.png", optimize=True)


def actordqn():
    im, d = canvas()
    flow_header(d, "REPLAY TRANSITION   (s, a, r, s', done)")
    txt(d, (76, 110), "TARGET", 28, True, TEAL, "lm")
    txt(d, (76, 394), "ONLINE Q", 28, True, TEAL, "lm")
    txt(d, (76, 770), "ACTOR", 28, True, TEAL, "lm")

    box(d, (75, 160, 150, 230), "s'", LABEL, size=40, bold=True)
    box(d, (245, 135, 550, 255), "Target Q\nnetwork", NET, size=39, bold=True)
    vector(d, (650, 171, 960, 233), "Q_T(s', ·)", size=30)
    box(d, (1060, 159, 1215, 235), "max", LABEL, size=39, bold=True)
    box(d, (1320, 130, 1730, 260),
        "TD target\ny = r + γ(1 − done)\n· max Q_T(s', ·)", LABEL, size=28, bold=True)
    box(d, (1455, 294, 1595, 351), "r, done", WHITE, size=27)

    box(d, (75, 449, 150, 519), "s", LABEL, size=42, bold=True)
    box(d, (245, 420, 550, 550), "Online Q\nnetwork", NET, size=39, bold=True)
    vector(d, (650, 457, 960, 519), "Q(s, ·)", size=30)
    box(d, (1060, 448, 1215, 528), "select a", LABEL, size=31, bold=True)
    box(d, (1320, 439, 1730, 537), "Q(s, a)", VALUE, size=39, bold=True)
    box(d, (1105, 545, 1170, 595), "a", WHITE, size=31, bold=True)
    box(d, (1975, 346, 2340, 509), "Regression\nloss (MSE)", LOSS, size=39, bold=True)

    box(d, (1070, 615, 1260, 683), "argmax", LABEL, size=34, bold=True)
    box(d, (1360, 615, 1730, 683), "Label a*  (detached)", VALUE, size=34, bold=True)
    box(d, (245, 781, 550, 901), "Actor Q\nnetwork", NET, size=39, bold=True)
    vector(d, (650, 818, 960, 880), "Actor scores  z(s, ·)", size=30)
    box(d, (1060, 813, 1215, 885), "softmax", LABEL, size=30, bold=True)
    box(d, (1320, 800, 1730, 896), "p(a | s)", VALUE, size=39, bold=True)
    box(d, (1975, 734, 2340, 903), "Classification\nloss (cross-entropy)", LOSS, size=37, bold=True)

    for p in [((150, 195), (245, 195)), ((550, 195), (650, 202)),
              ((960, 202), (1060, 197)), ((1215, 197), (1320, 197)),
              ((150, 484), (245, 484)), ((550, 484), (650, 488)),
              ((960, 488), (1060, 488)), ((1215, 488), (1320, 488)),
              ((550, 841), (650, 849)), ((960, 849), (1060, 849)),
              ((1215, 849), (1320, 849))]:
        arrow(d, list(p))
    arrow(d, [(1525, 294), (1525, 260)])
    arrow(d, [(1137, 545), (1137, 528)])
    arrow(d, [(150, 484), (190, 484), (190, 841), (245, 841)])
    dot(d, 190, 484)
    arrow(d, [(805, 519), (805, 650), (1070, 650)])
    arrow(d, [(1260, 650), (1360, 650)])
    arrow(d, [(1730, 197), (1860, 197), (1860, 393), (1975, 393)])
    arrow(d, [(1730, 488), (1860, 488), (1860, 464), (1975, 464)])
    arrow(d, [(1730, 649), (1850, 649), (1850, 765), (1975, 765)])
    arrow(d, [(1730, 849), (1975, 849)])
    im.save(OUT / "model-actordqn.png", optimize=True)


def maze(d, x, y, cell=49):
    walls = {(1, 0), (2, 0), (3, 0), (0, 2), (2, 2), (3, 2)}
    path = {(0, 0), (0, 1), (1, 1), (2, 1), (1, 2), (1, 3), (2, 3), (3, 3)}
    for row in range(4):
        for col in range(4):
            xy = (x + col * cell, y + row * cell,
                  x + (col + 1) * cell, y + (row + 1) * cell)
            fill = INK if (col, row) in walls else "#d9eeee" if (col, row) in path else WHITE
            d.rectangle(xy, fill=fill, outline=LINE, width=3)
    txt(d, (x + cell / 2, y + cell / 2), "S", 31, True)
    txt(d, (x + 3.5 * cell, y + 3.5 * cell), "G", 31, True)


def justification():
    im, d = canvas()
    txt(d, (157, 332), "Maze state  s", 31, True)
    maze(d, 59, 371, 49)
    txt(d, (870, 55), "LABEL I", 31, True, TEAL)
    txt(d, (1660, 55), "LABEL II", 31, True, TEAL)
    box(d, (420, 99, 770, 228), "Frozen target\nQ network", NET, size=40, bold=True)
    vector(d, (905, 130, 1225, 206), "Teacher Q(s, ·)", size=30)
    box(d, (1340, 135, 1520, 205), "argmax", LABEL, size=33, bold=True)
    box(d, (1635, 117, 1925, 223), "Action label  a*", VALUE, size=34, bold=True)
    box(d, (420, 425, 770, 554), "Q network I\n(regression student)", NET, size=34, bold=True)
    vector(d, (905, 457, 1225, 533), "Predicted Q(s, ·)", size=30)
    box(d, (1890, 427, 2305, 559), "Regression\nMSE loss", LOSS, size=39, bold=True)
    box(d, (420, 748, 770, 877), "Q network II\n(classification student)", NET, size=33, bold=True)
    vector(d, (905, 781, 1225, 857), "Action scores / logits", size=29)
    box(d, (1890, 749, 2305, 881), "Classification\ncross-entropy loss", LOSS, size=36, bold=True)

    # The same maze observation branches into teacher and both students.
    d.line([(255, 469), (330, 469)], fill=LINE, width=5)
    d.line([(330, 164), (330, 813)], fill=LINE, width=5)
    for yy, xx in [(164, 420), (489, 420), (813, 420)]:
        dot(d, 330, yy)
        arrow(d, [(330, yy), (xx, yy)])
    arrow(d, [(770, 164), (905, 168)])
    arrow(d, [(1225, 168), (1340, 170)])
    arrow(d, [(1520, 170), (1635, 170)])
    arrow(d, [(770, 489), (905, 495)])
    arrow(d, [(1225, 495), (1890, 495)])
    arrow(d, [(770, 813), (905, 819)])
    arrow(d, [(1225, 819), (1890, 819)])
    arrow(d, [(1065, 206), (1065, 341), (2070, 341), (2070, 427)])
    arrow(d, [(1925, 170), (2350, 170), (2350, 814), (2305, 814)])
    im.save(OUT / "justification-flow.png", optimize=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    dqn()
    actordqn()
    justification()
