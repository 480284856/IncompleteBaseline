# Generalization report draft

`main.tex` contains the Introduction, Related Work, Environment Design, Actor DQN, and Experiment sections. The report diagrams are stored in `img/`. Compile the report with a LaTeX installation:

```sh
pdflatex main.tex
pdflatex main.tex
```

The classification-versus-regression results are in `../dev/maze4x4/cls_fatsterthan_reg/probe_results_greedy/metrics.csv` and `../dev/maze4x4/cls_fatsterthan_reg/probe_results_greedy/maze_success.csv`. The earlier figure references `img/env_design-constraint.png` and `img/maze-path_finding.png` still need their image files.
