# Assignment-5-Quicksort-Algorithm-Implementation-Analysis-and-Randomization

- Deterministic Quicksort using fixed pivot rule
- Randomized Quicksort using random pivot
- Empirical timing on Random/Sorted/Reverse-Sorted inputs

How to run the script 
  python quicksort_assignment5.py

- Results


Running empirical analysis...
Sizes=[100, 1000, 5000, 10000, 20000] | trials=5 | seed=42 | outdir='results'

[Random        ] n=   100  det=0.000061s  rand=0.000093s
[Random        ] n=  1000  det=0.001012s  rand=0.001498s
[Random        ] n=  5000  det=0.006115s  rand=0.007678s
[Random        ] n= 10000  det=0.013364s  rand=0.015944s
[Random        ] n= 20000  det=0.028227s  rand=0.036865s
[Sorted        ] n=   100  det=0.000086s  rand=0.000171s
[Sorted        ] n=  1000  det=0.001341s  rand=0.002322s
[Sorted        ] n=  5000  det=0.008297s  rand=0.014005s
[Sorted        ] n= 10000  det=0.024538s  rand=0.031542s
[Sorted        ] n= 20000  det=0.058879s  rand=0.031794s
[Reverse Sorted] n=   100  det=0.000046s  rand=0.000087s
[Reverse Sorted] n=  1000  det=0.000748s  rand=0.001220s
[Reverse Sorted] n=  5000  det=0.004633s  rand=0.007091s
[Reverse Sorted] n= 10000  det=0.010023s  rand=0.014909s
[Reverse Sorted] n= 20000  det=0.021652s  rand=0.036373s
