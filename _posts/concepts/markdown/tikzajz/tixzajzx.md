---
date: 02-13-2024
type: notes
summary: 
tags: 
aliases: 
status: 
subject:
---
# [[tixzajzx]]

```tikz 
\begin{document} 
	\begin{tikzpicture}[domain=0:4] 
		\draw[very thin,color=gray] (-0.1,-1.1) grid (3.9,3.9); 
		\draw[->] (-0.2,0) -- (4.2,0) node[right] {$x$}; 
		\draw[->] (0,-1.2) -- (0,4.2) node[above] {$f(x)$}; 
		\draw[color=red] plot (\x,\x) node[right] {$f(x) =x$}; 
		\draw[color=blue] plot (\x,{sin(\x r)}) node[right] {$f(x) = \sin x$}; 
		\draw[color=orange] plot (\x,{0.05*exp(\x)}) node[right] {$f(x) = \frac{1}{20} \mathrm e^x$}; 
	\end{tikzpicture} 
\end{document} 
```


```tikz
\usepackage{pgfplots}
\pgfplotsset{compat=1.16}

\begin{document}

\begin{tikzpicture}
\begin{axis}[colormap/viridis]
\addplot3[
	surf,
	samples=18,
	domain=-3:3
]
{exp(-x^2-y^2)*x};
\end{axis}
\end{tikzpicture}

\end{document}
```

```tikz
\usepackage{circuitikz}
\begin{document}

\begin{circuitikz}[american, voltage shift=0.5]
\draw (0,0)
to[isource, l=$I_0$, v=$V_0$] (0,3)
to[short, -*, i=$I_0$] (2,3)
to[R=$R_1$, i>_=$i_1$] (2,0) -- (0,0);
\draw (2,3) -- (4,3)
to[R=$R_2$, i>_=$i_2$]
(4,0) to[short, -*] (2,0);
\end{circuitikz}

\end{document}
```
