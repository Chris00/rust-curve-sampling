use std::{error::Error,
          fs::File,
          f64::NAN,
          io::Write,
          process::Command};
use curve_sampling::Sampling;

fn main() -> Result<(), Box<dyn Error>> {
    let s = Sampling::from_iter([[0., 0.], [1., 1.], [NAN, NAN],
                                 [1., 1.], [3., -1.]]);
    s.latex().arrow_pos(0.3).write(&mut File::create("/tmp/arrow0.tex")?)?;

    let s = Sampling::uniform(|x| -0.7 * (x - 1.).powi(2), 0., 2.5).build();
    s.latex().arrow_pos(0.6).write(&mut File::create("/tmp/arrow1.tex")?)?;


    File::create("/tmp/arrows.tex")?.write_all(
        r"\documentclass{article}
\usepackage{tikz}
\begin{document}
\begin{tikzpicture}[x=3cm, y=3cm]
  \draw[->] (-1.2, 0) -- (3.2, 0);
  \draw[->] (0, -1.2) -- (0, 1.7);
  \foreach \x in {-1, -0.5, 0.5, 1, 1.5,..., 3}{
    \draw (\x, 3pt) -- (\x, -3pt) node[below]{$\scriptstyle \x$};
  }
  \foreach \y in {-1, -0.5, 0.5, 1, 1.5}{
    \draw (3pt, \y) -- (-3pt, \y) node[left]{$\scriptstyle \y$};
  }
  \begin{scope}[color=blue, line width=1pt]
    \input{arrow0.tex}
  \end{scope}
  \begin{scope}[color=orange, line width=1pt]
    \input{arrow1.tex}
  \end{scope}
\end{tikzpicture}
\end{document}".as_bytes())?;

    std::env::set_current_dir("/tmp")?;
    Command::new("pdflatex")
        .args(["-interaction=batchmode", "arrows.tex"]).output()?;
    Ok(())
}
