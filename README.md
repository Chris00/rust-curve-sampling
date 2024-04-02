Curve Sampling
==============

This module provide a collection of routines to perform adaptive
sampling of curves as well as manipulating those samplings.


Usage
-----

Add this to your `Cargo.toml`:

```
[dependencies]
curve-sampling = "0.5"
```

See the [documentation](https://docs.rs/curve-sampling/).

Example
-------

To sample the function x ↦ x sin(1/x) on the interval [-0.4, 0.4] with
227 function evaluations, simply do

```rust
use curve_sampling::Sampling;
let s = Sampling::fun(|x| x * (1. / x).sin(), -0.4, 0.4).n(227).build();
```

You can save the resulting sampling to [TikZ][] with

```rust
s.latex().write(&mut File::create("graph.tex")?)?;
```

or to a data file (whose format is compatible with [Gnuplot][]) with

```rust
s.write(&mut File::create("graph.dat")?)?;
```

Asking [Gnuplot][] to draw the resulting curve (with `plot 'graph.dat'`) 
yields:

![x sin(1/x)](https://user-images.githubusercontent.com/1255665/186882845-81dcbe02-808b-40d7-9fad-a838e326ce78.png)

P.S. The number of evaluations (227) was chosen to match a depth 5
recursion for
[Mathematica](https://github.com/Chris00/rust-curve-sampling/wiki).


[TikZ]: https://tikz.dev/
[Gnuplot]: http://www.gnuplot.info/
