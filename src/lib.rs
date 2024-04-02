//! This module provide a collection of routines to perform adaptive
//! sampling of curves as well as manipulating those samplings.
//!
//! As of now, the following is available:
//! - uniform sampling of the graph of functions ℝ → ℝ
//!   (see [`Sampling::uniform`]);
//! - adaptive sampling of the graph functions ℝ → ℝ
//!   (see [`Sampling::fun`]);
//! - adaptive sampling of the image functions ℝ → ℝ²
//!   (see [`Sampling::param`]).
//!
//! Samplings can be saved as a list of coordinates x-y, one point per
//! line with blank lines to indicate that the path is interrupted,
//! with [`Sampling::write`].  They can also be saved in a format
//! usable from the [LaTeX][] package [TikZ][] using
//! [`Sampling::latex`].  This format allows to add arrows to the
//! curves to indicate their orientation.
//!
//! [LaTeX]: https://www.latex-project.org/
//! [TikZ]: https://tikz.dev/

use std::{fmt::{self, Display, Formatter},
          cell::Cell,
          io::{self, Write},
          iter::Iterator,
          ops::ControlFlow};
use rgb::*;

// mod fibonacci_heap;
// use fibonacci_heap as pq;
// mod priority_queue;
// use priority_queue as pq;
mod approx_priority_queue;
use approx_priority_queue as pq;

mod list;
use list::List;

////////////////////////////////////////////////////////////////////////
//
// Bounding box

/// A two dimensional rectangle \[`xmin`, `xmax`\] × \[`ymin`, `ymax`\].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundingBox {
    pub xmin: f64,
    pub xmax: f64,
    pub ymin: f64,
    pub ymax: f64,
}

impl BoundingBox {
    #[inline]
    pub(crate) fn empty() -> Self {
        Self { xmin: f64::INFINITY,  xmax: f64::NEG_INFINITY,
               ymin: f64::INFINITY,  ymax: f64::NEG_INFINITY }
    }

    /// Return `true` if the bounding box has a non-empty interior.
    #[inline]
    pub fn is_empty(&self) -> bool {
        !(self.xmin < self.xmax && self.ymin < self.ymax) // NAN ⟹ empty
    }

    /// Return `true` if the point `p` belongs to `bb` (possibly on
    /// the boundary).
    #[inline]
    fn contains(&self, p: &Point) -> bool {
        let [x, y] = p.xy;
        self.xmin <= x && x <= self.xmax && self.ymin <= y && y <= self.ymax
    }

    /// Return the smaller bounding-box containing both `self` and
    /// `other`.
    #[inline]
    pub fn hull(&self, other: &Self) -> Self {
        BoundingBox { xmin: self.xmin.min(other.xmin),
                      xmax: self.xmax.max(other.xmax),
                      ymin: self.ymin.min(other.ymin),
                      ymax: self.ymax.max(other.ymax) }
    }
}


////////////////////////////////////////////////////////////////////////
//
// Sampling datastructure

/// A 2D point with coordinates (`x`, `y`) supposed to be given by a
/// function evaluated at "time" `t`.  A path is a sequence of points.
/// Points that are invalid (see [`Point::is_valid`]) are considered
/// cuts in the path (the line is interrupted).  As most 2 cuts may
/// follow each other (to track ranges of t outside the domain of the
/// function) except at the boundary of the path where at most 1 cut
/// is allowed.
#[derive(Debug)]
struct Point {
    t: f64, // "time" parameter, ALWAYS finite.
    xy: [f64; 2],
    cost: f64, // Cache the cost of the point (not the segment).  If
               // the point is not valid, the cost has no meaning.

    // Pointer to the priority queue node if the point is the initial
    // point of a segment.  `None` otherwise.
    witness: Option<pq::Witness<list::Witness<Point>>>,
}

impl Clone for Point {
    // As a cloned point may likely go to another path, reset the
    // priority queue pointer.
    fn clone(&self) -> Self {
        Self { witness: None, .. *self }
    }
}

impl Point {
    /// Return a new point.  `t` is assumed to be finite.
    #[inline]
    fn new_unchecked(t: f64, xy: [f64; 2]) -> Self {
        Point { t, xy,  cost: 0.,
                witness: None }
    }

    #[inline]
    fn cut(t: f64) -> Self {
        Point { t, xy: [f64::NAN; 2],
                cost: 0.,
                witness: None }
    }

    /// Return `true` if the point is valid — otherwise it is a cut.
    #[inline]
    fn is_valid(&self) -> bool {
        self.xy.iter().all(|z| z.is_finite())
    }

}

/// A 2D sampling.  This can be thought as a path, with possible
/// "cuts" because of discontinuities or leaving the domain of the
/// (parametric) function describing the path.
pub struct Sampling {
    // Priority queue of segments.  The value points to the first
    // `Point` of the segment (thus its .next pointer must not be
    // null).  At least one of the endpoint of all segments must be
    // valid (see `Point::is_valid`) If `pq` is empty but `begin` is
    // not null, it means that the costs need to to be updated.
    pq: PQ,
    path: List<Point>,
    // Guess on the path length (to allocate vectors) without making
    // `self` mutable (which the caller would not understand).
    guess_len: Cell<usize>,
    vp: Option<BoundingBox>, // viewport (zone of interest)
}

type PQ = pq::PQ<list::Witness<Point>>;

impl Clone for Sampling {
    fn clone(&self) -> Self {
        // It is guessed that one clones the sampling to transform it.
        // Thus start with an empty queue.
        Self { pq: PQ::new(),
               path: self.path.clone(),
               guess_len: self.guess_len.get().into(),
               vp: self.vp }
    }
}

/// `t` is the length of total range of time, `x` and `y` are the
/// dimensions of the bounding box.
#[derive(Debug, Clone, Copy)]
struct Lengths {
    t: f64,
    x: f64,
    y: f64,
}

impl Sampling {
    /// Return `true` if the sampling contains no point.
    #[inline]
    pub fn is_empty(&self) -> bool { self.path.is_empty() }

    /// Create an empty sampling.
    #[inline]
    pub(crate) fn empty() -> Self {
        Self { pq: PQ::new(),
               path: List::new(),
               guess_len: 0.into(),
               vp: None }
    }

    #[inline]
    pub(crate) fn singleton(p: Point) -> Self {
        debug_assert!(p.t.is_finite() && p.xy.iter().all(|z| z.is_finite()));
        let mut path = List::new();
        path.push_back(p);
        Self { pq: PQ::new(), path, guess_len: 1.into(), vp: None }
    }

    #[inline]
    pub(crate) fn set_vp(&mut self, bb: BoundingBox) {
        self.vp = Some(bb);
    }

    /// Return the length of the "time interval" as well as the
    /// lengths of the viewport.
    pub(crate) fn lengths(&self) -> Option<Lengths> {
        if self.is_empty() { return None }
        let p0 = self.path.first().unwrap();
        let p1 = self.path.last().unwrap();
        let len_x;
        let len_y;
        match self.vp {
            Some(vp) => {
                len_x = vp.xmax - vp.xmin;
                len_y = vp.ymax - vp.ymin;
            }
            None => {
                len_x = 1.;
                len_y = 1.;
            }
        }
        Some(Lengths { t: (p1.t - p0.t).abs(), x: len_x, y: len_y })
    }

    /// Returns an iterator on the points (and cuts) of the path.
    /// More precisely, a path is made of continuous segments whose
    /// points are given by contiguous values `[x,y]` with both `x`
    /// and `y` not NaN, interspaced by "cuts" `[f64::NAN; 2]`.  Two
    /// cuts never follow each other.  Isolated points `p` are given
    /// by ... `[f64::NAN; 2]`, `p`, `None`,...
    pub fn iter(&self) -> SamplingIter<'_> {
        SamplingIter {
            path: self.path.iter(),
            prev_is_cut: true,
            guess_len: self.guess_len.get(),
        }
    }

    /// Returns an iterator that allows to modify the points and cuts
    /// of the path.  Unlike [`iter`], this iterates on all the nodes
    /// even if several cuts (i.e., node with a non finite coordinate)
    /// follow each other.
    pub fn iter_mut(&mut self) -> SamplingIterMut<'_> {
        SamplingIterMut {
            path: self.path.iter_mut(),
            guess_len: self.guess_len.get(),
        }
    }

    /// Iterator on the x-coordinates of the sampling.
    /// See [`Self::iter`] for more information.
    #[inline]
    pub fn x(&self) -> Vec<f64> {
        let mut v = Vec::with_capacity(self.guess_len.get());
        for [x, _] in self.iter() { v.push(x) }
        v
    }

    /// Iterator on the y-coordinates of the sampling.
    /// See [`Self::iter`] for more information.
    #[inline]
    pub fn y(&self) -> Vec<f64> {
        let mut v = Vec::with_capacity(self.guess_len.get());
        for [_, y] in self.iter() { v.push(y) }
        v
    }
}

/// Iterator on the curve points (and cuts).
///
/// Created by [`Sampling::iter`].
pub struct SamplingIter<'a> {
    path: list::Iter<'a, Point>,
    prev_is_cut: bool,
    guess_len: usize,
}

impl<'a> Iterator for SamplingIter<'a> {
    type Item = [f64; 2];

    fn next(&mut self) -> Option<Self::Item> {
        match self.path.next() {
            None => None,
            Some(p) => {
                self.guess_len -= 1;
                if p.is_valid() {
                    self.prev_is_cut = false;
                    Some(p.xy)
                } else if self.prev_is_cut {
                    // Find the next valid point.
                    let r = self.path.try_fold(0, |n, p| {
                        if p.is_valid() {
                            ControlFlow::Break((n, p))
                        } else {
                            ControlFlow::Continue(n+1)
                        }
                    });
                    match r {
                        ControlFlow::Continue(_) => {
                            // Iterator exhausted
                            self.guess_len = 0;
                            None
                        }
                        ControlFlow::Break((n, p)) => {
                            self.guess_len -= n;
                            self.prev_is_cut = false;
                            Some(p.xy)
                        }
                    }
                } else {
                    self.prev_is_cut = true;
                    Some([f64::NAN; 2])
                }
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.guess_len))
    }
}

/// Mutable iterator on the curve points (and cuts).
///
/// Created by [`Sampling::iter_mut`].
pub struct SamplingIterMut<'a> {
    path: list::IterMut<'a, Point>,
    guess_len: usize,
}

impl<'a> Iterator for SamplingIterMut<'a> {
    type Item = &'a mut [f64; 2];

    fn next(&mut self) -> Option<Self::Item> {
        match self.path.next() {
            None => None,
            Some(p) => {
                self.guess_len -= 1;
                Some(&mut p.xy)
            }
        }
    }
}

/// Intersection of a segment with the bounding box.
#[derive(Debug)]
enum Intersection {
    Empty,
    Pt(Point),
    Seg(Point, Point),
}

impl Sampling {
    /// Return the smallest rectangle enclosing all the points of the
    /// sampling `self`.  If the path is empty, the "min" fields of
    /// the bounding box are set to +∞ and "max" fields to -∞.
    pub fn bounding_box(&self) -> BoundingBox {
        let mut points = self.iter().skip_while(|[x,_]| x.is_nan());
        let mut bb = match &points.next() {
            Some([x, y]) => BoundingBox { xmin: *x,  xmax: *x,
                                         ymin: *y,  ymax: *y },
            None => return BoundingBox::empty()
        };
        for [x, y] in points {
            if x < bb.xmin { bb.xmin = x }
            else if bb.xmax < x { bb.xmax = x };
            if y < bb.ymin { bb.ymin = y }
            else if bb.ymax < y { bb.ymax = y };
        }
        bb
    }

    /// Transpose in place the x and y coordinates of the sampling.
    pub fn transpose(&mut self) -> &mut Self {
        for p in self.path.iter_mut() {
            let [x, y] = p.xy;
            p.xy = [y, x];
        }
        self
    }

    /// Assume `p0` ∈ `bb` and `p1` ∉ `bb`.  Return the point
    /// intersecting the boundary of `bb`.  If the intersection point
    /// is the same as `p0`, return `None`
    #[inline]
    #[must_use]
    fn intersect(p0: &Point, p1: &Point, bb: BoundingBox) -> Option<Point> {
        let mut t = 1.; // t ∈ [0, 1]
        let [x0, y0] = p0.xy;
        let [x1, y1] = p1.xy;
        let dx = x1 - x0; // May be 0.
        let r = (if dx >= 0. {bb.xmax} else {bb.xmin} - x0) / dx;
        if r < t { t = r } // ⟹ r finite (as r ≥ 0 or NaN)
        let dy = y1 - y0; // May be 0.
        let r = (if dy >= 0. {bb.ymax} else {bb.ymin} - y0) / dy;
        if r < t { t = r };
        if t <= 1e-14 {
            None
        } else {
            Some(Point::new_unchecked(p0.t + t * (p1.t - p0.t),
                [x0 + t * dx, y0 + t * dy]))
        }
    }

    /// Assume `p0` ∉ `bb` and `p1` ∉ `bb` (thus, not even on the
    /// boundary of `bb`) and `p0` ≠ `p1`.  Return the endpoints of
    /// the segment intersecting the boundary of `bb` if any.  The
    /// "parameter direction" of `p0`, `p1` is preserved.
    #[inline]
    #[must_use]
    fn intersect_seg(p0: &Point, p1: &Point, bb: BoundingBox) -> Intersection {
        let mut t0 = 0.; // t0 ∈ [0, 1]
        let mut t1 = 1.; // t1 ∈ [0, 1]
        let [x0, y0] = p0.xy;
        let [x1, y1] = p1.xy;
        let dx = x1 - x0; // may be 0.
        let r0;
        let r1; // r0 ≤ r1 or NAN if on x-boundary lines.
        if dx >= 0. {
            r0 = (bb.xmin - x0) / dx;
            r1 = (bb.xmax - x0) / dx;
        } else {
            r0 = (bb.xmax - x0) / dx;
            r1 = (bb.xmin - x0) / dx;
        }
        if r0 > 1. || r1 < 0. { return Intersection::Empty }
        if r0 > 0. { t0 = r0 } // if r0 is NAN, keep the whole segment
        if r1 < 1. { t1 = r1 }
        let dy = y1 - y0; // may be 0.
        let r0;
        let r1;
        if dy >= 0. {
            r0 = (bb.ymin - y0) / dy;
            r1 = (bb.ymax - y0) / dy;
        } else {
            r0 = (bb.ymax - y0) / dy;
            r1 = (bb.ymin - y0) / dy;
        }
        if r0 > t1 || r1 < t0 { return Intersection::Empty }
        if r0 > t0 { t0 = r0 }
        if r1 < t1 { t1 = r1 }
        if t0 < t1 { // segment not reduced to a point
            let dt = p1.t - p0.t;
            let q0 = Point::new_unchecked(p0.t + t0 * dt,
                [x0 + t0 * dx, y0 + t0 * dy]);
            let q1 = Point::new_unchecked(p0.t + t1 * dt,
                [x0 + t1 * dx, y0 + t1 * dy]);
            Intersection::Seg(q0, q1)
        } else if t0 == t1 {
            let q0 = Point::new_unchecked(
                p0.t + t0 * (p1.t - p0.t),
                [x0 + t0 * dx, y0 + t0 * dy]);
            Intersection::Pt(q0)
        } else {
            Intersection::Empty
        }
    }

    /// Returns the sampling `self` but clipped to the 2D box `bb`.  A
    /// path that crosses the boundary will get additional nodes at
    /// the points of crossing and the part outside the bounding box
    /// will be dropped.  (Thus a path entirely out of the bounding
    /// box will be removed.)
    #[must_use]
    pub fn clip(&self, bb: BoundingBox) -> Self {
        let mut s = Sampling::empty();
        let mut new_len: usize = 0;
        // First point of the current segment, if any.
        let mut p0_opt: Option<&Point> = None;
        let mut p0_inside = false;
        let mut prev_cut = true; // New path ("virtual" cut)
        for p1 in self.path.iter() {
            if prev_cut {
                // A cut was pushed at an earlier step (and the path
                // has not subsequently intersected `bb`).  This may
                // be because the original path was just cut (`p0_opt
                // = None`) or because the previous segment was cut
                // (`p0_opt = Some(p0)` with `p0` ∉ `bb`) or because
                // there was an earlier cut and the next point `p0` is
                // the possible new start.
                match (p0_opt, p1.is_valid()) {
                    (Some(p0), true) => {
                        let p1_inside = bb.contains(p1);
                        if p0_inside { // p0 ∈ bb with cut before
                            s.path.push_back(p0.clone());
                            if p1_inside {
                                s.path.push_back(p1.clone());
                                new_len += 2;
                                prev_cut = false;
                            } else if
                                let Some(p) = Self::intersect(p0, p1, bb) {
                                    let t = p.t;
                                    s.path.push_back(p);
                                    s.path.push_back(Point::cut(t));
                                    new_len += 3;
                                } else {
                                    s.path.push_back(Point::cut(p0.t));
                                    new_len += 2;
                                }
                        } else if p1_inside { // p0 ∉ bb, p1 ∈ bb
                            if let Some(p) = Self::intersect(p1, p0, bb) {
                                s.path.push_back(p); // p ≠ p1
                                new_len += 1;
                            }
                            s.path.push_back(p1.clone());
                            new_len += 1;
                            prev_cut = false;
                        } else { // p0, p1 ∉ bb but maybe intersection
                            match Self::intersect_seg(p0, p1, bb) {
                                Intersection::Seg(q0, q1) => {
                                    let t1 = q1.t;
                                    s.path.push_back(q0);
                                    s.path.push_back(q1);
                                    s.path.push_back(Point::cut(t1));
                                    new_len += 3;
                                }
                                Intersection::Pt(p) => {
                                    let t = p.t;
                                    s.path.push_back(p);
                                    s.path.push_back(Point::cut(t));
                                    new_len += 2;
                                }
                                Intersection::Empty => (),
                            }
                        }
                        p0_opt = Some(p1);
                        p0_inside = p1_inside;
                    }
                    (None, true) => {
                        p0_opt = Some(p1);
                        p0_inside = bb.contains(p1);
                    }
                    (Some(p0), false) => {
                        if p0_inside {
                            s.path.push_back(p0.clone());
                            s.path.push_back(Point::cut(p0.t));
                            new_len += 2;
                        }
                        p0_opt = None;
                    }
                    (None, false) => (), // p0_opt == None
                }
            } else {
                // Previous step was not a cut which also implies that
                // `p0_opt = Some(p0)` with `p0` ∈ `bb`, and `p0` is
                // already present in the final sampling.
                let p0 = p0_opt.expect("No cut ⟹ previous point");
                debug_assert!(p0_inside);
                if p1.is_valid() {
                    p0_opt = Some(p1);
                    p0_inside = bb.contains(p1); // update for next step
                    if p0_inside { // p0, p1 ∈ bb
                        s.path.push_back(p1.clone());
                        new_len += 1;
                    } else { // p0 ∈ bb, p1 ∉ bb
                        if let Some(p) = Self::intersect(p0, p1, bb) {
                            let t = p.t;
                            s.path.push_back(p);
                            s.path.push_back(Point::cut(t));
                            new_len += 2;
                        } else {
                            s.path.push_back(Point::cut(p0.t));
                            new_len += 1;
                        }
                        prev_cut = true;
                    }
                } else { // p1 is invalid (i.e., represent a cut)
                    p0_opt = None;
                    s.path.push_back(p1.clone());
                    new_len += 1;
                    prev_cut = true
                }
            }
        }
        if prev_cut { s.path.pop_back(); }
        s.set_vp(bb);
        s.guess_len.set(new_len);
        s
    }

    /// Create a sampling from an iterator of points.  Beware that the
    /// invariant "`p.y` is finite ⇒ `p.x` is finite" is not checked.
    fn from_point_iterator<P>(points: P) -> Self
    where P: IntoIterator<Item = Point> {
        let mut s = Sampling::empty();
        let mut points = points.into_iter();
        let mut len: usize = 0;
        macro_rules! skip_until_last_cut { () => {
            let mut cut = None;
            let mut first_pt = None;
            for p in &mut points {
                if p.is_valid() { first_pt = Some(p); break; }
                cut = Some(p);
            }
            match (cut, first_pt) {
                (_, None) => {
                    s.guess_len.set(len);
                    return s
                }
                (None, Some(p)) => {
                    s.path.push_back(p);
                    len += 1;
                }
                (Some(c), Some(p)) => {
                    s.path.push_back(Point::cut(c.t));
                    s.path.push_back(p);
                    len += 2;
                }
            }
        }}
        skip_until_last_cut!();
        while let Some(p) = points.next() {
            if p.is_valid() {
                s.path.push_back(p);
                len += 1;
            } else {
                s.path.push_back(Point::cut(p.t));
                len += 1;
                skip_until_last_cut!();
            }
        }
        s.guess_len.set(len);
        s
    }

    /// Create a sampling from `points` after sorting them by
    /// increasing (if `incr`) or decreasing (if `! incr`) values of
    /// the field `t`.  Beware that the invariant "`p.y` is finite ⇒
    /// `p.x` is finite" is not checked.
    fn from_vec(mut points: Vec<Point>, incr: bool) -> Self {
        if incr {
            points.sort_unstable_by(|p1, p2| {
                // We know that `t1` and `t2` are finite.
                p1.t.partial_cmp(&p2.t).unwrap() });
        } else {
            points.sort_unstable_by(|p1, p2| {
                p2.t.partial_cmp(&p1.t).unwrap() });
        }
        Self::from_point_iterator(points)
    }
}


impl FromIterator<[f64; 2]> for Sampling {
    /// Return an sampling from the points.  Points with non-finite
    /// coordinates are interpreted as cuts.
    fn from_iter<T>(points: T) -> Self
    where T: IntoIterator<Item = [f64; 2]> {
        Sampling::from_point_iterator(
            points.into_iter().enumerate().map(|(i, xy @ [x, y])| {
                let t = i as f64;
                if x.is_finite() && y.is_finite() {
                    Point::new_unchecked(t, xy)
                } else {
                    Point::cut(t)
                } }))
    }
}

////////////////////////////////////////////////////////////////////////
//
// Acceptable types & functions that provide "points".
// These "conversions" must enforce the specification: `p.x` finite ⟹
// `p.y` finite.
// This is internal to this library.

impl From<(f64, f64)> for Point {
    #[inline]
    fn from((x, y): (f64, f64)) -> Self {
        // `x` ∈ [a, b] by `init_pt` checks.
        Point::new_unchecked(x, [x, y])
    }
}

impl From<(f64, [f64; 2])> for Point {
    /// Assume `t` is finite.
    #[inline]
    fn from((t, xy): (f64, [f64;2])) -> Self {
        // Enforce the invariant: y finite ⟹ x finite
        if xy[0].is_finite() { Point::new_unchecked(t, xy) }
        else { Point::cut(t) }
    }
}

/// Values that can be treated as Fn(f64) -> Point.
trait IntoFnPoint {
    fn eval(&mut self, t: f64) -> Point;
}

// This trait cannot implemented for both `FnMut(f64) -> f64` and
// `FnMut(f64) -> [f64; 2]` (that conflicts), so we wrap the types of
// interest.

struct FnPoint<T>(T);

impl<T> IntoFnPoint for FnPoint<T> where T: FnMut(f64) -> f64 {
    #[inline]
    fn eval(&mut self, t: f64) -> Point {
        Point::new_unchecked(t, [t, self.0(t)])
    }
}

struct ParamPoint<T>(T);

impl<T> IntoFnPoint for ParamPoint<T> where T: FnMut(f64) -> [f64; 2] {
    #[inline]
    fn eval(&mut self, t: f64) -> Point {
        let xy @ [x, y] = self.0(t);
        if x.is_finite() && y.is_finite() {
            Point::new_unchecked(t, xy)
        } else {
            Point::new_unchecked(t, [xy[0], f64::NAN])
        }
    }
}

////////////////////////////////////////////////////////////////////////
//
// Defining a sampling with standard options & checks

/// Define a structure with standard fields, standard options, and a
/// function to generate it.
macro_rules! new_sampling_fn {
    // Function to init the struct.
    ($(#[$docfn: meta])*, $(#[$docfn_extra: meta])* $fun: ident -> $ft: ty,
     // The structure to hold the options (and other fields).
     $(#[$doc: meta])* $struct: ident,
     $wrap_f: ident
    ) => {
        impl Sampling {
            $(#[$docfn])*
            ///
            /// Panics if `a` or `b` is not finite.
            ///
            $(#[$docfn_extra])*
            #[must_use]
            pub fn $fun<F>(f: F, a: f64, b: f64) -> $struct<F>
            where F: FnMut(f64) -> $ft {
                if !a.is_finite() {
                    panic!("curve_sampling::{}: a = {} must be finite",
                           stringify!($fun), a);
                }
                if !b.is_finite() {
                    panic!("curve_sampling::{}: b = {} must be finite",
                           stringify!($fun), b);
                }
                $struct { f: $wrap_f(f),
                          a, b,  // Order of `a`, `b` reflect orientation
                          n: 100,
                          viewport: None,
                          init: vec![],
                          init_pt: vec![],
                }
            }
        }

        $(#[$doc])*
        pub struct $struct<F> {
            f: $wrap_f<F>,  a: f64,  b: f64,
            n: usize,
            viewport: Option<BoundingBox>,
            init: Vec<f64>,
            init_pt: Vec<Point>,
        }

        impl<F> $struct<F>
        where F: FnMut(f64) -> $ft {
            /// Set the maximum number of evaluations of the function
            /// to build the sampling.  Panic if `n < 2`.
            pub fn n(mut self, n: usize) -> Self {
                if n < 2 {
                    panic!("curve_sampling: n = {} must at least be 2", n)
                }
                self.n = n;
                self
            }

            /// Set the zone of interest for the sampling.  Segments
            /// that end up outside this box will not be refined.
            pub fn viewport(mut self, vp: BoundingBox) -> Self {
                self.viewport = Some(vp);
                self
            }

            /// Add initial values of `t` such that `f(t)` (see [`
            #[doc = stringify!(Sampling::$fun)]
            /// `]) must be included into the sampling in addition to
            /// the `n` evaluations.  Only the values between `a` and
            /// `b` are taken into account (other values are ignored).
            pub fn init<'a, I>(mut self, ts: I) -> Self
            where I: IntoIterator<Item = &'a f64> {
                for &t in ts {
                    if self.a <= t && t <= self.b { // ⟹ t is finite
                        self.init.push(t);
                    }
                }
                self
            }

            /// Add initial points `(t, f(t))` to include into the
            /// sampling in addition to the `n` evaluations.  This
            /// allows you to use previous evaluations of `f`.  Only
            /// the couples with first coordinate `t` between `a` and
            /// `b` (see [`
            #[doc = stringify!(Sampling::$fun)]
            /// `]) are considered (other values are ignored).
            pub fn init_pt<'a, I>(mut self, pts: I) -> Self
            where I: IntoIterator<Item = &'a (f64, $ft)> {
                for &p in pts {
                    if self.a <= p.0 && p.0 <= self.b { // ⟹ p.0 = t is finite
                        self.init_pt.push(Point::from(p));
                    }
                }
                self
            }

            /// Evaluate the function at all initial values that where
            /// provided through [`Self::init`].
            fn eval_init(&mut self) {
                // `t` ∈ \[`a`, `b`\] already checked by [`init`] and
                // [`init_pt`].
                for &t in &self.init {
                    self.init_pt.push(self.f.eval(t))
                }
                self.init.clear()
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////
//
// Uniform sampling

new_sampling_fn!(
    /// Create a sampling for the graph of `f` on the interval
    /// \[`a`, `b`\] with evenly spaced values of the argument.
    ,
    /// # Example
    ///
    /// ```
    /// use std::{fs::File, io::BufWriter};
    /// use curve_sampling::Sampling;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let s = Sampling::uniform(|x| x.sin(), 0., 4.).build();
    /// s.write(&mut BufWriter::new(File::create("target/uniform.dat")?))?;
    /// # Ok(()) }
    /// ```
    uniform -> f64,
    /// Options for uniform sampling graphs of function ℝ → ℝ.
    /// See [`Sampling::uniform`].
    Uniform,
    FnPoint);

impl<F> Uniform<F>
where F: FnMut(f64) -> f64 {
    /// Return a uniform sampling of the function.
    pub fn build(mut self) -> Sampling {
        if self.a == self.b {
            let p = self.f.eval(self.a); // `a` is finite by previous tests
            if p.is_valid() {
                return Sampling::singleton(p);
            } else {
                return Sampling::empty()
            }
        }
        self.eval_init();
        let mut points = self.init_pt;
        let dt = (self.b - self.a) / (self.n - 1) as f64;
        for i in 0 .. self.n {
            let t = self.a + i as f64 * dt;
            points.push(self.f.eval(t));
        }
        Sampling::from_vec(points, self.a < self.b)
    }
}

////////////////////////////////////////////////////////////////////////
//
// Cost

mod cost {
    use super::{Point, Lengths};

    // The cost of a point is a measure of the curvature at this
    // point.  This requires segments before and after the point.  In
    // case the point is a cut, or first, or last, it has a cost of 0.
    // If it is an endpoint of a segment with the other point a cut,
    // the cost is set to [`HANGING_NODE`] because the segment with
    // the invalid point needs to be cut of too long to better
    // determine the boundary.  Only the absolute value of the cost
    // matters for the priority queue (see `segment`), its sign must
    // reflect the orientation of the angle.
    //
    // The cost of a point is apportioned to the segments of which it is
    // an endpoint according to their relative lengths.  More precisely,
    // the cost c of a point p is distributed on the segments s1 and s2
    // (of respective lengths l1 and l2) it is an endpoint of as
    //
    //   c * l1/(l1+l2) for s1 and c * l2/(l1+l2) for s2.
    //
    // In order to be able to update the cost of s1 without accessing
    // s2, p.cost holds c/(l1+l2).

    /// Cost for new "hanging" nodes — nodes created splitting a
    /// segment with an invalid endpoint.  Note that this cost will be
    /// multiplied by a function of `dt` in `segment` so it must be
    /// set high enough to ensure proper resolution of the endpoints
    /// of the domain.
    pub const HANGING_NODE: f64 = 5e5;

    /// Set the cost of the middle point `pm`.  Assumes `p0`, `pm`,
    /// and `p1` are valid points.
    #[inline]
    pub(crate) fn set_middle(p0: &Point, pm: &mut Point, p1: &Point,
                             len: Lengths)
    {
        let [x0, y0] = p0.xy;
        let [xm, ym] = pm.xy;
        let [x1, y1] = p1.xy;
        let dx0m = (x0 - xm) / len.x;
        let dy0m = (y0 - ym) / len.y;
        let dx1m = (x1 - xm) / len.x;
        let dy1m = (y1 - ym) / len.y;
        let len0m = dx0m.hypot(dy0m);
        let len1m = dx1m.hypot(dy1m);
        if len0m == 0. || len1m == 0. {
            pm.cost = 0.; // Do not subdivide
        } else {
            let dx = - dx0m * dx1m - dy0m * dy1m;
            let dy = dy0m * dx1m - dx0m * dy1m;
            pm.cost = dy.atan2(dx); // ∈ [-π, π]
            debug_assert!(!pm.cost.is_nan());
        }
    }

    /// Compute the cost of a segment according to the costs of its
    /// endpoints unless `in_vp` is false in which case -∞ is returned.
    #[inline]
    pub(crate) fn segment_vp(p0: &Point, p1: &Point, len: Lengths,
                             in_vp: bool) -> f64 {
        if ! in_vp { return f64::NEG_INFINITY }
        segment(p0, p1, len)
    }

    /// Compute the cost of a segment according to the costs of its
    /// endpoints.
    #[inline]
    pub(crate) fn segment(p0: &Point, p1: &Point, len: Lengths) -> f64 {
        let dt = (p1.t - p0.t).abs() / len.t; // ∈ [0,1]
        debug_assert!((0. ..=1.).contains(&dt), "dt = {dt} ∉ [0,1]");
        // Put less efforts when `dt` is small.  For functions, the
        // Y-variation may be large but, if it happens for a small range
        // of `t`, there is no point in adding indistinguishable details.
        let [x0, y0] = p0.xy;
        let [x1, y1] = p1.xy;
        let dx = ((x1 - x0) / len.x).abs();
        let dy = ((y1 - y0) / len.y).abs();
        let mut cost = p0.cost.abs() + p1.cost.abs(); // ≥ 0
        if p0.cost * p1.cost < 0. {
            // zigzag are bad on a large scale but less important on a
            // small scale.
            if dx <= 0.01 && dy <= 0.01 { cost *= 0.5 }
            else if !(dx <= 0.05 && dy <= 0.05) { cost *= 8. }
        }
        if dt >= 0.8 { cost }
        else {
            let dt = dt / 0.8;
            dt * dt * (6. + (-8. + 3. * dt) * dt) * cost
        }
    }

}

/// Compute the cost of the segment \[`p0`, `p1`\] (taking into
/// account `in_vp`) and push it to the queue `pq`.  `p0` is pointed
/// to by a list-witness so one can update it from the PQ elements.
/// `p0` is updated with the PQ-witness.
fn push_segment(pq: &mut PQ,
                p0: &mut list::Witness<Point>, p1: &Point,
                len: Lengths, in_vp: bool) {
    // FIXME: do we really want to push the segment when `!in_vp`?
    // In not all points are in the queue, one must beware of the
    // validity of witnesses though.
    let cost_segment = cost::segment_vp(unsafe { p0.as_ref() },
                                        p1, len, in_vp);
    // The segment is referred to by its first point.
    let w = pq.push(cost_segment, p0.clone());
    unsafe { p0.as_mut().witness = Some(w) }
}

/// With the (new) costs in points `p0` and `p1`, update the position
/// of the segment \[`p0`, `p1`\] in the priority queue of `self`.
///
/// # Safety
/// `p0` must be in `pq`, otherwise it is UB.
unsafe fn update_segment(pq: &mut PQ, p0: &Point, p1: &Point, len: Lengths) {
    match &p0.witness {
        Some(w) => {
            let priority = cost::segment(p0, p1, len);
            pq.increase_priority(w, priority)
        }
        None => panic!("Sampling::update_segment: unset witness"),
    }
}

////////////////////////////////////////////////////////////////////////
//
// Function sampling

/// Update the cost of all points in the sampling and add segments
/// to the priority queue.
fn compute(s: &mut Sampling, in_vp: impl Fn(&Point) -> bool) {
    if let Some(len) = s.lengths() {
        macro_rules! r { ($x: ident) => { unsafe { $x.as_ref() } } }
        macro_rules! m { ($x: ident) => { unsafe { $x.as_mut() } } }
        // Path is not empty.
        let mut pts = s.path.iter_witness_mut();
        let mut p0 = pts.next().unwrap();
        m!(p0).cost = 0.;
        let mut p0_is_valid = r!(p0).is_valid();
        let mut p0_in_vp = p0_is_valid && in_vp(r!(p0));
        let mut pm = match pts.next() {
            Some(p) => p,
            None => return };
        for p1 in pts {
            let pm_is_valid = r!(pm).is_valid();
            let pm_in_vp;
            if pm_is_valid {
                pm_in_vp = in_vp(r!(pm));
                if p0_is_valid && r!(p1).is_valid() {
                    cost::set_middle(r!(p0), m!(pm), r!(p1), len);
                } else {
                    m!(pm).cost = cost::HANGING_NODE;
                }
            } else { // pm is the location of a cut
                pm_in_vp = false;
                m!(pm).cost = 0.;
            }
            if p0_is_valid || pm_is_valid {
                // Add segment [p0, pm] to the PQ and set `p0` witness.
                push_segment(&mut s.pq, &mut p0, r!(pm), len,
                             p0_in_vp || pm_in_vp);
            }
            p0 = pm;
            p0_is_valid = pm_is_valid;
            p0_in_vp = pm_in_vp;
            pm = p1;
        }
        m!(pm).cost = 0.; // last point
        if p0_is_valid || r!(pm).is_valid() {
            let mut vp = p0_in_vp;
            if r!(pm).is_valid() { vp = vp || in_vp(r!(pm)) };
            push_segment(&mut s.pq, &mut p0, r!(pm), len, vp);
        }
    }
}

fn refine_gen(s: &mut Sampling, n: usize,
              mut f: impl FnMut(f64) -> Point,
              in_vp: impl Fn(&Point) -> bool) {
    let len = match s.lengths() {
        Some(lengths) => lengths,
        None => return };
    s.guess_len.set(s.guess_len.get() + n);
    macro_rules! r { ($x: ident) => { unsafe { $x.as_ref() } } }
    macro_rules! m { ($x: ident) => { unsafe { $x.as_mut() } } }
    for _ in 0 .. n {
        let mut p0: list::Witness<Point> = match s.pq.pop() {
            None => break,
            Some(p) => p };
        m!(p0).witness = None; // PQ element it points to just popped.
        let mut p1 = unsafe { p0.next().unwrap() };
        // Refine the segment [p0, p1] inserting a middle point `pm`.
        let t = (r!(p0).t + r!(p1).t) * 0.5;
        let mut pm = f(t);
        if r!(p0).is_valid() {
            if r!(p1).is_valid() {
                let mut pm = unsafe { s.path.insert_after(&mut p0, pm) };
                let mut pm_in_vp = false;
                if r!(pm).is_valid() {
                    pm_in_vp = in_vp(r!(pm));
                    cost::set_middle(r!(p0), m!(pm), r!(p1), len);
                    if let Some(p_1) = unsafe { p0.prev() } {
                        if r!(p_1).is_valid() {
                            cost::set_middle(r!(p_1), m!(p0), r!(pm), len);
                            unsafe {
                                update_segment(&mut s.pq, p_1.as_ref(),
                                               p0.as_ref(), len) }
                        }
                    }
                    if let Some(p2) = unsafe { p1.next() } {
                        if r!(p2).is_valid() {
                            cost::set_middle(r!(pm), m!(p1), r!(p2), len);
                            unsafe {
                                update_segment(&mut s.pq, p1.as_ref(),
                                               p2.as_ref(), len) }
                        }
                    }
                } else { // `pm` invalid ⟹ cut between `p0` and `p1`
                    m!(p0).cost = cost::HANGING_NODE;
                    m!(pm).cost = 0.;
                    m!(p1).cost = cost::HANGING_NODE;
                    unsafe {
                        if let Some(p_1) = p0.prev() {
                            update_segment(&mut s.pq, p_1.as_ref(),
                                           p0.as_ref(), len)
                        }
                        if let Some(p2) = p1.next() {
                            update_segment(&mut s.pq, p1.as_ref(),
                                           p2.as_ref(), len);
                        }
                    }
                }
                let vp = pm_in_vp || in_vp(r!(p0));
                push_segment(&mut s.pq, &mut p0, r!(pm), len, vp);
                let vp = pm_in_vp || in_vp(r!(p1));
                push_segment(&mut s.pq, &mut pm, r!(p1), len, vp);
            } else { // `p0` valid, `p1` invalid (i.e. is a cut)
                // Thus `p0` is a hanging node.
                if pm.is_valid() {
                    pm.cost = cost::HANGING_NODE;
                    let mut pm = unsafe { s.path.insert_after(&mut p0, pm) };
                    if let Some(p_1) = unsafe { p0.prev() } {
                        if r!(p_1).is_valid() {
                            cost::set_middle(r!(p_1), m!(p0), r!(pm), len);
                            unsafe{update_segment(&mut s.pq, p_1.as_ref(),
                                                  p0.as_ref(), len)}
                        }
                    }
                    let pm_in_vp = in_vp(r!(pm));
                    let vp = pm_in_vp || in_vp(r!(p0));
                    push_segment(&mut s.pq, &mut p0, r!(pm), len, vp);
                    push_segment(&mut s.pq, &mut pm, r!(p1), len, pm_in_vp)
                } else { // `pm` invalid
                    // Insert only \[`p0`, `pm`\] and forget
                    // \[`pm`, `p1`\].  The cost of `p0` stays
                    // `cost::HANGING_NODE`.  We can see this as
                    // reducing the uncertainty of the boundary in the
                    // segment \[`p0`, `p1`\].
                    pm.cost = 0.;
                    let pm = unsafe {
                        if p1.as_ref().witness.is_none() {
                            // `p1` is not part of a segment.  One can
                            // replace it by `pm`.
                            s.path.replace(&mut p1, pm);
                            p1 // witness for `pm` now.
                        } else {
                            s.path.insert_after(&mut p0, pm)
                        } };
                    let vp = in_vp(r!(p0));
                    push_segment(&mut s.pq, &mut p0, r!(pm), len, vp)
                }
            }
        } else { // `p0` invalid (i.e., cut) ⟹ `p1` valid
            debug_assert!(r!(p1).is_valid());
            if pm.is_valid() {
                pm.cost = cost::HANGING_NODE;
                let mut pm = unsafe { s.path.insert_after(&mut p0, pm) };
                if let Some(p2) = unsafe { p1.next() } {
                    if r!(p2).is_valid() {
                        cost::set_middle(r!(pm), m!(p1), r!(p2), len);
                        unsafe{update_segment(&mut s.pq, p1.as_ref(),
                                              p2.as_ref(), len)}
                    }
                }
                let pm_in_vp = in_vp(r!(pm));
                push_segment(&mut s.pq, &mut p0, r!(pm), len, pm_in_vp);
                push_segment(&mut s.pq, &mut pm, r!(p1), len,
                             pm_in_vp || in_vp(r!(p1)))
            } else { // `pm` invalid ⟹ drop segment \[`p0`, `pm`\].
                // Cost of `p1` stays `cost::HANGING_NODE`.
                pm.cost = 0.;
                let mut pm = unsafe {
                    if let Some(p_1) = p0.prev() {
                        if p_1.as_ref().is_valid() {
                            s.path.insert_after(&mut p0, pm)
                        } else {
                            // `p_1` is the cut ending the previous segment.
                            s.path.replace(&mut p0, pm);
                            p0
                        }
                    } else {
                        s.path.insert_after(&mut p0, pm)
                    } };
                let vp = in_vp(r!(p1));
                push_segment(&mut s.pq, &mut pm, r!(p1), len, vp)
            }
        }
    }
}

fn push_almost_uniform_sampling(points: &mut Vec<Point>,
                                f: &mut impl FnMut(f64) -> Point,
                                a: f64, b: f64, n: usize) {
    debug_assert!(n >= 4);
    let dt = (b - a) / (n - 3) as f64;
    // Pseudorandom number generator from the "Xorshift RNGs" paper by
    // George Marsaglia.
    // See https://matklad.github.io/2023/01/04/on-random-numbers.html and
    // https://github.com/rust-lang/rust/blob/1.55.0/library/core/src/slice/sort.rs#L559-L573
    let mut random = (n as u32).wrapping_mul(1_000_000);
    const NORMALIZE_01: f64 = 1. / u32::MAX as f64;
    let mut rand = move || {
        random ^= random << 13;
        random ^= random >> 17;
        random ^= random << 5;
        random as f64 * NORMALIZE_01
    };
    points.push(f(a));
    points.push(f(a + 0.0625 * dt));
    for i in 1 ..= n - 4 {
        let j = i as f64 + rand() * 0.125 - 0.0625;
        points.push(f(a + j * dt));
    }
    points.push(f(b - 0.0625 * dt));
    points.push(f(b));
}

impl Sampling {
    /// Return a sampling from the initial list of `points`.
    fn build(mut points: Vec<Point>,
             mut f: impl FnMut(f64) -> Point,
             a: f64, b: f64, n: usize,
             viewport: Option<BoundingBox>) -> Sampling {
        if a == b {
            let p = f(a);
            if p.is_valid() {
                return Sampling::singleton(p);
            } else {
                return Sampling::empty()
            }
        }
        // Uniform sampling requires ≥ 4 points but actually makes no
        // sense with less than 10 points.
        let n0 = (n / 10).max(10);
        push_almost_uniform_sampling(&mut points, &mut f, a, b, n0);
        let mut s = Sampling::from_vec(points, a < b);
        match viewport {
            Some(vp) => {
                let in_vp = |p: &Point| vp.contains(p);
                compute(&mut s, in_vp);
                refine_gen(&mut s, n - n0, &mut f, in_vp)
            }
            None => {
                compute(&mut s, |_| true);
                refine_gen(&mut s, n - n0, &mut f, |_| true)
            }
        }
        s
    }
}


new_sampling_fn!(
    /// Create a sampling of the *graph* of `f` on the interval
    /// \[`a`, `b`\] by evaluating `f` at `n` points.
    ,
    /// # Example
    ///
    /// ```
    /// use std::{fs::File, io::BufWriter};
    /// use curve_sampling::Sampling;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let s = Sampling::fun(|x| x.sin(), 0., 4.).build();
    /// s.write(&mut BufWriter::new(File::create("target/fun.dat")?))?;
    /// # Ok(()) }
    /// ```
    fun -> f64,
    /// Options for sampling graphs of functions ℝ → ℝ.
    /// See [`Sampling::fun`].
    Fun,
    FnPoint);

impl<F> Fun<F>
where F: FnMut(f64) -> f64 {
    /// Return the sampling.
    pub fn build(mut self) -> Sampling {
        self.eval_init();
        Sampling::build(self.init_pt, |x| self.f.eval(x),
                        self.a, self.b, self.n, self.viewport)
    }
}


new_sampling_fn!(
    /// Create a sampling of the *image* of `f` on the interval
    /// \[`a`, `b`\] by evaluating `f` at `n` points.
    ,
    /// # Example
    ///
    /// ```
    /// use std::{fs::File, io::BufWriter};
    /// use curve_sampling::Sampling;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let s = Sampling::param(|t| [t.sin(), t.cos()], 0., 4.).build();
    /// s.write(&mut BufWriter::new(File::create("target/param.dat")?))?;
    /// # Ok(()) }
    /// ```
    param -> [f64; 2],
    /// Options for sampling the image of functions ℝ → ℝ².
    /// See [`Sampling::param`].
    Param,
    ParamPoint);

impl<F> Param<F>
where F: FnMut(f64) -> [f64; 2] {
    /// Return the sampling.
    pub fn build(mut self) -> Sampling {
        self.eval_init();
        Sampling::build(self.init_pt, |t| self.f.eval(t),
                        self.a, self.b, self.n, self.viewport)
    }
}


////////////////////////////////////////////////////////////////////////
//
// Output

/// LaTeX output, created by [`Sampling::latex`].
///
/// # Example
///
/// ```
/// use std::fs::File;
/// use curve_sampling::Sampling;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let s = Sampling::from_iter([[0., 0.], [1., 1.]]);
/// s.latex().write(&mut File::create("target/sampling.tex")?)?;
/// # Ok(()) }
/// ```
pub struct LaTeX<'a> {
    sampling: &'a Sampling,
    n: usize,
    color: Option<RGB8>,
    arrow: Option<&'a str>,
    arrow_pos: Option<f64>,
}

impl<'a> LaTeX<'a> {
    #[inline]
    fn new(s: &'a Sampling) -> Self {
        Self { sampling: s,  n: 20_000,  color: None,
               arrow: None,  arrow_pos: None }
    }

    /// Set the maximum number of points of a PGF path to `n`.  If it
    /// contains more than `n` points, the sampling curve is drawn as
    /// several PGF paths.  Default: 20_000.
    pub fn n(&mut self, n: usize) -> &mut Self {
        self.n = n;
        self
    }

    /// Set the color of the curve to `color`.  If not specified the
    /// active LaTeX color will be used.
    pub fn color(&mut self, color: RGB8) -> &mut Self {
        self.color = Some(color);
        self
    }

    /// Set the type of arrow to draw to `arrow`.
    /// See the documentation of `\pgfsetarrowsend` in the
    /// [TikZ manual](https://tikz.dev/base-actions#sec-103.3).
    pub fn arrow(&mut self, arrow: &'a str) -> &mut Self {
        self.arrow = Some(arrow);
        self
    }

    /// The position of the arrow as a percent of the curve length (in
    /// the interval \[0.,1.\]).  If [`LaTeX::arrow`] is specified but
    /// not this, it defaults to `0.5`.
    pub fn arrow_pos(&mut self, arrow_pos: f64) -> &mut Self {
        if ! arrow_pos.is_finite() {
            panic!("curve_sampling::LaTeX::arrow_pos: \
                    position must be finite");
        }
        self.arrow_pos = Some(arrow_pos.clamp(0., 1.));
        self
    }

    /// Write the sampling with lines segments.
    fn write_with_lines(&self, f: &mut impl Write) -> Result<(), io::Error> {
        let mut n = 0;
        let mut new_path = true;
        for [x,y] in self.sampling.iter() {
            if x.is_nan() {
                writeln!(f, "\\pgfusepath{{stroke}}")?;
                n = 0;
                new_path = true;
            } else {
                n += 1;
                if new_path {
                    writeln!(f, "\\pgfpathmoveto{{\\pgfpointxy\
                                 {{{:.16}}}{{{:.16}}}}}", x, y)?
                } else if n >= self.n {
                    write!(f, "\\pgfpathlineto{{\\pgfpointxy\
                               {{{:.16}}}{{{:.16}}}}}\n\
                               \\pgfusepath{{stroke}}\n\
                               \\pgfpathmoveto{{\\pgfpointxy\
                               {{{:.16}}}{{{:.16}}}}}\n", x, y, x, y)?;
                    n = 0;
                } else {
                    writeln!(f, "\\pgfpathlineto{{\\pgfpointxy\
                                 {{{:.16}}}{{{:.16}}}}}", x, y)?
                }
                new_path = false;
            }
        }
        Ok(())
    }

    /// Write the path, each continuous sub-path with an arrow.
    fn write_with_arrows(&self, f: &mut impl Write, arrow: &str,
                         arrow_pos: f64) -> Result<(), io::Error> {
        // Compute the length of all sub-paths.
        let mut lens = vec![];
        let mut prev_pt: Option<[f64; 2]> = None;
        let mut cur_len = 0.;
        for p @ [x, y] in self.sampling.iter() {
            if x.is_nan() {
                lens.push(arrow_pos * cur_len);
                prev_pt = None;
                cur_len = 0.;
            } else {
                if let Some([x0, y0]) = prev_pt {
                    cur_len += (x - x0).hypot(y - y0);
                }
                prev_pt = Some(p);
            }
        }
        lens.push(arrow_pos * cur_len);
        if lens.is_empty() { return Ok(()) }
        let mut lens = lens.iter();
        let mut rem_len = *lens.next().unwrap(); // remaining before arrow
        prev_pt = None;
        let mut n = 0;
        for p @ [x, y] in self.sampling.iter() {
            if x.is_nan() {
                writeln!(f, "\\pgfusepath{{stroke}}")?;
                rem_len = *lens.next().unwrap();
                prev_pt = None;
            } else {
                n += 1;
                if let Some([x0, y0]) = prev_pt {
                    let dx = x - x0;
                    let dy = y - y0;
                    let l = dx.hypot(dy);
                    if rem_len <= l {
                        writeln!(f, "\\pgfusepath{{stroke}}")?;
                        // Drawing a long path with an arrow specified is
                        // extremely expensive.  Just draw the current segment.
                        let pct = rem_len / l;
                        if pct <= 1e-14 {
                            write!(f, "\\pgfsetarrowsstart{{{}}}\n\
                                       \\pgfpathmoveto{{\\pgfpointxy
                                       {{{:.16}}}{{{:.16}}}}}\n\
                                       \\pgfpathlineto{{\\pgfpointxy\
                                       {{{:.16}}}{{{:.16}}}}}\n\
                                       \\pgfusepath{{stroke}}\n\
                                       \\pgfsetarrowsend{{}}\n\
                                       \\pgfpathmoveto{{\\pgfpointxy\
                                       {{{:.16}}}{{{:.16}}}}}\n",
                                   arrow, x0, y0, x, y, x, y)?;
                        } else {
                            let xm = x0 + pct * dx;
                            let ym = y0 + pct * dy;
                            write!(f, "\\pgfsetarrowsend{{{}}}\n\
                                       \\pgfpathmoveto{{\\pgfpointxy\
                                       {{{:.16}}}{{{:.16}}}}}\n\
                                       \\pgfpathlineto{{\\pgfpointxy\
                                       {{{:.16}}}{{{:.16}}}}}\n\
                                       \\pgfusepath{{stroke}}\n\
                                       \\pgfsetarrowsend{{}}\n\
                                       \\pgfpathmoveto{{\\pgfpointxy\
                                       {{{:.16}}}{{{:.16}}}}}\n\
                                       \\pgfpathlineto{{\\pgfpointxy\
                                       {{{:.16}}}{{{:.16}}}}}\n",
                                   arrow, x0, y0, xm, ym, xm, ym, x, y)?;
                            n = 2;
                        }
                        rem_len = f64::INFINITY; // No more arrow for this sub-path
                    } else if n >= self.n {
                        write!(f, "\\pgfpathlineto{{\\pgfpointxy\
                                   {{{:.16}}}{{{:.16}}}}}\n\
                                   \\pgfusepath{{stroke}}\n\
                                   \\pgfpathmoveto{{\\pgfpointxy\
                                   {{{:.16}}}{{{:.16}}}}}\n", x, y, x, y)?;
                        n = 0;
                    } else {
                        writeln!(f, "\\pgfpathlineto{{\\pgfpointxy\
                                     {{{:.16}}}{{{:.16}}}}}", x, y)?
                    }
                    rem_len -= l;
                } else {
                    // No previous point.  New sub-path.
                    writeln!(f, "\\pgfpathmoveto{{\\pgfpointxy\
                                 {{{:.16}}}{{{:.16}}}}}", x, y)?
                }
                prev_pt = Some(p);
            }
        }
        Ok(())
    }

    /// Write the sampling to the formatter as PGF/TikZ commands.
    pub fn write(&self, f: &mut impl Write) -> Result<(), io::Error> {
        writeln!(f, "% Written by the Rust curve-sampling crate.")?;
        writeln!(f, "\\begin{{pgfscope}}")?;
        if let Some(RGB8 {r, g, b}) = self.color {
            write!(f, "\\definecolor{{RustCurveSamplingColor}}{{RGB}}\
                       {{{},{},{}}}\n\
                       \\pgfsetstrokecolor{{RustCurveSamplingColor}}\n",
                   r, g, b)?
        }
        match (self.arrow, self.arrow_pos) {
            (None, None) => self.write_with_lines(f)?,
            (Some(arrow), None) =>
                self.write_with_arrows(f, arrow, 0.5)?,
            (None, Some(arrow_pos)) =>
                self.write_with_arrows(f, ">",arrow_pos)?,
            (Some(arrow), Some(arrow_pos)) =>
                self.write_with_arrows(f, arrow, arrow_pos)?,
        }
        write!(f, "\\pgfusepath{{stroke}}\n\\end{{pgfscope}}\n")
    }
}

/// # Output
impl Sampling {
    /// Write the sampling `self` using PGF/TikZ commands.
    pub fn latex(&self) -> LaTeX<'_> { LaTeX::new(self) }

    /// Write the sampling to `f` in a tabular form: each point is
    /// written as "x y" on a single line (in scientific notation).
    /// If the path is interrupted, a blank line is printed.  This
    /// format is compatible with Gnuplot.
    pub fn write(&self, f: &mut impl Write) -> Result<(), io::Error> {
        for [x, y] in self.iter() {
            if x.is_nan() {
                writeln!(f)?
            } else {
                writeln!(f, "{:e} {:e}", x, y)?
            }
        }
        Ok(())
    }
}

impl Display for Sampling {
    /// Display the sampling in a tabular form: each point is written
    /// as "x y" on a single line (in scientific notation).  If the
    /// path is interrupted, a blank line is printed.  This format is
    /// compatible with Gnuplot.
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), fmt::Error> {
        for [x, y] in self.iter() {
            if x.is_nan() {
                writeln!(f)?
            } else {
                writeln!(f, "{:e} {:e}", x, y)?
            }
        }
        Ok(())
    }
}

////////////////////////////////////////////////////////////////////////
//
// Tests

#[cfg(test)]
mod tests {
    use std::{error::Error,
              fs::File,
              io::Write, path::Path};
    use crate::{Sampling, BoundingBox, Point};

    type R<T> = Result<T, Box<dyn Error>>;

    fn xy_of_sampling(s: &Sampling) -> Vec<Option<(f64, f64)>> {
        s.path.iter().map(|p| {
            if p.is_valid() { Some((p.xy[0], p.xy[1])) } else { None }
        })
            .collect()
    }

    #[test]
    fn clone_sampling() {
        let s = Sampling::from_iter([[0.,0.], [1.,1.]]);
        let xy0: Vec<_> = s.iter().collect();
        let xy1: Vec<_> = s.clone().iter().collect();
        assert_eq!(xy0, xy1)
    }

    #[test]
    fn io() {
        let s = Sampling::from_iter([[0.,0.], [1.,1.]]);
        assert_eq!(xy_of_sampling(&s), vec![Some((0.,0.)), Some((1.,1.))]);
        let s = Sampling::from_iter([[0.,0.], [1.,1.], [f64::NAN, 1.],
                                     [2.,2.]]);
        assert_eq!(xy_of_sampling(&s),
                   vec![Some((0.,0.)), Some((1.,1.)), None, Some((2.,2.))]);
    }

    #[test]
    fn bounding_box_singleton() {
        let s = Sampling::singleton(
            Point::new_unchecked(0., [1., 2.]));
        let bb = BoundingBox {xmin: 1., xmax: 1., ymin: 2., ymax: 2.};
        assert_eq!(s.bounding_box(), bb);
    }

    fn test_clip(bb: BoundingBox,
                 path: Vec<[f64;2]>,
                 expected: Vec<Option<(f64,f64)>>) {
        let s = Sampling::from_iter(path).clip(bb);
        assert_eq!(xy_of_sampling(&s), expected);
    }

    #[test]
    fn clip_base () {
        let bb = BoundingBox { xmin: 0.,  xmax: 3., ymin: 0.,  ymax: 2.};
        test_clip(bb, vec![[0.,1.], [2.,3.]],
                  vec![Some((0.,1.)), Some((1.,2.))]);
        test_clip(bb,
                  vec![[-1.,0.], [2.,3.], [4.,1.]],
                  vec![Some((0.,1.)), Some((1.,2.)), None, Some((3., 2.))]);
        test_clip(bb,
                  vec![[0.,1.], [2.,3.], [4.,1.], [2.,1.], [2., -1.],
                       [0., -1.], [0., 1.] ],
                  vec![Some((0.,1.)), Some((1.,2.)), None,
                       Some((3., 2.)), None,
                       Some((3., 1.)), Some((2., 1.)), Some((2., 0.)), None,
                       Some((0., 0.)), Some((0., 1.))]);
    }

    #[test]
    fn clip_empty() {
        let bb = BoundingBox {xmin: 0., xmax: 1., ymin: 0., ymax: 1.};
        let path = xy_of_sampling(&Sampling::empty().clip(bb));
        assert_eq!(path, vec![]);
    }

    #[test]
    fn clip_double_cut() {
        let bb = BoundingBox { xmin: 0.,  xmax: 4., ymin: 0.,  ymax: 2.};
        test_clip(bb,
                  vec![[1., 2.], [2.,3.], [5.,0.], [-1., 0.] ],
                  vec![Some((1., 2.)), None,
                       Some((3., 2.)), Some((4., 1.)), None,
                       Some((4., 0.)), Some((0., 0.)) ]);
    }

    #[test]
    fn clip_almost_horiz() {
        let bb = BoundingBox { xmin: 0.,  xmax: 2., ymin: -1.,  ymax: 1.};
        test_clip(bb,
                  vec![[1., 1e-100], [3., -1e-100] ],
                  vec![Some((1., 1e-100)), Some((2., 0.))]);
    }

    #[test]
    fn clip_touches_1pt_cut() {
        let bb = BoundingBox {xmin: 0., xmax: 2., ymax: 0., ymin: -1.};
        let cut = [f64::NAN, f64::NAN];
        test_clip(bb,
                  vec![[0.,1.], cut, [1., 0.], cut, cut, [2., 1.]],
                  vec![Some((1., 0.))])
    }

    #[test]
    fn clip_final_cut() {
        let bb = BoundingBox {xmin: 0., xmax: 1., ymin: 0., ymax: 2.};
        test_clip(bb,
                  vec![[0., 0.], [2., 2.]],
                  vec![Some((0., 0.)), Some((1., 1.))])
    }

    #[test]
    fn uniform1() {
        let s = Sampling::uniform(|x| x, 0., 4.).n(3)
            .init(&[1.]).init_pt(&[(3., 0.)]).build();
        assert_eq!(xy_of_sampling(&s),
                   vec![Some((0.,0.)), Some((1.,1.)), Some((2.,2.)),
                        Some((3., 0.)), Some((4.,4.))]);
    }

    #[test]
    fn uniform2() {
        let s = Sampling::uniform(|x| (4. - x).sqrt(), 0., 6.).n(4).build();
        assert_eq!(xy_of_sampling(&s),
                   vec![Some((0.,2.)), Some((2., 2f64.sqrt())),
                        Some((4., 0.)), None]);
    }

    /// In order the judge the quality of the sampling, we save it
    /// with the internal cost data.
    fn write_with_point_costs(s: &Sampling, fname: impl AsRef<Path>) -> R<()> {
        let mut fh = File::create(fname)?;
        for p in s.path.iter() {
            if p.is_valid() {
                let [x, y] = p.xy;
                writeln!(fh, "{x} {y} {}", p.cost)?;
            } else {
                writeln!(fh)?;
            }
        }
        Ok(())
    }

    fn write_segments(mut s: Sampling, fname: impl AsRef<Path>) -> R<()> {
        let mut fh = File::create(fname)?;
        let mut seg: Vec<(f64, Point, Point, f64)> = vec![];
        loop {
            let priority = s.pq.max_priority();
            if let Some(p0) = s.pq.pop() {
                let p1 = unsafe { p0.next().unwrap() };
                let p1 = unsafe { p1.as_ref() };
                let p0 = unsafe { p0.as_ref() };
                let tm = (p0.t + p1.t) / 2.;
                seg.push((tm, p0.clone(), p1.clone(), priority))
            } else {
                break;
            }
        }
        seg.sort_by(|(t1,_,_,_), (t2,_,_,_)| t1.partial_cmp(t2).unwrap());
        for (tm, p0, p1, priority) in seg {
            let [x0, y0] = p0.xy;
            let [x1, y1] = p1.xy;
            writeln!(fh, "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                     tm, p0.t, x0, y0,  p1.t, x1, y1, priority)?;
        }
        Ok(())
    }

    #[derive(Clone)]
    struct Plot {
        xmin: f64,  xmax: f64,
        ymin: f64,  ymax: f64,
        n: usize,
        init: Vec<f64>,
    }

    impl Plot {
        /// Return the Gnuplot instructions to plot the data.
        fn plot<F>(&self, f: F, title: &str) -> R<String>
        where F: FnMut(f64) -> f64 {
            let vp = BoundingBox { xmin: self.xmin, xmax: self.xmax,
                                   ymin: self.ymin, ymax: self.ymax };
            let s = Sampling::fun(f, self.xmin, self.xmax)
                .n(self.n).init(self.init.iter()).viewport(vp).build();
            static mut NDAT: usize = 0;
            let ndat = unsafe { NDAT += 1;  NDAT };
            let dir = Path::new("target");
            let fname = format!("horror{}.dat", ndat);
            s.write(&mut File::create(dir.join(&fname))?)?;
            let fname_p = format!("horror{}_p.dat", ndat);
            write_with_point_costs(&s, dir.join(&fname_p))?;
            let fname_s = format!("horror{}_s.dat", ndat);
            write_segments(s, dir.join(&fname_s))?;
        Ok(format!(
            "unset title\n\
             unset y2tics\n\
             plot [{}:{}] \"{}\" with l lt 5 title \"{}\", \
             \"{}\" with p lt 3 pt 6 ps 0.2 title \"n={}\"\n\
             set title \"Restricted to viewport [{}:{}]×[{}:{}]\"\n\
             set y2tics\n\
             set y2range [-1e-6:]\n\
             plot [{}:{}] [{}:{}] \"{}\" with l lt 5 title \"{}\", \
             \"{}\" with p lt 3 pt 7 ps 0.2 title \"n={}\", \
             \"{}\" using 1:3  with lp ps 0.2 lt rgb \"#760b0b\" \
             title \"cost points\", \
             \"{}\" using 1:8 with lp ps 0.2 lt rgb \"#909090\" \
             axes x1y2 title \"cost segments\"\n",
            self.xmin, self.xmax, &fname, title, &fname, self.n,
            self.xmin, self.xmax, self.ymin, self.ymax,
            self.xmin, self.xmax, self.ymin, self.ymax, &fname, title,
            &fname, self.n, &fname_p, &fname_s))
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)] // very slow under Miri
    fn horror() -> R<()> {
        let d = Plot {
            xmin: -5., xmax: 5., ymin: -5., ymax: 5., n: 100, init: vec![] };
        macro_rules! p {
            ($($id:ident $l:tt $e:expr),*) => {
                Plot{ $($id: $e,)* ..d.clone() } };
        }
        let s = [
            p!(n: 10).plot(|_| 2., "x ↦ 2")?,
            p!().plot(|x| x, "x ↦ x")?,
            p!().plot(|x| 5. * x, "x ↦ 5x")?,
            p!().plot(|x| 1e6 * x, "x ↦ 10⁶ x")?, // high slope
            p!().plot(|x| 1e50 * x, "x ↦ 10⁵⁰ x")?, // high slope
            p!().plot(|x| 1. / x, "x ↦ 1/x")?, // check singularity
            p!(xmin: 0., xmax: 5., ymax: 100.).plot(
                |x| 1. / x, "x ↦ 1/x")?, // singularity at starting point
            p!(xmin: -0.4, xmax: 2., ymin: 0., ymax: 1.6, n: 50).plot(
                |x| x.sqrt(), "x ↦ √x")?,
            // Test cuts also to the right:
            p!(xmin: -2., xmax: 1., ymin: 0., ymax: 1.6, n: 50).plot(
                |x| (-x).sqrt(), "x ↦ √(-x)")?,
            p!(n: 200).plot(|x| x.tan(), "tan")?,
            p!().plot(|x| 1. / x.abs(), "x ↦ 1/|x|")?,
            p!(xmin: -6., xmax: 6., ymin: -2., ymax: 2.).plot(
                |x| (1. + x.cos().sin()).ln(), "1 + sin(cos x)")?,
            p!(xmin: 0., xmax: 6.28, ymin: -1.5, ymax: 1.5, n: 400).plot(
                |x| x.powi(3).sin() + x.powi(3).cos(), "sin x³ + cos x³")?,
            p!(xmin: -5., xmax:200., ymin: -1., ymax: 1., n: 400).plot(
                |x| x.sin(), "sin")?,
            // Examples from R. Avitzur, O. Bachmann, N. Kajler, "From
            // Honest to Intelligent Plotting", proceedings of ISSAC'
            // 95, pages 32-41, July 1995.
            p!(xmin: -4., xmax: 4., ymin: -1., ymax: 1.).plot(
                |x| (300. * x).sin(), "sin(300 x)")?,
            p!(xmin: -4., xmax: 4., ymin: -1., ymax: 1., n: 1000).plot(
                |x| (300. * x).sin(), "sin(300 x)")?,
            p!(xmin: -2., xmax: 2., ymin: 0., ymax: 3.).plot(
                |x| 1. + x * x + 0.0125 * (1. - 3. * (x - 1.)).abs().ln(),
                "1 + x² + 0.0125 ln|1 - 3(x-1)|")?,
            p!(xmin: -2., xmax: 2., ymin: 0., ymax: 3., n: 300,
               init:vec![4. / 3.]).plot(
                |x| 1. + x * x + 0.0125 * (1. - 3. * (x - 1.)).abs().ln(),
                "1 + x² + 0.0125 ln|1 - 3(x-1)| (specifying x:4/3")?,
            p!(xmin: -0.5, xmax: 0.5, ymin: -1., ymax: 1.).plot(
                |x| x * (1. / x).sin(), "x sin(1/x)")?,
            p!(xmin: -0.5, xmax: 0.5, ymin: -1., ymax: 1., n:200).plot(
                |x| x * (1. / x).sin(), "x sin(1/x)")?,
            p!(xmin: -2., xmax: 2., ymin: -1., ymax: 1.).plot(
                |x| (1. / x).sin(), "sin(1/x)")?,
            p!(xmin: -2., xmax: 2., ymin: -1., ymax: 1., n: 400).plot(
                |x| (1. / x).sin(), "sin(1/x)")?,
            p!(xmin: -4., xmax: 4., ymin: -1., ymax: 1.).plot(
                |x| x.powi(4).sin(), "sin(x⁴)")?,
            p!(xmin: -4., xmax: 4., ymin: -1., ymax: 1., n: 600).plot(
                |x| x.powi(4).sin(), "sin(x⁴)")?,
            p!(xmin: -6., xmax: 6., ymin: -1., ymax: 1.).plot(
                |x| x.exp().sin(), "sin(exp x)")?,
            p!(xmin: -6., xmax: 6., ymin: -1., ymax: 1., n: 500).plot(
                |x| x.exp().sin(), "sin(exp x)")?,
            p!(xmin: -10., xmax: 10., ymin: 0., ymax: 10.).plot(
                |x| 1. / x.sin(), "1 / sin x")?,
            p!(xmin: -6., xmax: 6., ymin: 0., ymax: 2.).plot(
                |x| x.sin() / x, "(sin x)/x")?,
            p!(xmin: -2., xmax: 2., ymin: -15., ymax: 15.).plot(
                |x| (x.powi(3) - x + 1.).tan() + 1. / (x + 3. * x.exp()),
                "tan(x³ - x + 1) + 1/(x + 3 eˣ)")?,
            p!(xmin: 0., xmax: 17., ymin: 0., ymax: 2.).plot(
                |x| (1. + x.cos()) * (-0.1 * x).exp(),
                "(1 + cos x) exp(-x/10)")?,
            p!(xmin: -2., xmax: 17., ymin: 0., ymax: 2.).plot(
                |x| (1. + x.cos()) * (-0.1 * x).exp(),
                "(1 + cos x) exp(-x/10)")?,
            p!(xmin: 0., xmax: 17., ymin: 0., ymax: 2.).plot(
                |x| (1. + x.cos()) * (-0.01 * x * x).exp(),
                "(1 + cos x) exp(-x²/100)")?,
        ].join("");
        let mut fh = File::create("target/horror.gp")?;
        write!(fh, "set terminal pdfcairo\n\
                    set output \"horror.pdf\"\n\
                    set grid\n\
                    {}\n", s)?;
        Ok(())
    }


}
