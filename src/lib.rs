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
          io::{self, Write},
          iter::Iterator,
          mem::swap};
use rgb::*;

mod priority_queue;
use priority_queue as pq;

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
    fn contains(&self, p: TXY) -> bool {
        self.xmin <= p.x && p.x <= self.xmax
            && self.ymin <= p.y && p.y <= self.ymax
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
    // The path is given by `t`, `x`, `y` in the order of the vectors.
    // A point on the path is just an index to `t`, `x`, `y`, `cost`
    // and `witness` (these vectors all have the same length).
    // Invariants:
    // - `t` is made only of finite f64,
    // - `y[p]` is finite ⟹ `x[p]` is also finite.
    t: Vec<f64>,
    x: Vec<f64>,
    y: Vec<f64>,
    // `cost[p]` cache the cost of the point `p` (not the segment).
    // If the point is not valid, the cost has no meaning.
    cost: Vec<f64>,
    // Pointer to the priority queue node if the point is the initial
    // point of a segment (in order to be able to increase the
    // priority of segments).  `None` otherwise.
    witness: Vec<Option<pq::Witness<usize>>>,
    vp: Option<BoundingBox>, // viewport (zone of interest)
}

/// Priority queue storing indices to points.
type PQ = pq::PQ<usize>;

impl Clone for Sampling {
    fn clone(&self) -> Self {
        // It is guessed that one clones the sampling to transform it.
        // Thus start with an empty queue.
        let mut witness = Vec::with_capacity(self.witness.len());
        for _ in 0 .. self.witness.len() { witness.push(None) }
        Self { pq: PQ::new(),
               t: self.t.clone(),
               x: self.x.clone(),
               y: self.y.clone(),
               cost: self.cost.clone(),
               witness,
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

/// A 2D point with coordinates (`x`, `y`) supposed to be given by a
/// function evaluated at "time" `t`.  A path is a sequence of points.
/// Points that are invalid (see [`Point::is_valid`]) are considered
/// cuts in the path (the line is interrupted).  As most 2 cuts may
/// follow each other (to track ranges of t outside the domain of the
/// function) except at the boundary of the path where at most 1 cut
/// is allowed.
#[derive(Debug, Clone, Copy)]
struct TXY {
    t: f64,
    x: f64,
    y: f64,
}

impl TXY {
    fn cut(t: f64) -> Self { Self {t, x: f64::NAN, y: f64::NAN } }

    fn is_valid(self) -> bool { self.y.is_finite() }
}

impl Sampling {
    /// Return `true` if the sampling contains no point.
    #[inline]
    pub fn is_empty(&self) -> bool { self.t.is_empty() }

    /// Return the number of points and cuts in the sampling.
    pub fn len(&self) -> usize { self.t.len() }

    /// Create an empty sampling.
    #[inline]
    pub(crate) fn empty() -> Self {
        Self { pq: PQ::new(),
               t: Vec::new(),  x: Vec::new(),  y: Vec::new(),
               cost: Vec::new(),  witness: Vec::new(),
               vp: None }
    }

    #[inline]
    pub(crate) fn singleton(p: TXY) -> Self {
        debug_assert!(p.t.is_finite() && p.x.is_finite() && p.y.is_finite());
        Self { pq: PQ::new(), vp: None,
               t: vec![p.t],  x: vec![p.x],  y: vec![p.y],
               cost: vec![0.],  witness: vec![None]  }
    }

    fn set_cost(&mut self, index: usize, cost: f64) {
        debug_assert!(index < self.len());
        unsafe { *self.cost.get_unchecked_mut(index) = cost }
    }

    /// Return the coordinates `[t, x, y]` for the point of index `p`
    /// in the path.
    fn txy(&self, index: usize) -> TXY {
        debug_assert!(index < self.len());
        unsafe { TXY { t: *self.t.get_unchecked(index),
                       x: *self.x.get_unchecked(index),
                       y: *self.y.get_unchecked(index) } }
    }

    /// Add a new point at the end of the sampling.  `t` is assumed to
    /// be finite.
    #[inline]
    fn push_unchecked(&mut self, p: TXY) {
        // This does not change the existing indices in the queue.
        self.t.push(p.t);
        self.x.push(p.x);
        self.y.push(p.y);
        self.cost.push(0.); // Dummy cost
        self.witness.push(None);
    }

    #[inline]
    fn push_cut(&mut self, t: f64) {
        self.t.push(t);
        self.x.push(f64::NAN);
        self.y.push(f64::NAN);
        self.cost.push(0.); // Dummy cost
        self.witness.push(None);
    }

    /// Insert `p` at position `index` in the path, shifting all
    /// elements after it to the right.
    fn insert(&mut self, index: usize, p: TXY) {
        self.t.insert(index, p.t);
        self.x.insert(index, p.x);
        self.y.insert(index, p.y);
        self.cost.insert(index, 0.);
        self.witness.insert(index, None);
        // Increase indices ≥ `index`.
        self.pq.forall(|i| if *i >= index { *i += 1 })
    }

    /// Replace the element of the sampling at `index` with `p`.
    /// Does not change the cost not the priority queue witness.
    fn replace(&mut self, index: usize, p: TXY) {
        debug_assert!(index < self.len());
        unsafe {
            *self.t.get_unchecked_mut(index) = p.t;
            *self.x.get_unchecked_mut(index) = p.x;
            *self.y.get_unchecked_mut(index) = p.y;
            // *self.cost.get_unchecked_mut(index) = 0.;
            debug_assert!(self.witness[index].is_none());
            // *self.witness.get_unchecked_mut(index) = None;
        }
    }

    /// Remove the last element of the path.
    fn pop(&mut self) {
        self.t.pop();
        self.x.pop();
        self.y.pop();
        self.cost.pop();
        self.witness.pop();
    }

    /// Push the `index` of the path to the priority queue with
    /// `priority` and record its witness.
    fn push_to_pq(&mut self, priority: f64, index: usize) {
        debug_assert!(self.witness[index].is_none());
        let w = self.pq.push(priority, index);
        self.witness[index] = Some(w);
    }

    /// Return and removes the index of the path of the priority queue
    /// with the higher priority.  Make sure its witness is reset.
    fn pop_from_pq(&mut self) -> Option<usize> {
        self.pq.pop().and_then(|i| { self.witness[i] = None;  Some(i) })
    }

    #[inline]
    pub(crate) fn set_vp(&mut self, bb: BoundingBox) {
        self.vp = Some(bb);
    }

    /// Return the length of the "time interval" as well as the
    /// lengths of the viewport.
    pub(crate) fn len_txy(&self) -> Option<Lengths> {
        if self.is_empty() { return None }
        let t0 = self.t[0];
        let t1 = self.t.last().unwrap();
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
        Some(Lengths { t: (t1 - t0).abs(), x: len_x, y: len_y })
    }

    /// Iterate on the points (and cuts) of the path.  More precisely,
    /// a path is made of continuous segments whose points are given
    /// by contiguous values `[x,y]` with both `x` and `y` not NaN,
    /// interspaced by "cuts" `[f64::NAN; 2]`.  Two cuts never follow
    /// each other.  Isolated points `p` are given by ... `[f64::NAN;
    /// 2]`, `p`, `None`,...
    pub fn iter(&self) -> impl Iterator<Item = [f64; 2]> + '_ {
        // FIXME: Do we want to remove 2 cuts following each other?
        let mut prev_is_cut = false;
        self.x.iter().zip(&self.y).filter_map(move |(&x, &y)| {
            if y.is_finite() { // valid point
                prev_is_cut = false;
                Some([x, y])
            } else if prev_is_cut {
                // Do not issue several cuts following each other.
               None
            } else {
                prev_is_cut = true;
                Some([f64::NAN; 2])
            }})
    }

    /// Slice of the x-coordinates of the sampling.
    /// See [`Self::iter`] for more information.
    // FIXME: there may be 2 cuts (NAN) following each other.
    #[inline]
    pub fn x(&self) -> &[f64] { &self.x }

    /// Iterator on the y-coordinates of the sampling.
    /// See [`Self::iter`] for more information.
    #[inline]
    pub fn y(&self) -> &[f64] { &self.y }

    // #[inline]
    // pub fn iter_mut(&mut self)
    //                 -> impl Iterator<Item = &mut Option<[f64; 2]>> + '_ {
    //     todo!()
    // }
}

/// Intersection of a segment with the bounding box.
#[derive(Debug)]
enum Intersection {
    Empty,
    Pt(TXY),
    Seg(TXY, TXY),
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
        swap(&mut self.x, &mut self.y);
        self
    }

    /// Assume `p0` ∈ `bb` and `p1` ∉ `bb`.  Return the point
    /// intersecting the boundary of `bb`.  If the intersection point
    /// is the same as `p0`, return `None`
    #[inline]
    #[must_use]
    fn intersect(p0: TXY, p1: TXY, bb: BoundingBox) -> Option<TXY> {
        let mut t = 1.; // t ∈ [0, 1]
        let dx = p1.x - p0.x; // May be 0.
        let r = (if dx >= 0. {bb.xmax} else {bb.xmin} - p0.x) / dx;
        if r < t { t = r } // ⟹ r finite (as r ≥ 0 or NaN)
        let dy = p1.y - p0.y; // May be 0.
        let r = (if dy >= 0. {bb.ymax} else {bb.ymin} - p0.y) / dy;
        if r < t { t = r };
        if t <= 1e-14 {
            None
        } else {
            Some(TXY { t: p0.t + t * (p1.t - p0.t),
                       x: p0.x + t * dx,
                       y: p0.y + t * dy })
        }
    }

    /// Assume `p0` ∉ `bb` and `p1` ∉ `bb` (thus, not even on the
    /// boundary of `bb`) and `p0` ≠ `p1`.  Return the endpoints of
    /// the segment intersecting the boundary of `bb` if any.  The
    /// "parameter direction" of `p0`, `p1` is preserved.
    #[inline]
    #[must_use]
    fn intersect_seg(p0: TXY, p1: TXY, bb: BoundingBox) -> Intersection {
        // Convex combination of `p0` and `p1` parametrized by `s`.
        let mut s0 = 0.; // s0 ∈ [0, 1]
        let mut s1 = 1.; // s1 ∈ [0, 1]
        let dx = p1.x - p0.x; // may be 0.
        let r0;
        let r1; // r0 ≤ r1 or NAN if on x-boundary lines.
        if dx >= 0. {
            r0 = (bb.xmin - p0.x) / dx;
            r1 = (bb.xmax - p0.x) / dx;
        } else {
            r0 = (bb.xmax - p0.x) / dx;
            r1 = (bb.xmin - p0.x) / dx;
        }
        if r0 > 1. || r1 < 0. { return Intersection::Empty }
        if r0 > 0. { s0 = r0 } // if r0 is NAN, keep the whole segment
        if r1 < 1. { s1 = r1 }
        let dy = p1.y - p0.y; // may be 0.
        let r0;
        let r1;
        if dy >= 0. {
            r0 = (bb.ymin - p0.y) / dy;
            r1 = (bb.ymax - p0.y) / dy;
        } else {
            r0 = (bb.ymax - p0.y) / dy;
            r1 = (bb.ymin - p0.y) / dy;
        }
        if r0 > s1 || r1 < s0 { return Intersection::Empty }
        if r0 > s0 { s0 = r0 }
        if r1 < s1 { s1 = r1 }
        if s0 < s1 { // segment not reduced to a point
            let dt = p1.t - p0.t;
            let q0 = TXY { t: p0.t + s0 * dt,
                           x: p0.x + s0 * dx,
                           y: p0.y + s0 * dy };
            let q1 = TXY { t: p0.t + s1 * dt,
                           x: p0.x + s1 * dx,
                           y: p0.y + s1 * dy };
            Intersection::Seg(q0, q1)
        } else if s0 == s1 {
            let q0 = TXY { t: p0.t + s0 * (p1.t - p0.t),
                           x: p0.x + s0 * dx,
                           y: p0.y + s0 * dy };
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
        // First point of the current segment, if any.
        let mut p0_opt: Option<TXY> = None;
        let mut p0_inside = false;
        let mut prev_cut = true; // New path ("virtual" cut)
        for ((&t1, &x1), &y1) in self.t.iter()
                                 .zip(&self.x).zip(&self.y) {
            let p1 = TXY { t: t1, x: x1, y: y1 };
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
                            s.push_unchecked(p0);
                            if p1_inside {
                                s.push_unchecked(p1);
                                prev_cut = false;
                            } else if
                                let Some(p) = Self::intersect(p0, p1, bb) {
                                    s.push_unchecked(p);
                                    s.push_cut(p.t);
                                } else {
                                    s.push_cut(p0.t);
                                }
                        } else if p1_inside { // p0 ∉ bb, p1 ∈ bb
                            if let Some(p) = Self::intersect(p1, p0, bb) {
                                s.push_unchecked(p); // p ≠ p1
                            }
                            s.push_unchecked(p1);
                            prev_cut = false;
                        } else { // p0, p1 ∉ bb but maybe intersection
                            match Self::intersect_seg(p0, p1, bb) {
                                Intersection::Seg(q0, q1) => {
                                    s.push_unchecked(q0);
                                    s.push_unchecked(q1);
                                    s.push_cut(q1.t);
                                }
                                Intersection::Pt(p) => {
                                    s.push_unchecked(p);
                                    s.push_cut(p.t);
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
                            s.push_unchecked(p0);
                            s.push_cut(p0.t);
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
                        s.push_unchecked(p1);
                    } else { // p0 ∈ bb, p1 ∉ bb
                        if let Some(p) = Self::intersect(p0, p1, bb) {
                            s.push_unchecked(p);
                            s.push_cut(p.t);
                        } else {
                            s.push_cut(p0.t);
                        }
                        prev_cut = true;
                    }
                } else { // p1 is invalid (i.e., represent a cut)
                    p0_opt = None;
                    s.push_unchecked(p1);
                    prev_cut = true
                }
            }
        }
        if prev_cut { s.pop() }
        s.set_vp(bb);
        s
    }

    /// Create a sampling from an iterator of points.  Beware that the
    /// invariant "`p.y` is finite ⇒ `p.x` is finite" is not checked.
    fn from_point_iterator<P>(points: P) -> Self
    where P: IntoIterator<Item = TXY> {
        let mut s = Sampling::empty();
        let mut points = points.into_iter();
        macro_rules! skip_until_last_cut { () => {
            let mut cut = None;
            let mut first_pt = None;
            for p in &mut points {
                if p.is_valid() { first_pt = Some(p); break; }
                cut = Some(p);
            }
            match (cut, first_pt) {
                (_, None) => return s,
                (None, Some(p)) => {
                    s.push_unchecked(p);
                }
                (Some(c), Some(p)) => {
                    s.push_cut(c.t);
                    s.push_unchecked(p);
                }
            }
        }}
        skip_until_last_cut!();
        while let Some(p) = points.next() {
            if p.is_valid() {
                s.push_unchecked(p);
            } else {
                s.push_cut(p.t);
                skip_until_last_cut!();
            }
        }
        s
    }

    /// Create a sampling from `points` after sorting them by
    /// increasing (if `incr`) or decreasing (if `! incr`) values of
    /// the field `t`.  Beware that the invariant "`p.y` is finite ⇒
    /// `p.x` is finite" is not checked.
    fn from_vec(mut points: Vec<TXY>, incr: bool) -> Self {
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
            points.into_iter().enumerate().map(|(i, [x, y])| {
                let t = i as f64;
                if x.is_finite() { TXY {t, x, y} }
                else { TXY::cut(t) } }))
    }
}

////////////////////////////////////////////////////////////////////////
//
// Acceptable types & functions that provide "points" `TXY`.
// These "conversions" must enforce the specification: `p.x` finite ⟹
// `p.y` finite.
// This is internal to this library (to handle graphs of functions
// and parametric graphs with a single generic code).

impl From<(f64, f64)> for TXY {
    #[inline]
    fn from((x, y): (f64, f64)) -> Self {
        TXY { t: x, x, y } // `x` ∈ [a, b] by `init_pt` checks.
    }
}

impl From<(f64, [f64; 2])> for TXY {
    /// Assume `t` is finite.
    #[inline]
    fn from((t, [x,y]): (f64, [f64;2])) -> Self {
        // Enforce the invariant: y finite ⟹ x finite
        if x.is_finite() { TXY {t, x, y} }
        else { TXY::cut(t) }
    }
}

/// Values that can be treated as Fn(f64) -> TXY
trait IntoFnTXY {
    fn eval(&mut self, t: f64) -> TXY;
}

// This trait cannot implemented for both `FnMut(f64) -> f64` and
// `FnMut(f64) -> [f64; 2]` (that conflicts), so we wrap the types of
// interest.

struct FnTXY<T>(T);

impl<T> IntoFnTXY for FnTXY<T> where T: FnMut(f64) -> f64 {
    #[inline]
    fn eval(&mut self, t: f64) -> TXY {
        TXY {t, x: t, y: self.0(t)}
    }
}

struct ParamTXY<T>(T);

impl<T> IntoFnTXY for ParamTXY<T> where T: FnMut(f64) -> [f64; 2] {
    #[inline]
    fn eval(&mut self, t: f64) -> TXY {
        let [x, y] = self.0(t);
        // `TXY::is_valid()` only checks `y`; make sure non-finite
        // `x` leads to an invalid point.
        if x.is_finite() { TXY {t, x, y} }
        else { TXY {t, x, y: f64::NAN} }
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
            init_pt: Vec<TXY>,
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
                        self.init_pt.push(TXY::from(p));
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
    FnTXY);

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
    use super::{TXY, Lengths};

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

    /// Return the cost of the middle point `pm`.  Assumes `p0`, `pm`,
    /// and `p1` are valid points.
    #[inline]
    pub(crate) fn middle(p0: TXY, pm: TXY, p1: TXY, len: Lengths) -> f64 {
        let dx0m = (p0.x - pm.x) / len.x;
        let dy0m = (p0.y - pm.y) / len.y;
        let dx1m = (p1.x - pm.x) / len.x;
        let dy1m = (p1.y - pm.y) / len.y;
        let len0m = dx0m.hypot(dy0m);
        let len1m = dx1m.hypot(dy1m);
        if len0m == 0. || len1m == 0. {
            0. // Do not subdivide
        } else {
            let dx = - dx0m * dx1m - dy0m * dy1m;
            let dy = dy0m * dx1m - dx0m * dy1m;
            let cost = dy.atan2(dx); // ∈ [-π, π]
            debug_assert!(!cost.is_nan());
            cost
        }
    }

    /// Compute the cost of a segment according to the costs of its
    /// endpoints unless `in_vp` is false in which case -∞ is returned.
    #[inline]
    pub(crate) fn segment_vp(p0: TXY, cost0: f64, p1: TXY, cost1: f64,
                             len: Lengths, in_vp: bool) -> f64 {
        if ! in_vp { return f64::NEG_INFINITY }
        segment(p0, cost0, p1, cost1, len)
    }

    /// Compute the cost of a segment according to the costs of its
    /// endpoints.
    #[inline]
    pub(crate) fn segment(p0: TXY, cost0: f64, p1: TXY, cost1: f64,
                          len: Lengths) -> f64 {
        let dt = (p1.t - p0.t) / len.t; // ∈ [0,1]
        debug_assert!((0. ..=1.).contains(&dt));
        // Put less efforts when `dt` is small.  For functions, the
        // Y-variation may be large but, if it happens for a small range
        // of `t`, there is no point in adding indistinguishable details.
        let dx = ((p1.x - p0.x) / len.x).abs();
        let dy = ((p1.y - p0.y) / len.y).abs();
        let mut cost = cost0.abs() + cost1.abs();
        if cost0 * cost1 < 0. {
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

impl Sampling {
    /// Compute the cost of the segment \[`p0`, `p1`\] (taking into
    /// account `in_vp`) and push it to the queue `pq`.  `p0` is at
    /// index `i0` and `p1` at `i0 + 1` (next point in the path).
    /// `p0` is updated with the PQ-witness.
    fn push_segment(&mut self, i0: usize, p0: TXY, p1: TXY,
                    len: Lengths, in_vp: bool) {
        // FIXME: do we really want to push the segment when `!in_vp`?
        // In not all points are in the queue, one must beware of the
        // validity of witnesses though.
        let cost0 = self.cost[i0];
        let cost1 = self.cost[i0 + 1];
        let cost_seg = cost::segment_vp(p0, cost0, p1, cost1, len, in_vp);
        // The segment is referred to by its first point.
        self.push_to_pq(cost_seg, i0);
    }

    /// With the (new) costs in points `p0` and `p1`, update the position
    /// of the segment \[`p0`, `p1`\] in the priority queue of `self`.
    /// `p0` is at position `i0` and `p1` at position `i0 + 1`.
    ///
    /// # Safety
    /// `p0` must be in `pq`, otherwise it is UB.
    fn update_segment(&mut self, i0: usize, p0: TXY, p1: TXY, len: Lengths) {
        match &self.witness[i0] {
            Some(w) => {
                let cost0 = self.cost[i0];
                let cost1 = self.cost[i0 + 1];
                let priority = cost::segment(p0, cost0, p1, cost1, len);
                unsafe { self.pq.increase_priority(w, priority) }
            }
            None => panic!("Sampling::update_segment: unset witness"),
        }
    }
}

////////////////////////////////////////////////////////////////////////
//
// Function sampling

/// Update the cost of all points in the sampling and add segments
/// to the priority queue.
fn compute(s: &mut Sampling, in_vp: impl Fn(TXY) -> bool) {
    if let Some(len) = s.len_txy() {
        if s.len() <= 2 { return }
        // Path has at least 2 elements.
        let mut i0 = 0;
        let mut p0 = s.txy(i0);
        s.cost[i0] = 0.;
        let mut p0_is_valid = p0.is_valid();
        let mut p0_in_vp = p0_is_valid && in_vp(p0);
        let mut im = 1;
        let mut pm = s.txy(im);
        for i1 in 2 .. s.len() {
            let p1 = s.txy(i1);
            let pm_is_valid = pm.is_valid();
            let pm_in_vp;
            if pm_is_valid {
                pm_in_vp = in_vp(pm);
                if p0_is_valid && p1.is_valid() {
                    s.set_cost(im, cost::middle(p0, pm, p1, len));
                } else {
                    s.set_cost(im, cost::HANGING_NODE);
                }
            } else { // pm is the location of a cut
                pm_in_vp = false;
                s.set_cost(im, 0.);
            }
            if p0_is_valid || pm_is_valid {
                // Add segment [p0, pm] to the PQ and set `p0` witness.
                s.push_segment(i0, p0, pm, len, p0_in_vp || pm_in_vp);
            }
            i0 = im;
            p0 = pm;
            p0_is_valid = pm_is_valid;
            p0_in_vp = pm_in_vp;
            im = i1;
            pm = p1;
        }
        s.set_cost(im, 0.); // last point
        if p0_is_valid || pm.is_valid() {
            let vp = p0_in_vp || (pm.is_valid() && in_vp(pm));
            s.push_segment(i0, p0, pm, len, vp);
        }
    }
}

fn refine_gen(s: &mut Sampling, n: usize,
              mut f: impl FnMut(f64) -> TXY,
              in_vp: impl Fn(TXY) -> bool) {
    let len = match s.len_txy() {
        Some(txy) => txy,
        None => return };
    for _ in 0 .. n {
        let i0: usize = match s.pop_from_pq() { None => break,
                                                Some(p) => p };
        let p0 = s.txy(i0);
        let i1 = i0 + 1;
        let p1 = s.txy(i1);
        // Refine the segment [p0, p1] inserting a middle point `pm`.
        let t = (p0.t + p1.t) * 0.5; // FIXME: better middle point
        let pm = f(t);
        if p0.is_valid() {
            if p1.is_valid() {
                let im = i1;
                s.insert(im, pm);
                let i1 = im + 1; // Point `p1` shifted right.
                let mut pm_in_vp = false;
                if pm.is_valid() {
                    pm_in_vp = in_vp(pm);
                    s.set_cost(im, cost::middle(p0, pm, p1, len));
                    if i0 > 0 {
                        let i_1 = i0 - 1;
                        let p_1 = s.txy(i_1);
                        if p_1.is_valid() {
                            let c = cost::middle(p_1, p0, pm, len);
                            s.set_cost(i0, c);
                            s.update_segment(i_1, p_1, p0, len)
                        }
                    }
                    let i2 = i1 + 1;
                    if i2 < s.len() {
                        let p2 = s.txy(i2);
                        if p2.is_valid() {
                            let c = cost::middle(pm, p1, p2, len);
                            s.set_cost(i1, c);
                            s.update_segment(i1, p1, p2, len)
                        }
                    }
                } else { // `pm` invalid ⟹ cut between `p0` and `p1`
                    s.set_cost(i0, cost::HANGING_NODE);
                    // s.set_cost(im, 0.); // 0 = default
                    s.set_cost(i1, cost::HANGING_NODE);
                    if i0 > 0 {
                        let i_1 = i0 - 1;
                        let p_1 = s.txy(i_1);
                        s.update_segment(i_1, p_1, p0, len)
                    }
                    let i2 = i1 + 1;
                    if i2 < s.len() {
                        let p2 = s.txy(i2);
                        s.update_segment(i1, p1, p2, len)
                    }
                }
                s.push_segment(i0, p0, pm, len, pm_in_vp || in_vp(p0));
                s.push_segment(im, pm, p1, len, pm_in_vp || in_vp(p1));
            } else { // `p0` valid, `p1` invalid (i.e. is a cut)
                // Thus `p0` is a hanging node.
                if pm.is_valid() {
                    let im = i1; // `i1` no longer read
                    s.insert(im, pm);
                    s.set_cost(im, cost::HANGING_NODE);
                    if i0 > 0 {
                        let i_1 = i0 - 1;
                        let p_1 = s.txy(i_1);
                        if p_1.is_valid() {
                            let c = cost::middle(p_1, p0, pm, len);
                            s.set_cost(i0, c);
                            s.update_segment(i_1, p_1, p0, len)
                        }
                    }
                    let pm_in_vp = in_vp(pm);
                    let vp = pm_in_vp || in_vp(p0);
                    s.push_segment(i0, p0, pm, len, vp);
                    s.push_segment(im, pm, p1, len, pm_in_vp)
                } else { // `pm` invalid
                    // Insert only \[`p0`, `pm`\] and forget \[`pm`, `p1`\].
                    // The cost of `p0` stays `cost::HANGING_NODE`.  We can
                    // see this as reducing the uncertainty of the boundary
                    // in the segment \[`p0`, `p1`\].
                    let im = i1;
                    if s.witness[i1].is_none() {
                        // `p1` is not part of a segment.  One can replace
                        // it by `pm`.
                        s.replace(im, pm);
                        s.set_cost(im, 0.);
                    } else {
                        s.insert(im, pm); // ⟹ cost = 0
                    }
                    s.push_segment(i0, p0, pm, len, in_vp(p0))
                }
            }
        } else { // `p0` invalid (i.e., cut) ⟹ `p1` valid
            debug_assert!(p1.is_valid());
            if pm.is_valid() {
                let im = i1;
                s.insert(im, pm);
                s.set_cost(im, cost::HANGING_NODE);
                let i1 = im + 1; // `p1` shifted right
                let i2 = i1 + 1;
                if i2 < s.len() {
                    let p2 = s.txy(i2);
                    if p2.is_valid() {
                        let c = cost::middle(pm, p1, p2, len);
                        s.set_cost(i1, c);
                        s.update_segment(i1, p1, p2, len)
                    }
                }
                let pm_in_vp = in_vp(pm);
                s.push_segment(i0, p0, pm, len, pm_in_vp);
                s.push_segment(im, pm, p1, len, pm_in_vp || in_vp(p1))
            } else { // `pm` invalid ⟹ drop segment \[`p0`, `pm`\].
                // Cost of `p1` stays `cost::HANGING_NODE`.
                let im = {
                    if i0 > 0 {
                        let i_1 = i0 - 1;
                        let p_1 = s.txy(i_1);
                        if p_1.is_valid() {
                            s.insert(i1, pm);
                            i1
                        } else {
                            // `p_1` is the cut ending the previous segment.
                            s.replace(i0, pm);
                            i0
                        }
                    } else {
                        s.insert(i1, pm);
                        i1
                    } };
                s.set_cost(im, 0.);
                s.push_segment(im, pm, p1, len, in_vp(p1))
            }
        }
    }
}

fn push_almost_uniform_sampling(points: &mut Vec<TXY>,
                                f: &mut impl FnMut(f64) -> TXY,
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
    fn build(mut points: Vec<TXY>,
             mut f: impl FnMut(f64) -> TXY,
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
                let in_vp = |p: TXY| vp.contains(p);
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
    FnTXY);

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
    ParamTXY);

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
    use crate::{Sampling, TXY, BoundingBox};

    type R<T> = Result<T, Box<dyn Error>>;

    fn assert_path(s: &Sampling, exact: &[[f64; 2]])  {
        if !s.iter().zip(exact).all(|(p, q)| {
            (p[0] == q[0] && p[1] == q[1])
                || (p[1].is_nan() && q[1].is_nan())}) {
            eprintln!("left: {:?}\n\
                       right: {:?}",
                      s.iter().collect::<Vec<_>>(), exact)
            }
    }

    const CUT: [f64; 2] = [f64::NAN; 2];

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
        assert_path(&s, &[[0.,0.], [1.,1.]]);
        let s = Sampling::from_iter([[0.,0.], [1.,1.],
                                     [f64::NAN, 1.], [2.,2.]]);
        assert_path(&s, &[[0.,0.], [1.,1.], CUT, [2.,2.]]);
    }

    #[test]
    fn bounding_box_singleton() {
        let s = Sampling::singleton(TXY{ t: 0., x: 1., y: 2. });
        let bb = BoundingBox {xmin: 1., xmax: 1., ymin: 2., ymax: 2.};
        assert_eq!(s.bounding_box(), bb);
    }

    fn test_clip(bb: BoundingBox,
                 path: &[[f64;2]],
                 expected: &[[f64; 2]]) {
        let s = Sampling::from_iter(path.to_owned()).clip(bb);
        assert_path(&s, expected);
    }

    #[test]
    fn clip_base () {
        let bb = BoundingBox { xmin: 0.,  xmax: 3., ymin: 0.,  ymax: 2.};
        test_clip(bb, &[[0.,1.], [2.,3.]],
                  &[[0.,1.], [1.,2.]]);
        test_clip(bb,
                  &[[-1.,0.], [2.,3.], [4.,1.]],
                  &[[0.,1.], [1.,2.], CUT, [3., 2.]]);
        test_clip(bb,
                  &[[0.,1.], [2.,3.], [4.,1.], [2.,1.], [2., -1.],
                    [0., -1.], [0., 1.] ],
                  &[[0.,1.], [1.,2.], CUT,
                    [3., 2.], CUT,
                    [3., 1.], [2., 1.], [2., 0.], CUT,
                    [0., 0.], [0., 1.]]);
    }

    #[test]
    fn clip_empty() {
        let bb = BoundingBox {xmin: 0., xmax: 1., ymin: 0., ymax: 1.};
        assert_path(&Sampling::empty().clip(bb),
                    &Vec::<[f64; 2]>::new());
    }

    #[test]
    fn clip_double_cut() {
        let bb = BoundingBox { xmin: 0.,  xmax: 4., ymin: 0.,  ymax: 2.};
        test_clip(bb,
                  &[[1., 2.], [2.,3.], [5.,0.], [-1., 0.] ],
                  &[[1., 2.], CUT,
                    [3., 2.], [4., 1.], CUT,
                    [4., 0.], [0., 0.] ]);
    }

    #[test]
    fn clip_almost_horiz() {
        let bb = BoundingBox { xmin: 0.,  xmax: 2., ymin: -1.,  ymax: 1.};
        test_clip(bb,
                  &[[1., 1e-100], [3., -1e-100] ],
                  &[[1., 1e-100], [2., 0.]]);
    }

    #[test]
    fn clip_touches_1pt_cut() {
        let bb = BoundingBox {xmin: 0., xmax: 2., ymax: 0., ymin: -1.};
        let cut = [f64::NAN, f64::NAN];
        test_clip(bb,
                  &[[0.,1.], cut, [1., 0.], cut, cut, [2., 1.]],
                  &[[1., 0.]])
    }

    #[test]
    fn clip_final_cut() {
        let bb = BoundingBox {xmin: 0., xmax: 1., ymin: 0., ymax: 2.};
        test_clip(bb,
                  &[[0., 0.], [2., 2.]],
                  &[[0., 0.], [1., 1.]])
    }

    #[test]
    fn uniform1() {
        let s = Sampling::uniform(|x| x, 0., 4.).n(3)
            .init(&[1.]).init_pt(&[(3., 0.)]).build();
        assert_path(&s,
                    &[[0., 0.], [1.,1.], [2.,2.], [3., 0.], [4.,4.] ]);
    }

    #[test]
    fn uniform2() {
        let s = Sampling::uniform(|x| (4. - x).sqrt(), 0., 6.).n(4).build();
        assert_path(&s,
                    &[[0., 2.], [2., 2f64.sqrt()], [4., 0.], CUT]);
    }

    /// In order the judge the quality of the sampling, we save it
    /// with the internal cost data.
    fn write_with_point_costs(s: &Sampling, fname: impl AsRef<Path>) -> R<()> {
        let mut fh = File::create(fname)?;
        for i in 0 .. s.len() {
            let p = s.txy(i);
            if p.is_valid() {
                writeln!(fh, "{} {} {}", p.x, p.y, s.cost[i])?;
            } else {
                writeln!(fh)?;
            }
        }
        Ok(())
    }

    fn write_segments(mut s: Sampling, fname: impl AsRef<Path>) -> R<()> {
        let mut fh = File::create(fname)?;
        let mut seg: Vec<(f64, TXY, TXY, f64)> = vec![];
        loop {
            let priority = s.pq.max_priority();
            if let Some(i0) = s.pop_from_pq() {
                let p0 = s.txy(i0);
                let p1 = s.txy(i0 + 1);
                let tm = (p0.t + p1.t) / 2.;
                seg.push((tm, p0, p1, priority))
            } else {
                break;
            }
        }
        seg.sort_by(|(t1,_,_,_), (t2,_,_,_)| t1.partial_cmp(t2).unwrap());
        for (tm, p0, p1, priority) in seg {
            writeln!(fh, "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                     tm, p0.t, p0.x, p0.y,  p1.t, p1.x, p1.y, priority)?;
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
