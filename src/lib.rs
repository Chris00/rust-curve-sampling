//!

use std::{fmt::{self, Display, Formatter},
          io::{self, Write},
          iter::Iterator,
          mem::swap,
          ptr, marker::PhantomData};
use rand::prelude::*;
use rgb::*;

mod priority_queue;
use priority_queue::{PQ, Witness};

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

/// A 2D point with coordinates (`x`, `y`) supposed to be given by a
/// function evaluated at "time" `t`.  A path is a sequence of points.
/// Points that are invalid (see [`Point::is_valid`]) are considered
/// cuts in the path (the line is interrupted).  As most 2 cuts may
/// follow each other (to track ranges of t outside the domain of the
/// function) except at the boundary of the path where at most 1 cut
/// is allowed.
#[derive(Debug, Clone)]
struct Point {
    t: f64, // "time" parameter, ALWAYS finite.
    x: f64,
    y: f64, // `y` is finite ⇒ `x` is finite
    cost: f64, // Cache the cost of the point (not the segment)
    prev: *mut Point,
    next: *mut Point,
    witness: Witness<*mut Point>,
}

impl Point {
    /// Return a new point (with no cost, no predecessor and no
    /// successor).  `t` is assumed to be finite.
    #[inline]
    fn new(t: f64, x: f64, y: f64) -> Self {
        Point { t, x, y,  cost: 0.,
                prev: ptr::null_mut(),
                next: ptr::null_mut(),
                witness: Witness::invalid() }
    }

    #[inline]
    fn cut(t: f64) -> Self {
        Point { t, x: f64::NAN, y: f64::NAN,
                cost: 0.,
                prev: ptr::null_mut(),
                next: ptr::null_mut(),
                witness: Witness::invalid() }
    }

    /// Return `true` if the point is valid — otherwise it is a cut.
    #[inline]
    fn is_valid(&self) -> bool { self.y.is_finite() }
}

/// A 2D sampling.  This can be thought as a path, with possible
/// "cuts" because of discontinuities or leaving the domain of the
/// (parametric) function describing the path.
pub struct Sampling {
    // Priority queue of segments.  The value points to the first
    // `Point` of the segment (thus its .next pointer must not be
    // null).  If `pq` is empty but `begin` is not null, it means that
    // the costs need to to be updated.
    pq: PQ<*mut Point>,
    head: *mut Point,
    tail: *mut Point, // tail.is_null() ⇔ head.is_null()
    vp: Option<BoundingBox>, // viewport (zone of interest)
}

impl Drop for Sampling {
    fn drop(&mut self) {
        let mut pt_ptr = self.head;
        while !pt_ptr.is_null() {
            let pt = unsafe { Box::from_raw(pt_ptr) };
            pt_ptr = pt.next;
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Lengths {
    t: f64,
    x: f64,
    y: f64,
}

impl Sampling {
    /// Return `true` if the sampling contains no point.
    #[inline]
    pub fn is_empty(&self) -> bool { self.head.is_null() }

    /// Create an empty sampling.
    #[inline]
    pub(crate) fn empty() -> Self {
        Self { pq: PQ::new(),
               head: ptr::null_mut(),
               tail: ptr::null_mut(),
               vp: None }
    }

    #[inline]
    pub(crate) fn singleton(mut p: Point) -> Self {
        debug_assert!(p.t.is_finite() && p.x.is_finite() && p.y.is_finite());
        // `p` may be obtained cloning some other point.  Sanitize it.
        p.prev = ptr::null_mut();
        p.next = ptr::null_mut();
        let pt = Box::into_raw(Box::new(p));
        Self { pq: PQ::new(),
               head: pt,
               tail: pt,
               vp: None }
    }

    /// Push a new point at the end of the sampling (which the sampling then
    /// owns).  Does not modify the priority queue.  Does not verify
    /// that at most 2 cuts can follow each other or that `p.y` is
    /// finite ⇒ `p.x` is finite.
    #[inline]
    pub(crate) fn push_unchecked(&mut self, mut p: Point) {
        debug_assert!(!p.y.is_finite() || p.x.is_finite());
        if self.is_empty() {
            // `p` may be obtained cloning some other point.  Sanitize it.
            p.prev = ptr::null_mut();
            p.next = ptr::null_mut();
            let p = Box::into_raw(Box::new(p));
            self.head = p;
            self.tail = p;
        } else {
            p.prev = self.tail;
            p.next = ptr::null_mut();
            let p = Box::into_raw(Box::new(p));
            unsafe { (*self.tail).next = p };
            self.tail = p;
        }
    }

    /// Remove the final point (if any) of the path and return it.
    pub(crate) fn pop(&mut self) -> Option<Point> {
        if self.tail.is_null() {
            None
        } else {
            let p = unsafe { *Box::from_raw(self.tail) };
            if p.prev.is_null() {
                self.head = ptr::null_mut();
            } else {
                unsafe { (*p.prev).next = ptr::null_mut() }
            }
            self.tail = p.prev;
            Some(p)
        }
    }

        }
    }

    #[inline]
    pub(crate) fn set_vp(&mut self, bb: BoundingBox) {
        self.vp = Some(bb);
    }

    /// Return the length of the "time interval" as well as the
    /// lengths of the viewport.
    pub(crate) fn len_txy(&self) -> Option<Lengths> {
        if self.is_empty() { return None }
        let p0 = unsafe { &*self.head as &Point };
        let p1 = unsafe { &*self.tail as &Point };
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

    /// Iterate on points indices (one output per point).
    pub(crate) fn iter_points(&self) -> IterPoints<'_> {
        IterPoints { next: self.head,  marker: PhantomData }
    }

    /// Iterate on points indices (one output per point).  Also
    /// returns the priority queue as we may want to mutate it using
    /// this iterator (which prevents `self.vp` to be borrowed).
    pub(crate) fn iter_points_mut(&mut self) -> (IterPointsMut<'_>,
                                                &mut PQ<*mut Point>) {
        (IterPointsMut { next: self.head,  marker: PhantomData },
         &mut self.pq)
    }

    /// Iterate on the points (and cuts) of the path.  More
    /// precisely, a path is made of continuous segments whose
    /// points are given by contiguous values `Some(p)`
    /// interspaced by `None`.  Two `None` never follow each
    /// other.  Isolated points `p` are given by ... `None`,
    /// `Some(p)`, `None`,...
    pub fn iter(&self) -> Iter<'_> {
        Iter { next: self.head,
               prev_is_cut: true, // avoid issuing a cut at start
               marker: PhantomData }
    }
}

/// "Raw" mutable iterator on the list of points composing the path.
struct IterPoints<'a> {
    next: *mut Point,
    marker: PhantomData<&'a mut Point>,
}

impl<'a> Iterator for IterPoints<'a> {
    type Item = &'a Point;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next.is_null() { return None }
        let p = unsafe { &*self.next as &Point };
        self.next = p.next;
        Some(p)
    }
}

/// "Raw" mutable iterator on the list of points composing the path.
struct IterPointsMut<'a> {
    next: *mut Point,
    marker: PhantomData<&'a mut Point>,
}

impl<'a> Iterator for IterPointsMut<'a> {
    type Item = &'a mut Point;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next.is_null() { return None }
        let p = unsafe { &mut *self.next as &mut Point };
        self.next = p.next;
        Some(p)
    }
}

/// Iterator on the points of the [`Sampling`].
/// See [`Sampling::iter`] for more information.
pub struct Iter<'a> {
    next: *const Point,
    prev_is_cut: bool, // avoid two cuts following each other
    marker: PhantomData<&'a Point>,
}

impl<'a> Iterator for Iter<'a> {
    type Item = Option<[f64; 2]>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next.is_null() { return None }
        let p = unsafe { &*self.next as &Point };
        self.next = p.next;
        if p.is_valid() {
            self.prev_is_cut = false;
            Some(Some([p.x, p.y]))
        } else if self.prev_is_cut {
            // Unroll `self.next()` as tail recursion optimization is
            // not guaranteed.  There are at most 2 consecutive cuts,
            // so the next one (if it exists) must be a valid point.
            if self.next.is_null() { return None }
            let p = unsafe { &*self.next as &Point };
            self.next = p.next;
            debug_assert!(p.is_valid());
            self.prev_is_cut = false;
            Some(Some([p.x, p.y]))
        } else if self.next.is_null() {
            // Cut but at the end of the path; don't issue it.
            None
        } else {
            self.prev_is_cut = true;
            Some(None)
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
        let mut points = self.iter().skip_while(|p| p.is_none());
        let mut bb = match &points.next() {
            Some(Some([x, y])) => BoundingBox { xmin: *x,  xmax: *x,
                                               ymin: *y,  ymax: *y },
            Some(None) => unreachable!(),
            None => return BoundingBox::empty()
        };
        for p_opt in points {
            match p_opt {
                Some([x, y]) => {
                    if x < bb.xmin { bb.xmin = x }
                    else if bb.xmax < x { bb.xmax = x };
                    if y < bb.ymin { bb.ymin = y }
                    else if bb.ymax < y { bb.ymax = y };
                }
                None => ()
            }
        }
        bb
    }

    /// Transpose in place the x and y coordinates of the sampling.
    pub fn transpose(&mut self) -> &mut Self {
        let (iter, _) = self.iter_points_mut();
        for p in iter {
            swap(&mut p.x, &mut p.y)
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
        let dx = p1.x - p0.x; // May be 0.
        let r = (if dx >= 0. {bb.xmax} else {bb.xmin} - p0.x) / dx;
        if r < t { t = r } // ⟹ r finite (as r ≥ 0 or NaN)
        let dy = p1.y - p0.y; // May be 0.
        let r = (if dy >= 0. {bb.ymax} else {bb.ymin} - p0.y) / dy;
        if r < t { t = r };
        if t <= 1e-14 {
            None
        } else {
            Some(Point::new(p0.t + t * (p1.t - p0.t),
                            p0.x + t * dx,
                            p0.y + t * dy))
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
        if r0 > 0. { t0 = r0 } // if r0 is NAN, keep the whole segment
        if r1 < 1. { t1 = r1 }
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
        if r0 > t1 || r1 < t0 { return Intersection::Empty }
        if r0 > t0 { t0 = r0 }
        if r1 < t1 { t1 = r1 }
        if t0 < t1 { // segment not reduced to a point
            let dt = p1.t - p0.t;
            let q0 = Point::new(p0.t + t0 * dt,
                                p0.x + t0 * dx,
                                p0.y + t0 * dy);
            let q1 = Point::new(p0.t + t1 * dt,
                                p0.x + t1 * dx,
                                p0.y + t1 * dy);
            Intersection::Seg(q0, q1)
        } else if t0 == t1 {
            let q0 = Point::new(p0.t + t0 * (p1.t - p0.t),
                                p0.x + t0 * dx,
                                p0.y + t0 * dy);
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
        let mut p0_opt: Option<&Point> = None;
        let mut p0_inside = false;
        let mut prev_cut = true; // New path ("virtual" cut)
        for p1 in self.iter_points() {
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
                            s.push_unchecked(p0.clone());
                            if p1_inside {
                                s.push_unchecked(p1.clone());
                                prev_cut = false;
                            } else {
                                if let Some(p) = Self::intersect(p0, p1, bb) {
                                    let t = p.t;
                                    s.push_unchecked(p);
                                    s.push_unchecked(Point::cut(t));
                                } else {
                                    s.push_unchecked(Point::cut(p0.t));
                                }
                            }
                        } else if p1_inside { // p0 ∉ bb, p1 ∈ bb
                            if let Some(p) = Self::intersect(p1, p0, bb) {
                                s.push_unchecked(p); // p ≠ p1
                            }
                            s.push_unchecked(p1.clone());
                            prev_cut = false;
                        } else { // p0, p1 ∉ bb but maybe intersection
                            match Self::intersect_seg(p0, p1, bb) {
                                Intersection::Seg(q0, q1) => {
                                    let t1 = q1.t;
                                    s.push_unchecked(q0);
                                    s.push_unchecked(q1);
                                    s.push_unchecked(Point::cut(t1));
                                }
                                Intersection::Pt(p) => {
                                    let t = p.t;
                                    s.push_unchecked(p);
                                    s.push_unchecked(Point::cut(t));
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
                            s.push_unchecked(p0.clone());
                            s.push_unchecked(Point::cut(p0.t));
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
                        s.push_unchecked(p1.clone());
                    } else { // p0 ∈ bb, p1 ∉ bb
                        if let Some(p) = Self::intersect(p0, p1, bb) {
                            let t = p.t;
                            s.push_unchecked(p);
                            s.push_unchecked(Point::cut(t));
                        } else {
                            s.push_unchecked(Point::cut(p0.t));
                        }
                        prev_cut = true;
                    }
                } else { // p1 is invalid (i.e., represent a cut)
                    p0_opt = None;
                    s.push_unchecked(p1.clone());
                    prev_cut = true
                }
            }
        }
        if prev_cut { s.pop(); }
        s.set_vp(bb);
        s
    }

    /// Create a sampling from an iterator of points.  Beware that the
    /// invariant "`p.y` is finite ⇒ `p.x` is finite" is not checked.
    fn from_point_iterator<P>(points: P) -> Self
    where P: IntoIterator<Item = Point> {
        let mut prev_is_cut = true;
        let mut s = Sampling::empty();
        for p in points.into_iter() {
            if p.is_valid() {
                s.push_unchecked(p);
                prev_is_cut = false;
            } else if ! prev_is_cut {
                s.push_unchecked(Point::cut(p.t));
                prev_is_cut = true;
            }
        }
        s
    }

    /// Create a sampling from `points` after sorting them by
    /// increasing (if `incr`) or decreasing (if `! incr`) values of
    /// the field `t`.
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
            points.into_iter().enumerate().map(|(i, [x, y])| {
                let t = i as f64;
                if x.is_finite() {Point::new(t, x, y)}
                else {Point::cut(t)} }))
    }
}

////////////////////////////////////////////////////////////////////////
//
// Acceptable types & functions that provide "points".
// This is internal to this library.

impl From<(f64, f64)> for Point {
    fn from((x, y): (f64, f64)) -> Self {
        Point::new(x, x, y) // `x` ∈ \[a, b\] by [`init_pt`] checks.
    }
}

impl From<(f64, [f64; 2])> for Point {
    /// Assume `t` is finite.
    fn from((t, [x,y]): (f64, [f64;2])) -> Self {
        // Enforce the invariant: y finite ⟹ x finite
        if x.is_finite() { Point::new(t, x, y) }
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
        Point::new(t, t, self.0(t))
    }
}

struct ParamPoint<T>(T);

impl<T> IntoFnPoint for ParamPoint<T> where T: FnMut(f64) -> [f64; 2] {
    #[inline]
    fn eval(&mut self, t: f64) -> Point {
        let [x, y] = self.0(t);
        // `Point::is_valid()` only checks `y`; make sure non-finite
        // `x` leads to an invalid point.
        if x.is_finite() { Point::new(t, x, y) }
        else { Point::new(t, x, f64::NAN) }
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
    /// use std::fs::File;
    /// use curve_sampling::Sampling;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let s = Sampling::uniform(|x| x.sin(), 0., 4.).build();
    /// s.write(&mut File::create("target/uniform.dat")?)?;
    /// # Ok(()) }
    /// ```
    uniform -> f64,
    /// Uniform sampling options.  See [`Sampling::uniform`].
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
    use super::{Point, Sampling, Lengths, PQ};

    // The cost of a point is a measure of the curvature at this
    // point.  This requires segments before and after the point.  In
    // case the point is a cut, or first, or last, it has a cost of 0.
    // If it is an endpoint of a segment with the other point a cut,
    // the cost is set to [`HANGING_NODE`] because the segment with
    // the invalid point needs to be cut of too long to better
    // determine the boundary.
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
    const HANGING_NODE: f64 = 5e5;

    /// Return the cost of the middle point `pm`.  Assumes `p0` and
    /// `p1` are valid points.
    #[inline]
    pub(crate) fn estimate(p0: &Point, pm: &Point, p1: &Point,
                           len: Lengths) -> f64 {
        let dx0m = (p0.x - pm.x) / len.x;
        let dy0m = (p0.y - pm.y) / len.y;
        let dx1m = (p1.x - pm.x) / len.x;
        let dy1m = (p1.y - pm.y) / len.y;
        let len0m = dx0m.hypot(dy0m);
        let len1m = dx1m.hypot(dy1m);
        if len0m == 0. || len1m == 0. {
            f64::NEG_INFINITY // Do not subdivide
        } else {
            let dx = - dx0m * dx1m - dy0m * dy1m;
            let dy = dy0m * dx1m - dx0m * dy1m;
            dy.atan2(dx) // ∈ [-π, π]
        }
    }

    #[inline]
    pub(crate) fn segment_vp(p0: &Point, p1: &Point, len: Lengths,
                          in_vp: bool) -> f64 {
        if ! in_vp { return f64::NEG_INFINITY }
        segment(p0, p1, len)
    }

    /// Compute the cost of a segment according to the costs of its
    /// endpoints.  `len_t` is the length of total range of time.
    /// `len_x` and `len_y` are the dimensions of the bounding box.
    #[inline]
    pub(crate) fn segment(p0: &Point, p1: &Point, len: Lengths) -> f64 {
        let dt = (p1.t - p0.t) / len.t; // ∈ [0,1]
        debug_assert!(dt >= 0. && dt <= 1.);
        // Put less efforts when `dt` is small.  For functions, the
        // Y-variation may be large but, if it happens for a small range
        // of `t`, there is no point in adding indistinguishable details.
        let dx = ((p1.x - p0.x) / len.x).abs();
        let dy = ((p1.y - p0.y) / len.y).abs();
        let mut cost = p0.cost.abs() + p1.cost.abs();
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

    fn push_segment(pq: &mut PQ<*mut Point>, p0: &mut Point, p1: &Point,
                    len: Lengths, in_vp: bool) {
        let cost_segment = segment_vp(p0, p1, len, in_vp);
        // The segment is referred to by its first point.
        let w = pq.push(cost_segment, p0 as *mut _);
        p0.witness = w;
    }

    /// Update the cost of all points in the sampling and add segments
    /// to the priority queue.
    pub(crate) fn compute(s: &mut Sampling, in_vp: impl Fn(&Point) -> bool) {
        if let Some(len) = s.len_txy() {
            // Path is not empty.
            let (mut pts, pq) = s.iter_points_mut();
            let mut p0 = pts.next().unwrap();
            p0.cost = 0.;
            let mut p0_in_vp = p0.is_valid() && in_vp(p0);
            let mut pm = match pts.next() {
                Some(p) => p,
                None => return };
            for p1 in pts {
                let pm_in_vp;
                if pm.is_valid() {
                    pm_in_vp = in_vp(pm);
                    if p0.is_valid() && p1.is_valid() {
                        pm.cost = estimate(p0, pm, p1, len);
                    } else {
                        pm.cost = HANGING_NODE;
                    }
                } else { // pm is the location of a cut
                    pm_in_vp = false;
                    pm.cost = 0.;
                }
                // Segment [p0, pm]
                push_segment(pq, p0, pm, len, p0_in_vp || pm_in_vp);
                p0 = pm;
                p0_in_vp = pm_in_vp;
                pm = p1;
            }
            pm.cost = 0.; // last point
            push_segment(pq, p0, pm, len, p0_in_vp || in_vp(pm));
        }
    }
}

////////////////////////////////////////////////////////////////////////
//
// Function sampling

fn refine_gen(s: &mut Sampling, n: usize,
              mut f: impl FnMut(f64) -> Point,
              in_vp: impl Fn(&Point) -> bool) {

}

        }
    }
}

fn push_almost_uniform_sampling(points: &mut Vec<Point>,
                                f: &mut impl FnMut(f64) -> Point,
                                a: f64, b: f64, n: usize) {
    let dt = (b - a) / (n - 1) as f64;
    let mut rng = rand::thread_rng();
    points.push(f(a));
    points.push(f(a + 0.0625 * dt));
    for i in 0 .. n - 4 {
        let j = i as f64 + rng.gen::<f64>() * 0.125 - 0.0625;
        points.push(f(a + j * dt));
    }
    points.push(f(b - 0.0625 * dt));
    points.push(f(b));
}

impl Sampling {
    /// Return a sampling from the initial list of `points`
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
        let n0 = (n / 10).min(10);
        push_almost_uniform_sampling(&mut points, &mut f, a, b, n0);
        let mut s = Sampling::from_vec(points, a < b);
        match viewport {
            Some(vp) => {
                let in_vp = |p: &Point| vp.contains(p);
                cost::compute(&mut s, in_vp);
                refine_gen(&mut s, n - n0, &mut f, in_vp)
            }
            None => {
                cost::compute(&mut s, |_| true);
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
    /// use std::fs::File;
    /// use curve_sampling::Sampling;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let s = Sampling::fun(|x| x.sin(), 0., 4.).build();
    /// s.write(&mut File::create("target/fun.dat")?)?;
    /// # Ok(()) }
    /// ```
    fun -> f64,
    /// Options for sampling a function ℝ → ℝ.  See [`Sampling::fun`].
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
    /// use std::fs::File;
    /// use curve_sampling::Sampling;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let s = Sampling::param(|t| [t.sin(), t.cos()], 0., 4.).build();
    /// s.write(&mut File::create("target/param.dat")?)?;
    /// # Ok(()) }
    /// ```
    param -> [f64; 2],
    /// Options for sampling a function ℝ → ℝ.  See [`Sampling::param`].
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

/// LaTeX output.
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
        for p in self.sampling.iter() { match p {
            Some([x, y]) => {
                n += 1;
                if new_path {
                    write!(f, "\\pgfpathmoveto{{\\pgfpointxy\
                               {{{:.16}}}{{{:.16}}}}}\n", x, y)?
                } else if n >= self.n {
                    write!(f, "\\pgfpathlineto{{\\pgfpointxy\
                               {{{:.16}}}{{{:.16}}}}}\n\
                               \\pgfusepath{{stroke}}\n\
                               \\pgfpathmoveto{{\\pgfpointxy\
                               {{{:.16}}}{{{:.16}}}}}\n", x, y, x, y)?;
                    n = 0;
                } else {
                    write!(f, "\\pgfpathlineto{{\\pgfpointxy\
                               {{{:.16}}}{{{:.16}}}}}\n", x, y)?
                }
                new_path = false;
            }
            None =>  {
                write!(f, "\\pgfusepath{{stroke}}\n")?;
                n = 0;
                new_path = true;
            }
        }}
        Ok(())
    }

    /// Write the path, each continuous sub-path with an arrow.
    fn write_with_arrows(&self, f: &mut impl Write, arrow: &str,
                         arrow_pos: f64) -> Result<(), io::Error> {
        // Compute the length of all sub-paths.
        let mut lens = vec![];
        let mut prev_pt: Option<[f64; 2]> = None;
        let mut cur_len = 0.;
        for p in self.sampling.iter() { match p {
            Some([x, y]) => {
                if let Some([x0, y0]) = prev_pt {
                    cur_len += (x - x0).hypot(y - y0);
                }
                prev_pt = p;
            }
            None => {
                lens.push(arrow_pos * cur_len);
                prev_pt = None;
                cur_len = 0.;
            }
        }}
        lens.push(arrow_pos * cur_len);
        if lens.is_empty() { return Ok(()) }
        let mut lens = lens.iter();
        let mut rem_len = *lens.next().unwrap(); // remaining before arrow
        prev_pt = None;
        let mut n = 0;
        for p in self.sampling.iter() { match p {
            Some([x, y]) => {
                n += 1;
                if let Some([x0, y0]) = prev_pt {
                    let dx = x - x0;
                    let dy = y - y0;
                    let l = dx.hypot(dy);
                    if rem_len <= l {
                        write!(f, "\\pgfusepath{{stroke}}\n")?;
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
                        write!(f, "\\pgfpathlineto{{\\pgfpointxy\
                                   {{{:.16}}}{{{:.16}}}}}\n", x, y)?
                    }
                    rem_len -= l;
                } else {
                    // No previous point.  New sub-path.
                    write!(f, "\\pgfpathmoveto{{\\pgfpointxy\
                               {{{:.16}}}{{{:.16}}}}}\n", x, y)?
                }
                prev_pt = p;
            }
            None => {
                write!(f, "\\pgfusepath{{stroke}}\n")?;
                rem_len = *lens.next().unwrap();
                prev_pt = None;
            }
        }}
        Ok(())
    }

    /// Write the sampling to the formatter as PGF/TikZ commands.
    pub fn write(&self, f: &mut impl Write) -> Result<(), io::Error> {
        write!(f, "% Written by the Rust curve_sampling crate.\n")?;
        write!(f, "\\begin{{pgfscope}}\n")?;
        match self.color {
            Some(RGB8 {r, g, b}) =>
                write!(f, "\\definecolor{{RustCurveSamplingColor}}{{RGB}}\
                           {{{},{},{}}}\n\
                           \\pgfsetstrokecolor{{RustCurveSamplingColor}}\n",
                       r, g, b)?,
            None => (),
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
        for p in self.iter() {
            match p {
                Some([x, y]) => write!(f, "{:e} {:e}\n", x, y)?,
                None => write!(f, "\n")?,
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
        for p in self.iter() {
            match p {
                Some([x, y]) => write!(f, "{:e} {:e}\n", x, y)?,
                None => write!(f, "\n")?,
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
    use crate::{Sampling, BoundingBox, Point};

    fn xy_of_sampling(s: &Sampling) -> Vec<Option<(f64, f64)>> {
        s.iter_points().map(|p| {
            if p.is_valid() { Some((p.x, p.y)) } else { None }})
            .collect()
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
        let s = Sampling::singleton(Point::new(0., 1., 2.));
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
}
