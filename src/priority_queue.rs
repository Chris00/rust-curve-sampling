//! Priority Queue adapted to our case.
//!

use std::{collections::BinaryHeap,
          cmp::Ordering};

/// `f64` numbers at the exclusion of NaN.
#[derive(Debug, Clone, Copy, PartialEq)]
struct NotNAN(f64);

impl Eq for NotNAN {}

impl PartialOrd for NotNAN {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl Ord for NotNAN {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

struct Node<T> {
    priority: NotNAN,
    elem: T,
}

// Only compare the priorities (equality has to be compatible with
// ordering).
impl<T> PartialEq for Node<T> {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}

impl<T> Eq for Node<T> {}

impl<T> Ord for Node<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority.cmp(&other.priority)
    }
}

impl<T> PartialOrd for Node<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(Node::cmp(self, other))
    }
}

// Priority queue holding elements of type T.
pub struct PQ<T>(BinaryHeap<Node<T>>);

impl<T> PQ<T> {
    /// Create an empty priority queue.
    pub fn new() -> Self {
        PQ(BinaryHeap::new())
    }

    /// Pushes `elem` with `priority` to the queue.
    pub fn push(&mut self, priority: f64, elem: T) {
        assert!(! priority.is_nan());
        self.0.push(Node { priority: NotNAN(priority), elem })
    }

    /// Removes the greatest element from the priority queue and
    /// returns it, or `None` if the queue is empty.
    pub fn pop(&mut self) -> Option<T> {
        self.0.pop().map(|n| n.elem)
    }
}

