//! A max priority queue implemented as a Fibonacci heap.
//! We need it
//! - to use non-NaN `f64` values as priorities;
//! - to store pointers (it is made generic);
//! - to have "witnesses" that can invalidate these pointers in order
//!   to change the priority of already stored elements (without the
//!   cost of removing the old ones).

// See e.g. https://www.cs.princeton.edu/~wayne/teaching/fibonacci-heap.pdf

use std::{cmp::Ordering,
          ptr::NonNull};

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
    item: T,
    left: NonNull<Node<T>>,
    right: NonNull<Node<T>>,
    parent: Option<NonNull<Node<T>>>,
    // Pointer to one of the children.
    child: Option<NonNull<Node<T>>>,
    // Number of children.
    rank: usize,
    mark: bool,
}

impl<T> Node<T> {
    #[inline]
    fn new_dangling(priority: f64, item: T) -> NonNull<Self> {
        assert!(!priority.is_nan());
        let n = Self { priority: NotNAN(priority),  item,
                       left: NonNull::dangling(),
                       right: NonNull::dangling(),
                       parent: None,
                       child: None,
                       rank: 0,
                       mark: false };
        unsafe { NonNull::new_unchecked(Box::into_raw(Box::new(n))) }
    }
}

/// Priority queue holding elements of type T.
pub struct PQ<T> {
    max_node: Option<NonNull<Node<T>>>,
}

fn pq_drop<T>(node: Option<NonNull<Node<T>>>) {
    if let Some(mut node) = node {
        let end_ptr = node;
        loop {
            unsafe {
                pq_drop(node.as_ref().child);
                let right = node.as_ref().right;
                drop(Box::from_raw(node.as_ptr()));
                if right == end_ptr { break }
                else { node = right }
            }
        }
    }
}

impl<T> Drop for PQ<T> {
    fn drop(&mut self) { pq_drop(self.max_node) }
}

/// Witness for a node.  It enables increasing the node priority.
#[derive(Debug)]
pub struct Witness<T> {
    node: NonNull<Node<T>>
}

impl<T> PQ<T> {
    /// Return an empty priority queue.
    #[inline]
    pub fn new() -> Self { PQ { max_node: None } }

    #[cfg(test)]
    pub fn max_priority(&self) -> f64 {
        match self.max_node {
            Some(node) => unsafe { node.as_ref().priority.0 },
            None => f64::NAN
        }
    }

    /// Pushes `item` with `priority` to the queue and set a reference
    /// to the internal of the PQ in order to be able to increase the
    /// priority.
    ///
    /// # Safety
    /// The [`Witness`] is valid as long as the item stays in PQ.
    /// This is not enforced by the type system.
    pub fn push(&mut self, priority: f64, item: T) -> Witness<T> {
        let node = Node::new_dangling(priority, item);
        unsafe { self.push_to_roots(node) }
        Witness { node }
    }

    /// Push `node` to the roots.
    /// Return a possibly updated pointer to the max element of the list.
    /// We assume `node` is not already a root.
    unsafe fn push_to_roots(&mut self, mut node: NonNull<Node<T>>) {
        node.as_mut().parent = None;
        match self.max_node {
            None => {
                node.as_mut().left = node;
                node.as_mut().right = node;
                self.max_node = Some(node);
            }
            Some(mut max_node) => {
                if max_node.as_ref().priority >= node.as_ref().priority {
                    let mut right = max_node.as_ref().right;
                    max_node.as_mut().right = node;
                    node.as_mut().left = max_node;
                    node.as_mut().right = right;
                    right.as_mut().left = node;
                    // No change to `max_node`.
                } else {
                    let mut left = max_node.as_ref().left;
                    left.as_mut().right = node;
                    node.as_mut().left = left;
                    node.as_mut().right = max_node;
                    max_node.as_mut().left = node;
                    self.max_node = Some(node)
                }
            }
        }
    }

    /// Remove the node pointed by `node` (together with its subtree)
    /// from the circular list.  The node is left unmodified.
    /// It is assumed that there is another node in the list.
    unsafe fn remove_from_root_list(node: NonNull<Node<T>>) {
        let mut left = node.as_ref().left;
        let mut right = node.as_ref().right;
        left.as_mut().right = right;
        right.as_mut().left = left;
    }

    /// Remove `node` from its circular list, updating `parent`.
    /// # Invariant
    /// Assume `node.parent == parent`.
    unsafe fn remove_from_circular_list(mut parent: NonNull<Node<T>>,
                                        node: NonNull<Node<T>>) {
        let mut left = node.as_ref().left;
        if left == node {
            parent.as_mut().child = None;
        } else {
            let mut right = node.as_ref().right;
            left.as_mut().right = right;
            right.as_mut().left = left;
            // Update `parent` in case its child node was `node`.
            parent.as_mut().child = Some(right);
        }
    }

    /// Unparent all the nodes in the circular list pointed by `node`
    /// and return the (first) one with the highest priority.
    unsafe fn unparent_and_get_max(mut node: NonNull<Node<T>>)
                                   -> NonNull<Node<T>> {
        let end_ptr = node;
        let mut mx = node;
        node.as_mut().parent = None;
        node = node.as_ref().right;
        while node != end_ptr {
            node.as_mut().parent = None;
            if node.as_ref().priority > mx.as_ref().priority {
                mx = node;
            }
            node = node.as_ref().right;
        }
        mx
    }

    /// Add the tree `child` to the children of `node`.
    ///
    /// # Invariant
    /// It is assumed that the priority of `child` is smaller than the
    /// one of `node`.
    unsafe fn add_child(mut node: NonNull<Node<T>>,
                        mut child: NonNull<Node<T>>) {
        node.as_mut().rank += 1;
        child.as_mut().parent = Some(node);
        if let Some(mut child0) = node.as_ref().child {
            let mut right = child0.as_ref().right;
            child0.as_mut().right = child;
            child.as_mut().left = child0;
            child.as_mut().right = right;
            right.as_mut().left = child;
        } else {
            // Turn `child` into a circular list and add it.
            child.as_mut().right = child;
            child.as_mut().left = child;
            node.as_mut().child = Some(child);
        }
    }

    unsafe fn consolidate(&mut self) {
        // We will only reorganize the nodes, the one with the highest
        // priority does not change so `self.max_node` stays the same.
        if let Some(max_node) = self.max_node {
            let last_node = max_node.as_ref().left;
            if last_node == max_node {
                return // Single root.
            }
            let r0 = max_node.as_ref().rank;
            let mut rank = vec![None; (r0 + 1).max(16)];
            rank[r0] = Some(max_node);
            let mut is_last_node = false;
            let mut n = max_node.as_ref().right; // ≠ max_node
            loop {
                // Once we reach the last node, we loop further only
                // to merge it with other nodes.
                is_last_node = is_last_node || n == last_node;
                let r = n.as_ref().rank;
                if r >= rank.len() {
                    rank.resize(r + 1, None);
                } else if let Some(mut n1) = rank[r] {
                    rank[r] = None;
                    Self::remove_from_root_list(n1);
                    // If `n` has the same priority than `n1`, the
                    // older `n1` may be the max_root, so stays root.
                    if n.as_ref().priority > n1.as_ref().priority {
                        Self::add_child(n, n1);
                    } else {
                        // Make `n` a child of `n1` and replace `n` with `n1`.
                        let mut left = n.as_ref().left;
                        if left == n {
                            // Only `n` left after the removal of `n1`.
                            // Turn `n1 as a 1-element circular list
                            Self::add_child(n1, n);
                            n1.as_mut().left = n1;
                            n1.as_mut().right = n1;
                            break  // `n1` = `max_node` = single root
                        } else {
                            let mut right = n.as_ref().right;
                            Self::add_child(n1, n);
                            left.as_mut().right = n1;
                            n1.as_mut().left = left;
                            n1.as_mut().right = right;
                            right.as_mut().left = n1;
                            n = n1;
                        }
                    }
                    continue // Reexamine the merged node
                }
                if is_last_node { break }
                else {
                    rank[r] = Some(n);
                    n = n.as_ref().right
                }
            }
        }
    }

    pub fn pop(&mut self) -> Option<T> {
        if let Some(node) = self.max_node {
            unsafe {
                let mut next_max = node.as_ref().right;
                if next_max == node {
                    //  Sole root.  Let `node` children (if any) be new roots.
                    if let Some(c) = node.as_ref().child {
                        self.max_node = Some(Self::unparent_and_get_max(c))
                    } else {
                        self.max_node = None
                    }
                } else {
                    // Another root exists.  Find a maximum among the roots.
                    let mut n = next_max.as_ref().right;
                    while n != node {
                        if n.as_ref().priority > next_max.as_ref().priority {
                            next_max = n;
                        }
                        n = n.as_ref().right;
                    }
                    // Delete `node`, replace it with its children (if any).
                    if let Some(c) = node.as_ref().child {
                        let mut c = Self::unparent_and_get_max(c);
                        let mut left = node.as_ref().left;
                        let mut right = node.as_ref().right;
                        let mut c_left = c.as_ref().left;
                        left.as_mut().right = c;
                        c.as_mut().left = left;
                        c_left.as_mut().right = right;
                        right.as_mut().left = c_left;
                        if c.as_ref().priority > next_max.as_ref().priority {
                            self.max_node = Some(c);
                        } else {
                            self.max_node = Some(next_max);
                        }
                    } else {
                        Self::remove_from_root_list(node);
                        self.max_node = Some(next_max);
                    }
                }
                self.consolidate();
                Some(Box::from_raw(node.as_ptr()).item)
            }
        } else {
            None
        }
    }

    /// Set the priority of node pointed by `w` to `priority` if it is
    /// higher than the existing priority (otherwise, do nothing).
    ///
    /// # Safety
    /// BEWARE that the node pointed by `w` must be in the queue
    /// `self` otherwise it is undefined behavior.
    //
    // REMARK: There is a risk that the witness is not for the
    // appropriate queue.  Since this is an internal API and there
    // will be a single PQ per sampling, the risk is minimal and we
    // did not add a pointer to the queue in the witness.
    pub unsafe fn increase_priority(&mut self, w: &Witness<T>, priority: f64) {
        let mut node = w.node;
        assert!(!priority.is_nan());
        let priority = NotNAN(priority);
        if priority <= node.as_ref().priority {
            return
        }
        node.as_mut().priority = priority;
        if let Some(parent) = node.as_ref().parent {
            if priority <= parent.as_ref().priority {
                return;
            }
            // Cut `node` and add it to the roots.
            Self::remove_from_circular_list(parent, node);
            self.push_to_roots(node);
            self.cascading_cut(parent);
        }
        if priority > self.max_node.unwrap().as_ref().priority {
            self.max_node = Some(node);
        }
    }

    unsafe fn cascading_cut(&mut self, mut node: NonNull<Node<T>>) {
        if let Some(parent) = node.as_ref().parent {
            if node.as_ref().mark {
                node.as_mut().mark = false;
                Self::remove_from_circular_list(parent, node);
                self.push_to_roots(node);
                self.cascading_cut(parent);
            } else {
                node.as_mut().mark = true;
            }
        }
    }
}


#[cfg(test)]
mod test {
    use super::PQ;

    #[test]
    fn pair() {
        let mut q = PQ::new();
        q.push(1., "a");
        q.push(2., "b");
        assert_eq!(q.pop(), Some("b"));
        assert_eq!(q.pop(), Some("a"));
        assert_eq!(q.pop(), None);
    }

    #[test]
    fn basic() {
        let mut q = PQ::new();
        q.push(1., "a");
        q.push(2., "b");
        q.push(0., "c");
        q.push(-1., "d");
        assert_eq!(q.pop(), Some("b"));
        assert_eq!(q.pop(), Some("a"));
        assert_eq!(q.pop(), Some("c"));
        assert_eq!(q.pop(), Some("d"));
        assert_eq!(q.pop(), None);
    }

    #[test]
    fn test_drop() {
        let mut q = PQ::new();
        q.push(1., "a");
        q.push(2., "b");
        q.push(0., "c");
        q.push(-1., "d");
        q.pop(); // Exercise [`consolidate`].
    }

    #[test]
    fn increase_priority_simple() {
        let mut q = PQ::new();
        q.push(1., "a");
        q.push(2., "b");
        let w = q.push(0., "c");
        unsafe { q.increase_priority(&w, 3.); }
        assert_eq!(q.pop(), Some("c"));
        assert_eq!(q.pop(), Some("b"));
        assert_eq!(q.pop(), Some("a"));
        assert_eq!(q.pop(), None);
    }

    #[test]
    fn increase_priority() {
        let mut q = PQ::new();
        q.push(4., "d");
        q.push(1., "a");
        q.push(3., "c");
        q.push(2., "b");
        let w = q.push(0., "e");
        assert_eq!(q.pop(), Some("d"));
        unsafe { q.increase_priority(&w, 2.5); }
        assert_eq!(q.pop(), Some("c"));
        assert_eq!(q.pop(), Some("e"));
        assert_eq!(q.pop(), Some("b"));
        assert_eq!(q.pop(), Some("a"));
        assert_eq!(q.pop(), None);
    }

    #[test]
    fn spike_priority() {
        let n0 = 8;
        let mut q = PQ::new();
        let n1 = n0 / 2;
        for i in 0 .. n1 { q.push(i as f64, "a"); }
        for i in n1 .. n0 { q.push((n0 - i) as f64,"b"); }
        let mut n = 0;
        while q.pop().is_some() { n += 1 }
        assert_eq!(n, n0)
    }

    #[test]
    fn random_priority() {
        let n0 = 100;
        let mut q = PQ::new();
        // Pseudo-random priorities
        let mut random = 1000;
        const NORMALIZE_01: f64 = 1. / u32::MAX as f64;
        let mut rand = move || {
            random ^= random << 13;
            random ^= random >> 17;
            random ^= random << 5;
            random as f64 * NORMALIZE_01
        };
        for _ in 0 .. n0 {
            q.push(rand(), "a");
        }
        let mut n = 0;
        while q.pop().is_some() { n += 1 }
        assert_eq!(n, n0)
    }
}
