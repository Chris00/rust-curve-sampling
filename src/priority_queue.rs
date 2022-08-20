//! Priority Queue adapted to our case.  Wee need it
//! - to use non-NaN `f64` values as priorities;
//! - to store pointers (it is made generic);
//! - to have "witnesses" that can invalidate these pointers in order
//!   to change the priority of already stored elements (without the
//!   cost of removing the old ones).

//  Maximum priority queue.  Implemented as a Pairing heap
// (http://en.wikipedia.org/wiki/Pairing_heap) following the paper:
//
// Fredman, Michael L.; Sedgewick, Robert; Sleator, Daniel D.; Tarjan,
// Robert E. (1986). "The pairing heap: a new form of self-adjusting
// heap" (PDF). Algorithmica. 1 (1): 111–129. doi:10.1007/BF01840439.

use std::{cmp::Ordering,
          ptr};

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
    child: *mut Node<T>, // points to the node itself if no child.
    sibling: *mut Node<T>, // next older sibling, parent if last, null if root.
    parent: *mut Node<T>, // null if root.
}

impl<T> Node<T> {
    /// Return a pointer to a newly created node (i.e., a priority
    /// queue with a single node).  The user is responsible for
    /// freeing the returned node.
    fn new_ptr(priority: f64, item: T) -> *mut Self {
        debug_assert!(!priority.is_nan());
        let n = Node { priority: NotNAN(priority), item,
                       child: ptr::null_mut(),
                       sibling: ptr::null_mut(),
                       parent: ptr::null_mut() };
        let n_ptr = Box::into_raw(Box::new(n));
        unsafe { (*n_ptr).child = n_ptr };
        n_ptr
    }

    #[inline]
    fn has_children(node_ptr: *mut Self) -> bool {
        debug_assert!(!node_ptr.is_null());
        unsafe { (*node_ptr).child != node_ptr }
    }

    #[inline]
    fn is_root(&self) -> bool { self.parent.is_null() }

    /// Recursively drop all nodes from a priority queue.  When
    /// passing the pointer to the root as `node_ptr`, use null for
    /// `last_sibling_ptr`.
    fn drop(mut node_ptr: *mut Node<T>, last_sibling_ptr: *mut Node<T>) {
        while node_ptr != last_sibling_ptr {
            debug_assert!(!node_ptr.is_null());
            let node = unsafe { Box::from_raw(node_ptr) };
            if Self::has_children(node_ptr) {
                Self::drop(node.child, node_ptr);
            }
            node_ptr = node.sibling;
        }
    }

    /// Link two priority queues.
    #[inline]
    fn link(n1_ptr: *mut Self, n2_ptr: *mut Self) -> *mut Self {
        debug_assert!(!n1_ptr.is_null() && !n2_ptr.is_null());
        let mut n1 = unsafe { &mut *n1_ptr as &mut Self };
        let mut n2 = unsafe { &mut *n2_ptr as &mut Self };
        if n1.priority >= n2.priority {
            // Because of the convention that the sibling = parent if
            // last, we do not have to make a special case if n2 is
            // the first child (in which case n1.child == n1_ptr).
            n2.sibling = n1.child;  n2.parent = n1_ptr;
            n1.child = n2_ptr;
            n1_ptr
        } else {
            n1.sibling = n2.child;  n1.parent = n2_ptr;
            n2.child = n1_ptr;
            n2_ptr
        }
    }
}

/// Priority queue holding elements of type T.
pub struct PQ<T> {
    root: *mut Node<T>
}

impl<T> Drop for PQ<T> {
    fn drop(&mut self) {
        if !self.root.is_null() {
            Node::drop(self.root, ptr::null_mut())
        }
    }
}

/// Witness for a node.  It enables increasing the node priority.
#[derive(Debug)]
pub struct Witness<T> {
    node_ptr: *mut Node<T>
}

impl<T> Clone for Witness<T> {
    // Reset the pointer on Clone because only one value `T` should be
    // able to change the `Node` in the priority queue.
    fn clone(&self) -> Self {
        Self { node_ptr: ptr::null_mut() }
    }
}

impl<T> Witness<T> {
    /// Witness that cannot be used with `increase_priority`.
    pub fn invalid() -> Self {
        Self { node_ptr: ptr::null_mut() }
    }
}


impl<T> PQ<T> {
    /// Create an empty priority queue.
    #[inline]
    pub fn new() -> Self { PQ { root: ptr::null_mut() } }

    /// Return `true` if the priority queue is empty.
    #[inline]
    pub fn is_empty(&self) -> bool { self.root.is_null() }

    /// Return a reference to the maximal item in the queue, or `None`
    /// if the queue is empty.
    pub fn max(&self) -> Option<&T> {
        if self.is_empty() { None }
        else { unsafe { Some(&(*self.root).item) } }
    }

    #[inline]
    fn push_node(&mut self, node_ptr: *mut Node<T>) {
        if self.is_empty() {
            self.root = node_ptr;
        } else {
            self.root = Node::link(node_ptr, self.root);
        }
    }

    /// Pushes `item` with `priority` to the queue.  Return a
    /// reference to the internal node in order to be able to increase
    /// the priority.  That reference only lives as long as the node
    /// is in `self`.  This is not enforced by the type system.
    pub fn push(&mut self, priority: f64, item: T) -> Witness<T> {
        assert!(!priority.is_nan());
        let node_ptr = Node::new_ptr(priority, item);
        self.push_node(node_ptr);
        // The node is owned by the priority queue.
        Witness { node_ptr }
    }

    /// Link all the children of a deleted node, the first child being
    /// pointed to by `p0_ptr` (which thus cannot be the root).
    fn link_pairs(n0_ptr: *mut Node<T>, last_sibling_ptr: *mut Node<T>)
                  -> *mut Node<T> {
        // Instead of using recursion, we accumulate the pairs in a vector.
        let mut pair = Vec::new();
        let mut n1_ptr = n0_ptr;
        while n1_ptr != last_sibling_ptr {
            debug_assert!(!n1_ptr.is_null());
            let n2_ptr = unsafe { (*n1_ptr).sibling };
            if n2_ptr != last_sibling_ptr {
                pair.push(Node::link(n1_ptr, n2_ptr));
                n1_ptr = unsafe { (*n2_ptr).sibling };
            } else { // n1_ptr is the last sibling
                pair.push(n1_ptr);
                break
            }
        }
        // Combine the pairs right-to-left.
        debug_assert!(!pair.is_empty());
        let mut pair = pair.iter().rev();
        let mut combined = *pair.next().unwrap();
        for &n in pair {
            combined = Node::link(n, combined)
        }
        combined
    }

    /// Removes the greatest element from the priority queue and
    /// returns it, or `None` if the queue is empty.
    pub fn pop(&mut self) -> Option<T> {
        if self.is_empty() { return None }
        let root_has_children = Node::has_children(self.root);
        let root = unsafe { Box::from_raw(self.root) };
        if root_has_children {
            let root1_ptr = Self::link_pairs(root.child, self.root);
            unsafe {
                (*root1_ptr).parent = root1_ptr;
                (*root1_ptr).sibling = root1_ptr;
            }
            self.root = root1_ptr;
        } else {
            self.root = ptr::null_mut()
        }
        Some(root.item)
    }

    /// Set the priority of node pointed by `w` to `priority` if it is
    /// higher than the existing priority (otherwise, do nothing).
    /// BEWARE that the node pointed by `w` must be in the queue
    /// `self` otherwise it is undefined behavior.
    #[allow(unused_unsafe)]
    pub unsafe fn increase_priority(&mut self, w: Witness<T>, priority: f64) {
        debug_assert!(!w.node_ptr.is_null());
        let mut node = unsafe { &mut *w.node_ptr as &mut Node<T> };
        if priority > node.priority.0 { // ⟹ `priority` is not NaN
            let priority = NotNAN(priority);
            if node.is_root() {
                node.priority = priority;
            } else {
                // Cut `node` (and its children) from the tree and
                // re-insert it with the new `priority`.
                debug_assert!(!node.parent.is_null());
                let mut parent = unsafe { &mut *node.parent as &mut Node<T> };
                if parent.child == w.node_ptr { // first (possibly only) child
                    parent.child = node.sibling;
                } else {
                    let mut n_prev = parent.child;
                    while unsafe { (*n_prev).sibling } != w.node_ptr {
                        n_prev = unsafe { (*n_prev).sibling };
                    }
                    unsafe { (*n_prev).sibling = node.sibling }
                }
                node.priority = priority;       // detached node
                node.sibling = ptr::null_mut();
                node.parent = ptr::null_mut();
                self.push_node(w.node_ptr)
            }
        }
    }
}



#[cfg(test)]
mod test {
    use super::PQ;

    #[test]
    fn basic() {
        let mut q = PQ::new();
        q.push(1., "a");
        q.push(2., "b");
        q.push(0., "c");
        assert_eq!(q.pop(), Some("b"));
        assert_eq!(q.pop(), Some("a"));
        assert_eq!(q.pop(), Some("c"));
        assert_eq!(q.pop(), None);
    }

        #[test]
    fn increase_priority() {
        let mut q = PQ::new();
        q.push(1., "a");
        q.push(2., "b");
        let w = q.push(0., "c");
        unsafe { q.increase_priority(w, 3.); }
        assert_eq!(q.pop(), Some("c"));
        assert_eq!(q.pop(), Some("b"));
        assert_eq!(q.pop(), Some("a"));
        assert_eq!(q.pop(), None);
    }

}
