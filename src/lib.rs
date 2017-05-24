//! Per-object thread local storage.
//!
//! The `ThreadLocal` type which stores a distinct (nullable) value of some type for each thread
//! that accesses it.
//!
//! A thread's values are destroyed when it exits, but the values associated with a `ThreadLocal`
//! instance are not destroyed when it is dropped. These are in some ways the opposite semantics of
//! those provided by the `thread_local` crate, where values are cleaned up when a `ThreadLocal`
//! object is dropped, but not when individual threads exit.
//!
//! Because of this, this crate is an appropriate choice for use cases where you have long lived
//! `ThreadLocal` instances which are widely shared among threads that are created and destroyed
//! through the runtime of a program, while the `thread_local` crate is an appropriate choice for
//! short lived values.
//!
//! # Examples
//!
//! ```rust
//! use std::sync::Arc;
//! use std::thread;
//! use instance_thread_local::ThreadLocal;
//!
//! let tls = Arc::new(ThreadLocal::new());
//!
//! tls.set(1);
//!
//! let tls2 = tls.clone();
//! thread::spawn(move || {
//!     // the other thread doesn't see the 1
//!     assert_eq!(tls2.get_cloned(), None);
//!     tls2.set(2);
//! }).join().unwrap();
//!
//! // we still see our original value
//! assert_eq!(tls.get_cloned(), Some(1));
//! ```
#![warn(missing_docs)]
extern crate unsafe_any;

use std::cell::RefCell;
use std::collections::hash_map::{self, HashMap};
use std::marker::PhantomData;
use std::mem;
use std::sync::atomic::{AtomicUsize, ATOMIC_USIZE_INIT, Ordering};
use unsafe_any::UnsafeAny;

thread_local! {
    static VALUES: RefCell<HashMap<usize, Box<UnsafeAny>>> = RefCell::new(HashMap::new());
}

static NEXT_ID: AtomicUsize = ATOMIC_USIZE_INIT;

// if IDs ever wrap around we'll run into soundness issues with downcasts, so panic if we're out of
// IDs. On 64 bit platforms this can literally never happen (it'd take 584 years even if you were
// generating a billion IDs per second), but is more realistic a concern on 32 bit platforms.
//
// FIXME use AtomicU64 when it's stable
fn next_id() -> usize {
    let mut id = NEXT_ID.load(Ordering::SeqCst);
    loop {
        assert!(id != usize::max_value(), "thread local ID overflow");
        let old = id;
        id = NEXT_ID.compare_and_swap(old, old + 1, Ordering::SeqCst);
        if id == old {
            return id;
        }
    }
}

/// A thread local variable wrapper.
pub struct ThreadLocal<T: 'static> {
    id: usize,
    _p: PhantomData<T>,
}

impl<T: 'static> ThreadLocal<T> {
    /// Creates a new `ThreadLocal` with no values for any threads.
    ///
    /// # Panics
    ///
    /// Panics if more than `usize::max_value()` `ThreadLocal` objects have already been created.
    /// This can only ever realistically happen on 32 bit platforms.
    pub fn new() -> ThreadLocal<T> {
        ThreadLocal {
            id: next_id(),
            _p: PhantomData,
        }
    }

    /// Sets this thread's value, returning the previous value if present.
    pub fn set(&self, value: T) -> Option<T> {
        self.entry(|e| match e {
                       Entry::Occupied(mut e) => Some(e.insert(value)),
                       Entry::Vacant(e) => {
                           e.insert(value);
                           None
                       }
                   })
    }

    /// Removes this thread's value, returning it if it existed.
    pub fn remove(&self) -> Option<T> {
        VALUES.with(|v| {
                        v.borrow_mut()
                            .remove(&self.id)
                            .map(|v| unsafe { *v.downcast_unchecked::<T>() })
                    })
    }

    /// Passes a handle to the current thread's value to a closure for in-place manipulation.
    ///
    /// The closure is required for the same soundness reasons it is required for the standard
    /// library's `thread_local!` values.
    pub fn entry<F, R>(&self, f: F) -> R
        where F: FnOnce(Entry<T>) -> R
    {
        VALUES.with(|v| {
            let mut v = v.borrow_mut();
            let entry = match v.entry(self.id) {
                hash_map::Entry::Occupied(e) => Entry::Occupied(OccupiedEntry(e, PhantomData)),
                hash_map::Entry::Vacant(e) => Entry::Vacant(VacantEntry(e, PhantomData)),
            };
            f(entry)
        })
    }

    /// Passes a mutable reference to the current thread's value to a closure.
    ///
    /// The closure is required for the same soundness reasons it is required for the standard
    /// library's `thread_local!` values.
    pub fn get<F, R>(&self, f: F) -> R
        where F: FnOnce(Option<&T>) -> R
    {
        VALUES.with(|v| {
                        let v = v.borrow();
                        let value = v.get(&self.id)
                            .map(|v| unsafe { v.downcast_ref_unchecked() });
                        f(value)
                    })
    }

    /// Passes a mutable reference to the current thread's value to a closure.
    ///
    /// The closure is required for the same soundness reasons it is required for the standard
    /// library's `thread_local!` values.
    pub fn get_mut<F, R>(&self, f: F) -> R
        where F: FnOnce(Option<&mut T>) -> R
    {
        VALUES.with(|v| {
                        let mut v = v.borrow_mut();
                        let value = v.get_mut(&self.id)
                            .map(|v| unsafe { v.downcast_mut_unchecked() });
                        f(value)
                    })
    }
}

impl<T> ThreadLocal<T>
    where T: 'static + Clone
{
    /// Returns a copy of the current thread's value.
    pub fn get_cloned(&self) -> Option<T> {
        VALUES.with(|v| {
                        v.borrow()
                            .get(&self.id)
                            .map(|v| unsafe { v.downcast_ref_unchecked::<T>().clone() })
                    })
    }
}

/// A view into a thread's slot in a `ThreadLocal` that may be empty.
pub enum Entry<'a, T: 'static> {
    /// An occupied entry.
    Occupied(OccupiedEntry<'a, T>),
    /// A vacant entry.
    Vacant(VacantEntry<'a, T>),
}

impl<'a, T: 'static> Entry<'a, T> {
    /// Ensures a value is in the entry by inserting the default if it is empty, and returns a
    /// mutable reference to the value in the entry.
    pub fn or_insert(self, default: T) -> &'a mut T {
        match self {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e) => e.insert(default),
        }
    }

    /// Ensures a value is in the entry by inserting the result of the default function if it is
    /// empty, and returns a mutable reference to the value in the entry.
    pub fn or_insert_with<F>(self, default: F) -> &'a mut T
        where F: FnOnce() -> T
    {
        match self {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e) => e.insert(default()),
        }
    }
}

/// A view into a thread's slot in a `ThreadLocal` which is occupied.
pub struct OccupiedEntry<'a, T: 'static>(hash_map::OccupiedEntry<'a, usize, Box<UnsafeAny>>,
                                         PhantomData<&'a mut T>);

impl<'a, T: 'static> OccupiedEntry<'a, T> {
    /// Returns a reference to the value in the entry.
    pub fn get(&self) -> &T {
        unsafe { self.0.get().downcast_ref_unchecked() }
    }

    /// Returns a mutable reference to the value in the entry.
    pub fn get_mut(&mut self) -> &mut T {
        unsafe { self.0.get_mut().downcast_mut_unchecked() }
    }

    /// Converts an `OccupiedEntry` into a mutable reference to the value in the entry with a
    /// lifetime bound of the slot itself.
    pub fn into_mut(self) -> &'a mut T {
        unsafe { self.0.into_mut().downcast_mut_unchecked() }
    }

    /// Sets the value of the entry, and returns the entry's old value.
    pub fn insert(&mut self, value: T) -> T {
        mem::replace(self.get_mut(), value)
    }

    /// Takes the value out of the entry, and returns it.
    pub fn remove(self) -> T {
        unsafe { *self.0.remove().downcast_unchecked() }
    }
}

/// A view into a thread's slot in a `ThreadLocal` which is unoccupied.
pub struct VacantEntry<'a, T: 'static>(hash_map::VacantEntry<'a, usize, Box<UnsafeAny>>,
                                       PhantomData<&'a mut T>);

impl<'a, T: 'static> VacantEntry<'a, T> {
    /// Sets the value of the entry, and returns a mutable reference to it.
    pub fn insert(self, value: T) -> &'a mut T {
        unsafe { self.0.insert(Box::new(value)).downcast_mut_unchecked() }
    }
}
