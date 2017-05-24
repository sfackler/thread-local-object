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

pub struct ThreadLocal<T: 'static> {
    id: usize,
    _p: PhantomData<T>,
}

impl<T: 'static> Drop for ThreadLocal<T> {
    fn drop(&mut self) {
        self.remove();
    }
}

impl<T: 'static> ThreadLocal<T> {
    pub fn new() -> ThreadLocal<T> {
        ThreadLocal {
            id: next_id(),
            _p: PhantomData,
        }
    }

    pub fn set(&mut self, value: T) -> Option<T> {
        self.entry(|e| match e {
                       Entry::Occupied(mut e) => Some(e.insert(value)),
                       Entry::Vacant(e) => {
                           e.insert(value);
                           None
                       }
                   })
    }

    pub fn remove(&mut self) -> Option<T> {
        VALUES.with(|v| {
                        v.borrow_mut()
                            .remove(&self.id)
                            .map(|v| unsafe { *v.downcast_unchecked::<T>() })
                    })
    }

    pub fn entry<F, R>(&mut self, f: F) -> R
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
}

pub enum Entry<'a, T> {
    Occupied(OccupiedEntry<'a, T>),
    Vacant(VacantEntry<'a, T>),
}

impl<'a, T: 'static> Entry<'a, T> {
    pub fn or_insert(self, default: T) -> &'a mut T {
        match self {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e) => e.insert(default),
        }
    }

    pub fn or_insert_with<F>(self, default: F) -> &'a mut T
        where F: FnOnce() -> T
    {
        match self {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e) => e.insert(default()),
        }
    }
}

pub struct OccupiedEntry<'a, T>(hash_map::OccupiedEntry<'a, usize, Box<UnsafeAny>>, PhantomData<T>);

impl<'a, T: 'static> OccupiedEntry<'a, T> {
    pub fn get(&self) -> &T {
        unsafe { self.0.get().downcast_ref_unchecked() }
    }

    pub fn get_mut(&mut self) -> &mut T {
        unsafe { self.0.get_mut().downcast_mut_unchecked() }
    }

    pub fn into_mut(self) -> &'a mut T {
        unsafe { self.0.into_mut().downcast_mut_unchecked() }
    }

    pub fn insert(&mut self, value: T) -> T {
        mem::replace(self.get_mut(), value)
    }

    pub fn remove(self) -> T {
        unsafe { *self.0.remove().downcast_unchecked() }
    }
}

pub struct VacantEntry<'a, T>(hash_map::VacantEntry<'a, usize, Box<UnsafeAny>>, PhantomData<T>);

impl<'a, T: 'static> VacantEntry<'a, T> {
    pub fn insert(self, value: T) -> &'a mut T {
        unsafe { self.0.insert(Box::new(value)).downcast_mut_unchecked() }
    }
}
