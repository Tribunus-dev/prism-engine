//! Activation transaction (constitutional home, Metal FFI half).
//!
//! Per the inventory v2.1, the engine's activation_transaction.rs
//! is split: state half is `state::activation_transaction`; FFI
//! half is here. The FFI half owns the actual Metal activation
//! buffer handle; the state half owns the transaction guard.

pub struct ActivationTransactionFfi {
    _placeholder: (),
}

impl ActivationTransactionFfi {
    pub fn new() -> Self {
        Self { _placeholder: () }
    }
}

impl Default for ActivationTransactionFfi {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn activation_transaction_ffi_constructs() {
        let _ = ActivationTransactionFfi::new();
    }
}
