use crate::registry::types::*;
use sqlite::Value;

pub struct HardenedLedger {
    pub connection: std::sync::Mutex<sqlite::Connection>,
    pub checkpoint_sequence: std::sync::atomic::AtomicU64,
}

impl HardenedLedger {
    pub fn new(path: &str) -> Result<Self, String> {
        let conn = sqlite::open(path).map_err(|e| e.to_string())?;
        // Create tables
        conn.execute(
            "CREATE TABLE IF NOT EXISTS execution_receipts_v1 (
                id BLOB PRIMARY KEY,
                parent_id BLOB,
                contract_hash BLOB,
                output_hash BLOB,
                bytes BLOB,
                created_at INTEGER
            )",
        )
        .map_err(|e| e.to_string())?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS ledger_tail_v1 (
                hash BLOB PRIMARY KEY
            )",
        )
        .map_err(|e| e.to_string())?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS checkpoints_v1 (
                sequence INTEGER PRIMARY KEY,
                tail_digest BLOB,
                checkpoint_signature BLOB,
                created_at INTEGER
            )",
        )
        .map_err(|e| e.to_string())?;
        // Ensure tail row exists
        conn.execute("INSERT OR IGNORE INTO ledger_tail_v1 (hash) VALUES (X'00')")
            .map_err(|e| e.to_string())?;
        Ok(Self {
            connection: std::sync::Mutex::new(conn),
            checkpoint_sequence: std::sync::atomic::AtomicU64::new(0),
        })
    }

    pub fn log_execution(&self, mut receipt: ExecutionReceipt) -> Result<Digest256, String> {
        let tx = self.connection.lock().map_err(|e| e.to_string())?;
        // Read current tail
        let mut parent_digest: Option<Digest256> = None;
        let mut stmt = tx
            .prepare("SELECT hash FROM ledger_tail_v1 LIMIT 1")
            .map_err(|e| e.to_string())?;
        if let Ok(sqlite::State::Row) = stmt.next() {
            let bytes: Vec<u8> = stmt.read::<Vec<u8>, usize>(0).unwrap_or_default();
            if bytes.len() == 32 {
                let mut arr = [0u8; 32];
                arr.copy_from_slice(&bytes);
                parent_digest = Some(Digest256(arr));
            }
        }
        drop(stmt);

        receipt.previous_receipt_digest = parent_digest;
        let final_digest = receipt.compute_canonical_digest();
        receipt.receipt_id = ReceiptId(final_digest);

        // Insert receipt — use Value enum for mixed-type binding
        let receipt_bytes = serde_json::to_vec(&receipt).map_err(|e| e.to_string())?;
        let now_secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        let mut stmt = tx.prepare(
            "INSERT INTO execution_receipts_v1 (id, parent_id, contract_hash, output_hash, bytes, created_at) VALUES (?, ?, ?, ?, ?, ?)"
        ).map_err(|e| e.to_string())?;

        stmt.bind::<&[(_, Value)]>(&[
            (1_usize, Value::Binary(final_digest.as_bytes().to_vec())),
            (
                2_usize,
                match parent_digest {
                    Some(d) => Value::Binary(d.as_bytes().to_vec()),
                    None => Value::Null,
                },
            ),
            (
                3_usize,
                Value::Binary(receipt.deployment_digest.as_bytes().to_vec()),
            ),
            (
                4_usize,
                Value::Binary(receipt.output_digest.as_bytes().to_vec()),
            ),
            (5_usize, Value::Binary(receipt_bytes)),
            (6_usize, Value::Integer(now_secs)),
        ])
        .map_err(|e| e.to_string())?;
        stmt.next().map_err(|e| e.to_string())?;

        // Update tail
        drop(stmt);
        let mut stmt = tx
            .prepare("UPDATE ledger_tail_v1 SET hash = ?")
            .map_err(|e| e.to_string())?;
        stmt.bind::<&[(_, Value)]>(&[(1_usize, Value::Binary(final_digest.as_bytes().to_vec()))])
            .map_err(|e| e.to_string())?;
        stmt.next().map_err(|e| e.to_string())?;

        Ok(final_digest)
    }

    pub fn issue_secure_checkpoint(&self, tail_digest: Digest256) -> Result<(), String> {
        let sequence = self
            .checkpoint_sequence
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut payload = Vec::new();
        payload.extend_from_slice(b"prism.checkpoint.v1\0");
        payload.extend_from_slice(tail_digest.as_bytes());
        payload.extend_from_slice(&sequence.to_be_bytes());

        let signature = PlatformSecureSigner::sign(&payload).map_err(|e| format!("{e}"))?;
        PlatformSecureSigner::persist_checkpoint_record(sequence, tail_digest, signature)
            .map_err(|e| format!("{e}"))
    }
}
