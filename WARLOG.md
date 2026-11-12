## Day 1 - [today's date]

- Wrote full HNSW implementation (~350 lines of Rust)
- 4 unit tests pass, recall@10 = 1.000 on 1,000 vectors × 20 queries
- Fought through: linker, dbghelp, encoding, 5 clippy lints
- CI green ✅
- Next: WAL with CRC for crash-safe writes

## Time spent: ~2.5 hours (setup + code + debugging)
