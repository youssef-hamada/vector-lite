//! Hierarchical Navigable Small World graph.
//!
//! Multi-layer graph for approximate nearest neighbor search.
//! Insert walks down from the top layer; search does the same.

use crate::distance::l2_squared;
use crate::vector::Vector;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

#[derive(Clone, Debug)]
pub struct HnswConfig {
    /// Max edges per node per upper layer.
    pub m: usize,
    /// Max edges per node at layer 0 (usually 2 * m).
    pub m0: usize,
    /// Beam width during insertion.
    pub ef_construction: usize,
    /// Beam width during search.
    pub ef_search: usize,
    /// Seed for deterministic level assignment (needed for tests).
    pub seed: u64,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 16,
            m0: 32,
            ef_construction: 200,
            ef_search: 64,
            seed: 42,
        }
    }
}

/// Wrapper for min-heap ordering by distance.
#[derive(PartialEq)]
struct Candidate(f32, u64);

impl Eq for Candidate {}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reversed: BinaryHeap is a max-heap, we want min by distance.
        other.0.partial_cmp(&self.0).unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub struct Hnsw {
    pub config: HnswConfig,
    pub vectors: HashMap<u64, Vector>,
    /// layers[layer_id] maps node_id -> list of neighbor ids
    pub layers: Vec<HashMap<u64, Vec<u64>>>,
    pub entry_point: Option<u64>,
    pub max_layer: usize,
    rng_state: u64,
}

impl Hnsw {
    pub fn new(config: HnswConfig) -> Self {
        Self {
            rng_state: config.seed,
            config,
            vectors: HashMap::new(),
            layers: vec![HashMap::new()],
            entry_point: None,
            max_layer: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Simple xorshift64 RNG — deterministic, seeded, no thread-local.
    fn next_rand(&mut self) -> u64 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        x
    }

    /// Assign a level to a new node: exponentially distributed, mean = 1/ln(m).
    fn random_level(&mut self) -> usize {
        let ml = 1.0 / (self.config.m as f64).ln();
        // Use next_rand() as uniform [0,1)
        let u = (self.next_rand() >> 11) as f64 / (1u64 << 53) as f64;
        let u = u.max(1e-12);
        (-u.ln() * ml) as usize
    }

    /// Insert a vector into the index.
    pub fn insert(&mut self, id: u64, vector: Vector) {
        if self.vectors.contains_key(&id) {
            // Overwrite: just replace the vector.
            self.vectors.insert(id, vector);
            return;
        }

        self.vectors.insert(id, vector.clone());
        let level = self.random_level();

        // First node ever
        if self.entry_point.is_none() {
            self.entry_point = Some(id);
            self.max_layer = level;
            while self.layers.len() <= level {
                self.layers.push(HashMap::new());
            }
            for l in 0..=level {
                self.layers[l].insert(id, Vec::new());
            }
            return;
        }

        let mut current = self.entry_point.unwrap();
        let mut current_layer = self.max_layer;

        // Descend from top to level+1, greedy walk
        while current_layer > level {
            current = self.greedy_search(&vector, current, current_layer);
            current_layer -= 1;
        }

        // From min(level, max_layer) down to 0: connect
        let mut layer = level.min(self.max_layer);
        loop {
            let candidates = self.search_layer(&vector, current, self.config.ef_construction, layer);
            let max_m = if layer == 0 {
                self.config.m0
            } else {
                self.config.m
            };
            let neighbors: Vec<u64> = candidates
                .iter()
                .take(max_m)
                .map(|(_, id)| *id)
                .collect();

            self.layers[layer].insert(id, neighbors.clone());

            // Add reverse edges + prune
            for &n in &neighbors {
                let entry = self.layers[layer].entry(n).or_insert_with(Vec::new);
                entry.push(id);
                if entry.len() > max_m {
                    // Keep max_m nearest to n
                    let nv = self.vectors[&n].clone();
                    entry.sort_by(|a, b| {
                        l2_squared(&nv, &self.vectors[a])
                            .partial_cmp(&l2_squared(&nv, &self.vectors[b]))
                            .unwrap()
                    });
                    entry.truncate(max_m);
                }
            }

            if let Some(first) = candidates.first() {
                current = first.1;
            }
            if layer == 0 {
                break;
            }
            layer -= 1;
        }

        // If our level exceeds current max, we become the new entry point
        if level > self.max_layer {
            self.max_layer = level;
            self.entry_point = Some(id);
            while self.layers.len() <= level {
                self.layers.push(HashMap::new());
            }
            for l in 0..=level {
                self.layers[l].entry(id).or_insert_with(Vec::new);
            }
        }
    }

    /// Greedy nearest-neighbor walk at a single layer.
    pub fn greedy_search(&self, query: &Vector, entry: u64, layer: usize) -> u64 {
        let mut best = entry;
        let mut best_dist = l2_squared(query, &self.vectors[&entry]);
        loop {
            let mut changed = false;
            if let Some(neighbors) = self.layers[layer].get(&best) {
                for &n in neighbors {
                    let d = l2_squared(query, &self.vectors[&n]);
                    if d < best_dist {
                        best = n;
                        best_dist = d;
                        changed = true;
                    }
                }
            }
            if !changed {
                return best;
            }
        }
    }

    /// Beam search at a single layer. Returns (distance, id) pairs, sorted ascending.
    pub fn search_layer(
        &self,
        query: &Vector,
        entry: u64,
        ef: usize,
        layer: usize,
    ) -> Vec<(f32, u64)> {
        let mut visited: HashSet<u64> = HashSet::new();
        visited.insert(entry);
        let d0 = l2_squared(query, &self.vectors[&entry]);
        let mut candidates: BinaryHeap<Candidate> = BinaryHeap::new();
        candidates.push(Candidate(d0, entry));
        let mut results: Vec<(f32, u64)> = vec![(d0, entry)];

        while let Some(Candidate(d, id)) = candidates.pop() {
            let worst = results
                .iter()
                .map(|(d, _)| *d)
                .fold(f32::NEG_INFINITY, f32::max);
            if results.len() >= ef && d > worst {
                break;
            }
            if let Some(neighbors) = self.layers[layer].get(&id) {
                for &n in neighbors {
                    if !visited.insert(n) {
                        continue;
                    }
                    let dn = l2_squared(query, &self.vectors[&n]);
                    let worst = results
                        .iter()
                        .map(|(d, _)| *d)
                        .fold(f32::NEG_INFINITY, f32::max);
                    if results.len() < ef || dn < worst {
                        candidates.push(Candidate(dn, n));
                        results.push((dn, n));
                        if results.len() > ef {
                            results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                            results.truncate(ef);
                        }
                    }
                }
            }
        }
        results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        results
    }

    /// Top-k search.
    pub fn search(&self, query: &Vector, k: usize) -> Vec<(f32, u64)> {
        let entry = match self.entry_point {
            None => return vec![],
            Some(e) => e,
        };
        let mut current = entry;
        // Descend layers
        for layer in (1..=self.max_layer).rev() {
            current = self.greedy_search(query, current, layer);
        }
        // Final beam search at layer 0
        let mut results = self.search_layer(query, current, self.config.ef_search.max(k), 0);
        results.truncate(k);
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng}; 

    fn random_vectors(n: usize, dim: usize, seed: u64) -> Vec<(u64, Vector)> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        (0..n as u64)
            .map(|i| {
                let v: Vec<f32> = (0..dim).map(|_| rng.gen()).collect();
                (i, Vector::new(v))
            })
            .collect()
    }

    #[test]
    fn insert_single_vector() {
        let mut h = Hnsw::new(HnswConfig::default());
        h.insert(1, Vector::new(vec![1.0, 2.0, 3.0]));
        assert_eq!(h.len(), 1);
        assert_eq!(h.entry_point, Some(1));
    }

    #[test]
    fn insert_and_find_exact() {
        let mut h = Hnsw::new(HnswConfig::default());
        h.insert(1, Vector::new(vec![0.0, 0.0]));
        h.insert(2, Vector::new(vec![10.0, 10.0]));
        h.insert(3, Vector::new(vec![5.0, 5.0]));

        let results = h.search(&Vector::new(vec![0.1, 0.1]), 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1, 1, "should find id=1 as nearest");
    }

    #[test]
    fn recall_on_1k_vectors() {
        let mut h = Hnsw::new(HnswConfig::default());
        let data = random_vectors(1000, 32, 7);
        for (id, v) in &data {
            h.insert(*id, v.clone());
        }

        let queries = random_vectors(20, 32, 99);
        let mut hits = 0;
        for (_, q) in &queries {
            // Exact ground truth
            let mut exact: Vec<(f32, u64)> = data
                .iter()
                .map(|(id, v)| (l2_squared(q, v), *id))
                .collect();
            exact.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let truth: HashSet<u64> = exact.iter().take(10).map(|(_, id)| *id).collect();

            // HNSW
            let got: HashSet<u64> = h.search(q, 10).iter().map(|(_, id)| *id).collect();
            hits += truth.intersection(&got).count();
        }
        let recall = hits as f32 / (20 * 10) as f32;
        println!("recall@10 = {recall:.3}");
        assert!(
            recall >= 0.90,
            "recall too low: {recall:.3} (need >= 0.90)"
        );
    }
}
