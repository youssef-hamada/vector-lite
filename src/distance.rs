use crate::vector::Vector;

pub fn l2_squared(a: &Vector, b: &Vector) -> f32 {
    debug_assert_eq!(a.dim(), b.dim());
    a.data
        .iter()
        .zip(b.data.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn l2_correct() {
        let a = Vector::new(vec![0.0, 0.0]);
        let b = Vector::new(vec![3.0, 4.0]);
        assert!((l2_squared(&a, &b) - 25.0).abs() < 1e-6);
    }
}
