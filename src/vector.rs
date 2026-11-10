#[derive(Clone, Debug, PartialEq)]
pub struct Vector {
    pub data: Vec<f32>,
}

impl Vector {
    pub fn new(data: Vec<f32>) -> Self {
        Self { data }
    }
    pub fn dim(&self) -> usize {
        self.data.len()
    }
}
