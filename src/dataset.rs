use ndarray::{Array2, Array1};
use std::path::Path;
use tch::Tensor;

#[derive(Debug)]
pub struct IrisDataset {
    pub train_features: Tensor,
    pub train_labels: Tensor,
    pub test_features: Tensor,
    pub test_labels: Tensor,
}

impl IrisDataset {
    pub fn load(test_ratio: f64) -> Self {
        // 内置鸢尾花数据集 (UCI标准数据)
        let data = include_str!("iris.data");
        
        let mut features = Vec::new();
        let mut labels = Vec::new();
        let mut label_map = std::collections::HashMap::new();
        label_map.insert("Iris-setosa", 0);
        label_map.insert("Iris-versicolor", 1);
        label_map.insert("Iris-virginica", 2);

        for line in data.lines() {
            if line.is_empty() { continue }
            let parts: Vec<_> = line.split(',').collect();
            let feature: Vec<f32> = parts[..4].iter().map(|x| x.parse().unwrap()).collect();
            features.push(feature);
            labels.push(*label_map.get(parts[4]).unwrap());
        }

        // 随机分割训练集/测试集
        let indices = shuffle((0..features.len()).collect());
        let split_idx = (features.len() as f64 * test_ratio) as usize;
        let test_indices = &indices[..split_idx];
        let train_indices = &indices[split_idx..];

        // 转换为Tensor
        Self {
            train_features: create_tensor(&features, train_indices),
            train_labels: create_label_tensor(&labels, train_indices, labels.len()),
            test_features: create_tensor(&features, test_indices),
            test_labels: create_label_tensor(&labels, test_indices, labels.len()),
        }
    }
}

fn create_tensor(data: &[Vec<f32>], indices: &[usize]) -> Tensor {
    let flat_data: Vec<f32> = indices.iter()
        .flat_map(|&i| data[i].iter().cloned())
        .collect();
    Tensor::of_slice(&flat_data).view((indices.len() as i64, 4))
}

fn create_label_tensor(labels: &[usize], indices: &[usize], num_classes: usize) -> Tensor {
    let labels: Vec<i64> = indices.iter().map(|&i| labels[i] as i64).collect();
    Tensor::of_slice(&labels) /*.one_hot(num_classes)*/ // 原始标签格式即可
}

fn shuffle<T>(mut items: Vec<T>) -> Vec<T> {
    use rand::seq::SliceRandom;
    let mut rng = rand::thread_rng();
    items.shuffle(&mut rng);
    items
}
