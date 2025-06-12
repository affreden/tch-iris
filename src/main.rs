use tch::{
    nn, nn::Module, nn::OptimizerConfig, Device, Kind, Tensor,
    kind::Kind::Float
};
use std::{error::Error, fs};


// 为Net结构体添加#[derive(Debug)]
#[derive(Debug)]
struct Net {
    fc1: nn::Linear,
    fc2: nn::Linear,
    fc3: nn::Linear,
}



impl Net {
    fn new(vs: &nn::Path) -> Self {
        Self {
            fc1: nn::linear(vs / "fc1", 4, 64, nn::LinearConfig {
                ws_init: nn::Init::Uniform { lo: -1.0, up: 1.0 },
                ..Default::default()
            }),
            fc2: nn::linear(vs / "fc2", 64, 16, nn::LinearConfig {
                ws_init: nn::Init::Uniform { lo: -1.0, up: 1.0 },
                ..Default::default()
            }),
            fc3: nn::linear(vs / "fc3", 16, 3, nn::LinearConfig {
                ws_init: nn::Init::Uniform { lo: -1.0, up: 1.0 },
                ..Default::default()
            }),
        }
    }
}

impl Module for Net {
    fn forward(&self, x: &Tensor) -> Tensor {
        x.apply(&self.fc1)
            .relu()
            .dropout(0.1, true) // 添加第二个参数true表示在训练模式
            .apply(&self.fc2)
            .relu()
            .apply(&self.fc3)
    }
}

struct IrisDataset {
    train_features: Tensor,
    train_labels: Tensor,
    test_features: Tensor,
    test_labels: Tensor,
}

impl IrisDataset {
    fn loadall(data_path: &str, device: Device) -> Result<Self, Box<dyn Error>> {
        let data_content = fs::read_to_string(data_path)?;
        let mut features = Vec::new();
        let mut labels = Vec::new();
        let mut categories = std::collections::HashMap::new();
        categories.insert("Iris-setosa", 0);
        categories.insert("Iris-versicolor", 1);
        categories.insert("Iris-virginica", 2);
        for line in data_content.lines() {
            if line.trim().is_empty() { continue; }
            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() < 5 { continue; }
            let mut feature_vec = Vec::with_capacity(4);
            for i in 0..4 {
                feature_vec.push(parts[i].parse::<f32>()?);
            }
            let label = *categories.get(parts[4]).unwrap();
            
            features.push(feature_vec);
            labels.push(label);
            
        }
        // 创建 Tensor 时直接放置到指定的设备上
        let features_to_device = |f: &[Vec<f32>]| -> Tensor {
            let flat: Vec<f32> = f.iter()
                .flat_map(|v| v.iter().copied())
                .collect();
            Tensor::from_slice(&flat)
                .to_device(device)
                .view((f.len() as i64, 4))
        };
        
        let labels_to_device = |l: &[i64]| -> Tensor {
            Tensor::from_slice(l)
                .to_device(device)
        };
        Ok(IrisDataset {
            train_features: features_to_device(&features),
            train_labels: labels_to_device(&labels),
            test_features: features_to_device(&features),
            test_labels: labels_to_device(&labels),
        })
        
    }
    fn load(test_ratio: f32, data_path: &str, device: Device) -> Result<Self, Box<dyn Error>> {
        let data_content = fs::read_to_string(data_path)?;
        let mut features = Vec::new();
        let mut labels = Vec::new();
        let mut categories = std::collections::HashMap::new();
        categories.insert("Iris-setosa", 0);
        categories.insert("Iris-versicolor", 1);
        categories.insert("Iris-virginica", 2);
        for line in data_content.lines() {
            if line.trim().is_empty() { continue; }
            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() < 5 { continue; }
            let mut feature_vec = Vec::with_capacity(4);
            for i in 0..4 {
                feature_vec.push(parts[i].parse::<f32>()?);
            }
            let label = *categories.get(parts[4]).unwrap();
            
            features.push(feature_vec);
            labels.push(label);
        }
        Self::shuffle_data(&mut features, &mut labels);
        
        let total = features.len() as f32;
        let split_idx = (total * test_ratio) as usize;
        
        // 创建 Tensor 时直接放置到指定的设备上
        let features_to_device = |f: &[Vec<f32>]| -> Tensor {
            let flat: Vec<f32> = f.iter()
                .flat_map(|v| v.iter().copied())
                .collect();
            Tensor::from_slice(&flat)
                .to_device(device)
                .view((f.len() as i64, 4))
        };
        
        let labels_to_device = |l: &[i64]| -> Tensor {
            Tensor::from_slice(l)
                .to_device(device)
        };
        Ok(IrisDataset {
            train_features: features_to_device(&features[split_idx..]),
            train_labels: labels_to_device(&labels[split_idx..]),
            test_features: features_to_device(&features[..split_idx]),
            test_labels: labels_to_device(&labels[..split_idx]),
        })
    }
    fn shuffle_data(features: &mut Vec<Vec<f32>>, labels: &mut Vec<i64>) {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        
        let mut indices: Vec<usize> = (0..features.len()).collect();
        indices.shuffle(&mut rng);
        
        let new_features = indices.iter().map(|&i| features[i].clone()).collect();
        let new_labels = indices.iter().map(|&i| labels[i]).collect();
        
        *features = new_features;
        *labels = new_labels;
    }
}

fn set_rand_seed(seed: u64) {
    use rand::SeedableRng;
    rand::rngs::StdRng::seed_from_u64(seed);
    tch::manual_seed(seed.try_into().unwrap());
}

fn main() -> Result<(), Box<dyn Error>> {
    // 1. 添加库初始化
    tch::maybe_init_cuda();

    set_rand_seed(3);
    
    // 2. 检查CUDA可用性
    let device = Device::cuda_if_available();
    println!("🚀 运行设备: {:?}", device);
    
    // 3. 加载数据集
    let dataset_path = "src/iris.data";
    println!("📊 从 '{}' 加载数据集...", dataset_path);
    // let dataset = IrisDataset::load(0.5, dataset_path,device)?;
    let dataset = IrisDataset::loadall(dataset_path,device)?;

    
    // 4. 创建模型
    let vs = nn::VarStore::new(device);
    let net = Net::new(&vs.root());
    let mut opt = nn::Adam::default().build(&vs, 1e-5)?;
    
    // 5. 训练循环
    println!("🔁 开始训练 (共300轮)");
    for epoch in 1..=300 {
        // 前向传播
        let logits = net.forward(&dataset.train_features);
        let loss = logits.cross_entropy_for_logits(&dataset.train_labels);
        
        // 反向传播
        opt.backward_step(&loss);
        
        // 每30轮输出指标
        if epoch % 30 == 0 {
            // 修正准确率计算
            let train_preds = logits.argmax(-1, true);
            let train_eq = train_preds.eq_tensor(&dataset.train_labels);
            let train_accuracy = train_eq
                .to_kind(Float)
                .mean_dim(Some([0].as_ref()), false, Kind::Float)
                .double_value(&[0]);
            
            let test_logits = net.forward(&dataset.test_features);
            let test_preds = test_logits.argmax(-1, true);
            let test_eq = test_preds.eq_tensor(&dataset.test_labels);
            let test_accuracy = test_eq
                .to_kind(Float)
                .mean_dim(Some([0].as_ref()), false, Kind::Float)
                .double_value(&[0]);
            
            println!(
                "Epoch {:03}: loss={:.4} | 训练集准确率={:.1}% | 测试集准确率={:.1}%",
                epoch,
                loss.double_value(&[]),
                train_accuracy * 100.0,
                test_accuracy * 100.0
            );
        }
    }
    
    // 6. 保存模型
    vs.save("iris_model.ot")?;
    println!("✅ 训练完成! 模型已保存至 'iris_model.ot'");
    
    // 7. 测试推断
    // 修复创建样本的方式
    let sample: Vec<f32> = vec![5.1, 3.5, 1.4, 0.2];
    let sample_tensor = Tensor::f_from_slice(&sample)
        .expect("Invalid data")
        .to_device(device)
        .view([1, 4]);
    
    let predict = net.forward(&sample_tensor).softmax(1, Float);
    println!("\n🌸 预测示例:");
    println!("输入特征: {:?}", sample);
    
    // 获取预测概率
    let probs: Vec<f64> = (0..3).map(|i| 
        predict.double_value(&[0, i as i64])
    ).collect();
    
    println!("类别概率: {:.1}% setosa, {:.1}% versicolor, {:.1}% virginica",
        probs[0] * 100.0,
        probs[1] * 100.0,
        probs[2] * 100.0
    );
    
    Ok(())
}
