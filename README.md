# LLM Project - Fine-tuning et Tokenization

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Overview

Ce projet démontre l'expertise en Large Language Models (LLMs) à travers plusieurs notebooks Jupyter couvrant le fine-tuning, la tokenization, et l'utilisation de modèles de pointe comme Mistral 7B. Ce projet fait partie du cours MVA (Master en Vision Artificielle) et illustre les techniques avancées de traitement du langage naturel.

## 🚀 Key Features

### Fine-tuning Mistral 7B
- **LoRA Fine-tuning** - Adaptation efficace de Mistral 7B avec LoRA (Low-Rank Adaptation)
- **Dataset Preparation** - Préparation et formatage des données pour l'entraînement
- **Model Download** - Téléchargement et configuration du modèle Mistral 7B v0.3
- **Training Pipeline** - Pipeline complet d'entraînement avec validation

### Tokenization et Transformers
- **Tokenization Analysis** - Analyse approfondie des techniques de tokenization
- **Transformer Architecture** - Implémentation et compréhension des Transformers
- **Text Processing** - Traitement et préparation des données textuelles

### Tests et Évaluation
- **Model Testing** - Tests complets des modèles fine-tunés
- **Performance Evaluation** - Évaluation des performances et métriques
- **Interactive Demos** - Démonstrations interactives des capacités

## 📁 Project Structure

```
LLM-project/
├── mistral_finetune_7b.ipynb      # Fine-tuning complet de Mistral 7B
├── LLM_Tokenization_Finetuning.ipynb  # Tokenization et fine-tuning
├── Test_LLM.ipynb                 # Tests et évaluation des modèles
├── Trasnformers.ipynb             # Architecture et implémentation Transformers
├── session2.ipynb                 # Session pratique supplémentaire
└── README.md                      # Documentation du projet
```

## 🛠️ Installation

### Prérequis
- Python 3.8+
- Jupyter Notebook ou JupyterLab
- Compte Google Colab Pro+ (recommandé pour l'entraînement)
- GPU A100 avec 40GB RAM (pour le fine-tuning)

### Installation des dépendances

```bash
# Cloner le repository
git clone https://github.com/arthurriche/LLM-project.git
cd LLM-project

# Installer les dépendances
pip install torch==2.2
pip install transformers
pip install datasets
pip install accelerate
pip install peft
pip install bitsandbytes
pip install huggingface_hub
```

## 📈 Quick Start

### 1. Fine-tuning Mistral 7B

```python
# Ouvrir le notebook principal
jupyter notebook mistral_finetune_7b.ipynb

# Ou utiliser Google Colab
# Le notebook inclut un lien direct vers Colab
```

### 2. Tokenization et Transformers

```python
# Analyser la tokenization
jupyter notebook LLM_Tokenization_Finetuning.ipynb

# Explorer l'architecture Transformers
jupyter notebook Trasnformers.ipynb
```

### 3. Tests et Évaluation

```python
# Tester les modèles
jupyter notebook Test_LLM.ipynb
```

## 🧮 Technical Implementation

### LoRA Fine-tuning
Le projet utilise LoRA (Low-Rank Adaptation) pour adapter efficacement Mistral 7B :

```python
# Configuration LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

### Dataset Preparation
Formatage des données selon les spécifications Mistral :

```python
# Format JSONL requis
{
    "messages": [
        {"role": "user", "content": "Question utilisateur"},
        {"role": "assistant", "content": "Réponse assistant"}
    ]
}
```

### Model Architecture
- **Mistral 7B v0.3** - Modèle de base avec 7 milliards de paramètres
- **Sliding Window Attention** - Mécanisme d'attention optimisé
- **Grouped-query Attention** - Attention par groupes pour l'efficacité

## 📊 Performance Metrics

### Fine-tuning Results
- **Training Loss**: Réduction progressive de la perte d'entraînement
- **Validation Loss**: Performance sur l'ensemble de validation
- **Memory Usage**: Optimisation de l'utilisation mémoire avec LoRA
- **Training Speed**: Accélération grâce aux techniques d'optimisation

### Model Evaluation
- **Perplexity**: Mesure de la qualité du langage généré
- **BLEU Score**: Évaluation de la traduction (si applicable)
- **Human Evaluation**: Évaluation qualitative des réponses

## 🔬 Advanced Features

### Memory Optimization
- **LoRA** - Réduction drastique de l'utilisation mémoire
- **Gradient Checkpointing** - Optimisation du gradient
- **Mixed Precision Training** - Entraînement en précision mixte

### Data Processing
- **Dynamic Padding** - Padding dynamique pour l'efficacité
- **Data Streaming** - Traitement en streaming pour les gros datasets
- **Multi-GPU Training** - Entraînement distribué

### Model Serving
- **Quantization** - Réduction de la taille du modèle
- **Inference Optimization** - Optimisation de l'inférence
- **API Integration** - Intégration API pour le déploiement

## 🚀 Deployment

### Local Deployment
```bash
# Sauvegarder le modèle fine-tuné
model.save_pretrained("./fine_tuned_mistral")

# Charger pour l'inférence
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("./fine_tuned_mistral")
```

### Cloud Deployment
- **Google Colab** - Pour le développement et les tests
- **AWS/GCP** - Pour le déploiement en production
- **Hugging Face Spaces** - Pour le partage et la démonstration

## 📚 Learning Resources

### Cours MVA
Ce projet fait partie du cours MVA sur les Large Language Models, couvrant :
- **Architecture des Transformers**
- **Techniques de Fine-tuning**
- **Optimisation des performances**
- **Applications pratiques**

### Documentation
- [Mistral AI Documentation](https://docs.mistral.ai/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [PEFT Documentation](https://huggingface.co/docs/peft/)

## 🤝 Contributing

1. Fork le repository
2. Créer une branche feature (`git checkout -b feature/amazing-feature`)
3. Commit les changements (`git commit -m 'Add amazing feature'`)
4. Push vers la branche (`git push origin feature/amazing-feature`)
5. Ouvrir une Pull Request

## 📝 License

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour plus de détails.

## 👨‍💻 Author

**Arthur Riche**
- LinkedIn: [Arthur Riche](https://www.linkedin.com/in/arthurriche/)
- Email: arthur.riche@example.com

## 🙏 Acknowledgments

- **Mistral AI** pour le modèle Mistral 7B
- **Hugging Face** pour les outils et bibliothèques
- **École MVA** pour le cadre académique
- **Google Colab** pour l'infrastructure de calcul

---

⭐ **Star ce repository si vous le trouvez utile !**

*Ce projet démontre l'expertise en Large Language Models et les techniques avancées de fine-tuning pour des applications pratiques.*
