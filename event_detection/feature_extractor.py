import numpy as np
from transformers import pipeline
import torch
from sentence_transformers import SentenceTransformer, util
from config import MODEL_PATHS
from utils import clean_text

class NERFeatureExtractor:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = MODEL_PATHS["ner_model"]
        print("初始化NER模型...")
        try:
            self.ner_pipeline = pipeline(
                "token-classification",
                model=model_path,
                aggregation_strategy="simple",
                device=0 if torch.cuda.is_available() else -1
            )
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("GPU内存不足，切换到CPU模式...")
                torch.cuda.empty_cache()
                self.ner_pipeline = pipeline(
                    "token-classification",
                    model=model_path,
                    aggregation_strategy="simple",
                    device=-1
                )
            else:
                raise e
    
    def extract_entities(self, text):
        """提取文本中的人名、地名等实体"""
        if not text or len(text.strip()) == 0:
            return {'persons': [], 'locations': [], 'organizations': []}
        
        try:
            entities = self.ner_pipeline(text)
            persons = []
            locations = []
            organizations = []
            
            for entity in entities:
                if entity['entity_group'] == 'PER':
                    persons.append(entity['word'])
                elif entity['entity_group'] == 'LOC':
                    locations.append(entity['word'])
                elif entity['entity_group'] == 'ORG':
                    organizations.append(entity['word'])
            
            return {
                'persons': list(set(persons)),
                'locations': list(set(locations)),
                'organizations': list(set(organizations))
            }
        except Exception as e:
            print(f"NER处理错误: {e}, 文本: {text[:50]}...")
            return {'persons': [], 'locations': [], 'organizations': []}

def create_entity_vector(texts, entity_extractor, entity_vocab):
    """基于实体词汇表创建实体特征向量"""
    n_samples = len(texts)
    n_entities = len(entity_vocab)
    entity_vectors = np.zeros((n_samples, n_entities))
    
    entity_to_idx = {entity: idx for idx, entity in enumerate(entity_vocab)}
    
    print("提取实体特征...")
    for i, text in enumerate(texts):
        if i % 100 == 0:
            print(f"  处理进度: {i}/{len(texts)}")
            
        entities = entity_extractor.extract_entities(text)
        all_entities = entities['persons'] + entities['locations'] + entities['organizations']
        
        for entity in all_entities:
            if entity in entity_to_idx:
                entity_vectors[i, entity_to_idx[entity]] += 1.0
    
    return entity_vectors

def combine_features(text_embeddings, entity_vectors, entity_weight=0.15):
    """特征融合"""
    if entity_vectors.shape[1] == 0:
        return text_embeddings.copy()
    
    text_dim = text_embeddings.shape[1]
    ent_dim = entity_vectors.shape[1]
    target_dim = min(text_dim, ent_dim)
    
    if target_dim < text_dim:
        text_embeddings = text_embeddings[:, :target_dim]
    if target_dim < ent_dim:
        entity_vectors = entity_vectors[:, :target_dim]
    elif ent_dim < target_dim:
        padding = np.zeros((entity_vectors.shape[0], target_dim - ent_dim))
        entity_vectors = np.hstack([entity_vectors, padding])
    
    text_norm = text_embeddings / (np.linalg.norm(text_embeddings, axis=1, keepdims=True) + 1e-8)
    entity_norms = np.linalg.norm(entity_vectors, axis=1)
    valid_entity_mask = entity_norms > 0
    entity_norm = np.zeros_like(entity_vectors)
    entity_norm[valid_entity_mask] = entity_vectors[valid_entity_mask] / entity_norms[valid_entity_mask, np.newaxis]
    
    text_weight = 1 - entity_weight
    combined_embeddings = np.zeros_like(text_norm)
    for i in range(len(text_norm)):
        if valid_entity_mask[i]:
            combined_embeddings[i] = text_weight * text_norm[i] + entity_weight * entity_norm[i]
        else:
            combined_embeddings[i] = text_norm[i]
    
    combined_embeddings = combined_embeddings / (np.linalg.norm(combined_embeddings, axis=1, keepdims=True) + 1e-8)
    return combined_embeddings

def build_entity_vocab(texts, ner_extractor, sample_size=2000):
    """构建实体词汇表"""
    print("构建实体词汇表...")
    sample_texts = texts[:sample_size] if len(texts) > sample_size else texts
    all_entities = set()
    
    for i, text in enumerate(sample_texts):
        if i % 100 == 0:
            print(f"  采样进度: {i}/{len(sample_texts)}")
        entities = ner_extractor.extract_entities(text)
        all_entities.update(entities['persons'])
        all_entities.update(entities['locations'])
        all_entities.update(entities['organizations'])
    
    entity_vocab = [entity for entity in all_entities if len(entity) > 1]
    print(f"构建实体词汇表完成，共 {len(entity_vocab)} 个实体")
    return entity_vocab