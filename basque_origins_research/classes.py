import asyncio
import aiohttp
import json
import os
import sys
import time
import random
import re
import requests
from pathlib import Path
from datetime import datetime
import argparse
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import backoff
from collections import defaultdict, Counter
import itertools
from dataclasses import dataclass
import pickle
import hashlib
from scipy import stats
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/basque_research.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class CognateCandidate:
    """Represents a potential cognate relationship between words"""
    source_language: str
    target_language: str
    source_word: str
    target_word: str
    semantic_field: str
    sound_correspondence: str
    probability: float
    evidence: List[str]
    reconstructed_form: Optional[str] = None
    cognate_class: str = "unknown"  # true_cognate, false_cognate, loan, substrate

class PhylogeneticTreeBuilder:
    """Builds and analyzes phylogenetic trees for language families"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.language_data = {}
        self.reconstructed_nodes = {}
    
    def add_language(self, lang_code: str, family: str, features: Dict[str, Any]):
        """Add a language to the phylogenetic tree"""
        self.language_data[lang_code] = {
            'family': family,
            'features': features,
            'ancestors': [],
            'descendants': []
        }
        self.graph.add_node(lang_code, **features)
    
    def add_relationship(self, parent: str, child: str, time_depth: float, probability: float):
        """Add a parent-child relationship with time depth and probability"""
        self.graph.add_edge(parent, child, time_depth=time_depth, probability=probability)
        
        # Update ancestor/descendant relationships
        if child in self.language_data:
            self.language_data[child]['ancestors'].append(parent)
        if parent in self.language_data:
            self.language_data[parent]['descendants'].append(child)
    
    def find_common_ancestors(self, lang1: str, lang2: str) -> List[str]:
        """Find common ancestors between two languages"""
        if lang1 not in self.language_data or lang2 not in self.language_data:
            return []
        
        ancestors1 = set(self._get_all_ancestors(lang1))
        ancestors2 = set(self._get_all_ancestors(lang2))
        
        return list(ancestors1 & ancestors2)
    
    def _get_all_ancestors(self, lang: str) -> List[str]:
        """Recursively get all ancestors of a language"""
        ancestors = []
        current_ancestors = self.language_data[lang]['ancestors']
        
        for ancestor in current_ancestors:
            ancestors.append(ancestor)
            ancestors.extend(self._get_all_ancestors(ancestor))
        
        return ancestors
    
    def reconstruct_proto_language(self, lang_group: List[str], time_depth: float) -> Dict[str, Any]:
        """Reconstruct proto-language for a group of languages"""
        logger.info(f"Reconstructing proto-language for {lang_group} at {time_depth} years ago")
        
        # Collect features from all languages in the group
        feature_votes = defaultdict(Counter)
        feature_weights = defaultdict(float)
        
        for lang in lang_group:
            if lang in self.language_data:
                features = self.language_data[lang]['features']
                for feature, value in features.items():
                    if value is not None:
                        feature_votes[feature][value] += 1
                        feature_weights[feature] += 1.0  # Equal weight for now
        
        # Determine most common features
        reconstructed_features = {}
        for feature, votes in feature_votes.items():
            if feature_weights[feature] > 0:
                # Get the most voted value
                most_common = votes.most_common(1)[0][0]
                confidence = votes.most_common(1)[0][1] / feature_weights[feature]
                
                reconstructed_features[feature] = {
                    'value': most_common,
                    'confidence': confidence,
                    'votes': dict(votes)
                }
        
        return {
            'proto_language': f"Proto-{'-'.join(lang_group[:3])}",
            'time_depth': time_depth,
            'reconstructed_features': reconstructed_features,
            'languages_in_group': lang_group,
            'confidence': sum(f['confidence'] for f in reconstructed_features.values()) / len(reconstructed_features) if reconstructed_features else 0
        }
    
    def visualize_tree(self, output_path: str = "phylogenetic_tree.png"):
        """Visualize the phylogenetic tree"""
        plt.figure(figsize=(15, 10))
        
        # Try to use graphviz layout if available
        try:
            pos = graphviz_layout(self.graph, prog='dot')
        except:
            pos = nx.spring_layout(self.graph)
        
        # Draw nodes and edges
        nx.draw(self.graph, pos, with_labels=True, node_color='lightblue', 
                node_size=3000, font_size=8, font_weight='bold', arrows=True)
        
        plt.title("Phylogenetic Tree of Language Relationships")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Phylogenetic tree saved to {output_path}")

class CognateDetector:
    """Detects potential cognates between languages"""
    
    def __init__(self):
        self.sound_analyzer = None  # Will be initialized later
        self.cognate_candidates = []
    
    def calculate_semantic_similarity(self, word1: str, word2: str, semantic_field: str) -> float:
        """Calculate semantic similarity between words"""
        # Simple semantic similarity based on field
        if semantic_field in ["basic_vocabulary", "core_meaning"]:
            # Core vocabulary items have higher prior for cognacy
            return 0.8 if word1.lower() == word2.lower() else 0.1
        else:
            # Other fields have lower prior
            return 0.3 if word1.lower() == word2.lower() else 0.05
    
    def calculate_phonetic_similarity(self, word1: str, word2: str) -> float:
        """Calculate phonetic similarity between words"""
        # Calculate edit distance normalized by length
        max_len = max(len(word1), len(word2))
        if max_len == 0:
            return 1.0
        
        # Simple edit distance approximation
        common_chars = len(set(word1.lower()) & set(word2.lower()))
        char_similarity = common_chars / max(len(word1), len(word2))
        
        # Length similarity
        len_diff = abs(len(word1) - len(word2)) / max_len
        len_similarity = 1.0 - len_diff
        
        return (char_similarity + len_similarity) / 2
    
    def detect_potential_cognates(self, lang1_data: Dict[str, Any], lang2_data: Dict[str, Any], 
                                 lang1_code: str, lang2_code: str) -> List[CognateCandidate]:
        """Detect potential cognates between two languages"""
        candidates = []
        
        # Get word lists from both languages
        words1 = lang1_data.get('wordlist', {})
        words2 = lang2_data.get('wordlist', {})
        
        # Find overlapping semantic fields
        common_fields = set(words1.keys()) & set(words2.keys())
        
        for field in common_fields:
            word1 = words1[field]
            word2 = words2[field]
            
            # Calculate various similarity measures
            semantic_sim = self.calculate_semantic_similarity(word1, word2, field)
            phonetic_sim = self.calculate_phonetic_similarity(word1, word2)
            
            # Calculate sound correspondence probability
            sound_prob = 0.0  # Placeholder - would need sound analyzer
            if hasattr(self, 'sound_analyzer') and self.sound_analyzer:
                sound_prob = self.sound_analyzer.calculate_correspondence_probability(
                    lang1_code, word1[0] if word1 else '', lang2_code, word2[0] if word2 else ''
                )
            
            # Combine probabilities
            combined_prob = (semantic_sim * 0.4 + phonetic_sim * 0.4 + sound_prob * 0.2)
            
            if combined_prob > 0.3:  # Threshold for potential cognate
                evidence = []
                if semantic_sim > 0.5:
                    evidence.append(f"High semantic similarity ({semantic_sim:.2f})")
                if phonetic_sim > 0.5:
                    evidence.append(f"High phonetic similarity ({phonetic_sim:.2f})")
                if sound_prob > 0.3:
                    evidence.append(f"Regular sound correspondence ({sound_prob:.2f})")
                
                candidate = CognateCandidate(
                    source_language=lang1_code,
                    target_language=lang2_code,
                    source_word=word1,
                    target_word=word2,
                    semantic_field=field,
                    sound_correspondence=f"{word1[0] if word1 else '?'}->{word2[0] if word2 else '?'}",
                    probability=combined_prob,
                    evidence=evidence
                )
                
                candidates.append(candidate)
        
        return candidates

class BayesianPhylogeneticAnalyzer:
    """Performs Bayesian phylogenetic analysis for language divergence"""
    
    def __init__(self):
        self.divergence_times = {}
        self.confidence_intervals = {}
    
    def estimate_divergence_time(self, lang1: str, lang2: str, lexical_similarity: float, 
                                sound_correspondences: List[Tuple[str, str]], 
                                calibration_points: Dict[str, float]) -> Tuple[float, float, float]:
        """
        Estimate divergence time using Bayesian approach
        
        Returns: (estimated_time, lower_bound, upper_bound)
        """
        # Use lexical similarity to estimate time (simplified model)
        # Higher similarity suggests more recent divergence
        base_rate = 0.002  # 0.2% per year loss of shared vocabulary
        estimated_time = -np.log(lexical_similarity) / base_rate if lexical_similarity > 0 else float('inf')
        
        # Apply calibration points if available
        if lang1 in calibration_points:
            cal_time = calibration_points[lang1]
            estimated_time = (estimated_time + cal_time) / 2
        
        if lang2 in calibration_points:
            cal_time = calibration_points[lang2]
            estimated_time = (estimated_time + cal_time) / 2
        
        # Calculate confidence intervals based on sound correspondence regularity
        regularity = len([sc for sc in sound_correspondences if sc[0] != sc[1]]) / len(sound_correspondences) if sound_correspondences else 0.5
        ci_factor = 1.0 - regularity  # Less regular = wider CI
        
        lower_bound = estimated_time * (1 - ci_factor * 0.3)
        upper_bound = estimated_time * (1 + ci_factor * 0.3)
        
        return estimated_time, lower_bound, upper_bound