#!/usr/bin/env python3
"""
Knowledge Base Diagnostic and Fix Tool

This script diagnoses and fixes issues with the vector search functionality
including data structure problems and search parameter tuning.
"""

import json
import os
import logging
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import faiss

class KnowledgeBaseDiagnostic:
    """Comprehensive diagnostic tool for the knowledge base"""
    
    def __init__(self, 
                 index_path: str = 'knowledge_base/vector_store/faiss_index',
                 metadata_path: str = 'knowledge_base/vector_store/metadata.json'):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.logger = logging.getLogger(__name__)
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
    def run_full_diagnostic(self) -> Dict[str, Any]:
        """Run comprehensive diagnostic checks"""
        results = {
            'file_status': self.check_file_existence(),
            'data_structure': self.analyze_data_structure(),
            'index_integrity': self.check_index_integrity(),
            'content_quality': self.analyze_content_quality(),
            'search_parameters': self.test_search_parameters(),
            'recommendations': []
        }
        
        # Generate recommendations based on findings
        results['recommendations'] = self.generate_recommendations(results)
        
        return results
    
    def check_file_existence(self) -> Dict[str, Any]:
        """Check if required files exist and are accessible"""
        status = {
            'index_file_exists': os.path.exists(f"{self.index_path}.index"),
            'metadata_file_exists': os.path.exists(self.metadata_path),
            'files_readable': True,
            'file_sizes': {}
        }
        
        try:
            if status['index_file_exists']:
                status['file_sizes']['index'] = os.path.getsize(f"{self.index_path}.index")
            
            if status['metadata_file_exists']:
                status['file_sizes']['metadata'] = os.path.getsize(self.metadata_path)
                
        except Exception as e:
            status['files_readable'] = False
            status['error'] = str(e)
            
        return status
    
    def analyze_data_structure(self) -> Dict[str, Any]:
        """Analyze the structure and content of stored documents"""
        analysis = {
            'total_documents': 0,
            'content_types': {},
            'sources': {},
            'empty_content_count': 0,
            'sample_documents': [],
            'data_issues': []
        }
        
        try:
            if not os.path.exists(self.metadata_path):
                analysis['data_issues'].append("Metadata file does not exist")
                return analysis
            
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
            
            analysis['total_documents'] = len(metadata)
            
            for i, doc in enumerate(metadata):
                # Count content types and sources
                content_type = doc.get('content_type', 'unknown')
                source = doc.get('source', 'unknown')
                
                analysis['content_types'][content_type] = analysis['content_types'].get(content_type, 0) + 1
                analysis['sources'][source] = analysis['sources'].get(source, 0) + 1
                
                # Check for empty content
                text = doc.get('text', '')
                content = doc.get('content', '')
                
                if not text and not content:
                    analysis['empty_content_count'] += 1
                    analysis['data_issues'].append(f"Document {i} has no searchable text content")
                
                # Collect sample documents for inspection
                if i < 3:
                    analysis['sample_documents'].append({
                        'id': doc.get('id', i),
                        'content_type': content_type,
                        'source': source,
                        'has_text': bool(text),
                        'has_content': bool(content),
                        'text_length': len(text) if text else 0,
                        'text_preview': text[:200] if text else "No text content"
                    })
                    
        except Exception as e:
            analysis['data_issues'].append(f"Error reading metadata: {str(e)}")
            
        return analysis
    
    def check_index_integrity(self) -> Dict[str, Any]:
        """Check FAISS index integrity and dimensions"""
        integrity = {
            'index_loaded': False,
            'total_vectors': 0,
            'vector_dimension': 0,
            'index_type': 'unknown',
            'index_issues': []
        }
        
        try:
            if os.path.exists(f"{self.index_path}.index"):
                index = faiss.read_index(f"{self.index_path}.index")
                integrity['index_loaded'] = True
                integrity['total_vectors'] = index.ntotal
                integrity['vector_dimension'] = index.d
                integrity['index_type'] = type(index).__name__
                
                # Check if vectors and metadata count match
                if os.path.exists(self.metadata_path):
                    with open(self.metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    if len(metadata) != index.ntotal:
                        integrity['index_issues'].append(
                            f"Mismatch: {len(metadata)} metadata entries vs {index.ntotal} vectors"
                        )
                        
            else:
                integrity['index_issues'].append("Index file does not exist")
                
        except Exception as e:
            integrity['index_issues'].append(f"Error loading index: {str(e)}")
            
        return integrity
    
    def analyze_content_quality(self) -> Dict[str, Any]:
        """Analyze the quality and searchability of content"""
        quality = {
            'avg_content_length': 0,
            'content_length_distribution': {'short': 0, 'medium': 0, 'long': 0},
            'language_diversity': {},
            'keyword_coverage': {},
            'quality_issues': []
        }
        
        try:
            if not os.path.exists(self.metadata_path):
                return quality
                
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
            
            total_length = 0
            keyword_sets = {
                'financial': ['earnings', 'revenue', 'profit', 'financial', 'market', 'stock'],
                'asian_companies': ['tsmc', 'samsung', 'alibaba', 'sony', 'softbank', 'tencent'],
                'tech_terms': ['semiconductor', 'technology', 'chip', 'innovation', 'digital']
            }
            
            for doc in metadata:
                text = doc.get('text', '')
                content = doc.get('content', '')
                full_text = f"{text} {content}".lower()
                
                length = len(full_text)
                total_length += length
                
                # Categorize by length
                if length < 100:
                    quality['content_length_distribution']['short'] += 1
                elif length < 500:
                    quality['content_length_distribution']['medium'] += 1
                else:
                    quality['content_length_distribution']['long'] += 1
                
                # Check keyword coverage
                for category, keywords in keyword_sets.items():
                    if category not in quality['keyword_coverage']:
                        quality['keyword_coverage'][category] = 0
                    
                    if any(keyword in full_text for keyword in keywords):
                        quality['keyword_coverage'][category] += 1
            
            if metadata:
                quality['avg_content_length'] = total_length / len(metadata)
            
            # Identify quality issues
            short_content_ratio = quality['content_length_distribution']['short'] / len(metadata)
            if short_content_ratio > 0.5:
                quality['quality_issues'].append(
                    f"High ratio of short content: {short_content_ratio:.2%}"
                )
                
        except Exception as e:
            quality['quality_issues'].append(f"Error analyzing content quality: {str(e)}")
            
        return quality
    
    def test_search_parameters(self) -> Dict[str, Any]:
        """Test different search parameters and thresholds"""
        test_results = {
            'threshold_tests': [],
            'query_tests': [],
            'embedding_tests': [],
            'parameter_recommendations': []
        }
        
        try:
            # Load index and metadata
            if not os.path.exists(f"{self.index_path}.index") or not os.path.exists(self.metadata_path):
                test_results['parameter_recommendations'].append("Cannot test - missing files")
                return test_results
            
            index = faiss.read_index(f"{self.index_path}.index")
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Test different similarity thresholds
            test_query = "TSMC earnings financial results"
            query_embedding = self.embedding_model.encode([test_query], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)
            
            # Get all similarities
            scores, indices = index.search(query_embedding, min(50, len(metadata)))
            
            for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
                results_count = sum(1 for score in scores[0] if score >= threshold)
                test_results['threshold_tests'].append({
                    'threshold': threshold,
                    'results_count': results_count,
                    'max_score': float(scores[0].max()) if len(scores[0]) > 0 else 0.0
                })
            
            # Test different query types
            test_queries = [
                "TSMC earnings",
                "financial news Asia",
                "market data technology",
                "semiconductor industry"
            ]
            
            for query in test_queries:
                q_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
                faiss.normalize_L2(q_embedding)
                q_scores, q_indices = index.search(q_embedding, 10)
                
                test_results['query_tests'].append({
                    'query': query,
                    'top_score': float(q_scores[0][0]) if len(q_scores[0]) > 0 else 0.0,
                    'results_above_03': sum(1 for score in q_scores[0] if score >= 0.3)
                })
                
        except Exception as e:
            test_results['parameter_recommendations'].append(f"Error testing parameters: {str(e)}")
            
        return test_results
    
    def generate_recommendations(self, diagnostic_results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on diagnostic results"""
        recommendations = []
        
        # File existence issues
        file_status = diagnostic_results.get('file_status', {})
        if not file_status.get('index_file_exists') or not file_status.get('metadata_file_exists'):
            recommendations.append("CRITICAL: Missing index or metadata files - run knowledge base setup")
        
        # Data structure issues
        data_structure = diagnostic_results.get('data_structure', {})
        if data_structure.get('total_documents', 0) == 0:
            recommendations.append("CRITICAL: No documents in knowledge base - check data collection pipeline")
        
        if data_structure.get('empty_content_count', 0) > 0:
            recommendations.append(
                f"WARNING: {data_structure['empty_content_count']} documents have no searchable content"
            )
        
        # Index integrity issues
        index_integrity = diagnostic_results.get('index_integrity', {})
        if index_integrity.get('index_issues'):
            for issue in index_integrity['index_issues']:
                recommendations.append(f"INDEX ISSUE: {issue}")
        
        # Search parameter issues
        search_params = diagnostic_results.get('search_parameters', {})
        threshold_tests = search_params.get('threshold_tests', [])
        
        if threshold_tests:
            # Find optimal threshold
            best_threshold = None
            for test in threshold_tests:
                if test['results_count'] > 0 and test['results_count'] <= 10:
                    best_threshold = test['threshold']
                    break
            
            if best_threshold:
                recommendations.append(f"OPTIMIZATION: Consider using similarity threshold of {best_threshold}")
            else:
                recommendations.append("CRITICAL: No similarity threshold produces good results - check embeddings")
        
        # Content quality issues
        quality = diagnostic_results.get('content_quality', {})
        if quality.get('quality_issues'):
            for issue in quality['quality_issues']:
                recommendations.append(f"CONTENT QUALITY: {issue}")
        
        return recommendations
    
    def fix_common_issues(self) -> Dict[str, Any]:
        """Attempt to fix common issues automatically"""
        fixes_applied = {
            'fixes_attempted': [],
            'fixes_successful': [],
            'fixes_failed': [],
            'manual_fixes_needed': []
        }
        
        try:
            # Fix 1: Rebuild index if metadata exists but index is corrupted
            if os.path.exists(self.metadata_path) and not os.path.exists(f"{self.index_path}.index"):
                fixes_applied['fixes_attempted'].append("Rebuilding FAISS index from metadata")
                if self._rebuild_index():
                    fixes_applied['fixes_successful'].append("Successfully rebuilt FAISS index")
                else:
                    fixes_applied['fixes_failed'].append("Failed to rebuild FAISS index")
            
            # Fix 2: Clean up documents with empty content
            fixes_applied['fixes_attempted'].append("Cleaning empty content documents")
            cleaned_count = self._clean_empty_documents()
            if cleaned_count > 0:
                fixes_applied['fixes_successful'].append(f"Cleaned {cleaned_count} empty documents")
            
            # Fix 3: Optimize search parameters
            fixes_applied['fixes_attempted'].append("Optimizing search parameters")
            optimal_params = self._find_optimal_search_params()
            if optimal_params:
                fixes_applied['fixes_successful'].append(f"Found optimal parameters: {optimal_params}")
            
        except Exception as e:
            fixes_applied['fixes_failed'].append(f"Error during automated fixes: {str(e)}")
        
        return fixes_applied
    
    def _rebuild_index(self) -> bool:
        """Rebuild FAISS index from metadata"""
        try:
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
            
            if not metadata:
                return False
            
            # Extract texts and create embeddings
            texts = []
            for doc in metadata:
                text_content = doc.get('text', '')
                if not text_content:
                    # Reconstruct text from other fields
                    content_parts = []
                    for field in ['title', 'summary', 'content']:
                        if doc.get(field):
                            content_parts.append(str(doc[field]))
                    text_content = ' '.join(content_parts)
                texts.append(text_content)
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
            faiss.normalize_L2(embeddings)
            
            # Create new index
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            index.add(embeddings)
            
            # Save index
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            faiss.write_index(index, f"{self.index_path}.index")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error rebuilding index: {e}")
            return False
    
    def _clean_empty_documents(self) -> int:
        """Remove or fix documents with empty content"""
        try:
            if not os.path.exists(self.metadata_path):
                return 0
            
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
            
            cleaned_count = 0
            updated_metadata = []
            
            for doc in metadata:
                text = doc.get('text', '')
                
                if not text:
                    # Try to reconstruct text from other fields
                    content_parts = []
                    for field in ['title', 'summary', 'content']:
                        if doc.get(field):
                            content_parts.append(str(doc[field]))
                    
                    if content_parts:
                        doc['text'] = ' '.join(content_parts)
                        cleaned_count += 1
                        updated_metadata.append(doc)
                    # If no content can be reconstructed, skip this document
                else:
                    updated_metadata.append(doc)
            
            if cleaned_count > 0:
                # Save updated metadata
                with open(self.metadata_path, 'w') as f:
                    json.dump(updated_metadata, f, indent=2)
                
                # Rebuild index to reflect changes
                self._rebuild_index()
            
            return cleaned_count
            
        except Exception as e:
            self.logger.error(f"Error cleaning documents: {e}")
            return 0
    
    def _find_optimal_search_params(self) -> Dict[str, float]:
        """Find optimal search parameters"""
        try:
            if not os.path.exists(f"{self.index_path}.index"):
                return {}
            
            index = faiss.read_index(f"{self.index_path}.index")
            
            # Test query
            test_query = "financial news technology earnings"
            query_embedding = self.embedding_model.encode([test_query], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)
            
            scores, indices = index.search(query_embedding, min(20, index.ntotal))
            
            if len(scores[0]) == 0:
                return {}
            
            # Find threshold that gives 3-10 results
            for threshold in [0.1, 0.15, 0.2, 0.25, 0.3]:
                results_count = sum(1 for score in scores[0] if score >= threshold)
                if 3 <= results_count <= 10:
                    return {
                        'optimal_threshold': threshold,
                        'expected_results': results_count,
                        'max_similarity': float(scores[0].max())
                    }
            
            return {
                'max_similarity': float(scores[0].max()),
                'min_similarity': float(scores[0].min()),
                'recommended_threshold': 0.1
            }
            
        except Exception as e:
            self.logger.error(f"Error finding optimal parameters: {e}")
            return {}

def main():
    """Run the diagnostic tool"""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print("=" * 60)
    print("KNOWLEDGE BASE DIAGNOSTIC TOOL")
    print("=" * 60)
    
    diagnostic = KnowledgeBaseDiagnostic()
    
    # Run full diagnostic
    print("\n1. Running comprehensive diagnostic...")
    results = diagnostic.run_full_diagnostic()
    
    # Display results
    print("\n2. DIAGNOSTIC RESULTS:")
    print("-" * 40)
    
    for section, data in results.items():
        if section == 'recommendations':
            continue
        print(f"\n{section.upper().replace('_', ' ')}:")
        if isinstance(data, dict):
            for key, value in data.items():
                print(f"  {key}: {value}")
        else:
            print(f"  {data}")
    
    # Display recommendations
    print("\n3. RECOMMENDATIONS:")
    print("-" * 40)
    for i, rec in enumerate(results.get('recommendations', []), 1):
        print(f"{i}. {rec}")
    
    # Attempt fixes
    print("\n4. ATTEMPTING AUTOMATED FIXES:")
    print("-" * 40)
    fixes = diagnostic.fix_common_issues()
    
    for category, items in fixes.items():
        if items:
            print(f"\n{category.upper().replace('_', ' ')}:")
            for item in items:
                print(f"  • {item}")
    
    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()