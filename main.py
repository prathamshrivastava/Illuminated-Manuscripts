# enhanced_text_summarizer.py
import re
import math
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
import heapq
import warnings
from typing import Dict, List, Tuple, Optional
import PyPDF2
import io

warnings.filterwarnings('ignore')

class EnhancedTextSummarizer:
    def __init__(self):
        self.download_nltk_data()
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        
    def download_nltk_data(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            print("Downloading NLTK data...")
            nltk.download('punkt', quiet=True)
            nltk.download("punkt_tab")
            nltk.download('stopwords', quiet=True)
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove special characters and extra whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.!?]', '', text)
        return text.strip()
    
    def sentence_tokenize(self, text: str) -> List[str]:
        """Tokenize text into sentences"""
        sentences = sent_tokenize(text)
        return [s.strip() for s in sentences if len(s.strip()) > 20]
    
    def word_tokenize_and_clean(self, text: str) -> List[str]:
        """Tokenize and clean words"""
        words = word_tokenize(text.lower())
        return [word for word in words if word.isalpha() and word not in self.stop_words and len(word) > 2]
    
    def calculate_readability_score(self, text: str) -> float:
        """Calculate Flesch Reading Ease score"""
        sentences = self.sentence_tokenize(text)
        words = word_tokenize(text)
        syllables = sum(self.count_syllables(word) for word in words)
        
        if len(sentences) == 0 or len(words) == 0:
            return 0
        
        avg_sentence_length = len(words) / len(sentences)
        avg_syllables_per_word = syllables / len(words)
        
        score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        return max(0, min(100, score))
    
    def count_syllables(self, word: str) -> int:
        """Count syllables in a word"""
        word = word.lower()
        vowels = "aeiouy"
        syllable_count = 0
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel
        
        # Handle silent e
        if word.endswith('e'):
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    def frequency_based_summary(self, text: str, num_sentences: int = 3) -> str:
        """Generate summary using word frequency"""
        sentences = self.sentence_tokenize(text)
        if len(sentences) <= num_sentences:
            return ' '.join(sentences)
        
        # Calculate word frequencies
        words = self.word_tokenize_and_clean(text)
        word_freq = Counter(words)
        
        # Score sentences based on word frequencies
        sentence_scores = {}
        for sentence in sentences:
            sentence_words = self.word_tokenize_and_clean(sentence)
            if sentence_words:
                score = sum(word_freq[word] for word in sentence_words)
                sentence_scores[sentence] = score / len(sentence_words)
            else:
                sentence_scores[sentence] = 0
        
        # Get top sentences and maintain order
        top_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
        summary_sentences = [s for s in sentences if s in top_sentences]
        
        return ' '.join(summary_sentences)
    
    def tfidf_based_summary(self, text: str, num_sentences: int = 3) -> str:
        """Generate summary using TF-IDF"""
        sentences = self.sentence_tokenize(text)
        if len(sentences) <= num_sentences:
            return ' '.join(sentences)
        
        # Create TF-IDF matrix
        vectorizer = TfidfVectorizer(
            stop_words='english', 
            lowercase=True, 
            max_features=1000,
            ngram_range=(1, 2)
        )
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # Calculate sentence scores
        sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
        
        # Get top sentences while maintaining order
        top_indices = sentence_scores.argsort()[-num_sentences:][::-1]
        top_indices.sort()
        
        return ' '.join([sentences[i] for i in top_indices])
    
    def textrank_summary(self, text: str, num_sentences: int = 3) -> str:
        """Generate summary using TextRank algorithm"""
        sentences = self.sentence_tokenize(text)
        if len(sentences) <= num_sentences:
            return ' '.join(sentences)
        
        # Create TF-IDF matrix for similarity calculation
        vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # Calculate similarity matrix with threshold
        similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        # Apply threshold to create sparse graph
        threshold = np.percentile(similarity_matrix, 75)
        similarity_matrix[similarity_matrix < threshold] = 0
        
        # Create graph and apply PageRank
        nx_graph = nx.from_numpy_array(similarity_matrix)
        try:
            scores = nx.pagerank(nx_graph, max_iter=100, tol=1e-6)
        except:
            # Fallback if PageRank fails
            scores = {i: similarity_matrix[i].sum() for i in range(len(sentences))}
        
        # Get top sentences maintaining order
        ranked_sentences = sorted(((scores[i], s, i) for i, s in enumerate(sentences)), reverse=True)
        top_sentences_with_idx = sorted(ranked_sentences[:num_sentences], key=lambda x: x[2])
        
        return ' '.join([sentence for score, sentence, idx in top_sentences_with_idx])
    
    def luhn_summary(self, text: str, num_sentences: int = 3) -> str:
        """Generate summary using Luhn's algorithm"""
        sentences = self.sentence_tokenize(text)
        if len(sentences) <= num_sentences:
            return ' '.join(sentences)
        
        # Calculate word frequencies
        words = self.word_tokenize_and_clean(text)
        word_freq = Counter(words)
        
        # Define significant words (top third by frequency)
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        cutoff = max(1, len(sorted_words) // 3)
        significant_words = set([word for word, freq in sorted_words[:cutoff]])
        
        # Score sentences using Luhn's method
        sentence_scores = {}
        for sentence in sentences:
            sentence_words = self.word_tokenize_and_clean(sentence)
            significant_count = sum(1 for word in sentence_words if word in significant_words)
            if significant_count > 0 and len(sentence_words) > 0:
                # Luhn's formula: (significant_words^2) / total_words
                sentence_scores[sentence] = (significant_count ** 2) / len(sentence_words)
            else:
                sentence_scores[sentence] = 0
        
        # Get top sentences maintaining order
        top_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
        summary_sentences = [s for s in sentences if s in top_sentences]
        
        return ' '.join(summary_sentences)
    
    def lsa_summary(self, text: str, num_sentences: int = 3) -> str:
        """Generate summary using Latent Semantic Analysis"""
        sentences = self.sentence_tokenize(text)
        if len(sentences) <= num_sentences:
            return ' '.join(sentences)
        
        try:
            # Create TF-IDF matrix
            vectorizer = TfidfVectorizer(
                stop_words='english', 
                lowercase=True, 
                max_features=200,
                min_df=1,
                max_df=0.8
            )
            tfidf_matrix = vectorizer.fit_transform(sentences)
            
            # Apply LSA
            n_components = min(50, len(sentences), tfidf_matrix.shape[1] - 1)
            if n_components <= 0:
                return self.tfidf_based_summary(text, num_sentences)
                
            lsa = TruncatedSVD(n_components=n_components, random_state=42)
            lsa_matrix = lsa.fit_transform(tfidf_matrix)
            
            # Score sentences based on their representation in latent space
            sentence_scores = np.linalg.norm(lsa_matrix, axis=1)
            
            # Get top sentences maintaining order
            top_indices = sentence_scores.argsort()[-num_sentences:][::-1]
            top_indices.sort()
            
            return ' '.join([sentences[i] for i in top_indices])
            
        except Exception as e:
            print(f"LSA failed: {e}")
            return self.tfidf_based_summary(text, num_sentences)
    
    def clustering_summary(self, text: str, num_sentences: int = 3) -> str:
        """Generate summary using K-means clustering"""
        sentences = self.sentence_tokenize(text)
        if len(sentences) <= num_sentences:
            return ' '.join(sentences)
        
        try:
            # Create TF-IDF matrix
            vectorizer = TfidfVectorizer(
                stop_words='english', 
                lowercase=True, 
                max_features=200,
                min_df=1
            )
            tfidf_matrix = vectorizer.fit_transform(sentences)
            
            # Apply K-means clustering
            n_clusters = min(num_sentences, len(sentences), 10)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
            clusters = kmeans.fit_predict(tfidf_matrix.toarray())
            
            # Select representative sentence from each cluster
            summary_sentences = []
            for cluster_id in range(n_clusters):
                cluster_indices = [i for i, c in enumerate(clusters) if c == cluster_id]
                if not cluster_indices:
                    continue
                    
                # Find sentence closest to cluster centroid
                cluster_vectors = tfidf_matrix[cluster_indices]
                centroid = kmeans.cluster_centers_[cluster_id]
                
                similarities = cosine_similarity(cluster_vectors, centroid.reshape(1, -1)).flatten()
                best_idx = cluster_indices[np.argmax(similarities)]
                summary_sentences.append((best_idx, sentences[best_idx]))
            
            # Sort by original order
            summary_sentences.sort(key=lambda x: x[0])
            return ' '.join([sentence for idx, sentence in summary_sentences[:num_sentences]])
            
        except Exception as e:
            print(f"Clustering failed: {e}")
            return self.tfidf_based_summary(text, num_sentences)
    
    def extractive_summary(self, text: str, num_sentences: int = 3) -> str:
        """Advanced extractive summarization combining multiple signals"""
        sentences = self.sentence_tokenize(text)
        if len(sentences) <= num_sentences:
            return ' '.join(sentences)
        
        # Get scores from different methods
        methods = [
            self.frequency_based_summary,
            self.tfidf_based_summary,
            self.textrank_summary,
            self.luhn_summary
        ]
        
        # Collect sentence rankings from each method
        sentence_votes = defaultdict(int)
        
        for method in methods:
            try:
                method_summary = method(text, len(sentences) // 2)
                method_sentences = self.sentence_tokenize(method_summary)
                for sentence in method_sentences:
                    if sentence in sentences:
                        sentence_votes[sentence] += 1
            except:
                continue
        
        # Score sentences based on position, length, and votes
        final_scores = {}
        for i, sentence in enumerate(sentences):
            # Position score (earlier sentences get higher scores)
            position_score = 1.0 / (1 + i * 0.1)
            
            # Length score (prefer medium-length sentences)
            words = len(sentence.split())
            length_score = 1.0 if 10 <= words <= 30 else 0.5
            
            # Vote score
            vote_score = sentence_votes.get(sentence, 0) / len(methods)
            
            # Combined score
            final_scores[sentence] = (position_score * 0.3 + 
                                    length_score * 0.2 + 
                                    vote_score * 0.5)
        
        # Get top sentences maintaining order
        top_sentences = heapq.nlargest(num_sentences, final_scores, key=final_scores.get)
        summary_sentences = [s for s in sentences if s in top_sentences]
        
        return ' '.join(summary_sentences)
    
    def get_all_summaries(self, text: str, num_sentences: int = 3) -> Dict[str, str]:
        """Generate summaries using all techniques"""
        text = self.preprocess_text(text)
        
        if len(text.strip()) == 0:
            return {"Error": "No valid text provided"}
        
        summaries = {}
        techniques = {
            'Frequency-Based': self.frequency_based_summary,
            'TF-IDF': self.tfidf_based_summary,
            'TextRank (PageRank)': self.textrank_summary,
            'Luhn Algorithm': self.luhn_summary,
            'LSA (Latent Semantic Analysis)': self.lsa_summary,
            'K-means Clustering': self.clustering_summary,
            'Advanced Extractive': self.extractive_summary
        }
        
        for name, method in techniques.items():
            try:
                summary = method(text, num_sentences)
                summaries[name] = summary if summary else "No summary generated"
            except Exception as e:
                summaries[name] = f"Error: {str(e)}"
        
        return summaries
    
    def calculate_rouge_scores(self, reference: str, candidate: str) -> Tuple[float, float, float]:
        """Calculate ROUGE-1 scores (Precision, Recall, F1)"""
        ref_words = set(self.word_tokenize_and_clean(reference))
        cand_words = set(self.word_tokenize_and_clean(candidate))
        
        if not ref_words or not cand_words:
            return 0.0, 0.0, 0.0
        
        intersection = ref_words.intersection(cand_words)
        precision = len(intersection) / len(cand_words)
        recall = len(intersection) / len(ref_words)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return precision, recall, f1
    
    def calculate_compression_ratio(self, original: str, summary: str) -> float:
        """Calculate compression ratio"""
        orig_words = len(original.split())
        summ_words = len(summary.split())
        return summ_words / orig_words if orig_words > 0 else 0
    
    def evaluate_summaries(self, summaries: Dict[str, str], original_text: str, 
                          reference_summary: Optional[str] = None) -> pd.DataFrame:
        """Comprehensive evaluation of summaries"""
        evaluation_data = []
        
        for technique, summary in summaries.items():
            if "Error:" in str(summary):
                evaluation_data.append({
                    'Technique': technique,
                    'Word Count': 0,
                    'Sentence Count': 0,
                    'Compression Ratio': 0,
                    'Readability Score': 0,
                    'ROUGE-1 Precision': 0,
                    'ROUGE-1 Recall': 0,
                    'ROUGE-1 F1': 0,
                    'Status': 'Error'
                })
                continue
            
            # Basic metrics
            sentences = self.sentence_tokenize(summary)
            word_count = len(summary.split())
            sentence_count = len(sentences)
            compression_ratio = self.calculate_compression_ratio(original_text, summary)
            readability = self.calculate_readability_score(summary)
            
            # ROUGE scores if reference is provided
            if reference_summary:
                precision, recall, f1 = self.calculate_rouge_scores(reference_summary, summary)
            else:
                precision, recall, f1 = 0, 0, 0
            
            evaluation_data.append({
                'Technique': technique,
                'Word Count': word_count,
                'Sentence Count': sentence_count,
                'Compression Ratio': round(compression_ratio, 3),
                'Readability Score': round(readability, 1),
                'ROUGE-1 Precision': round(precision, 3),
                'ROUGE-1 Recall': round(recall, 3),
                'ROUGE-1 F1': round(f1, 3),
                'Status': 'Success'
            })
        
        return pd.DataFrame(evaluation_data)
    
    def get_text_statistics(self, text: str) -> Dict[str, int]:
        """Get basic statistics about the text"""
        sentences = self.sentence_tokenize(text)
        words = text.split()
        characters = len(text)
        paragraphs = len([p for p in text.split('\n\n') if p.strip()])
        
        return {
            'Characters': characters,
            'Words': len(words),
            'Sentences': len(sentences),
            'Paragraphs': paragraphs,
            'Average Words per Sentence': round(len(words) / len(sentences), 1) if sentences else 0
        }

# Example usage
if __name__ == "__main__":
    summarizer = EnhancedTextSummarizer()
    
    sample_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. 
    Leading AI textbooks define the field as the study of intelligent agents: any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals. 
    Colloquially, the term artificial intelligence is often used to describe machines that mimic cognitive functions that humans associate with the human mind, such as learning and problem solving.

    The traditional problems of AI research include reasoning, knowledge representation, planning, learning, natural language processing, perception and the ability to move and manipulate objects. 
    General intelligence is among the field's long-term goals. Approaches include statistical methods, computational intelligence, and traditional symbolic AI. 
    Many tools are used in AI, including versions of search and mathematical optimization, artificial neural networks, and methods based on statistics, probability and economics. 
    The AI field draws upon computer science, information engineering, mathematics, psychology, linguistics, philosophy, and many other fields.

    The field was founded as an academic discipline in 1956, and in the years since has experienced several waves of optimism, followed by disappointment and the loss of funding, followed by new approaches, success and renewed funding. 
    For most of its history, AI research has been divided into sub-fields that often fail to communicate with each other. 
    These sub-fields are based on technical considerations, such as particular goals, the use of particular tools or the satisfaction of particular applications.
    """
    
    # Generate summaries
    summaries = summarizer.get_all_summaries(sample_text, num_sentences=2)
    
    # Evaluate summaries
    evaluation_df = summarizer.evaluate_summaries(summaries, sample_text)
    
    print("Text Statistics:")
    stats = summarizer.get_text_statistics(sample_text)
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\n" + "="*80)
    print("GENERATED SUMMARIES")
    print("="*80)
    
    for technique, summary in summaries.items():
        print(f"\n{technique}:")
        print("-" * 50)
        print(summary)
    
    print("\n" + "="*80)
    print("EVALUATION METRICS")
    print("="*80)
    print(evaluation_df.to_string(index=False))