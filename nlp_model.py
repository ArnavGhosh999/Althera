import torch
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
from googletrans import Translator
import lancedb
import os
import requests
import json

class MedicalNLPModel:
    def __init__(self, model_name='bert-base-uncased', lancedb_path='./medicine_lancedb'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()
        self.translator = Translator()
        self.db = lancedb.connect(lancedb_path)
        try:
            self.query_table = self.db.open_table("medicine_queries")
        except:
            print("Medicine queries table not found. Will create on first store.")
            self.query_table = None
        self.condition_mapping = self._load_condition_mapping()
    
    def _load_condition_mapping(self):
        return {
            "cold": "common cold", 
            "flu": "influenza",
            "headache": "headache",
            "migraine": "migraine",
            "fever": "fever",
            "cough": "cough",
            "sore throat": "sore throat",
            "vomiting": "vomiting",
            "nausea": "nausea",
            "diarrhea": "diarrhea",
            "constipation": "constipation",
            "allergies": "allergic rhinitis",
            "rash": "skin rash",
            "pain": "pain",
            "insomnia": "insomnia",
            "diabetes": "diabetes",
            "hypertension": "high blood pressure",
            "stomach": "stomach pain"
        }
    
    def translate_if_needed(self, text, target_lang='en'):
        return text
    
    def extract_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        vector_dim = 1536
        embedding = embeddings[0]
        if len(embedding) < vector_dim:
            padded_embedding = np.pad(embedding, (0, vector_dim - len(embedding)))
            return padded_embedding
        elif len(embedding) > vector_dim:
            return embedding[:vector_dim]
        else:
            return embedding
    
    def extract_condition(self, query):
        translated_query = self.translate_if_needed(query)
        lower_query = translated_query.lower()
        for keyword, condition in self.condition_mapping.items():
            if keyword in lower_query:
                return condition, 'mapping'
        try:
            if self.query_table is not None:
                query_embedding = self.extract_embedding(translated_query)
                results = self.query_table.search(query_embedding).limit(1).to_pandas()
                if not results.empty and 'condition' in results.columns:
                    return results['condition'].iloc[0], 'vector'
        except Exception as e:
            print(f"Vector search error: {str(e)}")
        return translated_query, 'fallback'
    
    def search_similar_queries(self, query, n_results=3):
        try:
            if self.query_table is None:
                return pd.DataFrame()
            translated_query = self.translate_if_needed(query)
            query_embedding = self.extract_embedding(translated_query)
            results = self.query_table.search(query_embedding).limit(n_results).to_pandas()
            return results
        except Exception as e:
            print(f"Error searching similar queries: {str(e)}")
            return pd.DataFrame()
    
    def store_query(self, query, condition, recommendations):
        try:
            translated_query = self.translate_if_needed(query)
            query_id = str(hash(translated_query))
            query_embedding = self.extract_embedding(translated_query)
            medicine_names = [rec['medicine'] for rec in recommendations]
            medicine_text = ", ".join(medicine_names)
            data = pd.DataFrame([{
                "id": query_id,
                "query": translated_query,
                "condition": condition,
                "medicines": medicine_text,
                "timestamp": pd.Timestamp.now().isoformat(),
                "vector": query_embedding.tolist()
            }])
            if self.query_table is None:
                try:
                    self.query_table = self.db.create_table("medicine_queries", data=data)
                except Exception as e:
                    print(f"Error creating table: {str(e)}")
                    return False
            else:
                self.query_table.add(data)
            return True
        except Exception as e:
            print(f"Error storing query: {str(e)}")
            return False

class MedicineRecommendationSystem:
    def __init__(self, mistral_api_key=None):
        self.nlp_model = MedicalNLPModel()
        self.mistral_api_key = mistral_api_key or os.getenv("ALTHERA")
    
    def query_mistral_llm(self, prompt):
        if not self.mistral_api_key:
            return {"error": "No API key provided"}
        headers = {
            "Authorization": f"Bearer {self.mistral_api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "mistral-tiny",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 800
        }
        try:
            response = requests.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            if response.status_code != 200:
                return {"error": f"API returned status code {response.status_code}"}
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_recommendations(self, query, from_medicines_df, from_ses_df):
        similar_queries = self.nlp_model.search_similar_queries(query)
        has_similar = len(similar_queries) > 0
        if has_similar:
            print(f"Found {len(similar_queries)} similar queries in database")
        condition, extraction_method = self.nlp_model.extract_condition(query)
        print(f"Extracted condition '{condition}' using {extraction_method} method")
        recommendations = self.get_medicines_for_condition(condition, from_medicines_df, from_ses_df)
        enhanced_recommendations = self.enhance_with_llm(query, condition, recommendations)
        self.nlp_model.store_query(query, condition, recommendations)
        return {
            'query': query,
            'condition': condition,
            'recommendations': recommendations,
            'enhanced': enhanced_recommendations,
            'had_similar_queries': has_similar
        }
    
    def get_medicines_for_condition(self, condition, medicines_df, ses_df):
        condition_terms = condition.lower().split()
        matched_medicines = []
        for _, row in medicines_df.iterrows():
            medicine_name = row['medicine_name'] if 'medicine_name' in row else str(row.get('name', ''))
            score = 0
            med_text = f"{medicine_name} {row.get('short_composition1', '')} {row.get('short_composition2', '')}"
            med_text = med_text.lower()
            for term in condition_terms:
                if term in med_text:
                    score += 1
            if score > 0:
                matched_medicines.append({
                    'medicine': medicine_name,
                    'similarity_score': score / len(condition_terms),
                    'price': row.get('price(₹)', 'N/A'),
                    'manufacturer': row.get('manufacturer_name', 'N/A')
                })
        matched_medicines.sort(key=lambda x: x['similarity_score'], reverse=True)
        for med in matched_medicines[:5]:
            self.add_ses_data(med, ses_df)
        return matched_medicines[:5] if matched_medicines else []
    
    def add_ses_data(self, medicine_rec, ses_df):
        medicine_name = medicine_rec['medicine']
        matches = ses_df[ses_df['medicine_name'].str.lower() == medicine_name.lower()]
        if len(matches) > 0:
            ses_data = matches.iloc[0]
            side_effects = []
            for col in ses_df.columns:
                if col.startswith('sideEffect') and ses_data.get(col):
                    if pd.notna(ses_data[col]):
                        side_effects.append(str(ses_data[col]))
            medicine_rec['side_effects'] = ', '.join(side_effects) if side_effects else 'No side effects listed'
            substitutes = []
            for col in ses_df.columns:
                if col.startswith('substitute') and ses_data.get(col):
                    if pd.notna(ses_data[col]):
                        substitutes.append(str(ses_data[col]))
            medicine_rec['substitutes'] = ', '.join(substitutes) if substitutes else 'No substitutes listed'
            uses = []
            for col in ses_df.columns:
                if col.startswith('use') and ses_data.get(col):
                    if pd.notna(ses_data[col]):
                        uses.append(str(ses_data[col]))
            medicine_rec['uses'] = ', '.join(uses) if uses else 'No uses listed'
        else:
            medicine_rec['side_effects'] = 'No side effects data available'
            medicine_rec['substitutes'] = 'No substitutes data available'
            medicine_rec['uses'] = 'No uses data available'
    
    def enhance_with_llm(self, query, condition, recommendations):
        if not recommendations:
            return "No suitable medications found for your condition."
        prompt = f"""
        User query: "{query}"
        
        Based on our medical database, the following medications might be appropriate for {condition}.
        Please evaluate these options and provide advice, highlighting important keywords.
        
        Recommended medications:
        """
        for i, rec in enumerate(recommendations, 1):
            prompt += f"\n{i}. {rec['medicine']}\n"
            if 'uses' in rec:
                prompt += f"   Uses: {rec['uses']}\n"
            if 'side_effects' in rec:
                prompt += f"   Side Effects: {rec['side_effects']}\n"
            if 'substitutes' in rec:
                prompt += f"   Substitutes: {rec['substitutes']}\n"
        prompt += """
        
        Please provide a helpful response that:
        1. Addresses the user's medical condition
        2. Recommends the most appropriate medication(s) from the list
        3. Highlights important keywords in **bold** text
        4. Warns about potential side effects
        5. Suggests alternative medications if appropriate
        6. Includes a disclaimer about consulting a healthcare professional
        """
        llm_response = self.query_mistral_llm(prompt)
        if 'error' in llm_response:
            response_text = f"Based on your condition ({condition}), these medications may help:\n\n"
            for i, rec in enumerate(recommendations, 1):
                response_text += f"{i}. {rec['medicine']}\n"
                if 'uses' in rec and rec['uses'] != 'No uses listed':
                    response_text += f"   Uses: {rec['uses']}\n"
                if 'side_effects' in rec and rec['side_effects'] != 'No side effects listed':
                    response_text += f"   Side Effects: {rec['side_effects']}\n"
                if 'substitutes' in rec and rec['substitutes'] != 'No substitutes listed':
                    response_text += f"   Alternatives: {rec['substitutes']}\n"
                response_text += "\n"
            response_text += "Please consult a healthcare professional before taking any medication."
            return response_text
        elif 'choices' in llm_response and len(llm_response['choices']) > 0:
            return llm_response['choices'][0]['message']['content']
        else:
            return "Could not generate enhanced recommendations."