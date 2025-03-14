import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import re

class UserDatabase:
    def __init__(self, db_path="user_history.json"):
        self.db_path = db_path
        self.users = self._load_db()
    
    def _load_db(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_db(self):
        with open(self.db_path, 'w') as f:
            json.dump(self.users, f, indent=2)
    
    def get_user(self, name, age=None, gender=None):
        name_lower = name.lower()
        if name_lower not in self.users:
            self.users[name_lower] = {
                "name": name,
                "age": age,
                "gender": gender,
                "sessions": []
            }
            self._save_db()
        return self.users[name_lower]
    
    def add_session(self, name, symptoms, medications, recommendations):
        name_lower = name.lower()
        if name_lower in self.users:
            session = {
                "date": datetime.now().isoformat(),
                "symptoms": symptoms,
                "medications": medications,
                "recommendations": recommendations
            }
            self.users[name_lower]["sessions"].append(session)
            self._save_db()
    
    def get_last_session(self, name):
        name_lower = name.lower()
        if name_lower in self.users and self.users[name_lower]["sessions"]:
            return self.users[name_lower]["sessions"][-1]
        return None

class MedicineRecommender:
    def __init__(self, medicines_path="All_medicines/medicines.csv", ses_path="All_medicines/SeS_dataset.csv"):
        print("Loading medicine data...")
        self.medicines_df = pd.read_csv(medicines_path)
        self.ses_df = pd.read_csv(ses_path, low_memory=False)
        
        # Simple rename operations
        self.medicines_df = self.medicines_df.rename(columns={'name': 'medicine_name'})
        self.ses_df = self.ses_df.rename(columns={'name': 'medicine_name'})
        
        print(f"Loaded {len(self.medicines_df)} medicines and {len(self.ses_df)} side effect entries")
        
        self.condition_keywords = {
            "headache": ["headache", "head pain", "migraine", "head ache", "head hurts", "head"],
            "nausea": ["nausea", "sick to stomach", "feel sick", "queasy"],
            "vomiting": ["vomit", "throw up", "vomiting", "puking"],
            "dizziness": ["dizzy", "dizziness", "light headed", "vertigo"],
            "fever": ["fever", "high temperature", "febrile", "hot"],
            "cough": ["cough", "coughing", "sore throat", "throat pain"],
            "pain": ["pain", "ache", "hurt", "sore", "aching"],
            "fatigue": ["fatigue", "tired", "exhaustion", "weakness", "lethargy", "exhausted", "weak"],
            "insomnia": ["insomnia", "can't sleep", "difficulty sleeping", "sleeplessness"],
            "cold": ["cold", "runny nose", "stuffy nose", "congestion"],
            "stomach": ["stomach", "abdomen", "belly", "tummy", "gut"],
            "allergy": ["allergy", "allergic", "allergies", "hay fever"]
        }
        
        self.severity_levels = ["mild", "moderate", "severe", "extreme", "intense", "unbearable", "constant", "persistent", "recurring"]
        self.user_db = UserDatabase()
        self.query_cache = {}
        
        # Define default medications for quick lookup
        self.default_meds = {
            "headache": ["Paracetamol", "Ibuprofen", "Aspirin"],
            "pain": ["Paracetamol", "Ibuprofen", "Diclofenac"],
            "fever": ["Paracetamol", "Ibuprofen", "Aspirin"],
            "nausea": ["Ondansetron", "Domperidone", "Metoclopramide"],
            "vomiting": ["Ondansetron", "Domperidone", "Metoclopramide"],
            "dizziness": ["Meclizine", "Dimenhydrinate", "Prochlorperazine"],
            "cough": ["Dextromethorphan", "Codeine", "Guaifenesin"],
            "fatigue": ["Multivitamin", "Caffeine", "Coenzyme Q10"],
            "cold": ["Cetirizine", "Pseudoephedrine", "Phenylephrine"],
            "allergy": ["Cetirizine", "Loratadine", "Fexofenadine"],
            "stomach": ["Ranitidine", "Omeprazole", "Pantoprazole"]
        }
    
    def extract_keywords(self, text):
        text_lower = text.lower()
        found_conditions = []
        found_severity = []
        durations = []
        
        # Find conditions
        for condition, keywords in self.condition_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    found_conditions.append(condition)
                    break
        
        # Find severity
        for severity in self.severity_levels:
            if severity in text_lower:
                found_severity.append(severity)
        
        # Find duration
        day_pattern = r'(\d+)\s*(day|days|week|weeks|month|months)'
        day_matches = re.finditer(day_pattern, text_lower)
        for match in day_matches:
            durations.append(match.group(0))
        
        return list(set(found_conditions)), list(set(found_severity)), durations
    
    def get_recommendations(self, conditions, severity=None, duration=None):
        # Create a string-based cache key
        cache_key = "_".join(sorted(conditions))
        if severity:
            cache_key += "_" + "_".join(sorted(severity))
        if duration:
            cache_key += "_" + "_".join(sorted(duration))
        
        # Check cache
        if cache_key in self.query_cache:
            return self.query_cache[cache_key]
        
        matching_medicines = []
        
        # Simple approach - sample a small subset of data for quicker processing
        sample_size = min(5000, len(self.ses_df))
        ses_sample = self.ses_df.sample(n=sample_size) if len(self.ses_df) > sample_size else self.ses_df
        
        # Find medicines that match conditions
        for _, row in ses_sample.iterrows():
            medicine_name = row.get('medicine_name', '')
            if not isinstance(medicine_name, str) or pd.isna(medicine_name):
                continue
            
            # Extract use text
            use_cols = [col for col in row.index if col.startswith('use')]
            use_text = " ".join([str(row[col]) for col in use_cols if pd.notna(row[col])]).lower()
            
            # Calculate match score
            match_score = 0
            for condition in conditions:
                if condition.lower() in use_text:
                    match_score += 1
                    if severity:
                        for sev in severity:
                            if sev.lower() in use_text:
                                match_score += 0.5
            
            # Only process medicines with a positive match score
            if match_score > 0:
                # Get medicine details
                med_row = self.medicines_df[self.medicines_df['medicine_name'] == medicine_name]
                if not med_row.empty:
                    price = med_row.iloc[0].get('price(₹)', 'N/A')
                    manufacturer = med_row.iloc[0].get('manufacturer_name', 'N/A')
                else:
                    price = "N/A"
                    manufacturer = "N/A"
                
                # Extract side effects
                side_effect_cols = [col for col in row.index if col.startswith('sideEffect')]
                side_effects = ", ".join([str(row[col]) for col in side_effect_cols if pd.notna(row[col])])
                
                # Extract substitutes
                substitute_cols = [col for col in row.index if col.startswith('substitute')]
                substitutes = ", ".join([str(row[col]) for col in substitute_cols if pd.notna(row[col])])
                
                # Determine which condition this medicine is for
                medicine_condition = next((c for c in conditions if c.lower() in use_text), conditions[0])
                
                # Add to results
                matching_medicines.append({
                    "name": medicine_name,
                    "score": match_score,
                    "condition": medicine_condition,
                    "uses": use_text,
                    "side_effects": side_effects,
                    "substitutes": substitutes,
                    "price": price,
                    "manufacturer": manufacturer
                })
        
        # Sort by score
        matching_medicines.sort(key=lambda x: x["score"], reverse=True)
        
        # If no results found, use default medications
        if not matching_medicines:
            default_meds = self._get_default_medications(conditions)
            self.query_cache[cache_key] = default_meds
            return default_meds
        
        # Cache and return results
        self.query_cache[cache_key] = matching_medicines
        return matching_medicines
    
    def get_recommendations_by_condition(self, conditions, severity=None, duration=None):
        result = {}
        for condition in conditions:
            # Get recommendations for a single condition
            single_condition_recs = self.get_recommendations([condition], severity, duration)
            result[condition] = single_condition_recs[:3] if single_condition_recs else []
        return result
    
    def _get_default_medications(self, conditions):
        default_medicines = []
        
        for condition in conditions:
            if condition in self.default_meds:
                for med_name in self.default_meds[condition]:
                    # Get medicine details by name search
                    med_matches = self.medicines_df[self.medicines_df['medicine_name'].str.contains(med_name, case=False, na=False)]
                    
                    if not med_matches.empty:
                        for _, row in med_matches.iterrows():
                            medicine_name = row.get('medicine_name', '')
                            
                            # Skip if already added
                            if any(med["name"] == medicine_name for med in default_medicines):
                                continue
                            
                            price = row.get('price(₹)', 'N/A')
                            manufacturer = row.get('manufacturer_name', 'N/A')
                            
                            # Get side effects data
                            ses_row = self.ses_df[self.ses_df['medicine_name'] == medicine_name]
                            if not ses_row.empty:
                                # Extract use text
                                use_cols = [col for col in ses_row.iloc[0].index if col.startswith('use')]
                                uses = ", ".join([str(ses_row.iloc[0][col]) for col in use_cols if pd.notna(ses_row.iloc[0][col])])
                                
                                # Extract side effects
                                side_effect_cols = [col for col in ses_row.iloc[0].index if col.startswith('sideEffect')]
                                side_effects = ", ".join([str(ses_row.iloc[0][col]) for col in side_effect_cols if pd.notna(ses_row.iloc[0][col])])
                                
                                # Extract substitutes
                                substitute_cols = [col for col in ses_row.iloc[0].index if col.startswith('substitute')]
                                substitutes = ", ".join([str(ses_row.iloc[0][col]) for col in substitute_cols if pd.notna(ses_row.iloc[0][col])])
                            else:
                                uses = f"Used for {condition}"
                                side_effects = "No side effects information available"
                                substitutes = "No substitutes information available"
                            
                            # Add to results
                            default_medicines.append({
                                "name": medicine_name,
                                "score": 0.5,
                                "condition": condition,
                                "uses": uses,
                                "side_effects": side_effects,
                                "substitutes": substitutes,
                                "price": price,
                                "manufacturer": manufacturer
                            })
        
        return default_medicines

class MedicalChatbot:
    def __init__(self):
        self.recommender = MedicineRecommender()
        self.state = "greeting"
        self.current_user = None
        self.conversation = []
        self.conditions = []
        self.severity = []
        self.duration = []
        self.last_message = ""
    
    def process_message(self, message):
        if message:
            self.last_message = message
            self.conversation.append(message)
        
        if self.state == "greeting":
            self.state = "ask_name"
            return "Hello! I'm your medical assistant. Before we proceed, may I know your name?"
        
        elif self.state == "ask_name":
            name = message
            user = self.recommender.user_db.get_user(name)
            self.current_user = user
            
            if user.get("age") and user.get("gender"):
                self.state = "returning_user"
                last_session = self.recommender.user_db.get_last_session(name)
                if last_session:
                    return f"Welcome back, {name}! How are you feeling today?"
                else:
                    self.state = "ask_symptoms"
                    return f"Welcome back, {name}! How can I help you today? Please describe your symptoms."
            else:
                self.state = "ask_age"
                return f"Nice to meet you, {name}. Could you please tell me your age?"
        
        elif self.state == "ask_age":
            try:
                age = int(message)
                self.current_user["age"] = age
                self.state = "ask_gender"
                return "Thank you. What is your gender (male/female/other)?"
            except:
                return "Please provide a valid age as a number."
        
        elif self.state == "ask_gender":
            gender = message.lower()
            if gender in ["male", "female", "other"]:
                self.current_user["gender"] = gender
                self.current_user = self.recommender.user_db.get_user(
                    self.current_user["name"],
                    self.current_user["age"],
                    gender
                )
                self.state = "ask_symptoms"
                return "Thank you for providing your information. How can I help you today? Please describe your symptoms."
            else:
                return "Please specify your gender as male, female, or other."
        
        elif self.state == "returning_user":
            user_response = message.lower()
            last_session = self.recommender.user_db.get_last_session(self.current_user["name"])
            
            if "not" in user_response or "worse" in user_response or "still" in user_response or "bad" in user_response:
                self.state = "ask_symptoms"
                return f"I'm sorry to hear that. Let me check your last visit. You mentioned {last_session['symptoms']} and I recommended {last_session['medications']}. Has anything changed in your symptoms or do you have new ones?"
            else:
                self.state = "ask_symptoms"
                return f"Great to hear that! Last time you had {last_session['symptoms']}. Is there anything else I can help you with today?"
        
        elif self.state == "ask_symptoms":
            new_conditions, new_severity, new_duration = self.recommender.extract_keywords(message)
            
            any_new_info = False
            for condition in new_conditions:
                if condition not in self.conditions:
                    self.conditions.append(condition)
                    any_new_info = True
            
            for sev in new_severity:
                if sev not in self.severity:
                    self.severity.append(sev)
                    any_new_info = True
            
            for dur in new_duration:
                if dur not in self.duration:
                    self.duration.append(dur)
                    any_new_info = True
            
            if not self.conditions:
                return "I couldn't identify any specific medical conditions in your description. Could you please be more specific about your symptoms?"
            
            if "what" in message.lower() and "headache" in message.lower():
                headache_recs = self.recommender.get_recommendations(["headache"], self.severity, self.duration)
                return self.format_single_condition_recommendations("headache", headache_recs)
            
            if "what" in message.lower() and "fatigue" in message.lower():
                fatigue_recs = self.recommender.get_recommendations(["fatigue"], self.severity, self.duration)
                return self.format_single_condition_recommendations("fatigue", fatigue_recs)
            
            if any_new_info or not hasattr(self, 'last_recommendations'):
                recommendations_by_condition = self.recommender.get_recommendations_by_condition(
                    self.conditions, self.severity, self.duration
                )
                self.last_recommendations = recommendations_by_condition
                self.state = "recommendations_provided"
                return self.format_recommendations_by_condition(recommendations_by_condition)
            else:
                return "Is there anything specific about your symptoms that you'd like me to address?"
        
        elif self.state == "recommendations_provided":
            self.state = "ask_symptoms"
            
            if "what" in message.lower() and any(cond in message.lower() for cond in self.recommender.condition_keywords.keys()):
                mentioned_conditions = []
                for cond in self.recommender.condition_keywords.keys():
                    if cond in message.lower():
                        mentioned_conditions.append(cond)
                
                if mentioned_conditions:
                    condition_recs = self.recommender.get_recommendations([mentioned_conditions[0]], self.severity, self.duration)
                    return self.format_single_condition_recommendations(mentioned_conditions[0], condition_recs)
            
            return "Is there anything else I can help you with regarding your health?"
            
        return "I'm not sure how to respond to that. Could you please rephrase?"
    
    def format_single_condition_recommendations(self, condition, medications):
        if not medications:
            return f"I couldn't find specific medications for {condition}. Please consult a healthcare professional."
        
        top_meds = medications[:3]
        
        recommendation_text = f"For {condition}, I recommend:\n\n"
        
        for i, med in enumerate(top_meds, 1):
            recommendation_text += f"{i}. {med['name']}\n"
            if "uses" in med:
                recommendation_text += f"   Uses: {med['uses']}\n"
            if "side_effects" in med:
                recommendation_text += f"   Side Effects: {med['side_effects']}\n"
            if "substitutes" in med:
                recommendation_text += f"   Alternatives: {med['substitutes']}\n"
                
            if condition == "headache" or condition == "pain":
                recommendation_text += f"   Dosage: For adults, take as directed on the package, typically 1-2 tablets every 4-6 hours. For {self.current_user.get('gender', 'adults')} aged {self.current_user.get('age', '18+')}.\n"
            elif condition == "nausea" or condition == "vomiting":
                recommendation_text += f"   Dosage: For {self.current_user.get('gender', 'adults')} aged {self.current_user.get('age', '18+')}, take 1 tablet up to three times daily.\n"
            else:
                recommendation_text += f"   Dosage: Follow package instructions for proper dosage. For {self.current_user.get('gender', 'adults')} aged {self.current_user.get('age', '18+')}, consult a doctor for precise dosing.\n"
                
            recommendation_text += "\n"
        
        return recommendation_text
    
    def format_recommendations_by_condition(self, recommendations_by_condition):
        if not recommendations_by_condition:
            return "I couldn't find specific medications for your symptoms. Please consult a healthcare professional for proper diagnosis and treatment."
        
        recommendation_text = f"Based on your symptoms "
        if self.severity:
            recommendation_text += f"which appear to be {', '.join(self.severity)} "
        if self.duration:
            recommendation_text += f"and have persisted for {', '.join(self.duration)} "
        recommendation_text += ", here are my recommendations:\n\n"
        
        all_medications = []
        
        for condition, medications in recommendations_by_condition.items():
            if not medications:
                continue
                
            recommendation_text += f"FOR {condition.upper()}:\n"
            for i, med in enumerate(medications[:2], 1):
                recommendation_text += f"{i}. {med['name']}\n"
                if "uses" in med:
                    recommendation_text += f"   Uses: {med['uses']}\n"
                if "side_effects" in med:
                    recommendation_text += f"   Side Effects: {med['side_effects']}\n"
                if "substitutes" in med:
                    recommendation_text += f"   Alternatives: {med['substitutes']}\n"
                    
                if condition == "headache" or condition == "pain":
                    recommendation_text += f"   Dosage: For adults, take as directed on the package, typically 1-2 tablets every 4-6 hours. For {self.current_user.get('gender', 'adults')} aged {self.current_user.get('age', '18+')}.\n"
                elif condition == "nausea" or condition == "vomiting":
                    recommendation_text += f"   Dosage: For {self.current_user.get('gender', 'adults')} aged {self.current_user.get('age', '18+')}, take 1 tablet up to three times daily.\n"
                else:
                    recommendation_text += f"   Dosage: Follow package instructions for proper dosage. For {self.current_user.get('gender', 'adults')} aged {self.current_user.get('age', '18+')}, consult a doctor for precise dosing.\n"
                
                all_medications.append(med['name'])    
                recommendation_text += "\n"
        
        recommendation_text += "IMPORTANT: These are just recommendations. Please consult a healthcare professional before starting any medication."
        
        symptom_text = ", ".join(self.conditions)
        if self.severity:
            symptom_text = f"{', '.join(self.severity)} {symptom_text}"
        if self.duration:
            symptom_text += f" for {', '.join(self.duration)}"
        
        self.recommender.user_db.add_session(
            self.current_user["name"],
            symptom_text,
            ", ".join(all_medications),
            recommendation_text
        )
        
        return recommendation_text

def chat_loop():
    print("Medical Recommendation System")
    print("=============================")
    print("Type 'exit' to quit the program.")
    
    chatbot = MedicalChatbot()
    first_response = chatbot.process_message("")
    print(first_response)
    
    while True:
        user_input = input("> ")
        
        if user_input.lower() == "exit":
            print("Thank you for using the medical recommendation system. Stay healthy!")
            break
        
        response = chatbot.process_message(user_input)
        print(response)

if __name__ == "__main__":
    chat_loop()