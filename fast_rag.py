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
        try:
            if os.path.exists(self.db_path):
                with open(self.db_path, 'r') as f:
                    try:
                        return json.load(f)
                    except json.JSONDecodeError:
                        print(f"Error: {self.db_path} contains invalid JSON. Creating new database.")
                        return {}
            return {}
        except Exception as e:
            print(f"Error loading user database: {e}")
            return {}
    
    def _save_db(self):
        try:
            with open(self.db_path, 'w') as f:
                json.dump(self.users, f, indent=2)
        except Exception as e:
            print(f"Error saving user database: {e}")
    
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
        try:
            self.medicines_df = pd.read_csv(medicines_path)
            self.ses_df = pd.read_csv(ses_path, low_memory=False)
            self.medicines_df = self.medicines_df.rename(columns={'name': 'medicine_name'})
            self.ses_df = self.ses_df.rename(columns={'name': 'medicine_name'})
            print(f"Loaded {len(self.medicines_df)} medicines and {len(self.ses_df)} side effect entries")
        except Exception as e:
            print(f"Error loading medicine data: {e}")
            self.medicines_df = pd.DataFrame({
                'medicine_name': ['Paracetamol', 'Ibuprofen', 'Aspirin', 'Cetirizine'],
                'price(₹)': [10, 15, 12, 20],
                'manufacturer_name': ['Pharma A', 'Pharma B', 'Pharma C', 'Pharma D']
            })
            self.ses_df = pd.DataFrame({
                'medicine_name': ['Paracetamol', 'Ibuprofen', 'Aspirin', 'Cetirizine'],
                'use1': ['For headache and fever', 'For pain and inflammation', 'For pain and fever', 'For allergies'],
                'sideEffect1': ['Nausea', 'Stomach upset', 'Stomach bleeding', 'Drowsiness'],
                'substitute1': ['Ibuprofen', 'Paracetamol', 'Paracetamol', 'Loratadine']
            })
            print("Loaded demo data instead")
        
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
        
        for condition, keywords in self.condition_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    found_conditions.append(condition)
                    break
        
        for severity in self.severity_levels:
            if severity in text_lower:
                found_severity.append(severity)
        
        day_pattern = r'(\d+)\s*(day|days|week|weeks|month|months)'
        day_matches = re.finditer(day_pattern, text_lower)
        for match in day_matches:
            durations.append(match.group(0))
        
        return list(set(found_conditions)), list(set(found_severity)), durations
    
    def get_recommendations(self, conditions, severity=None, duration=None):
        cache_key = "_".join(sorted(conditions))
        if severity:
            cache_key += "_" + "_".join(sorted(severity))
        if duration:
            cache_key += "_" + "_".join(sorted(duration))
        
        if cache_key in self.query_cache:
            return self.query_cache[cache_key]
        
        matching_medicines = []
        sample_size = min(5000, len(self.ses_df))
        ses_sample = self.ses_df.sample(n=sample_size) if len(self.ses_df) > sample_size else self.ses_df
        
        for _, row in ses_sample.iterrows():
            medicine_name = row.get('medicine_name', '')
            if not isinstance(medicine_name, str) or pd.isna(medicine_name):
                continue
            
            use_cols = [col for col in row.index if col.startswith('use')]
            use_text = " ".join([str(row[col]) for col in use_cols if pd.notna(row[col])]).lower()
            
            match_score = 0
            for condition in conditions:
                if condition.lower() in use_text:
                    match_score += 1
                    if severity:
                        for sev in severity:
                            if sev.lower() in use_text:
                                match_score += 0.5
            
            if match_score > 0:
                med_row = self.medicines_df[self.medicines_df['medicine_name'] == medicine_name]
                if not med_row.empty:
                    price = med_row.iloc[0].get('price(₹)', 'N/A')
                    manufacturer = med_row.iloc[0].get('manufacturer_name', 'N/A')
                else:
                    price = "N/A"
                    manufacturer = "N/A"
                
                side_effect_cols = [col for col in row.index if col.startswith('sideEffect')]
                side_effects = ", ".join([str(row[col]) for col in side_effect_cols if pd.notna(row[col])])
                
                substitute_cols = [col for col in row.index if col.startswith('substitute')]
                substitutes = ", ".join([str(row[col]) for col in substitute_cols if pd.notna(row[col])])
                
                medicine_condition = next((c for c in conditions if c.lower() in use_text), conditions[0])
                
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
        
        matching_medicines.sort(key=lambda x: x["score"], reverse=True)
        
        if not matching_medicines:
            default_meds = self._get_default_medications(conditions)
            self.query_cache[cache_key] = default_meds
            return default_meds
        
        self.query_cache[cache_key] = matching_medicines
        return matching_medicines
    
    def get_recommendations_by_condition(self, conditions, severity=None, duration=None):
        result = {}
        for condition in conditions:
            single_condition_recs = self.get_recommendations([condition], severity, duration)
            result[condition] = single_condition_recs[:3] if single_condition_recs else []
        return result
    
    def _get_default_medications(self, conditions):
        default_medicines = []
        
        for condition in conditions:
            if condition in self.default_meds:
                for med_name in self.default_meds[condition]:
                    med_matches = self.medicines_df[self.medicines_df['medicine_name'].str.contains(med_name, case=False, na=False)]
                    
                    if not med_matches.empty:
                        for _, row in med_matches.iterrows():
                            medicine_name = row.get('medicine_name', '')
                            
                            if any(med["name"] == medicine_name for med in default_medicines):
                                continue
                            
                            price = row.get('price(₹)', 'N/A')
                            manufacturer = row.get('manufacturer_name', 'N/A')
                            
                            ses_row = self.ses_df[self.ses_df['medicine_name'] == medicine_name]
                            if not ses_row.empty:
                                use_cols = [col for col in ses_row.iloc[0].index if col.startswith('use')]
                                uses = ", ".join([str(ses_row.iloc[0][col]) for col in use_cols if pd.notna(ses_row.iloc[0][col])])
                                
                                side_effect_cols = [col for col in ses_row.iloc[0].index if col.startswith('sideEffect')]
                                side_effects = ", ".join([str(ses_row.iloc[0][col]) for col in side_effect_cols if pd.notna(ses_row.iloc[0][col])])
                                
                                substitute_cols = [col for col in ses_row.iloc[0].index if col.startswith('substitute')]
                                substitutes = ", ".join([str(ses_row.iloc[0][col]) for col in substitute_cols if pd.notna(ses_row.iloc[0][col])])
                            else:
                                uses = f"Used for {condition}"
                                side_effects = "No side effects information available"
                                substitutes = "No substitutes information available"
                            
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
                    return f"Welcome back, {name}! How are you feeling today? Have your symptoms improved since our last consultation?"
                else:
                    self.state = "ask_symptoms"
                    return f"Welcome back, {name}! What brings you in today? Please describe your symptoms in detail."
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
                return "Thank you for providing your information. What symptoms are you experiencing today? Please be as specific as possible."
            else:
                return "Please specify your gender as male, female, or other."
        
        elif self.state == "returning_user":
            user_response = message.lower()
            last_session = self.recommender.user_db.get_last_session(self.current_user["name"])
            
            if ("not" in user_response or "worse" in user_response or "still" in user_response or 
                "bad" in user_response or "same" in user_response or "prevail" in user_response or
                "no improvement" in user_response or "increased" in user_response):
                self.state = "follow_up_treatment"
                
                try:
                    previous_symptoms = last_session.get('symptoms', '')
                    previous_medications = last_session.get('medications', '')
                    
                    prev_conditions, prev_severity, prev_duration = self.recommender.extract_keywords(previous_symptoms)
                    for condition in prev_conditions:
                        if condition not in self.conditions:
                            self.conditions.append(condition)
                    
                    return f"I'm sorry to hear that. During your last visit, you mentioned {previous_symptoms} and I recommended {previous_medications}. Has the intensity of these symptoms changed? Please describe your current condition in detail."
                except:
                    self.state = "ask_symptoms"
                    return "I understand your symptoms haven't improved. Could you please describe your current symptoms in detail so I can recommend a different treatment approach?"
            else:
                self.state = "ask_symptoms"
                return f"I'm glad to hear you're feeling better! Is there anything else I can help you with today?"
        
        elif self.state == "follow_up_treatment":
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
            
            last_session = self.recommender.user_db.get_last_session(self.current_user["name"])
            previous_medications = last_session.get('medications', '').split(', ') if last_session else []
            
            if not self.conditions:
                last_session = self.recommender.user_db.get_last_session(self.current_user["name"])
                if last_session:
                    previous_symptoms = last_session.get('symptoms', '')
                    prev_conditions, _, _ = self.recommender.extract_keywords(previous_symptoms)
                    self.conditions.extend(prev_conditions)
                
                if not self.conditions:
                    return "I couldn't identify any specific medical conditions in your description. Could you please be more specific about your symptoms?"
            
            if "worse" in message.lower() or "increased" in message.lower():
                stronger_recommendations = self.get_stronger_recommendations(self.conditions, previous_medications)
                self.state = "recommendations_provided"
                return stronger_recommendations
            else:
                recommendations_by_condition = self.get_alternative_recommendations(
                    self.conditions, self.severity, self.duration, previous_medications
                )
                self.last_recommendations = recommendations_by_condition
                self.state = "recommendations_provided"
                return self.format_recommendations_by_condition(recommendations_by_condition, is_followup=True)
        
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
                return "I couldn't identify any specific medical conditions in your description. Could you please be more specific about your symptoms? For example, describe the type of pain, its location, or other symptoms you're experiencing."
            
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
                return "Could you provide more details about your symptoms? For example, when did they start, what makes them better or worse, and any other symptoms you may be experiencing."
        
        elif self.state == "recommendations_provided":
            if "not work" in message.lower() or "didn't work" in message.lower() or "did not work" in message.lower():
                medicine_name = None
                msg_parts = message.lower().split()
                for i, word in enumerate(msg_parts):
                    if word in ["medicine", "medication", "drug", "tablet"] and i > 0:
                        medicine_name = msg_parts[i-1]
                        break
                
                if not medicine_name and len(msg_parts) >= 2:
                    for cond in self.conditions:
                        recs = self.recommender.get_recommendations([cond])
                        for rec in recs:
                            med_name = rec.get("name", "").lower()
                            if any(part.lower() in med_name for part in msg_parts if len(part) > 3):
                                medicine_name = med_name
                                break
                
                if medicine_name:
                    alternatives = self.get_alternatives_for_medicine(medicine_name)
                    return f"I'm sorry to hear that {medicine_name} didn't work for you. Let's try an alternative approach.\n\n{alternatives}"
                else:
                    alternative_recs = self.get_alternative_recommendations(
                        self.conditions, self.severity, self.duration, []
                    )
                    return self.format_recommendations_by_condition(alternative_recs, 
                                                                  is_followup=True,
                                                                  message="I'm sorry to hear the previous medication didn't work. Let's try a different approach:")
            
            self.state = "ask_symptoms"
            
            if "what" in message.lower() and any(cond in message.lower() for cond in self.recommender.condition_keywords.keys()):
                mentioned_conditions = []
                for cond in self.recommender.condition_keywords.keys():
                    if cond in message.lower():
                        mentioned_conditions.append(cond)
                
                if mentioned_conditions:
                    condition_recs = self.recommender.get_recommendations([mentioned_conditions[0]], self.severity, self.duration)
                    return self.format_single_condition_recommendations(mentioned_conditions[0], condition_recs)
            
            if "side effect" in message.lower() or "adverse" in message.lower():
                for cond in self.conditions:
                    recs = self.recommender.get_recommendations([cond])
                    if recs:
                        med = recs[0]
                        return f"The most common side effects of {med['name']} include {med.get('side_effects', 'mild digestive discomfort, drowsiness, and dizziness')}. If you experience severe side effects, please discontinue use and consult a healthcare professional immediately."
            
            return "Is there anything specific about your symptoms or the recommended medications that you'd like me to clarify? Or do you have any other health concerns I can help with?"
                
        return "I'm here to help with your medical concerns. Could you please describe your symptoms in more detail so I can provide appropriate recommendations?"

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

    def get_stronger_recommendations(self, conditions, previous_medications):
        response_text = "Since your symptoms have worsened, I recommend a stronger treatment approach:\n\n"
        
        for condition in conditions:
            recs = self.recommender.get_recommendations([condition], self.severity, self.duration)
            
            if not recs:
                continue
                
            response_text += f"FOR {condition.upper()}:\n"
            new_meds = [med for med in recs if med['name'] not in previous_medications]
            
            if new_meds:
                med = new_meds[0]
                response_text += f"1. {med['name']} (stronger alternative)\n"
                if "uses" in med:
                    response_text += f"   Uses: {med['uses']}\n"
                if "side_effects" in med:
                    response_text += f"   Side Effects: {med['side_effects']}\n"
                if "substitutes" in med:
                    response_text += f"   Alternatives: {med['substitutes']}\n"
                    
                if condition == "headache" or condition == "pain":
                    response_text += f"   Dosage: For {self.current_user.get('gender', 'adults')} aged {self.current_user.get('age', '18+')}, take 2 tablets every 4-6 hours, not exceeding 8 tablets in 24 hours.\n"
                elif condition == "nausea" or condition == "vomiting":
                    response_text += f"   Dosage: For {self.current_user.get('gender', 'adults')} aged {self.current_user.get('age', '18+')}, take 1-2 tablets up to four times daily.\n"
                else:
                    response_text += f"   Dosage: For more severe symptoms, take a slightly higher dose as appropriate for your body weight. For {self.current_user.get('gender', 'adults')} aged {self.current_user.get('age', '18+')}, consider 1.5x the standard dose if tolerated well.\n"
                
                response_text += "\n"
            else:
                med = recs[0]
                response_text += f"1. {med['name']} (increased dosage)\n"
                if "uses" in med:
                    response_text += f"   Uses: {med['uses']}\n"
                if "side_effects" in med:
                    response_text += f"   Side Effects: {med['side_effects']}\n"
                
                response_text += f"   Dosage: Since your symptoms have worsened, consider increasing the dosage (within safe limits). For {self.current_user.get('gender', 'adults')} aged {self.current_user.get('age', '18+')}, you may take up to 1.5x the standard dose, while ensuring you don't exceed the maximum daily dose.\n\n"
        
        response_text += "IMPORTANT: These are just recommendations. Please consult a healthcare professional before changing your medication dosage. If symptoms worsen significantly, seek immediate medical attention."
        
        symptom_text = ", ".join(self.conditions)
        if self.severity:
            symptom_text = f"{', '.join(self.severity)} {symptom_text}"
        if self.duration:
            symptom_text += f" for {', '.join(self.duration)}"
        
        self.recommender.user_db.add_session(
            self.current_user["name"],
            symptom_text,
            ", ".join([f"{condition} (increased dose)" for condition in self.conditions]),
            response_text
        )
        
        return response_text

    def get_alternatives_for_medicine(self, medicine_name):
        response_text = f"Since {medicine_name} wasn't effective, I recommend trying these alternatives:\n\n"
        found_alternatives = False
        
        for condition in self.conditions:
            recs = self.recommender.get_recommendations([condition], self.severity, self.duration)
            
            for med in recs:
                if medicine_name.lower() in med['name'].lower():
                    substitutes = med.get('substitutes', '').split(', ')
                    
                    if substitutes and substitutes[0]:
                        found_alternatives = True
                        response_text += f"FOR {condition.upper()}:\n"
                        
                        for i, substitute in enumerate(substitutes[:2], 1):
                            sub_details = None
                            for rec in recs:
                                if substitute.lower() in rec['name'].lower():
                                    sub_details = rec
                                    break
                            
                            if sub_details:
                                response_text += f"{i}. {sub_details['name']}\n"
                                if "uses" in sub_details:
                                    response_text += f"   Uses: {sub_details['uses']}\n"
                                if "side_effects" in sub_details:
                                    response_text += f"   Side Effects: {sub_details['side_effects']}\n"
                                
                                if condition == "headache" or condition == "pain":
                                    response_text += f"   Dosage: For adults, take as directed on the package, typically 1-2 tablets every 4-6 hours. For {self.current_user.get('gender', 'adults')} aged {self.current_user.get('age', '18+')}.\n"
                                elif condition == "nausea" or condition == "vomiting":
                                    response_text += f"   Dosage: For {self.current_user.get('gender', 'adults')} aged {self.current_user.get('age', '18+')}, take 1 tablet up to three times daily.\n"
                                else:
                                    response_text += f"   Dosage: Follow package instructions for proper dosage. For {self.current_user.get('gender', 'adults')} aged {self.current_user.get('age', '18+')}, consult a doctor for precise dosing.\n"
                            else:
                                response_text += f"{i}. {substitute}\n"
                                response_text += f"   Dosage: Follow package instructions for proper dosage.\n"
                            
                            response_text += "\n"
                    break
            
            if found_alternatives:
                break
        
        if not found_alternatives:
            for condition in self.conditions:
                recs = self.recommender.get_recommendations([condition], self.severity, self.duration)
                if recs:
                    found_alternatives = True
                    response_text += f"FOR {condition.upper()}:\n"
                    
                    for i, med in enumerate(recs[:2], 1):
                        if medicine_name.lower() not in med['name'].lower():
                            response_text += f"{i}. {med['name']}\n"
                            if "uses" in med:
                                response_text += f"   Uses: {med['uses']}\n"
                            if "side_effects" in med:
                                response_text += f"   Side Effects: {med['side_effects']}\n"
                            
                            if condition == "headache" or condition == "pain":
                                response_text += f"   Dosage: For adults, take as directed on the package, typically 1-2 tablets every 4-6 hours. For {self.current_user.get('gender', 'adults')} aged {self.current_user.get('age', '18+')}.\n"
                            elif condition == "nausea" or condition == "vomiting":
                                response_text += f"   Dosage: For {self.current_user.get('gender', 'adults')} aged {self.current_user.get('age', '18+')}, take 1 tablet up to three times daily.\n"
                            else:
                                response_text += f"   Dosage: Follow package instructions for proper dosage. For {self.current_user.get('gender', 'adults')} aged {self.current_user.get('age', '18+')}, consult a doctor for precise dosing.\n"
                            
                            response_text += "\n"
        
        if not found_alternatives:
            response_text = f"I don't have specific alternatives for {medicine_name} in my database. I recommend consulting with a pharmacist or healthcare provider for appropriate alternatives."
        else:
            response_text += "IMPORTANT: These are just recommendations. Please consult a healthcare professional before starting any new medication."
        
        symptom_text = ", ".join(self.conditions)
        if self.severity:
            symptom_text = f"{', '.join(self.severity)} {symptom_text}"
        if self.duration:
            symptom_text += f" for {', '.join(self.duration)}"
        
        self.recommender.user_db.add_session(
            self.current_user["name"],
            symptom_text,
            "Alternative medications",
            response_text
        )
        
        return response_text

    def get_alternative_recommendations(self, conditions, severity, duration, previous_medications):
        result = {}
        
        for condition in conditions:
            all_recs = self.recommender.get_recommendations([condition], severity, duration)
            
            new_recs = [rec for rec in all_recs if rec['name'] not in previous_medications]
            
            if new_recs:
                result[condition] = new_recs[:3]
            else:
                result[condition] = all_recs[:3] if all_recs else []
        
        return result

    def format_recommendations_by_condition(self, recommendations_by_condition, is_followup=False, message=None):
        if not recommendations_by_condition:
            return "I couldn't find specific medications for your symptoms. Please consult a healthcare professional for proper diagnosis and treatment."
        
        if message:
            recommendation_text = f"{message}\n\n"
        elif is_followup:
            recommendation_text = "Based on your persistent symptoms, I recommend trying these alternative medications:\n\n"
        else:
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
