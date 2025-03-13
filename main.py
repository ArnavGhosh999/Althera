import pandas as pd
from nlp_model import MedicineRecommendationSystem

def main():
    print("Loading medicine datasets...")
    medicines_df = pd.read_csv('All_medicines/medicines.csv')
    ses_df = pd.read_csv('All_medicines/SeS_dataset.csv')
    
    medicines_df = medicines_df.rename(columns={'name': 'medicine_name'})
    ses_df = ses_df.rename(columns={'name': 'medicine_name'})
    
    print("Initializing recommendation system...")
    rec_system = MedicineRecommendationSystem()
    
    print("\nMedicine Recommendation System")
    print("==============================")
    print("Type 'exit' to quit\n")
    
    while True:
        query = input("What medical condition can I help you with? ")
        if query.lower() == 'exit':
            break
        
        print("\nProcessing your query...")
        result = rec_system.get_recommendations(query, medicines_df, ses_df)
        
        print(f"\nCondition identified: {result['condition']}")
        print("\nRecommended medications:")
        for i, med in enumerate(result['recommendations'], 1):
            print(f"{i}. {med['medicine']}")
            if 'uses' in med:
                print(f"   Uses: {med['uses']}")
            if 'side_effects' in med:
                print(f"   Side Effects: {med['side_effects']}")
            if 'substitutes' in med:
                print(f"   Substitutes: {med['substitutes']}")
        
        print("\nEnhanced recommendation:")
        print(result['enhanced'])
        print("\n" + "-"*50 + "\n")

if __name__ == "__main__":
    main()