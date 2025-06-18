import os
import sys
import subprocess

def main():
    """Main console menu to select and run different pipelines."""
    
    print("="*60)
    print("         TIME SERIES FORECASTING PIPELINE SELECTOR")
    print("="*60)
    
    # Define available pipelines
    pipelines = {
        "1": {
            "name": "Random Forest Pipeline",
            "path": "pipelines/random_forest/random_forest_pipeline.py",
            "description": "Random Forest with lag features and seasonal components"
        },
        "2": {
            "name": "Transformer Pipeline", 
            "path": "pipelines/transformer/transformer_pipeline.py",
            "description": "Time Series Transformer with deep learning"
        },
        "3": {
            "name": "XGBoost Pipeline",
            "path": "pipelines/xgboost/xgboost_pipeline.py", 
            "description": "XGBoost with extensive feature engineering"
        }
    }
    
    while True:
        print("\nAvailable Pipelines:")
        print("-" * 40)
        
        for key, pipeline in pipelines.items():
            print(f"{key}. {pipeline['name']}")
            print(f"   📋 {pipeline['description']}")
            print()
        
        print("q. Quit")
        print("-" * 40)
        
        choice = input("Select a pipeline to run (1-3 or q): ").strip().lower()
        
        if choice == 'q':
            print("👋 Goodbye!")
            break
        elif choice in pipelines:
            pipeline = pipelines[choice]
            pipeline_path = pipeline["path"]
            
            if not os.path.exists(pipeline_path):
                print(f"❌ Error: Pipeline file not found at {pipeline_path}")
                continue
            
            print(f"\n🚀 Starting {pipeline['name']}...")
            print(f"📁 Running: {pipeline_path}")
            print("="*60)
            
            try:
                # Run the selected pipeline
                result = subprocess.run([sys.executable, pipeline_path], 
                                      capture_output=False, 
                                      text=True)
                
                if result.returncode == 0:
                    print("\n" + "="*60)
                    print(f"✅ {pipeline['name']} completed successfully!")
                else:
                    print("\n" + "="*60)
                    print(f"❌ {pipeline['name']} failed with return code {result.returncode}")
                    
            except KeyboardInterrupt:
                print("\n🛑 Pipeline execution interrupted by user.")
            except Exception as e:
                print(f"❌ Error running pipeline: {e}")
            
            print("\nPress Enter to return to menu...")
            input()
            
        else:
            print("❌ Invalid choice. Please select 1-3 or q.")

if __name__ == "__main__":
    main() 